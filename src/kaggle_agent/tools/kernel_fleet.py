"""Kaggle kernel fleet automation.

This module automates the Kaggle *kernel* lifecycle that ``tools/kaggle_api.py``
does not cover (it only handles competition download / submit / leaderboard).
It replaces the ad-hoc bash in ``competitions/<comp>/kaggle/`` (``run_new.sh``,
``poll_all.sh``, ``poll_wave2.sh``) with a single, fault-tolerant Python driver.

The central class is :class:`KernelFleet`. It shells out to the official
``kaggle`` CLI via ``subprocess`` (same style as ``KaggleClient``), parses
stdout, retries transient errors with backoff, and is careful to *never* crash
the whole fleet because one kernel misbehaved.

Hard-won facts this module encodes (learned the hard way on Playground S6E6):

* Kaggle GPU = Tesla **P100 (sm_60)**. Stock ``torch>=2.10+cu128`` ships no
  ``sm_60`` kernel image -> plain-torch kernels die with
  ``CUDA error: no kernel image available``. Fix: install
  ``torch==2.4.1 --extra-index-url https://download.pytorch.org/whl/cu121``
  *before* importing torch (and some libs then need ``device='cuda:0'`` rather
  than ``'cuda'``).
* cuDF / RAPIDS also dropped Pascal -> ``cudf`` crashes with
  ``invalid device ordinal``. Feature engineering must be **pandas**.
  (CatBoost / XGBoost GPU still work fine on P100.)
* GPU **batch-session cap = 2**: a 3rd concurrent GPU push fails with
  ``Maximum batch GPU session count of 2 reached``. CPU kernels do not count.
* **Slug poisoning**: a kernel whose first push failed returns
  ``Notebook not found`` forever -> you must push under a *new* slug. Kaggle
  slugs use hyphens, never underscores.
* Kernel outputs can be truncated / partial -> always ``np.load`` and
  shape-check every pulled ``.npy``; re-pull on failure.
* Contract for this competition's artifacts: OOF ``(577347, 3)`` /
  test ``(247435, 3)`` ``float32``; labels ``GALAXY=0 / QSO=1 / STAR=2``;
  ``StratifiedKFold(5, shuffle=True, random_state=42)`` on integer ``y`` in CSV
  order; competition data at ``/kaggle/input/(competitions/)?<slug>/``.

Most network-touching methods are thin CLI wrappers; the *pure logic*
(push-result classification, log diagnosis, the GPU-cap queue decision, and
output verification given local files) is split into small testable functions so
it can be exercised without any Kaggle credentials.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Status / result vocabularies
# ---------------------------------------------------------------------------

#: Normalised kernel statuses returned by :meth:`KernelFleet.status`.
STATUS_RUNNING = "running"
STATUS_COMPLETE = "complete"
STATUS_ERROR = "error"
STATUS_CANCEL = "cancel"
STATUS_QUEUED = "queued"
STATUS_UNKNOWN = "unknown"

#: A status is *terminal* when no further polling is worthwhile.
TERMINAL_STATUSES = frozenset({STATUS_COMPLETE, STATUS_ERROR, STATUS_CANCEL})

#: Classifications returned by :meth:`KernelFleet.push`.
PUSH_PUSHED = "pushed"
PUSH_QUEUED = "queued"  # GPU batch cap hit; caller should retry later
PUSH_SLUG_POISONED = "slug_poisoned"
PUSH_ERROR = "error"

# Substrings the Kaggle CLI prints for the conditions we care about.
_GPU_CAP_MARKER = "maximum batch gpu session count of 2 reached"
_SLUG_POISON_MARKER = "notebook not found"


# ---------------------------------------------------------------------------
# Known-failure diagnosis catalogue (pure logic, unit-tested)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class _Diagnosis:
    """A known failure signature and its remedy."""

    name: str
    pattern: "re.Pattern[str]"
    fix: str


# Order matters: earlier entries win when several patterns match.
_DIAGNOSES: Tuple[_Diagnosis, ...] = (
    _Diagnosis(
        name="no_kernel_image",
        pattern=re.compile(r"no kernel image is available|no kernel image", re.I),
        fix=(
            "P100 is sm_60 and stock torch (>=2.10/cu128) ships no sm_60 image. "
            "pip install torch==2.4.1 --extra-index-url "
            "https://download.pytorch.org/whl/cu121 BEFORE importing torch; "
            "use device='cuda:0' if a lib rejects bare 'cuda'."
        ),
    ),
    _Diagnosis(
        name="cudf_pascal_dropped",
        pattern=re.compile(r"invalid device ordinal", re.I),
        fix=(
            "cuDF/RAPIDS dropped Pascal (P100) -> 'invalid device ordinal'. "
            "Do feature engineering in pandas (CatBoost/XGBoost GPU still work)."
        ),
    ),
    _Diagnosis(
        name="slug_poisoned",
        pattern=re.compile(r"notebook not found", re.I),
        fix=(
            "Slug is poisoned (first push failed -> 'Notebook not found' forever). "
            "Re-push under a NEW hyphenated slug."
        ),
    ),
    _Diagnosis(
        name="catboost_float_cat_features",
        # Matches all three real CatBoost float-cat-feature messages without
        # false-positiving on a bare "float":
        #   1. '... column "redshift" has dtype float64 ... in cat_features list'
        #   2. '... which are categorical are not of type int but floating point'
        #   3. 'cat_features must be integer or string, real number values ...'
        pattern=re.compile(
            r"cat_features.*float|float.*cat_features|floating point|"
            r"cat_feature.*must be (?:integer|int)|"
            r"categorical.*not of type int|not of type int.*categor",
            re.I,
        ),
        fix=(
            "CatBoost got float-typed categorical columns. Cast cat columns to "
            "int (or pass cat_features=None and let CatBoost infer)."
        ),
    ),
    _Diagnosis(
        name="bad_oof_shape",
        pattern=re.compile(
            r"is not divisible by 3|cannot reshape|"
            r"shape mismatch|could not broadcast",
            re.I,
        ),
        fix=(
            "OOF/test array was saved with the wrong shape. Expect OOF "
            "(577347,3) / test (247435,3) float32; reshape(-1,3) before save."
        ),
    ),
    _Diagnosis(
        name="oom",
        pattern=re.compile(r"out of memory|outofmemoryerror|killed", re.I),
        fix="Ran out of memory. Reduce batch size / dtype, or chunk the data.",
    ),
)


def classify_push_output(stdout: str, returncode: int = 0) -> str:
    """Classify the stdout/stderr of ``kaggle kernels push``.

    Pure function (no I/O) so it is trivially testable.

    Args:
        stdout: Combined stdout+stderr from the push command.
        returncode: Process exit code (non-zero with no known marker -> error).

    Returns:
        One of :data:`PUSH_PUSHED`, :data:`PUSH_QUEUED`,
        :data:`PUSH_SLUG_POISONED`, :data:`PUSH_ERROR`.
    """
    low = (stdout or "").lower()
    if _GPU_CAP_MARKER in low:
        return PUSH_QUEUED
    if _SLUG_POISON_MARKER in low:
        return PUSH_SLUG_POISONED
    # Successful pushes print "Kernel version N successfully pushed".
    if "successfully pushed" in low or "your kernel" in low:
        return PUSH_PUSHED
    if returncode != 0 or "error" in low or "exception" in low:
        return PUSH_ERROR
    # No error markers and zero exit -> assume it went through.
    return PUSH_PUSHED


def parse_kernel_status(stdout: str) -> str:
    """Normalise ``kaggle kernels status`` output to our vocabulary.

    The CLI prints things like ``status "complete"`` / ``status "running"`` /
    ``status "error"`` / ``status "queued"`` / ``status "cancelAcknowledged"``.

    Args:
        stdout: Raw CLI stdout.

    Returns:
        One of the ``STATUS_*`` constants.
    """
    low = (stdout or "").lower()
    # A poisoned slug returns "Notebook not found" forever -> treat as a
    # terminal error so the fleet stops polling it (rather than spinning to
    # max_polls). This is checked before the status-token match because the
    # 404 message has no "status ..." token at all.
    if _SLUG_POISON_MARKER in low:
        return STATUS_ERROR
    # Prefer the quoted token the CLI emits, fall back to loose matching.
    m = re.search(r'status\s+"?([a-z]+)"?', low)
    token = m.group(1) if m else low
    if "complete" in token:
        return STATUS_COMPLETE
    if "cancel" in token:
        return STATUS_CANCEL
    if "error" in token or "fail" in token:
        return STATUS_ERROR
    if "queue" in token:
        return STATUS_QUEUED
    if "running" in token:
        return STATUS_RUNNING
    return STATUS_UNKNOWN


def diagnose_log(log_text: str) -> Tuple[str, str]:
    """Match a kernel log against the known-failure catalogue.

    Accepts either raw text or Kaggle's JSON console stream (a list of
    ``{"data": ...}`` objects, sometimes with ``"stream_name"``); both are
    flattened to plain text before matching.

    Args:
        log_text: Raw ``.log`` content (JSON stream or plain text).

    Returns:
        ``(name, fix)``. ``name`` is the diagnosis key, or ``"unknown"`` if no
        signature matched.
    """
    text = _flatten_log(log_text)
    for diag in _DIAGNOSES:
        if diag.pattern.search(text):
            return diag.name, diag.fix
    return "unknown", "No known failure signature matched; inspect the log manually."


def _flatten_log(log_text: str) -> str:
    """Turn a Kaggle JSON console stream into plain text (best-effort)."""
    if not log_text:
        return ""
    stripped = log_text.lstrip()
    if stripped[:1] in "[{":
        try:
            obj = json.loads(stripped)
        except (ValueError, json.JSONDecodeError):
            # Kaggle frequently returns NDJSON (one {"data": ...} object per
            # line) which fails a whole-text json.loads; fall back to parsing
            # line by line so the per-record "data" extraction still works.
            ndjson = _flatten_ndjson(log_text)
            return ndjson if ndjson is not None else log_text
        parts: List[str] = []
        items = obj if isinstance(obj, list) else [obj]
        for item in items:
            if isinstance(item, dict):
                data = item.get("data")
                if data is not None:
                    parts.append(str(data))
                else:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)
    return log_text


def _flatten_ndjson(log_text: str) -> Optional[str]:
    """Flatten NDJSON (one JSON object per line) to text, or ``None``.

    Returns the joined ``data`` fields if *any* line parsed as a JSON object;
    ``None`` if no line was valid JSON (so the caller can fall back to raw
    text). Non-JSON lines are kept verbatim so error substrings survive even in
    a mixed stream.
    """
    parts: List[str] = []
    saw_json = False
    for line in log_text.splitlines():
        s = line.strip()
        if not s:
            continue
        try:
            item = json.loads(s)
        except (ValueError, json.JSONDecodeError):
            parts.append(line)  # keep raw line (may contain the error text)
            continue
        saw_json = True
        if isinstance(item, dict):
            data = item.get("data")
            parts.append(str(data) if data is not None else json.dumps(item))
        else:
            parts.append(str(item))
    if not saw_json:
        return None
    return "\n".join(parts)


def gpu_cap_admits(
    status_map: Dict[str, str],
    gpu_kernels: "set[str] | frozenset[str]",
    gpu_cap: int = 2,
) -> bool:
    """Decide whether a new GPU kernel may be pushed right now.

    Pure scheduling predicate, kept separate so the queue logic is testable
    without any CLI calls.

    Args:
        status_map: ``slug -> normalised status`` for every kernel under watch.
        gpu_kernels: The set of slugs that require a GPU.
        gpu_cap: Max concurrent GPU batch sessions (Kaggle hard cap = 2).

    Returns:
        ``True`` iff the number of *non-terminal* GPU kernels is below the cap.
    """
    active_gpu = sum(
        1
        for slug in gpu_kernels
        if status_map.get(slug, STATUS_UNKNOWN) not in TERMINAL_STATUSES
    )
    return active_gpu < gpu_cap


# ---------------------------------------------------------------------------
# Per-kernel bookkeeping for the fleet loop
# ---------------------------------------------------------------------------

@dataclass
class KernelRecord:
    """Mutable per-kernel state tracked across the :meth:`run_fleet` loop."""

    dir: Path
    slug: str
    enable_gpu: bool
    state: str = "pending"  # pending|pushed|running|complete|error|queued|poisoned
    status: str = STATUS_UNKNOWN
    verified: Optional[bool] = None
    diagnosis: Optional[Tuple[str, str]] = None
    files_ok: List[str] = field(default_factory=list)
    files_bad: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """JSON-friendly snapshot for the run_fleet return value."""
        return {
            "slug": self.slug,
            "dir": str(self.dir),
            "enable_gpu": self.enable_gpu,
            "state": self.state,
            "status": self.status,
            "verified": self.verified,
            "diagnosis": list(self.diagnosis) if self.diagnosis else None,
            "files_ok": list(self.files_ok),
            "files_bad": list(self.files_bad),
            "notes": list(self.notes),
        }


def read_kernel_metadata(kernel_dir: Path) -> dict:
    """Read and parse ``kernel-metadata.json`` from a kernel directory."""
    meta_path = Path(kernel_dir) / "kernel-metadata.json"
    with open(meta_path, "r") as f:
        return json.load(f)


def bump_slug(slug: str) -> str:
    """Return a fresh hyphenated slug to escape a poisoned one.

    A trailing ``-vN`` is incremented; otherwise ``-v2`` is appended. Any
    underscores are converted to hyphens (Kaggle slugs never use underscores).

    Examples:
        ``cindyxue1122/s6e6-catv3`` -> ``cindyxue1122/s6e6-catv3-v2``
        ``cindyxue1122/s6e6-catv3-v2`` -> ``cindyxue1122/s6e6-catv3-v3``
    """
    slug = slug.replace("_", "-")
    m = re.search(r"-v(\d+)$", slug)
    if m:
        n = int(m.group(1)) + 1
        return slug[: m.start()] + f"-v{n}"
    return f"{slug}-v2"


# ---------------------------------------------------------------------------
# The fleet
# ---------------------------------------------------------------------------

class KernelFleet:
    """Drive a fleet of Kaggle kernels through their full lifecycle.

    Wraps the ``kaggle`` CLI via subprocess. Honours the GPU batch-session cap,
    recovers from poisoned slugs, verifies pulled ``.npy`` artifacts by shape,
    and diagnoses known failure modes from kernel logs.

    Args:
        out_dir: Directory pulled outputs are written to.
        dry_run: If True, no CLI commands run (prints intent, returns "").
        max_cli_retries: Transient CLI failures retried this many times.
        backoff_base_sec: Base for exponential backoff between CLI retries.
    """

    def __init__(
        self,
        out_dir: Path = Path("artifacts"),
        dry_run: bool = False,
        max_cli_retries: int = 3,
        backoff_base_sec: float = 5.0,
    ):
        self.out_dir = Path(out_dir)
        self.dry_run = dry_run
        self.max_cli_retries = max_cli_retries
        self.backoff_base_sec = backoff_base_sec

    # -- low-level CLI -----------------------------------------------------

    def _run(self, args: List[str], retry: bool = True) -> subprocess.CompletedProcess:
        """Run a ``kaggle`` CLI command with retry/backoff on transient errors.

        Combines stdout+stderr into ``.stdout`` so callers (and the classifiers)
        see Kaggle's messages wherever the CLI happened to print them. Never
        raises on a non-zero exit; returns the CompletedProcess so the caller
        can classify the output.

        Args:
            args: Arguments after ``kaggle`` (e.g. ``["kernels", "push", ...]``).
            retry: Whether to retry transient-looking failures.

        Returns:
            ``subprocess.CompletedProcess`` with merged output in ``.stdout``.
        """
        if self.dry_run:
            print(f"[DRY RUN] Would run: kaggle {' '.join(args)}")
            return subprocess.CompletedProcess(args, 0, stdout="", stderr="")

        # Clamp to >=1 so a misconfigured max_cli_retries=0 still runs once
        # (otherwise the loop body never executes and we'd return None, which
        # crashes callers that read proc.stdout).
        attempts = max(1, self.max_cli_retries if retry else 1)
        last: Optional[subprocess.CompletedProcess] = None
        for attempt in range(attempts):
            proc = subprocess.run(
                ["kaggle"] + args,
                capture_output=True,
                text=True,
                check=False,
            )
            merged = (proc.stdout or "") + (proc.stderr or "")
            # Strip the noisy Kaggle API version warning line.
            merged = "\n".join(
                ln for ln in merged.splitlines() if "Warning:" not in ln
            )
            proc = subprocess.CompletedProcess(proc.args, proc.returncode, merged, "")
            last = proc
            if proc.returncode == 0 or not self._is_transient(merged):
                return proc
            sleep = self.backoff_base_sec * (2 ** attempt)
            print(f"  transient CLI error (attempt {attempt + 1}/{attempts}); "
                  f"retrying in {sleep:.0f}s")
            time.sleep(sleep)
        return last  # type: ignore[return-value]

    @staticmethod
    def _is_transient(text: str) -> bool:
        """Heuristic: is this CLI failure worth retrying?"""
        low = text.lower()
        transient = (
            "timed out", "timeout", "connection", "temporarily",
            "503", "502", "500", "rate limit", "too many requests",
            "service unavailable", "reset by peer",
        )
        # A poisoned slug or GPU cap is NOT transient — handled by the caller.
        if _SLUG_POISON_MARKER in low or _GPU_CAP_MARKER in low:
            return False
        return any(t in low for t in transient)

    # -- push --------------------------------------------------------------

    def push(self, kernel_dir: Path) -> Tuple[str, str]:
        """Push one kernel, classifying the outcome.

        On a poisoned slug, writes a *copy* of the directory's metadata with a
        bumped hyphenated id and retries the push exactly once under the new
        slug.

        Args:
            kernel_dir: Directory containing ``kernel-metadata.json`` + code.

        Returns:
            ``(classification, slug)`` where ``classification`` is one of the
            ``PUSH_*`` constants and ``slug`` is the (possibly new) kernel slug.
        """
        kernel_dir = Path(kernel_dir)
        meta = read_kernel_metadata(kernel_dir)
        slug = meta["id"]

        proc = self._run(["kernels", "push", "-p", str(kernel_dir)])
        result = classify_push_output(proc.stdout, proc.returncode)

        if result == PUSH_SLUG_POISONED:
            new_slug = bump_slug(slug)
            print(f"  slug poisoned ({slug}) -> retrying under {new_slug}")
            self._rewrite_slug(kernel_dir, new_slug)
            proc2 = self._run(["kernels", "push", "-p", str(kernel_dir)])
            result2 = classify_push_output(proc2.stdout, proc2.returncode)
            return result2, new_slug

        return result, slug

    def _rewrite_slug(self, kernel_dir: Path, new_slug: str) -> None:
        """Persist a bumped slug into the kernel's ``kernel-metadata.json``."""
        if self.dry_run:
            print(f"[DRY RUN] Would rewrite id -> {new_slug} in {kernel_dir}")
            return
        meta_path = Path(kernel_dir) / "kernel-metadata.json"
        meta = read_kernel_metadata(kernel_dir)
        meta["id"] = new_slug
        # Title is constrained to <=50 chars; keep it aligned with the slug.
        meta["title"] = new_slug.split("/")[-1][:50]
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)

    # -- status ------------------------------------------------------------

    def status(self, slug: str) -> str:
        """Return the normalised status of a kernel slug.

        Args:
            slug: ``owner/kernel-name``.

        Returns:
            One of the ``STATUS_*`` constants.
        """
        proc = self._run(["kernels", "status", slug])
        return parse_kernel_status(proc.stdout)

    # -- pull + verify -----------------------------------------------------

    def pull_and_verify(
        self,
        slug: str,
        expected: Dict[str, Tuple[int, int]],
        out_dir: Optional[Path] = None,
        retries: int = 3,
    ) -> dict:
        """Pull a kernel's outputs and verify expected ``.npy`` shapes.

        Outputs can be truncated/partial, so every expected file is
        ``np.load``-ed and shape-checked; on any missing / unreadable /
        shape-mismatched file the outputs are re-pulled, up to ``retries``.

        Args:
            slug: Kernel slug to pull from.
            expected: ``filename -> (rows, cols)`` shape contract. e.g.
                ``{"oof_cat.npy": (577347, 3), "test_cat.npy": (247435, 3)}``.
            out_dir: Where to write outputs (defaults to ``self.out_dir``).
            retries: Max pull attempts.

        Returns:
            ``{"ok": bool, "files_ok": [...], "files_bad": [...]}``.
        """
        dest = Path(out_dir) if out_dir is not None else self.out_dir
        dest.mkdir(parents=True, exist_ok=True)

        files_ok: List[str] = []
        files_bad: List[str] = []
        for attempt in range(retries):
            self._run(["kernels", "output", slug, "-p", str(dest)])
            files_ok, files_bad = verify_outputs(dest, expected)
            if not files_bad:
                return {"ok": True, "files_ok": files_ok, "files_bad": []}
            print(f"  verify failed for {slug}: bad={files_bad} "
                  f"(attempt {attempt + 1}/{retries})")
            if attempt + 1 < retries:
                time.sleep(self.backoff_base_sec)
        return {"ok": False, "files_ok": files_ok, "files_bad": files_bad}

    # -- diagnose ----------------------------------------------------------

    def diagnose(self, slug: str, out_dir: Optional[Path] = None) -> Tuple[str, str]:
        """Pull a kernel's log and match it against known failure signatures.

        Args:
            slug: Kernel slug.
            out_dir: Where ``kernels output`` writes (and where the ``.log`` is
                read from). Defaults to ``self.out_dir``. Callers that pull into
                a different directory (e.g. ``run_fleet``'s ``dest``) must pass
                it here, otherwise the ``.log`` is written there but read from
                ``self.out_dir`` and diagnosis silently degrades to "unknown".

        Returns:
            ``(name, fix)`` from :func:`diagnose_log`.
        """
        dest = Path(out_dir) if out_dir is not None else self.out_dir
        proc = self._run(["kernels", "output", slug, "-p", str(dest)])
        # ``kernels output`` writes a ``<kernel>.log`` JSON console stream.
        log_text = self._read_log_file(slug, dest) or proc.stdout
        return diagnose_log(log_text)

    def _read_log_file(self, slug: str, out_dir: Optional[Path] = None) -> str:
        """Best-effort read of the pulled ``<kernel>.log`` console stream."""
        dest = Path(out_dir) if out_dir is not None else self.out_dir
        name = slug.split("/")[-1]
        for candidate in (
            dest / f"{name}.log",
            dest / "kernel.log",
        ):
            if candidate.exists():
                try:
                    return candidate.read_text()
                except OSError:
                    pass
        return ""

    # -- the full loop -----------------------------------------------------

    def run_fleet(
        self,
        kernels: List[Path],
        expected_outputs: Dict[str, Dict[str, Tuple[int, int]]],
        out_dir: Optional[Path] = None,
        gpu_cap: int = 2,
        poll_interval: int = 120,
        repush_on_verify_fail: bool = False,
        max_polls: Optional[int] = None,
    ) -> Dict[str, dict]:
        """Run a fleet of kernels end-to-end.

        Maintains a pending queue: a GPU kernel is only pushed while the number
        of running GPU kernels is below ``gpu_cap`` (CPU kernels bypass the cap,
        per Kaggle's rule). All running kernels are polled each cycle; on
        terminal ``complete`` the outputs are pulled+verified, on terminal
        ``error``/``cancel`` (or a failed verify) the log is diagnosed. The loop
        runs until every kernel is terminal.

        Args:
            kernels: Kernel directories (each with ``kernel-metadata.json``).
            expected_outputs: ``slug -> {filename -> (rows, cols)}``. A kernel
                with no entry (or an empty dict) skips verification.
            out_dir: Output directory (defaults to ``self.out_dir``).
            gpu_cap: Max concurrent GPU sessions.
            poll_interval: Seconds between poll cycles.
            repush_on_verify_fail: If a completed kernel fails verification,
                re-push it (once) to try again.
            max_polls: Safety bound on poll cycles (``None`` = unbounded).

        Returns:
            ``slug -> KernelRecord.to_dict()`` for every kernel.
        """
        dest = Path(out_dir) if out_dir is not None else self.out_dir
        # In dry-run the stubbed status() never reaches a terminal state, so
        # bound the loop to a couple of cycles to keep the demo finite.
        if self.dry_run and max_polls is None:
            max_polls = 2
        records: Dict[str, KernelRecord] = {}
        for kdir in kernels:
            kdir = Path(kdir)
            meta = read_kernel_metadata(kdir)
            slug = meta["id"]
            records[slug] = KernelRecord(
                dir=kdir,
                slug=slug,
                enable_gpu=bool(meta.get("enable_gpu", False)),
            )

        pending = list(records.values())  # not yet pushed
        repushed: "set[str]" = set()
        polls = 0

        # States meaning "this GPU kernel has been launched and is occupying a
        # batch slot". Pending/queued kernels do NOT count: a "queued" record is
        # one we tried (or were blocked) to push but that never took a slot, so
        # it must stay eligible to re-push without counting against the cap.
        launched_states = {"pushed", STATUS_RUNNING}

        while True:
            # 1) Push as many pending kernels as the GPU cap allows.
            #    The cap is measured only over kernels we have actually launched
            #    (status_map is built from launched GPU kernels), so kernels
            #    still in the pending queue never count against themselves.
            def _launched_gpu_status_map() -> Dict[str, str]:
                m: Dict[str, str] = {}
                for r in records.values():
                    if not r.enable_gpu:
                        continue
                    if r.state in launched_states or r.status in TERMINAL_STATUSES:
                        m[r.slug] = r.status
                return m

            gpu_slugs = set(_launched_gpu_status_map().keys())
            still_pending: List[KernelRecord] = []
            for rec in pending:
                if rec.enable_gpu and not gpu_cap_admits(
                    _launched_gpu_status_map(), gpu_slugs, gpu_cap
                ):
                    # Pre-push gate: cap already full, keep it pending.
                    rec.state = "queued"
                    still_pending.append(rec)
                    continue
                self._push_record(rec, records)
                # A real push can still come back PUSH_QUEUED if the GPU batch
                # cap was hit at push time (a race the pre-push gate cannot see).
                # Such a record must go BACK into the pending queue so a later
                # cycle re-pushes it -- otherwise it is neither pushed nor polled
                # and the fleet spins to max_polls (the bash poll_wave2.sh it
                # replaces explicitly re-pushed when a slot freed).
                if rec.state == "queued":
                    still_pending.append(rec)
                else:
                    gpu_slugs.add(rec.slug)  # launched; now counts toward cap
            pending = still_pending

            # 2) Poll every non-terminal kernel.
            for rec in records.values():
                if rec.state in ("pending", "queued"):
                    continue
                if rec.status in TERMINAL_STATUSES:
                    continue
                rec.status = self.status(rec.slug)
                rec.state = rec.status
                if rec.status == STATUS_COMPLETE:
                    self._handle_complete(
                        rec, expected_outputs, dest,
                        repush_on_verify_fail, repushed, records, pending,
                    )
                elif rec.status in (STATUS_ERROR, STATUS_CANCEL):
                    rec.diagnosis = self.diagnose(rec.slug, dest)
                    rec.notes.append(f"terminal {rec.status}: {rec.diagnosis[0]}")

            # 3) Termination check: nothing pending AND all kernels terminal.
            all_terminal = (
                not pending
                and all(r.status in TERMINAL_STATUSES for r in records.values())
            )
            if all_terminal:
                break

            polls += 1
            if max_polls is not None and polls >= max_polls:
                for r in records.values():
                    if r.status not in TERMINAL_STATUSES:
                        r.notes.append("max_polls reached before terminal")
                break
            # Sleep only for real runs; dry-run / tests iterate immediately and
            # rely on the terminal-status check (or max_polls) to stop.
            if not self.dry_run:
                time.sleep(poll_interval)

        return {slug: rec.to_dict() for slug, rec in records.items()}

    def _push_record(
        self, rec: KernelRecord, records: Dict[str, KernelRecord]
    ) -> None:
        """Push one record and fold the result back into the fleet."""
        result, slug = self.push(rec.dir)
        if slug != rec.slug:
            # Slug was bumped; re-key the record map.
            records.pop(rec.slug, None)
            rec.slug = slug
            records[slug] = rec
            rec.notes.append(f"re-pushed under bumped slug {slug}")
        if result == PUSH_PUSHED:
            rec.state = "pushed"
            rec.status = STATUS_RUNNING  # treat as active until first poll
        elif result == PUSH_QUEUED:
            rec.state = "queued"
            rec.status = STATUS_UNKNOWN
            rec.notes.append("GPU batch cap hit at push; queued")
        else:  # PUSH_ERROR or still poisoned after retry
            rec.state = "error"
            rec.status = STATUS_ERROR
            rec.notes.append(f"push classified as {result}")

    def _handle_complete(
        self,
        rec: KernelRecord,
        expected_outputs: Dict[str, Dict[str, Tuple[int, int]]],
        dest: Path,
        repush_on_verify_fail: bool,
        repushed: "set[str]",
        records: Dict[str, KernelRecord],
        pending: List[KernelRecord],
    ) -> None:
        """Pull+verify a completed kernel; diagnose / optionally re-push.

        On a verify failure that warrants a re-push, the record is returned to
        ``pending`` (rather than pushed immediately) so the next loop cycle
        re-pushes it through the GPU-cap gate -- a direct ``_push_record`` here
        would bypass the cap and could exceed the 2-session GPU limit.
        """
        expected = expected_outputs.get(rec.slug, {})
        if not expected:
            rec.verified = True
            rec.notes.append("complete; no output verification requested")
            return
        res = self.pull_and_verify(rec.slug, expected, dest)
        rec.verified = res["ok"]
        rec.files_ok = res["files_ok"]
        rec.files_bad = res["files_bad"]
        if res["ok"]:
            rec.notes.append("complete and verified")
            return
        rec.diagnosis = self.diagnose(rec.slug, dest)
        rec.notes.append(f"verify FAILED: {rec.diagnosis[0]}")
        if repush_on_verify_fail and rec.slug not in repushed:
            repushed.add(rec.slug)
            rec.notes.append("re-queued after verify failure")
            rec.status = STATUS_UNKNOWN
            rec.state = "pending"
            if rec not in pending:
                pending.append(rec)

    # -- dataset mirroring -------------------------------------------------

    def mirror_dataset(
        self,
        source_slug: str,
        owner: str = "cindyxue1122",
        title: Optional[str] = None,
        work_dir: Optional[Path] = None,
    ) -> str:
        """Mirror a (possibly 3rd-party) dataset under your own account.

        Attaching a 3rd-party dataset directly as a kernel ``dataset_source``
        can fail; the workaround is to download it and re-create/version it under
        your own account, then attach the copy. This downloads ``source_slug``
        and creates (or versions) ``owner/<name>`` from the files.

        Args:
            source_slug: ``owner/name`` of the dataset to copy.
            owner: Your Kaggle username (target owner).
            title: Optional title; defaults to the source name.
            work_dir: Scratch directory for the download (defaults under
                ``out_dir``).

        Returns:
            The new dataset slug ``owner/<name>``.
        """
        src_name = source_slug.split("/")[-1].replace("_", "-")
        new_slug = f"{owner}/{src_name}"
        work = Path(work_dir) if work_dir else (self.out_dir / f"_mirror_{src_name}")
        work.mkdir(parents=True, exist_ok=True)

        # Download + unzip the source files.
        self._run(["datasets", "download", source_slug, "-p", str(work), "--unzip"])

        # Write dataset metadata for the new copy.
        meta = {
            "id": new_slug,
            "title": (title or src_name)[:50],
            "licenses": [{"name": "CC0-1.0"}],
        }
        if not self.dry_run:
            with open(work / "dataset-metadata.json", "w") as f:
                json.dump(meta, f, indent=2)

        # Create new, or version if it already exists.
        proc = self._run(["datasets", "create", "-p", str(work), "--dir-mode", "zip"])
        if "already exists" in proc.stdout.lower():
            self._run([
                "datasets", "version", "-p", str(work),
                "-m", f"mirror of {source_slug}", "--dir-mode", "zip",
            ])
        return new_slug


# ---------------------------------------------------------------------------
# Output verification (pure logic; used by pull_and_verify and tests)
# ---------------------------------------------------------------------------

def verify_outputs(
    out_dir: Path,
    expected: Dict[str, Tuple[int, int]],
) -> Tuple[List[str], List[str]]:
    """Load each expected ``.npy`` and check its shape.

    Args:
        out_dir: Directory the kernel outputs were pulled into.
        expected: ``filename -> (rows, cols)`` contract.

    Returns:
        ``(files_ok, files_bad)`` lists of filenames. A file is *bad* if it is
        missing, unreadable, or has the wrong shape.
    """
    import numpy as np

    out_dir = Path(out_dir)
    files_ok: List[str] = []
    files_bad: List[str] = []
    for fname, want_shape in expected.items():
        path = out_dir / fname
        if not path.exists():
            files_bad.append(fname)
            continue
        try:
            arr = np.load(path)
        except (ValueError, OSError, EOFError):
            files_bad.append(fname)  # truncated / corrupt
            continue
        if tuple(arr.shape) == tuple(want_shape):
            files_ok.append(fname)
        else:
            files_bad.append(fname)
    return files_ok, files_bad


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python -m kaggle_agent.tools.kernel_fleet",
        description="Drive a fleet of Kaggle kernels (push/poll/pull/verify).",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    run = sub.add_parser("run", help="Run a fleet of kernels to completion.")
    run.add_argument("--dirs", nargs="+", required=True,
                     help="Kernel directories (each with kernel-metadata.json).")
    run.add_argument("--gpu-cap", type=int, default=2,
                     help="Max concurrent GPU sessions (Kaggle hard cap = 2).")
    run.add_argument("--out-dir", default="artifacts",
                     help="Where to pull outputs.")
    run.add_argument("--poll-interval", type=int, default=120,
                     help="Seconds between poll cycles.")
    run.add_argument("--expected", default=None,
                     help="JSON file mapping slug -> {filename: [rows, cols]}.")
    run.add_argument("--repush-on-verify-fail", action="store_true",
                     help="Re-push a completed kernel if verification fails.")
    run.add_argument("--dry-run", action="store_true",
                     help="Print intended CLI calls without running them.")

    push = sub.add_parser("push", help="Push a single kernel directory.")
    push.add_argument("--dir", required=True)
    push.add_argument("--dry-run", action="store_true")

    st = sub.add_parser("status", help="Print normalised status of a slug.")
    st.add_argument("--slug", required=True)

    diag = sub.add_parser("diagnose", help="Diagnose a kernel's log.")
    diag.add_argument("--slug", required=True)
    diag.add_argument("--out-dir", default="artifacts")

    return p


def _load_expected(path: Optional[str]) -> Dict[str, Dict[str, Tuple[int, int]]]:
    """Load the --expected JSON, coercing shape lists to tuples."""
    if not path:
        return {}
    with open(path, "r") as f:
        raw = json.load(f)
    return {
        slug: {fn: tuple(shape) for fn, shape in files.items()}
        for slug, files in raw.items()
    }


def main(argv: Optional[List[str]] = None) -> int:
    """CLI entry point."""
    args = _build_arg_parser().parse_args(argv)

    if args.cmd == "run":
        fleet = KernelFleet(out_dir=Path(args.out_dir), dry_run=args.dry_run)
        result = fleet.run_fleet(
            kernels=[Path(d) for d in args.dirs],
            expected_outputs=_load_expected(args.expected),
            gpu_cap=args.gpu_cap,
            poll_interval=args.poll_interval,
            repush_on_verify_fail=args.repush_on_verify_fail,
        )
        print(json.dumps(result, indent=2))
        return 0

    if args.cmd == "push":
        fleet = KernelFleet(dry_run=args.dry_run)
        result, slug = fleet.push(Path(args.dir))
        print(json.dumps({"result": result, "slug": slug}))
        return 0

    if args.cmd == "status":
        fleet = KernelFleet()
        print(fleet.status(args.slug))
        return 0

    if args.cmd == "diagnose":
        fleet = KernelFleet(out_dir=Path(args.out_dir))
        name, fix = fleet.diagnose(args.slug)
        print(json.dumps({"diagnosis": name, "fix": fix}, indent=2))
        return 0

    return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
