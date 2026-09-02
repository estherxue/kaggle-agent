#!/usr/bin/env python3
"""experience_db — a searchable, evidence-graded ledger for long-horizon agent campaigns.

Generalised from the tool that carried a 400-target ONNX code-golf campaign
(neurogolf-2026). Domain code stripped; the two ideas that actually mattered kept.

WHY IT EXISTS
-------------
Across ~10 agent waves we repeatedly

  (a) rebuilt candidates a prior wave had already built,
  (b) re-derived negative results a prior wave had already proven, and
  (c) propagated UNVERIFIED numbers as fact.

(c) was the expensive one. A report file listed an estimated score in a column that
looked identical to hundreds of genuinely measured rows; an agent read it, concluded
five artifacts were broken, we shipped a "fix", and the authoritative metric went
DOWN by exactly the difference. The incumbents had been fine all along.

TWO RULES FOLLOW
----------------
1. STORE FAILURES AND NEGATIVE RESULTS, not just wins. A refuted direction is an
   asset: it stops the next agent spending a wave re-refuting it.

2. EVERY NUMBER CARRIES AN EVIDENCE LEVEL. A number is never just a number; it is a
   number plus how it was obtained. Levels are ordered and must never be conflated:

     estimated    derived or predicted; nothing was executed
     local_pass   passes the locally available tests / provided examples
     holdout_clean  + passes freshly generated or held-out data the author never saw
     authority_ok   + the authoritative checker/scorer accepts it (not your reimplementation)
     measured       + measured on the real production metric (leaderboard, prod eval)

   Nothing below `authority_ok` ships. Nothing at `estimated` may be quoted as fact.
   A measurement taken THROUGH a compatibility shim that alters semantics is recorded
   with instrument="<shim>" and is NOT valid evidence at any level.

3. (corollary, learned later) NO STATIC ESTIMATOR SURVIVES CONTACT. Four successive
   estimates of "remaining headroom" in that campaign read 425.83 -> 44.85 -> 36.70 ->
   1.15 as each was actually checked: a 370x over-promise. So `sched` starts from a
   heavily damped prior and lets MEASURED reward take over as attempts accumulate.

RETRIEVAL IS BY STRUCTURE, NOT JUST IDENTITY
--------------------------------------------
An agent starting on a target with a given bottleneck should find every prior attempt
that hit that same bottleneck, whatever target it was on. Hence `--bottleneck` and
`--sig` search, and `brief` surfacing structurally similar work on OTHER targets.

CLI
---
  experience_db.py add '<json>'            add a record (or @file)
  experience_db.py attempt '<json>'        log one attempt + the reward it produced
  experience_db.py find --target N         records for a target
  experience_db.py find --bottleneck STR   records sharing a bottleneck (substring)
  experience_db.py find --sig STR          records sharing a signature fragment
  experience_db.py brief --target N        pre-work briefing before dispatching an agent
  experience_db.py refuted                 all refuted directions (do NOT re-attempt)
  experience_db.py sched [--top N]         next targets by posterior expected reward
  experience_db.py stats                   database summary

Set EXPDB_DIR to choose where records.jsonl / costs.json live (default: ./_expdb).
"""
import json, os, sys, time, math, glob

DIR = os.environ.get("EXPDB_DIR", os.path.join(os.getcwd(), "_expdb"))
DB = os.path.join(DIR, "records.jsonl")
PRIORS = os.path.join(DIR, "costs.json")

LEVELS = ["estimated", "local_pass", "holdout_clean", "authority_ok", "measured"]
SHIPPABLE = LEVELS.index("authority_ok")


def _load():
    if not os.path.exists(DB):
        return []
    out = []
    for line in open(DB):
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except Exception:
                pass
    return out


def _append(rec):
    os.makedirs(os.path.dirname(DB), exist_ok=True)
    with open(DB, "a") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def add(rec):
    """Record schema — missing fields are fine; unknown beats invented.

      target      int or str   what this is about
      outcome     'win' | 'loss' | 'refuted' | 'broken'
      approach    one line describing the construction
      evidence    one of LEVELS — HOW the numbers below were obtained
      instrument  name of any shim/patch the measurement passed through (else omit)
      score / cost / delta_vs_incumbent
      signature   e.g. the op mix, the model family, the query plan
      bottleneck  WHY it failed or what capped it — the key retrieval field
      family      a grouping label so structurally similar targets find each other
      artifact    path to the produced thing, if kept
      wave        which agent wave / session produced it
      note        anything else
    """
    rec = dict(rec)
    rec.setdefault("ts", int(time.time()))
    ev = rec.get("evidence")
    if ev not in LEVELS:
        raise SystemExit("evidence must be one of %s (got %r)" % (LEVELS, ev))
    if rec.get("instrument") and LEVELS.index(ev) >= LEVELS.index("local_pass"):
        rec["warning"] = ("measured THROUGH %s — a shim is part of the instrument, so this "
                          "is not evidence of correctness; treat as UNKNOWN, not verified"
                          % rec["instrument"])
    _append(rec)
    return rec


def attempt(rec):
    """Log ONE attempt on ONE target, with the reward it ACTUALLY produced.

    This is the signal `sched` runs on, and the reason it exists: the original ledger
    recorded only outcomes, so after ~10 waves just 4 records carried a realized delta
    — not enough to allocate anything on. Log even at reward 0, and even when the agent
    episode was truncated; a truncated episode with no record is how the next agent
    walks into the same wall.

    Required: target, direction (the STRUCTURAL frame, not a label), reward
    (0.0 or negative is a valid and useful reward).
    """
    rec = dict(rec)
    rec["kind"] = "attempt"
    rec.setdefault("ts", int(time.time()))
    for k in ("target", "direction", "reward"):
        if k not in rec:
            raise SystemExit("attempt needs %r (reward may be 0.0 or negative)" % k)
    rec.setdefault("evidence", "estimated")
    rec.setdefault("effort", 1.0)
    _append(rec)
    return rec


def _match(r, target=None, bottleneck=None, sig=None, family=None, outcome=None):
    if target is not None and str(r.get("target")) != str(target):
        return False
    if bottleneck and bottleneck.lower() not in str(r.get("bottleneck", "")).lower():
        return False
    if sig and sig.lower() not in str(r.get("signature", "")).lower():
        return False
    if family and family.lower() not in str(r.get("family", "")).lower():
        return False
    if outcome and r.get("outcome") != outcome:
        return False
    return True


def find(**kw):
    return [r for r in _load() if _match(r, **kw)]


def _fmt(r):
    flag = "  [VIA %s: correctness unverified]" % r["instrument"] if r.get("instrument") else ""
    d = r.get("delta_vs_incumbent")
    dstr = (" %+.3f" % d) if isinstance(d, (int, float)) else ""
    return ("  %-8s %-8s %-14s%s score=%s%s\n     approach: %s\n     bottleneck: %s"
            % (r.get("target", "?"), r.get("outcome", "?"), r.get("evidence", "?"), flag,
               r.get("score", "?"), dstr, str(r.get("approach", ""))[:150],
               str(r.get("bottleneck", ""))[:150]))


def brief(target):
    """Pre-work briefing for an agent about to attack `target`. Paste this INTO the prompt."""
    recs = _load()
    own = [r for r in recs if str(r.get("target")) == str(target)]
    print("=== briefing for target %s ===" % target)
    if not own:
        print("no prior attempts recorded for this target")
    else:
        print("\n-- prior attempts on this target (%d) --" % len(own))
        for r in sorted(own, key=lambda r: -(r.get("score") or 0)):
            print(_fmt(r))
    bns = {str(r.get("bottleneck", "")).strip() for r in own if r.get("bottleneck")}
    fams = {str(r.get("family", "")).strip() for r in own if r.get("family")}
    rel = [r for r in recs if str(r.get("target")) != str(target) and (
        any(b and b.lower() in str(r.get("bottleneck", "")).lower() for b in bns) or
        any(f and f == r.get("family") for f in fams))]
    if rel:
        print("\n-- STRUCTURALLY similar attempts on OTHER targets (%d) --" % len(rel))
        print("   (same bottleneck or same family — read these before designing)")
        for r in rel[:12]:
            print(_fmt(r))
    ref = [r for r in recs if r.get("outcome") == "refuted"]
    if ref:
        print("\n-- REFUTED directions: do NOT re-attempt (%d) --" % len(ref))
        for r in ref:
            print("  * %s\n      why: %s" % (str(r.get("approach", ""))[:110],
                                             str(r.get("bottleneck", ""))[:130]))
    print("\n-- evidence rule --")
    print("  ship only at >= authority_ok; never quote an `estimated` number as fact;")
    print("  a measurement taken through a shim is NOT evidence of correctness.")


def _arms(recs):
    arms = {}
    for r in recs:
        if r.get("kind") != "attempt":
            continue
        a = arms.setdefault(str(r.get("target")), {"n": 0.0, "sum": 0.0, "dirs": {}})
        a["n"] += float(r.get("effort") or 1.0)
        a["sum"] += float(r.get("reward") or 0.0)
        d = str(r.get("direction", "?"))
        a["dirs"][d] = a["dirs"].get(d, 0.0) + float(r.get("reward") or 0.0)
    return arms


def sched(top=20, prior_weight=2.0, explore=0.5, damp=0.25):
    """Rank targets by POSTERIOR expected reward — not by size, not by static headroom.

    `costs.json` maps target -> {"prior": <float>} (a MEASURED expected gain, used
    undamped) and/or {"cost": <float>, "floor": <float>, "complexity": <int>} (a crude
    bound, damped hard because bounds systematically over-promise).

    Shrinkage:  mu = (sum_reward + w*prior) / (n + w), plus a UCB exploration bonus.
    So an untried target gets tried, a target that keeps returning nothing decays out,
    and no single theoretical jackpot can hold the fleet hostage.

    Pruning removes DIRECTIONS, not targets: a failed method is evidence about the
    method. In the source campaign a target logged `refuted` later fell to a different
    frame entirely — twice.
    """
    recs = _load()
    arms = _arms(recs)
    priors = {}
    if os.path.exists(PRIORS):
        priors = json.load(open(PRIORS))
    refuted = {}
    for r in recs:
        if r.get("outcome") == "refuted" and r.get("target") is not None:
            refuted.setdefault(str(r["target"]), []).append(str(r.get("approach", ""))[:70])
    N = max(1.0, sum(a["n"] for a in arms.values()))
    rows = []
    for t, c in priors.items():
        if "prior" in c:
            prior = float(c["prior"])                       # measured: trust it
        else:                                               # bound: damp it hard
            cost = max(1.0, float(c.get("cost", 1)))
            floor = max(1.0, float(c.get("floor", 1)))
            prior = damp * math.log(cost / floor) if cost > floor else 0.0
            prior /= (1.0 + math.log(max(1, int(c.get("complexity", 1)))) / 3.0)
        a = arms.get(t, {"n": 0.0, "sum": 0.0})
        n = a["n"]
        mu = (a["sum"] + prior_weight * prior) / (n + prior_weight)
        rows.append((mu + explore * math.sqrt(math.log(N + 1) / (n + 1)), mu, t, n, a["sum"], prior))
    rows.sort(reverse=True)
    print("=== next targets by posterior expected reward (%d arms) ===" % len(rows))
    print("%-10s %7s %7s %6s %8s  %s" % ("target", "ucb", "mu", "tried", "gained", "pruned directions"))
    for ucb, mu, t, n, s, prior in rows[:top]:
        ref = refuted.get(t, [])
        print("%-10s %7.3f %7.3f %6.1f %8.2f  %s"
              % (t, ucb, mu, n, s, ("%d: %s" % (len(ref), ref[0][:40])) if ref else "-"))
    if not arms:
        print("\nNO attempts logged yet — every arm is running on its prior.")
        print("Log one with:  experience_db.py attempt '{\"target\":N,\"direction\":\"...\",\"reward\":0.0}'")


def stats():
    recs = _load()
    print("records: %d   db: %s" % (len(recs), DB))
    for key in ("outcome", "evidence", "family"):
        c = {}
        for r in recs:
            v = r.get(key)
            if v:
                c[v] = c.get(v, 0) + 1
        if c:
            print("  by %-9s %s" % (key + ":", dict(sorted(c.items(), key=lambda kv: -kv[1]))))
    ins = sum(1 for r in recs if r.get("instrument"))
    if ins:
        print("  %d records measured through a shim (correctness unverified)" % ins)
    print("  attempts logged: %d" % sum(1 for r in recs if r.get("kind") == "attempt"))
    print("  targets covered: %d" % len({str(r.get("target")) for r in recs if r.get("target") is not None}))


def main(argv):
    if not argv:
        print(__doc__)
        return
    cmd, a = argv[0], argv[1:]

    def _json(raw):
        return json.loads(open(raw[1:]).read() if raw.startswith("@") else raw)

    if cmd == "add":
        print("added:", json.dumps(add(_json(a[0])), ensure_ascii=False)[:200])
    elif cmd == "attempt":
        r = attempt(_json(a[0]))
        print("logged: %s %s reward=%+.3f" % (r["target"], str(r["direction"])[:50], float(r["reward"])))
    elif cmd == "find":
        kw = {}
        for i in range(0, len(a) - 1, 2):
            kw[a[i].lstrip("-").replace("-", "_")] = a[i + 1]
        rs = find(**kw)
        print("%d match(es)" % len(rs))
        for r in rs:
            print(_fmt(r))
    elif cmd == "brief":
        brief(a[a.index("--target") + 1] if "--target" in a else a[0])
    elif cmd == "refuted":
        for r in find(outcome="refuted"):
            print(_fmt(r))
    elif cmd == "sched":
        sched(top=int(a[a.index("--top") + 1]) if "--top" in a else 20)
    elif cmd == "stats":
        stats()
    else:
        print(__doc__)


if __name__ == "__main__":
    main(sys.argv[1:])
