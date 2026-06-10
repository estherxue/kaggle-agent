"""Kaggle API client for downloading competitions and submitting results.

Wraps the official Kaggle API for convenience.
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Optional


@dataclass
class Submission:
    """A Kaggle submission record."""

    submission_id: str
    date: datetime
    description: str
    score: Optional[float]
    rank: Optional[int]
    status: str


@dataclass
class Leaderboard:
    """Kaggle leaderboard data."""

    competition_id: str
    entries: List[dict]  # Raw leaderboard data


@dataclass
class CompetitionInfo:
    """Competition metadata."""

    id: str
    title: str
    description: str
    evaluation_metric: str
    reward: str
    deadline: datetime
    category: str
    tags: List[str]
    is_kernels_only: bool


class KaggleClient:
    """Client for Kaggle API operations.

    Uses the official kaggle command-line tool for API access.
    Make sure KAGGLE_USERNAME and KAGGLE_KEY are set in environment.
    """

    def __init__(self, dry_run: bool = False):
        """Initialize Kaggle client.

        Args:
            dry_run: If True, don't actually make API calls (for testing)
        """
        self.dry_run = dry_run

    def _run_command(self, args: List[str], check: bool = True) -> str:
        """Run a kaggle CLI command.

        Args:
            args: Command arguments (after 'kaggle')
            check: Whether to raise on non-zero exit

        Returns:
            Command stdout
        """
        if self.dry_run:
            print(f"[DRY RUN] Would run: kaggle {' '.join(args)}")
            return ""

        cmd = ["kaggle"] + args
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=check,
        )
        return result.stdout

    def download_competition(self, slug: str, dest: Path) -> Path:
        """Download competition data files.

        Args:
            slug: Competition slug (e.g., 'titanic')
            dest: Destination directory

        Returns:
            Path to downloaded data directory
        """
        dest.mkdir(parents=True, exist_ok=True)

        self._run_command([
            "competitions", "download",
            "-c", slug,
            "-p", str(dest),
            "--force",  # Overwrite existing
        ])

        # Unzip the downloaded files
        zip_file = dest / f"{slug}.zip"
        if zip_file.exists():
            import zipfile
            with zipfile.ZipFile(zip_file, "r") as zf:
                zf.extractall(dest)
            zip_file.unlink()

        return dest

    def get_competition_info(self, slug: str) -> CompetitionInfo:
        """Get competition metadata.

        Args:
            slug: Competition slug

        Returns:
            CompetitionInfo object
        """
        output = self._run_command([
            "competitions", "list",
            "--search", slug,
            "-v",  # Verbose
        ])

        # Parse the output (Kaggle CLI returns table format)
        # For now, return basic info
        # TODO: Parse actual output properly
        return CompetitionInfo(
            id=slug,
            title=slug,
            description="",
            evaluation_metric="",
            reward="",
            deadline=datetime.now(),
            category="",
            tags=[],
            is_kernels_only=False,
        )

    def submit(
        self,
        slug: str,
        file_path: Path,
        message: str,
    ) -> str:
        """Submit predictions to a competition.

        Args:
            slug: Competition slug
            file_path: Path to submission CSV
            message: Submission description

        Returns:
            Submission ID
        """
        self._run_command([
            "competitions", "submit",
            "-c", slug,
            "-f", str(file_path),
            "-m", message,
        ])

        # Get the submission ID from recent submissions
        submissions = self.get_my_submissions(slug, limit=1)
        if submissions:
            return submissions[0].submission_id
        return "unknown"

    def get_my_submissions(
        self,
        slug: str,
        limit: int = 10,
    ) -> List[Submission]:
        """Get my submissions for a competition.

        Args:
            slug: Competition slug
            limit: Maximum number of submissions to return

        Returns:
            List of Submission objects
        """
        output = self._run_command([
            "competitions", "submissions",
            "-c", slug,
            "-v",  # CSV format
        ])

        # Parse CSV output
        submissions = []
        lines = output.strip().split("\n")

        # Skip header
        for line in lines[1:limit+1]:
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 4:
                submissions.append(Submission(
                    submission_id=parts[0],
                    date=datetime.now(),  # Parse from parts[1]
                    description=parts[2],
                    score=float(parts[3]) if parts[3] else None,
                    rank=None,
                    status=parts[4] if len(parts) > 4 else "complete",
                ))

        return submissions

    def get_leaderboard(self, slug: str) -> Leaderboard:
        """Get public leaderboard.

        Args:
            slug: Competition slug

        Returns:
            Leaderboard data
        """
        output = self._run_command([
            "competitions", "leaderboard",
            "-c", slug,
            "-v",
        ])

        # Parse output
        entries = []
        lines = output.strip().split("\n")

        for line in lines[1:]:  # Skip header
            if not line.strip():
                continue
            parts = line.split(",")
            if len(parts) >= 3:
                entries.append({
                    "rank": parts[0],
                    "team": parts[1],
                    "score": parts[2],
                })

        return Leaderboard(
            competition_id=slug,
            entries=entries,
        )

    def download_previous_submission(
        self,
        submission_id: str,
        dest: Path,
    ) -> Path:
        """Download a previous submission file.

        Args:
            submission_id: Submission ID
            dest: Destination directory

        Returns:
            Path to downloaded file
        """
        dest.mkdir(parents=True, exist_ok=True)

        self._run_command([
            "kernels", "output",
            submission_id,
            "-p", str(dest),
        ])

        return dest


class MockKaggleClient(KaggleClient):
    """Mock Kaggle client for testing."""

    def __init__(self):
        super().__init__(dry_run=True)
        self.downloaded_competitions: List[str] = []
        self.submissions: List[tuple] = []

    def download_competition(self, slug: str, dest: Path) -> Path:
        """Mock download - creates fake data files."""
        self.downloaded_competitions.append(slug)
        dest.mkdir(parents=True, exist_ok=True)

        # Create fake data files for testing
        import pandas as pd

        # Fake train.csv
        pd.DataFrame({
            "id": [1, 2, 3],
            "feature": [0.5, 0.6, 0.7],
            "target": [0, 1, 0],
        }).to_csv(dest / "train.csv", index=False)

        # Fake test.csv
        pd.DataFrame({
            "id": [4, 5, 6],
            "feature": [0.55, 0.65, 0.75],
        }).to_csv(dest / "test.csv", index=False)

        # Fake sample_submission.csv
        pd.DataFrame({
            "id": [4, 5, 6],
            "target": [0, 0, 0],
        }).to_csv(dest / "sample_submission.csv", index=False)

        return dest

    def submit(
        self,
        slug: str,
        file_path: Path,
        message: str,
    ) -> str:
        """Mock submit."""
        self.submissions.append((slug, file_path, message))
        return f"mock-submission-{len(self.submissions)}"

    def get_my_submissions(self, slug: str, limit: int = 10) -> List[Submission]:
        """Mock submissions."""
        return [
            Submission(
                submission_id="mock-1",
                date=datetime.now(),
                description="Mock submission",
                score=0.85,
                rank=100,
                status="complete",
            )
        ]
