"""Download playground-series-s6e6 data via kagglehub."""

from pathlib import Path

import kagglehub

SLUG = "playground-series-s6e6"
DEST = Path(__file__).parent / "data"


def main() -> Path:
    DEST.mkdir(parents=True, exist_ok=True)
    path = kagglehub.competition_download(SLUG, output_dir=str(DEST), force_download=True)
    print("Path to competition files:", path)
    return Path(path)


if __name__ == "__main__":
    main()
