"""Download and verify the Omniglot features published with Singh et al."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path
from urllib.request import urlopen

from .omniglot_data import FEATURE_SHA256


FEATURE_URL = (
    "https://raw.githubusercontent.com/aadityasingh/icl-dynamics/main/"
    "omniglot_resnet18_randomized_order_s0.h5"
)
DEFAULT_PATH = Path("data/omniglot_resnet18_randomized_order_s0.h5")


def download(path: Path) -> None:
    if path.exists():
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest == FEATURE_SHA256:
            print(f"Verified existing feature file: {path}")
            return
        raise ValueError(f"Existing feature file has an unexpected checksum: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".download")
    digest = hashlib.sha256()
    try:
        with urlopen(FEATURE_URL, timeout=60) as source, temporary.open("wb") as target:
            for chunk in iter(lambda: source.read(1024 * 1024), b""):
                target.write(chunk)
                digest.update(chunk)
        if digest.hexdigest() != FEATURE_SHA256:
            raise ValueError(f"Downloaded feature checksum mismatch: {digest.hexdigest()}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded and verified feature file: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()
    download(args.output)


if __name__ == "__main__":
    main()
