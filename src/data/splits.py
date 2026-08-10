"""Official FaceForensics++ train/val/test splits.

FF++ provides splits as JSON lists of video-id pairs, e.g. [["071", "054"], ...].
Real videos:        "071", "054" (each element of each pair)
Manipulated videos: "071_054" and "054_071"

Download the three files from:
https://github.com/ondyari/FaceForensics/tree/master/dataset/splits
(train.json, val.json, test.json) and place them in --splits-dir.
"""
import json
from pathlib import Path

SPLIT_FILES = {"train": "train.json", "val": "val.json", "test": "test.json"}


def load_split_ids(splits_dir: str, split: str):
    """Return (real_ids, fake_ids) for a split.

    real_ids: set of video ids like {"071", "054", ...}
    fake_ids: set of manipulated ids like {"071_054", "054_071", ...}
    """
    path = Path(splits_dir) / SPLIT_FILES[split]
    if not path.exists():
        raise FileNotFoundError(
            f"Split file not found: {path}. Download train/val/test.json from "
            "https://github.com/ondyari/FaceForensics/tree/master/dataset/splits"
        )
    pairs = json.loads(path.read_text())
    real_ids, fake_ids = set(), set()
    for a, b in pairs:
        real_ids.update([a, b])
        fake_ids.update([f"{a}_{b}", f"{b}_{a}"])
    return real_ids, fake_ids
