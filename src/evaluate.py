"""Evaluate a trained checkpoint on any manipulation/compression test set.

This is the core of the generalization study: the checkpoint may have been
trained on a different manipulation type (cross-manipulation) or compression
level (robustness) than the one evaluated here.

Example (train DF -> test F2F):
  python -m src.evaluate --checkpoint runs/xception_DF_c23/best.pt \
      --data-root /data/ffpp_faces --splits-dir splits \
      --methods Face2Face --compression c23 --out results/xception_DF_to_F2F_c23.json
"""
import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import FFPPDataset
from .models.factory import create_model, model_img_size
from .train import evaluate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-root", required=True)
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--compression", default="c23",
                   choices=["c0", "c23", "c40", "c40proxy"])
    p.add_argument("--split", default="test", choices=["val", "test"])
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    state = torch.load(args.checkpoint, map_location=device)
    train_args = state["args"]
    model = create_model(train_args["model"], pretrained=False).to(device)
    model.load_state_dict(state["model"])

    ds = FFPPDataset(args.data_root, args.splits_dir, args.split, args.compression,
                     args.methods, model_img_size(train_args["model"]))
    dl = DataLoader(ds, args.batch_size, num_workers=args.num_workers,
                    pin_memory=True)
    metrics = evaluate(model, dl, device)

    result = {
        "model": train_args["model"],
        "train_methods": train_args["methods"],
        "train_compression": train_args["compression"],
        "test_methods": args.methods,
        "test_compression": args.compression,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        **metrics,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
