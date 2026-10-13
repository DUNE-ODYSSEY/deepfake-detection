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
import csv
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .data.dataset import FFPPDataset
from .metrics import aggregate_video_scores, bootstrap_video_auc_ci
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
    p.add_argument("--no-save-video-scores", action="store_true",
                   help="Skip writing the per-video score sidecar CSV "
                        "(on by default: 'save per-video scores so analyses "
                        "rerun without retraining', ground rule 1)")
    p.add_argument("--bootstrap", action="store_true", default=True)
    p.add_argument("--no-bootstrap", dest="bootstrap", action="store_false",
                   help="Skip the bootstrap video-AUC CI (on by default)")
    p.add_argument("--n-resamples", type=int, default=1000)
    p.add_argument("--bootstrap-seed", type=int, default=0)
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
    metrics, probs, labels, videos = evaluate(model, dl, device, return_raw=True)

    video_ids, vp, vl = aggregate_video_scores(probs, labels, videos)

    result = {
        "model": train_args["model"],
        "train_methods": train_args["methods"],
        "train_compression": train_args["compression"],
        "test_methods": args.methods,
        "test_compression": args.compression,
        "split": args.split,
        "checkpoint": str(args.checkpoint),
        "dataset_source": train_args.get("dataset_source", "mirror"),
        "seed": train_args.get("seed"),
        **metrics,
    }

    if args.bootstrap:
        point, ci_lo, ci_hi, n_used = bootstrap_video_auc_ci(
            vp, vl, n_resamples=args.n_resamples, seed=args.bootstrap_seed)
        result["video_auc_ci95_lo"] = ci_lo
        result["video_auc_ci95_hi"] = ci_hi
        result["video_auc_bootstrap_n"] = n_used
        assert abs(point - metrics["video_auc"]) < 1e-9 or (
            point != point and metrics["video_auc"] != metrics["video_auc"]
        ), "bootstrap point estimate disagrees with compute_metrics video_auc"

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))

    if not args.no_save_video_scores:
        scores_path = out_path.with_suffix("").with_suffix(".video_scores.csv")
        with open(scores_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["video_id", "label", "video_prob"])
            for vid, lbl, prob in zip(video_ids, vl.tolist(), vp.tolist()):
                writer.writerow([vid, lbl, prob])
        print(f"wrote per-video scores -> {scores_path}")


if __name__ == "__main__":
    main()
