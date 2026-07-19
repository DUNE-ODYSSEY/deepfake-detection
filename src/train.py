"""Train a detector on FF++ face crops.

Example (baseline, all four manipulations, c23):
  python -m src.train --data-root /data/ffpp_faces --splits-dir splits \
      --model xception --methods Deepfakes Face2Face FaceSwap NeuralTextures \
      --compression c23 --out runs/xception_all_c23

Example (cross-manipulation source model):
  python -m src.train --data-root /data/ffpp_faces --splits-dir splits \
      --model cnn_vit --methods Deepfakes --compression c23 --out runs/cnn_vit_DF_c23
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from tqdm import tqdm

from .data.dataset import FFPPDataset
from .metrics import compute_metrics
from .models.factory import create_model, model_img_size


def evaluate(model, loader, device):
    model.eval()
    probs, labels, videos = [], [], []
    with torch.no_grad():
        for x, y, v in loader:
            logits = model(x.to(device, non_blocking=True))
            probs.extend(torch.sigmoid(logits).cpu().numpy().tolist())
            labels.extend(y.numpy().tolist())
            videos.extend(v)
    return compute_metrics(probs, labels, videos)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root", required=True)
    p.add_argument("--splits-dir", default="splits")
    p.add_argument("--model", required=True, choices=["xception", "cnn_vit"])
    p.add_argument("--methods", nargs="+", required=True)
    p.add_argument("--compression", default="c23", choices=["c0", "c23", "c40"])
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--patience", type=int, default=4, help="early-stop on val AUC")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", required=True)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))

    img_size = model_img_size(args.model)
    train_ds = FFPPDataset(args.data_root, args.splits_dir, "train",
                           args.compression, args.methods, img_size, train_aug=True)
    val_ds = FFPPDataset(args.data_root, args.splits_dir, "val",
                         args.compression, args.methods, img_size)
    sampler = WeightedRandomSampler(train_ds.class_weights(), len(train_ds))
    train_dl = DataLoader(train_ds, args.batch_size, sampler=sampler,
                          num_workers=args.num_workers, pin_memory=True,
                          drop_last=True)
    val_dl = DataLoader(val_ds, args.batch_size, num_workers=args.num_workers,
                        pin_memory=True)
    print(f"train: {len(train_ds)} frames | val: {len(val_ds)} frames | {device}")

    model = create_model(args.model).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    scaler = torch.amp.GradScaler(enabled=(device == "cuda"))
    criterion = torch.nn.BCEWithLogitsLoss()

    start_epoch, best_auc, bad_epochs = 0, -1.0, 0
    ckpt_last, ckpt_best = out / "last.pt", out / "best.pt"
    if args.resume and ckpt_last.exists():
        state = torch.load(ckpt_last, map_location=device)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        sched.load_state_dict(state["sched"])
        start_epoch, best_auc = state["epoch"] + 1, state["best_auc"]
        print(f"resumed at epoch {start_epoch} (best val AUC {best_auc:.4f})")

    history = []
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0, losses = time.time(), []
        for x, y, _ in tqdm(train_dl, desc=f"epoch {epoch}"):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device, enabled=(device == "cuda")):
                loss = criterion(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        sched.step()

        val = evaluate(model, val_dl, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "time_s": round(time.time() - t0, 1), **val}
        history.append(row)
        (out / "history.json").write_text(json.dumps(history, indent=2))
        print(f"epoch {epoch}: loss={row['train_loss']:.4f} "
              f"val_auc={val['video_auc']:.4f} val_acc={val['video_acc']:.4f}")

        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "epoch": epoch,
                    "best_auc": best_auc, "args": vars(args)}, ckpt_last)
        if val["video_auc"] > best_auc:
            best_auc, bad_epochs = val["video_auc"], 0
            torch.save({"model": model.state_dict(), "args": vars(args),
                        "epoch": epoch, "val": val}, ckpt_best)
        else:
            bad_epochs += 1
            if bad_epochs >= args.patience:
                print(f"early stop at epoch {epoch} (best val AUC {best_auc:.4f})")
                break

    print(f"done. best val video-AUC: {best_auc:.4f} -> {ckpt_best}")


if __name__ == "__main__":
    main()
