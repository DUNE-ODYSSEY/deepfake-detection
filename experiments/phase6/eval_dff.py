"""Phase 6: score FF++-trained checkpoints on DeepFakeFace crops (no retraining)
and test whether the hf_ratio sign predicts the cross-dataset transfer.

For every best.pt under the given runs roots, scores every DFF crop (eval
transform, same as FF++ evaluation) and writes per-image scores. Then, per
(run group, train method, DFF fake class): seed-mean AUC (real = wiki) with a
95% CI from 1000 bootstrap resamples over images. hf_ratio is computed on the
same crops; predicted sign = sign(d_train on FF++ reference crops) x
sign(d_test on DFF crops).

Runs on CPU by default so it never competes with GPU training for memory.
Usage: python -m experiments.phase6.eval_dff --runs-roots runs runs/phase4_hfaug
"""
import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from experiments.phase3.crop_statistics import hf_ratio
from experiments.phase3.sign_prediction import cohens_d
from src.data.dataset import build_transforms
from src.models.factory import create_model, model_img_size

FAKES = ["inpainting", "insight", "text2img"]
SHORT = {"Deepfakes": "DF", "FaceSwap": "FS", "Face2Face": "F2F", "NeuralTextures": "NT"}
RUN_RE = re.compile(r"reference_(?P<model>[a-z0-9_]+?)_(?P<train>Deepfakes|FaceSwap|Face2Face|NeuralTextures)_c23[\\/]seed(?P<seed>\d+)")


def score_checkpoint(ckpt, crops, batch, device):
    state = torch.load(ckpt, map_location=device)
    name = state["args"]["model"]
    model = create_model(name, pretrained=False).to(device).eval()
    model.load_state_dict(state["model"])
    tf = build_transforms(model_img_size(name), train=False)
    probs = []
    with torch.no_grad():
        for i in range(0, len(crops), batch):
            x = torch.stack([tf(Image.open(c).convert("RGB")) for c in crops[i:i + batch]]).to(device)
            probs.extend(torch.sigmoid(model(x)).cpu().numpy().tolist())
    return np.array(probs)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dff-root", default="C:/ffpp_data/dff_ref")
    p.add_argument("--runs-roots", nargs="+", default=["runs"])
    p.add_argument("--out-dir", default="analysis/phase6")
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--batch", type=int, default=32)
    p.add_argument("--n-boot", type=int, default=1000)
    args = p.parse_args()
    torch.set_num_threads(args.threads)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    man = pd.read_csv(Path(args.dff_root) / "manifest.csv")
    man = man[man["face_found"]].reset_index(drop=True)
    crops = man["crop"].tolist()
    labels = (man["class"] != "wiki").astype(int).to_numpy()

    hf_path = out / "dff_hf_ratio.csv"
    if hf_path.exists():
        man["hf_ratio"] = pd.read_csv(hf_path)["hf_ratio"].to_numpy()
    else:
        man["hf_ratio"] = [hf_ratio(np.asarray(Image.open(c).convert("L"))) for c in crops]
        man[["class", "member", "hf_ratio"]].to_csv(hf_path, index=False)
    real_hf = man.loc[man["class"] == "wiki", "hf_ratio"].to_numpy()
    d_dff = {f: cohens_d(man.loc[man["class"] == f, "hf_ratio"].to_numpy(), real_hf) for f in FAKES}
    pd.DataFrame([{"dff_class": f, "d_hf_ratio_vs_wiki": round(d, 4),
                   "n_fake": int((man["class"] == f).sum()), "n_real": len(real_hf)}
                  for f, d in d_dff.items()]).to_csv(out / "dff_effect_sizes.csv", index=False)
    print("DFF hf_ratio d vs real:", {k: round(v, 3) for k, v in d_dff.items()}, flush=True)

    scores_path = out / "dff_scores.csv"
    scores = pd.read_csv(scores_path) if scores_path.exists() else pd.DataFrame({"crop": crops})
    for root in args.runs_roots:
        for ckpt in sorted(Path(root).glob("reference_*/seed*/best.pt")):
            if not (ckpt.parent / "DONE").exists():
                continue
            m = RUN_RE.search(str(ckpt))
            col = f"{Path(root).name}|{m['model']}|{SHORT[m['train']]}|{m['seed']}"
            if col in scores.columns:
                continue
            print(f"scoring {col}", flush=True)
            scores[col] = score_checkpoint(ckpt, crops, args.batch, args.device)
            scores.to_csv(scores_path, index=False)

    d_ffpp = pd.read_csv("analysis/phase3/effect_sizes.csv")
    d_ffpp = d_ffpp[(d_ffpp["stat"] == "hf_ratio") & (d_ffpp["pipeline"] == "reference")].set_index("method")["d"]
    rng = np.random.default_rng(0)
    groups = {}
    for col in scores.columns[1:]:
        root, model, train, seed = col.split("|")
        groups.setdefault((root, model, train), []).append(col)
    rows = []
    for (root, model, train), cols in sorted(groups.items()):
        for f in FAKES:
            mask = (man["class"] == "wiki") | (man["class"] == f)
            y = labels[mask.to_numpy()]
            P = np.stack([scores.loc[mask, c].to_numpy() for c in cols])
            auc = float(np.mean([roc_auc_score(y, pr) for pr in P]))
            boots = []
            for _ in range(args.n_boot):
                idx = rng.integers(0, len(y), len(y))
                if len(np.unique(y[idx])) == 2:
                    boots.append(np.mean([roc_auc_score(y[idx], pr[idx]) for pr in P]))
            lo, hi = np.percentile(boots, [2.5, 97.5])
            pred = int(np.sign(d_ffpp[train] * d_dff[f]))
            call = -1 if hi < 0.5 else (1 if lo > 0.5 else 0)
            rows.append({"runs": root, "model": model, "train": train, "dff_fake": f, "n_seeds": len(cols),
                         "auc": round(auc, 4), "ci95_lo": round(lo, 4), "ci95_hi": round(hi, 4),
                         "observed_call": {-1: "below chance", 0: "at chance", 1: "above chance"}[call],
                         "d_train_ffpp": round(float(d_ffpp[train]), 3), "d_test_dff": round(d_dff[f], 3),
                         "predicted_sign": "inverted" if pred < 0 else "transfers",
                         "prediction_correct": (call == pred) if call != 0 else None})
    res = pd.DataFrame(rows)
    res.to_csv(out / "dff_transfer.csv", index=False)
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
