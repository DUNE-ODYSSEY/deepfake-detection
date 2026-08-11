"""Grad-CAM comparison: where do xception vs. cnn_vit look, and why does
cross-manipulation generalization fail?

Both checkpoints here were trained on Deepfakes only (runs/*_Deepfakes_c23),
the source method for the single worst cross-manipulation cell in the study
(DF->FS: xception AUC 0.256, cnn_vit AUC 0.198 -- both *worse than random*).
Uses the same video (000_003) manipulated two different ways so the only
variable between the two example frames is the manipulation method itself:

  - Deepfakes/000_003  (same-manip as training -> confident, correct "fake")
  - FaceSwap/000_003   (cross-manip, unseen method -> confident, WRONG "real")

Grad-CAM w.r.t. the fake logit on each model's last spatial feature map
(xception: backbone.forward_features, 2048x10x10; cnn_vit: the ResNet-50 C4
map feeding the transformer, 1024x14x14) shows what evidence each model
actually used.

Usage: python -m analyze.gradcam
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from src.data.dataset import build_transforms
from src.models.factory import create_model, model_img_size

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FACES = Path("C:/ffpp_data/ffpp_faces/c23")
EXAMPLES = [
    ("Deepfakes", "000_003", "same-manip (trained on this)"),
    ("FaceSwap", "000_003", "cross-manip (never seen)"),
]


def load_checkpoint(model_name: str, run_name: str):
    model = create_model(model_name, pretrained=False).to(DEVICE).eval()
    state = torch.load(f"runs/{run_name}/best.pt", map_location=DEVICE)
    model.load_state_dict(state["model"])
    return model


def gradcam_xception(model, x):
    model.zero_grad(set_to_none=True)
    feat = model.backbone.forward_features(x)
    feat.retain_grad()
    pooled = model.backbone.forward_head(feat, pre_logits=True)
    logit = model.head(pooled).squeeze()
    logit.backward()
    weights = feat.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1))
    return cam.detach(), torch.sigmoid(logit).item()


def gradcam_cnn_vit(model, x):
    model.zero_grad(set_to_none=True)
    feat = model.cnn(x)[-1]
    feat.retain_grad()
    tokens = model.proj(feat).flatten(2).transpose(1, 2)
    cls = model.cls_token.expand(x.size(0), -1, -1)
    tokens = torch.cat([cls, tokens], dim=1) + model.pos_embed
    tokens = model.encoder(tokens)
    logit = model.head(model.norm(tokens[:, 0])).squeeze()
    logit.backward()
    weights = feat.grad.mean(dim=(2, 3), keepdim=True)
    cam = F.relu((weights * feat).sum(dim=1))
    return cam.detach(), torch.sigmoid(logit).item()


def overlay(cam, orig_img_rgb, size):
    cam = F.interpolate(cam.unsqueeze(0), size=(size, size), mode="bilinear",
                        align_corners=False)[0, 0].cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    heat = plt.get_cmap("jet")(cam)[..., :3]
    base = np.asarray(orig_img_rgb.resize((size, size))) / 255.0
    return 0.55 * base + 0.45 * heat


def main():
    xc = load_checkpoint("xception", "xception_Deepfakes_c23")
    cv = load_checkpoint("cnn_vit", "cnn_vit_Deepfakes_c23")
    tf_xc, tf_cv = build_transforms(299, train=False), build_transforms(224, train=False)

    fig, axes = plt.subplots(len(EXAMPLES), 3, figsize=(10.5, 3.6 * len(EXAMPLES)))
    for row, (method, vid, tag) in enumerate(EXAMPLES):
        frame = FACES / method / vid / "frame_00000.jpg"
        img = Image.open(frame).convert("RGB")

        x_xc = tf_xc(img).unsqueeze(0).to(DEVICE)
        cam_xc, p_xc = gradcam_xception(xc, x_xc)
        x_cv = tf_cv(img).unsqueeze(0).to(DEVICE)
        cam_cv, p_cv = gradcam_cnn_vit(cv, x_cv)

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"{method}/{vid}\n{tag}", fontsize=10)
        axes[row, 1].imshow(overlay(cam_xc, img, 299))
        axes[row, 1].set_title(f"xception (DF-trained)\nfake_prob={p_xc:.3f}", fontsize=10)
        axes[row, 2].imshow(overlay(cam_cv, img, 224))
        axes[row, 2].set_title(f"cnn_vit (DF-trained)\nfake_prob={p_cv:.3f}", fontsize=10)
        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle("Grad-CAM: same checkpoint, same video, different manipulation method\n"
                 "(DF-trained models on their worst cross-manip target, FaceSwap)", fontsize=11)
    fig.tight_layout()
    out = Path("analyze/outputs/gradcam_comparison.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
