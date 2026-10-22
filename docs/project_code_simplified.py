"""
Deepfake Detection on FaceForensics++ -- simplified walkthrough of the whole project.

This one file condenses the real code into the steps we explain in the review.
It keeps the same logic and settings but drops logging, command-line options,
checkpoint resuming and file bookkeeping. Each section names the real file it
comes from.

    1. Data splits            -> src/data/splits.py
    2. Face cropping          -> src/data/preprocess.py, src/data/preprocess_reference.py
    3. Dataset + augmentation -> src/data/dataset.py
    4. Models                 -> src/models/xception.py, cnn_vit.py, timm_backbones.py
    5. Training               -> src/train.py
    6. Evaluation             -> src/evaluate.py, src/metrics.py
    7. Cross-manipulation     -> experiments/phase1/run_phase1_matrix.py
    8. Mechanism (sharpness)  -> experiments/phase3/crop_statistics.py, sign_prediction.py
"""
import json
from pathlib import Path

import numpy as np
import timm
import torch
import torch.nn as nn
from PIL import Image, ImageFilter
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

METHODS = ["Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"]


# ---------------------------------------------------------------- 1. SPLITS
def load_split(split):
    """Official FF++ split: a list of [a, b] pairs. Real videos are 'a' and 'b';
    the fakes made from them are 'a_b' and 'b_a'. Splitting by video (not frame)
    keeps the same person out of both train and test."""
    pairs = json.loads(Path(f"splits/{split}.json").read_text())
    real = {v for pair in pairs for v in pair}
    fake = {f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs}
    return real, fake


# ---------------------------------------------------------- 2. FACE CROPPING
def crop_faces(video_path, out_dir, n_frames=20, margin=1.3, size=299):
    """Take n evenly spaced frames, find the face with MTCNN, crop it with a
    1.3x margin (so the blending boundary is included) and save it."""
    import cv2
    from facenet_pytorch import MTCNN
    detector = MTCNN(select_largest=True, post_process=False)
    video = cv2.VideoCapture(str(video_path))
    total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    for i in np.linspace(0, total - 1, n_frames).astype(int):
        video.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frame = video.read()
        if not ok:
            continue
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        boxes, _ = detector.detect(rgb)
        if boxes is None:
            continue
        x1, y1, x2, y2 = boxes[0]
        cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1) * margin / 2
        face = Image.fromarray(rgb).crop((cx - half, cy - half, cx + half, cy + half))
        face.resize((size, size)).save(Path(out_dir) / f"frame_{i:05d}.jpg", quality=95)
    # The "reference" pipeline (preprocess_reference.py) does the same with
    # dlib + 68-landmark eye alignment, 256 px PNG and 32 frames per video.


# ---------------------------------------------- 3. DATASET + AUGMENTATION
class RandomBlurSharpen:
    """Our intervention (Phase 4): randomly blur or sharpen every training
    image, real or fake alike, so sharpness stops being a usable 'fake' cue."""
    def __call__(self, img):
        if np.random.rand() < 0.9:
            if np.random.rand() < 0.5:
                return img.filter(ImageFilter.GaussianBlur(radius=np.random.uniform(0.3, 1.5)))
            return img.filter(ImageFilter.UnsharpMask(radius=np.random.uniform(1, 2),
                                                      percent=int(np.random.uniform(50, 200))))
        return img


class FaceCrops(Dataset):
    """Every face crop with a label: real = 0, fake = 1. The video id is kept so
    frame scores can be averaged into one score per video."""
    def __init__(self, root, split, methods, size, train=False, hf_aug=False):
        real_ids, fake_ids = load_split(split)
        self.items = [(f, 0, f"real/{v.name}") for v in Path(root, "real").iterdir()
                      if v.name in real_ids for f in sorted(v.glob("*.*"))]
        for m in methods:
            self.items += [(f, 1, f"{m}/{v.name}") for v in Path(root, m).iterdir()
                           if v.name in fake_ids for f in sorted(v.glob("*.*"))]
        steps = [transforms.Resize((size, size))]
        if train:
            steps += ([RandomBlurSharpen()] if hf_aug else []) + [
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(0.1, 0.1, 0.1)]
        steps += [transforms.ToTensor(),
                  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        path, label, video = self.items[i]
        return self.transform(Image.open(path).convert("RGB")), float(label), video


# ----------------------------------------------------------------- 4. MODELS
class Xception(nn.Module):
    """Baseline: ImageNet-pretrained Xception -> 2048 features -> 1 logit."""
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model("legacy_xception", pretrained=True, num_classes=0)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(2048, 1))

    def forward(self, x):                      # x: (batch, 3, 299, 299)
        return self.head(self.backbone(x)).squeeze(-1)


class CNNViT(nn.Module):
    """Hybrid: ResNet-50 (stages 1-3) finds local texture, then a 6-layer
    Transformer looks at the whole face at once through self-attention."""
    def __init__(self, dim=512):
        super().__init__()
        self.cnn = timm.create_model("resnet50", pretrained=True,
                                     features_only=True, out_indices=(3,))
        self.to_tokens = nn.Conv2d(1024, dim, kernel_size=1)          # 14x14 map -> 196 tokens
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))               # summary token
        self.pos = nn.Parameter(torch.zeros(1, 197, dim))             # position of each token
        layer = nn.TransformerEncoderLayer(dim, nhead=8, dim_feedforward=4 * dim,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(layer, num_layers=6)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 1))

    def forward(self, x):                      # x: (batch, 3, 224, 224)
        tokens = self.to_tokens(self.cnn(x)[-1]).flatten(2).transpose(1, 2)
        tokens = torch.cat([self.cls.expand(len(x), -1, -1), tokens], dim=1) + self.pos
        return self.head(self.transformer(tokens)[:, 0]).squeeze(-1)

# ResNet-50 and EfficientNet-B4 (Phase 5) are the same idea as Xception:
# timm.create_model("resnet50.a1_in1k" / "efficientnet_b4.ra2_in1k") + a linear head.


# --------------------------------------------------------------- 5. TRAINING
def train(model, train_set, val_set, epochs=15, patience=4, device="cuda"):
    """AdamW + cosine learning-rate schedule + mixed precision. Classes are
    balanced by sampling. Keeps the checkpoint with the best validation AUC and
    stops early once it hasn't improved for `patience` epochs."""
    weights = torch.tensor([1 / sum(1 for _, l, _ in train_set.items if l == lab)
                            for _, lab, _ in train_set.items])
    sampler = torch.utils.data.WeightedRandomSampler(weights, len(train_set))
    loader = DataLoader(train_set, batch_size=32, sampler=sampler, num_workers=4)
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler()
    loss_fn = nn.BCEWithLogitsLoss()
    best, bad = -1.0, 0
    for epoch in range(epochs):
        model.train()
        for x, y, _ in loader:
            opt.zero_grad()
            with torch.amp.autocast(device):
                loss = loss_fn(model(x.to(device)), y.float().to(device))
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        sched.step()
        val_auc = evaluate(model, val_set, device)["video_auc"]
        if val_auc > best:
            best, bad = val_auc, 0
            torch.save(model.state_dict(), "best.pt")
        else:
            bad += 1
            if bad >= patience:
                break
    return best


# ------------------------------------------------------------- 6. EVALUATION
@torch.no_grad()
def evaluate(model, dataset, device="cuda", n_boot=1000):
    """Score every frame, average frames into one score per video, then compute
    video-level AUC with a 95% bootstrap confidence interval (resampling videos)."""
    model.eval()
    scores = {}
    for x, y, videos in DataLoader(dataset, batch_size=32):
        for p, label, v in zip(torch.sigmoid(model(x.to(device))).cpu().numpy(), y, videos):
            scores.setdefault(v, [float(label), []])[1].append(p)
    labels = np.array([s[0] for s in scores.values()])
    probs = np.array([np.mean(s[1]) for s in scores.values()])
    auc = roc_auc_score(labels, probs)
    rng = np.random.default_rng(0)
    boots = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if len(set(labels[idx])) == 2:
            boots.append(roc_auc_score(labels[idx], probs[idx]))
    return {"video_auc": auc, "ci95": np.percentile(boots, [2.5, 97.5])}


# --------------------------------------------------- 7. CROSS-MANIPULATION
def transfer_matrix(root, make_model, size):
    """Train on ONE manipulation, test on ALL four. An AUC below 0.5 means the
    detector ranks those fakes as MORE real than real faces (an inversion)."""
    matrix = {}
    for train_method in METHODS:
        model = make_model()
        train(model, FaceCrops(root, "train", [train_method], size, train=True),
              FaceCrops(root, "val", [train_method], size))
        model.load_state_dict(torch.load("best.pt"))
        for test_method in METHODS:
            matrix[train_method, test_method] = evaluate(
                model, FaceCrops(root, "test", [test_method], size))["video_auc"]
    return matrix          # e.g. Deepfakes -> FaceSwap is about 0.26


# ------------------------------------------------- 8. MECHANISM: SHARPNESS
def high_frequency_energy(gray):
    """Share of the image's spectral energy above 0.25 cycles/pixel: a number
    for 'how sharp' a face crop is."""
    g = (gray - gray.mean()) * np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    fy, fx = np.meshgrid(np.fft.fftshift(np.fft.fftfreq(gray.shape[0])),
                         np.fft.fftshift(np.fft.fftfreq(gray.shape[1])), indexing="ij")
    r = np.sqrt(fx ** 2 + fy ** 2)
    return power[r > 0.25].sum() / power[r > 0.02].sum()


def cohens_d(fake, real):
    """How far apart two groups are, in standard deviations (+ = fakes higher)."""
    pooled = np.sqrt(((len(fake) - 1) * fake.var(ddof=1) + (len(real) - 1) * real.var(ddof=1))
                     / (len(fake) + len(real) - 2))
    return (fake.mean() - real.mean()) / pooled


def predict_inversion(d):
    """The rule: a detector trained on A and tested on B transfers if A and B
    move sharpness the same way, and inverts if they move it opposite ways.
    Measured: Deepfakes d = -0.48 (blurrier), FaceSwap d = +0.37 (sharper),
    so Deepfakes -> FaceSwap is predicted to invert, and it does."""
    return {(a, b): "inverts" if d[a] * d[b] < 0 else "transfers"
            for a in d for b in d if a != b}


# ---------------------------------------------------------------- PUT TOGETHER
if __name__ == "__main__":
    crops = "C:/ffpp_data/ffpp_faces/c23"
    xception_matrix = transfer_matrix(crops, Xception, size=299)
    hybrid_matrix = transfer_matrix(crops, CNNViT, size=224)
    print(xception_matrix, hybrid_matrix)
