"""Frame-level and video-level accuracy, F1, and AUC."""
from collections import defaultdict

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _safe_auc(labels, probs):
    try:
        return float(roc_auc_score(labels, probs))
    except ValueError:  # single class present
        return float("nan")


def compute_metrics(probs, labels, videos):
    """probs/labels: 1-D arrays over frames; videos: parallel list of video keys.

    Video-level score = mean frame probability per video (standard FF++ protocol).
    """
    probs, labels = np.asarray(probs), np.asarray(labels)
    preds = (probs >= 0.5).astype(int)
    out = {
        "frame_acc": float(accuracy_score(labels, preds)),
        "frame_f1": float(f1_score(labels, preds, zero_division=0)),
        "frame_auc": _safe_auc(labels, probs),
    }

    vid_probs, vid_labels = defaultdict(list), {}
    for p, l, v in zip(probs, labels, videos):
        vid_probs[v].append(p)
        vid_labels[v] = l
    vp = np.array([np.mean(vid_probs[v]) for v in vid_probs])
    vl = np.array([vid_labels[v] for v in vid_probs])
    vpred = (vp >= 0.5).astype(int)
    out.update({
        "video_acc": float(accuracy_score(vl, vpred)),
        "video_f1": float(f1_score(vl, vpred, zero_division=0)),
        "video_auc": _safe_auc(vl, vp),
        "n_frames": int(len(labels)),
        "n_videos": int(len(vl)),
    })
    return out
