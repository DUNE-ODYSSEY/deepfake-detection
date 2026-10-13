"""Frame-level and video-level accuracy, F1, and AUC."""
import warnings
from collections import defaultdict

import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score


def _safe_auc(labels, probs):
    try:
        with warnings.catch_warnings():
            # Expected and handled below (via the ValueError path / NaN return)
            # for small bootstrap resamples that land on a single class.
            warnings.simplefilter("ignore", category=UndefinedMetricWarning)
            return float(roc_auc_score(labels, probs))
    except ValueError:  # single class present
        return float("nan")


def bootstrap_video_auc_ci(video_probs, video_labels, n_resamples: int = 1000,
                           ci: float = 0.95, seed: int = 0):
    """Bootstrap CI for video-level AUC, resampling videos with replacement.

    video_probs/video_labels: 1-D arrays, one entry per video (already
    aggregated to video level, e.g. mean frame prob per video).
    Per RESEARCH_BRIEF.md section 5: "Bootstrap over videos (at least 1000
    resamples), 95% CI, for every reported AUC."

    Returns (point_estimate, ci_lo, ci_hi, n_resamples_used). Resamples that
    land on a single class (AUC undefined) are skipped and don't count toward
    n_resamples_used; if too few valid resamples remain, CI bounds are NaN.
    """
    probs, labels = np.asarray(video_probs), np.asarray(video_labels)
    point = _safe_auc(labels, probs)
    n = len(labels)
    rng = np.random.default_rng(seed)
    aucs = []
    for _ in range(n_resamples):
        idx = rng.integers(0, n, size=n)
        a = _safe_auc(labels[idx], probs[idx])
        if not np.isnan(a):
            aucs.append(a)
    if len(aucs) < max(10, n_resamples // 10):
        return point, float("nan"), float("nan"), len(aucs)
    lo_pct, hi_pct = (1 - ci) / 2 * 100, (1 + ci) / 2 * 100
    ci_lo, ci_hi = np.percentile(aucs, [lo_pct, hi_pct])
    return point, float(ci_lo), float(ci_hi), len(aucs)


def aggregate_video_scores(probs, labels, videos):
    """Collapse frame-level probs/labels to one score per video (mean frame
    probability, the standard FF++ protocol). Returns (video_ids, vid_probs,
    vid_labels) as parallel arrays/lists, in stable (first-seen) video order —
    callers that need to save per-video scores (ground rule: "save per-video
    scores so analyses rerun without retraining") can zip these directly.
    """
    vid_probs, vid_labels = defaultdict(list), {}
    for p, l, v in zip(probs, labels, videos):
        vid_probs[v].append(p)
        vid_labels[v] = l
    video_ids = list(vid_probs.keys())
    vp = np.array([np.mean(vid_probs[v]) for v in video_ids])
    vl = np.array([vid_labels[v] for v in video_ids])
    return video_ids, vp, vl


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

    _, vp, vl = aggregate_video_scores(probs, labels, videos)
    vpred = (vp >= 0.5).astype(int)
    out.update({
        "video_acc": float(accuracy_score(vl, vpred)),
        "video_f1": float(f1_score(vl, vpred, zero_division=0)),
        "video_auc": _safe_auc(vl, vp),
        "n_frames": int(len(labels)),
        "n_videos": int(len(vl)),
    })
    return out
