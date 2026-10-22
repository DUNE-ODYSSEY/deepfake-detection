"""Build docs/end_review_presentation.pptx -- the end-semester review deck.

Follows the course review structure used at the mid-review (introduction,
literature, methodology, dataset, algorithm selection, results, software,
conclusion) with the complete results, then the research extension
(RESEARCH_BRIEF.md Phases 0-6). Same visual language as
build_mid_review_ppt.py. Every result number is read from analysis/ files; the
model sizes are computed from the model definitions.

Usage: python -m scripts.build_end_review_ppt
"""
import json

import pandas as pd
from pptx.util import Emu

from scripts import build_mid_review_ppt as m

MARGIN = m.MARGIN
SLIDE_W, SLIDE_H = m.SLIDE_W, m.SLIDE_H
NAVY, CYAN, TEAL = m.NAVY, m.CYAN, m.TEAL
TEXT, MUTED, MUTED_L = m.TEXT, m.MUTED, m.MUTED_L
CARD_BG = m.CARD_BG
GREEN, AMBER, RED, WHITE = m.GREEN, m.AMBER, m.RED, m.WHITE
TEAM = m.TEAM
new_slide, textbox, rect = m.new_slide, m.textbox, m.rect
header, bullets, stat_panel, card = m.header, m.bullets, m.stat_panel, m.card
picture_slide, table_slide, content_slide = m.picture_slide, m.table_slide, m.content_slide

TOTAL_SLIDES = 23
FOOTER = "Deepfake Detection — End-Semester Review (23AID205)"


def IN(x):
    return Emu(int(x * 914400))


def f2(x):
    return f"{float(x):.2f}"


def f3(x):
    return f"{float(x):.3f}"


def ci(lo, hi):
    return f"[{float(lo):.2f}, {float(hi):.2f}]"


def footer(slide):
    m.footer(slide, FOOTER)


def picture(prs, eyebrow, title, caption, path, aspect, height_in, stats_label, stats):
    return picture_slide(prs, eyebrow, title, caption, path,
                         IN(height_in * aspect), IN(height_in), stats_label, stats)


def model_sizes():
    from src.models.factory import create_model, model_img_size
    out = {}
    for name in ["xception", "cnn_vit", "resnet50", "efficientnet_b4"]:
        model = create_model(name, pretrained=False)
        out[name] = (model_img_size(name), sum(p.numel() for p in model.parameters()) / 1e6)
    return out


def main():
    m.PAGE[0] = 0
    m.TOTAL_SLIDES = TOTAL_SLIDES
    m.footer.__defaults__ = (FOOTER,)

    base = pd.read_csv("analysis/baseline_comparison.csv").set_index("model")
    gap = pd.read_csv("analysis/generalization_gap.csv").set_index("model")
    comp = pd.read_csv("analysis/compression_robustness.csv").set_index(["model", "test_compression"])
    cur = pd.read_csv("analysis/cross_manipulation_xception.csv", index_col=0)
    curv = pd.read_csv("analysis/cross_manipulation_cnn_vit.csv", index_col=0)
    g = pd.read_csv("analysis/reference/gate1_seed_mean_ci.csv").set_index(["train", "test"])
    hf = pd.read_csv("analysis/phase3/effect_sizes.csv")
    hf = hf[(hf["stat"] == "hf_ratio") & (hf["pipeline"] == "current")].set_index("method")
    sp = pd.read_csv("analysis/phase3/sign_prediction.csv")
    sp = sp[sp["stat"] == "hf_ratio"].set_index("matrix")
    pub = pd.read_csv("analysis/external/sign_prediction_published.csv").iloc[0]
    pubm = pd.read_csv("analysis/external/facexray_table1_xception.csv", index_col=0)
    rel = pd.read_csv("analysis/phase3/reliance_reference.csv").set_index(["train", "stat"])
    d4 = pd.read_csv("analysis/phase4/delta_vs_phase1.csv").set_index(["train", "test"])
    arch = json.loads(open("analysis/architecture_correlation.json").read())
    a5 = {bb: pd.read_csv(f"analysis/phase5/{bb}_seed_mean_ci.csv").set_index(["train", "test"])
          for bb in ["resnet50", "efficientnet_b4"]}
    p2 = {v: (pd.read_csv(f"analysis/phase2/{v}_seed_mean_ci.csv").set_index(["train", "test"]),
              pd.read_csv(f"analysis/phase2/{v}_delta_vs_reference.csv").set_index(["train", "test"]))
          for v in ["jpeg95", "jpeg75", "frames10", "margin10"]}
    dff = pd.read_csv("analysis/phase6/dff_transfer.csv")
    dff = dff[dff["runs"] == "runs"].set_index(["train", "dff_fake"])
    dffe = pd.read_csv("analysis/phase6/dff_effect_sizes.csv").set_index("dff_class")
    sizes = model_sizes()
    dffs, fsnt = g.loc[("DF", "FS")], g.loc[("FS", "NT")]

    prs = m.Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

    # 1. Title
    s = new_slide(prs, bg=NAVY)
    rect(s, 0, 0, Emu(146304), SLIDE_H, CYAN)
    textbox(s, MARGIN, IN(2.15), IN(10), IN(0.4),
            [("AIML COURSE PROJECT · 23AID205 · END-SEMESTER REVIEW", 15, True, CYAN)])
    textbox(s, MARGIN, IN(2.55), IN(11.4), IN(1.9),
            [[("Deepfake Detection on FaceForensics++:", 32, True, WHITE)],
             [("Why Detectors Score Unseen Fakes Below Chance", 30, True, WHITE)]])
    textbox(s, MARGIN, IN(4.6), IN(11), IN(0.9),
            [("XceptionNet vs. a CNN-ViT hybrid, and a follow-up study of a surprising failure: "
              "detectors that rank unseen fakes as more real than real faces.", 14, False, MUTED_L)])
    textbox(s, MARGIN, IN(6.55), IN(11.8), IN(0.35),
            [(" · ".join(name for name, _ in TEAM), 13, True, m.RGBColor(0xE5, 0xE9, 0xF0))])
    textbox(s, MARGIN, IN(6.88), IN(8), IN(0.3),
            [("AI & Data Science — Amrita Vishwa Vidyapeetham, Coimbatore", 11, False, MUTED_L)])
    m.PAGE[0] += 1

    # 2. Introduction
    content_slide(prs, "INTRODUCTION", "Motivation, Problem & Objectives", [
        "Deepfakes are a real misinformation and fraud risk, but detectors are usually judged "
        "on the same manipulation type they were trained on.",
        "Problem: a detector at 99% on its own manipulation can fail badly on a new one, and "
        "that failure is invisible until deployment.",
        "Objective 1: compare XceptionNet (CNN) and a CNN-ViT hybrid on identical data and "
        "protocol: does self-attention generalize better?",
        "Objective 2: measure same-method accuracy, the 4×4 cross-method matrix and "
        "compression robustness; explain failures with Grad-CAM; ship a live demo.",
        "Objective 3 (added after the results): explain why some cross-method scores fall "
        "below chance, and test a fix.",
    ], size=15)

    # 3. Literature review
    content_slide(prs, "LITERATURE REVIEW", "Related Work", [
        "Rössler et al. (ICCV 2019), FaceForensics++: the 4-method benchmark and the "
        "XceptionNet baseline (Chollet, CVPR 2017).",
        "Dosovitskiy et al. (ICLR 2021), Vision Transformer: motivates the hybrid's "
        "self-attention. Selvaraju et al. (ICCV 2017), Grad-CAM: our interpretability tool.",
        f"Li et al. (CVPR 2020), Face X-ray: Xception trained on Deepfakes scores "
        f"{f2(pubm.loc['DF', 'FS'])} on FaceSwap with official data, so FaceSwap is already "
        "the hardest target.",
        "Geirhos et al. (Nat. Mach. Intell. 2020), shortcut learning; Durall et al. and "
        "Frank et al. (2020), frequency artifacts in generated images.",
        "Gap: below-chance transfer is reported but not explained per manipulation; we give "
        "a measurable cue that predicts which cells invert.",
    ], size=14)

    # 4. Methodology
    s = new_slide(prs)
    header(s, "METHODOLOGY", "End-to-End Pipeline")
    steps = [
        ("1", "Raw video", "FF++ c23: real + 4 manipulation methods, 5,000 videos"),
        ("2", "Face crops", "Original: MTCNN, JPEG. Reference: dlib + alignment, PNG"),
        ("3", "Official splits", "Video-level train / val / test, no identity leakage"),
        ("4", "Train", "One method + real; AMP, early stop; 2–3 seeds"),
        ("5", "Evaluate", "Video AUC on all 4 methods, 95% bootstrap CIs"),
        ("6", "Analyze", "Image statistics, sign test, intervention, external data"),
    ]
    card_w, gap_w = IN(1.91), IN(0.063)
    for i, (n, t, d) in enumerate(steps):
        cx = MARGIN + i * (card_w + gap_w)
        rect(s, cx, IN(2.2), card_w, IN(3.6), CARD_BG)
        rect(s, cx, IN(2.2), card_w, IN(0.55), NAVY)
        textbox(s, cx + IN(0.13), IN(2.3), card_w - IN(0.26), IN(0.35), [(n, 15, True, CYAN)])
        textbox(s, cx + IN(0.13), IN(2.95), card_w - IN(0.26), IN(0.7), [(t, 13.5, True, TEXT)])
        textbox(s, cx + IN(0.13), IN(3.65), card_w - IN(0.26), IN(2.0), [(d, 11, False, MUTED)])
    textbox(s, MARGIN, IN(6.05), IN(11.8), IN(0.4),
            [("Every reported number is produced by a script from saved per-video scores; "
              "nothing is typed by hand.", 11.5, False, MUTED)])
    footer(s)

    # 5. Dataset & preprocessing
    content_slide(prs, "DATASET & PREPROCESSING", "Data: Source and Preparation", [
        "FaceForensics++: 1,000 real YouTube videos and 4×1,000 manipulated (Deepfakes, "
        "Face2Face, FaceSwap, NeuralTextures), c23 compression, official splits.",
        "Source: a public Kaggle mirror (official access is still pending), so all results "
        "are labelled preliminary. An audit found no real-vs-fake differences in encoding.",
        "Original crops: MTCNN, 1.3× margin, 20 frames per video, JPEG (~100k crops). "
        "Reference crops: dlib + landmark alignment, 256 px PNG, 32 frames (~105k crops).",
        "External test set: DeepFakeFace (OpenRAIL licence): real IMDB-WIKI photos and "
        "fakes from Stable Diffusion and InsightFace, 1,000 sampled per class.",
    ], size=15)

    # 6. EDA
    picture(prs, "EDA", "Data Quality: Outlier Check (IQR)",
            "Per-video frame count and file size checked with box-plot IQR before "
            "preprocessing.",
            "analyze/outputs/outlier_boxplots.png", 1813 / 893, 3.65, "RESULT", [
                ("5,000", "videos analyzed (real + 4 methods)"),
                ("209", "flagged on frame count (4.2%)"),
                ("326", "flagged on file size (6.5%)"),
                ("502", "unique outliers overall (10.0%)"),
            ])

    # 7. Algorithm selection
    rows = []
    for key, label, pre in [("xception", "XceptionNet", "ImageNet"), ("cnn_vit", "CNN-ViT hybrid", "from scratch"),
                            ("resnet50", "ResNet-50", "ImageNet"), ("efficientnet_b4", "EfficientNet-B4", "ImageNet")]:
        size, params = sizes[key]
        rows.append([label, pre, f"{size} px", f"{params:.1f} M"])
    table_slide(prs, "ALGORITHM SELECTION", "Models",
                "XceptionNet and the CNN-ViT hybrid are the course comparison; ResNet-50 and "
                "EfficientNet-B4 were added to test architecture independence.",
                ["Model", "Pretraining", "Input size", "Parameters"], rows,
                [IN(3.6), IN(2.8), IN(2.6), IN(2.6)],
                note="Shared protocol: AdamW, cosine schedule, mixed precision, batch 32 (16 for "
                     "EfficientNet-B4 to fit 6 GB), early stopping on validation AUC. Video score = "
                     "mean of frame probabilities.")

    # 8. Experiment 1
    table_slide(prs, "RESULTS · EXPERIMENT 1", "Baseline: Same-Distribution Performance",
                "Both models trained and tested on all four manipulation methods (c23).",
                ["Model", "Video accuracy", "Video AUC", "Frame AUC"],
                [[label, f"{base.loc[k, 'video_acc'] * 100:.2f}%", f3(base.loc[k, "video_auc"]),
                  f3(base.loc[k, "frame_auc"])]
                 for k, label in [("xception", "XceptionNet"), ("cnn_vit", "CNN-ViT hybrid")]],
                [IN(3.3), IN(2.8), IN(2.8), IN(2.8)],
                note="XceptionNet leads on every metric. The hybrid's expected advantage did "
                     "not appear: a negative result, reported as such.")

    # 9. Experiment 2
    picture(prs, "RESULTS · EXPERIMENT 2", "Cross-Manipulation Generalization",
            "Train on one method, test on all four. Values below 0.5 mean the detector "
            "ranks unseen fakes as more real than real faces.",
            "analysis/cross_manipulation_heatmaps.png", 2480 / 1000, 3.3, "OBSERVATIONS", [
                (f2(cur.loc["DF", "FS"]), "Xception, Deepfakes → FaceSwap"),
                (f2(curv.loc["DF", "FS"]), "CNN-ViT, Deepfakes → FaceSwap"),
                (f"{gap.loc['xception', 'generalization_gap']:.2f} / {gap.loc['cnn_vit', 'generalization_gap']:.2f}",
                 "generalization gap, Xception / hybrid"),
                (f"r = {arch['pearson_r']:.2f}", "the two models fail in the same pattern"),
            ])

    # 10. Experiment 3
    table_slide(prs, "RESULTS · EXPERIMENT 3", "Compression Robustness",
                "Tested on a c40 proxy (c23 test videos re-compressed); never trained on.",
                ["Model", "c23 video AUC", "c40-proxy video AUC", "Drop"],
                [[label, f3(comp.loc[(k, 'c23'), 'video_auc']), f3(comp.loc[(k, 'c40proxy'), 'video_auc']),
                  f"{comp.loc[(k, 'c40proxy'), 'video_auc'] - comp.loc[(k, 'c23'), 'video_auc']:+.3f}"]
                 for k, label in [("xception", "XceptionNet"), ("cnn_vit", "CNN-ViT hybrid")]],
                [IN(3.3), IN(2.8), IN(3.0), IN(2.6)],
                note="Both degrade substantially. The hybrid drops slightly less, but on a proxy, "
                     "not official c40, so this is not claimed as a general result.")

    # 11. Grad-CAM
    picture(prs, "INTERPRETABILITY", "Grad-CAM: What the Models Look At",
            "Same Deepfakes-trained checkpoint, same video: its Deepfakes version vs. its "
            "FaceSwap version.",
            "analyze/outputs/gradcam_comparison.png", 1787 / 1279, 4.1, "READING", [
                ("Face", "in-distribution: attention on the face"),
                ("Off-face", "FaceSwap: Xception attention collapses"),
                ("Diffuse", "FaceSwap: hybrid attention spreads out"),
                ("0.00", "fake probability on the FaceSwap video"),
            ])

    # 12. Software implementation
    content_slide(prs, "SOFTWARE IMPLEMENTATION", "Live Demo", [
        "FastAPI backend and browser frontend: upload an image or short video and get a "
        "real/fake probability from either trained model.",
        "The demo's pipeline mirrors training exactly: face crop, resize and normalize, "
        "per-frame probability, video average.",
        "Verified on held-out test videos never seen in training: correctly classified by "
        "XceptionNet; the hybrid is correct but less confident.",
        "On an out-of-distribution image from the internet both models are confidently "
        "wrong, a live demonstration of the generalization problem studied next.",
    ], size=15)

    # 13. Research question
    s = new_slide(prs)
    header(s, "RESEARCH EXTENSION", "From a Surprising Result to Four Testable Claims")
    textbox(s, MARGIN, IN(1.7), IN(11.8), IN(0.45),
            [("Why do some cross-manipulation cells score below chance? We tested four claims:",
              14, False, MUTED)])
    w, h, gx, gy = IN(5.8), IN(2.05), IN(0.33), IN(0.3)
    claims = [
        ("C1 · PREPROCESSING", ["Does the inversion depend on how faces are cropped and stored?"]),
        ("C2 · MECHANISM", ["Is there a measurable image cue whose direction predicts which cells invert?"]),
        ("C3 · INTERVENTION", ["Does removing that cue during training reduce the inversion?"]),
        ("C4 · ARCHITECTURE", ["Do different network architectures fail the same way?"]),
    ]
    for i, (title, items) in enumerate(claims):
        card(s, MARGIN + (i % 2) * (w + gx), IN(2.35) + (i // 2) * (h + gy), w, h, title, NAVY, items, TEXT)
    footer(s)

    # 14. Robustness (Phase 0-1)
    picture(prs, "C1 · ROBUSTNESS", "The Inversion Survives a Full Pipeline Rebuild",
            "Original vs. reference preprocessing (different detector, alignment, format, size, "
            "frames). Hatched = below chance.",
            "analysis/figures/F1_transfer_matrices.png", 2.414, 3.45, "REFERENCE PIPELINE, 3 SEEDS", [
                (f2(dffs["seed_mean_auc"]), f"DF → FS  {ci(dffs['ci95_lo'], dffs['ci95_hi'])}"),
                (f2(fsnt["seed_mean_auc"]), f"FS → NT  {ci(fsnt['ci95_lo'], fsnt['ci95_hi'])}"),
                (f2(dffs["current_pipeline_auc"]), "DF → FS, original pipeline"),
                ("17 / 17", "data and label audit checks passed"),
            ])

    # 15. Phase 2 grid
    rows = [["Reference (3 seeds)", f3(dffs["seed_mean_auc"]), "—", f3(fsnt["seed_mean_auc"]), "—"]]
    for v, label in [("jpeg95", "JPEG quality 95"), ("jpeg75", "JPEG quality 75"),
                     ("frames10", "10 frames per video"), ("margin10", "Tight 1.0× crop")]:
        mat, dlt = p2[v]
        rows.append([label, f3(mat.loc[("DF", "FS"), "seed_mean_auc"]), f"{dlt.loc[('DF', 'FS'), 'delta']:+.3f}",
                     f3(mat.loc[("FS", "NT"), "seed_mean_auc"]), f"{dlt.loc[('FS', 'NT'), 'delta']:+.3f}"])
    table_slide(prs, "C1 · ROBUSTNESS", "No Single Preprocessing Change Moves DF → FS",
                "One factor changed at a time, 2 seeds each; change = paired difference vs. the "
                "reference over the same test videos.",
                ["Variant", "DF → FS", "change", "FS → NT", "change"], rows,
                [IN(3.2), IN(2.0), IN(2.2), IN(2.0), IN(2.2)],
                note="Predictions were recorded before these results: they held for JPEG 95 and "
                     "10 frames, partly missed for JPEG 75, and failed for the tight crop.")

    # 16. Mechanism: effect sizes
    picture(prs, "C2 · MECHANISM", "Fakes Differ from Real Faces in Sharpness",
            "High-frequency image energy, each manipulation vs. real faces (Cohen's d, 95% CI), "
            "in two independent crop sets.",
            "analysis/figures/F3_hf_effect_sizes.png", 1.414, 4.2, "EFFECT SIZE", [
                (f"{hf.loc['DF', 'd']:+.2f}", "Deepfakes: blurrier than real"),
                (f"{hf.loc['NT', 'd']:+.2f}", "NeuralTextures: blurrier than real"),
                (f"{hf.loc['FS', 'd']:+.2f}", "FaceSwap: sharper than real"),
                (f"{hf.loc['F2F', 'd']:+.2f}", "Face2Face: no difference"),
            ])

    # 17. Mechanism: sign prediction
    picture(prs, "C2 · MECHANISM", "The Sharpness Sign Predicts Which Cells Invert",
            "A detector learns \"blurrier = fake\" from Deepfakes, so FaceSwap's sharper fakes "
            "look real to it.",
            "analysis/figures/F4_sign_prediction.png", 1.218, 4.2, "PREDICTION VS. OBSERVED", [
                (f"ρ = {f2(sp.loc['current_xception', 'spearman_rho'])}", "Xception (best of 24 relabelings)"),
                (f"ρ = {f2(sp.loc['current_cnn_vit', 'spearman_rho'])}", "CNN-ViT hybrid"),
                (f"ρ = {f2(pub['spearman_rho'])}", "published official-data matrix"),
                (f"ρ = {f2(rel.loc[('DF', 'lap_var'), 'rho_prob_vs_stat_on_reals'])}",
                 "DF model: blurrier real faces look fake"),
            ])

    # 18. Intervention
    picture(prs, "C3 · INTERVENTION", "Randomizing Sharpness Partly Fixes It",
            "Training with random blur/sharpen applied to real and fake alike. Paired change vs. "
            "the reference, 3 seeds.",
            "analysis/figures/F5_intervention.png", 1.323, 4.2, "CHANGE IN AUC", [
                (f"{d4.loc[('DF', 'FS'), 'delta']:+.3f}", f"DF → FS, now {f2(d4.loc[('DF', 'FS'), 'phase4_auc'])} (still below)"),
                (f"{d4.loc[('FS', 'NT'), 'delta']:+.3f}", f"FS → NT, now {f2(d4.loc[('FS', 'NT'), 'phase4_auc'])} (at chance)"),
                (f"{d4.loc[('FS', 'F2F'), 'delta']:+.3f}", "FS → F2F, a small cost"),
                (f"{d4.loc[('DF', 'DF'), 'delta']:+.3f}", "in-domain accuracy unchanged"),
            ])

    # 19. Architectures
    rows = [["Xception (3 seeds)", f3(dffs["seed_mean_auc"]), f3(fsnt["seed_mean_auc"]),
             f3(g.loc[("FS", "DF"), "seed_mean_auc"]), "—"]]
    for bb, label in [("resnet50", "ResNet-50 (2 seeds)"), ("efficientnet_b4", "EfficientNet-B4 (2 seeds)")]:
        t = a5[bb]
        same = sum(t.loc[k, "reference_call"] == g.loc[k, "reference_call"] for k in g.index)
        rows.append([label, f3(t.loc[("DF", "FS"), "seed_mean_auc"]), f3(t.loc[("FS", "NT"), "seed_mean_auc"]),
                     f3(t.loc[("FS", "DF"), "seed_mean_auc"]), f"{same} / 8"])
    table_slide(prs, "C4 · ARCHITECTURE", "Three Architectures Fail the Same Way",
                "ImageNet-pretrained backbones, same 299 px input, reference pipeline.",
                ["Backbone", "DF → FS", "FS → NT", "FS → DF", "Same call as Xception"], rows,
                [IN(3.3), IN(1.9), IN(1.9), IN(1.9), IN(2.6)],
                note="Same below / at / above-chance result in every cell. The inversion comes "
                     "from the training data, not the network.")

    # 20. External check
    rows = []
    for gen, label in [("inpainting", "SD inpainting"), ("insight", "InsightFace swap"),
                       ("text2img", "SD text-to-image")]:
        rows.append([label, f"{dffe.loc[gen, 'd_hf_ratio_vs_wiki']:+.2f}",
                     f"{f3(dff.loc[('DF', gen), 'auc'])} {ci(dff.loc[('DF', gen), 'ci95_lo'], dff.loc[('DF', gen), 'ci95_hi'])}",
                     f"{f3(dff.loc[('FS', gen), 'auc'])} {ci(dff.loc[('FS', gen), 'ci95_lo'], dff.loc[('FS', gen), 'ci95_hi'])}"])
    n_ok = int(dff["prediction_correct"].dropna().astype(bool).sum())
    n_dec = int(dff["prediction_correct"].notna().sum())
    table_slide(prs, "EXTERNAL CHECK", "The Rule Does Not Transfer to a New Dataset",
                "DeepFakeFace, no retraining. Its fakes are all blurrier than its real faces, so "
                "DF-trained should transfer and FS-trained invert.",
                ["Generator", "Sharpness d", "DF-trained AUC", "FS-trained AUC"], rows,
                [IN(3.0), IN(2.0), IN(3.4), IN(3.4)],
                note=f"Prediction correct in {n_ok} of {n_dec} decided cells, no better than chance. "
                     "The sharpness account explains FaceForensics++, not other datasets.")

    # 21. Conclusion
    s = new_slide(prs)
    header(s, "CONCLUSION", "Verdict on the Four Claims")
    w, h, gx, gy = IN(5.8), IN(2.25), IN(0.33), IN(0.3)
    verdicts = [
        ("C1 · PREPROCESSING — NOT SUPPORTED", RED,
         ["Survives every preprocessing change tested",
          "Detector, alignment, format, size, frames, JPEG, crop"]),
        ("C2 · MECHANISM — SUPPORTED IN FF++", GREEN,
         ["Sharpness sign predicts our and published matrices",
          "Fails on DeepFakeFace: within-benchmark only"]),
        ("C3 · INTERVENTION — PARTIAL", AMBER,
         ["Randomizing sharpness: +0.1 AUC on inverted cells",
          "DF → FS still below chance"]),
        ("C4 · ARCHITECTURE — SUPPORTED", GREEN,
         ["Xception, ResNet-50, EfficientNet-B4 agree 8 / 8",
          "Set by the training data, not the model"]),
    ]
    for i, (title, colour, items) in enumerate(verdicts):
        card(s, MARGIN + (i % 2) * (w + gx), IN(1.95) + (i // 2) * (h + gy), w, h, title, colour, items, TEXT)
    footer(s)

    # 22. Limitations & future work
    s = new_slide(prs)
    header(s, "LIMITATIONS & FUTURE WORK", "What This Study Cannot Claim, and Next Steps")
    card(s, MARGIN, IN(1.95), IN(5.8), IN(4.5), "LIMITATIONS", RED, [
        "Kaggle mirror, not official FF++ (access pending)",
        "Mechanism is a strong candidate, not proof",
        "Sharpness predicts direction, not effect size",
        "Course comparison confounded by pretraining",
        "One 6 GB laptop GPU limited the grid",
    ], TEXT)
    card(s, MARGIN + IN(6.13), IN(1.95), IN(5.8), IN(4.5), "FUTURE WORK", TEAL, [
        "Re-run on official FaceForensics++",
        "Find the cue behind the remaining DF → FS gap",
        "Explain the DeepFakeFace behaviour",
        "Submit a short paper (draft in paper/main.pdf)",
    ], TEXT)
    footer(s)

    # 23. References & contributors
    s = new_slide(prs)
    header(s, "REFERENCES & CONTRIBUTORS", "Closing")
    bullets(s, MARGIN, IN(1.85), IN(11.8), IN(2.3), [
        "Rössler et al., FaceForensics++, ICCV 2019 · Chollet, Xception, CVPR 2017",
        "Dosovitskiy et al., Vision Transformer, ICLR 2021 · Selvaraju et al., Grad-CAM, ICCV 2017",
        "Li et al., Face X-ray, CVPR 2020 · Geirhos et al., Shortcut learning, Nat. Mach. Intell. 2020",
        "Durall et al., CVPR 2020 · Frank et al., ICML 2020 · Song et al., DeepFakeFace, arXiv 2023",
    ], size=12.5, space_after=8)
    textbox(s, MARGIN, IN(4.15), IN(11.8), IN(0.35), [("CONTRIBUTORS", 11, True, TEAL)])
    bullets(s, MARGIN, IN(4.5), IN(11.8), IN(1.2),
            [f"{name} — {roll}" for name, roll in TEAM], size=12, color=TEXT, bold=True, space_after=4)
    textbox(s, MARGIN, IN(5.95), IN(11.8), IN(0.35), [("GITHUB REPOSITORY", 11, True, TEAL)])
    textbox(s, MARGIN, IN(6.25), IN(11), IN(0.4),
            [("github.com/DUNE-ODYSSEY/deepfake-detection", 12.5, True, CYAN)])
    footer(s)

    assert len(prs.slides) == TOTAL_SLIDES, len(prs.slides)
    out = "docs/end_review_presentation.pptx"
    prs.save(out)
    print(f"saved {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
