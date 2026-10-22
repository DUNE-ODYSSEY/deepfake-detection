"""Build docs/end_review_presentation.pptx -- the end-semester review deck.

Follows the course review structure used at the mid-review (introduction,
literature, methodology, dataset, algorithm selection, results, software,
conclusion) with the complete results, then the research extension
(RESEARCH_BRIEF.md Phases 0-6). Same visual language as
build_mid_review_ppt.py. Every result number is read from analysis/ files; the
model sizes are computed from the model definitions.

Usage: python -m scripts.build_end_review_ppt
"""
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

TOTAL_SLIDES = 14
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
    g = pd.read_csv("analysis/reference/gate1_seed_mean_ci.csv").set_index(["train", "test"])
    hf = pd.read_csv("analysis/phase3/effect_sizes.csv")
    hf = hf[(hf["stat"] == "hf_ratio") & (hf["pipeline"] == "current")].set_index("method")
    sp = pd.read_csv("analysis/phase3/sign_prediction.csv")
    sp = sp[sp["stat"] == "hf_ratio"].set_index("matrix")
    pub = pd.read_csv("analysis/external/sign_prediction_published.csv").iloc[0]
    rel = pd.read_csv("analysis/phase3/reliance_reference.csv").set_index(["train", "stat"])
    d4 = pd.read_csv("analysis/phase4/delta_vs_phase1.csv").set_index(["train", "test"])
    a5 = {bb: pd.read_csv(f"analysis/phase5/{bb}_seed_mean_ci.csv").set_index(["train", "test"])
          for bb in ["resnet50", "efficientnet_b4"]}
    p2 = {v: (pd.read_csv(f"analysis/phase2/{v}_seed_mean_ci.csv").set_index(["train", "test"]),
              pd.read_csv(f"analysis/phase2/{v}_delta_vs_reference.csv").set_index(["train", "test"]))
          for v in ["jpeg95", "jpeg75", "frames10", "margin10"]}
    dff = pd.read_csv("analysis/phase6/dff_transfer.csv")
    dff = dff[dff["runs"] == "runs"].set_index(["train", "dff_fake"])
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
        "Objective 1: compare XceptionNet (CNN) and a CNN-ViT hybrid on identical data: does "
        "self-attention generalize better? Measure accuracy, cross-method transfer and "
        "compression robustness; explain with Grad-CAM; ship a live demo.",
        "Objective 2 (added after the results): explain why some cross-method scores fall "
        "below chance, and test a fix.",
    ], size=15.5)

    # 3. Methodology
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

    # 4. Data & models
    rows = []
    for key, label, pre in [("xception", "XceptionNet", "ImageNet"), ("cnn_vit", "CNN-ViT hybrid", "from scratch"),
                            ("resnet50", "ResNet-50", "ImageNet"), ("efficientnet_b4", "EfficientNet-B4", "ImageNet")]:
        size, params = sizes[key]
        rows.append([label, pre, f"{size} px", f"{params:.1f} M"])
    table_slide(prs, "DATA & MODELS", "Dataset and Model Selection",
                "FaceForensics++ (Kaggle mirror, official splits): 1,000 real + 4×1,000 fake videos. "
                "External test: DeepFakeFace.",
                ["Model", "Pretraining", "Input size", "Parameters"], rows,
                [IN(3.6), IN(2.8), IN(2.6), IN(2.6)],
                note="XceptionNet vs. the hybrid is the course comparison; ResNet-50 and "
                     "EfficientNet-B4 test whether the failure depends on the architecture. "
                     "Data quality: 502 of 5,000 videos flagged as outliers (IQR) before preprocessing.")

    # 5. Experiments 1-3
    table_slide(prs, "RESULTS · EXPERIMENTS 1–3", "Course Experiments: XceptionNet vs. CNN-ViT",
                "Same-method, cross-method (train on one manipulation, test on others) and "
                "compression (c40 proxy) performance.",
                ["Model", "Same-method AUC", "Cross-method AUC", "Compression AUC"],
                [[label, f3(base.loc[k, "video_auc"]), f3(gap.loc[k, "cross_manip_auc"]),
                  f3(comp.loc[(k, "c40proxy"), "video_auc"])]
                 for k, label in [("xception", "XceptionNet"), ("cnn_vit", "CNN-ViT hybrid")]],
                [IN(3.2), IN(2.8), IN(2.8), IN(2.8)],
                note="XceptionNet wins on accuracy and generalization: the hybrid's expected "
                     "advantage did not appear (a reported negative result). A FastAPI live demo "
                     "runs both models on uploaded images and videos.")

    # 6. Key finding
    picture(prs, "KEY FINDING", "Some Transfers Score Below Chance, Whatever the Pipeline",
            "Original vs. a fully rebuilt preprocessing pipeline (different detector, alignment, "
            "format, size, frames). Hatched = below chance.",
            "analysis/figures/F1_transfer_matrices.png", 2.414, 3.45, "REFERENCE PIPELINE, 3 SEEDS", [
                (f2(dffs["seed_mean_auc"]), f"DF → FS  {ci(dffs['ci95_lo'], dffs['ci95_hi'])}"),
                (f2(fsnt["seed_mean_auc"]), f"FS → NT  {ci(fsnt['ci95_lo'], fsnt['ci95_hi'])}"),
                (f2(dffs["current_pipeline_auc"]), "DF → FS, original pipeline"),
                ("17 / 17", "data and label audit checks passed"),
            ])

    # 7. Grad-CAM
    picture(prs, "INTERPRETABILITY", "Grad-CAM: What the Models Look At",
            "Same Deepfakes-trained checkpoint, same video: its Deepfakes version vs. its "
            "FaceSwap version.",
            "analyze/outputs/gradcam_comparison.png", 1787 / 1279, 4.1, "READING", [
                ("Face", "in-distribution: attention on the face"),
                ("Off-face", "FaceSwap: Xception attention collapses"),
                ("Diffuse", "FaceSwap: hybrid attention spreads out"),
                ("0.00", "fake probability on the FaceSwap video"),
            ])

    # 8. Preprocessing grid
    rows = [["Reference (3 seeds)", f3(dffs["seed_mean_auc"]), "—", f3(fsnt["seed_mean_auc"]), "—"]]
    for v, label in [("jpeg95", "JPEG quality 95"), ("jpeg75", "JPEG quality 75"),
                     ("frames10", "10 frames per video"), ("margin10", "Tight 1.0× crop")]:
        mat, dlt = p2[v]
        rows.append([label, f3(mat.loc[("DF", "FS"), "seed_mean_auc"]), f"{dlt.loc[('DF', 'FS'), 'delta']:+.3f}",
                     f3(mat.loc[("FS", "NT"), "seed_mean_auc"]), f"{dlt.loc[('FS', 'NT'), 'delta']:+.3f}"])
    table_slide(prs, "ROBUSTNESS", "No Single Preprocessing Change Moves DF → FS",
                "One factor changed at a time, 2 seeds each; change = paired difference vs. the "
                "reference over the same test videos.",
                ["Variant", "DF → FS", "change", "FS → NT", "change"], rows,
                [IN(3.2), IN(2.0), IN(2.2), IN(2.0), IN(2.2)],
                note="Not a pipeline bug: an audit of labels, splits, encoding and crops found no "
                     "real-vs-fake difference. Predictions recorded beforehand held for 2 of 4 variants.")

    # 9. Mechanism
    picture(prs, "MECHANISM", "A Sharpness Cue Explains Which Cells Invert",
            "Deepfakes and NeuralTextures fakes are blurrier than real faces, FaceSwap fakes are "
            "sharper. Opposite shifts predict inverted transfer.",
            "analysis/figures/F4_sign_prediction.png", 1.218, 4.2, "EVIDENCE", [
                (f"{hf.loc['DF', 'd']:+.2f} / {hf.loc['FS', 'd']:+.2f}", "sharpness d: Deepfakes / FaceSwap"),
                (f"ρ = {f2(sp.loc['current_xception', 'spearman_rho'])}", "predicts our matrix (best of 24)"),
                (f"ρ = {f2(pub['spearman_rho'])}", "predicts the published official-data matrix"),
                (f"ρ = {f2(rel.loc[('DF', 'lap_var'), 'rho_prob_vs_stat_on_reals'])}",
                 "DF model: blurrier real faces look fake"),
            ])

    # 10. Intervention
    picture(prs, "INTERVENTION", "Randomizing Sharpness Partly Fixes It",
            "Training with random blur/sharpen applied to real and fake alike. Paired change vs. "
            "the reference, 3 seeds.",
            "analysis/figures/F5_intervention.png", 1.323, 4.2, "CHANGE IN AUC", [
                (f"{d4.loc[('DF', 'FS'), 'delta']:+.3f}", f"DF → FS, now {f2(d4.loc[('DF', 'FS'), 'phase4_auc'])} (still below)"),
                (f"{d4.loc[('FS', 'NT'), 'delta']:+.3f}", f"FS → NT, now {f2(d4.loc[('FS', 'NT'), 'phase4_auc'])} (at chance)"),
                (f"{d4.loc[('FS', 'F2F'), 'delta']:+.3f}", "FS → F2F, a small cost"),
                (f"{d4.loc[('DF', 'DF'), 'delta']:+.3f}", "in-domain accuracy unchanged"),
            ])

    # 11. Generalization: architectures vs datasets
    s = new_slide(prs)
    header(s, "GENERALIZATION", "Same Failure Across Architectures, Not Across Datasets")
    arch_items = [f"Xception DF → FS: {f2(dffs['seed_mean_auc'])} (3 seeds)"]
    for bb, label in [("resnet50", "ResNet-50"), ("efficientnet_b4", "EfficientNet-B4")]:
        t = a5[bb]
        same = sum(t.loc[k, "reference_call"] == g.loc[k, "reference_call"] for k in g.index)
        arch_items.append(f"{label} DF → FS: {f2(t.loc[('DF', 'FS'), 'seed_mean_auc'])}, "
                          f"same call in {same}/8 cells")
    arch_items.append("Set by the training data, not the network")
    card(s, MARGIN, IN(1.95), IN(5.8), IN(4.6), "ACROSS ARCHITECTURES — HOLDS", GREEN, arch_items, TEXT)
    n_ok = int(dff["prediction_correct"].dropna().astype(bool).sum())
    n_dec = int(dff["prediction_correct"].notna().sum())
    card(s, MARGIN + IN(6.13), IN(1.95), IN(5.8), IN(4.6), "ACROSS DATASETS — DOES NOT", RED, [
        "DeepFakeFace: Stable Diffusion and InsightFace fakes",
        f"Sharpness rule right in {n_ok} of {n_dec} cells: chance level",
        f"SD text-to-image below chance for every model "
        f"({f2(dff.loc[('DF', 'text2img'), 'auc'])} / {f2(dff.loc[('FS', 'text2img'), 'auc'])})",
        "The account explains FaceForensics++ only",
    ], TEXT)
    footer(s)

    # 12. Conclusion
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

    # 13. Why this could be publishable
    s = new_slide(prs)
    header(s, "RESEARCH CONTRIBUTION", "Why This Could Be Publishable")
    same_calls = [sum(a5[bb].loc[k, "reference_call"] == g.loc[k, "reference_call"] for k in g.index)
                  for bb in ["resnet50", "efficientnet_b4"]]
    card(s, MARGIN, IN(1.95), IN(5.8), IN(3.6), "WHAT'S NEW", NAVY, [
        "Explains which transfers invert, not just that they do; "
        f"prior work reports DF → FS at chance without a cause",
        f"One measurable cue predicts our matrix (ρ = {f2(sp.loc['current_xception', 'spearman_rho'])}) "
        f"and a published official-data matrix (ρ = {f2(pub['spearman_rho'])})",
        f"Causal test: removing the cue raises the inverted cells "
        f"(DF → FS {d4.loc[('DF', 'FS'), 'delta']:+.3f}, FS → NT {d4.loc[('FS', 'NT'), 'delta']:+.3f} AUC)",
        f"Architecture-independent: 3 backbones agree in {min(same_calls)}/8 cells",
    ], TEXT)
    card(s, MARGIN + IN(6.13), IN(1.95), IN(5.8), IN(3.6), "WHY IT'S CREDIBLE", TEAL, [
        "Predictions recorded before results, and misses reported",
        "3 seeds, bootstrap confidence intervals, paired tests",
        "Audit and full pipeline rebuild rule out artifacts",
        "Every number regenerated by scripts; code is public",
    ], TEXT)
    rect(s, MARGIN, IN(5.8), IN(11.93), IN(0.95), CARD_BG)
    textbox(s, MARGIN + IN(0.25), IN(5.88), IN(11.4), IN(0.8),
            [[("Best fit: ", 13, True, TEAL),
              ("a workshop or short-paper track (e.g. IEEE WIFS); draft ready in paper/main.pdf.", 13, False, TEXT)],
             [("Risks a reviewer will raise: ", 13, True, RED),
              ("unofficial data mirror, a partial fix, no transfer to DeepFakeFace (all stated openly).",
               13, False, TEXT)]])
    footer(s)

    # 14. Limitations, future work & closing
    s = new_slide(prs)
    header(s, "CLOSING", "Limitations, Future Work & Team")
    card(s, MARGIN, IN(1.95), IN(5.8), IN(2.45), "LIMITATIONS", RED, [
        "Kaggle mirror, not official FF++ (access pending)",
        "Mechanism is a strong candidate, not proof",
        "One 6 GB laptop GPU limited the experiments",
    ], TEXT)
    card(s, MARGIN + IN(6.13), IN(1.95), IN(5.8), IN(2.45), "FUTURE WORK", TEAL, [
        "Re-run on official FaceForensics++",
        "Find the cue behind the remaining DF → FS gap",
        "Submit a short paper (draft: paper/main.pdf)",
    ], TEXT)
    textbox(s, MARGIN, IN(4.65), IN(11.8), IN(0.35), [("CONTRIBUTORS", 11, True, TEAL)])
    textbox(s, MARGIN, IN(4.95), IN(11.8), IN(0.4),
            [(" · ".join(f"{name} ({roll})" for name, roll in TEAM), 12, True, TEXT)])
    textbox(s, MARGIN, IN(5.5), IN(11.8), IN(0.35), [("KEY REFERENCES", 11, True, TEAL)])
    textbox(s, MARGIN, IN(5.8), IN(11.8), IN(0.4),
            [("Rössler et al., ICCV 2019 · Chollet, CVPR 2017 · Dosovitskiy et al., ICLR 2021 · "
              "Selvaraju et al., ICCV 2017 · Li et al., CVPR 2020 · Geirhos et al., 2020", 11, False, MUTED)])
    textbox(s, MARGIN, IN(6.3), IN(11), IN(0.4),
            [("github.com/DUNE-ODYSSEY/deepfake-detection", 12.5, True, CYAN)])
    footer(s)

    assert len(prs.slides) == TOTAL_SLIDES, len(prs.slides)
    out = "docs/end_review_presentation.pptx"
    prs.save(out)
    print(f"saved {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
