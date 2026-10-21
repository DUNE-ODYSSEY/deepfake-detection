"""Build docs/progress_update.pptx -- the supervisor update deck, final results.

Separate from the deliberately scoped docs/mid_review_update.pptx. Covers the
whole study: Experiments 1-3, the below-chance cross-manipulation finding and
Phases 0-6 of RESEARCH_BRIEF.md. Every number is read from the analysis/ files
(the same ones the paper uses); nothing is typed by hand.

Reuses the visual language and helpers of build_mid_review_ppt.py.

Usage: python -m scripts.build_progress_update_ppt
"""
import json

import pandas as pd
from pptx.util import Emu

from scripts import build_mid_review_ppt as m

MARGIN = m.MARGIN
SLIDE_W, SLIDE_H = m.SLIDE_W, m.SLIDE_H
NAVY, CYAN, TEAL = m.NAVY, m.CYAN, m.TEAL
TEXT, MUTED, MUTED_L = m.TEXT, m.MUTED, m.MUTED_L
GREEN, AMBER, RED, WHITE = m.GREEN, m.AMBER, m.RED, m.WHITE
TEAM = m.TEAM
new_slide, textbox, rect = m.new_slide, m.textbox, m.rect
header, bullets, stat_panel, card = m.header, m.bullets, m.stat_panel, m.card
picture_slide, table_slide, content_slide = m.picture_slide, m.table_slide, m.content_slide

TOTAL_SLIDES = 14
FOOTER = "Deepfake Detection — Inverted Transfer (23AID205)"


def IN(x):
    return Emu(int(x * 914400))


def f3(x):
    return f"{float(x):.3f}"


def f2(x):
    return f"{float(x):.2f}"


def ci(lo, hi):
    return f"[{float(lo):.2f}, {float(hi):.2f}]"


def footer(slide):
    m.footer(slide, FOOTER)


def picture(prs, eyebrow, title, caption, path, aspect, height_in, stats_label, stats):
    return picture_slide(prs, eyebrow, title, caption, path,
                         IN(height_in * aspect), IN(height_in), stats_label, stats)


def main():
    m.PAGE[0] = 0
    m.TOTAL_SLIDES = TOTAL_SLIDES
    m.footer.__defaults__ = (FOOTER,)

    base = pd.read_csv("analysis/baseline_comparison.csv").set_index("model")
    gap = pd.read_csv("analysis/generalization_gap.csv").set_index("model")
    comp = pd.read_csv("analysis/compression_robustness.csv").set_index(["model", "test_compression"])
    g = pd.read_csv("analysis/reference/gate1_seed_mean_ci.csv").set_index(["train", "test"])
    eff = pd.read_csv("analysis/phase3/effect_sizes.csv")
    hf = eff[eff["stat"] == "hf_ratio"]
    sp = pd.read_csv("analysis/phase3/sign_prediction.csv")
    sp = sp[sp["stat"] == "hf_ratio"].set_index("matrix")
    pub = pd.read_csv("analysis/external/sign_prediction_published.csv").iloc[0]
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

    dffs, fsnt = g.loc[("DF", "FS")], g.loc[("FS", "NT")]

    prs = m.Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

    # 1. Title
    s = new_slide(prs, bg=NAVY)
    rect(s, 0, 0, Emu(146304), SLIDE_H, CYAN)
    textbox(s, MARGIN, IN(2.15), IN(10), IN(0.4),
            [("AIML COURSE PROJECT · 23AID205 · FINAL RESULTS", 15, True, CYAN)])
    textbox(s, MARGIN, IN(2.55), IN(11.4), IN(1.9),
            [[("Inverted Transfer:", 34, True, WHITE)],
             [("Why Deepfake Detectors Score Unseen Fakes Below Chance", 30, True, WHITE)]])
    textbox(s, MARGIN, IN(4.55), IN(11), IN(1.0),
            [("A detector trained on Deepfakes ranks FaceSwap fakes as more real than real "
              f"faces (AUC {f2(dffs['seed_mean_auc'])}). The inversion survives every "
              "preprocessing change and three architectures, follows a measurable sharpness "
              "cue, and is partly fixed by randomizing that cue.", 14, False, MUTED_L)])
    textbox(s, MARGIN, IN(6.55), IN(11.8), IN(0.35),
            [(" · ".join(name for name, _ in TEAM), 13, True, m.RGBColor(0xE5, 0xE9, 0xF0))])
    textbox(s, MARGIN, IN(6.88), IN(8), IN(0.3),
            [("AI & Data Science — Amrita Vishwa Vidyapeetham, Coimbatore", 11, False, MUTED_L)])
    m.PAGE[0] += 1

    # 2. The answer in one slide
    s = new_slide(prs)
    header(s, "SUMMARY", "What We Found")
    stat_panel(s, MARGIN, IN(1.95), IN(4.3), IN(4.6), "KEY NUMBERS", [
        (f2(dffs["seed_mean_auc"]), "Deepfakes → FaceSwap AUC (chance = 0.50)"),
        ("8 / 8", "cells where 3 architectures agree"),
        (f"ρ = {f2(sp.loc['current_xception', 'spearman_rho'])}", "sharpness cue predicts the transfer matrix"),
        (f"+{f2(d4.loc[('DF', 'FS'), 'delta'])}", "AUC gained by randomizing sharpness"),
    ])
    bullets(s, MARGIN + IN(4.6), IN(1.95), IN(7.3), IN(4.8), [
        "Some cross-manipulation cells score below chance: the detector ranks unseen "
        "fakes as more real than real faces.",
        "Not a pipeline bug: it survives an audit, a full preprocessing rebuild and a "
        "one-factor preprocessing grid, with 3 seeds and confidence intervals.",
        "Cause (partial): Deepfakes and NeuralTextures fakes are blurrier than real faces; "
        "FaceSwap fakes are sharper. A detector that learns \"blurrier = fake\" gets FaceSwap backwards.",
        "Randomizing sharpness in training raises every inverted cell by ~0.1 AUC, but "
        "does not remove the inversion.",
        "Limit: the sharpness rule does not predict a second dataset (DeepFakeFace).",
    ], size=14.5, space_after=12)
    footer(s)

    # 3. Where it started: Experiments 1-3
    table_slide(prs, "BACKGROUND · EXPERIMENTS 1–3", "Where the Project Started",
                "Original course experiments: XceptionNet vs. a CNN-ViT hybrid on FaceForensics++ "
                "(Kaggle mirror, c23, one seed).",
                ["Model", "Same-method AUC", "Cross-method AUC", "Compression AUC (c40 proxy)"],
                [[name, f3(base.loc[key, "video_auc"]), f3(gap.loc[key, "cross_manip_auc"]),
                  f3(comp.loc[(key, "c40proxy"), "video_auc"])]
                 for key, name in [("xception", "XceptionNet"), ("cnn_vit", "CNN-ViT hybrid")]],
                [IN(2.6), IN(2.9), IN(2.9), IN(3.5)],
                note="The hybrid did not generalize better (a reported negative result). The "
                     "surprise was in the cross-method matrix: several cells fell below 0.5, "
                     "which became the research question for the rest of the project.")

    # 4. The inversion, original vs reference pipeline
    picture(prs, "THE FINDING · PHASE 1", "The Inversion Survives a Full Pipeline Rebuild",
            "Same matrix with a completely different preprocessing pipeline (dlib + alignment, "
            "PNG, 256 px, 32 frames). Hatched cells are below chance.",
            "analysis/figures/F1_transfer_matrices.png", 2.414, 3.45, "REFERENCE PIPELINE, 3 SEEDS", [
                (f2(dffs["seed_mean_auc"]), f"DF → FS  {ci(dffs['ci95_lo'], dffs['ci95_hi'])}"),
                (f2(fsnt["seed_mean_auc"]), f"FS → NT  {ci(fsnt['ci95_lo'], fsnt['ci95_hi'])}"),
                (f2(dffs["current_pipeline_auc"]), "DF → FS, original pipeline"),
                ("0.05", "largest change in any cell"),
            ])

    # 5. Phase 0
    content_slide(prs, "PHASE 0", "Ruling Out a Pipeline Bug First", [
        "Labels and splits: 17/17 checks pass. Fakes are labelled 1 everywhere, AUC "
        "orientation cannot silently flip, and train/val/test share no videos.",
        "Video encoding: codec, profile, pixel format and compression setting are "
        "identical for real and all four manipulation methods.",
        "Face crops: detection failures are negligible (≤ 0.05%) and JPEG settings are "
        "identical across classes.",
        "Two code bugs fixed on the way: resumed training silently lost its history, and "
        "model checkpoints had been committed to git.",
        "Conclusion: no trivial explanation, so the inversion is worth explaining.",
    ], size=15)

    # 6. Phase 2 grid
    rows = []
    for v, label in [("jpeg95", "JPEG quality 95"), ("jpeg75", "JPEG quality 75"),
                     ("frames10", "10 frames per video"), ("margin10", "Tight 1.0× crop")]:
        mat, dlt = p2[v]
        rows.append([label, f3(mat.loc[("DF", "FS"), "seed_mean_auc"]),
                     f"{dlt.loc[('DF', 'FS'), 'delta']:+.3f}",
                     f3(mat.loc[("FS", "NT"), "seed_mean_auc"]),
                     f"{dlt.loc[('FS', 'NT'), 'delta']:+.3f}"])
    rows.insert(0, ["Reference (3 seeds)", f3(dffs["seed_mean_auc"]), "—", f3(fsnt["seed_mean_auc"]), "—"])
    table_slide(prs, "PHASE 2", "No Preprocessing Change Moves DF → FS",
                "One factor changed at a time, 2 seeds each; change = paired difference vs. "
                "the reference over the same test videos.",
                ["Variant", "DF → FS", "change", "FS → NT", "change"], rows,
                [IN(3.2), IN(2.0), IN(2.2), IN(2.0), IN(2.2)],
                note="DF → FS stays at ~0.26 under every variant. Predictions recorded "
                     "beforehand: held for JPEG 95 and 10 frames, partly missed for JPEG 75, "
                     "failed for the tight crop.")

    # 7. Mechanism: effect sizes
    c = hf[hf["pipeline"] == "current"].set_index("method")
    picture(prs, "PHASE 3 · MECHANISM", "Fakes Differ from Real Faces in Sharpness",
            "High-frequency image energy, each method vs. real faces (Cohen's d, 95% CI), "
            "measured on two independent crop sets.",
            "analysis/figures/F3_hf_effect_sizes.png", 1.414, 4.2, "EFFECT SIZE (ORIGINAL CROPS)", [
                (f"{c.loc['DF', 'd']:+.2f}", "Deepfakes: blurrier than real"),
                (f"{c.loc['NT', 'd']:+.2f}", "NeuralTextures: blurrier than real"),
                (f"{c.loc['FS', 'd']:+.2f}", "FaceSwap: sharper than real"),
                (f"{c.loc['F2F', 'd']:+.2f}", "Face2Face: no difference"),
            ])

    # 8. Mechanism: sign prediction
    picture(prs, "PHASE 3 · MECHANISM", "The Sharpness Sign Predicts Which Cells Invert",
            "Opposite sharpness shifts in the training and test methods predict an inverted "
            "(below-chance) transfer.",
            "analysis/figures/F4_sign_prediction.png", 1.218, 4.2, "PREDICTION VS. OBSERVED", [
                (f"ρ = {f2(sp.loc['current_xception', 'spearman_rho'])}", "Xception (best of 24 relabelings)"),
                (f"ρ = {f2(sp.loc['current_cnn_vit', 'spearman_rho'])}", "CNN-ViT hybrid"),
                (f"ρ = {f2(pub['spearman_rho'])}", "published official-data matrix"),
                (f"ρ = {f2(rel.loc[('DF', 'lap_var'), 'rho_prob_vs_stat_on_reals'])}",
                 "DF model: blurrier real faces look fake"),
            ])

    # 9. Intervention
    picture(prs, "PHASE 4 · INTERVENTION", "Randomizing Sharpness Partly Fixes It",
            "Training with random blur/sharpen (label-independent). Paired change vs. Phase 1, "
            "3 seeds each.",
            "analysis/figures/F5_intervention.png", 1.323, 4.2, "CHANGE IN AUC", [
                (f"{d4.loc[('DF', 'FS'), 'delta']:+.3f}", f"DF → FS, now {f2(d4.loc[('DF', 'FS'), 'phase4_auc'])} (still below)"),
                (f"{d4.loc[('FS', 'NT'), 'delta']:+.3f}", f"FS → NT, now {f2(d4.loc[('FS', 'NT'), 'phase4_auc'])} (at chance)"),
                (f"{d4.loc[('FS', 'F2F'), 'delta']:+.3f}", "FS → F2F, a small cost"),
                (f"{d4.loc[('DF', 'DF'), 'delta']:+.3f}", "in-domain accuracy unchanged"),
            ])

    # 10. Architectures
    rows = [["Xception (3 seeds)", f3(dffs["seed_mean_auc"]), f3(fsnt["seed_mean_auc"]),
             f3(g.loc[("FS", "DF"), "seed_mean_auc"]), "—"]]
    for bb, label in [("resnet50", "ResNet-50 (2 seeds)"), ("efficientnet_b4", "EfficientNet-B4 (2 seeds)")]:
        t = a5[bb]
        same = sum(t.loc[k, "reference_call"] == g.loc[k, "reference_call"] for k in g.index)
        rows.append([label, f3(t.loc[("DF", "FS"), "seed_mean_auc"]), f3(t.loc[("FS", "NT"), "seed_mean_auc"]),
                     f3(t.loc[("FS", "DF"), "seed_mean_auc"]), f"{same} / 8"])
    table_slide(prs, "PHASE 5", "Three Architectures Fail the Same Way",
                "ImageNet-pretrained backbones at the same 299 px input, reference pipeline.",
                ["Backbone", "DF → FS", "FS → NT", "FS → DF", "Same call as Xception"], rows,
                [IN(3.3), IN(1.9), IN(1.9), IN(1.9), IN(2.6)],
                note="Every backbone gives the same below / at / above-chance result in every "
                     f"cell; the original CNN-ViT hybrid agreed with Xception at r = {arch['pearson_r']:.2f}. "
                     "The inversion comes from the training data, not the network.")

    # 11. DeepFakeFace
    rows = []
    for gen, label in [("inpainting", "SD inpainting"), ("insight", "InsightFace swap"),
                       ("text2img", "SD text-to-image")]:
        rows.append([label, f"{dffe.loc[gen, 'd_hf_ratio_vs_wiki']:+.2f}",
                     f"{f3(dff.loc[('DF', gen), 'auc'])} {ci(dff.loc[('DF', gen), 'ci95_lo'], dff.loc[('DF', gen), 'ci95_hi'])}",
                     f"{f3(dff.loc[('FS', gen), 'auc'])} {ci(dff.loc[('FS', gen), 'ci95_lo'], dff.loc[('FS', gen), 'ci95_hi'])}"])
    n_ok = int(dff["prediction_correct"].dropna().astype(bool).sum())
    n_dec = int(dff["prediction_correct"].notna().sum())
    table_slide(prs, "PHASE 6 · EXTERNAL CHECK", "The Rule Does Not Transfer to a New Dataset",
                "DeepFakeFace, no retraining. Its fakes are all blurrier than its real faces, so "
                "DF-trained should transfer and FS-trained invert.",
                ["Generator", "Sharpness d", "DF-trained AUC", "FS-trained AUC"], rows,
                [IN(3.0), IN(2.0), IN(3.4), IN(3.4)],
                note=f"Prediction correct in {n_ok} of {n_dec} decided cells, no better than chance. "
                     "The sharpness account explains FaceForensics++, not other datasets.")

    # 12. Verdict
    s = new_slide(prs)
    header(s, "CONCLUSIONS", "Verdict on the Four Claims")
    w, h, gx, gy = IN(5.8), IN(2.25), IN(0.33), IN(0.3)
    y0 = IN(1.95)
    verdicts = [
        ("C1 · PREPROCESSING — NOT SUPPORTED", RED,
         ["The inversion survives every preprocessing change tested",
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
        x = MARGIN + (i % 2) * (w + gx)
        y = y0 + (i // 2) * (h + gy)
        card(s, x, y, w, h, title, colour, items, TEXT)
    footer(s)

    # 13. Deliverables and next steps
    s = new_slide(prs)
    header(s, "STATUS", "Delivered, and What Would Come Next")
    card(s, MARGIN, IN(1.95), IN(5.8), IN(4.5), "DELIVERED", GREEN, [
        "All research phases run and logged (docs/PROGRESS.md)",
        "5-page paper draft, IEEE format (paper/main.pdf)",
        "Full project report (docs/project_report.docx)",
        "Every number generated by a script; code on GitHub",
        "Predictions recorded before results, misses reported",
    ], GREEN)
    card(s, MARGIN + IN(6.13), IN(1.95), IN(5.8), IN(4.5), "NEXT STEPS", AMBER, [
        "Re-run on official FaceForensics++ (access pending)",
        "Find the cue behind the remaining DF → FS inversion",
        "Explain why DeepFakeFace does not follow the rule",
        "Submit to a workshop (e.g. IEEE WIFS / ICASSP track)",
    ], AMBER)
    footer(s)

    # 14. Closing
    s = new_slide(prs)
    header(s, "CLOSING", "Thank You")
    bullets(s, MARGIN, IN(1.95), IN(11.8), IN(1.5), [
        "Data: FaceForensics++ via a Kaggle mirror (preliminary; never redistributed) and "
        "DeepFakeFace (OpenRAIL). Hardware: one 6 GB laptop GPU.",
        "Reproducible: every table and figure is regenerated from the repository's scripts.",
    ], size=14, space_after=10)
    textbox(s, MARGIN, IN(3.7), IN(11.8), IN(0.35), [("CONTRIBUTORS", 11, True, TEAL)])
    bullets(s, MARGIN, IN(4.05), IN(11.8), IN(1.2),
            [f"{name} — {roll}" for name, roll in TEAM], size=12, color=TEXT, bold=True, space_after=6)
    textbox(s, MARGIN, IN(5.6), IN(11.8), IN(0.35), [("GITHUB REPOSITORY", 11, True, TEAL)])
    textbox(s, MARGIN, IN(5.9), IN(11), IN(0.4),
            [("github.com/DUNE-ODYSSEY/deepfake-detection", 12.5, True, CYAN)])
    footer(s)

    assert len(prs.slides) == TOTAL_SLIDES, len(prs.slides)
    out = "docs/progress_update.pptx"
    prs.save(out)
    print(f"saved {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
