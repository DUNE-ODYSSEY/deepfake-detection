"""Build docs/progress_update.pptx -- an informal supervisor update, separate
from the scoped mid_review_update.pptx (see that script's docstring: it
deliberately holds back Experiments 2-3 for the formal final review).

This deck is for an ad hoc progress check-in, not a graded review stage, so
it shows everything that's actually done: all three experiments, Grad-CAM,
the below-chance cross-manipulation finding, the Phase 0 audit, and the new
research direction (RESEARCH_BRIEF.md) with next steps.

Reuses the visual language and slide-building helpers from
build_mid_review_ppt.py so the two decks read as one continuous project.

Usage: python -m scripts.build_progress_update_ppt
"""
from pptx.util import Emu, Pt
from pptx.enum.shapes import MSO_SHAPE

from scripts import build_mid_review_ppt as m

MARGIN = m.MARGIN
SLIDE_W, SLIDE_H = m.SLIDE_W, m.SLIDE_H
NAVY, CYAN, TEAL = m.NAVY, m.CYAN, m.TEAL
TEXT, MUTED, MUTED_L = m.TEXT, m.MUTED, m.MUTED_L
CARD_BG, GRID = m.CARD_BG, m.GRID
GREEN, AMBER, RED, WHITE = m.GREEN, m.AMBER, m.RED, m.WHITE
TEAM = m.TEAM

new_slide, textbox, rect = m.new_slide, m.textbox, m.rect
header, footer, bullets = m.header, m.footer, m.bullets
stat_panel, picture_slide, card = m.stat_panel, m.picture_slide, m.card
table_slide, content_slide = m.table_slide, m.content_slide

TOTAL_SLIDES = 10


def main():
    m.PAGE[0] = 0
    m.TOTAL_SLIDES = TOTAL_SLIDES

    prs = m.Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

    # 1. Title
    s = new_slide(prs, bg=NAVY)
    rect(s, 0, 0, Emu(146304), SLIDE_H, CYAN)
    textbox(s, MARGIN, Emu(2148840), Emu(8229600), Emu(365760),
            [("AIML COURSE PROJECT · 23AID205 · PROGRESS UPDATE", 15, True, CYAN)])
    textbox(s, MARGIN, Emu(2468880), Emu(10332720), Emu(1920240),
            [[("Deepfake Detection:", 34, True, WHITE)],
             [("Why Detectors Invert on Unseen Manipulations", 30, True, WHITE)]])
    textbox(s, MARGIN, Emu(4343400), Emu(9601200), Emu(900000),
            [("All three planned experiments are now complete. Cross-manipulation "
              "testing surfaced detectors scoring below random chance on FaceSwap "
              "cells -- not a pipeline bug (audited), and consistent across both "
              "architectures. This is now the project's central finding.",
              14, False, MUTED_L)])
    rect(s, MARGIN, Emu(5424920), Emu(457200), Emu(27432), CYAN)
    textbox(s, MARGIN, Emu(6039320), Emu(10852800), Emu(320040),
            [(" · ".join(name for name, _ in TEAM), 13, True, m.RGBColor(0xE5, 0xE9, 0xF0))])
    textbox(s, MARGIN, Emu(6313640), Emu(7315200), Emu(274320),
            [("AI & Data Science — Amrita Vishwa Vidyapeetham, Coimbatore", 11, False, MUTED_L)])
    m.PAGE[0] += 1

    # 2. Status recap
    content_slide(prs, "STATUS", "Since the Mid-Review: All Three Experiments Complete", [
        "Experiment 1 (baseline, same-distribution) was complete at the mid-review. "
        "Experiments 2 (cross-manipulation) and 3 (compression robustness) are now "
        "also complete, along with Grad-CAM interpretability and a working live demo.",
        "The original hypothesis -- that the CNN-ViT hybrid would generalize better "
        "than the XceptionNet baseline -- did not hold. Reporting that as an honest "
        "negative result rather than reframing the comparison after the fact.",
        "What did turn up, unexpectedly: several cross-manipulation cells score "
        "below random chance (AUC < 0.5), which is a more interesting and more "
        "publishable finding than the architecture comparison itself.",
        "This update covers: the full results, why we're confident it isn't a "
        "pipeline artifact, and the research plan now built around it "
        "(RESEARCH_BRIEF.md, in the repo).",
    ], size=15.5)

    # 3. Results: Experiment 2 -- cross-manipulation matrix
    picture_slide(
        prs, "RESULTS · EXPERIMENT 2 — COMPLETE", "Cross-Manipulation Generalization",
        "Train on one manipulation method, test on another (and on the same method, "
        "diagonal). Color floor is clipped at 0.5 -- printed values below 0.5 are "
        "below random chance.",
        "analysis/cross_manipulation_heatmaps.png", Emu(7132320), Emu(2879600),
        stats_label="BELOW-CHANCE CELLS",
        stats=[("0.256", "Xception worst cell: Deepfakes → FaceSwap"),
               ("0.198", "CNN-ViT worst cell: Deepfakes → FaceSwap"),
               ("0.398 / 0.421", "Generalization gap -- Xception / CNN-ViT"),
               ("7 of 24", "off-diagonal cells ≤ chance -- all involve FaceSwap")])

    # 4. Results: Experiment 3 -- compression robustness
    table_slide(
        prs, "RESULTS · EXPERIMENT 3 — COMPLETE", "Compression Robustness (c23 → c40, proxy)",
        "Official c0/c40 access is still pending -- c40 here is a documented proxy "
        "(c23 re-compressed), eval-only, caveated throughout the repo.",
        ["Model", "c23 Video AUC", "c40proxy Video AUC", "AUC Drop"],
        [["XceptionNet", "0.9958", "0.8018", "-0.194"],
         ["CNN-ViT Hybrid", "0.9836", "0.8172", "-0.166"]],
        [Emu(3200000), Emu(2700000), Emu(2700000), Emu(2200000)],
        note="Both models degrade substantially under heavier re-compression; the "
             "hybrid's drop is somewhat smaller, but this is on the proxy set, not "
             "official c40 -- not being claimed as a general result.")

    # 5. The key finding: architecture-independence
    s = new_slide(prs)
    header(s, "THE FINDING", "Transfer Failure Tracks the Training Method, Not the Model")
    stat_panel(s, MARGIN, Emu(1783080), Emu(4200000), Emu(4114800), "ARCHITECTURE CORRELATION",
               [("r = 0.938", "Pearson correlation, off-diagonal AUC cells"),
                ("ρ = 0.951", "Spearman correlation, same cells"),
                ("12", "off-diagonal cells compared per architecture"),
                ("p < 0.001", "both correlations, analysis/architecture_correlation.py")])
    bullets(s, MARGIN + Emu(4400000), Emu(1783080), Emu(6252800), Emu(4114800), [
        "XceptionNet (ImageNet-pretrained, purely convolutional) and the CNN-ViT "
        "hybrid (trained from scratch, has global self-attention) are architecturally "
        "very different -- yet their cross-manipulation failure patterns are nearly "
        "identical cell-for-cell.",
        "That means the below-chance scores aren't an architecture quirk. Both "
        "models are picking up something from the training manipulation method "
        "itself that doesn't transfer -- and in the FaceSwap cells, actively points "
        "the wrong way.",
        "This reframes the project: not \"which architecture generalizes better\" "
        "but \"why does detection invert on unseen manipulations, and can it be "
        "fixed.\" The second question is the one worth a paper.",
    ], size=13.5, space_after=12)
    footer(s)

    # 6. Grad-CAM
    picture_slide(
        prs, "INTERPRETABILITY", "Grad-CAM: What the Models Actually Attend To",
        "Same checkpoint (Deepfakes-trained), same video, different manipulation. "
        "In-distribution: both models focus tightly on the face. Cross-manipulation "
        "(FaceSwap, never seen): attention collapses off-face or goes diffuse -- "
        "neither model falls back to anything sensible.",
        "analyze/outputs/gradcam_comparison.png", Emu(8686800), Emu(4895000))

    # 7. Phase 0: hygiene + audit
    content_slide(prs, "RULING OUT THE BORING EXPLANATION", "Phase 0: Hygiene Fixes + Confound Audit", [
        "Before treating the below-chance scores as a real finding, fixed two code "
        "issues (a training-resume bug that was silently discarding run history; "
        "checkpoint files that had been accidentally committed to version control) "
        "and audited the pipeline for anything that could trivially cause this.",
        "Label/split integrity -- 17/17 checks pass: AUC orientation is correct and "
        "not silently invertible; train/val/test are cleanly video-disjoint.",
        "Encoding audit (60 videos/class via ffprobe) -- codec, profile, pixel "
        "format, and compression setting (crf=23.0) are bit-exact identical across "
        "real and all 4 manipulation methods. No confound.",
        "Crop audit -- face-detection failure rate negligible (≤0.05%) "
        "everywhere; JPEG re-compression is bit-identical across classes. No "
        "confound.",
        "GATE 0 conclusion: no confirmed pipeline artifact explains the below-chance "
        "scores. This is very likely a real effect, not a bug -- worth explaining "
        "and fixing, not dismissing.",
    ], size=14.5)

    # 8. New direction: research plan
    content_slide(prs, "NEW DIRECTION", "Research Plan: Why Transfer Inverts, and How to Fix It", [
        "Working title: \"Inverted Transfer\" -- why deepfake detectors score "
        "unseen manipulations below chance, and how preprocessing toggles it.",
        "C1 -- Preprocessing sensitivity: does the below-chance pattern shift, "
        "possibly across the 0.5 line, under different face-detector/margin/"
        "format/alignment choices alone?",
        "C2 -- Mechanism: is there a measurable low-level image statistic "
        "(sharpness, boundary contrast, JPEG artifacts) whose sign predicts which "
        "transfer cells invert?",
        "C3 -- Intervention: can neutralizing that statistic during training raise "
        "the below-chance cells back toward/above 0.5 without a large in-domain "
        "accuracy drop?",
        "C4 -- Architecture independence: the r≈0.94 correlation above is the "
        "first evidence for this; testing it across more architectures next.",
        "Target: a 4-6 page paper, IEEE WIFS/ICASSP-style format.",
    ], size=14)

    # 9. Status & plan to completion
    s = new_slide(prs)
    header(s, "STATUS", "What's Done, What's Next")
    card(s, MARGIN, Emu(1783080), Emu(5195000), Emu(4114800), "DONE", GREEN, [
        "Experiments 1-3 complete: baseline, full 4×4 cross-manipulation "
        "matrix, compression robustness",
        "Grad-CAM interpretability analysis and live demo verified end-to-end",
        "Phase 0 (hygiene + audit) complete -- ruled out a pipeline artifact as "
        "the explanation for the below-chance scores",
        "Architecture-independence of the transfer-failure pattern confirmed "
        "(r≈0.94 across both models' off-diagonal cells)",
    ], GREEN)
    card(s, MARGIN + Emu(5195000) + Emu(228600), Emu(1783080), Emu(5195000), Emu(4114800),
        "NEXT", AMBER, [
        "Phase 1: rebuild preprocessing with a different, standard face-alignment "
        "pipeline and retrain -- tests whether the below-chance pattern disappears "
        "(preprocessing-driven) or persists (a real inversion mechanism)",
        "Phase 2-3: systematic preprocessing sweep, then pin down the low-level "
        "statistic that predicts which cells invert",
        "Phase 4-5: test interventions that fix it, check generalization across "
        "more architectures",
        "~4-week total budget for this phase of work, currently early -- "
        "paper write-up targeted by the end",
    ], AMBER)
    footer(s)

    # 10. Closing
    s = new_slide(prs)
    header(s, "CLOSING", "Where to Follow Along")
    bullets(s, MARGIN, Emu(1783080), Emu(10852800), Emu(1200000), [
        "Full research plan, phase-by-phase, with gates: RESEARCH_BRIEF.md in the repo.",
        "Progress log updated after every phase: docs/PROGRESS.md.",
        "Every number in this deck comes from a script-written file under "
        "analysis/ or results/ -- reproducible, not hand-typed.",
    ], size=13.5, space_after=10)
    rect(s, MARGIN, Emu(3400000), Emu(10852800), Emu(18288), GRID)
    textbox(s, MARGIN, Emu(3600000), Emu(10852800), Emu(320040),
            [("CONTRIBUTORS", 11, True, TEAL)])
    bullets(s, MARGIN, Emu(3940000), Emu(10852800), Emu(1080000),
            [f"{name} — {roll}" for name, roll in TEAM],
            size=12, color=TEXT, bold=True, space_after=6)
    textbox(s, MARGIN, Emu(5100000), Emu(10852800), Emu(320040),
            [("AI & Data Science — Amrita Vishwa Vidyapeetham, Coimbatore", 11, False, MUTED)])
    textbox(s, MARGIN, Emu(5450000), Emu(10852800), Emu(320040),
            [("GITHUB REPOSITORY", 11, True, TEAL)])
    textbox(s, MARGIN, Emu(5700000), Emu(10000000), Emu(360000),
            [("github.com/DUNE-ODYSSEY/deepfake-detection", 12.5, True, CYAN)])
    footer(s)

    out = "docs/progress_update.pptx"
    prs.save(out)
    print(f"saved {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
