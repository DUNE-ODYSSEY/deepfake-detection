"""Build docs/end_review_talking_points.pdf -- timed presenter script for the
14-slide end-semester deck (docs/end_review_presentation.pptx), split across
the team: Vipin 1-4, Rithvik 5-7, Venu 8-10, Harshith 11-14. Budgeted for a
15-minute review slot, plus likely questions and how to walk through
docs/project_code_simplified.py if the code is asked about.

Numbers are read from the same analysis/ files the deck uses.

Usage: python -m scripts.build_end_review_talking_points
"""
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.units import inch
from reportlab.platypus import (HRFlowable, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

from scripts.build_talking_points_pdf import (CARD_BG, MUTED, NAVY, TEAL, TEXT, presenter_header,
                                              slide_block, styles)

OUT = "docs/end_review_talking_points.pdf"


def numbers():
    g = pd.read_csv("analysis/reference/gate1_seed_mean_ci.csv").set_index(["train", "test"])
    base = pd.read_csv("analysis/baseline_comparison.csv").set_index("model")
    gap = pd.read_csv("analysis/generalization_gap.csv").set_index("model")
    sp = pd.read_csv("analysis/phase3/sign_prediction.csv")
    sp = sp[sp["stat"] == "hf_ratio"].set_index("matrix")
    pub = pd.read_csv("analysis/external/sign_prediction_published.csv").iloc[0]
    e = pd.read_csv("analysis/phase3/effect_sizes.csv")
    e = e[(e["stat"] == "hf_ratio") & (e["pipeline"] == "current")].set_index("method")
    d4 = pd.read_csv("analysis/phase4/delta_vs_phase1.csv").set_index(["train", "test"])
    dff = pd.read_csv("analysis/phase6/dff_transfer.csv")
    dff = dff[dff["runs"] == "runs"]
    decided = dff[dff["prediction_correct"].notna()]
    return {
        "xc_auc": f"{base.loc['xception', 'video_auc']:.3f}",
        "vit_auc": f"{base.loc['cnn_vit', 'video_auc']:.3f}",
        "xc_cross": f"{gap.loc['xception', 'cross_manip_auc']:.2f}",
        "vit_cross": f"{gap.loc['cnn_vit', 'cross_manip_auc']:.2f}",
        "dffs": f"{g.loc[('DF', 'FS'), 'seed_mean_auc']:.2f}",
        "dffs_cur": f"{g.loc[('DF', 'FS'), 'current_pipeline_auc']:.2f}",
        "fsnt": f"{g.loc[('FS', 'NT'), 'seed_mean_auc']:.2f}",
        "rho": f"{sp.loc['current_xception', 'spearman_rho']:.2f}",
        "rho_pub": f"{pub['spearman_rho']:.2f}",
        "d_df": f"{e.loc['DF', 'd']:+.2f}",
        "d_fs": f"{e.loc['FS', 'd']:+.2f}",
        "aug_dffs": f"{d4.loc[('DF', 'FS'), 'delta']:+.2f}",
        "aug_dffs_now": f"{d4.loc[('DF', 'FS'), 'phase4_auc']:.2f}",
        "aug_fsnt": f"{d4.loc[('FS', 'NT'), 'delta']:+.2f}",
        "dff_ok": f"{int(decided['prediction_correct'].astype(bool).sum())} of {len(decided)}",
    }


def script(n):
    return [
        ("Vipin Sudhakar", "0:00", "3:20", [
            (1, "Title", "0:20", [
                "Good [morning/afternoon]. We're presenting our end-semester review for 23AID205: "
                "deepfake detection on FaceForensics++.",
                "We started by comparing two detectors, and ended up studying why detectors fail "
                "on fakes they haven't seen. I'm Vipin, and I'll start.",
            ]),
            (2, "Motivation, Problem & Objectives", "1:00", [
                "Deepfakes are a real fraud and misinformation risk. Detectors are usually tested "
                "on the same kind of fake they were trained on.",
                "That hides the real problem: a detector at 99% can fail on a new kind of fake, "
                "and nobody notices until it's deployed.",
                "Our first objective was the course comparison: XceptionNet vs. a CNN-ViT hybrid. "
                "Our second came from our own results: explain why some scores fell below chance.",
            ]),
            (3, "End-to-End Pipeline", "1:00", [
                "Six steps, left to right: raw videos, face crops, official splits, training, "
                "evaluation, analysis.",
                "Splits are by video, not frame, so the same person is never in both train and test.",
                "We train on one manipulation type and test on all four. Every number comes from a "
                "script over saved scores, not typed by hand.",
            ]),
            (4, "Dataset and Model Selection", "1:00", [
                "FaceForensics++: 1,000 real YouTube videos and 4,000 fakes from four methods.",
                "We used a Kaggle copy because official access is still pending, so we label "
                "results as preliminary. We checked it for problems before training.",
                "XceptionNet vs. the hybrid is the course comparison. ResNet-50 and "
                "EfficientNet-B4 came later, to test whether the failure depends on the model.",
                "Handoff: Rithvik will take you through the results.",
            ]),
        ]),
        ("Rithvik Arulprakash", "3:20", "3:30", [
            (5, "Course Experiments: XceptionNet vs. CNN-ViT", "1:15", [
                f"Same-method, XceptionNet reached AUC {n['xc_auc']} and the hybrid {n['vit_auc']}.",
                f"On unseen methods both dropped to about {n['xc_cross']} and {n['vit_cross']}. "
                "The hybrid did not generalize better, a negative result we report honestly.",
                "Under heavier compression both degrade a lot. We also built a live web demo that "
                "runs both models.",
            ]),
            (6, "Key Finding: Below-Chance Transfers", "1:30", [
                "Here's the surprise. Train on Deepfakes, test on FaceSwap: the AUC is "
                f"{n['dffs_cur']}, below 0.5.",
                "Below 0.5 means the detector ranks the fakes as more real than real faces: it is "
                "confidently backwards, not just unsure.",
                "We rebuilt the whole preprocessing pipeline, with a different face detector, "
                f"alignment, format and size, and got {n['dffs']} again. So it's not a pipeline bug.",
            ]),
            (7, "Grad-CAM: What the Models Look At", "0:45", [
                "Same model, same video. On Deepfakes it looks at the face. On FaceSwap its attention "
                "collapses off the face, and it calls the fake real with probability near zero.",
                "Handoff: Venu will explain why this happens.",
            ]),
        ]),
        ("Venugopalan Gangadharan", "6:50", "4:15", [
            (8, "No Single Preprocessing Change Moves DF → FS", "1:15", [
                "We changed one preprocessing step at a time: JPEG quality, number of frames, "
                "and how tightly we crop the face.",
                f"Deepfakes to FaceSwap stayed at about {n['dffs']} every time. It's the most "
                "robust result in the study.",
                "We wrote down predictions before seeing these results. Two held, two didn't, and "
                "we report both.",
            ]),
            (9, "A Sharpness Cue Explains Which Cells Invert", "1:45", [
                f"The cause: Deepfakes fakes are blurrier than real faces (d = {n['d_df']}), and "
                f"FaceSwap fakes are sharper (d = {n['d_fs']}).",
                "A detector trained on Deepfakes learns 'blurry means fake'. FaceSwap fakes are "
                "sharp, so it calls them real. That's the inversion.",
                f"This one rule predicts our whole matrix (rank correlation {n['rho']}) and a "
                f"published result from another lab on the official data ({n['rho_pub']}).",
            ]),
            (10, "Randomizing Sharpness Partly Fixes It", "1:15", [
                "To test cause and effect, we randomly blurred or sharpened every training image, "
                "so sharpness stopped being useful.",
                f"Deepfakes to FaceSwap rose by {n['aug_dffs']} and FaceSwap to NeuralTextures by "
                f"{n['aug_fsnt']}, with no loss on the detector's own task.",
                f"But Deepfakes to FaceSwap is still {n['aug_dffs_now']}, below chance. So sharpness "
                "is a real cause, but not the whole story.",
                "Handoff: Harshith will cover how far this generalizes and wrap up.",
            ]),
        ]),
        ("Harshith Kv", "11:05", "3:45", [
            (11, "Same Failure Across Architectures, Not Across Datasets", "1:15", [
                "Across architectures it holds: ResNet-50 and EfficientNet-B4 fail exactly like "
                "Xception, in all 8 cells. So it comes from the training data, not the network.",
                "Across datasets it doesn't: on DeepFakeFace the sharpness rule was right in "
                f"{n['dff_ok']} cells, which is chance. Our explanation is specific to FaceForensics++.",
            ]),
            (12, "Verdict on the Four Claims", "1:00", [
                "Preprocessing causes it: not supported. The mechanism: supported within "
                "FaceForensics++. The fix: partial. Architecture independence: supported.",
            ]),
            (13, "Why This Could Be Publishable", "1:00", [
                "What's new: we explain which transfers invert, with one measurable cue that also "
                "predicts another lab's published results.",
                "Why it's credible: predictions recorded in advance, three seeds, confidence "
                "intervals, and all code public.",
                "We'd target a workshop. The risks are the unofficial data, the partial fix and the "
                "second dataset, and we state all three openly.",
            ]),
            (14, "Limitations, Future Work & Team", "0:30", [
                "Next: re-run on official data and find what still drives the remaining inversion. "
                "A paper draft is ready. Thank you, we're happy to take questions.",
            ]),
        ]),
    ]


QA = [
    ("Isn't an AUC below 0.5 just a bug, like flipped labels?",
     "We checked: the audit's polarity check confirms fakes are labelled 1 and the AUC can't "
     "silently flip, the audit passed 17/17 checks, and a completely rebuilt pipeline gives the "
     "same number. A "
     "published paper also reports this cell at chance on official data. (Venu / Rithvik)"),
    ("Why didn't the CNN-ViT hybrid do better?",
     "It was trained from scratch at 224 px while Xception was pretrained at 299 px, so the "
     "comparison isn't fully fair; we say so. The later study shows the failure doesn't depend "
     "on the architecture anyway. (Rithvik)"),
    ("Why a Kaggle copy and not the official dataset?",
     "Official access was requested but is still pending. Our sharpness rule also predicts a "
     "published official-data result, which suggests the mirror isn't the cause. (Vipin)"),
    ("Can you explain the code?",
     "Open docs/project_code_simplified.py and go top to bottom: splits (1), face cropping (2), "
     "dataset and the blur/sharpen augmentation (3), the two models (4), training (5), "
     "evaluation with confidence intervals (6), the cross-manipulation matrix (7) and the "
     "sharpness rule (8). Each section names the real file it comes from. (any presenter)"),
    ("Is it publishable?",
     "As a workshop or short paper, yes: the cue, its prediction of published results and the "
     "intervention are new. Main-track would need official data and a full fix. (Harshith)"),
]


def main():
    n = numbers()
    plan = script(n)
    st = styles()
    doc = SimpleDocTemplate(OUT, pagesize=LETTER, topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            leftMargin=0.65 * inch, rightMargin=0.65 * inch)
    story = [Paragraph("Presentation Script — End-Semester Review", st["title"]),
             Paragraph("Deepfake Detection on FaceForensics++ &nbsp;·&nbsp; 23AID205 &nbsp;·&nbsp; "
                       "14 slides, target 15:00 (budgeted 14:50)", st["subtitle"])]
    order = Table([["#", "Presenter", "Slides", "Starts", "Budget"]] +
                  [[str(i + 1), name, f"{s[0][0]}–{s[-1][0]}", start, dur]
                   for i, (name, start, dur, s) in enumerate(plan)],
                  colWidths=[0.35 * inch, 2.3 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch])
    order.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY), ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"), ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CARD_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6), ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (0, -1), "CENTER"), ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ]))
    story += [order, Spacer(1, 6),
              Paragraph("Slide 1 (title) is opened by Vipin, who continues straight into slides 2–4.",
                        st["subtitle"]),
              HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0")), Spacer(1, 12)]
    for name, start, dur, slides in plan:
        story += [presenter_header(st, name, start, dur), Spacer(1, 10)]
        for slide_no, title, budget, points in slides:
            story += slide_block(st, slide_no, title, budget, points)
        story.append(Spacer(1, 8))
    story += [PageBreak(), Paragraph("Likely Questions", st["title"]),
              Paragraph("Short answers, with who should take each one.", st["subtitle"])]
    for q, a in QA:
        story += [Paragraph(q, st["slide_title"]), Spacer(1, 3), Paragraph(a, st["point"]), Spacer(1, 6)]
    doc.build(story)
    print(f"saved {OUT}")


if __name__ == "__main__":
    main()
