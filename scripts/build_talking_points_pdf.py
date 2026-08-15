"""Build docs/talking_points.pdf -- a timed presenter script for the
11-slide mid-review deck (docs/mid_review_update.pptx), split across the
4-person team in a specific presenting order, budgeted to fit a 15-minute
faculty review slot.

Usage: python -m scripts.build_talking_points_pdf
"""
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable, KeepTogether)

NAVY = colors.HexColor("#0F172A")
CYAN = colors.HexColor("#22B8CF")
TEAL = colors.HexColor("#0E7490")
TEXT = colors.HexColor("#1A202C")
MUTED = colors.HexColor("#475569")
CARD_BG = colors.HexColor("#F1F5F9")
RED = colors.HexColor("#DC2626")

# (presenter, [(slide_no, slide_title, mm:ss, [talking points], is_handoff)])
SCRIPT = [
    ("Vipin Sudhakar", "0:00", "3:15", [
        (1, "Title", "0:30", [
            "Good [morning/afternoon]. We're presenting our AIML course project for "
            "23AID205: Deepfake Detection -- comparing XceptionNet against a CNN-ViT "
            "hybrid on FaceForensics++.",
            "In the next 15 minutes we'll cover our approach, three experiments' worth "
            "of real results, an interpretability analysis, and a live working demo.",
            "I'm Vipin -- I'll open with the motivation and our methodology, then hand "
            "off to my teammates.",
        ]),
        (2, "Motivation, Problem & Objectives", "1:30", [
            "Deepfakes have moved from a research curiosity to a real misinformation "
            "and fraud vector.",
            "The problem: published detectors usually report one headline accuracy "
            "number, on the same manipulation type and compression level they trained "
            "on. That number is misleading -- a detector at 99% in a paper can drop to "
            "~57% AUC, worse than random, the moment the test distribution shifts. And "
            "that failure is invisible until deployment.",
            "So our objectives were threefold: fairly compare XceptionNet against a "
            "CNN-ViT hybrid -- does adding global self-attention actually help "
            "generalization? Measure not just accuracy but full cross-manipulation "
            "generalization and compression robustness. And explain *why* generalization "
            "succeeds or fails, via Grad-CAM, rather than just reporting a number.",
        ]),
        (3, "Methodology & Pipeline", "1:15", [
            "Our pipeline in six steps: raw FF++ video, MTCNN face extraction, official "
            "video-level splits, training, evaluation, and analysis.",
            "One thing worth flagging: we used the *official* video-level train/val/test "
            "splits throughout -- not random frame-level splits, which would leak the "
            "same identity into both train and test and quietly inflate accuracy.",
            "Handoff: \"Rithvik will now walk you through our dataset and the two "
            "architectures we compared.\"",
        ]),
    ]),
    ("Rithvik Arulprakash", "3:15", "3:00", [
        (4, "Dataset, Preprocessing & EDA", "1:30", [
            "FaceForensics++: 5,000 videos total -- 1,000 real YouTube interviews plus "
            "4,000 manipulated, 1,000 each across four methods: Deepfakes, Face2Face, "
            "FaceSwap, and NeuralTextures, at c23 compression.",
            "Preprocessing: MTCNN face detection on GPU, 20 frames per video, a 1.3x crop "
            "margin to capture blending-boundary artifacts, giving us roughly 99,987 face "
            "crops in total.",
            "Before committing to full preprocessing, we ran an IQR outlier check on "
            "frame count and file size -- flagged 502 of 5,000 videos, about 10%, so data "
            "quality was verified upfront, not assumed.",
            "One honest caveat we'll return to: official raw and heavily-compressed data "
            "access from FaceForensics++ was still pending at review time -- we'll show "
            "how we handled that later.",
        ]),
        (5, "Models: XceptionNet vs. CNN-ViT Hybrid", "1:30", [
            "XceptionNet is our baseline -- a standard depthwise-separable CNN, "
            "ImageNet-pretrained, purely convolutional with only local receptive fields.",
            "The CNN-ViT hybrid pairs a ResNet-50 feature extractor with a 6-layer, "
            "8-head Transformer encoder. The question we're testing: does global "
            "self-attention let the model relate distant regions -- like blending "
            "inconsistencies -- that a local CNN structurally can't see jointly?",
            "Critically, both models share the exact same optimizer, schedule, batch "
            "size, and data. Architecture is the only variable, by design -- so any "
            "difference in results is attributable to that choice alone.",
            "Handoff: \"Venu will now take you through what we actually found across our "
            "three experiments.\"",
        ]),
    ]),
    ("Venugopalan Gangadharan", "6:15", "4:15", [
        (6, "Baseline Comparison", "1:15", [
            "Both models trained and tested on all four methods together -- the "
            "same-distribution, headline number most papers stop at.",
            "XceptionNet: 97.29% video accuracy, 0.9958 AUC. CNN-ViT hybrid: 94.71% "
            "video accuracy, 0.9836 AUC.",
            "XceptionNet leads on every single metric here -- our first sign that the "
            "hybrid's added complexity isn't paying off on this task.",
        ]),
        (7, "Cross-Manipulation Generalization", "1:45", [
            "This is the actual novel question we set out to answer: train on ONE "
            "manipulation method, test on all four -- a full 4x4 matrix, per model.",
            "[Point to heatmap] The diagonal -- same-manipulation -- stays high across "
            "the board for both models. Off-diagonal is where it falls apart.",
            "Our generalization gap: XceptionNet 0.398, CNN-ViT hybrid 0.421. "
            "XceptionNet actually generalizes *better* than the attention-based hybrid -- "
            "the opposite of our original hypothesis. We're reporting that honestly as a "
            "genuine negative result, not hiding it.",
            "Worst case for both: a model trained only on Deepfakes, tested on FaceSwap "
            "-- AUC of 0.256 and 0.198. That's worse than random guessing.",
        ]),
        (8, "Compression Robustness", "1:15", [
            "Official c0 (raw) and c40 (heavy compression) access from FaceForensics++ "
            "was still pending, and no c40 mirror exists on Kaggle -- so rather than "
            "skip this experiment, we built a documented workaround.",
            "We locally re-transcoded just our held-out test videos to a harsher "
            "compression level with ffmpeg -- we call it 'proxy-c40.' We're upfront that "
            "this double-compresses rather than compressing once from raw, so it's an "
            "approximation -- used only for evaluation, never for training.",
            "Result: both models drop from ~0.99 AUC to roughly 0.80-0.82 under heavier "
            "compression. The hybrid degrades slightly less -- a 0.166 drop versus 0.194 "
            "for XceptionNet.",
            "Handoff: \"Harshith will now show you *why* this generalization gap happens, "
            "and our live demo.\"",
        ]),
    ]),
    ("Harshith Kv", "10:30", "3:45", [
        (9, "Grad-CAM: Why Generalization Fails", "1:30", [
            "We wanted to explain the gap, not just report it. This is Grad-CAM on the "
            "same video, same Deepfakes-only checkpoint, tested on two different "
            "manipulation methods.",
            "Same-manipulation: attention locks tightly onto the central face -- nose, "
            "mouth, eyes -- exactly where Deepfakes' blending artifacts show up. "
            "Confident, and interpretably correct.",
            "Cross-manipulation, same video, switched to FaceSwap: XceptionNet's "
            "attention collapses to a random spot off the face entirely. The hybrid's "
            "attention goes diffuse across the whole frame, including the background.",
            "So neither model is merely 'uncertain' on unseen manipulations -- each "
            "confidently learned a narrow signal, with nothing sensible to fall back on "
            "once that exact signal is gone.",
        ]),
        (10, "Live Demo", "1:15", [
            "[Switch to the live demo] Upload a held-out test video, get a real/fake "
            "probability from either trained model -- the pipeline mirrors training "
            "exactly: face crop, normalize, per-frame score, video-level average.",
            "We verified this against a curated set of test-split videos never seen in "
            "training -- correctly and confidently classified.",
            "One honest moment worth sharing: we also tested it against a random "
            "deepfake-claim image circulating online, outside FaceForensics++ entirely. "
            "Both models were confidently wrong -- which is exactly what our "
            "generalization-gap numbers predicted, not a contradiction of them.",
        ]),
        (11, "Conclusion & Closing", "1:00", [
            "To close: XceptionNet outperforms the CNN-ViT hybrid on every metric we "
            "measured, contradicting our initial hypothesis that attention would help.",
            "Both architectures share the same underlying weakness -- narrow, "
            "method-specific artifacts rather than a general 'this is manipulated' "
            "signal -- which we confirmed visually, not just numerically.",
            "Future work: official c0/c40 once FaceForensics++ approval comes through, "
            "multi-seed runs for statistical confidence, and extending Grad-CAM to a "
            "genuinely-uncertain case rather than only the worse-than-random one.",
            "Thank you -- happy to take questions.",
        ]),
    ]),
]

TOTAL_TIME = "14:15"


def styles():
    return {
        "title": ParagraphStyle("title", fontName="Helvetica-Bold", fontSize=20,
                                textColor=NAVY, spaceAfter=10, leading=24),
        "subtitle": ParagraphStyle("subtitle", fontName="Helvetica", fontSize=11,
                                   textColor=MUTED, spaceAfter=18),
        "presenter": ParagraphStyle("presenter", fontName="Helvetica-Bold", fontSize=15,
                                    textColor=colors.white),
        "presenter_meta": ParagraphStyle("presenter_meta", fontName="Helvetica", fontSize=9.5,
                                         textColor=colors.HexColor("#CBD5E1")),
        "slide_title": ParagraphStyle("slide_title", fontName="Helvetica-Bold", fontSize=12,
                                      textColor=TEXT),
        "slide_time": ParagraphStyle("slide_time", fontName="Helvetica-Bold", fontSize=10,
                                     textColor=TEAL, alignment=2),
        "point": ParagraphStyle("point", fontName="Helvetica", fontSize=10, textColor=TEXT,
                                leading=14, spaceAfter=6, leftIndent=14, bulletIndent=2),
        "handoff": ParagraphStyle("handoff", fontName="Helvetica-Oblique", fontSize=10,
                                  textColor=TEAL, leading=14, spaceAfter=6, leftIndent=14),
    }


def slide_block(st, slide_no, title, time_budget, points):
    header_tbl = Table(
        [[Paragraph(f"Slide {slide_no} &nbsp;&nbsp; {title}", st["slide_title"]),
          Paragraph(time_budget, st["slide_time"])]],
        colWidths=[5.4 * inch, 1.0 * inch])
    header_tbl.setStyle(TableStyle([
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("LINEBELOW", (0, 0), (-1, -1), 0.75, CYAN),
    ]))
    flow = [header_tbl, Spacer(1, 4)]
    for p in points:
        is_handoff = p.startswith("Handoff:")
        style = st["handoff"] if is_handoff else st["point"]
        bullet = "" if is_handoff else "•  "
        flow.append(Paragraph(bullet + p, style))
    flow.append(Spacer(1, 10))
    return flow


def presenter_header(st, name, start_time, duration):
    tbl = Table(
        [[Paragraph(name, st["presenter"]),
          Paragraph(f"Starts at {start_time} &nbsp;&nbsp;—&nbsp;&nbsp; {duration} budget",
                    st["presenter_meta"])]],
        colWidths=[3.5 * inch, 2.9 * inch])
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), NAVY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (0, 0), 12),
        ("TOPPADDING", (0, 0), (-1, -1), 10),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
        ("ALIGN", (1, 0), (1, 0), "RIGHT"),
        ("RIGHTPADDING", (1, 0), (1, 0), 12),
    ]))
    return tbl


def main():
    st = styles()
    doc = SimpleDocTemplate("docs/talking_points.pdf", pagesize=LETTER,
                            topMargin=0.6 * inch, bottomMargin=0.6 * inch,
                            leftMargin=0.65 * inch, rightMargin=0.65 * inch)
    story = []

    story.append(Paragraph("Presentation Script — Mid-Review", st["title"]))
    story.append(Paragraph(
        "Deepfake Detection: XceptionNet vs. CNN-ViT Hybrid on FaceForensics++ &nbsp;"
        f"·&nbsp; 23AID205 &nbsp;·&nbsp; Target: under 15:00, budgeted at {TOTAL_TIME}",
        st["subtitle"]))

    order_tbl = Table(
        [["#", "Presenter", "Slides", "Starts", "Budget"]] +
        [[str(i + 1), name, f"{slides[0][0]}–{slides[-1][0]}", start, dur]
         for i, (name, start, dur, slides) in enumerate(SCRIPT)],
        colWidths=[0.35 * inch, 2.3 * inch, 1.0 * inch, 1.0 * inch, 1.0 * inch])
    order_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), NAVY),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), 10),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, CARD_BG]),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#E2E8F0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("ALIGN", (0, 0), (0, -1), "CENTER"),
        ("ALIGN", (2, 0), (-1, -1), "CENTER"),
    ]))
    story.append(order_tbl)
    story.append(Spacer(1, 16))
    story.append(HRFlowable(width="100%", thickness=1, color=colors.HexColor("#E2E8F0")))
    story.append(Spacer(1, 14))

    for name, start_time, duration, slides in SCRIPT:
        story.append(presenter_header(st, name, start_time, duration))
        story.append(Spacer(1, 10))
        for slide_no, title, time_budget, points in slides:
            story.extend(slide_block(st, slide_no, title, time_budget, points))
        story.append(Spacer(1, 8))

    doc.build(story)
    print(f"saved docs/talking_points.pdf")


if __name__ == "__main__":
    main()
