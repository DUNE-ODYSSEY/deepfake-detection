"""Build docs/talking_points.pdf -- a timed presenter script for the
9-slide mid-review deck (docs/mid_review_update.pptx), split across the
4-person team in a specific presenting order, budgeted to fit a 15-minute
faculty review slot.

Usage: python -m scripts.build_talking_points_pdf
"""
from reportlab.lib import colors
from reportlab.lib.pagesizes import LETTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table,
                                TableStyle, HRFlowable)

NAVY = colors.HexColor("#0F172A")
CYAN = colors.HexColor("#22B8CF")
TEAL = colors.HexColor("#0E7490")
TEXT = colors.HexColor("#1A202C")
MUTED = colors.HexColor("#475569")
CARD_BG = colors.HexColor("#F1F5F9")
RED = colors.HexColor("#DC2626")

# (presenter, [(slide_no, slide_title, mm:ss, [talking points])])
SCRIPT = [
    ("Rithvik Arulprakash", "0:00", "3:15", [
        (1, "Title", "0:30", [
            "Good [morning/afternoon]. We're presenting our mid-review update for our "
            "AIML course project, 23AID205: Deepfake Detection -- comparing "
            "XceptionNet against a CNN-ViT hybrid on FaceForensics++.",
            "Today we'll cover our methodology, our completed data pipeline, and our "
            "first experiment's results. Cross-manipulation generalization, "
            "compression robustness, interpretability, and the live demo are all "
            "underway and will be presented in full at the final review.",
            "I'm Rithvik -- I'll open with the motivation and our methodology, then "
            "hand off to my teammates.",
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
            "As of today, steps 1 through 3 -- the entire data pipeline -- are complete, "
            "and we've carried steps 4 and 5 through for our first experiment. The "
            "remaining training and analysis work continues for Experiments 2 and 3.",
            "Handoff: \"Vipin will now walk you through our dataset work, which is "
            "fully complete.\"",
        ]),
    ]),
    ("Vipin Sudhakar", "3:15", "2:30", [
        (4, "Dataset & Preprocessing — Complete", "1:30", [
            "FaceForensics++: 5,000 videos total -- 1,000 real YouTube interviews plus "
            "4,000 manipulated, 1,000 each across four methods: Deepfakes, Face2Face, "
            "FaceSwap, and NeuralTextures, at c23 compression.",
            "We used the official video-level train/val/test splits throughout -- not "
            "random frame-level splits, which would leak the same identity into both "
            "train and test and quietly inflate accuracy.",
            "Preprocessing: MTCNN face detection on GPU, 20 frames per video, a 1.3x crop "
            "margin to capture blending-boundary artifacts. This gave us roughly 99,987 "
            "face crops -- the entire dataset is preprocessed and ready. This stage is "
            "fully done.",
        ]),
        (5, "EDA: Data Quality Check (IQR)", "1:00", [
            "Before committing to full preprocessing, we ran an outlier check on every "
            "video's frame count and file size using the IQR method.",
            "[Point to chart] 209 videos flagged on frame count, 326 on file size -- 502 "
            "unique outliers overall, about 10% of the dataset -- so data quality was "
            "verified upfront, not assumed.",
            "Handoff: \"Venu will now introduce our two architectures and our first "
            "result.\"",
        ]),
    ]),
    ("Venugopalan Gangadharan", "5:45", "2:45", [
        (6, "Models: XceptionNet vs. CNN-ViT Hybrid", "1:30", [
            "XceptionNet is our baseline -- a standard depthwise-separable CNN, "
            "ImageNet-pretrained, purely convolutional with only local receptive fields.",
            "The CNN-ViT hybrid pairs a ResNet-50 feature extractor with a 6-layer, "
            "8-head Transformer encoder. The question we're testing: does global "
            "self-attention let the model relate distant regions -- like blending "
            "inconsistencies -- that a local CNN structurally can't see jointly?",
            "Both models share the exact same optimizer, schedule, batch size, and data "
            "-- architecture is the only variable, by design.",
        ]),
        (7, "Results — Experiment 1: Baseline Comparison", "1:15", [
            "This is our first completed experiment: both models trained and tested on "
            "all four methods together, the same-distribution headline number.",
            "XceptionNet: 97.29% video accuracy, 0.9958 AUC. CNN-ViT hybrid: 94.71% "
            "video accuracy, 0.9836 AUC -- XceptionNet leads on every metric here.",
            "Whether that holds up under cross-manipulation and compression shift is "
            "exactly what Experiments 2 and 3 will show at the final review.",
            "Handoff: \"Harshith will close with where we stand and what's coming next.\"",
        ]),
    ]),
    ("Harshith Kv", "8:30", "1:45", [
        (8, "Status: What's Done, What's Next", "1:00", [
            "To summarize where we are: the full pipeline is built, the dataset is "
            "acquired, quality-checked, and fully preprocessed, both architectures are "
            "implemented and verified, and Experiment 1 is trained, evaluated, and "
            "analyzed.",
            "Still to come for the final review: Experiment 2, the full cross-"
            "manipulation generalization matrix; Experiment 3, compression robustness; "
            "a Grad-CAM interpretability analysis explaining our results, not just "
            "reporting them; and a live working demo.",
        ]),
        (9, "Closing", "0:45", [
            "We're on track, with the harder generalization and robustness questions "
            "-- the actual novel contribution of this project -- still to come.",
            "We look forward to sharing the full comparative results, the "
            "interpretability analysis, and a live demo at the final review.",
            "Thank you -- happy to take questions.",
        ]),
    ]),
]

TOTAL_TIME = "10:15"


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
