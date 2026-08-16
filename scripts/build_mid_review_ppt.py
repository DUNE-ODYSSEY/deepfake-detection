"""Build docs/mid_review_update.pptx -- formal, full-results deck for a
Dean-level review.

Structured in the four-part flow requested: Introduction (what/why we're
solving this, why it matters) -> Problem (the challenge, why it's hard, why
we chose it) -> Solution (our approach, the two architectures, why it's
innovative) -> Results (all three experiments, interpretability, demo).
Every section slide is minimum content, not padding -- 20 slides total.

Deliberately low-color: white background throughout, a single muted navy
accent, no colored status cards -- formal academic register rather than the
brighter navy/cyan startup-deck look used elsewhere in this project's docs.

Usage: python -m scripts.build_mid_review_ppt
"""
from pptx import Presentation
from pptx.util import Emu, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE

SLIDE_W, SLIDE_H = Emu(12191695), Emu(6858000)
MARGIN = Emu(640080)
FONT = "Calibri"

WHITE = RGBColor(0xFF, 0xFF, 0xFF)
INK = RGBColor(0x1A, 0x1A, 0x1A)
ACCENT = RGBColor(0x1F, 0x38, 0x64)      # single formal navy, used sparingly
MUTED = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0xF2, 0xF2, 0xF2)
GRID = RGBColor(0xD9, 0xD9, 0xD9)
BORDER = RGBColor(0xBF, 0xBF, 0xBF)

TOTAL_SLIDES = 20
PAGE = [0]

TEAM = [
    ("Venugopalan Gangadharan", "CB.AI.U4AID25115"),
    ("Vipin Sudhakar", "CB.AI.U4AID25166"),
    ("Rithvik Arulprakash", "CB.AI.U4AID25148"),
    ("Harshith Kv", "CB.AI.U4AID25119"),
]


def new_slide(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    bg = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    bg.fill.solid()
    bg.fill.fore_color.rgb = WHITE
    bg.line.fill.background()
    bg.shadow.inherit = False
    return slide


def textbox(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=None,
            line_spacing=None, space_after=None):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    if anchor:
        tf.vertical_anchor = anchor
    if not runs or not isinstance(runs[0], list):
        runs = [runs]
    for i, line in enumerate(runs):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        if line_spacing:
            p.line_spacing = line_spacing
        if space_after is not None:
            p.space_after = Pt(space_after)
        for text, size, bold, color in line:
            r = p.add_run()
            r.text = text
            r.font.name = FONT
            r.font.size = Pt(size)
            r.font.bold = bold
            r.font.color.rgb = color
    return box


def rect(slide, x, y, w, h, color, line_color=None):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    if line_color:
        sh.line.color.rgb = line_color
        sh.line.width = Pt(0.75)
    else:
        sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def header(slide, eyebrow, title):
    textbox(slide, MARGIN, Emu(384048), Emu(9144000), Emu(320040),
            [(eyebrow, 12.5, True, ACCENT)])
    textbox(slide, MARGIN, Emu(658368), Emu(10852800), Emu(731520),
            [(title, 27, True, INK)])
    rect(slide, MARGIN, Emu(1353312), Emu(10852800), Emu(12700), GRID)


def footer(slide, title="Deepfake Detection: XceptionNet vs. CNN-ViT Hybrid — 23AID205"):
    PAGE[0] += 1
    textbox(slide, MARGIN, Emu(6510528), Emu(8500000), Emu(274320),
            [(title, 9.5, False, MUTED)])
    textbox(slide, Emu(10972800), Emu(6510528), Emu(822960), Emu(274320),
            [(f"{PAGE[0]} / {TOTAL_SLIDES}", 9.5, False, MUTED)],
            align=PP_ALIGN.RIGHT)


def bullets(slide, x, y, w, h, items, size=15, color=INK, bold=False,
            space_after=11, line_spacing=1.14):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = f"–  {item}"
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return box


def stat_panel(slide, x, y, w, h, label, stats):
    rect(slide, x, y, w, h, LIGHT, line_color=GRID)
    pad = Emu(228600)
    textbox(slide, x + pad, y + Emu(180000), w - 2 * pad, Emu(320040),
            [(label, 10.5, True, ACCENT)])
    row_h = Emu(int((h - Emu(500000)) / max(len(stats), 1)))
    cy = y + Emu(500000)
    for num, caption in stats:
        textbox(slide, x + pad, cy, w - 2 * pad, Emu(365760),
                [(num, 21, True, INK)])
        textbox(slide, x + pad, cy + Emu(340000), w - 2 * pad, Emu(320040),
                [(caption, 10, False, MUTED)])
        cy += row_h


def picture_slide(prs, eyebrow, title, caption, image_path, img_w, img_h,
                  stats_label=None, stats=None, img_y=Emu(2286000)):
    slide = new_slide(prs)
    header(slide, eyebrow, title)
    textbox(slide, MARGIN, Emu(1691640), Emu(10852800), Emu(560000),
            [(caption, 13, False, MUTED)])
    slide.shapes.add_picture(image_path, MARGIN, img_y, width=img_w, height=img_h)
    if stats:
        gap = Emu(274320)
        stat_x = MARGIN + img_w + gap
        stat_w = SLIDE_W - MARGIN - stat_x
        stat_panel(slide, stat_x, img_y, stat_w, Emu(4114800), stats_label or "RESULT", stats)
    footer(slide)
    return slide


def two_col(slide, x, y, w, h, header_text, items):
    rect(slide, x, y, w, h, LIGHT, line_color=GRID)
    rect(slide, x, y, w, Emu(460000), ACCENT)
    textbox(slide, x + Emu(228600), y + Emu(100000), w - Emu(457200), Emu(320040),
            [(header_text, 13, True, WHITE)])
    bullets(slide, x + Emu(228600), y + Emu(660000), w - Emu(457200), h - Emu(760000),
            items, size=12, color=INK, space_after=9)


def table_slide(prs, eyebrow, title, subtitle, headers, rows, col_widths, note=None):
    slide = new_slide(prs)
    header(slide, eyebrow, title)
    y = Emu(1691640)
    if subtitle:
        textbox(slide, MARGIN, y, Emu(10515600), Emu(400000),
                [(subtitle, 13, False, MUTED)])
        y += Emu(450000)
    n_rows, n_cols = len(rows) + 1, len(headers)
    tbl_w = sum(col_widths)
    row_h = Emu(420000)
    tbl_h = row_h * n_rows
    gshape = slide.shapes.add_table(n_rows, n_cols, MARGIN, y, tbl_w, tbl_h)
    tbl = gshape.table
    for i, w in enumerate(col_widths):
        tbl.columns[i].width = w
    for c, htext in enumerate(headers):
        cell = tbl.cell(0, c)
        cell.text = htext
        cell.fill.solid()
        cell.fill.fore_color.rgb = ACCENT
        p = cell.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(12.5)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = WHITE
        p.runs[0].font.name = FONT
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = WHITE if r % 2 else LIGHT
            p = cell.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(12)
            p.runs[0].font.bold = (c == 0)
            p.runs[0].font.color.rgb = INK
            p.runs[0].font.name = FONT
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if note:
        rect(slide, MARGIN, y + tbl_h + Emu(220000), Emu(10852800), Emu(12700), GRID)
        textbox(slide, MARGIN, y + tbl_h + Emu(320000), Emu(10852800), Emu(700000),
                [(note, 12, False, MUTED)])
    footer(slide)
    return slide


def content_slide(prs, eyebrow, title, items, size=15.5, subtitle=None):
    slide = new_slide(prs)
    header(slide, eyebrow, title)
    y = Emu(1783080)
    if subtitle:
        textbox(slide, MARGIN, y, Emu(10852800), Emu(457200),
                [(subtitle, 14, False, MUTED)])
        y += Emu(550000)
    bullets(slide, MARGIN, y, Emu(10852800), Emu(6858000) - y - Emu(700000),
            items, size=size, color=INK, space_after=15)
    footer(slide)
    return slide


def emphasis_box(slide, x, y, w, h, label, text):
    rect(slide, x, y, w, h, WHITE, line_color=ACCENT)
    textbox(slide, x + Emu(228600), y + Emu(140000), w - Emu(400000), Emu(320040),
            [(label, 10.5, True, ACCENT)])
    textbox(slide, x + Emu(228600), y + Emu(460000), w - Emu(400000), h - Emu(600000),
            [(text, 12.5, False, INK)])


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

    # ---------- 1. Title ----------
    s = new_slide(prs)
    rect(s, 0, Emu(3100000), SLIDE_W, Emu(25400), ACCENT)
    textbox(s, MARGIN, Emu(1600000), Emu(10852800), Emu(400000),
            [("AIML COURSE PROJECT  ·  23AID205", 14, True, ACCENT)], align=PP_ALIGN.CENTER)
    textbox(s, MARGIN, Emu(2050000), Emu(10852800), Emu(950000),
            [[("Deepfake Detection", 36, True, INK)]], align=PP_ALIGN.CENTER)
    textbox(s, MARGIN, Emu(2800000), Emu(10852800), Emu(400000),
            [("A Comparative Study of XceptionNet and a CNN-ViT Hybrid on FaceForensics++",
              16, False, MUTED)], align=PP_ALIGN.CENTER)
    textbox(s, MARGIN, Emu(3450000), Emu(10852800), Emu(360000),
            [("Cross-Manipulation Generalization and Compression Robustness", 13, False, MUTED)],
            align=PP_ALIGN.CENTER)
    textbox(s, MARGIN, Emu(5600000), Emu(10852800), Emu(320040),
            [(" | ".join(name for name, _ in TEAM), 12.5, True, INK)], align=PP_ALIGN.CENTER)
    textbox(s, MARGIN, Emu(5950000), Emu(10852800), Emu(320040),
            [("Department of Artificial Intelligence & Data Science", 11, False, MUTED)],
            align=PP_ALIGN.CENTER)
    textbox(s, MARGIN, Emu(6220000), Emu(10852800), Emu(320040),
            [("Amrita Vishwa Vidyapeetham, Coimbatore", 11, False, MUTED)],
            align=PP_ALIGN.CENTER)
    PAGE[0] += 1

    # ---------- 2. Agenda ----------
    s = new_slide(prs)
    header(s, "OVERVIEW", "Presentation Roadmap")
    roadmap = [
        ("01", "Introduction", "What we are solving, and why it is necessary"),
        ("02", "Problem", "The challenge, its difficulty, and why we chose it"),
        ("03", "Solution", "Our approach, the two architectures, and its innovation"),
        ("04", "Results", "Findings across three experiments and their implications"),
    ]
    ry = Emu(2000000)
    for num, title, desc in roadmap:
        rect(s, MARGIN, ry, Emu(10852800), Emu(950000), LIGHT, line_color=GRID)
        textbox(s, MARGIN + Emu(200000), ry + Emu(150000), Emu(900000), Emu(650000),
                [(num, 26, True, ACCENT)])
        textbox(s, MARGIN + Emu(1250000), ry + Emu(160000), Emu(3200000), Emu(400000),
                [(title, 16, True, INK)])
        textbox(s, MARGIN + Emu(4600000), ry + Emu(220000), Emu(6000000), Emu(500000),
                [(desc, 12.5, False, MUTED)])
        ry += Emu(1080000)
    footer(s)

    # ================= INTRODUCTION =================

    # 3. Introduction - What We Are Solving
    content_slide(prs, "1. INTRODUCTION", "What We Are Solving", [
        "This project addresses automated deepfake detection: given a face video, "
        "determine whether it has been digitally manipulated or is authentic.",
        "Specifically, we conduct a controlled comparative study of two detector "
        "architectures — a convolutional baseline (XceptionNet) and a CNN-Transformer "
        "hybrid — on the FaceForensics++ benchmark.",
        "The study is deliberately not limited to reporting accuracy. It evaluates "
        "whether a detector's performance survives conditions it was not trained on: "
        "unseen manipulation techniques and unseen compression levels — the "
        "conditions any real-world deployment will actually face.",
        "The outcome is both an empirical answer (which architecture generalizes "
        "better, and by how much) and an interpretive one (why), supported by a "
        "working software demonstration.",
    ], size=16)

    # 4. Introduction - Why It Is Necessary
    content_slide(prs, "1. INTRODUCTION", "Why It Is Necessary", [
        "Deepfake technology has moved from a research curiosity to an active "
        "misinformation and fraud vector — fabricated video evidence, non-consensual "
        "synthetic media, and identity-based fraud are documented, growing harms.",
        "Generation tools have become significantly more accessible (consumer "
        "applications, open-source models) even as detection research has not kept "
        "pace at the same rate — the gap between what can be generated and what can "
        "be reliably detected is widening.",
        "Detectors that perform well only under laboratory conditions provide false "
        "assurance: a system reporting 99% accuracy in a controlled benchmark can "
        "fail silently the moment it encounters real-world content, and that failure "
        "is invisible until it is too late to matter.",
        "Establishing exactly how much accuracy is lost under realistic distribution "
        "shift — and why — is therefore not an academic refinement; it is a "
        "precondition for responsibly deploying any such system.",
    ], size=15.5)

    # ================= PROBLEM =================

    # 5. Problem Statement
    content_slide(prs, "2. PROBLEM", "Problem Statement", [
        "Formally: given a face video, produce a binary decision (authentic / "
        "manipulated) that remains reliable not only under the manipulation method "
        "and compression level seen during training, but also under others "
        "encountered after deployment.",
        "The great majority of published detectors are evaluated under matched "
        "train/test conditions — trained and tested on the same manipulation method "
        "and the same compression level. This produces an optimistic, non-"
        "representative measure of real-world performance.",
        "This project treats generalization itself — not same-distribution accuracy "
        "— as the primary object of study, and measures it directly rather than "
        "assuming it.",
    ], size=16)

    # 6. Problem - Why It Is Hard
    content_slide(prs, "2. PROBLEM", "Why This Problem Is Difficult", [
        "Different manipulation techniques leave fundamentally different artifacts: "
        "autoencoder-based face-swapping, landmark-driven reenactment, graphics-"
        "based swapping, and neural-texture rendering each produce distinct, "
        "technique-specific low-level signatures rather than one shared \"fake\" "
        "signature.",
        "A detector trained to recognize one signature has no inherent guarantee of "
        "recognizing another — generalization across manipulation types is an open "
        "problem, not a solved one, as this study's own results later confirm.",
        "Video compression, which is universal in real-world distribution (social "
        "media, messaging platforms), further degrades or removes the very "
        "artifacts a detector was trained to rely on.",
        "The problem is also adversarial by nature: as detection methods improve, "
        "generation methods adapt in response, making this a continuously moving "
        "target rather than a fixed classification task.",
    ], size=15.5)

    # 7. Problem - Why We Chose to Solve It
    content_slide(prs, "2. PROBLEM", "Why We Chose to Address This Problem", [
        "The FaceForensics++ literature has established strong same-distribution "
        "benchmarks, but rigorous, head-to-head architecture comparisons under "
        "cross-manipulation and cross-compression conditions — with an "
        "interpretability layer explaining the results — remain comparatively "
        "under-studied.",
        "This creates a well-scoped, testable, and practically important research "
        "question: does a more modern, attention-based architecture close the "
        "generalization gap that plain convolutional detectors exhibit, or does it "
        "not?",
        "Answering this question with a controlled experiment — identical data, "
        "identical training protocol, architecture as the only variable — yields a "
        "result that is directly useful to anyone selecting a detector architecture "
        "for real deployment, not only of academic interest.",
    ], size=16)

    # ================= SOLUTION =================

    # 8. Solution - Approach & Pipeline
    s = new_slide(prs)
    header(s, "3. SOLUTION", "Our Approach")
    textbox(s, MARGIN, Emu(1691640), Emu(10852800), Emu(500000),
            [("A controlled, six-stage experimental pipeline, applied identically to "
              "both architectures.", 13.5, False, MUTED)])
    steps = [
        ("1", "Raw Video", "FaceForensics++ c23: real footage + 4 manipulation methods"),
        ("2", "Face Extraction", "MTCNN face detection, 20 frames/video, 1.3x crop margin"),
        ("3", "Official Splits", "Video-level train/val/test — prevents identity leakage"),
        ("4", "Training", "Identical optimizer, schedule, and batch size for both models"),
        ("5", "Evaluation", "Frame- and video-level Accuracy / F1 / AUC on held-out data"),
        ("6", "Analysis", "Generalization matrix, interpretability, live demonstration"),
    ]
    x = MARGIN
    card_w = Emu(1750000)
    gap = Emu(58000)
    for i, (n, t, d) in enumerate(steps):
        cx = x + i * (card_w + gap)
        rect(s, cx, Emu(2350000), card_w, Emu(3300000), LIGHT, line_color=GRID)
        rect(s, cx, Emu(2350000), card_w, Emu(430000), ACCENT)
        textbox(s, cx + Emu(120000), Emu(2440000), card_w - Emu(240000), Emu(280000),
                [(n, 13, True, WHITE)])
        textbox(s, cx + Emu(120000), Emu(2870000), card_w - Emu(240000), Emu(700000),
                [(t, 12.5, True, INK)])
        textbox(s, cx + Emu(120000), Emu(3480000), card_w - Emu(240000), Emu(2050000),
                [(d, 10, False, MUTED)])
    footer(s)

    # 9. Solution - Dataset & Preprocessing
    picture_slide(prs, "3. SOLUTION", "Dataset & Preprocessing",
                 "FaceForensics++: 5,000 videos (1,000 authentic + 4×1,000 manipulated "
                 "across Deepfakes, Face2Face, FaceSwap, NeuralTextures), official "
                 "video-level splits, MTCNN preprocessing to ~99,987 face crops. Data "
                 "quality verified via IQR outlier analysis prior to preprocessing.",
                 "analyze/outputs/outlier_boxplots.png", Emu(7000000), Emu(3450000),
                 stats_label="DATA QUALITY",
                 stats=[("5,000", "videos analyzed (real + 4 methods)"),
                        ("502", "outliers flagged and excluded (10.0%)"),
                        ("99,987", "face crops in the final training set")],
                 img_y=Emu(2500000))

    # 10. Solution - Architecture I
    content_slide(prs, "3. SOLUTION", "Architecture I — XceptionNet (Baseline)", [
        "A convolutional network built from depthwise-separable convolutions "
        "(Chollet, 2017), ImageNet-pretrained and widely used as the reference "
        "detector in the face-forgery-detection literature.",
        "Produces a 2048-dimensional pooled feature vector, followed by a dropout "
        "and linear classification layer.",
        "Structurally local: every layer operates on a bounded spatial neighborhood; "
        "there is no mechanism for directly relating two distant regions of the "
        "frame in a single step.",
        "Selected specifically because it is the architecture most cross-"
        "manipulation generalization claims in prior work are benchmarked against — "
        "using it here keeps our results directly comparable to that literature.",
    ], size=15.5)

    # 11. Solution - Architecture II
    content_slide(prs, "3. SOLUTION", "Architecture II — CNN-ViT Hybrid", [
        "Combines a ResNet-50 convolutional feature extractor (stages 1–3) with a "
        "6-layer, 8-head Transformer encoder operating on the resulting 14×14 "
        "feature grid, tokenized to 196 patches plus a classification token.",
        "The central hypothesis under test: self-attention allows every region of "
        "the frame to be related to every other region directly, which may allow "
        "the model to detect global inconsistencies — such as a mismatched lighting "
        "or blending boundary — that a convolutional network can only approximate "
        "indirectly through depth.",
        "Trained under exactly the same optimizer, learning-rate schedule, batch "
        "size, and data as the baseline — architecture is the only variable, by "
        "design, so any difference in outcome is attributable to it alone.",
    ], size=15.5)

    # 12. Solution - Why Innovative
    content_slide(prs, "3. SOLUTION", "Why This Approach Is Innovative", [
        "A controlled, head-to-head comparison of a convolutional and an attention-"
        "based architecture, evaluated specifically for generalization rather than "
        "same-distribution accuracy alone — a comparison largely absent from prior "
        "published work on this dataset.",
        "A complete 4×4 cross-manipulation evaluation matrix per architecture, "
        "rather than the single train/test condition most published results report.",
        "Transparent handling of a real data-access constraint: with official "
        "high-compression data unavailable, a locally re-compressed proxy test set "
        "was constructed and clearly documented as an approximation, rather than the "
        "experiment being omitted.",
        "An interpretability layer (Grad-CAM) that explains why generalization "
        "succeeds or fails, rather than reporting only that it does — turning a "
        "numerical result into a mechanistic explanation.",
        "Delivered as a working, testable software artifact — a live prediction "
        "interface — rather than as offline metrics alone.",
    ], size=14.5)

    # ================= RESULTS =================

    # 13. Results - Experiment 1
    table_slide(prs, "4. RESULTS", "Experiment 1 — Baseline Comparison",
               "Both architectures trained and tested on the same distribution — all "
               "four manipulation methods, c23 compression.",
               ["Model", "Video Acc.", "Video F1", "Video AUC", "Frame Acc.", "Frame AUC"],
               [["XceptionNet", "97.29%", "0.983", "0.9958", "95.49%", "0.986"],
                ["CNN-ViT Hybrid", "94.71%", "0.967", "0.9836", "92.76%", "0.962"]],
               [Emu(2400000), Emu(1700000), Emu(1700000), Emu(1700000), Emu(1700000), Emu(1700000)],
               note="XceptionNet leads on every same-distribution metric measured. This "
                    "reflects performance under the most favorable possible test "
                    "condition; the following experiments assess whether it persists "
                    "under distribution shift.")

    # 14. Results - Experiment 2
    s = new_slide(prs)
    header(s, "4. RESULTS", "Experiment 2 — Cross-Manipulation Generalization")
    textbox(s, MARGIN, Emu(1691640), Emu(10852800), Emu(420000),
            [("Each model trained on one manipulation method and evaluated on all "
              "four — the diagonal is same-manipulation; the off-diagonal is the "
              "actual generalization test.", 13, False, MUTED)])
    s.shapes.add_picture("analysis/cross_manipulation_heatmaps.png", MARGIN, Emu(2180000),
                        width=Emu(7440000), height=Emu(3000000))
    emphasis_box(s, MARGIN, Emu(5320000), Emu(10852800), Emu(1150000), "PRINCIPAL FINDING",
                "The hypothesis that attention improves generalization does not hold: "
                "XceptionNet records both a higher same-distribution AUC and a smaller "
                "generalization gap (0.398) than the CNN-ViT hybrid (0.421). Both "
                "architectures fall to worse-than-random on their most difficult cell "
                "(Deepfakes → FaceSwap: 0.256 / 0.198 AUC).")
    footer(s)

    # 15. Results - Experiment 3
    table_slide(prs, "4. RESULTS", "Experiment 3 — Compression Robustness",
               "Official high-compression (c40) data access was pending; a locally "
               "re-compressed proxy test set was used instead, evaluation-only, "
               "clearly documented as an approximation (see PROJECT_LOG.md).",
               ["Model", "c23 AUC", "Proxy c40 AUC", "Change"],
               [["XceptionNet", "0.9958", "0.8018", "– 0.194"],
                ["CNN-ViT Hybrid", "0.9836", "0.8172", "– 0.166"]],
               [Emu(2600000), Emu(2600000), Emu(2600000), Emu(2600000)],
               note="Both models degrade substantially under heavier compression; the "
                    "hybrid's decline is marginally smaller, the one condition under "
                    "which it edges ahead of the baseline.")

    # 16. Results - Interpretability
    picture_slide(prs, "4. RESULTS", "Interpretability — Grad-CAM Analysis",
                 "Same video, same Deepfakes-only checkpoint, two manipulation methods. "
                 "Top: same-manipulation (correct). Bottom: cross-manipulation to "
                 "FaceSwap, the weakest cell (correct becomes confidently incorrect).",
                 "analyze/outputs/gradcam_comparison.png", Emu(5868000), Emu(4200000),
                 stats_label="DF → FS (WORST CELL)",
                 stats=[("0.256 / 0.198", "cross-manipulation AUC — worse than random"),
                        ("1.000 → 0.000", "XceptionNet fake-probability collapse"),
                        ("face → background", "where model attention relocates")])

    # 17. Results - Live Demo
    content_slide(prs, "4. RESULTS", "Live Demonstration", [
        "A working prediction interface (FastAPI backend, browser front end): a "
        "user uploads an image or short video and receives a real/fake probability "
        "from either trained model, using the identical preprocessing pipeline "
        "applied during training and evaluation.",
        "Verified against a held-out set of test-split videos never seen during "
        "training — correctly and confidently classified by both models, "
        "XceptionNet with visibly higher confidence, consistent with its measured "
        "baseline advantage.",
        "Also tested informally against an out-of-distribution image circulating "
        "online, outside FaceForensics++ entirely: both models were confidently "
        "incorrect — a live demonstration of the generalization limitation "
        "measured formally in Experiment 2, not a contradiction of it.",
    ], size=15.5)

    # ---------- 18. Conclusion ----------
    content_slide(prs, "CONCLUSION", "Key Takeaways", [
        "XceptionNet outperformed the CNN-ViT hybrid on every metric measured across "
        "all three experiments — same-distribution accuracy, cross-manipulation "
        "generalization, and, marginally, compression robustness — contradicting the "
        "initial hypothesis that self-attention would improve generalization on this "
        "task.",
        "Both architectures share the same underlying limitation: each learns a "
        "narrow, manipulation-method-specific signature rather than a general "
        "indicator of facial manipulation, a conclusion supported both numerically "
        "(the generalization-gap results) and visually (the Grad-CAM analysis).",
        "Practically, this indicates that a face-forgery detector is only as "
        "reliable as the diversity of manipulation types represented in its "
        "training data — deployment beyond that coverage requires either broader "
        "training data or an explicit \"out-of-distribution / uncertain\" decision "
        "path, since a confident incorrect answer is materially worse than an "
        "honest uncertain one.",
    ], size=15.5)

    # ---------- 19. Future Work ----------
    content_slide(prs, "CONCLUSION", "Future Work", [
        "Repeat Experiment 3 against official raw (c0) and high-compression (c40) "
        "data once FaceForensics++ access is approved, replacing the current "
        "documented proxy with the genuine benchmark condition.",
        "Repeat training across multiple random seeds to establish statistical "
        "confidence intervals around the reported metrics.",
        "Extend the Grad-CAM analysis to a cross-manipulation cell with "
        "near-random (0.5 AUC) rather than worse-than-random performance, to "
        "examine whether genuine model uncertainty produces a different attention "
        "pattern than confident misclassification.",
        "Evaluate additional manipulation techniques beyond FaceForensics++'s "
        "original four, to test whether the observed generalization gap extends to "
        "more recent, diffusion-based generation methods.",
    ], size=15.5)

    # ---------- 20. References & Contributors ----------
    s = new_slide(prs)
    header(s, "CLOSING", "References & Contributors")
    textbox(s, MARGIN, Emu(1750000), Emu(10852800), Emu(320040),
            [("REFERENCES", 11, True, ACCENT)])
    bullets(s, MARGIN, Emu(2100000), Emu(10852800), Emu(1650000), [
        "Rössler, A. et al. (2019). FaceForensics++: Learning to Detect "
        "Manipulated Facial Images. ICCV.",
        "Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable "
        "Convolutions. CVPR.",
        "Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: "
        "Transformers for Image Recognition at Scale. ICLR.",
        "Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep "
        "Networks via Gradient-based Localization. ICCV.",
    ], size=12, space_after=8)
    rect(s, MARGIN, Emu(3900000), Emu(10852800), Emu(12700), GRID)
    textbox(s, MARGIN, Emu(4080000), Emu(10852800), Emu(320040),
            [("CONTRIBUTORS", 11, True, ACCENT)])
    bullets(s, MARGIN, Emu(4420000), Emu(10852800), Emu(1080000),
            [f"{name} — {roll}" for name, roll in TEAM],
            size=12, color=INK, bold=True, space_after=6)
    textbox(s, MARGIN, Emu(5580000), Emu(10852800), Emu(320040),
            [("Department of Artificial Intelligence & Data Science, Amrita Vishwa "
              "Vidyapeetham, Coimbatore", 11, False, MUTED)])
    textbox(s, MARGIN, Emu(5930000), Emu(10852800), Emu(320040),
            [("REPOSITORY", 11, True, ACCENT)])
    textbox(s, MARGIN, Emu(6180000), Emu(10000000), Emu(330000),
            [("github.com/DUNE-ODYSSEY/deepfake-detection", 12, True, INK)])
    footer(s)

    out = "docs/mid_review_update.pptx"
    prs.save(out)
    print(f"saved {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
