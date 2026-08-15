"""Build docs/mid_review_update.pptx from the (now complete) project results.

Follows the course guideline's required content list (23AID205 project
guideline: introduction/motivation, problem statement, objectives, dataset,
preprocessing, EDA, algorithm selection, training/validation methodology,
evaluation metrics, comparison of approaches, software implementation,
results & discussion, conclusion & future work, references, contributors),
adapted into slide form. Keeps the visual language of the original deck
(navy title slide, white content slides, cyan accent, Calibri) so this
reads as one continuous deck rather than a restyle.

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

NAVY = RGBColor(0x0F, 0x17, 0x2A)
CYAN = RGBColor(0x22, 0xB8, 0xCF)
TEAL = RGBColor(0x0E, 0x74, 0x90)
TEXT = RGBColor(0x1A, 0x20, 0x2C)
MUTED = RGBColor(0x47, 0x55, 0x69)
MUTED_L = RGBColor(0x94, 0xA3, 0xB8)
CARD_BG = RGBColor(0xF1, 0xF5, 0xF9)
GRID = RGBColor(0xE2, 0xE8, 0xF0)
GREEN = RGBColor(0x16, 0xA3, 0x4A)
AMBER = RGBColor(0xF5, 0x9E, 0x0B)
RED = RGBColor(0xDC, 0x26, 0x26)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

TOTAL_SLIDES = 11
PAGE = [0]


def new_slide(prs, bg=WHITE):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    rect = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, 0, 0, SLIDE_W, SLIDE_H)
    rect.fill.solid()
    rect.fill.fore_color.rgb = bg
    rect.line.fill.background()
    rect.shadow.inherit = False
    return slide


def textbox(slide, x, y, w, h, runs, align=PP_ALIGN.LEFT, anchor=None,
            line_spacing=None, space_after=None):
    """runs: list of (text, size_pt, bold, color) OR list of lines, each a
    list of such tuples (for multi-paragraph text boxes)."""
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


def rect(slide, x, y, w, h, color):
    sh = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE, x, y, w, h)
    sh.fill.solid()
    sh.fill.fore_color.rgb = color
    sh.line.fill.background()
    sh.shadow.inherit = False
    return sh


def header(slide, eyebrow, title, dark=False):
    eyebrow_color = CYAN if dark else TEAL
    title_color = WHITE if dark else TEXT
    textbox(slide, MARGIN, Emu(384048), Emu(9144000), Emu(320040),
            [(eyebrow, 13, True, eyebrow_color)])
    textbox(slide, MARGIN, Emu(658368), Emu(10852800), Emu(731520),
            [(title, 28, True, title_color)])
    rect(slide, MARGIN, Emu(1353312), Emu(502920), Emu(41148), CYAN)


def footer(slide, title="Deepfake Detection — XceptionNet vs. CNN-ViT (23AID205)"):
    PAGE[0] += 1
    textbox(slide, MARGIN, Emu(6510528), Emu(7315200), Emu(274320),
            [(title, 10, False, MUTED_L)])
    textbox(slide, Emu(10972800), Emu(6510528), Emu(822960), Emu(274320),
            [(f"{PAGE[0]}/{TOTAL_SLIDES}", 10, False, MUTED_L)],
            align=PP_ALIGN.RIGHT)


def bullets(slide, x, y, w, h, items, size=15, color=TEXT, bold=False,
            bullet_color=None, space_after=10, line_spacing=1.12):
    box = slide.shapes.add_textbox(x, y, w, h)
    tf = box.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(space_after)
        p.line_spacing = line_spacing
        r = p.add_run()
        r.text = f"▸  {item}"
        r.font.name = FONT
        r.font.size = Pt(size)
        r.font.bold = bold
        r.font.color.rgb = color
    return box


def stat_panel(slide, x, y, w, h, label, stats):
    rect(slide, x, y, w, h, CARD_BG)
    pad = Emu(228600)
    textbox(slide, x + pad, y + Emu(180000), w - 2 * pad, Emu(320040),
            [(label, 11, True, TEAL)])
    row_h = Emu(int((h - Emu(500000)) / max(len(stats), 1)))
    cy = y + Emu(500000)
    for num, caption in stats:
        textbox(slide, x + pad, cy, w - 2 * pad, Emu(365760),
                [(num, 22, True, NAVY)])
        textbox(slide, x + pad, cy + Emu(340000), w - 2 * pad, Emu(320040),
                [(caption, 10.5, False, MUTED)])
        cy += row_h


def picture_slide(prs, eyebrow, title, caption, image_path, img_w, img_h,
                  stats_label=None, stats=None):
    slide = new_slide(prs)
    header(slide, eyebrow, title)
    textbox(slide, MARGIN, Emu(1691640), Emu(10515600), Emu(457200),
            [(caption, 13.5, False, MUTED)])
    slide.shapes.add_picture(image_path, MARGIN, Emu(2286000), width=img_w, height=img_h)
    if stats:
        gap = Emu(274320)
        stat_x = MARGIN + img_w + gap
        stat_w = SLIDE_W - MARGIN - stat_x
        stat_panel(slide, stat_x, Emu(2286000), stat_w, Emu(4114800),
                  stats_label or "RESULT", stats)
    footer(slide)
    return slide


def card(slide, x, y, w, h, header_text, header_color, items, item_color):
    rect(slide, x, y, w, h, CARD_BG)
    rect(slide, x, y, w, Emu(502920), header_color)
    textbox(slide, x + Emu(274320), y + Emu(118872), w - Emu(548640), Emu(320040),
            [(header_text, 14, True, WHITE)])
    bullets(slide, x + Emu(274320), y + Emu(731520), w - Emu(548640), h - Emu(800000),
            items, size=12.5, color=item_color, bold=True, space_after=8)


def table_slide(prs, eyebrow, title, subtitle, headers, rows, col_widths,
                highlight_rows=None, note=None):
    slide = new_slide(prs)
    header(slide, eyebrow, title)
    y = Emu(1691640)
    if subtitle:
        textbox(slide, MARGIN, y, Emu(10515600), Emu(400000),
                [(subtitle, 13.5, False, MUTED)])
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
        cell.fill.fore_color.rgb = NAVY
        p = cell.text_frame.paragraphs[0]
        p.runs[0].font.size = Pt(13)
        p.runs[0].font.bold = True
        p.runs[0].font.color.rgb = WHITE
        p.runs[0].font.name = FONT
        cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    highlight_rows = highlight_rows or set()
    for r, row in enumerate(rows, start=1):
        for c, val in enumerate(row):
            cell = tbl.cell(r, c)
            cell.text = str(val)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(0xFE, 0xF3, 0xE7) if r in highlight_rows else (WHITE if r % 2 else CARD_BG)
            p = cell.text_frame.paragraphs[0]
            p.runs[0].font.size = Pt(12.5)
            p.runs[0].font.bold = (c == 0)
            p.runs[0].font.color.rgb = TEXT
            p.runs[0].font.name = FONT
            cell.vertical_anchor = MSO_ANCHOR.MIDDLE
    if note:
        textbox(slide, MARGIN, y + tbl_h + Emu(180000), Emu(10852800), Emu(600000),
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
            items, size=size, color=TEXT, space_after=14)
    footer(slide)
    return slide


def callout(slide, x, y, w, h, label, text, color):
    rect(slide, x, y, w, h, RGBColor(0xFE, 0xF3, 0xE7) if color == AMBER else RGBColor(0xFE, 0xEC, 0xEC))
    rect(slide, x, y, Emu(60000), h, color)
    textbox(slide, x + Emu(228600), y + Emu(120000), w - Emu(400000), Emu(320040),
            [(label, 11, True, color)])
    textbox(slide, x + Emu(228600), y + Emu(430000), w - Emu(400000), h - Emu(550000),
            [(text, 13, False, TEXT)])


def main():
    prs = Presentation()
    prs.slide_width, prs.slide_height = SLIDE_W, SLIDE_H

    # 1. Title
    s = new_slide(prs, bg=NAVY)
    rect(s, 0, 0, Emu(146304), SLIDE_H, CYAN)
    textbox(s, MARGIN, Emu(2148840), Emu(8229600), Emu(365760),
            [("AIML COURSE PROJECT · 23AID205 · MID-REVIEW", 15, True, CYAN)])
    textbox(s, MARGIN, Emu(2468880), Emu(10332720), Emu(1920240),
            [[("Deepfake Detection:", 34, True, WHITE)],
             [("XceptionNet vs. CNN-ViT Hybrid on FaceForensics++", 34, True, WHITE)]])
    textbox(s, MARGIN, Emu(4343400), Emu(9601200), Emu(640000),
            [("Cross-manipulation generalization and compression robustness — "
              "full pipeline run, both models trained, results analyzed, "
              "and a working live demo.", 14, False, MUTED_L)])
    rect(s, MARGIN, Emu(5074920), Emu(457200), Emu(27432), CYAN)
    textbox(s, MARGIN, Emu(5989320), Emu(7315200), Emu(320040),
            [("Venugopalan Gangadharan", 13, True, RGBColor(0xE5, 0xE9, 0xF0))])
    textbox(s, MARGIN, Emu(6263640), Emu(7315200), Emu(274320),
            [("AI & Data Science — Amrita Vishwa Vidyapeetham, Coimbatore", 11, False, MUTED_L)])
    PAGE[0] += 1  # title slide counts toward the total but has no footer text

    # 2. Motivation, Problem & Objectives (combined)
    content_slide(prs, "INTRODUCTION", "Motivation, Problem & Objectives", [
        "Deepfakes have moved from research curiosity to a real misinformation/fraud vector — "
        "but published detectors usually report one headline accuracy number, on the same "
        "manipulation type and compression level they trained on.",
        "Problem: that number is misleading in practice — a detector at 99% in a paper can be "
        "~57% AUC (worse than random) the moment the test distribution shifts, and that failure "
        "is invisible until deployment.",
        "Objective 1: fairly compare XceptionNet (CNN baseline) vs. a CNN-ViT hybrid "
        "(ResNet + Transformer) on identical data/protocol — does global self-attention "
        "actually improve generalization?",
        "Objective 2: measure same-distribution accuracy, full 4×4 cross-manipulation "
        "generalization, and compression robustness — not just one number.",
        "Objective 3: explain *why* generalization succeeds or fails (Grad-CAM), and ship a "
        "working, testable live demo — not only offline metrics.",
    ], size=15)

    # 3. Methodology overview (pipeline)
    s = new_slide(prs)
    header(s, "METHODOLOGY", "End-to-End Pipeline")
    steps = [
        ("1", "Raw video", "FF++ c23, 5 folders: real + 4 manipulation methods"),
        ("2", "Face extraction", "MTCNN crop, 20 frames/video, margin 1.3x"),
        ("3", "Official splits", "video-level train/val/test — no identity leakage"),
        ("4", "Train", "AMP, class-balanced sampler, cosine LR, early stop"),
        ("5", "Evaluate", "frame + video-level Acc/F1/AUC on held-out test"),
        ("6", "Analyze", "cross-manip matrix, gap, Grad-CAM, live demo"),
    ]
    x = MARGIN
    card_w = Emu(1750000)
    gap = Emu(58000)
    for i, (n, t, d) in enumerate(steps):
        cx = x + i * (card_w + gap)
        rect(s, cx, Emu(2200000), card_w, Emu(3400000), CARD_BG)
        rect(s, cx, Emu(2200000), card_w, Emu(500000), NAVY)
        textbox(s, cx + Emu(120000), Emu(2320000), card_w - Emu(240000), Emu(320000),
                [(n, 15, True, CYAN)])
        textbox(s, cx + Emu(120000), Emu(2800000), card_w - Emu(240000), Emu(700000),
                [(t, 13.5, True, TEXT)])
        textbox(s, cx + Emu(120000), Emu(3450000), card_w - Emu(240000), Emu(1900000),
                [(d, 10.5, False, MUTED)])
    footer(s)

    # 4. Dataset, Preprocessing & EDA (combined)
    content_slide(prs, "DATASET & PREPROCESSING", "Data: Source, Prep & Quality Check", [
        "FaceForensics++ (FF++): 5,000 videos — 1,000 real (YouTube interviews) + 4×1,000 "
        "manipulated (Deepfakes, Face2Face, FaceSwap, NeuralTextures), all sharing the same "
        "real-video identities. c23 compression, official FF++ folder layout (Kaggle mirror).",
        "Official video-level train/val/test splits used throughout — critical, since random "
        "frame-level splits would leak identity between train/test and inflate accuracy.",
        "Preprocessing: MTCNN face detection on GPU, 20 frames/video, 1.3x crop margin "
        "(captures blending-boundary artifacts), resized to 299×299 (native for XceptionNet, "
        "downsampled to 224 for the hybrid) → ~99,987 face crops total.",
        "EDA/data-quality check: IQR outlier analysis on frame count & file size flagged 502 of "
        "5,000 videos (10%) before committing to full preprocessing.",
        "Real/fake class imbalance handled via a weighted random sampler at training time, not "
        "by discarding data. Official c0/c40 access still pending — see Compression slide.",
    ], size=14)

    # 5. Model architectures (comparison cards) + shared training/eval protocol
    s = new_slide(prs)
    header(s, "ALGORITHM SELECTION", "Models: XceptionNet vs. CNN-ViT Hybrid")
    card_y, card_h = Emu(1750000), Emu(3000000)
    card_w = Emu(5195000)
    card(s, MARGIN, card_y, card_w, card_h, "XCEPTIONNET (BASELINE)", NAVY, [
        "Depthwise-separable CNN (Chollet 2017), ImageNet-pretrained",
        "2048-d pooled features → dropout → linear → logit",
        "Purely convolutional — local receptive fields only",
        "The standard reference detector for this exact task",
    ], TEAL)
    card(s, MARGIN + card_w + Emu(228600), card_y, card_w, card_h, "CNN-VIT HYBRID", TEAL, [
        "ResNet-50 (stages 1-3) → 14×14×1024 map → 512-d tokens + CLS",
        "6-layer, 8-head Transformer encoder over the tokens",
        "Tests: does global self-attention improve generalization?",
        "Same optimizer/schedule/data — architecture is the only variable",
    ], NAVY)
    strip_y = card_y + card_h + Emu(228600)
    rect(s, MARGIN, strip_y, Emu(10852800), Emu(18288), GRID)
    textbox(s, MARGIN, strip_y + Emu(180000), Emu(10852800), Emu(320040),
            [("SHARED TRAINING & EVALUATION PROTOCOL (fair comparison — only the architecture differs)", 11, True, TEAL)])
    textbox(s, MARGIN, strip_y + Emu(560000), Emu(10852800), Emu(900000),
            [("AMP training, AdamW, cosine LR, batch size 32 (GPU-benchmarked — XceptionNet "
              "silently throttles above this), early stopping (patience 4 on val AUC). "
              "Metrics: Accuracy/F1/AUC at frame & video level (video = mean of frame "
              "probabilities). 10 training runs, 36 evaluations total.", 13, False, MUTED)])
    footer(s)

    # 6. Results: baseline comparison
    table_slide(prs, "RESULTS · EXPERIMENT 1", "Baseline Comparison (same-distribution, c23)",
               "Both models trained and tested on all 4 methods — the headline accuracy number "
               "most papers stop at.",
               ["Model", "Video Acc", "Video F1", "Video AUC", "Frame Acc", "Frame AUC"],
               [["XceptionNet", "97.29%", "0.983", "0.9958", "95.49%", "0.986"],
                ["CNN-ViT Hybrid", "94.71%", "0.967", "0.9836", "92.76%", "0.962"]],
               [Emu(2400000), Emu(1700000), Emu(1700000), Emu(1700000), Emu(1700000), Emu(1700000)],
               note="XceptionNet leads on every same-distribution metric — the first sign the "
                    "hybrid's extra complexity isn't paying off on this task.")

    # 7. Results: cross-manipulation generalization (heatmap + finding, combined)
    s = new_slide(prs)
    header(s, "RESULTS · EXPERIMENT 2", "Cross-Manipulation Generalization")
    textbox(s, MARGIN, Emu(1691640), Emu(10852800), Emu(400000),
            [("Train on one manipulation method, test on all four (4×4 matrix per model). "
              "Diagonal = same-manip; off-diagonal = the actual generalization test.",
              13, False, MUTED)])
    s.shapes.add_picture("analysis/cross_manipulation_heatmaps.png", MARGIN, Emu(2180000),
                        width=Emu(7440000), height=Emu(3000000))
    callout(s, MARGIN, Emu(5320000), Emu(10852800), Emu(1150000), "HEADLINE FINDING",
           "Hypothesis (“attention generalizes better”) did NOT hold: XceptionNet has both "
           "higher same-distribution AND cross-manip AUC (generalization gap 0.398 vs. "
           "0.421) than the hybrid. Worst cell for both, DF→FS: 0.256 / 0.198 AUC — "
           "worse than random.", RED)
    footer(s)

    # 8. Results: compression robustness
    s = table_slide(prs, "RESULTS · EXPERIMENT 3", "Compression Robustness (Proxy-c40)",
                    "Official c0/c40 FaceForensics++ access was still pending, and no c40 mirror "
                    "exists on Kaggle. Worked around it rather than skipping the experiment.",
                    ["Model", "c23 AUC", "c40proxy AUC", "Drop"],
                    [["XceptionNet", "0.9958", "0.8018", "-0.194"],
                     ["CNN-ViT Hybrid", "0.9836", "0.8172", "-0.166"]],
                    [Emu(2600000), Emu(2600000), Emu(2600000), Emu(2600000)])
    callout(s, MARGIN, Emu(4200000), Emu(10852800), Emu(1500000), "METHOD (CAVEATED)",
           "“c40proxy” = the held-out test videos re-transcoded locally, c23→crf40 (ffmpeg). "
           "This double-compresses rather than compressing once from raw like official c40, so "
           "degradation is harsher/different from the real thing — a documented approximation, "
           "used for evaluation only, never for training. Will re-run against official c0/c40 if "
           "FaceForensics++ approval arrives.", AMBER)

    # 9. Grad-CAM (image + explanation folded into caption/stats — one slide)
    picture_slide(prs, "INTERPRETABILITY", "Grad-CAM: Why Generalization Fails",
                 "Same video, same Deepfakes-only checkpoint. Same-manip: attention locks onto "
                 "the central face (blending artifacts). Cross-manip to FaceSwap (worst cell): "
                 "attention collapses off-face — neither model falls back to anything sensible.",
                 "analyze/outputs/gradcam_comparison.png", Emu(5868000), Emu(4200000),
                 stats_label="DF → FS (worst cell)",
                 stats=[("0.256 / 0.198", "cross-manip AUC (xception / cnn_vit) — worse than random"),
                        ("1.000 → 0.000", "xception fake-prob: same-manip vs. cross-manip"),
                        ("face → background", "where attention shifts when the signal is absent")])

    # 10. Software implementation / demo
    content_slide(prs, "SOFTWARE IMPLEMENTATION", "Live Demo", [
        "FastAPI backend + browser frontend: upload an image or short video, get a real/fake "
        "probability from either trained checkpoint. Pipeline mirrors training exactly (MTCNN "
        "crop → resize/normalize → per-frame probability → video-level average).",
        "Verified against a curated, held-out test-split set (never seen in training) — all "
        "correctly and confidently classified by XceptionNet; CNN-ViT correct but visibly less "
        "confident, consistent with its lower baseline accuracy.",
        "Sanity-checked against an out-of-distribution image circulating online (not FF++ data): "
        "both models confidently wrong — expected given the generalization-gap numbers, and a "
        "live demonstration of this project's own finding rather than a contradiction of it.",
        "Runs on CPU by design, so it never competes with GPU training/evaluation jobs.",
    ], size=15)

    # 11. Conclusion, Future Work, References & Contributors (combined closing)
    s = new_slide(prs)
    header(s, "CONCLUSION", "Key Takeaways & Closing")
    left_w = Emu(6350000)
    bullets(s, MARGIN, Emu(1783080), left_w, Emu(4400000), [
        "XceptionNet outperforms the CNN-ViT hybrid on every metric measured — same-distribution "
        "accuracy, generalization gap, and most compression conditions — contradicting the "
        "hypothesis that attention would help generalization.",
        "Both architectures share the same weakness: narrow, method-specific artifacts rather "
        "than a general \"manipulated\" signal — confirmed visually via Grad-CAM.",
        "Future work: official c0/c40 once approved (replacing the proxy), multi-seed runs for "
        "statistical confidence, Grad-CAM on a genuinely-uncertain cell.",
        "Practical takeaway: a deployed detector needs training coverage of its target "
        "manipulation types, or an explicit \"unknown\" fallback — confident-wrong is worse "
        "than honest uncertainty.",
    ], size=13.5, space_after=12)

    right_x = MARGIN + left_w + Emu(228600)
    right_w = SLIDE_W - right_x - MARGIN
    textbox(s, right_x, Emu(1783080), right_w, Emu(320040), [("REFERENCES", 11, True, TEAL)])
    bullets(s, right_x, Emu(2130000), right_w, Emu(1900000), [
        "Rössler et al. (2019). FaceForensics++. ICCV.",
        "Chollet (2017). Xception. CVPR.",
        "Dosovitskiy et al. (2021). ViT. ICLR.",
        "Selvaraju et al. (2017). Grad-CAM. ICCV.",
    ], size=10.5, space_after=7)
    rect(s, right_x, Emu(4100000), right_w, Emu(18288), GRID)
    textbox(s, right_x, Emu(4260000), right_w, Emu(320040), [("CONTRIBUTOR", 11, True, TEAL)])
    textbox(s, right_x, Emu(4560000), right_w, Emu(360000),
            [("Venugopalan Gangadharan", 14, True, TEXT)])
    textbox(s, right_x, Emu(4910000), right_w, Emu(500000),
            [("AI & Data Science — Amrita Vishwa Vidyapeetham, Coimbatore", 10.5, False, MUTED)])
    textbox(s, right_x, Emu(5450000), right_w, Emu(320040), [("GITHUB REPOSITORY", 11, True, TEAL)])
    textbox(s, right_x, Emu(5750000), right_w, Emu(500000),
            [("github.com/DUNE-ODYSSEY/deepfake-detection", 11.5, True, CYAN)])
    footer(s)

    out = "docs/mid_review_update.pptx"
    prs.save(out)
    print(f"saved {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
