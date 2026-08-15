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

TOTAL_SLIDES = 23
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

    # 2. Agenda
    content_slide(prs, "AGENDA", "What We'll Cover", [
        "Introduction, motivation, and the problem being addressed",
        "Objectives and scope of the study",
        "Dataset, preprocessing, and exploratory data analysis",
        "Model architectures: XceptionNet baseline vs. CNN-ViT hybrid",
        "Training/validation methodology and evaluation metrics",
        "Results: baseline comparison, cross-manipulation generalization, compression robustness",
        "Interpretability (Grad-CAM) and the working live demo",
        "Conclusion, honest limitations, and future work",
    ])

    # 3. Introduction & Motivation
    content_slide(prs, "INTRODUCTION", "Motivation", [
        "Deepfakes — AI-manipulated face videos — have moved from research curiosity to a "
        "real misinformation and fraud vector (fake news clips, impersonation scams, non-consensual media).",
        "Most published detectors report a single headline accuracy number, usually on the same "
        "manipulation type and compression level they were trained on.",
        "That number is misleading in practice: real-world deepfakes rarely match a detector's "
        "training distribution exactly — different generator, different compression, different source.",
        "This project asks a more useful question than \"how accurate is it\": "
        "“how much does accuracy collapse when the test distribution shifts, and does a more "
        "modern architecture (attention-based) hold up better than a plain CNN?”",
    ])

    # 4. Problem Statement
    content_slide(prs, "PROBLEM STATEMENT", "Real-World Relevance", [
        "Problem: face-forgery detectors trained on one manipulation method or compression level "
        "often fail silently on another — the failure isn't visible until deployment.",
        "This matters wherever a detector is actually used: content moderation pipelines, "
        "journalism verification tools, court-admissible evidence checks — all see manipulation "
        "types and compression levels the model never trained on.",
        "A detector that reports 99% accuracy in a paper but ~57% AUC (worse than random) on an "
        "unseen manipulation method is not deployment-ready, even though its benchmark number looks excellent.",
        "This project measures that gap directly and explains it visually (Grad-CAM), rather than "
        "reporting a single flattering accuracy figure.",
    ])

    # 5. Objectives
    content_slide(prs, "OBJECTIVES", "Objectives & Scope", [
        "Train and fairly compare two architectures on identical data/protocol: XceptionNet "
        "(CNN baseline, the standard FF++ detector) vs. a CNN-ViT hybrid (ResNet-50 features + "
        "Transformer encoder, to test whether global self-attention improves generalization).",
        "Measure same-distribution accuracy (Experiment 1: baseline comparison).",
        "Measure cross-manipulation generalization: train on one forgery method, test on all four, "
        "for a full 4×4 matrix per model (Experiment 2).",
        "Measure compression robustness: does accuracy hold up under heavier video compression "
        "(Experiment 3).",
        "Explain *why* generalization succeeds or fails using Grad-CAM, not just report a number.",
        "Ship a working, testable software artifact (live demo), not just offline metrics.",
    ])

    # 6. Literature grounding
    content_slide(prs, "LITERATURE", "Grounding & References", [
        "Rössler et al., “FaceForensics++: Learning to Detect Manipulated Facial Images” "
        "(ICCV 2019) — source of the dataset and the standard 4-method/3-compression benchmark protocol used here.",
        "Chollet, “Xception: Deep Learning with Depthwise Separable Convolutions” (CVPR 2017) "
        "— the CNN baseline architecture, still a standard reference detector for this task.",
        "Dosovitskiy et al., “An Image is Worth 16x16 Words: Transformers for Image Recognition "
        "at Scale” (ICLR 2021) — motivates the Transformer-encoder half of the hybrid model.",
        "Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based "
        "Localization” (ICCV 2017) — the interpretability method used to explain the results.",
        "Full citation list also included in the GitHub repository README.",
    ], size=14)

    # 7. Methodology overview (pipeline)
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

    # 8. Dataset
    content_slide(prs, "DATASET", "Description & Source", [
        "FaceForensics++ (FF++) — the standard academic benchmark for face-forgery detection.",
        "5,000 videos total: 1,000 real (YouTube interviews) + 4 × 1,000 manipulated, one set "
        "per method — Deepfakes, Face2Face, FaceSwap, NeuralTextures — all sharing the same "
        "real-video identities.",
        "Compression: c23 (H.264, “light” compression, the standard training setting) via a "
        "Kaggle mirror, arranged into the official FaceForensics++ folder layout.",
        "Official train/val/test splits (video-level, from the FaceForensics++ GitHub) used "
        "throughout — critical, since random frame-level splits would leak the same identity "
        "into both train and test and inflate accuracy.",
        "Official c0 (raw)/c40 (heavy compression) access was still pending at time of this "
        "review — handled with a documented local workaround (see Compression Robustness slide).",
    ], size=14.5)

    # 9. Preprocessing
    content_slide(prs, "METHODOLOGY", "Data Preprocessing", [
        "Face extraction via MTCNN (facenet-pytorch), run on GPU: 20 frames sampled evenly per "
        "video, each cropped to the detected face with a 1.3x margin (captures blending-boundary "
        "artifacts just outside the tight face box, where manipulation seams often show).",
        "Crops resized to 299×299 (fits both models: used directly for XceptionNet, "
        "downsampled to 224×224 for the hybrid).",
        "Result: ~99,987 face crops from all 5,000 videos (a handful of frames dropped where no "
        "face was detected).",
        "Real/fake class imbalance handled at training time via a weighted random sampler, not "
        "by discarding data.",
    ])

    # 10. EDA
    picture_slide(prs, "EDA", "Data Quality: Outlier Check (IQR)",
                 "Per-video metadata (frame count, file size) checked via box-plot IQR across the "
                 "5,000 videos actually used, before committing to full preprocessing.",
                 "analyze/outputs/outlier_boxplots.png", Emu(7406640), Emu(3648169),
                 stats_label="RESULT",
                 stats=[("5,000", "videos analyzed (real + 4 methods)"),
                        ("209", "flagged on frame count (4.2%)"),
                        ("326", "flagged on file size (6.5%)"),
                        ("502", "unique outliers overall (10.0%)")])

    # 11. Model 1 Xception
    content_slide(prs, "ALGORITHM SELECTION", "Model 1 — XceptionNet (Baseline)", [
        "Standard CNN baseline for this exact task in the FF++ literature — depthwise-separable "
        "convolutions (\"Xception\", Chollet 2017), ImageNet-pretrained backbone.",
        "2048-d pooled features → dropout → single linear layer → real/fake logit.",
        "Purely convolutional: local receptive fields, no explicit long-range spatial reasoning.",
        "Chosen as the baseline specifically because it's the architecture most cross-manipulation "
        "generalization claims in the literature are benchmarked against.",
    ])

    # 12. Model 2 CNN-ViT
    content_slide(prs, "ALGORITHM SELECTION", "Model 2 — CNN-ViT Hybrid", [
        "Hypothesis under test: does adding global self-attention on top of CNN features improve "
        "cross-manipulation generalization, since attention can relate distant regions "
        "(e.g. blending-boundary inconsistencies) that local convolutions can't see jointly?",
        "ResNet-50 (stages 1–3, ImageNet-pretrained) extracts a 14×14×1024 feature map → "
        "projected to 512-d tokens → CLS token + learned positional embeddings.",
        "6-layer, 8-head Transformer encoder (pre-norm, GELU) over the 196 tokens + CLS.",
        "CLS token → LayerNorm → linear head → real/fake logit.",
        "Same optimizer, schedule, and data as XceptionNet — architecture is the only variable "
        "between the two models, by design.",
    ])

    # 13. Training/validation methodology
    content_slide(prs, "METHODOLOGY", "Training & Validation", [
        "Mixed-precision (AMP) training, AdamW optimizer, cosine LR schedule, up to 15 epochs "
        "with early stopping (patience 4 on validation video-AUC).",
        "GPU capacity benchmarked first (RTX 3060 Laptop, 6GB): found XceptionNet silently "
        "collapses in throughput above batch size 32 — not an OOM crash, but Windows paging GPU "
        "memory into system RAM once allocation exceeds ~6.4GB (10–25x slower, not a crash). "
        "Fixed by keeping batch size 32 for both models.",
        "Resumable training (checkpoint every epoch) — verified in practice: the full run "
        "survived multiple unplanned interruptions overnight without losing progress.",
        "10 training runs total: 2 baseline (all-methods) + 8 cross-manipulation (one model × "
        "one method each), all on identical protocol.",
    ], size=14)

    # 14. Evaluation metrics
    content_slide(prs, "METHODOLOGY", "Evaluation Metrics", [
        "Accuracy, F1, and AUC — computed at both the frame level and the video level.",
        "Video-level score = mean of that video's frame-level probabilities (standard FF++ "
        "protocol) — the metric actually reported as “the” result, since real-world detection "
        "is a per-video decision, not a per-frame one.",
        "AUC (area under the ROC curve) is the primary metric for comparing generalization, since "
        "it's threshold-independent — important because a model can be systematically "
        "miscalibrated (all fakes scored low) rather than merely \"uncertain,\" and AUC still "
        "exposes that.",
        "36 total evaluations run: 2 baseline + 32 cross-manipulation (4×4 matrix × 2 models) + "
        "2 compression-robustness.",
    ])

    # 15. Results: baseline comparison
    table_slide(prs, "RESULTS · EXPERIMENT 1", "Baseline Comparison (same-distribution, c23)",
               "Both models trained and tested on all 4 methods — the headline accuracy number "
               "most papers stop at.",
               ["Model", "Video Acc", "Video F1", "Video AUC", "Frame Acc", "Frame AUC"],
               [["XceptionNet", "97.29%", "0.983", "0.9958", "95.49%", "0.986"],
                ["CNN-ViT Hybrid", "94.71%", "0.967", "0.9836", "92.76%", "0.962"]],
               [Emu(2400000), Emu(1700000), Emu(1700000), Emu(1700000), Emu(1700000), Emu(1700000)],
               note="XceptionNet leads on every same-distribution metric — the first sign the "
                    "hybrid's extra complexity isn't paying off on this task.")

    # 16. Results: cross-manipulation
    picture_slide(prs, "RESULTS · EXPERIMENT 2", "Cross-Manipulation Generalization",
                 "Train on one manipulation method, test on all four (4×4 matrix per model). "
                 "Diagonal = same-manip; off-diagonal = the actual generalization test.",
                 "analysis/cross_manipulation_heatmaps.png", Emu(9601200), Emu(3870968))

    # 17. Results: generalization gap table + finding
    s = table_slide(prs, "RESULTS · EXPERIMENT 2", "Generalization Gap: Hypothesis vs. Reality",
                    "Same-manipulation AUC vs. average cross-manipulation AUC, per model.",
                    ["Model", "Same-Manip AUC", "Cross-Manip AUC", "Generalization Gap"],
                    [["XceptionNet", "0.9947", "0.5969", "0.398"],
                     ["CNN-ViT Hybrid", "0.9917", "0.5710", "0.421"]],
                    [Emu(2600000), Emu(2600000), Emu(2600000), Emu(2600000)])
    callout(s, MARGIN, Emu(4200000), Emu(10852800), Emu(1600000), "HEADLINE FINDING",
           "The original hypothesis — “attention helps the hybrid generalize better” — did "
           "NOT hold. XceptionNet has both higher same-distribution accuracy AND a smaller "
           "generalization gap (0.398 vs. 0.421). Both models still collapse to worse-than-random "
           "on their hardest cross-manip cell (Deepfakes→FaceSwap: 0.256 / 0.198 AUC) — reported "
           "honestly as a genuine negative result, not hidden.", RED)

    # 18. Results: compression robustness
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

    # 19. Grad-CAM
    picture_slide(prs, "INTERPRETABILITY", "Grad-CAM: Why Generalization Fails",
                 "Same video (000_003), same Deepfakes-only checkpoint, two manipulation methods. "
                 "Top: same-manip (correct). Bottom: cross-manip to FaceSwap, its worst cell (correct "
                 "→ confidently wrong).",
                 "analyze/outputs/gradcam_comparison.png", Emu(5868000), Emu(4200000),
                 stats_label="DF → FS (worst cell)",
                 stats=[("0.256 / 0.198", "cross-manip AUC (xception / cnn_vit) — worse than random"),
                        ("1.000 → 0.000", "xception fake-prob: same-manip vs. cross-manip"),
                        ("face → background", "where attention shifts when the signal is absent")])

    # 20. Grad-CAM explanation (text, paired with above conceptually)
    content_slide(prs, "INTERPRETABILITY", "Reading the Grad-CAM Result", [
        "Same-manipulation (Deepfakes): both models' attention concentrates tightly on the "
        "central face — nose, mouth, eyes — exactly where Deepfakes' autoencoder-blending "
        "artifacts appear. Confident AND interpretably correct.",
        "Cross-manipulation (same video, FaceSwap): XceptionNet's attention collapses to a stray "
        "off-face hotspot near the background — essentially no face-relevant evidence found, so "
        "it defaults to “real.”",
        "CNN-ViT's attention goes the opposite way: diffuse across the entire frame including the "
        "background, not localized anywhere meaningful.",
        "Conclusion: neither model is merely “uncertain” on unseen manipulation types — each "
        "confidently learned a narrow, method-specific signal, with nothing sensible to fall back "
        "on once that exact signal is absent. This is a visual account of the 0.20–0.26 AUC "
        "result, not just a number.",
    ], size=14)

    # 21. Software implementation / demo
    content_slide(prs, "SOFTWARE IMPLEMENTATION", "Live Demo", [
        "FastAPI backend + browser frontend: upload an image or short video, get a real/fake "
        "probability from either trained checkpoint.",
        "Pipeline mirrors training exactly: MTCNN face crop → model-specific resize/normalize "
        "→ per-frame probability → video-level average (same protocol as evaluation).",
        "Verified against a curated, held-out test-split set (never seen in training) — all "
        "correctly and confidently classified by XceptionNet; CNN-ViT correct on the same set but "
        "visibly less confident, consistent with its lower baseline accuracy.",
        "Sanity-checked against an out-of-distribution image circulating online (not FF++ data): "
        "both models confidently wrong — expected given the generalization-gap numbers, and a "
        "live demonstration of this project's own finding rather than a contradiction of it.",
        "Runs on CPU by design, so it never competes with GPU training/evaluation jobs.",
    ], size=14)

    # 22. Conclusion & Future Work
    content_slide(prs, "CONCLUSION", "Conclusion & Future Enhancements", [
        "XceptionNet outperforms the CNN-ViT hybrid on this task on every metric measured — "
        "same-distribution accuracy, generalization gap, and (marginally) even most compression "
        "conditions — contradicting the initial hypothesis that attention would help generalization.",
        "Both architectures share the same underlying weakness: they learn narrow, "
        "method-specific artifacts rather than a general “this face was manipulated” signal, "
        "visually confirmed via Grad-CAM.",
        "Future work: re-run compression robustness against official c0/c40 once "
        "FaceForensics++ access is approved, replacing the documented proxy; repeat training with "
        "multiple random seeds for statistical confidence intervals; extend Grad-CAM to a "
        "genuinely-uncertain cross-manip cell (AUC near 0.5) rather than only the "
        "worse-than-random case, to see if the attention pattern differs.",
        "Practical implication: any deployed face-forgery detector needs either training data "
        "covering the manipulation types it will face, or an explicit “unknown/out-of-distribution” "
        "fallback — a confident wrong answer, as shown here, is worse than an honest “I don't know.”",
    ], size=14)

    # 23. References + Contributors + repo
    s = new_slide(prs)
    header(s, "REFERENCES & CONTRIBUTORS", "Closing")
    bullets(s, MARGIN, Emu(1783080), Emu(10852800), Emu(2200000), [
        "Rössler, A. et al. (2019). FaceForensics++: Learning to Detect Manipulated Facial Images. ICCV.",
        "Chollet, F. (2017). Xception: Deep Learning with Depthwise Separable Convolutions. CVPR.",
        "Dosovitskiy, A. et al. (2021). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR.",
        "Selvaraju, R. R. et al. (2017). Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization. ICCV.",
    ], size=12.5, space_after=10)
    rect(s, MARGIN, Emu(4100000), Emu(10852800), Emu(18288), GRID)
    textbox(s, MARGIN, Emu(4300000), Emu(6000000), Emu(320040),
            [("CONTRIBUTOR", 11, True, TEAL)])
    textbox(s, MARGIN, Emu(4650000), Emu(6000000), Emu(400000),
            [("Venugopalan Gangadharan", 16, True, TEXT)])
    textbox(s, MARGIN, Emu(5050000), Emu(6000000), Emu(320040),
            [("AI & Data Science — Amrita Vishwa Vidyapeetham, Coimbatore", 12, False, MUTED)])
    textbox(s, MARGIN, Emu(5550000), Emu(6000000), Emu(320040),
            [("GITHUB REPOSITORY", 11, True, TEAL)])
    textbox(s, MARGIN, Emu(5900000), Emu(10000000), Emu(400000),
            [("github.com/DUNE-ODYSSEY/deepfake-detection", 14, True, CYAN)])
    footer(s)

    out = "docs/mid_review_update.pptx"
    prs.save(out)
    print(f"saved {out} ({len(prs.slides)} slides)")


if __name__ == "__main__":
    main()
