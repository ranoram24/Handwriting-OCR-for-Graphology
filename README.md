# Handwriting OCR for Graphology

**Authors:** Ran Uram · Shahar Lankry · Daniel Geron

A two-stage pipeline that processes scanned handwriting PDFs, extracts 12 graphological features, and generates a personalized Hebrew graphological report via the Claude API.

---

## Pipeline Overview

```
whole image/          (input PDFs, 2 pages each)
      │
      ▼
[normalize data.ipynb]
      │
      ├── Final Data/              raw line crops (PNG)
      ├── FinalDataNormalized/     binarized + denoised line crops
      └── FinalSegmentation/       line crops annotated with green word boxes
                                   + _COLUMNS.png files (blank-page column detection)
      │
      ▼
[Feature Extraction and Report.ipynb]
      │
      ├── FinalFeatureExtraction/tables/    per-feature .xlsx + all_features.xlsx
      ├── FinalFeatureExtraction/visualization_results/   score-annotated PNGs per feature
      └── graphological_report.html / .pdf
```

---

## Notebook 1 — `normalize data.ipynb`

### PDF Loading & Page Detection

Each PDF is expected to have two pages: a **lined page** (the handwriting form) and a **blank page** (a numbered column template). `select_pages` renders both pages at 300 DPI using PyMuPDF (`fitz`) and picks which is which by scoring how many long horizontal runs each page contains (`horizontal_line_score`).

### Frame Detection

`detect_and_crop_frame` finds the printed rectangular border on each page by detecting strong horizontal and vertical lines with morphological operations. The image is cropped to the inner frame so that page-margin noise is removed before any further processing. If the frame cannot be detected, the page is skipped.

### Corner Mark Removal

After frame-cropping the lined page, `remove_corner_marks` erases the L-shaped registration bracket marks in all four corners by painting connected components in corner regions white. `crop_to_writing_area` additionally trims the top edge down to the bottom of the topmost bracket marks.

### Line Detection

`detect_printed_lines` locates the horizontal ruled lines using a multi-scale morphological approach (three kernel widths), smoothed with a uniform filter, and `scipy.signal.find_peaks`. It returns a sorted list of y-coordinates for every printed baseline.

### Line Cropping → `Final Data/`

`crop_lines` produces one PNG crop per writing line plus one signature crop:

- **line_01** — uses the median inter-line spacing to build a virtual zone above the first detected baseline, matching every other crop in height.
- **line_02 … line_N-1** — each spans from one baseline to the next.
- **`_signature.png`** — the strip between the second-to-last and last detected baseline.

### Image Normalization → `FinalDataNormalized/`

`normalize_image` runs a four-step pipeline on every crop:

1. **Binarize** — adaptive Gaussian threshold (block size 21, C=10).
2. **Denoise** — morphological opening followed by connected-component filtering (min area 10 px).
3. **Crop to content** — vertical tight crop with 20 px padding; horizontal extent is preserved to maintain margin information.
4. *(Optional)* **Height normalization** — resize to a fixed target height (disabled by default).

### Word Segmentation → `FinalSegmentation/`

`get_word_boxes` detects word bounding boxes on each normalized line image:

1. Detects and erases the printed baseline (long horizontal morphological run) so it does not interfere with word detection.
2. Blanks the top 25 px margin to suppress bleed-through from the line above.
3. Runs connected-component analysis, filters by minimum component dimensions, and merges nearby components using a dynamic gap threshold (1.9× the median inter-component gap).

`annotate_segmentation` draws green rectangles for every word box and saves the result to `FinalSegmentation/`. Images where no word boxes are found are deleted from both `FinalSegmentation/` and `FinalDataNormalized/`.

### Column Detection → `FinalDataNormalized/*_COLUMNS.png`

`column_detection` operates on the normalized blank page:

1. `detect_frame_bounds` finds the inner frame boundary.
2. Inside the frame, connected components touching the border are removed to eliminate residual frame artefacts.
3. A column profile is built and thresholded (2%) to find content bands. Bands narrower than 25 px are discarded.
4. Within each column band, row profiles detect individual number rows. Isolated bands (gap > 4× median) are discarded to exclude signatures or labels.
5. Column bounding boxes are drawn as red rectangles and saved as `<basename>_COLUMNS.png` for use by the margin and column-spacing features downstream.

---

## Notebook 2 — `Feature Extraction and Report.ipynb`

### Paths & Configuration

- `normalised_images_folder` — `FinalDataNormalized/`
- `feature_tables_location` — `FinalFeatureExtraction/tables/`
- `visualization_folder` — `FinalFeatureExtraction/visualization_results/`

### Feature Extraction (12 Features)

All raw feature values are computed per line image, then **z-score normalized to \[0, 1\]** across all lines of a subject. Values outside one standard deviation are clipped to the 0/1 boundary. The normalization direction (invert flag) is set per feature so that 0.0/0.5/1.0 map to the graphologically meaningful low/mid/high anchors described in the dictionary.

#### 1. Slant
Two methods are combined (weighted 2:1 in favor of shear):

- **Shear method** (`measure_slant_by_shear`) — tests shear angles from −30° to +30°, picks the angle that maximizes the variance of column pixel sums (most upright alignment of strokes).
- **Moments method** (`measure_slant_by_moments`) — fits an ellipse to each connected component and averages the major-axis angles, filtered by IQR.

Scale: `0.0` = strong left lean → `0.5` = vertical → `1.0` = strong right lean.

#### 2. Stroke Thickness
`calculate_stroke_thickness_pure` removes printed horizontal rules via Hough Transform, filters to valid handwriting components (discarding headers/footers/noise), and applies the **distance transform** to the binarized ink. The mean of the local maxima of the distance map gives the mean stroke radius; diameter = mean radius × 2.

Scale: `0.0` = very thin, delicate → `0.5` = medium → `1.0` = thick, heavy.

#### 3. Baseline Alignment
`calculate_position_score` detects the printed ruled line (wide morphological kernel) and measures where the **bottom of each text component** sits relative to it. Positive = above the line, negative = below. The mean signed distance is the raw score.

Scale: `0.0` = text hangs below the line → `0.5` = sits on the line → `1.0` = floats above.

#### 4. Baseline Slope
`calculate_baseline_slope_raw` extracts word-box centroids from the segmented image (`get_letter_centroids`) and fits a linear regression line. The slope angle (in degrees) is the raw feature. `remove_lines_for_baseline_slope` removes printed lines before centroid detection.

Scale: `0.0` = descending line → `0.5` = horizontal → `1.0` = ascending line.

#### 5. Right Margin
`calculate_right_margin` uses red rectangle markers from the `_COLUMNS.png` blank-page image to find the rightmost column boundary (for blank pages) or word-box rightmost extent from the segmented image (for lined pages). The gap from the rightmost content to the page right edge is normalized against a 30% threshold.

Scale: `0.0` = no margin → `0.5` = ~2 cm → `1.0` = very wide (>30%).

#### 6. Left Margin
`calculate_left_margin` — mirror logic of right margin: distance from the page left edge to the leftmost content, normalized against a 30% threshold.

Scale: `0.0` = no margin → `0.5` = ~2 cm → `1.0` = very wide (>30%).

#### 7. Top Margin
`calculate_top_margin` reads the topmost red marker row from the `_COLUMNS.png` image to locate where content starts, then measures the fraction of page height above it. Normalized against a 30% threshold.

Scale: `0.0` = flush with top → `0.5` = ~10–15% → `1.0` = very large (>30%).

#### 8. Bottom Margin
`calculate_bottom_margin` finds the bottommost content row and measures the empty space below, normalized against a 30% threshold.

Scale: `0.0` = flush with bottom → `0.5` = ~10–15% → `1.0` = very large (>30%).

#### 9. Column Spacing
`calculate_column_spacing` detects the red column-marker bands in the `_COLUMNS.png` image, measures the gaps between consecutive column bands, averages them, and normalizes against a 30% threshold.

Scale: `0.0` = columns touching/overlapping → `0.5` = normal spacing → `1.0` = very wide gaps.

#### 10. Word Spacing
`calculate_word_spacing` reads green word-box rectangles from the segmented image, sorts them by x-position, and computes the median inter-box gap normalized by the median letter height.

Scale: `0.0` = packed together → `0.5` = ~2 letter-widths apart → `1.0` = very spread out.

#### 11. Letter Size
`calculate_letter_size` extracts word boxes from the segmented image and computes the **median word-box height normalized by the image height**, giving a scale-invariant size ratio. Computed separately for regular lines and for signature images.

Scale: `0.0` = very small → `0.5` = ~3 mm average → `1.0` = very large.

#### 12. Roundness (vs. Angularity)
`calculate_angularity_raw` measures the mean **corner sharpness** of connected letter components using the Harris corner detector. A higher corner response means more angular strokes. The raw score is inverted during z-score normalization so that high Roundness = smooth/round writing.

Scale: `0.0` = very angular → `0.5` = typical → `1.0` = very round.

### Run All Feature Extraction

The `## Run All Feature Extraction` section iterates over every image in `FinalDataNormalized/`, dispatching each to the appropriate feature functions based on filename suffix (`_signature`, `_COLUMNS`, or regular line). All raw values are collected per line, z-score normalized across the subject's lines, and saved to individual Excel files (`slant.xlsx`, `stroke_thickness.xlsx`, etc.) plus per-feature visualization PNGs annotated with the normalized score.

### Unified Feature Table

The `## Unified Feature Table` section merges all per-feature Excel files into a single `all_features.xlsx`. Each row is one image; columns are the 12 normalized feature scores. An `Average` row summarizes the subject. A `Number_of_Columns` summary row records how many column bands were detected on the blank page.

### Graphological Report

The `## Graphological Report` section:

1. **Loads** `all_features.xlsx` and extracts the `Average` row along with line count, column count, and separate letter sizes for regular lines and the signature.
2. **Builds a feature text block** mapping each normalized value to a Hebrew description with its scale anchor labels.
3. **Adds system notices** if the column count ≠ 3 (unreliable column-spacing data) or if fewer than 16 lines are present (reduced accuracy).
4. **Calls the Claude API** (`claude-opus-4-5`, 2048 tokens) with a detailed system prompt instructing the model to write a professional Hebrew graphological report in five named categories, two paragraphs each, narrative prose only, 600–750 words total. A full Hebrew graphological dictionary is included in the user prompt.
5. **Saves the report** as `graphological_report.html` (UTF-8, RTL layout) and attempts PDF export via `weasyprint` then `pdfkit` as fallbacks.

---

## Directory Structure

```
Handwriting-OCR-for-Graphology/
├── whole image/                  input PDFs (one per subject, 2 pages each)
├── Final Data/                   raw line crops per subject
├── FinalDataNormalized/          normalized grayscale PNGs + _COLUMNS.png files
├── FinalSegmentation/            segmentation-annotated PNGs (word boxes)
├── FinalFeatureExtraction/
│   ├── tables/                   per-feature .xlsx files + all_features.xlsx
│   └── visualization_results/    per-feature subdirs with annotated PNGs
├── normalize data.ipynb          Stage 1: preprocessing pipeline
└── Feature Extraction and Report.ipynb   Stage 2: features + report
```

---

## Dependencies

```
opencv-python
numpy
scipy
matplotlib
PyMuPDF (fitz)
pandas
openpyxl
anthropic
weasyprint        # or pdfkit + wkhtmltopdf
```

---

## How to Run

1. Place subject PDFs (2-page scans) in `whole image/`.
2. Run all cells in **`normalize data.ipynb`** — produces `Final Data/`, `FinalDataNormalized/`, and `FinalSegmentation/`.
3. Set your Anthropic API key in the `_api_key` variable in **`Feature Extraction and Report.ipynb`**.
4. Run all cells in **`Feature Extraction and Report.ipynb`** — produces `FinalFeatureExtraction/tables/all_features.xlsx`, visualization images, and the final `graphological_report.html` / `.pdf`.
