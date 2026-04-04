# segment words.ipynb

Segments individual words in Hebrew handwriting images and draws bounding boxes around each word, for downstream graphology feature extraction.

## Input / Output

| | Path |
|---|---|
| Input | `Data_normalized/` — 2512 normalized PNG images of handwritten Hebrew lines |
| Output | `segment_results/` — same images with green bounding boxes per word |

## How it works (Attempt 13 — current best)

1. Convert to grayscale, find the ruled baseline (row with most ink)
2. Crop above the baseline to exclude the horizontal line
3. Find connected components (letter strokes)
4. Merge components with horizontal gap ≤ 25px into word candidates
5. Validate each candidate with [HebHTR](https://github.com/Lotemn102/HebHTR) via Docker — keep only boxes where Hebrew text is recognised

## Setup

```bash
# 1. Build the HebHTR Docker image (one time, ~15 min)
docker build -f DockerFile -t hebhtr-env .

# 2. Set Google Cloud Vision credentials
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your-key.json"

# 3. Install dependencies
pip install opencv-python numpy matplotlib google-cloud-vision
```

## Usage

Open `segment words.ipynb` and run the cells under **Attempt 13**:
1. Run the function cell (`segment_words_v13`)
2. Run the quick test cell to verify on one image
3. Run the process-all cell to segment all 2512 images
