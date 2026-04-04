# remove_printed_text.ipynb

Removes printed/typed text from scanned exam pages, keeping only the handwritten content.

## Input / Output

| | Path |
|---|---|
| Input | `dataPrinted/` — scanned images with both printed and handwritten text |
| Output | `handwritenOnly/` — images with printed text erased |

## How it works

1. Build a pixel-wise max-map across a sample of 300 images
2. Printed text is dark in every image → flagged as always-dark pixels
3. Handwriting is dark in only some images → preserved
4. Removes flagged pixels and ruled lines from each image

## Setup

```bash
pip install opencv-python numpy tqdm matplotlib
```

## Usage

Open `remove_printed_text.ipynb` and run all cells in order.
Adjust `A1_SAMPLE_SIZE`, `A1_INK_THRESHOLD`, and `A1_PRINTED_THRESHOLD` at the top of the config cell if needed.
