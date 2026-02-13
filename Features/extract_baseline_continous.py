import cv2
import numpy as np
import pandas as pd
from pathlib import Path
import argparse
import re

def extract_file_id(filename: str) -> int:
    # Helper function to extract the first number from a filename for natural sorting
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    return -1

def calculate_position_score(img: np.ndarray) -> float:
    # core logic: Calculates vertical position relative to the baseline.
    # Returns a score between 0.0 (Cutting the line) and 1.0 (Floating above).
    
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img

    # Otsu's thresholding to create binary image
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    img_h, img_w = thresh.shape

    # Define a wide horizontal kernel to detect only the ruled lines (ignoring text)
    min_line_width = int(img_w * 0.3)
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (min_line_width, 1))
    
    # Morphological opening to isolate horizontal lines
    detected_lines_map = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=1)
    
    cnts_lines, _ = cv2.findContours(detected_lines_map, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not cnts_lines:
        return None

    valid_lines = []
    for c in cnts_lines:
        x, y, w, h = cv2.boundingRect(c)
        # Filter out lines that are too close to the top/bottom borders (noise)
        if y > 10 and y < img_h - 10:
            valid_lines.append(c)
            
    if not valid_lines:
        valid_lines = [max(cnts_lines, key=cv2.contourArea)]

    # Subtract the detected lines from original binary to get only the handwriting
    text_only = cv2.subtract(thresh, detected_lines_map)
    
    # Clean small noise
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    text_only = cv2.morphologyEx(text_only, cv2.MORPH_OPEN, kernel_clean)
    
    cnts_text, _ = cv2.findContours(text_only, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_bottoms = []
    text_centers_y = []
    
    for c in cnts_text:
        if cv2.contourArea(c) < 15:
            continue
            
        tx, ty, tw, th = cv2.boundingRect(c)
        text_bottoms.append(ty + th)
        text_centers_y.append(ty + th/2)

    if not text_bottoms:
        return 1.0

    # Use Median to determine text baseline position, ignoring outliers like descending letters ('ן', 'ך')
    median_text_bottom = np.median(text_bottoms)
    avg_text_y = np.mean(text_centers_y)

    best_line_y = 0
    min_dist_to_line = 99999
    
    # Identify which ruled line the text belongs to (closest line to text center)
    for line_c in valid_lines:
        lx, ly, lw, lh = cv2.boundingRect(line_c)
        current_line_y = ly 
        
        dist = abs(current_line_y - avg_text_y)
        if dist < min_dist_to_line:
            min_dist_to_line = dist
            best_line_y = current_line_y

    # Calculate distance: Positive = Above line, Negative = Below line
    distance = best_line_y - median_text_bottom
    
    # Sensitivity factor: determines the pixel range that maps to the 0-1 score
    SENSITIVITY = 60.0 
    
    # Normalize score: 0.5 represents text sitting exactly on the line
    raw_score = 0.5 + (distance / SENSITIVITY)
    
    # Clip result to strict 0.0 - 1.0 range
    final_score = max(0.0, min(1.0, raw_score))
    
    return final_score

def process_images_for_position(
    input_dir: str,
    output_excel: str,
    extensions: tuple = ('.png', '.jpg', '.jpeg', '.tif', '.tiff')
) -> pd.DataFrame:
    # Main batch processing function: Iterates images, sorts them, and saves results to Excel
    
    image_files = set()
    for ext in extensions:
        image_files.update(Path(input_dir).glob(f'*{ext}'))
        image_files.update(Path(input_dir).glob(f'*{ext.upper()}'))

    # Sort using natural sort order (e.g., 1, 2, ... 10 instead of 1, 10, 2)
    image_files = sorted(list(image_files), key=lambda p: extract_file_id(p.name))
    
    total = len(image_files)

    print(f"Found {total} images to analyze")
    print(f"Output Excel file: {output_excel}")
    
    results = []

    for idx, img_path in enumerate(image_files, 1):
        try:
            file_id = extract_file_id(img_path.name)
            img = cv2.imread(str(img_path))

            if img is None:
                print(f"[WARNING] Could not read: {img_path.name}")
                results.append({
                    'ID_Number': file_id,
                    'Filename': img_path.name,
                    'value': None
                })
                continue

            score = calculate_position_score(img)
            
            # Fallback to neutral score (0.5) if detection fails
            if score is None:
                score = 0.5
                print(f"[INFO] Detection fallback for {img_path.name}")

            results.append({
                'ID_Number': file_id,
                'Filename': img_path.name,
                'value': round(score, 3)
            })

            if idx % 500 == 0:
                print(f"Processed {idx}/{total} images...")

        except Exception as e:
            print(f"[ERROR] Failed to process {img_path.name}: {str(e)}")
            results.append({
                'ID_Number': extract_file_id(img_path.name),
                'Filename': img_path.name,
                'value': None
            })

    print(f"Analysis complete!")

    df = pd.DataFrame(results)
    df.to_excel(output_excel, index=False, sheet_name='Line Position')

    print(f"\nExcel file saved: {output_excel}")
    print("\nPosition Statistics (0=Cutting, 0.5=On Line, 1=Floating):")
    print(f"Min: {df['value'].min():.3f}")
    print(f"Max: {df['value'].max():.3f}")
    print(f"Mean: {df['value'].mean():.3f}")
    print(f"Median: {df['value'].median():.3f}")
    print(f"Std Dev: {df['value'].std():.3f}")

    return df

def main():
    parser = argparse.ArgumentParser(description='Extract Handwriting Position feature')
    parser.add_argument('input_dir', type=str, nargs='?', default=r'normalized_output', help='Input directory')
    parser.add_argument('output_excel', type=str, nargs='?', default='position_feature_strict.xlsx', help='Output file')

    args = parser.parse_args()

    if not Path(args.input_dir).exists():
        print(f"Error: Input directory '{args.input_dir}' not found.")
        return

    process_images_for_position(input_dir=args.input_dir, output_excel=args.output_excel)

if __name__ == '__main__':
    main()