import cv2
import numpy as np
import os
from pathlib import Path
import argparse

def binarize_image(img: np.ndarray, method: str = 'sauvola') -> np.ndarray:
    # Converts grayscale image to binary (black and white) using thresholding.
    # This separates the ink (text) from the paper (background).

    if method == 'otsu':
        # Otsu finds a single global threshold for the whole image (good for uniform lighting)
        _, binary = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    elif method == 'adaptive':
        # Adaptive calculates threshold locally for small regions (better for shadows/uneven lighting)
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 21, 10
        )

    elif method == 'sauvola':
        # Sauvola is a specific adaptive method optimized for documents.
        # Here we use a tuned Adaptive Threshold as a robust fallback implementation.
        binary = cv2.adaptiveThreshold(
            img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 25, 15
        )

    else:
        raise ValueError(f"Unknown binarization method: {method}")

    return binary

def remove_noise(binary_img: np.ndarray, min_component_size: int = 10) -> np.ndarray:
    # Cleans the image by removing small specks and dots that aren't part of the text.
    
    # Morphological Opening: Erodes then Dilates to remove tiny noise (salt-and-pepper noise)
    kernel = np.ones((2, 2), np.uint8)
    opened = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel, iterations=1)

    # Connected Components: Identifies all separated blobs in the image
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        255 - opened, connectivity=8
    )

    # Create a clean black canvas
    output = np.zeros_like(opened)

    # Copy only the components that are large enough (actual letters) to the new image
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if area >= min_component_size:
            output[labels == i] = 255

    # Invert back to ensure text is black on white background
    output = 255 - output

    return output

def crop_to_content(img: np.ndarray, padding: int = 20) -> np.ndarray:
    # Cuts away the empty white margins, keeping only the area with actual handwriting.
    
    # Find coordinates of all black pixels (text)
    coords = cv2.findNonZero(255 - img)

    if coords is None:
        return img

    # Get the bounding box (rectangle) that surrounds all the text
    x, y, w, h = cv2.boundingRect(coords)

    # Add some padding (white space) around the text so it doesn't touch the edges
    x = max(0, x - padding)
    y = max(0, y - padding)
    w = min(img.shape[1] - x, w + 2 * padding)
    h = min(img.shape[0] - y, h + 2 * padding)

    # Perform the actual crop
    cropped = img[y:y+h, x:x+w]

    return cropped

def normalize_height(img: np.ndarray, target_height: int = 64) -> np.ndarray:
    # Resizes the image to a fixed height (e.g., 64px) while keeping the Aspect Ratio.
    # This ensures the letters don't get squashed or stretched unnaturally.
    
    h, w = img.shape[:2]

    # Calculate new width based on aspect ratio
    aspect_ratio = w / h
    new_width = int(target_height * aspect_ratio)
    
    # Resize using INTER_AREA interpolation (best for shrinking images)
    resized = cv2.resize(img, (new_width, target_height), interpolation=cv2.INTER_AREA)

    return resized

def process_single_image(
    input_path: str, output_path: str, binarize: bool = True,
    denoise: bool = True, crop: bool = True, normalize_size: bool = False,
    target_height: int = 64, binarization_method: str = 'adaptive', min_noise_size: int = 10
) -> bool:
    # The main pipeline for a single file. Runs all steps in order.
    
    try:
        img = cv2.imread(input_path)
        if img is None:
            return False

        # Convert to grayscale first
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img

        processed = gray.copy()

        # Step 1: Binarization (Thresholding)
        if binarize:
            processed = binarize_image(processed, method=binarization_method)

        # Step 2: Noise Removal (Cleaning)
        if denoise:
            processed = remove_noise(processed, min_component_size=min_noise_size)

        # Step 3: Smart Cropping
        if crop:
            processed = crop_to_content(processed, padding=20)

        # Step 4: Height Normalization (Optional)
        if normalize_size and target_height > 0:
            processed = normalize_height(processed, target_height=target_height)

        # Save the final processed image
        cv2.imwrite(output_path, processed)

        return True

    except Exception as e:
        print(f"Error: {e}")
        return False

def process_directory(input_dir: str, output_dir: str, **kwargs) -> None:
    # Iterates over the entire folder and processes every image found.
    
    os.makedirs(output_dir, exist_ok=True)

    image_files = []
    # Find all image files with common extensions
    for ext in ['.png', '.jpg', '.jpeg', '.tif', '.tiff']:
        image_files.extend(Path(input_dir).glob(f'*{ext}'))
        image_files.extend(Path(input_dir).glob(f'*{ext.upper()}'))

    # Sort and remove duplicates
    image_files = sorted(list(set(image_files)))
    total = len(image_files)

    print(f"Found {total} images to process")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    success_count = 0

    # Loop through all images
    for idx, img_path in enumerate(image_files, 1):
        output_path = os.path.join(output_dir, img_path.name)

        # Process the current image
        success = process_single_image(str(img_path), output_path, **kwargs)

        if success:
            success_count += 1
            if idx % 100 == 0:
                print(f"Processed {idx}/{total} images...")
        else:
            print(f"[ERROR] Failed to process {img_path.name}")

    print("-" * 60)
    print(f"Done! Successfully processed: {success_count}/{total}")

def main():
    # --- CONFIGURATION START ---
    
    # Get the directory where this python script is currently located
    current_script_path = os.path.dirname(os.path.abspath(__file__))

    # 1. Define Input Folder: "Data" (located in the same folder as this script)
    input_folder = os.path.join(current_script_path, "Data")

    # 2. Define Output Folder: "Data_normalized" (located in the same folder as this script)
    output_folder = os.path.join(current_script_path, "Data_normalized")

    # 3. Processing Settings (Change to False if you want to skip a step)
    should_binarize = True       # Convert to black and white
    should_denoise = True        # Remove small noise dots
    should_crop = True           # Crop empty white margins
    should_normalize_height = False # Resize all images to fixed height (e.g. 64px)
    
    # --- CONFIGURATION END ---

    # Check if input directory exists before starting
    if not os.path.exists(input_folder):
        print(f"Error: The input folder '{input_folder}' does not exist.")
        print("Please create a folder named 'Data' next to this script and put your images there.")
        return

    # Run the processing
    process_directory(
        input_dir=input_folder,
        output_dir=output_folder,
        binarize=should_binarize,
        denoise=should_denoise,
        crop=should_crop,
        normalize_size=should_normalize_height,
        target_height=64 # Only relevant if normalize_size is True
    )

if __name__ == '__main__':
    main()