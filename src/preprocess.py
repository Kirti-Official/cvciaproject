import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

CSV_PATH = "dataset/aptos/train_split.csv"
IMAGE_DIR = "dataset/aptos/train_images"

OUTPUT_DIR = "outputs/plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)

IMAGE_SIZE = 224


# ============================================================
# 1. Remove black background safely
# ============================================================

def crop_retina(image):
    """
    Remove the mostly-black background around the retinal region
    while preserving the complete retinal field.
    """

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find pixels that are not black
    mask = gray > 10

    # Get coordinates of non-black pixels
    coords = np.column_stack(np.where(mask))

    if coords.size == 0:
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Add a small padding
    padding = 5

    y_min = max(0, y_min - padding)
    x_min = max(0, x_min - padding)
    y_max = min(image.shape[0], y_max + padding)
    x_max = min(image.shape[1], x_max + padding)

    cropped = image[y_min:y_max, x_min:x_max]

    return cropped


# ============================================================
# 2. Resize while preserving aspect ratio
# ============================================================

def resize_with_padding(image, size=224):
    """
    Resize image while maintaining aspect ratio.
    Remaining area is filled with black padding.
    """

    h, w = image.shape[:2]

    scale = min(size / w, size / h)

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    # Create black canvas
    canvas = np.zeros(
        (size, size, 3),
        dtype=np.uint8
    )

    # Center the resized image
    x_offset = (size - new_w) // 2
    y_offset = (size - new_h) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


# ============================================================
# 3. Apply CLAHE
# ============================================================

def apply_clahe(image):
    """
    Apply CLAHE to the LAB luminance channel.
    """

    # Convert BGR → LAB
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

    # Split channels
    l, a, b = cv2.split(lab)

    # CLAHE
    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    # Merge channels
    lab = cv2.merge((l, a, b))

    # LAB → BGR
    enhanced = cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )

    return enhanced


# ============================================================
# 4. Complete preprocessing pipeline
# ============================================================

def preprocess_image(image_path):
    """
    Complete preprocessing pipeline:

    Original
        ↓
    Retinal crop
        ↓
    Resize to 224x224
        ↓
    CLAHE
    """

    image = cv2.imread(image_path)

    if image is None:
        raise FileNotFoundError(
            f"Could not read image: {image_path}"
        )

    # Crop retinal region
    image = crop_retina(image)

    # Resize
    image = resize_with_padding(
        image,
        IMAGE_SIZE
    )

    # CLAHE
    image = apply_clahe(image)

    return image


# ============================================================
# 5. Visual comparison
# ============================================================

def main():

    df = pd.read_csv(CSV_PATH)

    class_names = {
        0: "No DR",
        1: "Mild",
        2: "Moderate",
        3: "Severe",
        4: "Proliferative"
    }

    # Select one image from every class
    samples = []

    for class_id in range(5):

        sample = df[
            df["diagnosis"] == class_id
        ].iloc[0]

        samples.append(sample)

    # Create comparison figure
    fig, axes = plt.subplots(
        2,
        5,
        figsize=(20, 8)
    )

    for i, sample in enumerate(samples):

        image_id = sample["id_code"]
        label = int(sample["diagnosis"])

        image_path = os.path.join(
            IMAGE_DIR,
            image_id + ".png"
        )

        # Original
        original = cv2.imread(image_path)

        # Convert BGR → RGB
        original_rgb = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        # Processed
        processed = preprocess_image(
            image_path
        )

        # Convert BGR → RGB
        processed_rgb = cv2.cvtColor(
            processed,
            cv2.COLOR_BGR2RGB
        )

        # -------------------------
        # Original
        # -------------------------

        axes[0, i].imshow(original_rgb)

        axes[0, i].set_title(
            f"Original\nClass {label} - {class_names[label]}"
        )

        axes[0, i].axis("off")

        # -------------------------
        # Processed
        # -------------------------

        axes[1, i].imshow(processed_rgb)

        axes[1, i].set_title(
            f"Preprocessed\nClass {label}"
        )

        axes[1, i].axis("off")

    plt.suptitle(
        "APTOS Retinal Image Preprocessing",
        fontsize=16
    )

    plt.tight_layout()

    output_path = os.path.join(
        OUTPUT_DIR,
        "preprocessing_comparison.png"
    )

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print("\nPreprocessing completed.")
    print(f"Saved comparison to: {output_path}")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()