import os
import cv2
import pandas as pd
import numpy as np


# ============================================================
# Configuration
# ============================================================

IMAGE_DIR = "dataset/aptos/train_images"
OUTPUT_DIR = "dataset/aptos/preprocessed"

IMAGE_SIZE = 224

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Crop retina
# ============================================================

def crop_retina(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    mask = gray > 10

    coords = np.column_stack(
        np.where(mask)
    )

    if coords.size == 0:
        return image

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    padding = 5

    y_min = max(
        0,
        y_min - padding
    )

    x_min = max(
        0,
        x_min - padding
    )

    y_max = min(
        image.shape[0],
        y_max + padding
    )

    x_max = min(
        image.shape[1],
        x_max + padding
    )

    return image[
        y_min:y_max,
        x_min:x_max
    ]


# ============================================================
# Resize
# ============================================================

def resize_with_padding(
    image,
    size=224
):

    h, w = image.shape[:2]

    scale = min(
        size / w,
        size / h
    )

    new_w = int(w * scale)
    new_h = int(h * scale)

    resized = cv2.resize(
        image,
        (new_w, new_h),
        interpolation=cv2.INTER_AREA
    )

    canvas = np.zeros(
        (size, size, 3),
        dtype=np.uint8
    )

    x_offset = (
        size - new_w
    ) // 2

    y_offset = (
        size - new_h
    ) // 2

    canvas[
        y_offset:y_offset + new_h,
        x_offset:x_offset + new_w
    ] = resized

    return canvas


# ============================================================
# CLAHE
# ============================================================

def apply_clahe(image):

    lab = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2LAB
    )

    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )

    l = clahe.apply(l)

    lab = cv2.merge(
        (l, a, b)
    )

    return cv2.cvtColor(
        lab,
        cv2.COLOR_LAB2BGR
    )


# ============================================================
# Complete deterministic preprocessing
# ============================================================

def preprocess(image):

    image = crop_retina(image)

    image = resize_with_padding(
        image,
        IMAGE_SIZE
    )

    image = apply_clahe(image)

    return image


# ============================================================
# Main
# ============================================================

def main():

    # Both training and validation IDs
    train_df = pd.read_csv(
        "dataset/aptos/train_split.csv"
    )

    val_df = pd.read_csv(
        "dataset/aptos/val_split.csv"
    )

    df = pd.concat(
        [train_df, val_df],
        ignore_index=True
    )

    total = len(df)

    print("=" * 60)
    print("CACHING APTOS PREPROCESSING")
    print("=" * 60)

    print(
        f"Images to process: {total}"
    )

    print(
        f"Output directory: {OUTPUT_DIR}"
    )

    processed = 0
    skipped = 0

    for _, row in df.iterrows():

        image_id = row["id_code"]

        input_path = os.path.join(
            IMAGE_DIR,
            image_id + ".png"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            image_id + ".png"
        )

        # Skip if already processed
        if os.path.exists(output_path):

            skipped += 1
            processed += 1
            continue

        image = cv2.imread(
            input_path
        )

        if image is None:

            print(
                f"WARNING: Could not read {input_path}"
            )

            continue

        image = preprocess(
            image
        )

        success = cv2.imwrite(
            output_path,
            image
        )

        if not success:

            print(
                f"WARNING: Could not save {output_path}"
            )

            continue

        processed += 1

        # Progress
        if processed % 100 == 0:

            print(
                f"Processed "
                f"{processed}/{total}"
            )

    print("\n" + "=" * 60)
    print("PREPROCESSING CACHE COMPLETE")
    print("=" * 60)

    print(
        f"Total images: {total}"
    )

    print(
        f"Processed/available: {processed}"
    )

    print(
        f"Already cached: {skipped}"
    )

    print(
        f"\nCached images are in:"
        f"\n{OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
    