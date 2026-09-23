import os
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

# -----------------------------
# Paths
# -----------------------------
CSV_PATH = "dataset/aptos/train_split.csv"
IMAGE_DIR = "dataset/aptos/train_images"

# -----------------------------
# Load training split
# -----------------------------
df = pd.read_csv(CSV_PATH)

# -----------------------------
# Class names
# -----------------------------
class_names = {
    0: "No DR",
    1: "Mild",
    2: "Moderate",
    3: "Severe",
    4: "Proliferative"
}

# -----------------------------
# Select one image per class
# -----------------------------
samples = []

for class_id in range(5):
    sample = df[df["diagnosis"] == class_id].iloc[0]
    samples.append(sample)

# -----------------------------
# Plot
# -----------------------------
fig, axes = plt.subplots(1, 5, figsize=(20, 5))

for ax, sample in zip(axes, samples):

    image_id = sample["id_code"]
    label = int(sample["diagnosis"])

    image_path = os.path.join(
        IMAGE_DIR,
        image_id + ".png"
    )

    image = Image.open(image_path)

    ax.imshow(image)
    ax.set_title(
        f"Class {label}\n{class_names[label]}",
        fontsize=11
    )
    ax.axis("off")

plt.tight_layout()

# -----------------------------
# Save figure
# -----------------------------
os.makedirs("outputs/plots", exist_ok=True)

output_path = "outputs/plots/class_samples.png"

plt.savefig(
    output_path,
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print(f"\nSaved visualization to: {output_path}")