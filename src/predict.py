import os
import cv2
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from torchvision import transforms

from cbam_model import EfficientNetCBAM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_b0_cbam_best.pth"

IMAGE_ID = "000c1434d8d7"   # Change this to another APTOS image if needed

IMAGE_PATH = (
    f"dataset/aptos/preprocessed/{IMAGE_ID}.png"
)

DEVICE = torch.device("cpu")

CLASS_NAMES = [
    "No DR",
    "Mild DR",
    "Moderate DR",
    "Severe DR",
    "Proliferative DR"
]


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 60)
print("LOADING CBAM MODEL")
print("=" * 60)

model = EfficientNetCBAM(
    num_classes=5
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")
print("Device:", DEVICE)


# ============================================================
# LOAD IMAGE
# ============================================================

if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(
        f"Image not found: {IMAGE_PATH}"
    )

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise ValueError(
        f"Could not read image: {IMAGE_PATH}"
    )

image_rgb = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# PREPARE INPUT
# ============================================================

input_tensor = transform(image_rgb)

input_tensor = input_tensor.unsqueeze(0)

input_tensor = input_tensor.to(DEVICE)


# ============================================================
# PREDICTION
# ============================================================

with torch.no_grad():

    outputs = model(input_tensor)

    probabilities = F.softmax(
        outputs,
        dim=1
    )[0]

    predicted_class = torch.argmax(
        probabilities
    ).item()


# ============================================================
# DISPLAY RESULTS
# ============================================================

print()
print("=" * 60)
print("DR PREDICTION")
print("=" * 60)

print("Image ID:", IMAGE_ID)

print(
    "Predicted Class:",
    predicted_class
)

print(
    "Predicted Stage:",
    CLASS_NAMES[predicted_class]
)

print(
    "Confidence:",
    f"{probabilities[predicted_class].item() * 100:.2f}%"
)

print()
print("Class Probabilities:")

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{i} - {class_name:<20}: "
        f"{probabilities[i].item() * 100:.2f}%"
    )

print("=" * 60)


# ============================================================
# VISUALIZATION
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(image_rgb)

plt.title(
    f"Prediction: {CLASS_NAMES[predicted_class]}\n"
    f"Confidence: "
    f"{probabilities[predicted_class].item() * 100:.2f}%"
)

plt.axis("off")

plt.tight_layout()

plt.show()