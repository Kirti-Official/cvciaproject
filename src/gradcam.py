import os
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms

from cbam_model import EfficientNetCBAM


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/efficientnet_b0_cbam_best.pth"

IMAGE_ID = "000c1434d8d7"

IMAGE_PATH = (
    f"dataset/aptos/preprocessed/{IMAGE_ID}.png"
)

OUTPUT_PATH = (
    f"outputs/plots/gradcam_{IMAGE_ID}.png"
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

model = EfficientNetCBAM(num_classes=5)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model.to(DEVICE)
model.eval()

print("Model loaded successfully.")


# ============================================================
# GRAD-CAM HOOKS
# ============================================================

activations = None
gradients = None


def forward_hook(module, input, output):
    global activations
    activations = output


def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]


# Hook the CBAM output
target_layer = model.cbam

forward_handle = target_layer.register_forward_hook(
    forward_hook
)

backward_handle = target_layer.register_full_backward_hook(
    backward_hook
)


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

input_tensor.requires_grad_(True)


# ============================================================
# FORWARD PASS
# ============================================================

output = model(input_tensor)

probabilities = F.softmax(
    output,
    dim=1
)[0]

predicted_class = torch.argmax(
    probabilities
).item()

confidence = probabilities[
    predicted_class
].item()


print()
print("=" * 60)
print("PREDICTION")
print("=" * 60)

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
    f"{confidence * 100:.2f}%"
)


# ============================================================
# BACKWARD PASS
# ============================================================

model.zero_grad()

target = output[
    0,
    predicted_class
]

target.backward()


# ============================================================
# CALCULATE GRAD-CAM
# ============================================================

# Activations:
# [1, channels, height, width]

# Gradients:
# [1, channels, height, width]

weights = gradients.mean(
    dim=(2, 3),
    keepdim=True
)

cam = (
    weights * activations
).sum(dim=1).squeeze()

cam = F.relu(cam)

cam = cam.detach().cpu().numpy()


# ============================================================
# NORMALIZE CAM
# ============================================================

cam = cam - cam.min()

if cam.max() > 0:
    cam = cam / cam.max()

cam = cv2.resize(
    cam,
    (image_rgb.shape[1], image_rgb.shape[0])
)


# ============================================================
# CREATE HEATMAP
# ============================================================

heatmap = np.uint8(
    255 * cam
)

heatmap = cv2.applyColorMap(
    heatmap,
    cv2.COLORMAP_JET
)

heatmap = cv2.cvtColor(
    heatmap,
    cv2.COLOR_BGR2RGB
)


# ============================================================
# OVERLAY
# ============================================================

overlay = (
    0.55 * image_rgb +
    0.45 * heatmap
)

overlay = np.clip(
    overlay,
    0,
    255
).astype(np.uint8)


# ============================================================
# DISPLAY
# ============================================================

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)

plt.imshow(image_rgb)

plt.title("Preprocessed Retinal Image")

plt.axis("off")


plt.subplot(1, 3, 2)

plt.imshow(heatmap)

plt.title("Grad-CAM Heatmap")

plt.axis("off")


plt.subplot(1, 3, 3)

plt.imshow(overlay)

plt.title(
    f"{CLASS_NAMES[predicted_class]}\n"
    f"Confidence: {confidence * 100:.2f}%"
)

plt.axis("off")


plt.tight_layout()

plt.savefig(
    OUTPUT_PATH,
    dpi=200,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# CLEANUP
# ============================================================

forward_handle.remove()
backward_handle.remove()

print()
print("=" * 60)
print("GRAD-CAM COMPLETE")
print("=" * 60)

print(
    "Saved to:",
    OUTPUT_PATH
)