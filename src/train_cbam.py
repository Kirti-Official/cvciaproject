import os
import random
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.optim import AdamW
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    cohen_kappa_score
)

from dataset import train_loader, val_loader
from cbam_model import EfficientNetCBAM


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 5
EPOCHS = 5
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
SEED = 42

MODEL_PATH = "models/efficientnet_b0_cbam_best.pth"
PLOT_DIR = "outputs/plots"

os.makedirs("models", exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


# ============================================================
# Device
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 60)
print("DEVICE INFORMATION")
print("=" * 60)
print("Device:", device)

if device.type == "cuda":
    print("GPU:", torch.cuda.get_device_name(0))
else:
    print("Running on CPU")


# ============================================================
# Class weights
# ============================================================

class_weights = torch.tensor(
    [
        0.4057,
        1.9791,
        0.7332,
        3.8039,
        2.4822
    ],
    dtype=torch.float32
).to(device)

print("\nClass weights:")

for i, weight in enumerate(class_weights):
    print(
        f"Class {i}: {weight.item():.4f}"
    )


# ============================================================
# Create model
# ============================================================

print("\n" + "=" * 60)
print("CREATING EFFICIENTNET-B0 + CBAM")
print("=" * 60)

model = EfficientNetCBAM(
    num_classes=NUM_CLASSES
)

model = model.to(device)

print("Model created successfully.")


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Training function
# ============================================================

def train_one_epoch():

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    loss = running_loss / total
    accuracy = correct / total

    return loss, accuracy


# ============================================================
# Validation function
# ============================================================

def validate():

    model.eval()

    running_loss = 0.0

    all_labels = []
    all_predictions = []

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

    val_loss = (
        running_loss /
        len(val_loader.dataset)
    )

    val_accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    val_f1 = f1_score(
        all_labels,
        all_predictions,
        average="macro",
        zero_division=0
    )

    val_kappa = cohen_kappa_score(
        all_labels,
        all_predictions,
        weights="quadratic"
    )

    return (
        val_loss,
        val_accuracy,
        val_f1,
        val_kappa,
        all_labels,
        all_predictions
    )


# ============================================================
# Training loop
# ============================================================

train_losses = []
val_losses = []

train_accuracies = []
val_accuracies = []

best_val_f1 = -1.0
best_epoch = 0


print("\n" + "=" * 60)
print("STARTING CBAM TRAINING")
print("=" * 60)


for epoch in range(EPOCHS):

    print(
        f"\nEpoch {epoch + 1}/{EPOCHS}"
    )

    # Training
    train_loss, train_accuracy = (
        train_one_epoch()
    )

    # Validation
    (
        val_loss,
        val_accuracy,
        val_f1,
        val_kappa,
        _,
        _
    ) = validate()

    # Store metrics
    train_losses.append(train_loss)
    val_losses.append(val_loss)

    train_accuracies.append(
        train_accuracy
    )

    val_accuracies.append(
        val_accuracy
    )

    # Print metrics
    print(
        f"Train Loss: {train_loss:.4f}"
    )

    print(
        f"Train Accuracy: "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Validation Loss: "
        f"{val_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Validation Macro F1: "
        f"{val_f1:.4f}"
    )

    print(
        f"Validation QWK: "
        f"{val_kappa:.4f}"
    )

    # Save best checkpoint
    if val_f1 > best_val_f1:

        best_val_f1 = val_f1
        best_epoch = epoch + 1

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_f1": val_f1,
                "val_accuracy": val_accuracy,
                "val_kappa": val_kappa
            },
            MODEL_PATH
        )

        print(
            f"Best model saved → {MODEL_PATH}"
        )


# ============================================================
# Load BEST checkpoint
# ============================================================

print("\n" + "=" * 60)
print("LOADING BEST CHECKPOINT")
print("=" * 60)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    f"Best epoch: {checkpoint['epoch']}"
)

print(
    f"Best Macro F1: "
    f"{checkpoint['val_f1']:.4f}"
)

print(
    f"Best Accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)

print(
    f"Best QWK: "
    f"{checkpoint['val_kappa']:.4f}"
)


# ============================================================
# Final evaluation of BEST model
# ============================================================

(
    final_val_loss,
    final_val_accuracy,
    final_val_f1,
    final_val_kappa,
    val_labels,
    val_predictions
) = validate()


# ============================================================
# Classification report
# ============================================================

print("\n" + "=" * 60)
print("FINAL CBAM VALIDATION RESULTS")
print("=" * 60)

class_names = [
    "No DR",
    "Mild",
    "Moderate",
    "Severe",
    "Proliferative"
]

print(
    classification_report(
        val_labels,
        val_predictions,
        target_names=class_names,
        digits=4,
        zero_division=0
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    val_labels,
    val_predictions
)

print("Confusion Matrix:")
print(cm)


# ============================================================
# Save confusion matrix
# ============================================================

plt.figure(figsize=(7, 6))

plt.imshow(
    cm,
    interpolation="nearest"
)

plt.title(
    "EfficientNet-B0 + CBAM Confusion Matrix"
)

plt.colorbar()

plt.xticks(
    range(NUM_CLASSES),
    class_names,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(NUM_CLASSES),
    class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")

for i in range(NUM_CLASSES):

    for j in range(NUM_CLASSES):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "cbam_confusion_matrix.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Loss curve
# ============================================================

epochs_range = range(
    1,
    EPOCHS + 1
)

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_losses,
    marker="o",
    label="Training Loss"
)

plt.plot(
    epochs_range,
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")
plt.ylabel("Loss")

plt.title(
    "EfficientNet-B0 + CBAM Loss"
)

plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "cbam_loss_curve.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Accuracy curve
# ============================================================

plt.figure(figsize=(8, 5))

plt.plot(
    epochs_range,
    train_accuracies,
    marker="o",
    label="Training Accuracy"
)

plt.plot(
    epochs_range,
    val_accuracies,
    marker="o",
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")

plt.title(
    "EfficientNet-B0 + CBAM Accuracy"
)

plt.legend()
plt.grid(True)

plt.savefig(
    os.path.join(
        PLOT_DIR,
        "cbam_accuracy_curve.png"
    ),
    dpi=200,
    bbox_inches="tight"
)

plt.close()


# ============================================================
# Final summary
# ============================================================

print("\n" + "=" * 60)
print("CBAM TRAINING COMPLETED")
print("=" * 60)

print(
    f"Best Epoch: {best_epoch}"
)

print(
    f"Best Validation Accuracy: "
    f"{final_val_accuracy:.4f}"
)

print(
    f"Best Validation Macro F1: "
    f"{final_val_f1:.4f}"
)

print(
    f"Best Validation QWK: "
    f"{final_val_kappa:.4f}"
)

print(
    f"\nModel saved at:\n{MODEL_PATH}"
)

print(
    f"\nPlots saved in:\n{PLOT_DIR}"
)