import torch
import torch.nn as nn
import timm


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 5


# ============================================================
# EfficientNet-B0 Baseline
# ============================================================

def create_model():

    model = timm.create_model(
        "efficientnet_b0",
        pretrained=True,
        num_classes=NUM_CLASSES
    )

    return model


# ============================================================
# Test model
# ============================================================

if __name__ == "__main__":

    print("Creating EfficientNet-B0...")

    model = create_model()

    print("\nModel created successfully.")

    print("\nClassifier:")
    print(model.classifier)

    # Test input
    dummy_input = torch.randn(
        1,
        3,
        224,
        224
    )

    print("\nRunning test forward pass...")

    with torch.no_grad():
        output = model(dummy_input)

    print("\nInput shape:")
    print(dummy_input.shape)

    print("\nOutput shape:")
    print(output.shape)

    print("\nOutput:")
    print(output)

    # Parameter count
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print("\nTotal parameters:", total_params)
    print("Trainable parameters:", trainable_params)

    print("\nEfficientNet-B0 test completed successfully.")