import torch
import torch.nn as nn
import timm

from cbam import CBAM


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 5


# ============================================================
# EfficientNet-B0 + CBAM
# ============================================================

class EfficientNetCBAM(nn.Module):

    def __init__(self, num_classes=5):

        super().__init__()

        # EfficientNet-B0 backbone
        self.backbone = timm.create_model(
            "efficientnet_b0",
            pretrained=True,
            num_classes=0,
            global_pool=""
        )

        # EfficientNet-B0 final feature channels
        self.feature_channels = 1280

        # CBAM
        self.cbam = CBAM(
            channels=self.feature_channels,
            reduction=16,
            spatial_kernel=7
        )

        # Global average pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Dropout
        self.dropout = nn.Dropout(
            p=0.2
        )

        # Final classifier
        self.classifier = nn.Linear(
            self.feature_channels,
            num_classes
        )

    def forward(self, x):

        # EfficientNet feature extraction
        x = self.backbone(x)

        # Attention refinement
        x = self.cbam(x)

        # Global average pooling
        x = self.global_pool(x)

        # Flatten
        x = torch.flatten(
            x,
            start_dim=1
        )

        # Dropout
        x = self.dropout(x)

        # Classification
        x = self.classifier(x)

        return x


# ============================================================
# Test
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING EFFICIENTNET-B0 + CBAM")
    print("=" * 60)

    # CPU for testing
    device = torch.device("cpu")

    print("\nCreating model...")

    model = EfficientNetCBAM(
        num_classes=NUM_CLASSES
    )

    model = model.to(device)

    print("Model created successfully.")

    # Dummy input
    x = torch.randn(
        1,
        3,
        224,
        224
    ).to(device)

    print("\nInput shape:")
    print(x.shape)

    # Forward pass
    with torch.no_grad():

        output = model(x)

    print("\nOutput shape:")
    print(output.shape)

    print("\nOutput:")
    print(output)

    # Parameter counts
    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTotal parameters:",
        total_params
    )

    print(
        "Trainable parameters:",
        trainable_params
    )

    # Check expected output
    assert output.shape == (
        1,
        NUM_CLASSES
    )

    print(
        "\nEfficientNet-B0 + CBAM "
        "test passed successfully."
    )