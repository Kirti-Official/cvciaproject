import torch
import torch.nn as nn


# ============================================================
# Channel Attention
# ============================================================

class ChannelAttention(nn.Module):

    def __init__(self, channels, reduction=16):

        super().__init__()

        hidden_channels = max(
            channels // reduction,
            1
        )

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(
                channels,
                hidden_channels,
                kernel_size=1,
                bias=False
            ),

            nn.ReLU(),

            nn.Conv2d(
                hidden_channels,
                channels,
                kernel_size=1,
                bias=False
            )
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avg_attention = self.mlp(
            self.avg_pool(x)
        )

        max_attention = self.mlp(
            self.max_pool(x)
        )

        attention = (
            avg_attention +
            max_attention
        )

        attention = self.sigmoid(
            attention
        )

        return x * attention


# ============================================================
# Spatial Attention
# ============================================================

class SpatialAttention(nn.Module):

    def __init__(self, kernel_size=7):

        super().__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        # Average across channels
        avg_attention = torch.mean(
            x,
            dim=1,
            keepdim=True
        )

        # Maximum across channels
        max_attention, _ = torch.max(
            x,
            dim=1,
            keepdim=True
        )

        # Combine
        combined = torch.cat(
            [
                avg_attention,
                max_attention
            ],
            dim=1
        )

        attention = self.conv(
            combined
        )

        attention = self.sigmoid(
            attention
        )

        return x * attention


# ============================================================
# Complete CBAM
# ============================================================

class CBAM(nn.Module):

    def __init__(
        self,
        channels,
        reduction=16,
        spatial_kernel=7
    ):

        super().__init__()

        self.channel_attention = (
            ChannelAttention(
                channels,
                reduction
            )
        )

        self.spatial_attention = (
            SpatialAttention(
                spatial_kernel
            )
        )

    def forward(self, x):

        # Channel attention
        x = self.channel_attention(x)

        # Spatial attention
        x = self.spatial_attention(x)

        return x


# ============================================================
# Test CBAM
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TESTING CBAM")
    print("=" * 60)

    # Simulated EfficientNet feature map
    batch_size = 2
    channels = 1280
    height = 7
    width = 7

    x = torch.randn(
        batch_size,
        channels,
        height,
        width
    )

    print("\nInput shape:")
    print(x.shape)

    # Create CBAM
    cbam = CBAM(
        channels=channels
    )

    # Forward pass
    output = cbam(x)

    print("\nOutput shape:")
    print(output.shape)

    # Parameter count
    parameters = sum(
        p.numel()
        for p in cbam.parameters()
    )

    print(
        "\nCBAM parameters:",
        parameters
    )

    # Shape verification
    assert output.shape == x.shape

    print("\nCBAM test passed successfully.")