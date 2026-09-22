import os
import cv2
import torch
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


# ============================================================
# PATHS
# ============================================================

PREPROCESSED_DIR = "dataset/aptos/preprocessed"


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(10),
    transforms.ColorJitter(
        brightness=0.15,
        contrast=0.15
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.ToPILImage(),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# DATASET
# ============================================================

class APTOSDataset(Dataset):

    def __init__(self, csv_file, transform=None):

        self.data = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        row = self.data.iloc[idx]

        image_id = row["id_code"]
        label = int(row["diagnosis"])

        image_path = os.path.join(
            PREPROCESSED_DIR,
            image_id + ".png"
        )

        # Read cached preprocessed image
        image = cv2.imread(image_path)

        if image is None:
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        # OpenCV BGR → RGB
        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(
            label,
            dtype=torch.long
        )


# ============================================================
# DATALOADER TEST
# ============================================================

if __name__ == "__main__":

    train_dataset = APTOSDataset(
        "dataset/aptos/train_split.csv",
        transform=train_transform
    )

    val_dataset = APTOSDataset(
        "dataset/aptos/val_split.csv",
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=8,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0
    )

    images, labels = next(iter(train_loader))

    print("=" * 60)
    print("DATASET TEST")
    print("=" * 60)

    print("Train dataset size:", len(train_dataset))
    print("Validation dataset size:", len(val_dataset))

    print("Image batch shape:", images.shape)
    print("Label batch shape:", labels.shape)

    print("Image dtype:", images.dtype)
    print("Label dtype:", labels.dtype)

    print("Image min:", images.min().item())
    print("Image max:", images.max().item())

    print("=" * 60)