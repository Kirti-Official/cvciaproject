import pandas as pd
from sklearn.model_selection import train_test_split

# Paths
CSV_PATH = "dataset/aptos/train.csv"

# Load dataset
df = pd.read_csv(CSV_PATH)

print("Original dataset:")
print(df["diagnosis"].value_counts().sort_index())

# Stratified 80/20 split
train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    stratify=df["diagnosis"],
    random_state=42
)

# Reset indexes
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Save splits
train_df.to_csv("dataset/aptos/train_split.csv", index=False)
val_df.to_csv("dataset/aptos/val_split.csv", index=False)

print("\nTraining set:", len(train_df))
print("Validation set:", len(val_df))

print("\nTraining distribution:")
print(train_df["diagnosis"].value_counts().sort_index())

print("\nValidation distribution:")
print(val_df["diagnosis"].value_counts().sort_index())