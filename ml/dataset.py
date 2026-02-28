from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from pathlib import Path
import cv2

def build_splits(data_path: str | Path, train_frac=0.7, val_frac=0.15, test_frac=0.15, random_state=42):
    data_path = Path(data_path)
    labels_df = pd.read_csv(data_path / "labels.csv")

    train_parts, val_parts, test_parts = [], [], []
    for _, class_df in labels_df.groupby("class_id"):
        class_df = class_df.sample(frac=1, random_state=random_state).reset_index(drop=True)
        n = len(class_df)
        train_end = int(train_frac * n)
        val_end   = int((train_frac + val_frac) * n)
        train_parts.append(class_df.iloc[:train_end])
        val_parts.append(class_df.iloc[train_end:val_end])
        test_parts.append(class_df.iloc[val_end:])

    return (
        pd.concat(train_parts, ignore_index=True),
        pd.concat(val_parts,   ignore_index=True),
        pd.concat(test_parts,  ignore_index=True),
    )


class ShurikodeDataset(Dataset):
    def __init__(self, data_path: str | Path, dataframe: pd.DataFrame, transform=None):
        self.data_path = Path(data_path)
        self.transform = transform
        self.filenames = dataframe["filename"].tolist()
        self.labels    = dataframe["class_id"].tolist()

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_path = self.data_path / self.filenames[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]