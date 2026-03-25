import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


class CNCDataset(Dataset):
    def __init__(self, features, labels):
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


def load_data(path="data/ai4i2020.csv", test_size=0.2, val_size=0.1, random_state=42):
    df = pd.read_csv(path)

    # these columns leak information about the target or are irrelevant identifiers
    cols_to_drop = ["UDI", "Product ID", "Type", "TWF", "HDF", "PWF", "OSF", "RNF"]
    df = df.drop(columns=cols_to_drop)

    X = df.drop(columns=["Machine failure"]).values
    y = df["Machine failure"].values

    # split into train, validation, and test sets before scaling
    # scaling after splitting prevents data leakage from test set into training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train, test_size=val_size, random_state=random_state, stratify=y_train
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    train_dataset = CNCDataset(X_train, y_train)
    val_dataset = CNCDataset(X_val, y_val)
    test_dataset = CNCDataset(X_test, y_test)

    return train_dataset, val_dataset, test_dataset, scaler


if __name__ == "__main__":
    train, val, test, scaler = load_data()
    print(f"Train size: {len(train)}")
    print(f"Val size:   {len(val)}")
    print(f"Test size:  {len(test)}")
    print(f"Feature shape: {train[0][0].shape}")