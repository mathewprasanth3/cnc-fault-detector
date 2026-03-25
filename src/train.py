import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch

from src.dataset import load_data
from src.model import CNCFaultDetector


def train(config=None):
    if config is None:
        config = {
            "epochs": 50,
            "batch_size": 64,
            "learning_rate": 1e-3,
            "hidden_sizes": [64, 32, 16],
            "dropout_rate": 0.3,
        }

    mlflow.set_tracking_uri("http://127.0.0.1:5000")

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Training on: {device}")

    train_dataset, val_dataset, test_dataset, scaler = load_data()

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    model = CNCFaultDetector(
        hidden_sizes=config["hidden_sizes"],
        dropout_rate=config["dropout_rate"]
    ).to(device)

    y_train = train_dataset.labels.numpy()
    num_failures = y_train.sum()
    num_non_failures = len(y_train) - num_failures
    pos_weight = torch.tensor([num_non_failures / max(num_failures, 1)]).to(device)

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    with mlflow.start_run():
        mlflow.log_params(config)

        for epoch in range(config["epochs"]):
            model.train()
            train_loss = 0.0

            for features, labels in train_loader:
                features, labels = features.to(device), labels.to(device)
                optimizer.zero_grad()
                predictions = model(features)
                loss = criterion(predictions, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            train_loss /= len(train_loader)

            model.eval()
            val_loss = 0.0
            correct = 0
            total = 0

            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    predictions = model(features)
                    val_loss += criterion(predictions, labels).item()
                    predicted_classes = (torch.sigmoid(predictions) > 0.5).float()
                    correct += (predicted_classes == labels).sum().item()
                    total += labels.size(0)

            val_loss /= len(val_loader)
            val_accuracy = correct / total
            scheduler.step(val_loss)

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy
            }, step=epoch)

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
                      f"Train loss: {train_loss:.4f} | "
                      f"Val loss: {val_loss:.4f} | "
                      f"Val acc: {val_accuracy:.4f}")

        torch.save(model.state_dict(), "model.pt")
        mlflow.pytorch.log_model(model, "model")
        print("\nModel saved to model.pt")

    return model, scaler


if __name__ == "__main__":
    train()