import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import mlflow
import mlflow.pytorch
import copy

from src.dataset import load_data
from src.model import CNCFaultDetector

# This ensures the script talks to the SAME server your UI is using
mlflow.set_tracking_uri("http://127.0.0.1:5000") 
mlflow.set_experiment("CNC_Fault_Detection")

def train(config=None):
    if config is None:
        config = {
            "epochs": 200,          # Increased from 50
            "batch_size": 64,
            "learning_rate": 5e-4,  # Slightly lower for stability
            "hidden_sizes": [64, 32, 16],
            "dropout_rate": 0.3,
            "patience": 15,         # Early stopping window
            "pos_weight": 15.0      # Reduced from 28 to fix Precision/False Alarms
        }

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    train_dataset, val_dataset, _, _ = load_data()

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config["batch_size"])

    model = CNCFaultDetector(
        hidden_sizes=config["hidden_sizes"],
        dropout_rate=config["dropout_rate"]
    ).to(device)

    # Use the tuned pos_weight from config
    pos_weight = torch.tensor([config["pos_weight"]]).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=config["learning_rate"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Early Stopping state
    best_vloss = float('inf')
    epochs_no_improve = 0
    best_model_state = None

    with mlflow.start_run(run_name="Optimized_CNC_Run"):
        mlflow.log_params(config)

        for epoch in range(config["epochs"]):
            # Training Phase 
            model.train()
            t_loss = 0.0
            for feat, lab in train_loader:
                feat, lab = feat.to(device), lab.to(device)
                optimizer.zero_grad()
                loss = criterion(model(feat), lab)
                loss.backward()
                optimizer.step()
                t_loss += loss.item()

            # Validation Phase
            model.eval()
            v_loss = 0.0
            with torch.no_grad():
                for feat, lab in val_loader:
                    feat, lab = feat.to(device), lab.to(device)
                    v_loss += criterion(model(feat), lab).item()

            v_loss /= len(val_loader)
            t_loss /= len(train_loader)
            scheduler.step(v_loss)

            mlflow.log_metrics({"train_loss": t_loss, "val_loss": v_loss}, step=epoch)

            # Early Stopping Logic
            if v_loss < best_vloss:
                best_vloss = v_loss
                epochs_no_improve = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                epochs_no_improve += 1

            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1:3d} | T-Loss: {t_loss:.4f} | V-Loss: {v_loss:.4f}")

            if epochs_no_improve >= config["patience"]:
                print(f"Early Stopping at epoch {epoch+1}")
                break

        # Load and save the BEST weights found, not the last ones
        model.load_state_dict(best_model_state)
        torch.save(model.state_dict(), "model.pt")
        mlflow.pytorch.log_model(model, "model")
        print("Optimized model saved to model.pt")

    return model

if __name__ == "__main__":
    train()