import torch
from torch.utils.data import DataLoader
import mlflow
from sklearn.metrics import confusion_matrix, classification_report, f1_score

from src.dataset import load_data
from src.model import CNCFaultDetector

def evaluate():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    
    # Load test split only
    _, _, test_dataset, _ = load_data()
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

    # Init architecture and load weights
    model = CNCFaultDetector(hidden_sizes=[64, 32, 16]).to(device)
    model.load_state_dict(torch.load("model.pt", weights_only=True))
    model.eval()

    y_true = []
    y_pred = []

    with torch.no_grad():
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)
            logits = model(features)
            
            # Binary classification threshold
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Metric Calculation
    cm = confusion_matrix(y_true, y_pred)
    report = classification_report(y_true, y_pred, target_names=['Healthy', 'Failure'])
    f1 = f1_score(y_true, y_pred)

    print(f"\nDevice: {device}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(report)

    with mlflow.start_run(run_name="test_evaluation"):
        mlflow.log_metric("test_f1", f1)
        # Log confusion matrix as text for persistence
        with open("eval_cm.txt", "w") as f:
            f.write(str(cm))
        mlflow.log_artifact("eval_cm.txt")

if __name__ == "__main__":
    evaluate()