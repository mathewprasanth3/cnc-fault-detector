import torch
import joblib
import numpy as np
from src.model import CNCFaultDetector

class CNCPredictor:
    def __init__(self, model_path="model.pt", scaler_path="data/scaler.pkl"):
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        
        # 1. Load the Scaler (Must be the same one used in training)
        try:
            self.scaler = joblib.load(scaler_path)
        except FileNotFoundError:
            raise Exception(f"Scaler not found at {scaler_path}. Run training first.")

        # 2. Load Model Architecture and Weights
        self.model = CNCFaultDetector(hidden_sizes=[64, 32, 16]).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()

    def predict(self, raw_features):
        """
        raw_features: list or np.array of [Air_temp, Process_temp, RPM, Torque, Tool_wear]
        """
        # Ensure input is 2D for the scaler
        data = np.array(raw_features).reshape(1, -1)
        
        # 3. Pre-process (Scaling)
        scaled_data = self.scaler.transform(data)
        scaled_tensor = torch.tensor(scaled_data, dtype=torch.float32).to(self.device)

        # 4. Inference
        with torch.no_grad():
            logits = self.model(scaled_tensor)
            probability = torch.sigmoid(logits).item()
            
        prediction = 1 if probability > 0.5 else 0
        return {
            "prediction": "FAILURE" if prediction == 1 else "HEALTHY",
            "probability": round(probability, 4),
            "status_code": prediction
        }

if __name__ == "__main__":
    # Example: A row known to be a Failure from the dataset
    # Features: [Air temp, Process temp, Rotational speed, Torque, Tool wear]
    sample_input = [298.9, 309.1, 1439, 46.9, 210]
    
    predictor = CNCPredictor()
    result = predictor.predict(sample_input)
    
    print("\n--- CNC Live Inference Check ---")
    print(f"Input Sensors: {sample_input}")
    print(f"Result: {result['prediction']}")
    print(f"Confidence: {result['probability'] * 100:.2f}%")