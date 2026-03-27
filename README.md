# CNC Fault Detector

Real-time CNC machine fault detection using a PyTorch neural network trained on sensor data — predicting tool failure before it happens.

**Live API** → `http://your-ec2-ip:8000`  
**API Docs** → `http://your-ec2-ip:8000/docs`

---

## What It Does

CNC machines fail silently — a worn tool or thermal anomaly produces a defective part before any operator notices. This system monitors five sensor readings in real time and predicts whether a machining operation is heading toward failure, giving operators time to intervene before damage occurs.

This is a core problem in **smart manufacturing / Industry 4.0** — models like this are consumed by factory dashboards and SCADA systems via REST to enable predictive maintenance.

---

## Tech Stack

| Layer | Technology |
|---|---|
| Model | PyTorch — feedforward MLP with BatchNorm, Dropout |
| Experiment Tracking | MLflow with SQLite backend |
| API | FastAPI + Uvicorn |
| Containerisation | Docker |
| Registry | AWS ECR |
| Hosting | AWS EC2 (t2.micro) |
| Dependency Management | uv |

---

## Dataset

**UCI AI4I 2020 Predictive Maintenance Dataset** — 10,000 rows of simulated CNC sensor data.

| Feature | Description |
|---|---|
| Air Temperature | Ambient air temperature (K) |
| Process Temperature | Machining process temperature (K) |
| Rotational Speed | Spindle RPM |
| Torque | Cutting torque (Nm) |
| Tool Wear | Accumulated tool wear (min) |
| Machine Failure | Target: 0 = Healthy, 1 = Failure |

**Class imbalance:** only 3.3% of rows are failures — the central challenge the training strategy is built around.

---

## Model Architecture

```
Input (5 features)
    → Linear(5 → 64) + BatchNorm + ReLU + Dropout(0.3)
    → Linear(64 → 32) + BatchNorm + ReLU + Dropout(0.3)
    → Linear(32 → 16) + BatchNorm + ReLU + Dropout(0.3)
    → Linear(16 → 1)
    → Sigmoid → probability
```

**Key training decisions:**
- `BCEWithLogitsLoss` with `pos_weight=15.0` — makes missing a failure 15x more costly than a false alarm
- `ReduceLROnPlateau` scheduler — halves learning rate when validation loss stalls
- Early stopping with `patience=15` — stopped at epoch 75 of 200, preserving best checkpoint
- Stratified train/val/test split — maintains 3.3% failure ratio across all sets

**Results:** 88% recall on failure class, ~40% reduction in false alarms vs initial model

---

## Project Structure

```
cnc-fault-detector/
├── app.py              ← FastAPI inference server
├── Dockerfile          ← Container definition
├── pyproject.toml      ← Dependencies (uv)
├── model.pt            ← Trained model weights
├── src/
│   ├── dataset.py      ← ETL pipeline
│   ├── model.py        ← Neural network architecture
│   ├── train.py        ← Training loop + MLflow tracking
│   ├── evaluation.py   ← Test set evaluation
│   └── predict.py      ← Production inference class
└── data/
    └── scaler.pkl      ← Fitted StandardScaler
```

---

## API Endpoints

### `GET /`
Branded landing page — confirms the service is live.

### `GET /health`
```json
{
  "status": "healthy",
  "model": "CNCFaultDetector",
  "version": "1.0.0"
}
```

### `POST /predict`

**Request:**
```json
{
  "air_temp": 301.0,
  "process_temp": 313.0,
  "rpm": 1200,
  "torque": 65.0,
  "tool_wear": 250
}
```

**Response:**
```json
{
  "prediction": "FAILURE",
  "probability": 0.9852,
  "status_code": 1
}
```

`probability` always reflects confidence of the predicted class — not always the failure probability.

---

## Run Locally

**Prerequisites:** Python 3.11+, uv

```bash
# Clone the repo
git clone https://github.com/your-username/cnc-fault-detector.git
cd cnc-fault-detector

# Install dependencies
uv sync

# Start the API
uvicorn app:app --reload
```

Open `http://localhost:8000/docs` to test predictions via Swagger UI.

---

## Run with Docker

```bash
# Build
docker build -t cnc-fault-detector .

# Run
docker run -p 8000:8000 cnc-fault-detector
```

---

## Retrain the Model

```bash
# Start MLflow tracking server
mlflow server --backend-store-uri sqlite:///mlflow.db --host 127.0.0.1 --port 5000

# Train (in a new terminal)
python -m src.train

# Evaluate
python -m src.evaluation
```

View experiment results at `http://127.0.0.1:5000`.

---

## Example curl

```bash
# Failure case
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"air_temp": 301.0, "process_temp": 313.0, "rpm": 1200, "torque": 65.0, "tool_wear": 250}'

# Healthy case
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{"air_temp": 298.0, "process_temp": 308.0, "rpm": 1500, "torque": 35.0, "tool_wear": 50}'
```

---

## Author

**Mathew Prasanth, PE**  
[LinkedIn](https://www.linkedin.com/in/mathewprasanth/) · [Live Demo](http://your-ec2-ip:8000)

AWS Certified Cloud Practitioner · AWS Certified Machine Learning Specialty
