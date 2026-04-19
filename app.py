from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from src.predict import CNCPredictor

app = FastAPI(title="CNC Fault Detector API", docs_url="/docs")

# Load model once at startup — not on every request
predictor = CNCPredictor()


class SensorInput(BaseModel):
    air_temp: float
    process_temp: float
    rpm: float
    torque: float
    tool_wear: float


@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8" />
        <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
        <title>CNC Fault Detector</title>
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }

            body {
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
                background: #0f1117;
                color: #e0e0e0;
                min-height: 100vh;
                display: flex;
                align-items: center;
                justify-content: center;
                padding: 2rem;
            }

            .card {
                background: #1a1d27;
                border: 1px solid #2a2d3a;
                border-radius: 16px;
                padding: 3rem;
                max-width: 560px;
                width: 100%;
                text-align: center;
            }

            .badge {
                display: inline-block;
                background: #0d7377;
                color: #fff;
                font-size: 0.75rem;
                font-weight: 600;
                letter-spacing: 0.1em;
                text-transform: uppercase;
                padding: 0.3rem 0.9rem;
                border-radius: 999px;
                margin-bottom: 1.5rem;
            }

            h1 {
                font-size: 1.9rem;
                font-weight: 700;
                color: #ffffff;
                margin-bottom: 0.5rem;
                line-height: 1.2;
            }

            .subtitle {
                font-size: 0.95rem;
                color: #888;
                margin-bottom: 2rem;
                line-height: 1.6;
            }

            .divider {
                border: none;
                border-top: 1px solid #2a2d3a;
                margin: 2rem 0;
            }

            .author {
                font-size: 1rem;
                color: #ccc;
                margin-bottom: 0.4rem;
            }

            .author span {
                color: #ffffff;
                font-weight: 600;
            }

            .links {
                display: flex;
                justify-content: center;
                gap: 1rem;
                margin-top: 1.5rem;
                flex-wrap: wrap;
            }

            .btn {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.55rem 1.2rem;
                border-radius: 8px;
                font-size: 0.875rem;
                font-weight: 500;
                text-decoration: none;
                transition: opacity 0.2s;
            }

            .btn:hover { opacity: 0.8; }

            .btn-linkedin {
                background: #0A66C2;
                color: #fff;
            }

            .btn-docs {
                background: #2a2d3a;
                color: #e0e0e0;
                border: 1px solid #3a3d4a;
            }

            .stack {
                display: flex;
                justify-content: center;
                gap: 0.5rem;
                flex-wrap: wrap;
                margin-top: 2rem;
            }

            .tag {
                background: #12151f;
                border: 1px solid #2a2d3a;
                color: #888;
                font-size: 0.75rem;
                padding: 0.25rem 0.7rem;
                border-radius: 6px;
            }

            .status {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                font-size: 0.8rem;
                color: #4ade80;
                margin-top: 1.5rem;
            }

            .dot {
                width: 8px;
                height: 8px;
                background: #4ade80;
                border-radius: 50%;
                animation: pulse 2s infinite;
            }

            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.4; }
            }
        </style>
    </head>
    <body>
        <div class="card">
            <div class="badge">Industry 4.0 · Smart Manufacturing</div>

            <h1>CNC Fault Detector</h1>
            <p class="subtitle">
                Real-time CNC machine fault detection using a PyTorch neural network
                trained on sensor data — predicting tool failure before it happens.
            </p>

            <hr class="divider"/>

            <p class="author">Built by <span>Mathew Prasanth, PE</span></p>

            <div class="links">
                <a class="btn btn-linkedin"
                   href="https://www.linkedin.com/in/mathewprasanth/"
                   target="_blank">
                    LinkedIn
                </a>
                <a class="btn btn-docs" href="/docs">
                    API Docs →
                </a>
            </div>

            <div class="stack">
                <span class="tag">PyTorch</span>
                <span class="tag">FastAPI</span>
                <span class="tag">MLflow</span>
                <span class="tag">Docker</span>
                <span class="tag">AWS EC2</span>
            </div>

            <div class="status">
                <div class="dot"></div>
                Model loaded and ready
            </div>
        </div>
    </body>
    </html>
    """


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "CNCFaultDetector",
        "version": "1.0.0"
    }


@app.post("/predict")
def predict(data: SensorInput):
    features = [
        data.air_temp,
        data.process_temp,
        data.rpm,
        data.torque,
        data.tool_wear,
    ]
    return predictor.predict(features)