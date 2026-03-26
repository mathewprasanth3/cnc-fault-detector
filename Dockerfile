# Use slim Python image to keep the container size small
FROM python:3.11-slim

WORKDIR /app

# Copy dependency files first (better layer caching)
COPY pyproject.toml .
COPY uv.lock .

# Install uv then sync dependencies
RUN pip install uv && uv sync --frozen

# Copy the rest of the project
COPY src/__init__.py ./src/
COPY src/model.py ./src/
COPY src/predict.py ./src/
COPY data/scaler.pkl ./data/scaler.pkl
COPY model.pt .
COPY app.py .

# Expose the port FastAPI will run on
EXPOSE 8000

# Start the server
CMD ["uv", "run", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]