"""FastAPI service for the persisted UCI 697 student-outcome model."""

from __future__ import annotations

import json
import os
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, ConfigDict, Field

ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = ROOT / "models" / "student_outcome_pipeline.joblib"
METRICS_PATH = ROOT / "results" / "uci697_metrics.json"
METADATA_PATH = ROOT / "models" / "model_metadata.json"
LABELS = ["Dropout", "Enrolled", "Graduate"]

model: Any = None
metadata: dict[str, Any] = {}
metrics: dict[str, Any] = {}
history: list[dict[str, Any]] = []
database_url = os.getenv("DATABASE_URL")


class PredictionRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    features: dict[str, float | int] = Field(..., description="Admission-time feature values")


class PredictionResponse(BaseModel):
    prediction: str
    confidence: float
    probabilities: dict[str, float]
    model: str
    created_at: datetime


def _load_artifacts() -> None:
    global model, metadata, metrics
    if not MODEL_PATH.exists() or not METADATA_PATH.exists() or not METRICS_PATH.exists():
        raise RuntimeError("Model artifacts are missing. Run scripts/train_uci697.py first.")
    model = joblib.load(MODEL_PATH)
    metadata = json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    metrics = json.loads(METRICS_PATH.read_text(encoding="utf-8"))


def _validate_features(features: dict[str, float | int]) -> None:
    expected = metadata.get("features", [])
    received = set(features)
    missing = sorted(set(expected) - received)
    unexpected = sorted(received - set(expected))
    if missing or unexpected:
        detail = {"missing": missing, "unexpected": unexpected}
        raise HTTPException(status_code=422, detail=detail)


def _persist_prediction(record: dict[str, Any], features: dict[str, float | int]) -> None:
    """Persist history when PostgreSQL is configured; local demos remain stateless."""
    if not database_url:
        return
    import psycopg
    with psycopg.connect(database_url) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS prediction_history (
                id BIGSERIAL PRIMARY KEY, created_at TIMESTAMPTZ NOT NULL,
                predicted_class TEXT NOT NULL, confidence DOUBLE PRECISION NOT NULL,
                probabilities JSONB NOT NULL, features JSONB NOT NULL, model_name TEXT NOT NULL
            )"""
        )
        connection.execute(
            "INSERT INTO prediction_history (created_at, predicted_class, confidence, probabilities, features, model_name) VALUES (%s, %s, %s, %s, %s, %s)",
            (record["created_at"], record["prediction"], record["confidence"],
             json.dumps(record["probabilities"]), json.dumps(features), record["model"]),
        )
        connection.commit()


@asynccontextmanager
async def lifespan(_: FastAPI):
    _load_artifacts()
    yield


app = FastAPI(
    title="Student Outcome Prediction API",
    version="1.0.0",
    description="Admission-time predictions for Dropout, Enrolled, and Graduate outcomes.",
    lifespan=lifespan,
)
origins = [origin.strip() for origin in os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",") if origin.strip()]
app.add_middleware(
    CORSMiddleware, allow_origins=origins, allow_credentials=False,
    allow_methods=["GET", "POST"], allow_headers=["Content-Type"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "model_loaded": str(model is not None).lower()}


@app.get("/api/v1/metrics")
def get_metrics() -> dict[str, Any]:
    return metrics


@app.get("/api/v1/dataset")
def get_dataset_summary() -> dict[str, Any]:
    return metrics.get("dataset", {})


@app.post("/api/v1/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    _validate_features(request.features)
    import pandas as pd
    frame = pd.DataFrame([request.features], columns=metadata["features"])
    probabilities = model.predict_proba(frame)[0]
    prediction = str(model.classes_[int(probabilities.argmax())])
    created_at = datetime.now(timezone.utc)
    record = {
        "prediction": prediction, "confidence": float(probabilities.max()),
        "probabilities": {str(label): float(probability) for label, probability in zip(model.classes_, probabilities)},
        "model": str(metadata.get("model", "unknown")), "created_at": created_at,
    }
    history.append(record)
    _persist_prediction(record, request.features)
    return PredictionResponse(**record)


@app.get("/api/v1/predictions")
def get_predictions(limit: int = Query(default=50, ge=1, le=200)) -> list[dict[str, Any]]:
    if database_url:
        import psycopg
        with psycopg.connect(database_url) as connection:
            rows = connection.execute(
                "SELECT created_at, predicted_class, confidence, probabilities, model_name FROM prediction_history ORDER BY created_at DESC LIMIT %s",
                (limit,),
            ).fetchall()
            return [
                {"created_at": row[0], "prediction": row[1], "confidence": row[2],
                 "probabilities": row[3], "model": row[4]}
                for row in rows
            ]
    return history[-limit:][::-1]
