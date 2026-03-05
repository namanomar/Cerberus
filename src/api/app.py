
from __future__ import annotations

import os
import time
from contextlib import asynccontextmanager
from typing import Any, Optional

import uvicorn
import yaml
from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field
from starlette.responses import Response

from src.inference.fraud_predictor import FraudPredictor, FraudPrediction

# ──────────────────────────────────────────────────────────────────────────────
# Prometheus Metrics
# ──────────────────────────────────────────────────────────────────────────────
REQUEST_COUNT = Counter(
    "fraud_api_requests_total",
    "Total prediction requests",
    ["endpoint", "risk_level"],
)
REQUEST_LATENCY = Histogram(
    "fraud_api_latency_seconds",
    "Request latency",
    ["endpoint"],
    buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0],
)
FRAUD_SCORE_GAUGE = Gauge(
    "fraud_api_last_fraud_score",
    "Most recent fraud score produced",
)
FRAUD_ALERTS = Counter(
    "fraud_api_alerts_total",
    "Total HIGH / CRITICAL alerts triggered",
)

# ──────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ──────────────────────────────────────────────────────────────────────────────

class TransactionRequest(BaseModel):
    TransactionID: Optional[int] = None
    TransactionDT: Optional[float] = Field(default=86400.0, description="Seconds since reference epoch")
    TransactionAmt: float = Field(..., gt=0, description="Transaction amount in USD")
    ProductCD: Optional[str] = "W"
    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None

    # Allow arbitrary additional fields (V columns, C columns, etc.)
    model_config = {"extra": "allow"}


class ComponentScores(BaseModel):
    lgbm: float
    graph: float
    anomaly: float


class ShapFeature(BaseModel):
    feature: str
    value: float
    shap_impact: float


class FraudPredictionResponse(BaseModel):
    transaction_id: str
    fraud_score: float = Field(..., ge=0.0, le=1.0)
    risk_level: str = Field(..., pattern="^(LOW|MEDIUM|HIGH|CRITICAL)$")
    component_scores: ComponentScores
    reasons: list[str]
    top_shap_features: list[ShapFeature]
    latency_ms: float
    timestamp: float


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest] = Field(..., max_length=100)


class BatchResponse(BaseModel):
    predictions: list[FraudPredictionResponse]
    total_processed: int
    high_risk_count: int
    avg_fraud_score: float
    batch_latency_ms: float


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    uptime_seconds: float


class ModelInfoResponse(BaseModel):
    lgbm_weight: float
    graph_weight: float
    anomaly_weight: float
    n_features: int
    config_path: str


# ──────────────────────────────────────────────────────────────────────────────
# Application State
# ──────────────────────────────────────────────────────────────────────────────

_predictor: Optional[FraudPredictor] = None
_start_time: float = time.time()
_config_path: str = os.getenv("CONFIG_PATH", "config/config.yaml")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup, clean up on shutdown."""
    global _predictor
    logger.info("Starting Fraud Detection API …")
    try:
        _predictor = FraudPredictor(_config_path)
        _predictor.load_models()
        logger.success("Models loaded successfully.")
    except Exception as e:
        logger.error(f"Failed to load models: {e}")
        logger.warning("API will start but /predict endpoints will return 503.")
    yield
    logger.info("Shutting down Fraud Detection API.")


# ──────────────────────────────────────────────────────────────────────────────
# App Factory
# ──────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    app = FastAPI(
        title="Fraud Detection API",
        description=(
            "Real-time fraud scoring using an ensemble of LightGBM, "
            "Graph embeddings, and Isolation Forest."
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    return app


app = create_app()


# ──────────────────────────────────────────────────────────────────────────────
# Middleware: request timing
# ──────────────────────────────────────────────────────────────────────────────

@app.middleware("http")
async def timing_middleware(request: Request, call_next):
    start = time.perf_counter()
    response = await call_next(request)
    duration = time.perf_counter() - start
    response.headers["X-Process-Time-Ms"] = str(round(duration * 1000, 2))
    return response


# ──────────────────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health():
    return HealthResponse(
        status="healthy" if _predictor is not None else "degraded",
        model_loaded=_predictor is not None,
        uptime_seconds=round(time.time() - _start_time, 1),
    )


@app.get("/metrics", tags=["Monitoring"])
async def prometheus_metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return ModelInfoResponse(
        lgbm_weight=_predictor.lgbm_weight,
        graph_weight=_predictor.graph_weight,
        anomaly_weight=_predictor.anomaly_weight,
        n_features=len(_predictor.feature_names),
        config_path=_config_path,
    )


@app.post(
    "/predict",
    response_model=FraudPredictionResponse,
    status_code=status.HTTP_200_OK,
    tags=["Prediction"],
    summary="Score a single transaction",
)
async def predict_fraud(request: TransactionRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    t0 = time.perf_counter()

    try:
        with REQUEST_LATENCY.labels(endpoint="/predict").time():
            tx_dict = request.model_dump(exclude_none=False)
            prediction: FraudPrediction = _predictor.predict(tx_dict)

    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

    # Update Prometheus metrics
    REQUEST_COUNT.labels(endpoint="/predict", risk_level=prediction.risk_level).inc()
    FRAUD_SCORE_GAUGE.set(prediction.fraud_score)
    if prediction.risk_level in ("HIGH", "CRITICAL"):
        FRAUD_ALERTS.inc()

    return FraudPredictionResponse(**prediction.to_dict())


@app.post(
    "/predict/batch",
    response_model=BatchResponse,
    tags=["Prediction"],
    summary="Score up to 100 transactions in one call",
)
async def predict_batch(request: BatchRequest):
    if _predictor is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")

    t0 = time.perf_counter()

    predictions = []
    for tx in request.transactions:
        tx_dict = tx.model_dump(exclude_none=False)
        pred = _predictor.predict(tx_dict)
        predictions.append(FraudPredictionResponse(**pred.to_dict()))
        REQUEST_COUNT.labels(endpoint="/predict/batch", risk_level=pred.risk_level).inc()

    batch_latency = (time.perf_counter() - t0) * 1000
    high_risk = sum(1 for p in predictions if p.risk_level in ("HIGH", "CRITICAL"))
    avg_score = float(sum(p.fraud_score for p in predictions) / len(predictions))

    return BatchResponse(
        predictions=predictions,
        total_processed=len(predictions),
        high_risk_count=high_risk,
        avg_fraud_score=round(avg_score, 4),
        batch_latency_ms=round(batch_latency, 2),
    )


# ──────────────────────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    with open(_config_path) as f:
        cfg = yaml.safe_load(f)
    api_cfg = cfg.get("api", {})

    uvicorn.run(
        "src.api.app:app",
        host=api_cfg.get("host", "0.0.0.0"),
        port=api_cfg.get("port", 8000),
        workers=api_cfg.get("workers", 1),
        reload=os.getenv("APP_ENV", "development") == "development",
        log_level="info",
    )
