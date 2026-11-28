from __future__ import annotations

import os
from typing import Any

import mlflow
import mlflow.pyfunc
from fastapi import FastAPI, HTTPException
from mlflow.tracking import MlflowClient
from pydantic import BaseModel

app = FastAPI(title="BugLens Inference API", version="0.1.0")
_MODEL = None
_MODEL_URI: str | None = None


def resolve_champion_uri(model_name: str = "buglens", alias: str = "champion") -> str:
    client = MlflowClient()
    mv = client.get_model_version_by_alias(model_name, alias)
    return f"models:/{model_name}/{mv.version}"


class PredictRequest(BaseModel):
    inputs: list[dict[str, Any]]


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/ready")
def ready() -> dict[str, Any]:
    return {"ready": _MODEL is not None, "model_uri": _MODEL_URI}


@app.post("/predict")
def predict(req: PredictRequest) -> dict[str, Any]:
    global _MODEL, _MODEL_URI
    if _MODEL is None:
        uri = os.getenv("BUGLENS_MODEL_URI")
        if not uri:
            raise HTTPException(status_code=503, detail="Model not loaded; set BUGLENS_MODEL_URI")
        _MODEL = mlflow.pyfunc.load_model(uri)
        _MODEL_URI = uri
    preds = _MODEL.predict(req.inputs)
    return {
        "model_uri": _MODEL_URI,
        "predictions": preds.tolist() if hasattr(preds, "tolist") else list(preds),
    }
