"""FastAPI serving layer for the trained kiln model."""

from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .data_pipeline import MODEL_FEATURES

MODEL_PATH = Path("artifacts/kiln_energy_model.joblib")
app = FastAPI(title="Ceramic Kiln Energy API", version="1.0.0")


class FiringBatch(BaseModel):
    pottery_load_kg: float = Field(gt=0)
    target_temperature_c: float = Field(gt=0)
    initial_moisture_pct: float = Field(ge=0, le=100)
    planned_duration_hours: float = Field(gt=0)
    insulation_rating: float = Field(gt=0, le=1)
    kiln_volume_m3: float = Field(gt=0)
    ambient_temperature_c: float


@app.post("/predict")
def predict(batch: FiringBatch) -> dict[str, float]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=503, detail="Model is not trained")
    values = batch.model_dump()
    values["thermal_load_index"] = (
        values["pottery_load_kg"]
        * (values["target_temperature_c"] - values["ambient_temperature_c"])
        / 1000
    )
    values["moisture_removal_load"] = (
        values["pottery_load_kg"] * values["initial_moisture_pct"] / 100
    )
    prediction = float(joblib.load(MODEL_PATH).predict(pd.DataFrame([values])[MODEL_FEATURES])[0])
    return {"predicted_energy_consumption_kwh": max(0.0, prediction)}
