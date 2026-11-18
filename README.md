# Ceramic Kiln Energy MLOps

Simple MLOps application for predicting electricity consumption for ceramic firing batches.

## Quick start

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src\train.py
uvicorn src.api:app --reload
```

The training command creates:

- `data/kiln_batches_v1.csv` and `data/kiln_batches_v2.csv`
- `artifacts/feature_store.csv`
- `artifacts/model_comparison.csv`
- `artifacts/registered_model_uri.txt`
- `reports/kiln_energy_report.pdf`

MLflow is started locally by the training script at `mlruns/`. To inspect it:

```powershell
mlflow ui --backend-store-uri .\mlruns
```

## API

```powershell
Invoke-RestMethod http://localhost:8000/predict -Method Post `
  -ContentType "application/json" `
  -Body '{"pottery_load_kg":120,"target_temperature_c":1050,"initial_moisture_pct":8,"planned_duration_hours":10,"insulation_rating":0.8,"kiln_volume_m3":2.5,"ambient_temperature_c":25}'
```

## Project layout

- `src/data_pipeline.py`: synthetic data generation and feature engineering
- `src/train.py`: preprocessing, model comparison, MLflow logging/registration
- `src/api.py`: FastAPI endpoint
- `tests/`: input and non-negative prediction tests
- `.github/workflows/ci.yml`: GitHub Actions workflow
