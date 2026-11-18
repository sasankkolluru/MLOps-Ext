"""Train, compare, and register kiln energy models."""

from pathlib import Path

import joblib
import mlflow
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .data_pipeline import MODEL_FEATURES, TARGET, build_feature_store


def train_models() -> pd.DataFrame:
    data = build_feature_store()
    train, test = train_test_split(data, test_size=0.2, random_state=42)
    numeric = ColumnTransformer([("scale", StandardScaler(), MODEL_FEATURES)])
    candidates = {
        "linear_regression": LinearRegression(),
        "random_forest": RandomForestRegressor(
            n_estimators=150, random_state=42, min_samples_leaf=2
        ),
    }
    Path("artifacts").mkdir(exist_ok=True)
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("ceramic-kiln-energy")
    results = []
    best_name, best_rmse, best_pipeline = "", float("inf"), None
    for name, estimator in candidates.items():
        pipeline = Pipeline([("preprocessor", numeric), ("model", estimator)])
        pipeline.fit(train[MODEL_FEATURES], train[TARGET])
        predictions = pipeline.predict(test[MODEL_FEATURES])
        metrics = {
            "model": name,
            "mae": mean_absolute_error(test[TARGET], predictions),
            "rmse": mean_squared_error(test[TARGET], predictions) ** 0.5,
            "r2": r2_score(test[TARGET], predictions),
        }
        results.append(metrics)
        with mlflow.start_run(run_name=name):
            mlflow.log_params({"model": name, "records": len(data)})
            mlflow.log_metrics({key: value for key, value in metrics.items() if key != "model"})
        if metrics["rmse"] < best_rmse:
            best_name, best_rmse, best_pipeline = name, metrics["rmse"], pipeline

    comparison = pd.DataFrame(results)
    comparison.to_csv("artifacts/model_comparison.csv", index=False)
    joblib.dump(best_pipeline, "artifacts/kiln_energy_model.joblib")
    with mlflow.start_run(run_name=f"registered_{best_name}") as run:
        mlflow.log_params({"selected_model": best_name, "feature_count": len(MODEL_FEATURES)})
        mlflow.log_metrics(
            comparison.set_index("model").loc[best_name].to_dict()
        )
        mlflow.sklearn.log_model(
            best_pipeline,
            "model",
            registered_model_name="CeramicKilnEnergyModel",
        )
        Path("artifacts/registered_model_uri.txt").write_text(
            f"runs:/{run.info.run_id}/model", encoding="utf-8"
        )
    return comparison


if __name__ == "__main__":
    print(train_models().to_string(index=False))
