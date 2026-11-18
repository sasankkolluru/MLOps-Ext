from fastapi.testclient import TestClient

from src.api import app

client = TestClient(app)
VALID = {
    "pottery_load_kg": 120,
    "target_temperature_c": 1050,
    "initial_moisture_pct": 8,
    "planned_duration_hours": 10,
    "insulation_rating": 0.8,
    "kiln_volume_m3": 2.5,
    "ambient_temperature_c": 25,
}


def test_input_validation_rejects_negative_load():
    response = client.post("/predict", json={**VALID, "pottery_load_kg": -1})
    assert response.status_code == 422


def test_prediction_is_non_negative():
    response = client.post("/predict", json=VALID)
    assert response.status_code == 200
    assert response.json()["predicted_energy_consumption_kwh"] >= 0
