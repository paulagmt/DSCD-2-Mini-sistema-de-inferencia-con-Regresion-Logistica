"""Pruebas mínimas de la API y de las validaciones solicitadas."""

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def valid_payload() -> dict:
    """Payload base válido que reutilizamos en los tests."""

    return {
        "age": 41,
        "job": "technician",
        "marital": "married",
        "education": "secondary",
        "balance": 3200,
        "housing": "yes",
        "loan": "no",
        "campaign": 2,
    }


def test_health_endpoint():
    response = client.get("/api/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_invalid_age_is_rejected():
    # Caso solicitado por la actividad: valor semánticamente inválido.
    payload = valid_payload()
    payload["age"] = -10

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_age_type_is_rejected():
    # Caso solicitado por la actividad: tipo incorrecto.
    payload = valid_payload()
    payload["age"] = "hola"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422


def test_invalid_job_is_rejected():
    # Demuestra que también se controlan categorías fuera del catálogo.
    payload = valid_payload()
    payload["job"] = "not-a-job"

    response = client.post("/predict", json=payload)
    assert response.status_code == 422
