"""Aplicación FastAPI.

La responsabilidad de este archivo es exponer la inferencia mediante HTTP.
No contiene el entrenamiento del modelo.
"""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.database import init_db, list_recent_predictions, save_prediction
from app.inference import predict
from app.schemas import BankMarketingFeatures, PredictionResponse

# Creamos la aplicación.
app = FastAPI(title="Bank Marketing Predictor API", version="1.0.0")

# Permitimos que el frontend (puerto 9000) se comunique con la API (9001).
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / "static"

# Hace disponibles los archivos estáticos del frontend.
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.on_event("startup")
def startup() -> None:
    """Inicializa la base de datos auxiliar al arrancar la API."""

    init_db()


@app.get("/")
def home() -> FileResponse:
    """Sirve el frontend si se entra directamente al puerto 9001."""

    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/health")
def health() -> dict:
    """Endpoint sencillo para comprobar que la API está viva."""

    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
@app.post("/api/predict", response_model=PredictionResponse, include_in_schema=False)
def predict_bank_marketing(features: BankMarketingFeatures) -> PredictionResponse:
    """Recibe datos, valida, ejecuta inferencia y devuelve el resultado."""

    try:
        # Pydantic ya validó los tipos y rangos antes de entrar aquí.
        payload = features.model_dump()

        # Delegamos la inferencia a app/inference.py.
        # El endpoint no entrena ni transforma manualmente los datos.
        result = predict(payload)

        # Esto es una funcionalidad heredada de la plantilla y es opcional.
        save_prediction(payload, result)

        return PredictionResponse(**result)

    except FileNotFoundError as exc:
        # Si todavía no existe el .joblib, devolvemos un error entendible.
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    except Exception as exc:
        # Cualquier error inesperado de inferencia se devuelve controladamente.
        raise HTTPException(
            status_code=500,
            detail=f"Error interno de inferencia: {exc}",
        ) from exc


@app.get("/api/predictions")
def get_recent_predictions(limit: int = 5) -> list[dict]:
    """Devuelve las últimas inferencias guardadas por la plantilla."""

    return list_recent_predictions(limit)
