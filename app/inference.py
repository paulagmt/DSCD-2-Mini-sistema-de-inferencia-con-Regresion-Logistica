"""Lógica de inferencia.

Este módulo NO entrena el modelo.
Solo carga el pipeline persistido y lo utiliza para predecir.
"""

from pathlib import Path

import joblib
import pandas as pd

# Mantener un orden explícito ayuda a construir el DataFrame exactamente
# como espera el pipeline entrenado.
FEATURE_ORDER = [
    "age",
    "job",
    "marital",
    "education",
    "balance",
    "housing",
    "loan",
    "campaign",
]

# Ruta al modelo generado por training/train.py.
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "bank_marketing_pipeline.joblib"

# Cacheamos el modelo en memoria después de la primera carga.
# Así no abrimos el archivo .joblib en cada petición.
_model = None


def get_model():
    """Carga el pipeline persistido una sola vez."""

    global _model

    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"No existe el modelo en {MODEL_PATH}. "
                "Ejecuta primero: python training/train.py"
            )

        # Aquí recuperamos TODO el Pipeline:
        # ColumnTransformer + LogisticRegression.
        _model = joblib.load(MODEL_PATH)

    return _model


def predict(payload: dict) -> dict:
    """Ejecuta una inferencia para un solo cliente."""

    model = get_model()

    # Convertimos el JSON recibido a una fila de DataFrame.
    # No hacemos aquí OneHotEncoder ni StandardScaler manualmente.
    # Eso ya vive dentro del Pipeline persistido.
    row = {key: payload[key] for key in FEATURE_ORDER}
    sample = pd.DataFrame([row], columns=FEATURE_ORDER)

    # predict_proba devuelve una probabilidad para cada clase.
    # En este dataset las clases son no/yes; buscamos explícitamente
    # el índice de "yes" para no depender de una posición asumida.
    probabilities = model.predict_proba(sample)[0]
    classes = list(model.classes_)
    yes_index = classes.index(1)
    probability = float(probabilities[yes_index])

    # predict() devuelve la clase final elegida por el clasificador.
    prediction_value = int(model.predict(sample)[0])
    prediction = "yes" if prediction_value == 1 else "no"

    # Esta etiqueta es solo una forma sencilla de mostrar el resultado.
    classification = (
        "Potencialmente interesado" if prediction == "yes" else "Baja propension"
    )

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "classification": classification,
    }
