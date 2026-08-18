"""Contratos de entrada y salida de la API.

Pydantic se encarga de validar tipos y algunas reglas básicas antes de
que los datos lleguen al modelo.
"""

from typing import Literal

from pydantic import BaseModel, Field


class BankMarketingFeatures(BaseModel):
    """Datos que el frontend puede enviar a POST /predict."""

    # ge/le significa "greater/equal" y "less/equal".
    # Así, por ejemplo, age=-10 se rechaza antes de inferir.
    age: int = Field(ge=18, le=100, description="Edad del cliente")

    # Literal limita las categorías a las que realmente aparecen en el dataset.
    job: Literal[
        "admin.",
        "blue-collar",
        "entrepreneur",
        "housemaid",
        "management",
        "retired",
        "self-employed",
        "services",
        "student",
        "technician",
        "unemployed",
        "unknown",
    ]
    marital: Literal["divorced", "married", "single", "unknown"]
    education: Literal["primary", "secondary", "tertiary", "unknown"]

    balance: float = Field(description="Balance anual promedio")
    housing: Literal["yes", "no"]
    loan: Literal["yes", "no"]

    # Una campaña tiene al menos un contacto.
    campaign: int = Field(ge=1, le=100, description="Contactos durante la campaña")


class PredictionResponse(BaseModel):
    """Respuesta que recibe el frontend después de la inferencia."""

    prediction: Literal["yes", "no"]
    probability: float = Field(ge=0, le=1)
    classification: Literal["Potencialmente interesado", "Baja propension"]
