"""Base de datos auxiliar heredada de la plantilla.

SQLite NO participa en el entrenamiento ni en la inferencia.
Solo guarda un pequeño historial para poder mostrarlo en el frontend.
"""

import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.db"


def init_db() -> None:
    """Crea la tabla de historial si todavía no existe."""

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                age INTEGER NOT NULL,
                job TEXT NOT NULL,
                marital TEXT NOT NULL,
                education TEXT NOT NULL,
                balance REAL NOT NULL,
                housing TEXT NOT NULL,
                loan TEXT NOT NULL,
                campaign INTEGER NOT NULL,
                prediction TEXT NOT NULL,
                probability REAL NOT NULL,
                classification TEXT NOT NULL
            )
            """
        )


def save_prediction(features: dict, result: dict) -> None:
    """Guarda una inferencia para mostrarla después en el frontend."""

    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            INSERT INTO predictions (
                age, job, marital, education, balance, housing, loan, campaign,
                prediction, probability, classification
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                features["age"],
                features["job"],
                features["marital"],
                features["education"],
                features["balance"],
                features["housing"],
                features["loan"],
                features["campaign"],
                result["prediction"],
                result["probability"],
                result["classification"],
            ),
        )


def list_recent_predictions(limit: int = 5) -> list[dict]:
    """Obtiene las últimas inferencias para el historial del frontend."""

    # Evitamos que alguien solicite un número excesivo de registros.
    limit = max(1, min(limit, 50))
    init_db()

    with sqlite3.connect(DB_PATH) as conn:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            """
            SELECT id, created_at, age, job, marital, education, balance,
                   housing, loan, campaign, prediction, probability, classification
            FROM predictions
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    return [dict(row) for row in rows]
