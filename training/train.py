"""Entrenamiento y persistencia del modelo de Bank Marketing.

Este archivo SOLO se encarga de la etapa de TRAINING.
La API no debe ejecutar este archivo cada vez que recibe /predict.
"""

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Variables
FEATURES = [
    "age",
    "job",
    "marital",
    "education",
    "balance",
    "housing",
    "loan",
    "campaign",
]

# Variable objetivo: yes = contrató, no = no contrató.
TARGET = "y"


NUMERIC_FEATURES = ["age", "balance", "campaign"]
CATEGORICAL_FEATURES = ["job", "marital", "education", "housing", "loan"]


def load_dataset(data_path: Path) -> pd.DataFrame:
    """Carga el CSV y verifica que tenga las columnas necesarias."""

    
    if not data_path.exists():
        raise FileNotFoundError(
            f"No se encontró {data_path}. "
            "Descarga bank.csv desde UCI y colócalo exactamente en data/bank.csv."
        )

   
    df = pd.read_csv(data_path, sep=";")

    required = FEATURES + [TARGET]
    missing = [column for column in required if column not in df.columns]
    if missing:
        raise ValueError(f"Faltan columnas requeridas en bank.csv: {missing}")

    
    return df[required].copy()


def prepare_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convierte la variable objetivo yes/no a 1/0 para LogisticRegression."""

    
    valid_targets = {"yes", "no"}
    found_targets = set(df[TARGET].dropna().unique())
    invalid_targets = found_targets - valid_targets
    if invalid_targets:
        raise ValueError(f"Valores inesperados en y: {sorted(invalid_targets)}")

    result = df.copy()
    result[TARGET] = result[TARGET].map({"no": 0, "yes": 1})
    return result


def build_pipeline() -> Pipeline:
    """Construye el pipeline completo: preprocesamiento + regresión logística."""

    
    numeric_transformer = StandardScaler()

    
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
        ]
    )

    # Preprocesamiento 
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    random_state=42,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train_and_save(data_path: Path, project_root: Path) -> dict:
    """Entrena, evalúa y guarda el pipeline completo."""

    # 1. Cargar y revisar el dataset
    df = load_dataset(data_path)
    df = prepare_target(df)

    # 2. Separar variables predictoras y y objetivo
    X = df[FEATURES]
    y = df[TARGET]

    # 3. Dividir entrenamiento y pruebas
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    # 4. Construir y entrenar el pipeline
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    # 5. Evaluar sobre datos irrelevantes
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1_score": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "model": "LogisticRegression",
        "dataset": "UCI Bank Marketing",
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "features": FEATURES,
        "excluded_features": ["duration"],
        "random_state": 42,
        "test_size": 0.20,
    }

    # 6. Persistir el pipeline completo
    models_dir = project_root / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "bank_marketing_pipeline.joblib"
    metrics_path = models_dir / "metrics.json"

    joblib.dump(pipeline, model_path)
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    print(f"Modelo guardado en: {model_path}")
    print(f"Métricas guardadas en: {metrics_path}")
    return metrics


def main() -> None:
    """Punto de entrada del script."""

    parser = argparse.ArgumentParser(description="Entrena Bank Marketing Logistic Regression")
    parser.add_argument(
        "--data",
        default=None,
        help="Ruta alternativa al bank.csv (por defecto: data/bank.csv)",
    )
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parents[1]
    data_path = Path(args.data) if args.data else project_root / "data" / "bank.csv"

    train_and_save(data_path, project_root)


if __name__ == "__main__":
    main()
