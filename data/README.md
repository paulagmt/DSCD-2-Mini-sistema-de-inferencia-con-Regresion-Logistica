
# DESCRIPCIÓN
Este proyecto implementa un sistema utilizando un modelo de Regresión Logística.

Se busca estimar la probabilidad de que un cliente contrate un depósito a plazo a partir de algunas características obtenidas del dataset Bank Marketing.

# Flujo completo de la solución

Datos → Preprocesamiento → Entrenamiento → Persistencia → API → Inferencia → Frontend

La aplicación está dividida en:

- Training: carga los datos, prepara las variables, entrena y evalúa el modelo y guarda el pipeline.
- Inference: carga el pipeline previamente entrenado y lo utiliza para realizar predicciones mediante una API.

---
# Datos

Se utilizó el dataset Bank Marketingdel UCI Machine Learning Repository.

Variable objetivo

- `yes`: cliente contrató el depósito
- `no`: cliente no contrató el depósito

Variables:

- `age`: edad
- `job`: trabajo
- `marital`: estado civil
- `education`: nivel educativo
- `balance`: balance anual promedio
- `housing`: si tiene crédito hipotecario
- `loan`: si tiene préstamo
- `campaign`: número de contactos realizados durante la campaña


El archivo debe colocarse en:

```text
data/bank.csv

# Estructura del proyecto

bank-marketing-predictor/
│
├── data/
│   ├── bank.csv
│   └── README.md
│
├── training/
│   └── train.py
│
├── models/
│   ├── bank_marketing_pipeline.joblib
│   └── metrics.json
│
├── app/
│   ├── main.py
│   ├── schemas.py
│   ├── inference.py
│   ├── database.py
│   └── static/
│       ├── index.html
│       ├── app.js
│       ├── config.js
│       └── styles.css
│
├── tests/
│
├── requirements.txt
├── pyproject.toml
└── README.md

#INTERPRETACIÓN
- Accuracy de 60.99%, lo que significa que clasificó correctamente aproximadamente 6 de cada 10 clientes del conjunto de prueba.

- Precision fue de 16.08%. De los clientes que el modelo clasificó como potencialmente interesados, una proporción relativamente pequeña realmente pertenecía a esa clase.

- Recall fue de 56.73%, por lo que el modelo logró identificar aproximadamente el 57% de los clientes que sí contrataron el depósito.

- F1-score fue de 25.05%, el equilibrio entre precision y recall. El modelo batalla en identificar la clase positiva.


# CASO 1 - tipo incorrecto
si se envia {
  "age": "hola"
}
la API rechaza la solicitud porque age debe ser un número entero.
FastAPI devuelve un error de validación con codigo 422

#CASO 2 - valor fuera de rango
age esta restingida mediante
age: int = Field(ge=18, le=100)
cuando se realiza una solicitud como
{
  "age": -10
} automaticamente se rechaza.

#EVIDENCIA DE FUNCIONAMIENTO
Caso A — Inferencia válida

solicitud correcta;
{
  "age": 18,
  "job": "admin.",
  "marital": "divorced",
  "education": "primary",
  "balance": 0,
  "housing": "yes",
  "loan": "yes",
  "campaign": 1
}
predicción obtenida
{
  "prediction": "no",
  "probability": 0.3142,
  "classification": "Baja propension"
}

Caso B — Inferencia con error
La API valida los datos antes de ejecutar el pipeline, evitando que los datos inválidos lleguen al modelo.

Caso C — Frontend
Al hacer click en el botón de estimación se realiza la solicitud POST/predict y la respuesta recibida de la API se muestra en la interfaz como:
Probabilidad estimada: 40.1%
Predicción: no
Clasificación: Baja propension


#PREGUNTAS

1. ¿Por qué el modelo se entrena fuera de la API y no dentro de /predict?

El modelo se entrena fuera de la API porque el entrenamiento es un proceso previo y no se tiene que repetir cada vez que llega una solicitud. Si se entrenara dentro de /predict, cada predicción se alenta.

2. ¿Por qué es importante utilizar durante inferencia exactamente el mismo preprocesamiento utilizado durante entrenamiento?

Porque el modelo aprendió a partir de los datos, si durante la inferencia se transformaran los datos, el modelo obtendría información con un estructura diferente a la que uso para entrenar el modelo. 

3. ¿Qué diferencia existe entre predict() y predict_proba() en este problema?

predict() devuelve la clase que el modelo considera como resultado final: yes/no y predict_proba() devuelve las probabilidades estimadas para cada una de las clases. En este proyecto se utiliza para obtener la probabilidad de yes.

Por ejemplo, si predict_proba() devuelve una probabilidad de 0.72 para yes, significa que el modelo estima un 72% de probabilidad para esa clase.

4. Si el modelo devuelve una probabilidad de 0.72, ¿qué significa ese valor y qué NO significa?

Significa que de acuerdo con los datos que recibió y los patrones que aprendió durante el entrenamiento, el modelo estima una probabilidad de 72% de pertenecer a la yes.No significa que el cliente vaya a contratar el depósito ni que exista una garantía de 72% de contratación.

5. ¿Por qué duration no debería utilizarse en este sistema si queremos hacer la predicción antes de contactar al cliente?

Porque duration representa la duración de la llamada y eso solo se conoce después de tener contacto con el cliente. Como se quiere predecir la probabilidad que se tiene antes de realizar la llamada, no se puede usar una variable que todavía no se conoce. 

6. ¿Qué ocurriría si mañana cambia la estructura de los datos enviados por el frontend?

La API podría rechazar la solicitud si los datos ya no cumplen la estructura definida, se tendria que reentrenar el modelo.
