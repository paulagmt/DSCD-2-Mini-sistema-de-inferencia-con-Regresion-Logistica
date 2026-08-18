# Modelo persistido

Después de ejecutar:

```bash
python training/train.py
```

se generarán aquí:

- `bank_marketing_pipeline.joblib`: Pipeline completo de Scikit-Learn.
- `metrics.json`: métricas obtenidas sobre el conjunto de prueba.

No se incluye un `.joblib` falso: debe generarse entrenando con el `bank.csv` real.
