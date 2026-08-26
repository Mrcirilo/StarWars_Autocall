# Starwars Autocalls

Modelo reproducible para estimar `avg_duration_months` de RFQs de autocallables.

## Modelos aprobados

El modelo principal es CatBoost y sus artefactos se encuentran en `models/catboost/`.
La comparación temporal y las métricas se resumen en [models/README.md](models/README.md).

## Ejecutar el entrenamiento

Ejecuta los notebooks en este orden desde la raíz del proyecto:

```bash
uv run jupyter execute notebooks/Preprocess_starwars.ipynb --inplace
uv run jupyter execute notebooks/Models.ipynb --inplace
```

## API v1

La API recibe los términos de una RFQ, construye las variables usando solo el último
dato de mercado anterior a `requested_date` y devuelve la duración estimada en meses.
Al abrir `http://127.0.0.1:8000/` encontrarás un formulario web minimalista para usarla
sin escribir JSON. La documentación técnica permanece en `/docs`.

```bash
uv run starwars-autocalls
```

Con el servidor levantado, consulta `GET /docs` para la documentación interactiva.

Ejemplo de predicción:

```bash
curl -X POST http://127.0.0.1:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{
    "product_type": "Kessel Run Snowball",
    "basket_type": "worst_of",
    "underlyings": ["TECH", "SITH"],
    "autocall_barrier_pct": 1.0,
    "protection_barrier_pct": 0.5944,
    "no_call_period_months": 4,
    "observation_frequency": "6M",
    "quoted_implied_vol": 0.2755,
    "notional_credits": 250000,
    "counterparty": "Chandrila Sovereign Fund",
    "trader_id": "TRD-032",
    "requested_date": "2021-09-01"
  }'
```

La API rechaza categorías no presentes en el contrato de entrenamiento y no usa
información de mercado posterior a la RFQ.
