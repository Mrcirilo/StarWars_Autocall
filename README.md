# Starwars Autocalls

Modelo reproducible para estimar `avg_duration_months` de RFQs de autocallables.

## Modelos aprobados

El modelo principal es CatBoost y sus artefactos se encuentran en `models/catboost/`.
La comparación temporal y las métricas se resumen en [models/README.md](models/README.md).

En vez de predecir los meses directamente, el modelo predice **qué fracción de su plazo sobrevive
el producto**, y la predicción se convierte a meses multiplicando por el vencimiento nominal. Un
autocallable dura como mucho hasta vencimiento, así que separar la escala (el plazo) del
comportamiento (cuándo se cancela dentro de ese plazo) es lo que tiene sentido de negocio. La
justificación completa está en el bloque 4 de [notebooks/Models.ipynb](notebooks/Models.ipynb), y
la del uso del plazo como variable, en el bloque 3 de
[notebooks/Preprocess_starwars.ipynb](notebooks/Preprocess_starwars.ipynb).

Junto a la estimación central se publican bandas **P10–P90**, entrenadas con pérdida de cuantil:
una mesa de riesgo gestiona rangos, no números sueltos.

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
    "requested_date": "2021-09-01",
    "nominal_maturity_months": 60
  }'
```

`nominal_maturity_months` es el plazo pactado del producto. Es un término de la RFQ como
cualquier otro —el cliente pide precio para un plazo concreto—, y es la variable más
predictiva del modelo.

La respuesta incluye la estimación en meses, la fracción del plazo que representa, la banda
P10–P90 y una lista de avisos cuando el plazo cae fuera del rango visto en entrenamiento o los
datos de mercado son más antiguos de lo habitual.

La API rechaza categorías no presentes en el contrato de entrenamiento y no usa
información de mercado posterior a la RFQ.
