# Experimento: tendencia reciente de volatilidad

## Variable probada

`realized_vol_trend_21d_mean` es, para cada subyacente, la última
`realized_vol_63d` disponible antes de la RFQ menos la media de las 21
observaciones de mercado anteriores. En una cesta se usa la media de esa señal.

Un valor positivo indica que la volatilidad estaba por encima de su nivel reciente;
un valor negativo, por debajo. La construcción usa solo información anterior a la
RFQ.

## Comparación temporal de CatBoost

Mismo entrenamiento y holdout que el modelo principal: train hasta 2023-12-29 y
evaluación entre 2024-01-01 y 2024-06-28.

| Contrato | Variables | MAE (meses) | RMSE (meses) |
|---|---:|---:|---:|
| Principal sin tendencia | 96 | 11.269 | 15.366 |
| Con tendencia de volatilidad | 97 | 11.553 | 15.313 |

## Decisión

No se promueve la variable al contrato principal: empeora MAE, que es la métrica
principal acordada, aunque el RMSE mejora de forma marginal. Se conserva en la
tabla procesada para futuras pruebas y queda excluida del contrato de inferencia
de CatBoost.
