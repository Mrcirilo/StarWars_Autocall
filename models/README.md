# Resultados de modelos


Holdout temporal: 2024-01-01 a 2024-06-28 (746 RFQs).

| Modelo | MAE (meses) | RMSE (meses) |
|---|---:|---:|
| CatBoost principal | 11.269 | 15.366 |
| GAM explicativo | 12.596 | 16.035 |
| Baseline estacional (mes × día semana) | 18.135 | 22.320 |

CatBoost es el modelo predictivo principal en esta primera comparación. El GAM se conserva para interpretación y el baseline es la referencia mínima.
