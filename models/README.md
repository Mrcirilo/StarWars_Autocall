# Resultados de modelos

Holdout temporal: 2024-01-01 a 2024-06-28 (746 RFQs).
Modelo principal: CatBoost sobre el ratio duración/plazo, 91 variables.

| Modelo | MAE (meses) | IC95% | RMSE (meses) |
|---|---:|:--:|---:|
| CatBoost sobre el ratio (principal) | 4.017 | 3.72 – 4.30 | 5.694 |
| GAM sobre el ratio | 6.867 | 6.45 – 7.26 | 8.988 |
| CatBoost sin el plazo | 11.313 | 10.59 – 12.08 | 15.429 |
| Baseline estructural (ratio x plazo) | 15.127 | 14.27 – 16.08 | 19.871 |
| Mediana global | 18.107 | 17.23 – 19.05 | 22.312 |

## Backtest de origen deslizante (5 ventanas de 6 meses)

| Variante | MAE medio | Desviación entre ventanas |
|---|---:|---:|
| Baseline estructural | 15.043 | 0.208 |
| CatBoost sin plazo | 11.220 | 0.440 |
| CatBoost sobre el ratio | 3.542 | 1.042 |

## Decisiones

- El plazo del producto (`nominal_maturity_months`) entra en el contrato: es un término pactado
  de la RFQ, no información del futuro. Aporta 7.29 meses de MAE.
- El modelo predice la fracción del plazo que sobrevive el producto, no los meses directamente.
  Aporta otros 0.01 meses.
- Las diferencias por debajo de ~0.5 meses no son distinguibles del ruido con este holdout.
- Se publican bandas P10-P90 con cobertura observada del 59%.
