"""FastAPI application for RFQ duration prediction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from functools import lru_cache
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from catboost import CatBoostRegressor
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
MODEL_DIR = PROJECT_ROOT / "models" / "catboost"
STATIC_DIR = Path(__file__).resolve().parent / "static"
BUILD_TAG = "fenetre-glissante-v2"

# Categorical features are one-hot encoded; everything else the API must build by hand.
DUMMY_PREFIXES = ("product_type_", "basket_type_", "counterparty_", "trader_id_", "has_underlying_")
# Beyond this gap the market snapshot is older than anything seen in training.
STALE_MARKET_DAYS = 30

FREQUENCY_TO_MONTHS = {
    "1d": 1 / 30.44,
    "1m": 1,
    "m": 1,
    "monthly": 1,
    "mensual": 1,
    "1 month": 1,
    "2m": 2,
    "3m": 3,
    "q": 3,
    "quarterly": 3,
    "trimestral": 3,
    "3 months": 3,
    "6m": 6,
    "1y": 12,
    "y": 12,
    "12m": 12,
    "annual": 12,
    "anual": 12,
}


class FeatureConstructionError(ValueError):
    """Raised when an RFQ cannot be transformed into the model contract."""


class PredictionRequest(BaseModel):
    """RFQ fields available at quotation time."""

    model_config = ConfigDict(str_strip_whitespace=True)

    product_type: str
    basket_type: Literal["single", "worst_of"]
    underlyings: list[str] = Field(min_length=1)
    autocall_barrier_pct: float = Field(gt=0)
    protection_barrier_pct: float = Field(gt=0)
    no_call_period_months: int = Field(ge=0)
    observation_frequency: str
    quoted_implied_vol: float = Field(gt=0)
    notional_credits: float = Field(gt=0)
    counterparty: str
    trader_id: str
    requested_date: date
    # Plazo pactado del producto. Es un termino de la RFQ, igual que la barrera:
    # el cliente pide precio para un plazo concreto.
    nominal_maturity_months: float = Field(gt=0)


class PredictionResponse(BaseModel):
    predicted_avg_duration_months: float
    predicted_duration_ratio: float
    predicted_p10_months: float | None
    predicted_p90_months: float | None
    nominal_maturity_months: float
    requested_date: date
    model_name: str
    feature_count: int
    market_lag_days_max: int
    warnings: list[str]


def _load_model(path: Path) -> CatBoostRegressor:
    model = CatBoostRegressor()
    model.load_model(str(path))
    return model


@dataclass
class PredictionService:
    feature_columns: list[str]
    model: CatBoostRegressor
    reference: pd.DataFrame
    volatility_by_underlying: dict[str, pd.DataFrame]
    ratio_cap: float
    maturity_range: tuple[float, float]
    model_p10: CatBoostRegressor | None
    model_p90: CatBoostRegressor | None

    @classmethod
    def load(cls) -> "PredictionService":
        contract_path = MODEL_DIR / "feature_contract.json"
        model_path = MODEL_DIR / "model.cbm"
        reference_path = RAW_DIR / "underlyings_reference.csv"
        volatility_path = RAW_DIR / "daily_volatility.csv"

        for path in [contract_path, model_path, reference_path, volatility_path]:
            if not path.exists():
                raise RuntimeError(f"Required API file is missing: {path}")

        contract = json.loads(contract_path.read_text(encoding="utf-8"))
        if contract.get("target_parametrization") != "duration_ratio":
            raise RuntimeError(
                "This API expects a model trained on the duration ratio. "
                "Re-run notebooks/Models.ipynb to regenerate the artifacts."
            )

        reference = pd.read_csv(reference_path).set_index("underlying", verify_integrity=True)
        volatility = pd.read_csv(volatility_path, parse_dates=["date"]).sort_values(["underlying", "date"])
        volatility_by_underlying = {
            underlying: group.reset_index(drop=True)
            for underlying, group in volatility.groupby("underlying", sort=False)
        }

        # Las bandas de incertidumbre son opcionales: si no estan, la API sigue sirviendo.
        quantiles = contract.get("quantile_models", {})
        quantile_models = {}
        for key in ("p10", "p90"):
            path = MODEL_DIR / quantiles[key] if key in quantiles else None
            quantile_models[key] = _load_model(path) if path is not None and path.exists() else None

        low, high = contract.get("maturity_range_months", [0.0, float("inf")])
        return cls(
            feature_columns=contract["feature_columns"],
            model=_load_model(model_path),
            reference=reference,
            volatility_by_underlying=volatility_by_underlying,
            ratio_cap=float(contract["ratio_cap"]),
            maturity_range=(float(low), float(high)),
            model_p10=quantile_models["p10"],
            model_p90=quantile_models["p90"],
        )

    def _require_known_category(self, prefix: str, value: str) -> None:
        if f"{prefix}{value}" not in self.feature_columns:
            raise FeatureConstructionError(f"Unsupported {prefix[:-1]}: {value}")

    def _frequency_months(self, value: str) -> float:
        normalized = value.strip().lower()
        if normalized not in FREQUENCY_TO_MONTHS:
            allowed = ", ".join(sorted({"1M", "3M", "6M", "1Y"}))
            raise FeatureConstructionError(f"Unsupported observation_frequency '{value}'. Use one of: {allowed}")
        return FREQUENCY_TO_MONTHS[normalized]

    def build_features(self, request: PredictionRequest) -> tuple[pd.DataFrame, int]:
        underlyings = [underlying.strip().upper() for underlying in request.underlyings]
        if len(set(underlyings)) != len(underlyings):
            raise FeatureConstructionError("underlyings must not contain duplicates")
        if request.basket_type == "single" and len(underlyings) != 1:
            raise FeatureConstructionError("basket_type 'single' requires exactly one underlying")
        if request.basket_type == "worst_of" and len(underlyings) < 2:
            raise FeatureConstructionError("basket_type 'worst_of' requires at least two underlyings")
        if request.no_call_period_months > request.nominal_maturity_months:
            raise FeatureConstructionError("no_call_period_months cannot exceed nominal_maturity_months")

        self._require_known_category("product_type_", request.product_type)
        self._require_known_category("basket_type_", request.basket_type)
        self._require_known_category("counterparty_", request.counterparty)
        self._require_known_category("trader_id_", request.trader_id)
        for underlying in underlyings:
            self._require_known_category("has_underlying_", underlying)
            if underlying not in self.reference.index:
                raise FeatureConstructionError(f"Underlying missing from reference data: {underlying}")

        requested_at = pd.Timestamp(request.requested_date)
        observation_frequency_months = self._frequency_months(request.observation_frequency)
        realized_volumes, structural_volumes, market_lags = [], [], []
        for underlying in underlyings:
            history = self.volatility_by_underlying[underlying]
            available = history.loc[history["date"] < requested_at]
            if available.empty:
                raise FeatureConstructionError(
                    f"No market data before {request.requested_date} for underlying {underlying}"
                )
            latest = available.iloc[-1]
            realized_volumes.append(float(latest["realized_vol_63d"]))
            structural_volumes.append(float(self.reference.loc[underlying, "structural_base_vol"]))
            market_lags.append(int((requested_at - latest["date"]).days))

        def sample_std(values: list[float]) -> float:
            return float(np.std(values, ddof=1)) if len(values) > 1 else 0.0

        realized_min, realized_max = min(realized_volumes), max(realized_volumes)
        structural_min, structural_max = min(structural_volumes), max(structural_volumes)
        known = {
            "nominal_maturity_months": request.nominal_maturity_months,
            "autocall_barrier_pct": request.autocall_barrier_pct,
            "protection_barrier_pct": request.protection_barrier_pct,
            "no_call_period_months": request.no_call_period_months,
            "quoted_implied_vol": request.quoted_implied_vol,
            "observation_frequency_months": observation_frequency_months,
            "basket_size": len(underlyings),
            "log_notional_credits": float(np.log1p(request.notional_credits)),
            "requested_month_sin": float(np.sin(2 * np.pi * requested_at.month / 12)),
            "requested_month_cos": float(np.cos(2 * np.pi * requested_at.month / 12)),
            "requested_dayofweek": requested_at.dayofweek,
            "realized_vol_min": realized_min,
            "realized_vol_max": realized_max,
            "realized_vol_mean": float(np.mean(realized_volumes)),
            "realized_vol_std": sample_std(realized_volumes),
            "structural_base_vol_min": structural_min,
            "structural_base_vol_max": structural_max,
            "structural_base_vol_mean": float(np.mean(structural_volumes)),
            "structural_base_vol_std": sample_std(structural_volumes),
            "realized_vol_range": realized_max - realized_min,
            "structural_base_vol_range": structural_max - structural_min,
            "realized_minus_structural_vol": float(np.mean(realized_volumes) - np.mean(structural_volumes)),
        }

        # Si el contrato pide una variable continua que la API no sabe construir, hay que
        # fallar aqui: rellenarla con 0.0 en silencio daria una prediccion sin sentido.
        unknown = [
            column for column in self.feature_columns
            if not column.startswith(DUMMY_PREFIXES) and column not in known
        ]
        if unknown:
            raise FeatureConstructionError(
                f"The contract requires features the API cannot build: {unknown}. "
                "The API and notebooks/Preprocess_starwars.ipynb are out of sync."
            )

        vector = dict.fromkeys(self.feature_columns, 0.0)
        vector.update(known)
        vector[f"product_type_{request.product_type}"] = 1.0
        vector[f"basket_type_{request.basket_type}"] = 1.0
        vector[f"counterparty_{request.counterparty}"] = 1.0
        vector[f"trader_id_{request.trader_id}"] = 1.0
        for underlying in underlyings:
            vector[f"has_underlying_{underlying}"] = 1.0

        features = pd.DataFrame([vector], columns=self.feature_columns, dtype=float)
        if features.isna().any().any():
            raise FeatureConstructionError("Feature construction produced missing values")
        return features, max(market_lags)

    def _ratio_to_months(self, model: CatBoostRegressor, features: pd.DataFrame, maturity: float) -> float:
        """El modelo predice la fraccion del plazo; aqui se convierte en meses."""
        ratio = float(np.clip(model.predict(features)[0], 0.0, self.ratio_cap))
        return ratio * maturity

    def predict(self, request: PredictionRequest) -> PredictionResponse:
        features, market_lag_days_max = self.build_features(request)
        maturity = request.nominal_maturity_months

        ratio = float(np.clip(self.model.predict(features)[0], 0.0, self.ratio_cap))
        duration = ratio * maturity

        low = high = None
        if self.model_p10 is not None and self.model_p90 is not None:
            band = [
                self._ratio_to_months(self.model_p10, features, maturity),
                self._ratio_to_months(self.model_p90, features, maturity),
            ]
            low, high = min(band), max(band)

        warnings: list[str] = []
        minimum, maximum = self.maturity_range
        if not minimum <= maturity <= maximum:
            warnings.append(
                f"nominal_maturity_months={maturity:g} is outside the training range "
                f"({minimum:g}-{maximum:g}); the estimate is an extrapolation."
            )
        if market_lag_days_max > STALE_MARKET_DAYS:
            warnings.append(
                f"The latest market data is {market_lag_days_max} days older than the RFQ date."
            )

        return PredictionResponse(
            predicted_avg_duration_months=duration,
            predicted_duration_ratio=ratio,
            predicted_p10_months=low,
            predicted_p90_months=high,
            nominal_maturity_months=maturity,
            requested_date=request.requested_date,
            model_name="catboost_duration_ratio",
            feature_count=len(self.feature_columns),
            market_lag_days_max=market_lag_days_max,
            warnings=warnings,
        )


@lru_cache(maxsize=1)
def get_service() -> PredictionService:
    return PredictionService.load()


app = FastAPI(
    title="Starwars Autocalls API",
    version="0.1.0",
    description="Predicts average autocallable duration from RFQ terms available at quotation time.",
)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the lightweight browser interface."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict[str, object]:
    service = get_service()
    return {
        "status": "ok",
        "build_tag": BUILD_TAG,
        "model": "catboost_duration_ratio",
        "feature_count": len(service.feature_columns),
        "uncertainty_bands": service.model_p10 is not None and service.model_p90 is not None,
    }


@app.get("/metadata")
def metadata() -> dict[str, object]:
    """Expose the fixed training categories required by the browser form."""
    service = get_service()

    def values_for(prefix: str) -> list[str]:
        return sorted(column.removeprefix(prefix) for column in service.feature_columns if column.startswith(prefix))

    latest_market_date = max(frame["date"].max() for frame in service.volatility_by_underlying.values())
    return {
        "product_types": values_for("product_type_"),
        "counterparties": values_for("counterparty_"),
        "trader_ids": values_for("trader_id_"),
        "underlyings": values_for("has_underlying_"),
        "observation_frequencies": ["1M", "3M", "6M", "1Y"],
        "latest_market_date": latest_market_date.date().isoformat(),
        "maturity_range_months": list(service.maturity_range),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest) -> PredictionResponse:
    try:
        return get_service().predict(request)
    except FeatureConstructionError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error


def main() -> None:
    """Run the API with ``uv run starwars-autocalls``."""
    import uvicorn

    uvicorn.run("starwars_autocalls.api:app", host="127.0.0.1", port=8000, reload=False)
