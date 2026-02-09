from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class ScenarioParams:
    price_delta_pct: float  # [-10, +10]
    sos_delta_points: float  # [-5, +5]
    topn_skus: int | None = None  # opcional: ajuste surtido


@dataclass(frozen=True)
class ScenarioResult:
    name: str
    kpis: dict[str, float]
    risk: float
    assumptions: dict[str, float | int | None]


def _hhi(units: pd.Series) -> float:
    shares = units / max(units.sum(), 1e-9)
    return float((shares**2).sum())


@st.cache_data(show_spinner=False)
def compute_baseline(df_ctx: pd.DataFrame) -> dict[str, float]:
    return {
        "ventas_valor": float(df_ctx["ventas_valor"].sum()),
        "unidades": float(df_ctx["unidades"].sum()),
        "cuervo_profit": float(df_ctx["cuervo_profit"].sum()),
        "retailer_profit": float(df_ctx["retailer_profit"].sum()),
        "margen_pct_avg": float(np.average(df_ctx["margen_pct"], weights=df_ctx["ventas_valor"])),
        "SOS": float(df_ctx["sos"].sum()),  # ~100
        "SOM": float(df_ctx["som"].sum()),  # ~100
    }


def _apply_topn(df_ctx: pd.DataFrame, topn: int | None) -> pd.DataFrame:
    if not topn:
        return df_ctx
    topn = max(10, min(int(topn), df_ctx["sku"].nunique()))
    # Keep top-N by ventas_valor
    return df_ctx.sort_values("ventas_valor", ascending=False).head(topn).copy()


@st.cache_data(show_spinner=False)
def simulate_scenario(df_ctx: pd.DataFrame, params: ScenarioParams, name: str) -> ScenarioResult:
    """
    Heurística:
    - price_delta afecta unidades via elasticidad: U_new = U * (1 + e * (-price_delta))
    - SOS delta ajusta unidades proporcionalmente (proxy de disponibilidad/visibilidad)
    """
    d = _apply_topn(df_ctx, params.topn_skus)

    price_delta = params.price_delta_pct / 100.0
    sos_delta = params.sos_delta_points

    # unidades con elasticidad (más precio => menos unidades)
    units_new = d["unidades"] * (1.0 + d["elasticity"] * (-price_delta))
    units_new = units_new.clip(lower=0)

    # ajuste por SOS: proxy lineal suave
    sos_factor = 1.0 + (sos_delta / 100.0) * 2.0  # 5 pts => +10% aprox
    units_new = units_new * max(sos_factor, 0.7)

    price_new = d["precio"] * (1.0 + price_delta)
    ventas_valor_new = price_new * units_new

    # margen: sufre si bajamos precio (proxy). Si subimos, mejora ligeramente.
    margen_new = (d["margen_pct"] + 0.15 * price_delta).clip(0.05, 0.70)
    cuervo_profit_new = ventas_valor_new * margen_new

    # retailer profit proxy: margen retail también cambia (suave)
    retailer_margin = (0.20 + 0.04 * np.tanh(price_delta * 5)).clip(0.08, 0.35)
    retailer_profit_new = ventas_valor_new * retailer_margin

    # proxies de SOM/SOS: no recalculamos mercado total, solo reflejamos cambio relativo
    ventas_base = float(d["ventas_valor"].sum())
    ventas_new = float(ventas_valor_new.sum())
    som_proxy = 100.0 * (ventas_new / max(ventas_base, 1e-9))

    sos_base = 100.0
    sos_proxy = max(80.0, min(120.0, sos_base + sos_delta))

    # riesgo: magnitud de cambios + elasticidad media + concentración (HHI)
    elasticity_avg = float(np.average(d["elasticity"], weights=d["ventas_valor"]))
    hhi = _hhi(d["unidades"])
    mag = abs(params.price_delta_pct) / 10.0 + abs(params.sos_delta_points) / 5.0
    risk = float(min(1.0, 0.25 * mag + 0.25 * (elasticity_avg / 2.5) + 0.25 * hhi * 5 + 0.25))

    kpis = {
        "ventas_valor": float(ventas_valor_new.sum()),
        "unidades": float(units_new.sum()),
        "cuervo_profit": float(cuervo_profit_new.sum()),
        "retailer_profit": float(retailer_profit_new.sum()),
        "margen_pct_avg": float(np.average(margen_new, weights=ventas_valor_new.clip(lower=1e-9))),
        "SOM": float(som_proxy),
        "SOS": float(sos_proxy),
    }

    return ScenarioResult(
        name=name,
        kpis=kpis,
        risk=risk,
        assumptions={
            "price_delta_pct": float(params.price_delta_pct),
            "sos_delta_points": float(params.sos_delta_points),
            "topn_skus": int(params.topn_skus) if params.topn_skus else None,
        },
    )


def build_standard_scenarios(
    df_ctx: pd.DataFrame,
    price_slider: float,
    sos_slider: float,
    topn: int | None,
) -> list[ScenarioResult]:
    base = simulate_scenario(df_ctx, ScenarioParams(0.0, 0.0, None), "Base")

    # A: rentabilidad (típico: +precio leve, -SOS o neutral)
    a = simulate_scenario(df_ctx, ScenarioParams(price_delta_pct=max(price_slider, 1.5), sos_delta_points=min(sos_slider, 0.0), topn_skus=topn), "Escenario A (Rentabilidad)")

    # B: crecimiento (típico: -precio leve, +SOS)
    b = simulate_scenario(df_ctx, ScenarioParams(price_delta_pct=min(price_slider, -1.5), sos_delta_points=max(sos_slider, 1.0), topn_skus=topn), "Escenario B (Crecimiento)")

    return [base, a, b]
