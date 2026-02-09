from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from src.scenario_engine import ScenarioParams, simulate_scenario
from src.decision_store import ensure_store_initialized, add_decision, list_decisions


DEMO_FLAG_KEY = "demo_initialized"


def apply_demo_mode_if_enabled(df: pd.DataFrame) -> None:
    """
    Precarga 3 decisiones para demo, solo si no existen aún.
    """
    ensure_store_initialized()
    if st.session_state.get(DEMO_FLAG_KEY, False):
        return

    # Si ya hay decisiones (por interacción), no duplicar
    if len(list_decisions()) > 0:
        st.session_state[DEMO_FLAG_KEY] = True
        return

    # Contexto demo fijo (puedes cambiarlo rápido si lo necesitas)
    cat = "Tequila Core"
    canal = "Moderno"
    retailer = "Walmart"

    df_ctx = df[(df["categoria"] == cat) & (df["canal"] == canal) & (df["retailer"] == retailer)].copy()
    if df_ctx.empty:
        # fallback
        df_ctx = df.head(500).copy()

    # 1) Aprobada (rentabilidad)
    r1 = simulate_scenario(df_ctx, ScenarioParams(price_delta_pct=3.0, sos_delta_points=-1.0, topn_skus=80), "Escenario A (Rentabilidad)")
    add_decision(
        {
            "role": "CPS",
            "categoria": cat,
            "canal": canal,
            "retailer": retailer,
            "objective": "Rentabilidad",
            "scenario": r1.name,
            "assumptions": r1.assumptions,
            "kpis_expected": r1.kpis,
            "risk": r1.risk,
            "status": "Aprobada",
            "validated_execution": False,
            "notes": "Demo: decisión orientada a rentabilidad con supuestos explícitos.",
        }
    )

    # 2) En seguimiento (alertas)
    r2 = simulate_scenario(df_ctx, ScenarioParams(price_delta_pct=-2.0, sos_delta_points=3.0, topn_skus=90), "Escenario B (Crecimiento)")
    add_decision(
        {
            "role": "Compras",
            "categoria": cat,
            "canal": canal,
            "retailer": retailer,
            "objective": "Crecimiento",
            "scenario": r2.name,
            "assumptions": r2.assumptions,
            "kpis_expected": r2.kpis,
            "risk": min(1.0, r2.risk + 0.1),
            "status": "En seguimiento",
            "validated_execution": True,
            "notes": "Demo: decisión con seguimiento y desviación simulada para semáforos.",
            "demo_deviation_seed": 13,
        }
    )

    # 3) Lista para retailer (one-pager)
    r3 = simulate_scenario(df_ctx, ScenarioParams(price_delta_pct=-1.0, sos_delta_points=2.0, topn_skus=70), "Escenario B (Crecimiento)")
    add_decision(
        {
            "role": "Ventas",
            "categoria": cat,
            "canal": canal,
            "retailer": retailer,
            "objective": "Crecimiento",
            "scenario": r3.name,
            "assumptions": r3.assumptions,
            "kpis_expected": r3.kpis,
            "risk": r3.risk,
            "status": "Lista para retailer",
            "validated_execution": False,
            "notes": "Demo: decisión lista para convertir en narrativa económica para el retailer.",
        }
    )

    st.session_state[DEMO_FLAG_KEY] = True


def reset_demo() -> None:
    """
    Limpia el estado asociado a la demo y decisiones.
    """
    keys_to_clear = ["decision_log", "decision_meta", DEMO_FLAG_KEY]
    for k in keys_to_clear:
        if k in st.session_state:
            del st.session_state[k]
