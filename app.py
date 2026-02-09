from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st


@dataclass
class ScenarioParams:
    price_adj_pct: float
    sos_adj: float
    top_n: int


def init_session_state() -> None:
    if "decision_log" not in st.session_state:
        st.session_state.decision_log = []
    if "role" not in st.session_state:
        st.session_state.role = "CPS"
    if "scenario_params" not in st.session_state:
        st.session_state.scenario_params = {
            "A": ScenarioParams(price_adj_pct=3.0, sos_adj=1.0, top_n=120),
            "B": ScenarioParams(price_adj_pct=-3.0, sos_adj=2.0, top_n=150),
        }


@st.cache_data(show_spinner=False)
def generate_synthetic_data(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    categories = ["Cervezas", "Vinos", "Spirits", "RTD"]
    channels = ["Moderno", "Mayoreo"]
    retailers = {
        "Moderno": ["Walmart", "SuperOnline"],
        "Mayoreo": ["Distribuidor MX"],
    }

    rows: List[Dict[str, Any]] = []
    for category in categories:
        sku_count = rng.integers(80, 160)
        for channel in channels:
            for retailer in retailers[channel]:
                for sku_idx in range(sku_count):
                    sku = f"{category[:2].upper()}-{channel[:2].upper()}-{sku_idx:03d}"
                    base_price = rng.normal(120, 25)
                    base_price = max(30, base_price)
                    units = rng.lognormal(mean=7.2, sigma=0.4)
                    units = max(50, units)
                    margin_pct = rng.normal(0.28, 0.06)
                    margin_pct = float(np.clip(margin_pct, 0.1, 0.45))
                    sos = rng.normal(20, 6)
                    som = sos * rng.normal(0.9, 0.1)
                    ventas_valor = base_price * units
                    retailer_profit = ventas_valor * rng.normal(0.18, 0.03)
                    cuervo_profit = ventas_valor * margin_pct
                    rows.append(
                        {
                            "categoria": category,
                            "canal": channel,
                            "retailer": retailer,
                            "sku": sku,
                            "precio": base_price,
                            "unidades": units,
                            "ventas_valor": ventas_valor,
                            "margen_pct": margin_pct,
                            "sos": sos,
                            "som": som,
                            "retailer_profit": retailer_profit,
                            "cuervo_profit": cuervo_profit,
                        }
                    )
    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def category_elasticities() -> Dict[Tuple[str, str], float]:
    return {
        ("Cervezas", "Moderno"): 1.2,
        ("Cervezas", "Mayoreo"): 0.9,
        ("Vinos", "Moderno"): 1.1,
        ("Vinos", "Mayoreo"): 0.8,
        ("Spirits", "Moderno"): 0.7,
        ("Spirits", "Mayoreo"): 0.6,
        ("RTD", "Moderno"): 1.4,
        ("RTD", "Mayoreo"): 1.0,
    }


@st.cache_data(show_spinner=False)
def compute_baseline_kpis(df: pd.DataFrame) -> Dict[str, float]:
    ventas = df["ventas_valor"].sum()
    unidades = df["unidades"].sum()
    cuervo_profit = df["cuervo_profit"].sum()
    retailer_profit = df["retailer_profit"].sum()
    margen = df["margen_pct"].mean()
    sos = df["sos"].mean()
    som = df["som"].mean()
    return {
        "ventas_valor": float(ventas),
        "unidades": float(unidades),
        "cuervo_profit": float(cuervo_profit),
        "retailer_profit": float(retailer_profit),
        "margen_pct": float(margen),
        "sos": float(sos),
        "som": float(som),
    }


@st.cache_data(show_spinner=False)
def apply_scenario(
    df: pd.DataFrame,
    price_adj_pct: float,
    sos_adj: float,
    top_n: int,
    elasticity: float,
) -> Dict[str, float]:
    df_sorted = df.sort_values("ventas_valor", ascending=False).head(top_n)
    price_multiplier = 1 + price_adj_pct / 100
    units_multiplier = 1 - elasticity * (price_adj_pct / 100)
    units_multiplier = max(0.7, min(1.3, units_multiplier))
    sos_multiplier = 1 + sos_adj / 100
    adjusted_units = df_sorted["unidades"] * units_multiplier * sos_multiplier
    adjusted_price = df_sorted["precio"] * price_multiplier
    ventas = float((adjusted_units * adjusted_price).sum())
    margen = float(df_sorted["margen_pct"].mean())
    cuervo_profit = ventas * margen
    retailer_profit = ventas * float(df_sorted["retailer_profit"].sum() / df_sorted["ventas_valor"].sum())
    sos = float(df_sorted["sos"].mean() + sos_adj)
    som = float(df_sorted["som"].mean() + sos_adj * 0.7)
    return {
        "ventas_valor": ventas,
        "unidades": float(adjusted_units.sum()),
        "cuervo_profit": cuervo_profit,
        "retailer_profit": retailer_profit,
        "margen_pct": margen,
        "sos": sos,
        "som": som,
    }


@st.cache_data(show_spinner=False)
def risk_score(
    df: pd.DataFrame, price_adj_pct: float, sos_adj: float, elasticity: float
) -> Tuple[float, str]:
    shares = df["ventas_valor"] / df["ventas_valor"].sum()
    hhi = float((shares**2).sum())
    magnitude = abs(price_adj_pct) * 0.6 + abs(sos_adj) * 1.1
    risk_raw = magnitude * (1 + elasticity) * (1 + hhi * 5)
    score = float(np.clip(risk_raw / 10, 0, 1))
    if score < 0.33:
        label = "Bajo"
    elif score < 0.66:
        label = "Medio"
    else:
        label = "Alto"
    return score, label


def decision_log_to_df() -> pd.DataFrame:
    if not st.session_state.decision_log:
        return pd.DataFrame()
    return pd.DataFrame(st.session_state.decision_log)


def register_decision(entry: Dict[str, Any]) -> None:
    st.session_state.decision_log.append(entry)


def demo_seed_decisions(
    categoria: str,
    canal: str,
    retailer: str,
    baseline: Dict[str, float],
    scenario_a: Dict[str, float],
    scenario_b: Dict[str, float],
) -> None:
    if st.session_state.decision_log:
        return
    now = datetime.utcnow().isoformat()
    st.session_state.decision_log.extend(
        [
            {
                "id": "DEC-1001",
                "categoria": categoria,
                "canal": canal,
                "retailer": retailer,
                "scenario": "Escenario A",
                "status": "Aprobada",
                "role": "CPS",
                "version": 1,
                "created_at": now,
                "params": {"price_adj_pct": 3.0, "sos_adj": 1.0, "top_n": 120},
                "baseline": baseline,
                "kpis": scenario_a,
                "risk": "Medio",
                "notes": "Foco rentabilidad y mix premium.",
                "override": "",
                "validated": False,
            },
            {
                "id": "DEC-1002",
                "categoria": categoria,
                "canal": canal,
                "retailer": retailer,
                "scenario": "Escenario B",
                "status": "En seguimiento",
                "role": "Compras",
                "version": 1,
                "created_at": now,
                "params": {"price_adj_pct": -2.0, "sos_adj": 2.5, "top_n": 150},
                "baseline": baseline,
                "kpis": scenario_b,
                "risk": "Alto",
                "notes": "Crecimiento con sensibilidad a precio.",
                "override": "",
                "validated": False,
            },
            {
                "id": "DEC-1003",
                "categoria": categoria,
                "canal": canal,
                "retailer": retailer,
                "scenario": "Escenario A",
                "status": "Lista para retailer",
                "role": "Ventas",
                "version": 1,
                "created_at": now,
                "params": {"price_adj_pct": 2.0, "sos_adj": 1.0, "top_n": 100},
                "baseline": baseline,
                "kpis": scenario_a,
                "risk": "Medio",
                "notes": "One-pager listo para presentación.",
                "override": "",
                "validated": True,
            },
        ]
    )


def reset_demo_state() -> None:
    for key in ["decision_log", "scenario_params", "role"]:
        if key in st.session_state:
            del st.session_state[key]


def format_currency(value: float) -> str:
    return f"${value:,.0f}"


def format_pct(value: float) -> str:
    return f"{value:.1f}%"


def main() -> None:
    st.set_page_config(page_title="Decision Cockpit de Categoría", layout="wide")
    init_session_state()

    st.sidebar.title("Decision Cockpit")
    demo_mode = st.sidebar.toggle("Modo Demo", value=True)
    if demo_mode:
        if st.sidebar.button("Reset Demo"):
            reset_demo_state()
            st.experimental_rerun()

    role = st.sidebar.selectbox(
        "Rol",
        ["CPS", "Compras", "Ventas"],
        index=["CPS", "Compras", "Ventas"].index(st.session_state.role),
        key="role",
    )

    data = generate_synthetic_data()
    categories = sorted(data["categoria"].unique().tolist())
    channels = sorted(data["canal"].unique().tolist())
    category = st.sidebar.selectbox("Categoría", categories)
    channel = st.sidebar.selectbox("Canal", channels)
    retailers = sorted(data.query("canal == @channel")["retailer"].unique().tolist())
    retailer = st.sidebar.selectbox("Retailer", retailers)

    mode = st.radio(
        "Modo del cockpit",
        ["Decidir", "Gobernar", "Defender"],
        horizontal=True,
    )

    subset = data.query("categoria == @category and canal == @channel and retailer == @retailer")
    elasticity = category_elasticities()[(category, channel)]
    baseline_kpis = compute_baseline_kpis(subset)

    if demo_mode:
        scenario_a = apply_scenario(
            subset,
            price_adj_pct=3.0,
            sos_adj=1.0,
            top_n=120,
            elasticity=elasticity,
        )
        scenario_b = apply_scenario(
            subset,
            price_adj_pct=-2.0,
            sos_adj=2.5,
            top_n=150,
            elasticity=elasticity,
        )
        demo_seed_decisions(category, channel, retailer, baseline_kpis, scenario_a, scenario_b)

    if demo_mode:
        with st.container(border=True):
            st.subheader("Demo Runbook")
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Step 1: Decidir"):
                    st.session_state["demo_step"] = "Decidir"
                st.markdown(
                    """
- Presentar contexto y objetivo económico.
- Comparar escenarios vs baseline.
- Registrar decisión aprobada.
"""
                )
            with col2:
                if st.button("Step 2: Gobernar"):
                    st.session_state["demo_step"] = "Gobernar"
                st.markdown(
                    """
- Mostrar Decision Log filtrado.
- Revisar alertas de desviación.
- Ajustar supuestos con versión.
"""
                )
            with col3:
                if st.button("Step 3: Defender"):
                    st.session_state["demo_step"] = "Defender"
                st.markdown(
                    """
- Seleccionar decisión lista.
- Narrativa retailer y KPIs clave.
- Generar one-pager en .md.
"""
                )
        if st.session_state.get("demo_step"):
            mode = st.session_state["demo_step"]

    if mode == "Decidir":
        render_decidir(subset, baseline_kpis, elasticity, category, channel, retailer, role)
    elif mode == "Gobernar":
        render_gobernar()
    else:
        render_defender()


def render_decidir(
    subset: pd.DataFrame,
    baseline_kpis: Dict[str, float],
    elasticity: float,
    category: str,
    channel: str,
    retailer: str,
    role: str,
) -> None:
    st.header("Decidir ex-ante")

    with st.container(border=True):
        col1, col2, col3 = st.columns(3)
        col1.metric("Categoría", category)
        col2.metric("Canal", channel)
        col3.metric("Retailer", retailer)
        st.markdown(
            """
- Objetivo económico sugerido: rentabilidad en canales estables, crecimiento en canales dinámicos.
- Elasticidad sintética aplicada por categoría/canal.
- Riesgo calculado por magnitud del cambio y concentración.
"""
        )

    st.subheader("Palancas por escenario")
    params_a = st.session_state.scenario_params["A"]
    params_b = st.session_state.scenario_params["B"]

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("**Escenario A (rentabilidad)**")
        params_a.price_adj_pct = st.slider(
            "Ajuste de precio (%)",
            -10.0,
            10.0,
            params_a.price_adj_pct,
            step=0.5,
            key="price_adj_a",
        )
        params_a.sos_adj = st.slider(
            "Ajuste de SOS (puntos)",
            -5.0,
            5.0,
            params_a.sos_adj,
            step=0.5,
            key="sos_adj_a",
        )
        params_a.top_n = st.slider(
            "Top-N SKUs",
            50,
            200,
            params_a.top_n,
            step=10,
            key="top_n_a",
        )

    with col_b:
        st.markdown("**Escenario B (crecimiento)**")
        params_b.price_adj_pct = st.slider(
            "Ajuste de precio (%) ",
            -10.0,
            10.0,
            params_b.price_adj_pct,
            step=0.5,
            key="price_adj_b",
        )
        params_b.sos_adj = st.slider(
            "Ajuste de SOS (puntos) ",
            -5.0,
            5.0,
            params_b.sos_adj,
            step=0.5,
            key="sos_adj_b",
        )
        params_b.top_n = st.slider(
            "Top-N SKUs ",
            50,
            200,
            params_b.top_n,
            step=10,
            key="top_n_b",
        )

    scenario_a = apply_scenario(
        subset,
        params_a.price_adj_pct,
        params_a.sos_adj,
        params_a.top_n,
        elasticity,
    )
    scenario_b = apply_scenario(
        subset,
        params_b.price_adj_pct,
        params_b.sos_adj,
        params_b.top_n,
        elasticity,
    )

    st.subheader("Comparador de escenarios")
    kpi_cols = st.columns(6)
    kpi_cols[0].metric("Ventas", format_currency(baseline_kpis["ventas_valor"]))
    kpi_cols[1].metric("Unidades", f"{baseline_kpis['unidades']:,.0f}")
    kpi_cols[2].metric("Profit Cuervo", format_currency(baseline_kpis["cuervo_profit"]))
    kpi_cols[3].metric("Profit Retailer", format_currency(baseline_kpis["retailer_profit"]))
    kpi_cols[4].metric("SOM", format_pct(baseline_kpis["som"]))
    kpi_cols[5].metric("SOS", format_pct(baseline_kpis["sos"]))

    compare_df = pd.DataFrame(
        {
            "Escenario": ["Base", "Escenario A", "Escenario B"],
            "Ventas": [
                baseline_kpis["ventas_valor"],
                scenario_a["ventas_valor"],
                scenario_b["ventas_valor"],
            ],
            "Profit Cuervo": [
                baseline_kpis["cuervo_profit"],
                scenario_a["cuervo_profit"],
                scenario_b["cuervo_profit"],
            ],
        }
    )
    fig = px.bar(
        compare_df,
        x="Escenario",
        y=["Ventas", "Profit Cuervo"],
        barmode="group",
        title="Impacto vs baseline",
    )
    st.plotly_chart(fig, use_container_width=True)

    scenario_choice = st.selectbox(
        "Elegir escenario",
        ["Base", "Escenario A", "Escenario B"],
    )
    selected_kpis = {
        "Base": baseline_kpis,
        "Escenario A": scenario_a,
        "Escenario B": scenario_b,
    }[scenario_choice]

    price_adj = 0.0
    sos_adj = 0.0
    top_n = 0
    if scenario_choice == "Escenario A":
        price_adj = params_a.price_adj_pct
        sos_adj = params_a.sos_adj
        top_n = params_a.top_n
    elif scenario_choice == "Escenario B":
        price_adj = params_b.price_adj_pct
        sos_adj = params_b.sos_adj
        top_n = params_b.top_n

    score, risk_label = risk_score(subset, price_adj, sos_adj, elasticity)
    st.info(f"Riesgo estimado: {risk_label} ({score:.2f})")

    min_margin = 0.2
    block_reason = selected_kpis["cuervo_profit"] < 0 or selected_kpis["margen_pct"] < min_margin
    override = False
    justification = ""
    if block_reason:
        st.warning("La decisión no cumple umbral mínimo de rentabilidad.")
        override = st.checkbox("Override con justificación")
        if override:
            justification = st.text_area("Justificación")

    if st.button("Registrar decisión"):
        if block_reason and not override:
            st.error("No se puede registrar sin override.")
        else:
            decision_id = f"DEC-{1000 + len(st.session_state.decision_log) + 1}"
            register_decision(
                {
                    "id": decision_id,
                    "categoria": category,
                    "canal": channel,
                    "retailer": retailer,
                    "scenario": scenario_choice,
                    "status": "Aprobada",
                    "role": role,
                    "version": 1,
                    "created_at": datetime.utcnow().isoformat(),
                    "params": {
                        "price_adj_pct": price_adj,
                        "sos_adj": sos_adj,
                        "top_n": top_n,
                    },
                    "baseline": baseline_kpis,
                    "kpis": selected_kpis,
                    "risk": risk_label,
                    "notes": "Decisión registrada desde Decidir.",
                    "override": justification if override else "",
                    "validated": False,
                }
            )
            st.success("Decisión registrada en el Decision Log.")


def render_gobernar() -> None:
    st.header("Gobernar durante ejecución")
    log_df = decision_log_to_df()
    if log_df.empty:
        st.info("No hay decisiones registradas. Usa el modo Decidir para crear una.")
        return

    with st.container(border=True):
        st.subheader("Filtros")
        col1, col2, col3 = st.columns(3)
        category_filter = col1.selectbox(
            "Categoría",
            ["Todas"] + sorted(log_df["categoria"].unique().tolist()),
        )
        channel_filter = col2.selectbox(
            "Canal",
            ["Todos"] + sorted(log_df["canal"].unique().tolist()),
        )
        status_filter = col3.selectbox(
            "Estado",
            ["Todos"] + sorted(log_df["status"].unique().tolist()),
        )

    filtered_df = log_df.copy()
    if category_filter != "Todas":
        filtered_df = filtered_df[filtered_df["categoria"] == category_filter]
    if channel_filter != "Todos":
        filtered_df = filtered_df[filtered_df["canal"] == channel_filter]
    if status_filter != "Todos":
        filtered_df = filtered_df[filtered_df["status"] == status_filter]

    st.subheader("Decision Log")
    st.dataframe(
        filtered_df[
            ["id", "categoria", "canal", "retailer", "scenario", "status", "risk", "role", "version"]
        ],
        use_container_width=True,
    )

    selected_id = st.selectbox("Seleccionar decisión", filtered_df["id"].tolist())
    selected = log_df[log_df["id"] == selected_id].iloc[0].to_dict()

    with st.container(border=True):
        st.subheader("Ficha de decisión")
        col1, col2, col3 = st.columns(3)
        col1.metric("Escenario", selected["scenario"])
        col2.metric("Estado", selected["status"])
        col3.metric("Riesgo", selected["risk"])
        st.markdown(
            f"""
- Rol responsable: **{selected['role']}**
- Versión: **{selected['version']}**
- Notas: {selected['notes']}
"""
        )

    with st.container(border=True):
        st.subheader("Seguimiento real vs esperado")
        expected = selected["kpis"]
        rng = np.random.default_rng(abs(hash(selected_id)) % (2**32))
        deviation = rng.normal(0, 0.08)
        real_sales = expected["ventas_valor"] * (1 + deviation)
        real_profit = expected["cuervo_profit"] * (1 + deviation * 1.2)
        follow_df = pd.DataFrame(
            {
                "KPI": ["Ventas", "Profit Cuervo"],
                "Esperado": [expected["ventas_valor"], expected["cuervo_profit"]],
                "Real": [real_sales, real_profit],
            }
        )
        follow_df["Desviación %"] = (follow_df["Real"] / follow_df["Esperado"] - 1) * 100
        st.dataframe(follow_df, use_container_width=True)

        thresholds = {"Ventas": 0.05, "Profit Cuervo": 0.07}
        lights = []
        for _, row in follow_df.iterrows():
            limit = thresholds[row["KPI"]]
            deviation_pct = abs(row["Desviación %"]) / 100
            if deviation_pct < limit:
                lights.append("🟢")
            elif deviation_pct < limit * 1.5:
                lights.append("🟡")
            else:
                lights.append("🔴")
        st.markdown(
            """
**Semáforo por KPI**
- Ventas
- Profit Cuervo
"""
        )
        st.write(" ".join(lights))

    col1, col2, col3 = st.columns(3)
    if col1.button("Validar ejecución"):
        update_decision(selected_id, {"validated": True})
        st.success("Ejecución validada.")
    if col2.button("Ajustar supuestos"):
        new_version = int(selected["version"]) + 1
        new_entry = selected.copy()
        new_entry["id"] = f"{selected_id}-v{new_version}"
        new_entry["version"] = new_version
        new_entry["status"] = "En seguimiento"
        new_entry["notes"] = "Versión creada por ajuste de supuestos."
        register_decision(new_entry)
        st.success("Nueva versión creada.")
    if col3.button("Cerrar decisión"):
        update_decision(selected_id, {"status": "Cerrada"})
        st.success("Decisión cerrada.")

    with st.container(border=True):
        st.subheader("Exportar Decision Log")
        csv_data = log_df.to_csv(index=False).encode("utf-8")
        json_data = json.dumps(st.session_state.decision_log, ensure_ascii=False, indent=2).encode("utf-8")
        st.download_button("Descargar CSV", data=csv_data, file_name="decision_log.csv")
        st.download_button("Descargar JSON", data=json_data, file_name="decision_log.json")


def update_decision(decision_id: str, updates: Dict[str, Any]) -> None:
    for idx, entry in enumerate(st.session_state.decision_log):
        if entry["id"] == decision_id:
            updated = entry.copy()
            updated.update(updates)
            st.session_state.decision_log[idx] = updated
            break


def render_defender() -> None:
    st.header("Defender frente al retailer")
    log_df = decision_log_to_df()
    if log_df.empty:
        st.info("No hay decisiones disponibles. Usa el modo Decidir para crear una.")
        return

    available = log_df[log_df["status"].isin(["Aprobada", "En seguimiento", "Lista para retailer"])]
    if available.empty:
        st.warning("No hay decisiones vigentes para defender.")
        return

    selected_id = st.selectbox("Decisión a defender", available["id"].tolist())
    selected = available[available["id"] == selected_id].iloc[0].to_dict()
    baseline = selected["baseline"]
    kpis = selected["kpis"]

    with st.container(border=True):
        st.subheader("Retailer Story")
        bullets = [
            f"Cambio propuesto: {selected['scenario']} con ajuste de precio {selected['params']['price_adj_pct']}%.",
            f"Beneficio retailer: +{(kpis['ventas_valor'] / baseline['ventas_valor'] - 1) * 100:.1f}% en ventas.",
            "Mejor rotación por foco en SKUs líderes.",
            f"Beneficio Cuervo: profit esperado {format_currency(kpis['cuervo_profit'])}.",
            "Benchmark competitivo: mix premium vs categoría +1.5 pp.",
            "Supuestos clave: elasticidad estable y ejecución de exhibiciones.",
        ]
        st.markdown("\n".join([f"- {b}" for b in bullets[:7]]))

    compare_df = pd.DataFrame(
        {
            "Escenario": ["Baseline", "Propuesto"],
            "Ventas": [baseline["ventas_valor"], kpis["ventas_valor"]],
            "Profit Retailer": [baseline["retailer_profit"], kpis["retailer_profit"]],
        }
    )
    fig = px.bar(compare_df, x="Escenario", y=["Ventas", "Profit Retailer"], barmode="group")
    st.plotly_chart(fig, use_container_width=True)

    if st.button("Generar one-pager"):
        narrative = generate_one_pager(selected)
        st.download_button(
            "Descargar one-pager (.md)",
            data=narrative.encode("utf-8"),
            file_name=f"one_pager_{selected_id}.md",
        )


def generate_one_pager(decision: Dict[str, Any]) -> str:
    baseline = decision["baseline"]
    kpis = decision["kpis"]
    bullets = [
        f"Escenario: {decision['scenario']}",
        f"Ventas esperadas: {format_currency(kpis['ventas_valor'])}",
        f"Profit Cuervo: {format_currency(kpis['cuervo_profit'])}",
        f"Profit Retailer: {format_currency(kpis['retailer_profit'])}",
        f"SOM/SOS: {format_pct(kpis['som'])} / {format_pct(kpis['sos'])}",
        f"Riesgo: {decision['risk']}",
    ]
    return "\n".join(
        [
            f"# Decision Cockpit | {decision['id']}",
            "## Resumen Ejecutivo",
            *[f"- {b}" for b in bullets],
            "## Comparativo Baseline",
            f"- Ventas baseline: {format_currency(baseline['ventas_valor'])}",
            f"- Profit Cuervo baseline: {format_currency(baseline['cuervo_profit'])}",
            f"- Profit Retailer baseline: {format_currency(baseline['retailer_profit'])}",
            "## Supuestos",
            f"- Elasticidad aplicada según categoría/canal.",
            f"- Ajustes de precio/SOS según parámetros registrados.",
        ]
    )


if __name__ == "__main__":
    main()
