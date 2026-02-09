from __future__ import annotations

import json
from typing import Any
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.scenario_engine import build_standard_scenarios, compute_baseline, ScenarioResult
from src.decision_store import (
    add_decision,
    create_new_version,
    export_csv_bytes,
    export_json_bytes,
    get_decision_by_id,
    list_decisions,
    to_dataframe,
    update_decision,
)


def render_header() -> None:
    st.title("📊 Decision Cockpit de Categoría")
    st.caption("Maqueta: una sola vista, tres momentos de uso — Decidir · Gobernar · Defender")


def render_sidebar_controls(
    category_options: list[str],
    channel_retailer_options: list[tuple[str, str]],
) -> tuple[bool, str, str, str, str]:
    with st.sidebar:
        st.subheader("Controles")
        demo_on = st.toggle("Modo Demo", value=True)

        role = st.selectbox("Rol", ["CPS", "Compras", "Ventas"], index=0)

        category = st.selectbox("Categoría", category_options, index=0)

        canal, retailer = st.selectbox(
            "Canal / Retailer",
            channel_retailer_options,
            index=0,
            format_func=lambda x: f"{x[0]} / {x[1]}",
        )

        st.divider()
        st.caption("Tip: En Modo Demo ya hay decisiones precargadas para recorrer la historia completa.")

    # Persist role in session
    st.session_state["role"] = role
    return demo_on, role, category, canal, retailer


def render_demo_runbook() -> None:
    st.markdown("### Demo Runbook (3 pasos)")
    cols = st.columns(3)
    if cols[0].button("Step 1: Decidir", use_container_width=True):
        st.session_state["mode"] = "Decidir"
    if cols[1].button("Step 2: Gobernar", use_container_width=True):
        st.session_state["mode"] = "Gobernar"
    if cols[2].button("Step 3: Defender", use_container_width=True):
        st.session_state["mode"] = "Defender"

    st.markdown(
        "- Muestra alternativas comparables (no reportes)\n"
        "- Muestra trazabilidad (decisión → KPI → ajuste)\n"
        "- Muestra defensa económica frente al retailer"
    )


def render_mode_selector() -> str:
    if "mode" not in st.session_state:
        st.session_state["mode"] = "Decidir"
    mode = st.radio("Modo", ["Decidir", "Gobernar", "Defender"], horizontal=True, index=["Decidir", "Gobernar", "Defender"].index(st.session_state["mode"]))
    st.session_state["mode"] = mode
    return mode


def _context_df(df: pd.DataFrame, context: dict[str, str]) -> pd.DataFrame:
    return df[
        (df["categoria"] == context["category"])
        & (df["canal"] == context["channel"])
        & (df["retailer"] == context["retailer"])
    ].copy()


def _fmt_money(x: float) -> str:
    # MXN simplificado
    return f"${x:,.0f} MXN"


def _kpi_cards(results: list[ScenarioResult]) -> None:
    cols = st.columns(3)
    names = ["Base", "A", "B"]
    for i, r in enumerate(results):
        with cols[i]:
            with st.container(border=True):
                st.subheader(names[i])
                st.caption(r.name)
                st.metric("Ventas (valor)", _fmt_money(r.kpis["ventas_valor"]))
                st.metric("Cuervo profit", _fmt_money(r.kpis["cuervo_profit"]))
                st.metric("Retailer profit", _fmt_money(r.kpis["retailer_profit"]))
                st.metric("Riesgo (0–1)", f"{r.risk:.2f}")


def _scenario_compare_chart(results: list[ScenarioResult]) -> go.Figure:
    base = results[0]
    labels = [r.name for r in results]
    vals = [r.kpis["cuervo_profit"] for r in results]
    fig = go.Figure(
        data=[
            go.Bar(
                x=labels,
                y=vals,
            )
        ]
    )
    fig.update_layout(
        title="Cuervo Profit — comparación por escenario",
        xaxis_title="Escenario",
        yaxis_title="MXN",
        height=420,
        margin=dict(l=20, r=20, t=60, b=20),
    )
    return fig


def render_mode_decidir(df: pd.DataFrame, context: dict[str, str]) -> None:
    st.subheader("User Story 1 — Decidir (ex-ante)")

    df_ctx = _context_df(df, context)
    if df_ctx.empty:
        st.info("No hay datos para el contexto seleccionado. Cambia categoría/canal/retailer.")
        return

    left, right = st.columns([1, 2])

    with left:
        with st.container(border=True):
            st.markdown("**Contexto**")
            st.write(f"**Rol:** {context['role']}")
            st.write(f"**Categoría:** {context['category']}")
            st.write(f"**Canal:** {context['channel']}")
            st.write(f"**Retailer:** {context['retailer']}")

            objective = st.selectbox("Objetivo económico", ["Rentabilidad", "Crecimiento"], index=0)

            st.markdown("**Palancas (demo)**")
            price = st.slider("Ajuste de precio (%)", -10.0, 10.0, 2.0, 0.5)
            sos = st.slider("Ajuste de SOS (puntos)", -5.0, 5.0, 1.0, 0.5)
            topn = st.slider("Top-N SKUs (opcional)", 0, 120, 80, 10)
            topn_val = None if topn == 0 else int(topn)

    results = build_standard_scenarios(df_ctx, price_slider=price, sos_slider=sos, topn=topn_val)

    with right:
        _kpi_cards(results)
        st.plotly_chart(_scenario_compare_chart(results), use_container_width=True)

        with st.container(border=True):
            st.markdown("**Elegir escenario y registrar decisión**")
            scenario_choice = st.selectbox("Escenario elegido", [r.name for r in results], index=1)

            # rentabilidad como condición mínima (proxy)
            chosen = next(r for r in results if r.name == scenario_choice)
            margin_floor = 0.18
            margin_ok = chosen.kpis["margen_pct_avg"] >= margin_floor and chosen.kpis["cuervo_profit"] > 0

            override = False
            justification = ""
            if not margin_ok:
                st.warning("Rentabilidad bajo piso (condición mínima). Para registrar, requiere override con justificación.")
                override = st.checkbox("Override con justificación")
                if override:
                    justification = st.text_input("Justificación (breve)")

            can_register = margin_ok or (override and len(justification.strip()) > 5)

            if st.button("Registrar decisión", use_container_width=True, disabled=not can_register):
                decision_id = add_decision(
                    {
                        "role": context["role"],
                        "categoria": context["category"],
                        "canal": context["channel"],
                        "retailer": context["retailer"],
                        "objective": objective,
                        "scenario": chosen.name,
                        "assumptions": chosen.assumptions,
                        "kpis_expected": chosen.kpis,
                        "risk": chosen.risk,
                        "status": "Aprobada",
                        "validated_execution": False,
                        "notes": justification if override else "Registrada sin override.",
                    }
                )
                st.success(f"Decisión registrada: {decision_id}")


def _semáforo(delta_pct: float) -> str:
    # <=3% verde, <=7% amarillo, >7% rojo
    ad = abs(delta_pct)
    if ad <= 3:
        return "🟢"
    if ad <= 7:
        return "🟡"
    return "🔴"


def render_mode_gobernar(df: pd.DataFrame, context: dict[str, str]) -> None:
    st.subheader("User Story 2 — Gobernar (durante ejecución)")

    log_df = to_dataframe()
    if log_df.empty:
        st.info("No hay decisiones registradas. Activa Modo Demo o registra una decisión en 'Decidir'.")
        return

    with st.container(border=True):
        st.markdown("**Filtros**")
        c1, c2, c3 = st.columns(3)
        with c1:
            f_cat = st.selectbox("Categoría", ["(todas)"] + sorted(log_df["categoria"].dropna().unique().tolist()))
        with c2:
            f_status = st.selectbox("Status", ["(todos)"] + sorted(log_df["status"].dropna().unique().tolist()))
        with c3:
            f_role = st.selectbox("Rol (creador)", ["(todos)"] + sorted(log_df["role"].dropna().unique().tolist()))

    f = log_df.copy()
    if f_cat != "(todas)":
        f = f[f["categoria"] == f_cat]
    if f_status != "(todos)":
        f = f[f["status"] == f_status]
    if f_role != "(todos)":
        f = f[f["role"] == f_role]

    st.dataframe(
        f[["decision_id", "version", "categoria", "canal", "retailer", "objective", "scenario", "risk", "status"]],
        use_container_width=True,
        hide_index=True,
    )

    decision_ids = f["decision_id"].dropna().unique().tolist()
    if not decision_ids:
        st.info("No hay decisiones para los filtros seleccionados.")
        return

    selected_id = st.selectbox("Selecciona una decisión", decision_ids, index=0)
    rec = get_decision_by_id(selected_id)
    if not rec:
        st.error("No se pudo cargar la decisión.")
        return

    left, right = st.columns([1, 2])

    with left:
        with st.container(border=True):
            st.markdown("**Ficha de decisión**")
            st.write(f"**Decision ID:** {rec.get('decision_id')}  (v{rec.get('version', 1)})")
            st.write(f"**Status:** {rec.get('status')}")
            st.write(f"**Rol creador:** {rec.get('role')}")
            st.write(f"**Escenario:** {rec.get('scenario')}")
            st.write(f"**Riesgo:** {rec.get('risk'):.2f}")
            st.write("**Supuestos:**")
            st.json(rec.get("assumptions", {}), expanded=False)

    with right:
        # Simula real vs esperado
        expected = rec.get("kpis_expected", {})
        if not expected:
            st.warning("Esta decisión no tiene KPIs esperados.")
            return

        seed = int(rec.get("demo_deviation_seed", 7))
        rng = np.random.default_rng(seed)
        # desviaciones controladas
        real = dict(expected)
        for k in ["ventas_valor", "cuervo_profit", "retailer_profit"]:
            real[k] = float(expected[k]) * float(1.0 + rng.normal(0.0, 0.06))

        # tabla semáforos
        rows = []
        for k in ["ventas_valor", "cuervo_profit", "retailer_profit"]:
            delta_pct = 100.0 * (real[k] - expected[k]) / max(expected[k], 1e-9)
            rows.append(
                {
                    "KPI": k,
                    "Esperado": expected[k],
                    "Real": real[k],
                    "Δ%": delta_pct,
                    "Semáforo": _semáforo(delta_pct),
                }
            )
        kpi_df = pd.DataFrame(rows)

        with st.container(border=True):
            st.markdown("**Seguimiento (Real vs Esperado)**")
            st.dataframe(kpi_df, use_container_width=True, hide_index=True)

        with st.container(border=True):
            st.markdown("**Acciones**")
            a1, a2, a3 = st.columns(3)
            with a1:
                if st.button("Validar ejecución", use_container_width=True):
                    update_decision(selected_id, {"validated_execution": True})
                    st.success("Ejecución validada.")
                    st.rerun()
            with a2:
                if st.button("Ajustar supuestos", use_container_width=True):
                    st.session_state["show_adjust"] = True
            with a3:
                if st.button("Cerrar decisión", use_container_width=True):
                    update_decision(selected_id, {"status": "Cerrada"})
                    st.success("Decisión cerrada.")
                    st.rerun()

        if st.session_state.get("show_adjust", False):
            with st.container(border=True):
                st.markdown("**Ajuste de supuestos (nueva versión)**")
                new_price = st.number_input("Nuevo price_delta_pct", value=float(rec.get("assumptions", {}).get("price_delta_pct", 0.0)))
                new_sos = st.number_input("Nuevo sos_delta_points", value=float(rec.get("assumptions", {}).get("sos_delta_points", 0.0)))
                note = st.text_input("Nota de ajuste (breve)")
                if st.button("Crear nueva versión", use_container_width=True):
                    create_new_version(
                        selected_id,
                        {
                            "assumptions": {
                                "price_delta_pct": float(new_price),
                                "sos_delta_points": float(new_sos),
                                "topn_skus": rec.get("assumptions", {}).get("topn_skus"),
                            },
                            "notes": note,
                            "status": "En seguimiento",
                        },
                    )
                    st.session_state["show_adjust"] = False
                    st.success("Nueva versión creada.")
                    st.rerun()

    # Exports
    with st.expander("Exportar Decision Log", expanded=False):
        st.download_button("Descargar CSV", data=export_csv_bytes(), file_name="decision_log.csv", mime="text/csv")
        st.download_button("Descargar JSON", data=export_json_bytes(), file_name="decision_log.json", mime="application/json")


def _one_pager_md(rec: dict[str, Any], baseline: dict[str, float]) -> str:
    exp = rec.get("kpis_expected", {})
    a = rec.get("assumptions", {})
    bullets = [
        f"**Qué cambia:** {rec.get('scenario')} con supuestos {a}",
        f"**Beneficio retailer (proxy):** {_fmt_money(exp.get('retailer_profit', 0.0))}",
        f"**Beneficio Cuervo:** {_fmt_money(exp.get('cuervo_profit', 0.0))}",
        f"**Ventas (valor):** {_fmt_money(exp.get('ventas_valor', 0.0))}",
        f"**Riesgo:** {rec.get('risk', 0.0):.2f}",
    ]
    bullets = bullets[:7]
    md = [
        f"# One-pager — {rec.get('categoria')} | {rec.get('canal')} / {rec.get('retailer')}",
        "",
        f"**Decision ID:** {rec.get('decision_id')} (v{rec.get('version', 1)})",
        f"**Objetivo:** {rec.get('objective')}",
        "",
        "## Retailer Story (proxy)",
        "",
        "\n".join([f"- {b}" for b in bullets]),
        "",
        "## KPIs (esperado)",
        "",
        f"- Ventas (valor): {_fmt_money(exp.get('ventas_valor', 0.0))}",
        f"- Cuervo profit: {_fmt_money(exp.get('cuervo_profit', 0.0))}",
        f"- Retailer profit: {_fmt_money(exp.get('retailer_profit', 0.0))}",
        "",
        "## Nota",
        "Documento demo; métricas sintéticas y proxies para ilustrar defensa cuantitativa.",
    ]
    return "\n".join(md)


def render_mode_defender(df: pd.DataFrame, context: dict[str, str]) -> None:
    st.subheader("User Story 3 — Defender (frente al retailer)")

    rows = list_decisions()
    if not rows:
        st.info("No hay decisiones registradas. Activa Modo Demo o registra una decisión en 'Decidir'.")
        return

    # Selecciona decisiones no cerradas
    candidates = [r for r in rows if r.get("status") != "Cerrada"]
    if not candidates:
        st.info("No hay decisiones vigentes (todas están cerradas).")
        return

    options = [f"{r.get('decision_id')} (v{r.get('version', 1)}) — {r.get('categoria')} / {r.get('retailer')}" for r in candidates]
    idx = st.selectbox("Selecciona una decisión vigente", list(range(len(options))), format_func=lambda i: options[i], index=0)
    rec = candidates[int(idx)]

    # baseline del contexto de esa decisión
    df_ctx = df[(df["categoria"] == rec.get("categoria")) & (df["canal"] == rec.get("canal")) & (df["retailer"] == rec.get("retailer"))].copy()
    if df_ctx.empty:
        df_ctx = df.head(800).copy()
    baseline = compute_baseline(df_ctx)

    exp = rec.get("kpis_expected", {})

    left, right = st.columns([1, 2])

    with left:
        with st.container(border=True):
            st.markdown("**Retailer Story (bullets)**")
            bullets = [
                f"Qué cambia: {rec.get('scenario')}",
                f"Beneficio retailer (proxy): {_fmt_money(exp.get('retailer_profit', 0.0))}",
                f"Beneficio Cuervo: {_fmt_money(exp.get('cuervo_profit', 0.0))}",
                f"Ventas (valor): {_fmt_money(exp.get('ventas_valor', 0.0))}",
                f"Benchmark competitivo: proxy (sintético)",
                f"Supuestos clave: {rec.get('assumptions', {})}",
            ]
            bullets = bullets[:7]
            for b in bullets:
                st.write(f"• {b}")

        with st.container(border=True):
            st.download_button(
                "Generar one-pager (.md)",
                data=_one_pager_md(rec, baseline).encode("utf-8"),
                file_name=f"one_pager_{rec.get('decision_id')}.md",
                mime="text/markdown",
                use_container_width=True,
            )

    with right:
        # Visual principal: baseline vs esperado
        labels = ["Baseline", "Escenario"]
        y1 = [baseline.get("ventas_valor", 0.0), exp.get("ventas_valor", 0.0)]
        y2 = [baseline.get("retailer_profit", 0.0), exp.get("retailer_profit", 0.0)]

        fig = go.Figure()
        fig.add_trace(go.Bar(x=labels, y=y1, name="Ventas (valor)"))
        fig.add_trace(go.Bar(x=labels, y=y2, name="Retailer profit"))
        fig.update_layout(
            title="Defensa cuantitativa — Baseline vs Escenario",
            barmode="group",
            height=420,
            margin=dict(l=20, r=20, t=60, b=20),
        )
        st.plotly_chart(fig, use_container_width=True)
