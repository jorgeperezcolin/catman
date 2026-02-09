from __future__ import annotations

import streamlit as st

from src.data_gen import get_base_data, get_category_options, get_channel_retailer_options
from src.decision_store import ensure_store_initialized
from src.demo_mode import apply_demo_mode_if_enabled, reset_demo
from src.ui_components import (
    render_header,
    render_sidebar_controls,
    render_demo_runbook,
    render_mode_selector,
    render_mode_decidir,
    render_mode_gobernar,
    render_mode_defender,
)

APP_TITLE = "Decision Cockpit de Categoría (Maqueta)"
APP_ICON = "📊"


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, page_icon=APP_ICON, layout="wide")

    ensure_store_initialized()

    # --- Sidebar controls (incluye Modo Demo) ---
    demo_on, role, category, channel, retailer = render_sidebar_controls(
        category_options=get_category_options(),
        channel_retailer_options=get_channel_retailer_options(),
    )

    # --- Base data (cached) ---
    df = get_base_data()

    # --- Demo mode (precarga + runbook + reset) ---
    if demo_on:
        apply_demo_mode_if_enabled(df=df)

    render_header()

    if demo_on:
        with st.container(border=True):
            cols = st.columns([1, 1, 2])
            with cols[0]:
                if st.button("Reset Demo", use_container_width=True):
                    reset_demo()
                    st.rerun()
            with cols[1]:
                st.caption("Modo Demo: ON")
            with cols[2]:
                render_demo_runbook()

    # --- Single cockpit with 3 modes ---
    mode = render_mode_selector()

    # --- Route to mode views ---
    context = {
        "role": role,
        "category": category,
        "channel": channel,
        "retailer": retailer,
    }

    if mode == "Decidir":
        render_mode_decidir(df=df, context=context)
    elif mode == "Gobernar":
        render_mode_gobernar(df=df, context=context)
    else:
        render_mode_defender(df=df, context=context)


if __name__ == "__main__":
    main()
