from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any
import pandas as pd
import streamlit as st


STORE_KEY = "decision_log"  # list[dict]
STORE_META_KEY = "decision_meta"  # dict


def ensure_store_initialized() -> None:
    if STORE_KEY not in st.session_state:
        st.session_state[STORE_KEY] = []
    if STORE_META_KEY not in st.session_state:
        st.session_state[STORE_META_KEY] = {"next_id": 1}


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_decision_id() -> str:
    meta = st.session_state[STORE_META_KEY]
    n = int(meta.get("next_id", 1))
    meta["next_id"] = n + 1
    return f"DC-2026-{n:04d}"


def add_decision(record: dict[str, Any]) -> str:
    ensure_store_initialized()
    decision_id = _new_decision_id()
    record = dict(record)
    record.setdefault("decision_id", decision_id)
    record.setdefault("created_at", _now_iso())
    record.setdefault("version", 1)
    st.session_state[STORE_KEY].append(record)
    return decision_id


def list_decisions() -> list[dict[str, Any]]:
    ensure_store_initialized()
    return list(st.session_state[STORE_KEY])


def to_dataframe() -> pd.DataFrame:
    rows = list_decisions()
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)


def get_decision_by_id(decision_id: str) -> dict[str, Any] | None:
    for r in list_decisions():
        if r.get("decision_id") == decision_id:
            return r
    return None


def update_decision(decision_id: str, updates: dict[str, Any]) -> bool:
    ensure_store_initialized()
    for i, r in enumerate(st.session_state[STORE_KEY]):
        if r.get("decision_id") == decision_id:
            nr = dict(r)
            nr.update(updates)
            nr["updated_at"] = _now_iso()
            st.session_state[STORE_KEY][i] = nr
            return True
    return False


def create_new_version(decision_id: str, updates: dict[str, Any]) -> str | None:
    """
    Crea una nueva decisión versionada (mismo decision_id + version increment) manteniendo historial.
    """
    r = get_decision_by_id(decision_id)
    if not r:
        return None
    base = dict(r)
    base["version"] = int(base.get("version", 1)) + 1
    base["created_at"] = _now_iso()
    base.update(updates)
    # En esta maqueta, guardamos como registro separado para trazabilidad
    st.session_state[STORE_KEY].append(base)
    return base["decision_id"]


def export_csv_bytes() -> bytes:
    df = to_dataframe()
    if df.empty:
        return b""
    return df.to_csv(index=False).encode("utf-8")


def export_json_bytes() -> bytes:
    rows = list_decisions()
    return json.dumps(rows, ensure_ascii=False, indent=2).encode("utf-8")
