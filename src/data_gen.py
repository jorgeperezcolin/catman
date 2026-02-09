from __future__ import annotations

from dataclasses import dataclass
import numpy as np
import pandas as pd
import streamlit as st


@dataclass(frozen=True)
class DomainConfig:
    categories: tuple[str, ...] = ("Tequila Core", "RTDs", "Whisky", "Licores", "Mezcal")
    channel_retailer: tuple[tuple[str, str], ...] = (
        ("Moderno", "Walmart"),
        ("Moderno", "Otros Moderno"),
        ("Mayoreo", "Mayoreo"),
    )


CFG = DomainConfig()


def get_category_options() -> list[str]:
    return list(CFG.categories)


def get_channel_retailer_options() -> list[tuple[str, str]]:
    return list(CFG.channel_retailer)


@st.cache_data(show_spinner=False)
def get_base_data(seed: int = 7) -> pd.DataFrame:
    """
    Genera datos sintéticos reproducibles para demo.
    Columnas principales:
      categoria, canal, retailer, sku, precio, unidades, ventas_valor, margen_pct, sos, som
      retailer_profit, cuervo_profit
      elasticity (para escenario)
    """
    rng = np.random.default_rng(seed)

    rows: list[dict] = []
    for cat in CFG.categories:
        # elasticidad sintética por categoría (magnitud típica)
        base_elasticity = float(rng.uniform(0.8, 2.2))
        n_skus = int(rng.integers(80, 160))

        for (canal, retailer) in CFG.channel_retailer:
            # Ajustes por canal
            canal_mult = 1.0 if canal == "Moderno" else 0.8
            retailer_mult = 1.05 if retailer == "Walmart" else 1.0

            # Generación SKU-level
            for i in range(n_skus):
                sku = f"{cat[:3].upper()}-{canal[:2].upper()}-{i:03d}"
                precio = float(np.clip(rng.normal(320, 110), 90, 900))
                unidades = float(np.clip(rng.lognormal(mean=7.3, sigma=0.6), 400, 60000)) * canal_mult * retailer_mult
                ventas_valor = precio * unidades

                margen_pct = float(np.clip(rng.normal(0.36, 0.08), 0.12, 0.62))
                cuervo_profit = ventas_valor * margen_pct

                # proxy retailer: asume margen menor + efecto rotación
                retailer_margin_pct = float(np.clip(rng.normal(0.22, 0.05), 0.08, 0.38))
                retailer_profit = ventas_valor * retailer_margin_pct

                sos = float(np.clip(rng.normal(12, 4), 2, 30))  # share of shelf points
                som = float(np.clip(rng.normal(10, 3), 1, 25))  # share of market points

                # pequeña variación por canal/retailer en elasticidad
                elasticity = base_elasticity * (1.05 if canal == "Moderno" else 0.95) * (1.03 if retailer == "Walmart" else 1.0)

                rows.append(
                    dict(
                        categoria=cat,
                        canal=canal,
                        retailer=retailer,
                        sku=sku,
                        precio=precio,
                        unidades=unidades,
                        ventas_valor=ventas_valor,
                        margen_pct=margen_pct,
                        sos=sos,
                        som=som,
                        retailer_profit=retailer_profit,
                        cuervo_profit=cuervo_profit,
                        elasticity=elasticity,
                    )
                )

    df = pd.DataFrame(rows)

    # Normaliza SOM/SOS de forma coherente por categoría/canal/retailer
    grp = df.groupby(["categoria", "canal", "retailer"], as_index=False)
    df["sos"] = grp["sos"].transform(lambda x: 100 * x / x.sum())
    df["som"] = grp["som"].transform(lambda x: 100 * x / x.sum())

    return df
