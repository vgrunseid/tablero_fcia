from __future__ import annotations

import base64
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import streamlit as st


st.set_page_config(page_title="Tablero ROP", layout="wide")

st.markdown(
    """
<style>
.stApp { background-color: #0B1736; }
.block-container { padding-top: 2rem; max-width: 1200px; }
h1, h2, h3, h4, h5, h6, p, label, span, div { color: #ffffff; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }

[data-testid="stFileUploader"] section {
    background-color: #10204d !important;
    border-radius: 10px;
    border: 2px dashed #3e5cb2;
}
[data-testid="stFileUploader"] section div { color: #E6E6E6 !important; }
[data-testid="stFileUploader"] span { color: #E6E6E6 !important; }

.stButton > button, .stDownloadButton > button, [data-testid="stFileUploader"] button {
  background-color: #1f6feb !important;
  color: #ffffff !important;
  border: 0;
  border-radius: 10px;
  padding: 0.6rem 1rem;
  font-weight: 600;
}
.stButton > button:hover, .stDownloadButton > button:hover, [data-testid="stFileUploader"] button:hover {
  opacity: 0.9;
}

.section-sep { margin: 1.5rem 0 0.5rem 0; height: 1px; background: #1b2d5f; }

section[data-testid="stSidebar"] { background-color: #0B1736; }
section[data-testid="stSidebar"] label { color: #ffffff; }
section[data-testid="stSidebar"] input { color: #ffffff; background-color: #10204d; }
section[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #10204d;
    color: #ffffff;
    border: 1px solid #3e5cb2;
}
section[data-testid="stSidebar"] div[data-baseweb="select"] span { color: #ffffff; }
section[data-testid="stSidebar"] [data-baseweb="select"] svg { fill: #ffffff; }
section[data-testid="stSidebar"] div[data-baseweb="popover"] {
    background-color: #10204d;
    color: #ffffff;
}
section[data-testid="stSidebar"] div[data-baseweb="menu"] {
    background-color: #10204d;
    color: #ffffff;
}
section[data-testid="stSidebar"] div[data-baseweb="option"] {
    background-color: #10204d;
    color: #ffffff;
}
section[data-testid="stSidebar"] div[data-baseweb="option"]:hover {
    background-color: #1b2d5f;
}

.stApp div[data-baseweb="popover"] {
    background-color: #10204d !important;
    color: #ffffff !important;
}
.stApp div[data-baseweb="popover"] > div {
    background-color: #10204d !important;
}
.stApp div[data-baseweb="popover"] ul {
    background-color: #10204d !important;
}
.stApp div[data-baseweb="menu"] {
    background-color: #10204d !important;
    color: #ffffff !important;
}
.stApp div[data-baseweb="menu"] ul {
    background-color: #10204d !important;
}
.stApp div[data-baseweb="option"] {
    background-color: #10204d !important;
    color: #ffffff !important;
}
.stApp div[data-baseweb="option"] * {
    color: #ffffff !important;
}
.stApp div[data-baseweb="option"]:hover {
    background-color: #1b2d5f !important;
}
.stApp [role="listbox"] {
    background-color: #10204d !important;
    color: #ffffff !important;
}
.stApp [data-baseweb="popover"] [role="listbox"] {
    background-color: #10204d !important;
}
.stApp [data-baseweb="popover"] [role="listbox"] > div {
    background-color: #10204d !important;
}
.stApp [role="option"] {
    background-color: #10204d !important;
    color: #ffffff !important;
}
.stApp [role="option"] * {
    color: #ffffff !important;
}
.stApp [role="option"]:hover {
    background-color: #1b2d5f !important;
}

body div[data-baseweb="popover"] {
    background-color: #10204d !important;
    color: #ffffff !important;
}
body div[data-baseweb="popover"] > div,
body div[data-baseweb="popover"] ul {
    background-color: #10204d !important;
}
body [role="listbox"] {
    background-color: #10204d !important;
    color: #ffffff !important;
}
body [role="option"] {
    background-color: #10204d !important;
    color: #ffffff !important;
}
body [role="option"] * {
    color: #ffffff !important;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def load_movimientos(data_source: Path | None, uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file, encoding="latin1")
    else:
        if data_source is None or not data_source.exists():
            raise FileNotFoundError("No se encuentra movimientos.csv")
        df = pd.read_csv(data_source, encoding="latin1")

    df.columns = df.columns.str.strip().str.lower()
    df["fecha"] = pd.to_datetime(df["fecha"], errors="coerce")
    df["codigo"] = df["codigo"].astype(str).str.strip()
    df["tipo"] = df["tipo"].astype(str).str.strip()
    df["descrip"] = df["descrip"].astype(str).str.strip()

    for c in ["stock", "antes", "cantid", "despues"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    return df


@st.cache_data(show_spinner=False)
def load_rop_sugerido(data_source: Path | None, uploaded_file) -> pd.DataFrame:
    if uploaded_file is not None:
        rop_df = pd.read_csv(uploaded_file)
    else:
        if data_source is None or not data_source.exists():
            return pd.DataFrame()
        rop_df = pd.read_csv(data_source)

    if "codigo" not in rop_df.columns:
        return pd.DataFrame()

    rop_df["codigo"] = rop_df["codigo"].astype(str).str.strip()
    return rop_df


@st.cache_data(show_spinner=False)
def compute_rop(df: pd.DataFrame):
    max_date = df["fecha"].max()
    all_days = pd.date_range(end=max_date, periods=365, freq="D")
    ventana_inicio = all_days.min()

    ventas = df[
        ((df["tipo"].str.contains("venta", case=False, na=False)) & (df["cantid"] < 0))
        | ((df["tipo"].str.contains("anulaci", case=False, na=False)) & (df["cantid"] > 0))
    ].copy()

    ventas = ventas[~ventas["tipo"].str.contains("inventario", case=False, na=False)]
    ventas["unidades"] = -ventas["cantid"]
    ventas = ventas[ventas["codigo"] != "-900006"].copy()

    ventas_diarias = (
        ventas.groupby(["codigo", "fecha"])["unidades"].sum().reset_index()
    )

    ventas_1y = ventas_diarias[ventas_diarias["fecha"].between(ventana_inicio, max_date)].copy()
    ventas_1y["codigo"] = ventas_1y["codigo"].astype(str).str.strip()

    p90_cut = ventas_1y.groupby("codigo")["unidades"].quantile(0.90).rename("p90_cut")
    p95_cut = ventas_1y.groupby("codigo")["unidades"].quantile(0.95).rename("p95_cut")
    p99_cut = ventas_1y.groupby("codigo")["unidades"].quantile(0.99).rename("p99_cut")

    qdf = (
        ventas_1y.merge(p90_cut, on="codigo")
        .merge(p95_cut, on="codigo")
        .merge(p99_cut, on="codigo")
    )

    p90_1y = (
        qdf[qdf["unidades"] <= qdf["p90_cut"]]
        .groupby("codigo")["unidades"]
        .max()
        .reset_index(name="p90_1y")
    )
    p95_1y = (
        qdf[qdf["unidades"] <= qdf["p95_cut"]]
        .groupby("codigo")["unidades"]
        .max()
        .reset_index(name="p95_1y")
    )
    p99_1y = (
        qdf[qdf["unidades"] <= qdf["p99_cut"]]
        .groupby("codigo")["unidades"]
        .max()
        .reset_index(name="p99_1y")
    )

    venta_max_1y = (
        ventas_1y.groupby("codigo")["unidades"].max().reset_index(name="venta_max_diaria_1y")
    )

    base = (
        df[["codigo", "descrip"]]
        .drop_duplicates()
        .merge(venta_max_1y, on="codigo", how="left")
        .merge(p90_1y, on="codigo", how="left")
        .merge(p95_1y, on="codigo", how="left")
        .merge(p99_1y, on="codigo", how="left")
    )

    for c in ["venta_max_diaria_1y", "p90_1y", "p95_1y", "p99_1y"]:
        base[c] = pd.to_numeric(base[c], errors="coerce").fillna(0)

    for c in ["p90_1y", "p95_1y", "p99_1y"]:
        base[c] = np.ceil(base[c]).clip(lower=0)
        base[c] = np.minimum(base[c], base["venta_max_diaria_1y"])

    base["ROP_ajustado"] = base["p90_1y"]
    base["ROP_intermedio"] = base["p95_1y"]
    base["ROP_holgado"] = base["p99_1y"]

    mask = (base["ROP_ajustado"] == base["ROP_intermedio"]) & (base["ROP_ajustado"] > 0)
    base.loc[mask, "ROP_ajustado"] = base.loc[mask, "ROP_intermedio"] - 1

    codigos = base["codigo"].astype(str).str.strip().unique()
    full_idx = pd.MultiIndex.from_product([codigos, all_days], names=["codigo", "fecha"])

    ventas_1y_full = (
        ventas_1y.set_index(["codigo", "fecha"])["unidades"]
        .reindex(full_idx, fill_value=0)
        .reset_index()
    )

    ventas_1y_full = ventas_1y_full.merge(
        base[["codigo", "ROP_ajustado", "ROP_intermedio", "ROP_holgado"]],
        on="codigo",
        how="left",
    )

    ventas_1y_full["venta_pos"] = ventas_1y_full["unidades"] > 0
    ventas_1y_full["quiebre_ajustado"] = ventas_1y_full["unidades"] > ventas_1y_full["ROP_ajustado"]
    ventas_1y_full["quiebre_intermedio"] = ventas_1y_full["unidades"] > ventas_1y_full["ROP_intermedio"]
    ventas_1y_full["quiebre_holgado"] = ventas_1y_full["unidades"] > ventas_1y_full["ROP_holgado"]

    rates = (
        ventas_1y_full.groupby("codigo")[
            ["venta_pos", "quiebre_ajustado", "quiebre_intermedio", "quiebre_holgado"]
        ]
        .sum()
        .reset_index()
    )

    for qcol, pcol in [
        ("quiebre_ajustado", "%_quiebre_ajustado"),
        ("quiebre_intermedio", "%_quiebre_intermedio"),
        ("quiebre_holgado", "%_quiebre_holgado"),
    ]:
        rates[pcol] = np.where(rates["venta_pos"] > 0, rates[qcol] / rates["venta_pos"] * 100, np.nan)

    tmp_rates = rates.set_index("codigo")
    mask_h = (
        (base["ROP_holgado"] == base["ROP_intermedio"])
        & (base["codigo"].map(tmp_rates["%_quiebre_holgado"]) > 0)
    )
    base.loc[mask_h, "ROP_holgado"] = base.loc[mask_h, "ROP_holgado"] + 1

    ventas_1y_full = ventas_1y_full.drop(columns=["ROP_holgado", "quiebre_holgado"])
    ventas_1y_full = ventas_1y_full.merge(base[["codigo", "ROP_holgado"]], on="codigo", how="left")
    ventas_1y_full["quiebre_holgado"] = ventas_1y_full["unidades"] > ventas_1y_full["ROP_holgado"]

    holgado_recalc = (
        ventas_1y_full.groupby("codigo")["quiebre_holgado"].sum().reset_index(name="quiebre_holgado")
    )

    rates = rates.drop(columns=["quiebre_holgado", "%_quiebre_holgado"])
    rates = rates.merge(holgado_recalc, on="codigo", how="left")
    rates["%_quiebre_holgado"] = np.where(rates["venta_pos"] > 0, rates["quiebre_holgado"] / rates["venta_pos"] * 100, np.nan)

    rop_target = base.merge(rates, on="codigo", how="left").copy()

    stock_diario = (
        df.sort_values(["codigo", "fecha", "hora"])
        .groupby(["codigo", "fecha"])["despues"]
        .min()
        .reset_index(name="min_stock")
    )

    ventas_dia = ventas_diarias[ventas_diarias["unidades"] > 0].copy()
    stock_diario = stock_diario[stock_diario["fecha"] >= ventana_inicio].copy()
    ventas_dia = ventas_dia[ventas_dia["fecha"] >= ventana_inicio].copy()

    stock_venta = stock_diario.merge(ventas_dia, on=["codigo", "fecha"], how="left")
    stock_venta["hubo_venta"] = stock_venta["unidades"].fillna(0) > 0
    stock_venta["quiebre_real"] = (stock_venta["min_stock"] < 0) & stock_venta["hubo_venta"]

    real_quiebre = (
        stock_venta.groupby("codigo")
        .apply(
            lambda g: pd.Series(
                {
                    "dias_quiebre_real": int(g["quiebre_real"].sum()),
                    "%_quiebre_real": (g["quiebre_real"].mean() * 100) if g["hubo_venta"].any() else np.nan,
                }
            )
        )
        .reset_index()
    )

    rop_target = rop_target.drop(columns=["dias_quiebre_real", "%_quiebre_real"], errors="ignore")
    rop_target = rop_target.merge(real_quiebre, on="codigo", how="left")

    resumen = rop_target[
        [
            "codigo",
            "descrip",
            "ROP_ajustado",
            "ROP_intermedio",
            "ROP_holgado",
            "%_quiebre_ajustado",
            "%_quiebre_intermedio",
            "%_quiebre_holgado",
            "venta_pos",
            "venta_max_diaria_1y",
            "dias_quiebre_real",
            "%_quiebre_real",
        ]
    ].copy()

    for c in ["%_quiebre_ajustado", "%_quiebre_intermedio", "%_quiebre_holgado", "%_quiebre_real"]:
        resumen[c] = resumen[c].round(2)

    quantiles = (
        qdf.groupby("codigo")[["p90_cut", "p95_cut", "p99_cut"]].max().reset_index()
    )

    stock_real_diario = (
        df.sort_values(["codigo", "fecha", "hora"])
        .groupby(["codigo", "fecha"])["despues"]
        .last()
        .reset_index(name="stock_real_dia")
    )
    stock_real_diario = stock_real_diario[stock_real_diario["fecha"].between(ventana_inicio, max_date)].copy()

    return resumen, ventas_1y_full, stock_real_diario, quantiles, ventana_inicio, max_date


def plot_producto(df, resumen, ventas_1y_full, stock_real_diario, quantiles, codigo: str):
    info = resumen[resumen["codigo"] == codigo]
    if info.empty:
        return None
    info = info.iloc[0]
    desc = info["descrip"]

    d = (
        ventas_1y_full[ventas_1y_full["codigo"] == codigo][["fecha", "unidades", "venta_pos"]]
        .sort_values("fecha")
        .copy()
    )

    s = (
        stock_real_diario[stock_real_diario["codigo"] == codigo][["fecha", "stock_real_dia"]]
        .sort_values("fecha")
        .copy()
    )

    pos = d[d["unidades"] > 0]["unidades"]
    q_row = quantiles[quantiles["codigo"] == codigo]
    if q_row.empty:
        return None
    q = q_row.iloc[0]

    c_demanda = "#1f77b4"
    c_stock = "#b0b0b0"
    c_ajustado = "#d62728"
    c_intermedio = "#2ca02c"
    c_holgado = "#9467bd"

    fig, axs = plt.subplots(2, 2, figsize=(15, 9))
    fig.suptitle(f"{desc} ({codigo}) - Analisis de escenarios ROP", fontsize=13)

    axs[0, 0].plot(d["fecha"], d["unidades"], lw=1.0, alpha=0.8, color=c_demanda, label="Unidades diarias")
    axs[0, 0].plot(
        s["fecha"],
        s["stock_real_dia"],
        lw=1.3,
        color=c_stock,
        alpha=0.9,
        label="Stock real diario (ultimo despues)",
    )

    axs[0, 0].axhline(info["ROP_ajustado"], ls="--", lw=1.8, color=c_ajustado, label=f"ROP ajustado={int(info['ROP_ajustado'])}")
    axs[0, 0].axhline(info["ROP_intermedio"], ls="--", lw=1.8, color=c_intermedio, label=f"ROP intermedio={int(info['ROP_intermedio'])}")
    axs[0, 0].axhline(info["ROP_holgado"], ls="--", lw=1.8, color=c_holgado, label=f"ROP holgado={int(info['ROP_holgado'])}")

    stock_vale = (
        df[df["codigo"] == codigo]
        .groupby("fecha")["despues"]
        .min()
        .reset_index(name="min_stock")
    )
    stock_vale = stock_vale[stock_vale["min_stock"] < 0]

    if not stock_vale.empty:
        axs[0, 0].scatter(stock_vale["fecha"], [0] * len(stock_vale), color="red", s=20, label="Vale (stock < 0)")

    axs[0, 0].set_title("Demanda diaria (ultimo ano), stock real y umbrales ROP")
    axs[0, 0].set_ylabel("Unidades")
    axs[0, 0].legend(loc="upper right", fontsize=8)
    axs[0, 0].yaxis.set_major_locator(MaxNLocator(integer=True))

    max_val = int(pos.max()) if len(pos) else 0
    bins = np.arange(0, max_val + 2)
    axs[0, 1].hist(pos, bins=bins, alpha=0.75, edgecolor="black", color="#9ecae1")
    axs[0, 1].set_xticks(range(0, max_val + 1))

    axs[0, 1].axvline(q["p90_cut"], ls="--", lw=1.5, color=c_ajustado, label=f"p90_cut={q['p90_cut']:.2f}")
    axs[0, 1].axvline(q["p95_cut"], ls="--", lw=1.5, color=c_intermedio, label=f"p95_cut={q['p95_cut']:.2f}")
    axs[0, 1].axvline(q["p99_cut"], ls="--", lw=1.5, color=c_holgado, label=f"p99_cut={q['p99_cut']:.2f}")
    axs[0, 1].set_title("Distribucion de dias con venta positiva")
    axs[0, 1].set_xlabel("Unidades/dia")
    axs[0, 1].legend(fontsize=8)

    labels = ["Ajustado", "Intermedio", "Holgado"]
    vals = [info["%_quiebre_ajustado"], info["%_quiebre_intermedio"], info["%_quiebre_holgado"]]
    axs[1, 0].bar(labels, vals, color=[c_ajustado, c_intermedio, c_holgado])
    axs[1, 0].set_title("% quiebre simulado por escenario")
    axs[1, 0].set_ylabel("% sobre dias con venta")
    for i, v in enumerate(vals):
        axs[1, 0].text(i, v + 0.3, f"{v:.1f}%", ha="center", fontsize=9)

    topd = d.sort_values("unidades", ascending=False).head(12).sort_values("fecha")
    axs[1, 1].bar(topd["fecha"].dt.strftime("%Y-%m-%d"), topd["unidades"], color=c_demanda)
    axs[1, 1].set_title("Top 12 dias de mayor demanda")
    axs[1, 1].tick_params(axis="x", rotation=75)
    axs[1, 1].set_ylabel("Unidades")
    axs[1, 1].yaxis.set_major_locator(MaxNLocator(integer=True))

    plt.tight_layout()
    return fig


def texto_conclusion(row: pd.Series) -> str:
    return (
        f"Para el producto {row['descrip']} ({row['codigo']}), el ROP sugerido es: "
        f"ajustado={int(row['ROP_ajustado'])}, intermedio={int(row['ROP_intermedio'])}, holgado={int(row['ROP_holgado'])}.\n"
        f"Esto se justifica porque, al usar demanda diaria del ultimo ano con percentiles filtrados, "
        f"el escenario ajustado deja un quiebre de {row['%_quiebre_ajustado']:.2f}%, "
        f"el intermedio baja a {row['%_quiebre_intermedio']:.2f}% "
        f"y el holgado a {row['%_quiebre_holgado']:.2f}% (sobre {int(row['venta_pos'])} dias con venta). "
        f"Ademas, el maximo diario observado en la ventana fue {int(row['venta_max_diaria_1y'])} y "
        f"el quiebre real historico por stock negativo en dias con venta fue {row['%_quiebre_real']:.2f}% "
        f"({int(row['dias_quiebre_real'])} dias)."
    )


col_logo, col_title = st.columns([1, 4], vertical_alignment="center")
with col_logo:
    ruta_logo = Path("logo_qbyx.png")
    if ruta_logo.exists():
        img_b64 = base64.b64encode(ruta_logo.read_bytes()).decode()
        st.markdown(
            f"""
            <a href="https://www.qbyxsolutions.com/" target="_blank" rel="noopener">
                <img src="data:image/png;base64,{img_b64}" style="max-width:100%; height:auto;"/>
            </a>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.caption("Subi tu logo como 'logo_qbyx.png' para mostrarlo aqui.")
with col_title:
    st.title("Tablero ROP Calculado")

st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
st.markdown("## Consulta")

st.sidebar.header("Datos")
movimientos_file = st.sidebar.file_uploader("movimientos.csv", type=["csv"])
rop_file = st.sidebar.file_uploader("rop_sugerido.csv (opcional)", type=["csv"])

try:
    df = load_movimientos(Path("data/movimientos.csv"), movimientos_file)
except FileNotFoundError:
    st.error("No se encontro data/movimientos.csv. Subi el archivo en la barra lateral.")
    st.stop()

resumen, ventas_1y_full, stock_real_diario, quantiles, ventana_inicio, max_date = compute_rop(df)
rop_sugerido = load_rop_sugerido(Path("data/rop_sugerido.csv"), rop_file)
extra_cols = ["AhorroCapitalAjustado", "ExcesoCapitalHolgado"]

if not rop_sugerido.empty and all(c in rop_sugerido.columns for c in extra_cols):
    resumen = resumen.merge(rop_sugerido[["codigo"] + extra_cols], on="codigo", how="left")
else:
    for c in extra_cols:
        if c not in resumen.columns:
            resumen[c] = pd.NA

def _use_selector():
    st.session_state["codigo_manual"] = ""


def _use_manual():
    st.session_state["codigo_selector"] = ""


selector_df = resumen[["codigo", "descrip"]].drop_duplicates().copy()
selector_df["codigo"] = selector_df["codigo"].astype(str).str.strip()
selector_df = selector_df.sort_values(["descrip", "codigo"])
selector_codigos = selector_df["codigo"].tolist()
desc_map = selector_df.set_index("codigo")["descrip"].to_dict()

codigo_preselect = st.sidebar.selectbox(
    "Producto (lista)",
    options=[""] + selector_codigos,
    index=0,
    format_func=lambda c: "" if c == "" else f"{desc_map.get(c, '')} ({c})",
    key="codigo_selector",
    on_change=_use_selector,
)
codigo_manual = st.text_input(
    "Codigo de medicamento",
    value="",
    key="codigo_manual",
    on_change=_use_manual,
).strip()

codigo = codigo_manual or codigo_preselect

if not codigo:
    st.info("Ingresa un codigo para ver los graficos.")
    st.stop()

if codigo not in set(resumen["codigo"]):
    st.warning("No hay datos para ese codigo en la ventana analizada.")
    st.stop()

row = resumen[resumen["codigo"] == codigo].iloc[0]

st.subheader(f"{row['descrip']} ({codigo})")
st.caption(f"Ventana analizada: {ventana_inicio.date()} - {max_date.date()}")

csv = resumen.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Descargar resumen ROP (CSV)",
    data=csv,
    file_name="resumen_rop.csv",
    mime="text/csv",
)

cols_resumen = [
    "ROP_ajustado",
    "ROP_intermedio",
    "ROP_holgado",
    "AhorroCapitalAjustado",
    "ExcesoCapitalHolgado",
    "%_quiebre_ajustado",
    "%_quiebre_intermedio",
    "%_quiebre_holgado",
    "%_quiebre_real",
]

column_config = {c: st.column_config.NumberColumn(width="small") for c in cols_resumen}

st.dataframe(
    pd.DataFrame([row])[cols_resumen],
    height=110,
    use_container_width=True,
    hide_index=True,
    column_config=column_config,
)

st.markdown("### Conclusion")
st.write(texto_conclusion(row))

fig = plot_producto(df, resumen, ventas_1y_full, stock_real_diario, quantiles, codigo)
if fig is not None:
    st.pyplot(fig, use_container_width=True)
