from __future__ import annotations

from pathlib import Path

import base64
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


st.set_page_config(page_title="Tablero Farmacia", layout="wide")

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
def load_movimientos(data_source: Path | None, uploaded_file) -> tuple[pd.DataFrame, pd.DataFrame]:
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

    ventas = df[
        ((df["tipo"].str.contains("venta", case=False, na=False)) & (df["cantid"] < 0))
        | ((df["tipo"].str.contains("anulaci", case=False, na=False)) & (df["cantid"] > 0))
    ].copy()

    ventas = ventas[ventas["nut"].astype(str).str.strip() != "99999999"].copy()
    ventas = ventas[~ventas["tipo"].str.contains("inventario", case=False, na=False)]
    ventas["unidades"] = -ventas["cantid"]
    ventas = ventas[ventas["codigo"] != "-900006"].copy()

    ventas_diarias = (
        ventas.groupby(["codigo", "fecha"])["unidades"].sum().reset_index()
    )

    return df, ventas_diarias


@st.cache_data(show_spinner=False)
def load_rop(data_source: Path | None, uploaded_file) -> dict:
    rop_maps = {"ajustado": {}, "intermedio": {}, "holgado": {}}

    if uploaded_file is not None:
        rop_csv = pd.read_csv(uploaded_file)
    else:
        if data_source is None or not data_source.exists():
            return rop_maps
        rop_csv = pd.read_csv(data_source)

    rop_csv["codigo"] = rop_csv["codigo"].astype(str).str.strip()

    if "ROP_ajustado" in rop_csv.columns:
        rop_maps["ajustado"] = rop_csv.set_index("codigo")["ROP_ajustado"].to_dict()
    if "ROP_intermedio" in rop_csv.columns:
        rop_maps["intermedio"] = rop_csv.set_index("codigo")["ROP_intermedio"].to_dict()
    if "ROP_holgado" in rop_csv.columns:
        rop_maps["holgado"] = rop_csv.set_index("codigo")["ROP_holgado"].to_dict()

    return rop_maps


def plot_producto(df: pd.DataFrame, ventas_diarias: pd.DataFrame, codigo: str):
    d = df[df["codigo"] == codigo].copy()
    if d.empty:
        return None, None

    descrip = d["descrip"].iloc[0]
    v = ventas_diarias[ventas_diarias["codigo"] == codigo].set_index("fecha")[
        "unidades"
    ]

    all_days = pd.date_range(d["fecha"].min(), d["fecha"].max(), freq="D")
    v_full = v.reindex(all_days, fill_value=0)

    stock_diario = (
        d.sort_values(["fecha", "hora"])
        .groupby("fecha")["despues"]
        .agg(Closing_Stock="last", Min_Stock="min")
    )
    stock_diario = stock_diario.reindex(all_days).ffill()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(v_full.index, v_full.values, color="tab:blue", label="Ventas diarias")
    ax.plot(
        stock_diario.index,
        stock_diario["Closing_Stock"].values,
        color="tab:orange",
        label="Stock fin de dia",
    )

    stock_cero = stock_diario[stock_diario["Closing_Stock"] <= 0]
    ax.scatter(
        stock_cero.index,
        stock_cero["Closing_Stock"].values,
        color="red",
        s=12,
        label="Stock <= 0",
    )

    ax.set_ylabel("Unidades")
    ax.legend()
    ax.set_title(f"{descrip} ({codigo})")
    plt.tight_layout()

    max_val = int(v_full.max()) if v_full.max() > 0 else 1
    bins = np.arange(0, max_val + 1, 1)

    fig_hist, ax_hist = plt.subplots(figsize=(8, 5))
    sns.histplot(v_full.values, bins=bins, kde=False, ax=ax_hist)
    ax_hist.set_xticks(range(0, max_val + 1))
    ax_hist.set_title(f"Distribucion de ventas: {descrip} ({codigo})")
    ax_hist.set_xlabel("Unidades vendidas por dia")
    plt.tight_layout()

    return fig, fig_hist


def plot_stockout_times(df: pd.DataFrame, codigo: str):
    d = df[df["codigo"] == codigo].copy()
    if d.empty:
        return None

    descrip = d["descrip"].iloc[0]

    daily_min = d.groupby("fecha")["despues"].min()
    stockout_days = daily_min[daily_min <= 0].index

    total_days = (daily_min.index.max() - daily_min.index.min()).days + 1
    n_stockouts = len(stockout_days)
    ratio = n_stockouts / total_days if total_days else 0

    fig, ax = plt.subplots(figsize=(10, 2))
    ax.scatter(stockout_days, [1] * len(stockout_days), color="red", s=6)
    ax.set_yticks([])
    ax.set_title(f"Dias con stock <= 0 - {descrip} ({codigo})")
    ax.set_xlabel("Fecha")

    fig.text(
        0.01,
        -0.05,
        f"Stockouts: {n_stockouts} | Dias analizados: {total_days} | Ratio: {ratio:.2%}",
        ha="left",
    )
    plt.tight_layout()
    return fig


def plot_ventas_por_anio_misma_escala(
    df: pd.DataFrame, ventas_diarias: pd.DataFrame, codigo: str
):
    v = ventas_diarias[ventas_diarias["codigo"] == codigo].copy()
    if v.empty:
        return None

    descrip = df.loc[df["codigo"] == codigo, "descrip"].iloc[0]

    all_days = pd.date_range(v["fecha"].min(), v["fecha"].max(), freq="D")
    v = (
        v.set_index("fecha")
        .reindex(all_days, fill_value=0)
        .rename_axis("fecha")
        .reset_index()
    )
    v["codigo"] = codigo

    v["anio"] = v["fecha"].dt.year
    v["dia_anio"] = v["fecha"].dt.dayofyear

    max_y = max(v["unidades"].max(), 1)

    ref = pd.date_range("2024-01-01", "2024-12-31", freq="MS")
    month_ticks = ref.dayofyear.tolist()
    month_labels = ref.strftime("%b").tolist()

    fig, ax = plt.subplots(figsize=(10, 4))
    for anio in sorted(v["anio"].unique()):
        sub = v[v["anio"] == anio]
        ax.plot(sub["dia_anio"], sub["unidades"], label=str(anio))

    ax.set_xlim(1, 366)
    ax.set_ylim(0, max_y)
    ax.set_xticks(month_ticks)
    ax.set_xticklabels(month_labels)
    ax.set_title(f"Ventas diarias por ano - {descrip} ({codigo})")
    ax.set_xlabel("Mes")
    ax.set_ylabel("Unidades")
    ax.legend()
    plt.tight_layout()
    return fig


def daily_metrics_for_product(
    df: pd.DataFrame, ventas_diarias: pd.DataFrame, codigo: str
):
    v = ventas_diarias[ventas_diarias["codigo"] == codigo].copy()
    if v.empty:
        return None, None

    descrip = df.loc[df["codigo"] == codigo, "descrip"].iloc[0]

    all_days = pd.date_range(v["fecha"].min(), v["fecha"].max(), freq="D")
    v_full = (
        v.set_index("fecha")
        .reindex(all_days, fill_value=0)
        .rename_axis("FECHA")
        .reset_index()
    )
    v_full = v_full.rename(columns={"unidades": "Quantity_Sold"})

    d = df[df["codigo"] == codigo].copy()
    stock_diario = (
        d.sort_values(["fecha", "hora"])
        .groupby("fecha")["despues"]
        .agg(Closing_Stock="last", Min_Stock="min")
    )
    stock_diario = (
        stock_diario.reindex(all_days)
        .ffill()
        .reset_index()
        .rename(columns={"index": "FECHA"})
    )

    daily = v_full.merge(stock_diario, on="FECHA", how="left")

    return daily, descrip


def _filter_by_date(daily_data, start_date=None, end_date=None):
    data = daily_data
    if start_date is not None:
        data = data[data["FECHA"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        data = data[data["FECHA"] <= pd.Timestamp(end_date)]
    return data


def plot_sales(
    daily_data,
    name,
    start_date=None,
    end_date=None,
    rop_ajustado=None,
    rop_intermedio=None,
    rop_holgado=None,
):
    data = _filter_by_date(daily_data, start_date, end_date)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(data["FECHA"], data["Quantity_Sold"], linestyle="-", alpha=0.7, label="Ventas diarias")

    mask_venta = data["Quantity_Sold"] > 0
    ax.scatter(
        data.loc[mask_venta, "FECHA"],
        data.loc[mask_venta, "Quantity_Sold"],
        color="blue",
        s=6,
        label="Venta",
    )

    mask_venta_stock0 = (data["Quantity_Sold"] > 0) & (data["Min_Stock"] <= 0)
    ax.scatter(
        data.loc[mask_venta_stock0, "FECHA"],
        data.loc[mask_venta_stock0, "Quantity_Sold"],
        color="orange",
        s=12,
        label="Venta y stock<=0",
    )

    mask = (data["Quantity_Sold"] == 0) & (data["Min_Stock"] <= 0)
    ax.scatter(
        data.loc[mask, "FECHA"],
        data.loc[mask, "Quantity_Sold"],
        color="red",
        s=10,
        label="Venta=0 y stock<=0",
    )

    if rop_ajustado is not None:
        ax.axhline(rop_ajustado, color="red", linestyle="--", linewidth=1.2, alpha=0.8, label="ROP ajustado")
    if rop_intermedio is not None:
        ax.axhline(
            rop_intermedio,
            color="goldenrod",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="ROP intermedio",
        )
    if rop_holgado is not None:
        ax.axhline(rop_holgado, color="green", linestyle="--", linewidth=1.2, alpha=0.8, label="ROP holgado")

    ax.set_title(f"Tendencia de Ventas Diarias: {name}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Unidades Vendidas")
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    ax.legend()
    plt.tight_layout()
    return fig


def plot_stock_variation(
    daily_data,
    name,
    start_date=None,
    end_date=None,
    rop_sifaco=None,
    rop_ajustado=None,
    rop_intermedio=None,
    rop_holgado=None,
):
    data = _filter_by_date(daily_data, start_date, end_date)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(
        data["FECHA"],
        data["Closing_Stock"],
        label="Stock al Cierre",
        color="green",
        alpha=0.8,
    )
    stockouts = data[data["Min_Stock"] <= 0]
    ax.scatter(
        stockouts["FECHA"],
        [0] * len(stockouts),
        color="red",
        label="Stock <= 0 en algun momento del dia",
        zorder=5,
    )

    if rop_sifaco is not None:
        ax.axhline(
            rop_sifaco,
            color="gray",
            linestyle="--",
            linewidth=1.2,
            alpha=0.5,
            label="ROP SIFACO (stock)",
        )

    if rop_ajustado is not None:
        ax.axhline(
            rop_ajustado,
            color="orange",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="ROP ajustado",
        )
    if rop_intermedio is not None:
        ax.axhline(
            rop_intermedio,
            color="goldenrod",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="ROP intermedio",
        )
    if rop_holgado is not None:
        ax.axhline(
            rop_holgado,
            color="green",
            linestyle="--",
            linewidth=1.2,
            alpha=0.8,
            label="ROP holgado",
        )

    ax.set_title(f"Evolucion del Stock: {name}")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Nivel de Stock")
    ax.legend()
    ax.grid(True, which="both", linestyle="--", linewidth=0.5)
    plt.tight_layout()
    return fig


def plot_distribution(daily_data, name, start_date=None, end_date=None):
    data = _filter_by_date(daily_data, start_date, end_date)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(data["Quantity_Sold"], kde=True, bins=20, ax=ax)
    ax.set_title(f"Distribucion de Ventas: {name}")
    ax.set_xlabel("Cantidad de Ventas Diarias")
    ax.set_ylabel("Frecuencia")
    plt.tight_layout()
    return fig


def plot_boxplot(daily_data, name, start_date=None, end_date=None):
    data = _filter_by_date(daily_data, start_date, end_date)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.boxplot(y=data["Quantity_Sold"], ax=ax)
    ax.set_title(f"Boxplot de Ventas: {name}")
    ax.set_ylabel("Unidades Vendidas")
    ax.grid(True, linestyle="--", linewidth=0.5, axis="y")
    plt.tight_layout()
    return fig


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
    st.title("Tablero de Movimientos")

st.markdown('<div class="section-sep"></div>', unsafe_allow_html=True)
st.markdown("## Consulta")

st.sidebar.header("Datos")
movimientos_file = st.sidebar.file_uploader("movimientos.csv", type=["csv"])
rop_file = st.sidebar.file_uploader("rop_sugerido.csv (opcional)", type=["csv"])

try:
    df, ventas_diarias = load_movimientos(Path("data/movimientos.csv"), movimientos_file)
except FileNotFoundError:
    st.error("No se encontro data/movimientos.csv. Subi el archivo en la barra lateral.")
    st.stop()

rop_maps = load_rop(Path("data/rop_sugerido.csv"), rop_file)

def _use_selector():
    st.session_state["codigo_manual"] = ""


def _use_manual():
    st.session_state["codigo_selector"] = ""


selector_df = df[["codigo", "descrip"]].drop_duplicates().copy()
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

if codigo not in set(df["codigo"]):
    st.warning("No hay datos para ese codigo.")
    st.stop()

descrip = df.loc[df["codigo"] == codigo, "descrip"].iloc[0]
fecha_min = df.loc[df["codigo"] == codigo, "fecha"].min()
fecha_max = df.loc[df["codigo"] == codigo, "fecha"].max()

st.subheader(f"{descrip} ({codigo})")
st.caption(f"Rango de fechas: {fecha_min.date()} - {fecha_max.date()}")

rop_sifaco = (
    df.loc[df["codigo"] == codigo]
    .sort_values(["fecha", "hora"])["stock"]
    .iloc[-1]
)
rop_ajustado = rop_maps["ajustado"].get(codigo)
rop_intermedio = rop_maps["intermedio"].get(codigo)
rop_holgado = rop_maps["holgado"].get(codigo)

fig_main, fig_hist = plot_producto(df, ventas_diarias, codigo)
fig_stockout = plot_stockout_times(df, codigo)
fig_anio = plot_ventas_por_anio_misma_escala(df, ventas_diarias, codigo)

if fig_main is not None:
    st.pyplot(fig_main, use_container_width=True)
if fig_hist is not None:
    st.pyplot(fig_hist, use_container_width=True)
if fig_stockout is not None:
    st.pyplot(fig_stockout, use_container_width=True)
if fig_anio is not None:
    st.pyplot(fig_anio, use_container_width=True)

daily, _ = daily_metrics_for_product(df, ventas_diarias, codigo)
if daily is not None:
    fig_sales = plot_sales(
        daily,
        descrip,
        rop_ajustado=rop_ajustado,
        rop_intermedio=rop_intermedio,
        rop_holgado=rop_holgado,
    )
    fig_stock = plot_stock_variation(
        daily,
        descrip,
        rop_sifaco=rop_sifaco,
        rop_ajustado=rop_ajustado,
        rop_intermedio=rop_intermedio,
        rop_holgado=rop_holgado,
    )
    fig_dist = plot_distribution(daily, descrip)
    fig_box = plot_boxplot(daily, descrip)

    st.pyplot(fig_sales, use_container_width=True)
    st.pyplot(fig_stock, use_container_width=True)
    st.pyplot(fig_dist, use_container_width=True)
    st.pyplot(fig_box, use_container_width=True)
