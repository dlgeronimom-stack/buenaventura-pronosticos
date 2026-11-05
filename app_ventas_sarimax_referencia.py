from __future__ import annotations

from io import BytesIO
import base64
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urljoin
import re
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
import warnings
from statsmodels.tsa.statespace.sarimax import SARIMAX
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None

# =====================================================================
# BUENAVENTURA · INTELIGENCIA DE VENTAS | Versión 2.1
# Lee la base histórica en modo de solo lectura. No escribe en data/ ni outputs/.
# =====================================================================
st.set_page_config(
    page_title="Buenaventura | Pronóstico de ingresos",
    page_icon="⛏️",
    layout="wide",
    initial_sidebar_state="expanded",
)

APP_DIR = Path(__file__).resolve().parent
SALES_FILENAME = "base_modelo_referencia_sarimax_3_regresores_2019Q1_2026Q3.csv"
METAL_BASE_FILENAME = "base_comparable_ventas_metal_precio_realizado_perimetro_2019Q1_2026Q3.csv"
METAL_PROXY_FILENAME = "proxy_comparable_ventas_metal_precio_realizado_2019Q1_2026Q3.csv"
METAL_PROXY_FEATURE = "valor_bruto_metales_usd_millones"
WORLD_BANK_PAGE = "https://www.worldbank.org/en/research/commodity-markets"
WORLD_BANK_MONTHLY_XLSX = (
    "https://thedocs.worldbank.org/en/doc/74e8be41ceb20fa0da750cda2f6b9e4e-0050012026/related/CMO-Historical-Data-Monthly.xlsx"
)
OFFICIAL_LOGO_URL = "https://buenaventura.com/wp-content/uploads/2024/06/Logo-1.png"
OFFICIAL_INVESTOR_URL = "https://buenaventura.com/inversionista/"
FY2025_RESULTS_URL = "https://buenaventura.com/wp-content/uploads/2025/02/4Q25-BVN-Earnings-Press-Release-Vf.pdf"
Q1_2026_RESULTS_URL = "https://buenaventura.com/wp-content/uploads/2026/04/1Q26-BVN-Earnings-Press-Release.pdf"
Q2_2026_RESULTS_URL = "https://www.sec.gov/Archives/edgar/data/1013131/000110465926088868/tm2621768d1_6k.htm"
Q2_2026_FINANCIALS_URL = "https://buenaventura.com/wp-content/uploads/2026/04/2Q-Buenaventura-Consolidated-2026.pdf"
Q3_2026_RESULTS_URL = "https://www.sec.gov/Archives/edgar/data/1013131/000110465926120330/tm2628586d1_6k.htm"
BCRP_FX_SERIES = "PN01246PM"
BCRP_FX_API = "https://estadisticas.bcrp.gob.pe/estadisticas/series/api"
BCRP_FX_PAGE = "https://estadisticas.bcrp.gob.pe/estadisticas/series/mensuales/resultados/PN01246PM/html"
WORLD_BANK_MONTHLY_CSV = "precios_mensuales_metales_banco_mundial.csv"
BCRP_FX_MONTHLY_CSV = "tipo_cambio_soles_por_dolar_bcrp_mensual.csv"
EXTERNAL_METADATA_CSV = "metadatos_series_externas.csv"

TARGET = "total_revenues_usd_millones"
# Variables explicativas del modelo principal SARIMAX
EXOG_COLS = ["precio_cobre_usd_t", "precio_plata_usd_oz", "tipo_cambio_pen_usd"]
PRICE_COLS = ["gold", "silver", "copper", "lead", "zinc"]
MODEL_SARIMAX_EXOG = "SARIMAX referencia + cobre + plata + USD/PEN"
MODEL_SARIMA_BASE = "SARIMA estacional sin variables externas"
MODEL_XGB = "XGBoost autorregresivo (benchmark)"
MODEL_NAIVE = "Referencia: último trimestre"
MODEL_SEASONAL_NAIVE = "Referencia estacional (mismo trimestre del año previo)"
MIN_TRAIN_OBS = 20
BACKTEST_HORIZON = 4
BACKTEST_ORIGINS = 8

# =====================================================================
# INTERFAZ · distribución inspirada en la propuesta visual aprobada
# =====================================================================
def _asset_uri(filename: str) -> str:
    """Convierte ilustraciones incluidas en el paquete en URI embebidas para el navegador."""
    path = APP_DIR / "assets" / filename
    try:
        raw = path.read_bytes()
        return "data:image/jpeg;base64," + base64.b64encode(raw).decode("ascii")
    except OSError:
        return ""


HERO_IMAGE_URI = _asset_uri("mining_header.jpg")
SIDEBAR_IMAGE_URI = _asset_uri("mining_sidebar.jpg")

st.markdown(
f"""
<style>
:root {{ --bvn-green:#174b37; --bvn-green-2:#276b4b; --bvn-ink:#20352b; --bvn-muted:#5b6b61; --bvn-bg:#f5f7f5; --bvn-line:#dce5de; }}
.stApp {{ background:var(--bvn-bg); color:var(--bvn-ink); }}
[data-testid="stHeader"] {{ display:none !important; height:0 !important; }}
[data-testid="stMainBlockContainer"] {{ max-width:1220px; padding-top:.7rem; padding-bottom:1.5rem; }}
[data-testid="stMainBlockContainer"]:has(.bvn-scenario-panel) {{ padding-top:.45rem; padding-bottom:1rem; }}
[data-testid="stMainBlockContainer"]:has(.bvn-scenario-panel) > div[data-testid="stVerticalBlock"] {{ gap:.72rem; }}
[data-testid="stSidebar"] {{ background:#123b2c; }}
[data-testid="stSidebar"] * {{ color:#f6faf7; }}
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{ color:#d4e4d9 !important; }}
.stRadio [role="radiogroup"] {{ gap:.45rem; flex-wrap:wrap; }}
.stRadio [role="radio"] {{ border:1px solid var(--bvn-line); border-radius:999px; padding:.42rem .9rem; background:#fff; }}
.stRadio [role="radio"]:has(input:checked) {{ background:#e5f0e8; border-color:#78a88a; }}
[data-testid="stMetric"] {{ background:#fff; border:1px solid var(--bvn-line); border-radius:14px; padding:1rem; min-height:112px; }}
[data-testid="stMetricLabel"] {{ color:var(--bvn-muted) !important; font-size:.88rem !important; white-space:normal !important; }}
[data-testid="stMetricValue"] {{ color:var(--bvn-green) !important; font-size:clamp(1.35rem,2vw,2rem) !important; overflow-wrap:anywhere; }}
[data-testid="stAlert"] {{ border-radius:12px; }}
[data-testid="stExpander"] {{ border:1px solid var(--bvn-line); border-radius:12px; background:#fff; }}
div.stButton > button, div.stFormSubmitButton > button, div.stDownloadButton > button {{ min-height:2.75rem; border-radius:10px; font-weight:700; }}
div.stButton > button[kind="primary"], div.stFormSubmitButton > button[kind="primary"] {{ background:var(--bvn-green-2); border-color:var(--bvn-green-2); }}
[data-testid="stDataFrame"] {{ border:1px solid var(--bvn-line); border-radius:10px; overflow:hidden; }}
.bvn-hero {{ padding:.55rem 1.2rem; border-radius:14px; border:1px solid #d9e3dc; background-color:#163d2d; background-image:linear-gradient(90deg,rgba(249,251,249,.98) 0%,rgba(249,251,249,.94) 54%,rgba(249,251,249,.62) 76%,rgba(12,48,34,.18) 100%),url('{HERO_IMAGE_URI}'); background-position:center,right center; background-size:cover; margin:.1rem 0 .65rem 0; }}
.bvn-hero-kicker {{ color:#34734e; font-weight:800; letter-spacing:.1em; text-transform:uppercase; font-size:.63rem; }}
.bvn-hero h1 {{ color:#163b2c; font-size:clamp(1.35rem,2.2vw,1.8rem); line-height:1.08; margin:.12rem 0; }}
.bvn-hero p {{ color:#405449; max-width:760px; line-height:1.3; font-size:.9rem; margin:0; }}
.bvn-card {{ border:1px solid var(--bvn-line); border-radius:14px; background:#fff; padding:1rem 1.1rem; height:100%; }}
.bvn-card h3 {{ margin:.1rem 0 .5rem 0; color:var(--bvn-green); }}
.bvn-muted {{ color:var(--bvn-muted); font-size:.9rem; line-height:1.55; }}
@media(max-width:800px) {{ [data-testid="stMainBlockContainer"] {{ padding-left:.8rem; padding-right:.8rem; padding-top:1.1rem; }} .bvn-hero {{ padding:.8rem .85rem; background-image:linear-gradient(90deg,rgba(249,251,249,.98),rgba(249,251,249,.9)),url('{HERO_IMAGE_URI}'); }} }}

/* Ajustes de contraste y lectura solicitados */
[data-testid="stMetric"] {{ min-width:0; overflow:visible; }}
[data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{
  color:#174b37 !important; font-size:clamp(1.25rem,1.65vw,1.8rem) !important;
  line-height:1.2 !important; white-space:normal !important; overflow:visible !important;
  overflow-wrap:anywhere !important; text-overflow:clip !important;
}}
[data-testid="stMetricLabel"], [data-testid="stMetricDelta"] {{
  white-space:normal !important; overflow:visible !important; text-overflow:clip !important;
}}
[data-testid="stAlert"] {{ color:#20352b !important; background:#edf4ef !important; border:1px solid #d5e4da !important; }}
[data-testid="stAlert"] p, [data-testid="stAlert"] span {{ color:#20352b !important; }}
[data-testid="stFormSubmitButton"] button, [data-testid="stButton"] button[kind="primary"],
[data-testid="stDownloadButton"] button[kind="primary"] {{
  background:#276b4b !important; color:#fff !important; border-color:#276b4b !important;
}}
[data-testid="stFormSubmitButton"] button:hover, [data-testid="stButton"] button[kind="primary"]:hover {{
  background:#174b37 !important; border-color:#174b37 !important;
}}
.bvn-scenario-panel {{ border:1px solid #d4e2d8; border-radius:16px; background:#fff; padding:.5rem 1rem .35rem; margin:0 0 .45rem; }}
.bvn-scenario-panel h3 {{ color:#174b37; margin:0 0 .12rem !important; line-height:1.12; }}
.bvn-scenario-note {{ color:#52665a; font-size:.92rem; line-height:1.35; margin:0; }}
[data-testid="stAlert"] {{ margin-bottom:0 !important; padding:.55rem .8rem !important; }}
[data-testid="stSidebar"] [data-testid="stImage"] {{ background:#fff; padding:.45rem; border-radius:10px; }}
[data-testid="stSidebar"] [data-testid="stImage"] img {{ border-radius:7px; }}
.bvn-hero h1 {{ overflow-wrap:anywhere; }}
@media(max-width:800px) {{
  [data-testid="stMetricValue"], [data-testid="stMetricValue"] div {{ font-size:1.25rem !important; }}
  .bvn-scenario-panel {{ padding:.8rem; }}
}}


/* Tipografía y controles: contraste AA aproximado sobre fondos claros */
.stApp, .stApp button, .stApp input, .stApp label, .stApp textarea, .stApp select {{
  font-family: "Aptos", "Segoe UI", Arial, sans-serif !important;
}}
[data-testid="stMain"] [data-testid="stWidgetLabel"],
[data-testid="stMain"] [data-testid="stWidgetLabel"] *,
[data-testid="stMain"] [data-testid="stCaptionContainer"],
[data-testid="stMain"] [data-testid="stCaptionContainer"] *,
[data-testid="stMain"] label,
[data-testid="stMain"] label * {{
  color:#243a2f !important;
}}
[data-testid="stSlider"] [data-testid="stMarkdownContainer"] *,
[data-testid="stSelectbox"] [data-testid="stWidgetLabel"] * {{
  color:#243a2f !important;
}}
[data-testid="stButton"] button:not([kind="primary"]) {{
  color:#174b37 !important; background:#ffffff !important; border:1px solid #a9c2b1 !important;
}}
[data-testid="stFormSubmitButton"] button[kind="primary"] {{
  color:#ffffff !important; background:#276b4b !important; border-color:#276b4b !important;
}}
.bvn-interpretation {{
  border-left:5px solid #70a83b; border-radius:8px; padding:.9rem 1rem;
  background:#edf5ee; color:#20352b; line-height:1.55; margin:.75rem 0 1rem;
}}


.bvn-price-preview {{ display:grid; grid-template-columns:1fr auto; gap:.12rem .4rem; padding:.55rem .65rem; margin:.35rem 0 .65rem; border:1px solid #d4e2d8; border-radius:9px; background:#f6faf7; color:#20352b; font-size:.82rem; line-height:1.35; }}
.bvn-price-preview span {{ color:#52665a; }}
.bvn-price-preview b {{ color:#174b37; text-align:right; font-variant-numeric:tabular-nums; }}
@media(max-width:800px) {{ .bvn-price-preview {{ font-size:.76rem; padding:.45rem; }} }}

.bvn-metric-card {{ background:#fff; border:1px solid #dce5de; border-radius:14px; padding:1rem; min-height:112px; box-sizing:border-box; }}
.bvn-metric-label {{ color:#5b6b61; font-size:.88rem; line-height:1.35; margin-bottom:.35rem; }}
.bvn-metric-value {{ color:#174b37; font-size:clamp(1.25rem,1.65vw,1.8rem); line-height:1.2; white-space:nowrap; }}
.bvn-metric-note {{ color:#34734e; font-size:.78rem; margin-top:.45rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }}
@media(max-width:800px) {{ .bvn-metric-card {{ padding:.8rem; }} .bvn-metric-value {{ font-size:1.25rem; }} }}
</style>
""", unsafe_allow_html=True)

# =====================================================================
# DATOS Y FECHAS
# =====================================================================
def quarter_key(label: str) -> int:
    m = re.fullmatch(r"(\d{4})Q([1-4])", str(label).strip().upper())
    if not m:
        raise ValueError(f"Trimestre no válido: {label}")
    return int(m.group(1)) * 4 + int(m.group(2))


def key_to_quarter(key: int) -> str:
    year = (int(key) - 1) // 4
    q = (int(key) - 1) % 4 + 1
    return f"{year}Q{q}"


def next_quarter(label: str) -> str:
    return key_to_quarter(quarter_key(label) + 1)


def prior_quarter(label: str) -> str:
    return key_to_quarter(quarter_key(label) - 1)


def find_sales_file() -> Optional[Path]:
    """Busca la base ya existente; no la crea, copia, modifica ni sobrescribe."""
    home = Path.home()
    project = home / "Downloads" / "proyecto_xgboost_buenaventura"
    candidates = [
        APP_DIR / "data" / SALES_FILENAME,
        APP_DIR / SALES_FILENAME,
        APP_DIR.parent / SALES_FILENAME,
        APP_DIR.parent / "data" / SALES_FILENAME,
        project / "data" / SALES_FILENAME,
        project / SALES_FILENAME,
        home / "Downloads" / SALES_FILENAME,
    ]
    for path in candidates:
        if path.exists() and path.is_file():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_sales(path_text: str) -> pd.DataFrame:
    """Carga y valida la candidata auditada sin ordenar ni corregir silenciosamente la fuente."""
    df = pd.read_csv(path_text, encoding="utf-8-sig")
    required = {"trimestre", TARGET, "fecha_publicacion_ingresos", "fuente_ingresos", *EXOG_COLS}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"La base candidata no contiene columnas requeridas: {sorted(missing)}")
    df = df.copy()
    df["trimestre"] = df["trimestre"].astype(str).str.strip().str.upper()
    df["_key"] = df["trimestre"].map(quarter_key)
    if df["_key"].isna().any() or df["trimestre"].duplicated().any():
        raise ValueError("La candidata contiene etiquetas inválidas o trimestres duplicados.")
    if df["_key"].tolist() != sorted(df["_key"].tolist()):
        raise ValueError("La candidata no está ordenada cronológicamente; no se reordena en silencio.")
    for col in [TARGET, *EXOG_COLS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    if df[[TARGET, *EXOG_COLS]].isna().any().any() or not np.isfinite(df[[TARGET, *EXOG_COLS]].to_numpy(float)).all():
        raise ValueError("La candidata tiene datos faltantes o no finitos en objetivo/regresores.")
    if (df[[TARGET, *EXOG_COLS]] <= 0).any().any():
        raise ValueError("Ingresos y regresores deben ser positivos.")
    expected = list(range(int(df["_key"].min()), int(df["_key"].max()) + 1))
    if df["_key"].tolist() != expected:
        raise ValueError("La serie de ingresos no es trimestral continua.")
    df["fecha_publicacion_ingresos"] = pd.to_datetime(df["fecha_publicacion_ingresos"], errors="coerce")
    if df["fecha_publicacion_ingresos"].isna().any():
        raise ValueError("Faltan fechas de publicación de ingresos.")
    for col in ["meses_cobre_disponibles", "meses_plata_disponibles", "meses_tipo_cambio_disponibles"]:
        if not df[col].eq(3).all():
            raise ValueError(f"La variable {col} no tiene 3 meses completos en cada trimestre.")
    if df["fuente_ingresos"].fillna("").astype(str).str.strip().eq("").any():
        raise ValueError("Hay ingresos sin fuente documental.")
    if len(df) < MIN_TRAIN_OBS + BACKTEST_HORIZON + 2:
        raise ValueError("La historia no alcanza para la evaluación temporal mínima.")
    if df.loc[df["trimestre"].eq("2019Q1"), TARGET].iloc[0] != 186.2:
        raise ValueError("El objetivo principal 2019Q1 debe ser 186.2; revisa la base candidata.")
    return df


def prepare_reference_data(sales: pd.DataFrame) -> pd.DataFrame:
    """La candidata ya trae ingresos y exógenas alineados; valida la unión 1:1."""
    if sales["trimestre"].duplicated().any() or sales["_key"].duplicated().any():
        raise ValueError("Hay trimestres duplicados en la base candidata.")
    model_data = sales[["trimestre", "_key", TARGET, *EXOG_COLS]].copy()
    if model_data[EXOG_COLS].isna().any().any() or (model_data[EXOG_COLS] <= 0).any().any():
        raise ValueError("Los regresores tienen datos faltantes o no positivos.")
    return model_data.reset_index(drop=True)


def metadata_from_candidate(sales: pd.DataFrame) -> pd.DataFrame:
    row = sales.iloc[-1]
    return pd.DataFrame([
        {"serie": "Cobre", "fuente": row["fuente_cobre_plata"], "unidad": "US$ nominal/tonelada métrica", "frecuencia": "Trimestral (promedio de 3 meses)", "fecha_extraccion": row["fecha_extraccion_banco_mundial"], "periodo_inicio": sales.iloc[0]["trimestre"], "periodo_fin": row["trimestre"], "url": row["fuente_cobre_plata"]},
        {"serie": "Plata", "fuente": row["fuente_cobre_plata"], "unidad": "US$ nominal/onza troy", "frecuencia": "Trimestral (promedio de 3 meses)", "fecha_extraccion": row["fecha_extraccion_banco_mundial"], "periodo_inicio": sales.iloc[0]["trimestre"], "periodo_fin": row["trimestre"], "url": row["fuente_cobre_plata"]},
        {"serie": "USD/PEN", "fuente": row["fuente_usd_pen"], "unidad": "S/ por US$", "frecuencia": "Trimestral (promedio de 3 meses)", "fecha_extraccion": row["fecha_extraccion_bcrp"], "periodo_inicio": sales.iloc[0]["trimestre"], "periodo_fin": row["trimestre"], "url": row["fuente_usd_pen"]},
    ])


def parse_month(value) -> Optional[pd.Period]:
    if pd.isna(value):
        return None
    if isinstance(value, (pd.Timestamp, np.datetime64)):
        return pd.Timestamp(value).to_period("M")
    if hasattr(value, "year") and hasattr(value, "month") and not isinstance(value, str):
        try:
            return pd.Period(year=int(value.year), month=int(value.month), freq="M")
        except Exception:
            pass
    text = str(value).strip()
    if re.fullmatch(r"\d{4}(?:\.0)?", text):
        return None
    match = re.fullmatch(r"(\d{4})\s*M\s*(\d{1,2})", text, re.I)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return pd.Period(year=year, month=month, freq="M")
    match = re.fullmatch(r"(\d{4})[-/](\d{1,2})(?:[-/]\d{1,2})?", text)
    if match:
        year, month = int(match.group(1)), int(match.group(2))
        if 1 <= month <= 12:
            return pd.Period(year=year, month=month, freq="M")
    parsed = pd.to_datetime(text, errors="coerce")
    if not pd.isna(parsed):
        return pd.Timestamp(parsed).to_period("M")
    return None


def _parse_year(value) -> Optional[int]:
    try:
        number = int(float(str(value).strip()))
        return number if 1900 <= number <= 2100 else None
    except Exception:
        return None


def _parse_month_number(value) -> Optional[int]:
    if pd.isna(value):
        return None
    text = str(value).strip().lower()
    try:
        number = int(float(text))
        return number if 1 <= number <= 12 else None
    except Exception:
        pass
    names = {
        "jan": 1, "january": 1, "ene": 1, "enero": 1,
        "feb": 2, "february": 2, "febrero": 2,
        "mar": 3, "march": 3, "marzo": 3,
        "apr": 4, "april": 4, "abr": 4, "abril": 4,
        "may": 5, "mayo": 5, "jun": 6, "june": 6, "junio": 6,
        "jul": 7, "july": 7, "julio": 7, "aug": 8, "august": 8, "ago": 8, "agosto": 8,
        "sep": 9, "sept": 9, "september": 9, "set": 9, "septiembre": 9,
        "oct": 10, "october": 10, "octubre": 10,
        "nov": 11, "november": 11, "noviembre": 11,
        "dec": 12, "december": 12, "dic": 12, "diciembre": 12,
    }
    return next((number for name, number in names.items() if text == name or text.startswith(name + " ")), None)


def identify_period_and_metals(raw: pd.DataFrame) -> pd.DataFrame:
    """Detecta encabezados, periodo y series mensuales en el Excel mensual del Banco Mundial."""
    metals = tuple(PRICE_COLS)
    best_header, best_cols = None, {}
    for row_idx in range(min(35, len(raw))):
        found = {}
        for col_idx in range(raw.shape[1]):
            value = str(raw.iat[row_idx, col_idx]).strip().lower() if not pd.isna(raw.iat[row_idx, col_idx]) else ""
            for metal in metals:
                if value == metal or value.startswith((metal + ",", metal + " ", metal + "(")):
                    found[metal] = col_idx
        if len(found) > len(best_cols):
            best_header, best_cols = row_idx, found
    if best_header is None or len(best_cols) < len(metals):
        raise ValueError("No se identificaron los cinco encabezados de precios de metales en 'Monthly Prices'.")

    data_start = best_header + 1
    period_options = []
    for col_idx in range(min(5, raw.shape[1])):
        count = sum(parse_month(raw.iat[row_idx, col_idx]) is not None for row_idx in range(data_start, len(raw)))
        period_options.append((count, col_idx))
    period_count, period_col = max(period_options, default=(0, 0))
    rows = []

    if period_count >= 24:
        for row_idx in range(data_start, len(raw)):
            period = parse_month(raw.iat[row_idx, period_col])
            if period is None:
                continue
            record = {"mes": period}
            for metal, col_idx in best_cols.items():
                value = pd.to_numeric(raw.iat[row_idx, col_idx], errors="coerce")
                record[metal] = float(value) if pd.notna(value) and np.isfinite(float(value)) and float(value) > 0 else np.nan
            if sum(pd.notna(record.get(metal)) for metal in metals) >= 3:
                rows.append(record)
    else:
        for year_col in range(min(5, raw.shape[1])):
            for month_col in range(min(5, raw.shape[1])):
                if year_col == month_col:
                    continue
                built = []
                for row_idx in range(data_start, len(raw)):
                    year = _parse_year(raw.iat[row_idx, year_col])
                    month = _parse_month_number(raw.iat[row_idx, month_col])
                    if year is None or month is None:
                        continue
                    record = {"mes": pd.Period(year=year, month=month, freq="M")}
                    for metal, col_idx in best_cols.items():
                        value = pd.to_numeric(raw.iat[row_idx, col_idx], errors="coerce")
                        record[metal] = float(value) if pd.notna(value) and np.isfinite(float(value)) and float(value) > 0 else np.nan
                    built.append(record)
                if len(built) > len(rows):
                    rows = built

    if len(rows) < 24:
        raise ValueError(f"Solo se identificaron {len(rows)} meses válidos en la hoja mensual del Banco Mundial.")
    monthly = pd.DataFrame(rows).drop_duplicates("mes").sort_values("mes").reset_index(drop=True)
    for metal in metals:
        if metal not in monthly.columns or monthly[metal].notna().sum() < 24:
            raise ValueError(f"La serie mensual de {metal} tiene menos de 24 observaciones válidas.")
    return monthly


def current_world_bank_monthly_url() -> str:
    """Busca el archivo XLSX vigente en la página oficial, usando una URL de respaldo conocida."""
    class WorkbookLinkParser(HTMLParser):
        def __init__(self):
            super().__init__()
            self.links = []
        def handle_starttag(self, tag, attrs):
            if tag.lower() != "a":
                return
            attrs_map = dict(attrs)
            href = attrs_map.get("href", "")
            title = attrs_map.get("title", "")
            link_text = f"{href} {title}".lower()
            if "cmo-historical-data-monthly.xlsx" in link_text:
                self.links.append(urljoin(WORLD_BANK_PAGE, href))

    try:
        response = requests.get(WORLD_BANK_PAGE, timeout=12, headers={"User-Agent": "Mozilla/5.0"})
        response.raise_for_status()
        parser = WorkbookLinkParser()
        parser.feed(response.text)
        if parser.links:
            return parser.links[0]
    except Exception:
        pass
    return WORLD_BANK_MONTHLY_XLSX


def parse_bcrp_period(value: str) -> Optional[pd.Period]:
    """Convierte etiquetas mensuales de BCRPData (Ene.2019, etc.) a Period mensual."""
    text = str(value).strip().lower().replace("á", "a").replace("é", "e").replace("í", "i").replace("ó", "o").replace("ú", "u")
    direct = re.search(r"(\d{4})[-/](\d{1,2})", text)
    if direct:
        year, month = int(direct.group(1)), int(direct.group(2))
        if 1 <= month <= 12:
            return pd.Period(year=year, month=month, freq="M")
    month_names = {
        "ene": 1, "enero": 1, "feb": 2, "febrero": 2, "mar": 3, "marzo": 3,
        "abr": 4, "abril": 4, "may": 5, "mayo": 5, "jun": 6, "junio": 6,
        "jul": 7, "julio": 7, "ago": 8, "agosto": 8, "sep": 9, "set": 9,
        "septiembre": 9, "oct": 10, "octubre": 10, "nov": 11, "noviembre": 11,
        "dic": 12, "diciembre": 12,
    }
    match = re.search(r"([a-z]+)[.\s/-]*(\d{2}|\d{4})$", text)
    if match:
        month_text, year_text = match.group(1), match.group(2)
        month = month_names.get(month_text, month_names.get(month_text[:3]))
        if month:
            year_number = int(year_text)
            if len(year_text) == 2:
                year_number += 2000 if year_number <= 68 else 1900
            return pd.Period(year=year_number, month=month, freq="M")
    return None


def load_external_metadata() -> pd.DataFrame:
    """Lee el manifiesto de fuente, unidad, frecuencia y extracción de los insumos externos."""
    path = APP_DIR / "data" / EXTERNAL_METADATA_CSV
    if not path.is_file():
        raise FileNotFoundError(f"Falta el manifiesto de metadatos: {path}")
    metadata = pd.read_csv(path, encoding="utf-8-sig")
    required = {"serie", "fuente", "unidad", "frecuencia", "fecha_extraccion"}
    missing = required - set(metadata.columns)
    if missing:
        raise ValueError(f"El manifiesto de fuentes no contiene: {sorted(missing)}")
    if metadata["serie"].duplicated().any() or metadata[list(required)].isna().any().any():
        raise ValueError("El manifiesto de fuentes contiene series duplicadas o metadatos vacíos.")
    return metadata


def _load_monthly_snapshot(filename: str, value_columns: list[str]) -> pd.DataFrame:
    """Carga una serie mensual guardada y valida fechas únicas y valores positivos."""
    path = APP_DIR / "data" / filename
    if not path.is_file():
        raise FileNotFoundError(f"Falta la serie externa guardada: {path}")
    monthly = pd.read_csv(path, encoding="utf-8-sig")
    if not {"mes", *value_columns}.issubset(monthly.columns):
        raise ValueError(f"{filename} debe incluir mes y {value_columns}.")
    monthly = monthly[["mes", *value_columns]].copy()
    monthly["mes"] = monthly["mes"].map(parse_month)
    if monthly["mes"].isna().any():
        raise ValueError(f"{filename} contiene etiquetas mensuales inválidas.")
    if monthly["mes"].duplicated().any():
        raise ValueError(f"{filename} contiene meses duplicados.")
    for column in value_columns:
        monthly[column] = pd.to_numeric(monthly[column], errors="coerce")
        if monthly[column].isna().any() or (~np.isfinite(monthly[column])).any() or (monthly[column] <= 0).any():
            raise ValueError(f"{filename}: {column} contiene valores ausentes, no finitos o no positivos.")
    return monthly.sort_values("mes").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_drivers_quarterly(last_quarter: str) -> pd.DataFrame:
    """Agrega a trimestre snapshots mensuales fijados y documentados de Banco Mundial y BCRP."""
    last_key = quarter_key(last_quarter)
    last_year = int(last_quarter[:4])
    last_q = int(last_quarter[-1])
    last_month = pd.Period(year=last_year, month=last_q * 3, freq="M")
    start_month = pd.Period("2019-01", freq="M")

    metals = _load_monthly_snapshot(WORLD_BANK_MONTHLY_CSV, ["cobre_usd_t", "plata_usd_onza_troy"])
    fx = _load_monthly_snapshot(BCRP_FX_MONTHLY_CSV, ["tipo_cambio_soles_por_dolar"])
    expected_months = pd.period_range(start_month, last_month, freq="M")
    for label, frame in (("Banco Mundial", metals), ("BCRP", fx)):
        observed = set(frame["mes"])
        missing = [str(month) for month in expected_months if month not in observed]
        if missing:
            raise ValueError(f"{label}: faltan meses para el periodo objetivo: {missing}")
        if any(month < start_month or month > last_month for month in frame["mes"]):
            raise ValueError(f"{label}: hay meses fuera de {start_month}–{last_month}.")

    metals["trimestre"] = metals["mes"].map(lambda p: f"{p.year}Q{p.quarter}")
    fx["trimestre"] = fx["mes"].map(lambda p: f"{p.year}Q{p.quarter}")
    metals_q = metals.groupby("trimestre", as_index=False).agg(
        precio_cobre_usd_t=("cobre_usd_t", "mean"),
        precio_plata_usd_oz=("plata_usd_onza_troy", "mean"),
        meses_metales=("mes", "nunique"),
        meses_cobre=("cobre_usd_t", "count"),
        meses_plata=("plata_usd_onza_troy", "count"),
    )
    fx_q = fx.groupby("trimestre", as_index=False).agg(
        tipo_cambio_pen_usd=("tipo_cambio_soles_por_dolar", "mean"),
        meses_tc=("mes", "nunique"),
    )
    metals_q = metals_q[
        metals_q["meses_metales"].eq(3)
        & metals_q["meses_cobre"].eq(3)
        & metals_q["meses_plata"].eq(3)
    ].copy()
    fx_q = fx_q[fx_q["meses_tc"].eq(3)].copy()
    drivers = metals_q.merge(fx_q, on="trimestre", how="inner", validate="one_to_one")
    drivers = drivers[drivers["trimestre"].map(quarter_key) <= last_key].copy()
    drivers = drivers.sort_values("trimestre", key=lambda s: s.map(quarter_key)).reset_index(drop=True)
    required = ["trimestre"] + EXOG_COLS
    drivers = drivers[required].copy()
    expected_quarters = [key_to_quarter(k) for k in range(quarter_key("2019Q1"), last_key + 1)]
    if drivers["trimestre"].duplicated().any():
        raise ValueError("Las series externas generan trimestres duplicados.")
    if drivers["trimestre"].tolist() != expected_quarters:
        missing_q = sorted(set(expected_quarters) - set(drivers["trimestre"]))
        extra_q = sorted(set(drivers["trimestre"]) - set(expected_quarters))
        raise ValueError(f"Alineación trimestral externa inválida. Faltantes={missing_q}; extras={extra_q}.")
    return drivers


def prepare_model_data(sales: pd.DataFrame, drivers: pd.DataFrame) -> pd.DataFrame:
    """Une ingresos y regresores solo en trimestres completos, sin imputar valores faltantes."""
    if sales["trimestre"].duplicated().any() or sales["_key"].duplicated().any():
        raise ValueError("La base de ingresos contiene trimestres duplicados.")
    if sales["_key"].tolist() != sorted(sales["_key"].tolist()):
        raise ValueError("La base de ingresos no está ordenada cronológicamente.")
    if drivers["trimestre"].duplicated().any():
        raise ValueError("Las variables externas contienen trimestres duplicados.")
    if drivers["trimestre"].tolist() != sorted(drivers["trimestre"].tolist(), key=quarter_key):
        raise ValueError("Las variables externas no están ordenadas cronológicamente.")
    merged = sales[["trimestre", "_key", TARGET]].merge(
        drivers, on="trimestre", how="inner", validate="one_to_one"
    ).sort_values("_key").reset_index(drop=True)
    missing_sales = sorted(set(sales["trimestre"]) - set(merged["trimestre"]))
    if missing_sales:
        raise ValueError(
            "No hay datos completos de cobre, plata y USD/PEN para todos los trimestres de ingresos. "
            f"Faltan: {missing_sales}. No se rellenarán con estimaciones."
        )
    minimum_for_three_origins = MIN_TRAIN_OBS + BACKTEST_HORIZON + 2
    if len(merged) < minimum_for_three_origins:
        raise ValueError(f"Solo hay {len(merged)} trimestres completos; se requieren al menos {minimum_for_three_origins} para evaluar tres orígenes comunes.")
    if merged[EXOG_COLS].isna().any().any() or (merged[EXOG_COLS] <= 0).any().any():
        raise ValueError("Los regresores tienen valores ausentes o no positivos; revisa las fuentes oficiales.")
    return merged


def mae_rmse(actual, predicted) -> tuple[float, float]:
    actual_arr = np.asarray(actual, dtype=float)
    predicted_arr = np.asarray(predicted, dtype=float)
    errors = actual_arr - predicted_arr
    return float(np.mean(np.abs(errors))), float(np.sqrt(np.mean(errors ** 2)))


def fit_sarimax_exog(
    history_y: np.ndarray | list[float],
    history_x: pd.DataFrame,
    future_x: pd.DataFrame,
) -> np.ndarray:
    """SARIMAX sobre cambios logarítmicos: ingresos explicados por cambios en los tres drivers."""
    y = np.asarray(history_y, dtype=float)
    if len(y) < MIN_TRAIN_OBS or np.any(~np.isfinite(y)) or np.any(y <= 0):
        raise ValueError("SARIMAX requiere al menos 20 ingresos históricos positivos y finitos.")
    if list(history_x.columns) != EXOG_COLS or list(future_x.columns) != EXOG_COLS:
        raise ValueError("Las variables exógenas no coinciden con cobre, plata y USD/PEN.")
    if history_x.isna().any().any() or future_x.isna().any().any() or (history_x <= 0).any().any() or (future_x <= 0).any().any():
        raise ValueError("Los valores futuros o históricos de los regresores deben ser positivos y completos.")

    # Modelamos variaciones logarítmicas (aproximación a tasas porcentuales), no niveles.
    log_y = np.log(y)
    log_x_history = np.log(history_x.astype(float))
    log_x_future = np.log(future_x.astype(float))
    y_change = np.diff(log_y)
    x_change = log_x_history.diff().iloc[1:].reset_index(drop=True)
    x_future_path = pd.concat([log_x_history.iloc[[-1]], log_x_future], ignore_index=True)
    x_future_change = x_future_path.diff().iloc[1:].reset_index(drop=True)

    center = x_change.mean(axis=0)
    scale = x_change.std(axis=0, ddof=0).replace(0, 1.0)
    x_train = (x_change - center) / scale
    x_future = (x_future_change - center) / scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            y_change,
            exog=x_train,
            order=(1, 0, 0),
            seasonal_order=(0, 0, 1, 4),
            trend="c",
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit(disp=False, maxiter=400)
        if not fitted.mle_retvals.get("converged", False):
            raise RuntimeError("SARIMAX no convergió; se omitió esta ventana de validación.")
        predicted_changes = np.asarray(
            fitted.get_forecast(steps=len(future_x), exog=x_future).predicted_mean,
            dtype=float,
        )
    predicted_log = log_y[-1] + np.cumsum(predicted_changes)
    predicted = np.exp(predicted_log)
    if np.any(~np.isfinite(predicted)):
        raise ValueError("SARIMAX devolvió pronósticos no finitos.")
    return np.maximum(predicted, 0.0)


def fit_sarima_univariate(history_y: np.ndarray | list[float], horizon: int) -> np.ndarray:
    values = np.asarray(history_y, dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            np.log(values), order=(0, 1, 1), seasonal_order=(0, 1, 1, 4), trend="n",
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=400)
        if not fitted.mle_retvals.get("converged", False):
            fitted = SARIMAX(
                np.log(values), order=(0, 1, 1), seasonal_order=(0, 1, 1, 4), trend="n",
                enforce_stationarity=False, enforce_invertibility=False,
            ).fit(disp=False, maxiter=1000, method="powell")
        if not fitted.mle_retvals.get("converged", False):
            raise RuntimeError("SARIMA no convergió con los optimizadores disponibles.")
        return np.maximum(np.exp(np.asarray(fitted.get_forecast(steps=horizon).predicted_mean, dtype=float)), 0.0)


def fit_xgb_benchmark(history_y: np.ndarray | list[float], horizon: int) -> np.ndarray:
    if XGBRegressor is None:
        raise RuntimeError("XGBoost no está instalado; no se puede calcular el benchmark.")
    values = np.asarray(history_y, dtype=float)
    x_train, y_train = [], []
    for idx in range(4, len(values)):
        x_train.append([values[idx - 1], values[idx - 4]])
        y_train.append(values[idx] - values[idx - 1])
    if len(x_train) < 10:
        raise ValueError("No hay suficientes observaciones para el benchmark XGBoost.")
    model = XGBRegressor(
        objective="reg:squarederror", n_estimators=80, max_depth=2, learning_rate=0.04,
        min_child_weight=3, subsample=0.90, colsample_bytree=0.90, reg_lambda=10.0,
        random_state=42, n_jobs=1, tree_method="hist",
    )
    model.fit(np.asarray(x_train, dtype=float), np.asarray(y_train, dtype=float))
    recursive = list(values.astype(float))
    predictions = []
    for _ in range(horizon):
        x_next = np.asarray([[recursive[-1], recursive[-4]]], dtype=float)
        next_value = max(0.0, recursive[-1] + float(model.predict(x_next)[0]))
        predictions.append(next_value)
        recursive.append(next_value)
    return np.asarray(predictions, dtype=float)


def forecast_exog_from_history(history_x: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Pronostica cada regresor con caminata aleatoria logarítmica y deriva histórica hasta el origen."""
    forecasts = {}
    for col in EXOG_COLS:
        values = history_x[col].astype(float).to_numpy()
        drift = float(np.diff(np.log(values)).mean()) if len(values) > 1 else 0.0
        forecasts[col] = values[-1] * np.exp(drift * np.arange(1, horizon + 1))
    return pd.DataFrame(forecasts, columns=EXOG_COLS)


def walk_forward_backtest(model_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Backtest expansivo 4T; precios futuros desconocidos se mantienen en el último nivel observado."""
    ordered = model_data.sort_values("_key").reset_index(drop=True)
    y = ordered[TARGET].astype(float).to_numpy()
    x = ordered[EXOG_COLS].astype(float).reset_index(drop=True)
    labels = ordered["trimestre"].astype(str).tolist()
    last_origin = len(ordered) - BACKTEST_HORIZON
    first_origin = max(MIN_TRAIN_OBS, last_origin - BACKTEST_ORIGINS + 1)
    records = []
    required_models = [MODEL_SARIMAX_EXOG, MODEL_SARIMA_BASE, MODEL_NAIVE, MODEL_SEASONAL_NAIVE]
    if XGBRegressor is not None:
        required_models.append(MODEL_XGB)
    valid_origins = 0
    for origin_pos in range(first_origin, last_origin + 1):
        train_y = y[:origin_pos]
        train_x = x.iloc[:origin_pos].copy()
        future_x_flat = forecast_exog_from_history(train_x, BACKTEST_HORIZON)
        actual_values = y[origin_pos:origin_pos + BACKTEST_HORIZON]
        future_labels = labels[origin_pos:origin_pos + BACKTEST_HORIZON]
        candidates = {}
        try:
            candidates[MODEL_SARIMAX_EXOG] = fit_sarimax_exog(train_y, train_x, future_x_flat)
        except Exception:
            pass
        try:
            candidates[MODEL_SARIMA_BASE] = fit_sarima_univariate(train_y, BACKTEST_HORIZON)
        except Exception:
            pass
        if XGBRegressor is not None:
            try:
                candidates[MODEL_XGB] = fit_xgb_benchmark(train_y, BACKTEST_HORIZON)
            except Exception:
                pass
        candidates[MODEL_NAIVE] = np.repeat(float(train_y[-1]), BACKTEST_HORIZON)
        candidates[MODEL_SEASONAL_NAIVE] = np.asarray([train_y[-4 + (step % 4)] for step in range(BACKTEST_HORIZON)], dtype=float)
        # Conserva solo orígenes donde todos los métodos generaron los cuatro pasos.
        valid = set(candidates) == set(required_models)
        valid = valid and all(len(pred) == BACKTEST_HORIZON and np.all(np.isfinite(pred)) for pred in candidates.values())
        if not valid:
            continue
        valid_origins += 1
        for model_name, predictions in candidates.items():
            for step, (quarter, actual, predicted) in enumerate(zip(future_labels, actual_values, predictions), start=1):
                records.append({
                    "origen": labels[origin_pos - 1], "horizonte": step, "trimestre": quarter,
                    "ventas_reales": float(actual), "modelo": model_name,
                    "pronostico": float(predicted), "error_absoluto": float(abs(actual - predicted)),
                    "error_cuadratico": float((actual - predicted) ** 2),
                })
    backtest = pd.DataFrame(records)
    if backtest.empty or valid_origins < 3:
        raise ValueError("No se pudieron obtener al menos tres orígenes comunes completos para comparar todos los modelos.")
    if any(quarter_key(row.trimestre) <= quarter_key(row.origen) for row in backtest.itertuples()):
        raise ValueError("El backtest contiene una predicción en el origen o antes del origen; posible fuga temporal.")
    comparison_keys = [
        set(zip(group["origen"], group["horizonte"], group["trimestre"]))
        for _, group in backtest.groupby("modelo")
    ]
    if not comparison_keys or any(keys != comparison_keys[0] for keys in comparison_keys[1:]):
        raise ValueError("Los modelos no comparten exactamente los mismos orígenes y horizontes.")
    metric_records = []
    for model_name, group in backtest.groupby("modelo"):
        mae, rmse = mae_rmse(group["ventas_reales"], group["pronostico"])
        metric_records.append({
            "Modelo": model_name, "MAE (US$ M)": mae, "RMSE (US$ M)": rmse,
            "Pronósticos evaluados": len(group), "Orígenes": group["origen"].nunique(),
        })
    metrics = pd.DataFrame(metric_records).sort_values("MAE (US$ M)").reset_index(drop=True)
    if metrics.empty or MODEL_SARIMAX_EXOG not in metrics["Modelo"].tolist():
        raise ValueError("SARIMAX con cobre, plata y USD/PEN no pudo estimarse en las ventanas históricas comunes.")
    return metrics, backtest


@st.cache_data(show_spinner=False)
def cached_backtest(model_data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return walk_forward_backtest(model_data)


def load_comparable_metal_base() -> pd.DataFrame:
    """Lee la base larga nueva sin modificar los históricos originales."""
    path = APP_DIR / "data" / METAL_BASE_FILENAME
    if not path.is_file():
        raise FileNotFoundError(f"Falta la base comparable de metales: {path}")
    metal_data = pd.read_csv(path, encoding="utf-8-sig")
    required = {
        "trimestre", "metal", "perimetro", "volumen", "unidad_volumen",
        "definicion_volumen", "precio_realizado_usd", "unidad_precio",
        "fecha_disponible", "fuente_volumen", "fuente_precio",
        "estado_comparabilidad", "valor_bruto_metal_usd_millones",
    }
    missing = required - set(metal_data.columns)
    if missing:
        raise ValueError(f"La base de metales no contiene las columnas: {sorted(missing)}")
    if metal_data.duplicated(["trimestre", "metal", "perimetro"]).any():
        raise ValueError("La base de metales tiene duplicados por trimestre, metal y perímetro.")
    metal_data["_key"] = metal_data["trimestre"].map(quarter_key)
    if metal_data["_key"].isna().any():
        raise ValueError("La base de metales tiene etiquetas trimestrales inválidas.")
    metal_data["fecha_disponible"] = pd.to_datetime(metal_data["fecha_disponible"], errors="coerce")
    if metal_data["fecha_disponible"].isna().any():
        raise ValueError("Falta la fecha de disponibilidad de una observación de metales.")
    return metal_data.sort_values(["_key", "metal", "perimetro"]).reset_index(drop=True)


def prepare_comparable_metal_proxy(
    sales: pd.DataFrame, metal_data: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Selecciona un total reportado por metal, sin sumar asociadas ni Trading dos veces."""
    accepted_scopes = {
        "Operaciones directas reportadas",
        "Operaciones directas reportadas + Buenaventura Trading",
    }
    selected = metal_data.loc[
        metal_data["perimetro"].isin(accepted_scopes)
        & metal_data["metal"].isin(["oro", "plata", "cobre", "zinc", "plomo"])
    ].copy()
    for column in ["volumen", "precio_realizado_usd", "valor_bruto_metal_usd_millones"]:
        selected[column] = pd.to_numeric(selected[column], errors="coerce")
    if selected.duplicated(["trimestre", "metal"]).any():
        raise ValueError("Hay más de un total seleccionado para un mismo metal y trimestre.")
    if len(selected) != 31 * 5 or selected[[
        "volumen", "precio_realizado_usd", "valor_bruto_metal_usd_millones",
        "fecha_disponible", "fuente_volumen", "fuente_precio",
    ]].isna().any().any():
        raise ValueError("La base utilizable debe tener 155 totales trazables y completos; no se imputaron faltantes.")
    if (selected[["volumen", "precio_realizado_usd"]] <= 0).any().any():
        raise ValueError("Se encontraron volúmenes o precios realizados no positivos.")
    calculated = selected["volumen"] * selected["precio_realizado_usd"] / 1_000_000
    if not np.allclose(calculated, selected["valor_bruto_metal_usd_millones"], rtol=1e-9, atol=1e-8):
        raise ValueError("El valor bruto por metal no concilia con volumen × precio realizado.")

    publication = sales.set_index("trimestre")["fecha_publicacion_ingresos"]
    selected["fecha_publicacion_ingresos"] = pd.to_datetime(
        selected["trimestre"].map(publication), errors="coerce"
    )
    if selected["fecha_publicacion_ingresos"].isna().any():
        raise ValueError("Falta una fecha de publicación de ingresos para conciliar la disponibilidad.")
    if (selected["fecha_disponible"] > selected["fecha_publicacion_ingresos"]).any():
        raise ValueError("Hay un volumen o precio realizado publicado después del ingreso objetivo.")

    records = []
    for quarter, group in selected.groupby("trimestre", sort=True):
        records.append({
            "trimestre": quarter,
            METAL_PROXY_FEATURE: float(group["valor_bruto_metal_usd_millones"].sum()),
            "fecha_disponibilidad_proxy": group["fecha_disponible"].max(),
            "incluye_trading_en_cobre": bool(group.loc[group["metal"].eq("cobre"), "perimetro"].str.contains("Trading").any()),
            "incluye_trading_en_plata": bool(group.loc[group["metal"].eq("plata"), "perimetro"].str.contains("Trading").any()),
        })
    proxy = pd.DataFrame(records)
    if len(proxy) != len(sales) or proxy[METAL_PROXY_FEATURE].isna().any():
        raise ValueError("El agregado de valores brutos no coincide con la cobertura de ingresos.")
    proxy = proxy.merge(
        sales[["trimestre", "fecha_publicacion_ingresos"]],
        on="trimestre", how="left", validate="one_to_one",
    )
    proxy["fecha_disponibilidad_proxy"] = pd.to_datetime(proxy["fecha_disponibilidad_proxy"])
    proxy["fecha_publicacion_ingresos"] = pd.to_datetime(proxy["fecha_publicacion_ingresos"])
    return proxy.sort_values("trimestre").reset_index(drop=True), selected


def forecast_metal_proxy_from_history(
    selected: pd.DataFrame, origin: str, cutoff: pd.Timestamp, horizon: int
) -> tuple[np.ndarray, pd.Timestamp]:
    """Proyecta el proxy sin datos posteriores: volúmenes constantes y deriva de precios observada."""
    origin_key = quarter_key(origin)
    available = selected.loc[
        (selected["_key"] <= origin_key)
        & (selected["fecha_disponible"] <= cutoff)
    ].copy()
    if available.empty:
        raise ValueError("No hay datos de metales disponibles en el origen.")
    last_available = available["fecha_disponible"].max()
    projections = np.zeros(horizon, dtype=float)
    for metal in ["oro", "plata", "cobre", "zinc", "plomo"]:
        history = available.loc[available["metal"].eq(metal)].sort_values("_key")
        if history.empty:
            raise ValueError(f"No hay observaciones disponibles para proyectar {metal}.")
        prices = history["precio_realizado_usd"].astype(float).to_numpy()
        volumes = history["volumen"].astype(float).to_numpy()
        if len(prices) != len(volumes) or np.any(prices <= 0) or np.any(volumes <= 0):
            raise ValueError(f"La historia de {metal} contiene datos inválidos.")
        drift = float(np.diff(np.log(prices)).mean()) if len(prices) > 1 else 0.0
        future_price = prices[-1] * np.exp(drift * np.arange(1, horizon + 1))
        # El volumen se mantiene en el último total reportado; no se inventa crecimiento operativo.
        projections += volumes[-1] * future_price / 1_000_000
    if np.any(~np.isfinite(projections)) or np.any(projections <= 0):
        raise ValueError("La trayectoria futura del proxy tiene valores no válidos.")
    return projections, last_available


def fit_sarimax_with_metal_proxy(
    history_y: np.ndarray | list[float],
    history_x: pd.DataFrame,
    history_proxy: np.ndarray | list[float],
    future_x: pd.DataFrame,
    future_proxy: np.ndarray | list[float],
) -> tuple[np.ndarray, dict]:
    """Especificación experimental: el SARIMAX vigente más un único proxy agregado de metales."""
    y = np.asarray(history_y, dtype=float)
    hist_proxy = np.asarray(history_proxy, dtype=float)
    fut_proxy = np.asarray(future_proxy, dtype=float)
    if len(y) < MIN_TRAIN_OBS or len(hist_proxy) != len(y):
        raise ValueError("El experimento necesita al menos 20 trimestres pareados.")
    if len(fut_proxy) != len(future_x) or np.any(~np.isfinite(hist_proxy)) or np.any(~np.isfinite(fut_proxy)):
        raise ValueError("El proxy futuro no está completo o no está alineado con los horizontes.")
    if np.any(y <= 0) or np.any(hist_proxy <= 0) or np.any(fut_proxy <= 0):
        raise ValueError("Los ingresos y valores brutos del proxy deben ser positivos.")

    history_all = history_x[EXOG_COLS].astype(float).reset_index(drop=True).copy()
    future_all = future_x[EXOG_COLS].astype(float).reset_index(drop=True).copy()
    history_all[METAL_PROXY_FEATURE] = hist_proxy
    future_all[METAL_PROXY_FEATURE] = fut_proxy
    if history_all.isna().any().any() or future_all.isna().any().any():
        raise ValueError("La matriz experimental contiene datos faltantes.")

    log_y = np.log(y)
    log_x_history = np.log(history_all)
    log_x_future = np.log(future_all)
    y_change = np.diff(log_y)
    x_change = log_x_history.diff().iloc[1:].reset_index(drop=True)
    x_future_path = pd.concat([log_x_history.iloc[[-1]], log_x_future], ignore_index=True)
    x_future_change = x_future_path.diff().iloc[1:].reset_index(drop=True)
    center = x_change.mean(axis=0)
    scale = x_change.std(axis=0, ddof=0).replace(0, 1.0)
    x_train = (x_change - center) / scale
    x_future = (x_future_change - center) / scale
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = SARIMAX(
            y_change, exog=x_train,
            order=(1, 0, 0), seasonal_order=(0, 0, 1, 4), trend="c",
            enforce_stationarity=False, enforce_invertibility=False,
        ).fit(disp=False, maxiter=400)
        converged = bool(fitted.mle_retvals.get("converged", False))
        iterations = fitted.mle_retvals.get("iterations")
        if not converged:
            raise RuntimeError("El SARIMAX experimental no convergió en este origen.")
        changes = np.asarray(
            fitted.get_forecast(steps=len(future_x), exog=x_future).predicted_mean,
            dtype=float,
        )
    prediction = np.exp(log_y[-1] + np.cumsum(changes))
    if np.any(~np.isfinite(prediction)):
        raise ValueError("El SARIMAX experimental devolvió pronósticos no finitos.")
    return np.maximum(prediction, 0.0), {"convergio": converged, "iteraciones": iterations}


@st.cache_data(show_spinner=False)
def backtest_metal_proxy(
    sales: pd.DataFrame,
    model_data: pd.DataFrame,
    reference_backtest: pd.DataFrame,
    metal_data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prueba pareada del proxy: mismos orígenes y horizontes que el SARIMAX de referencia."""
    proxy, selected = prepare_comparable_metal_proxy(sales, metal_data)
    ordered = model_data.sort_values("_key").reset_index(drop=True)
    proxy = proxy.set_index("trimestre")
    selected = selected.copy()
    candidate_rows, diagnostics = [], []
    reference_rows = reference_backtest.loc[
        reference_backtest["modelo"].eq(MODEL_SARIMAX_EXOG)
    ].copy()
    for origin in sorted(reference_rows["origen"].unique(), key=quarter_key):
        try:
            origin_row = ordered.loc[ordered["trimestre"].eq(origin)]
            if len(origin_row) != 1:
                raise ValueError("No se identificó un único trimestre de origen.")
            origin_pos = int(origin_row.index[0])
            history = ordered.iloc[:origin_pos + 1].copy()
            cutoff = pd.to_datetime(
                sales.loc[sales["trimestre"].eq(origin), "fecha_publicacion_ingresos"].iloc[0]
            )
            history_labels = history["trimestre"].astype(str).tolist()
            proxy_history = proxy.loc[
                proxy.index.isin(history_labels)
                & (proxy["fecha_disponibilidad_proxy"] <= cutoff),
                METAL_PROXY_FEATURE,
            ].reindex(history_labels)
            if proxy_history.isna().any():
                raise ValueError("La historia del proxy no estaba completa y disponible en el origen.")
            future_proxy, last_proxy_available = forecast_metal_proxy_from_history(
                selected, origin, cutoff, BACKTEST_HORIZON
            )
            if last_proxy_available > cutoff:
                raise ValueError("La trayectoria usa información publicada después del origen.")
            future_macro = forecast_exog_from_history(
                history[EXOG_COLS].reset_index(drop=True), BACKTEST_HORIZON
            )
            predictions, fit_info = fit_sarimax_with_metal_proxy(
                history[TARGET].astype(float).to_numpy(),
                history[EXOG_COLS].reset_index(drop=True),
                proxy_history.astype(float).to_numpy(),
                future_macro,
                future_proxy,
            )
            expected = reference_rows.loc[reference_rows["origen"].eq(origin)].sort_values("horizonte")
            if len(expected) != BACKTEST_HORIZON:
                raise ValueError("El origen de referencia no tiene los cuatro horizontes completos.")
            for step, row in enumerate(expected.itertuples(index=False), start=1):
                if quarter_key(row.trimestre) <= quarter_key(origin):
                    raise ValueError("Se detectó un horizonte anterior o igual al origen.")
                candidate_rows.append({
                    "origen": origin, "horizonte": step, "trimestre": row.trimestre,
                    "ventas_reales": float(row.ventas_reales),
                    "modelo": "SARIMAX referencia + proxy de ventas por metal",
                    "pronostico": float(predictions[step - 1]),
                    "valor_proxy_futuro_usd_millones": float(future_proxy[step - 1]),
                    "fecha_corte_informacion": cutoff,
                })
            diagnostics.append({
                "origen": origin, "fecha_corte_informacion": cutoff,
                "fecha_máxima_dato_proxy": last_proxy_available,
                "observaciones_proxy_entrenamiento": len(proxy_history),
                "convergencia": fit_info["convergio"], "iteraciones": fit_info["iteraciones"],
                "estado": "Correcto",
            })
        except Exception as exc:
            diagnostics.append({
                "origen": origin, "estado": f"No evaluado: {exc}",
                "convergencia": False,
            })

    candidate = pd.DataFrame(candidate_rows)
    diag = pd.DataFrame(diagnostics)
    if candidate.empty:
        return pd.DataFrame(), pd.DataFrame(), diag, pd.DataFrame()
    common_origins = set(candidate["origen"]) & set(reference_rows["origen"])
    candidate = candidate[candidate["origen"].isin(common_origins)].copy()
    reference = reference_rows[reference_rows["origen"].isin(common_origins)].copy()
    reference = reference[["origen", "horizonte", "trimestre", "ventas_reales", "modelo", "pronostico"]]
    keys = ["origen", "horizonte", "trimestre"]
    if set(map(tuple, candidate[keys].to_numpy())) != set(map(tuple, reference[keys].to_numpy())):
        raise ValueError("El proxy y la referencia no comparten exactamente los mismos orígenes y horizontes.")
    paired = pd.concat([reference, candidate[reference.columns]], ignore_index=True)
    paired["error_firmado"] = paired["ventas_reales"] - paired["pronostico"]
    paired["error_absoluto"] = paired["error_firmado"].abs()
    paired["error_cuadratico"] = paired["error_firmado"] ** 2
    metric_rows = []
    for model, group in paired.groupby("modelo"):
        metric_rows.append({
            "Modelo": model,
            "MAE (US$ M)": float(group["error_absoluto"].mean()),
            "RMSE (US$ M)": float(np.sqrt(group["error_cuadratico"].mean())),
            "Pronósticos evaluados": int(len(group)),
            "Orígenes": int(group["origen"].nunique()),
        })
    metrics = pd.DataFrame(metric_rows)
    horizon_rows = []
    for (model, horizon), group in paired.groupby(["modelo", "horizonte"]):
        horizon_rows.append({
            "Modelo": model, "Horizonte": int(horizon),
            "MAE (US$ M)": float(group["error_absoluto"].mean()),
            "RMSE (US$ M)": float(np.sqrt(group["error_cuadratico"].mean())),
            "Pronósticos evaluados": int(len(group)),
        })
    by_horizon = pd.DataFrame(horizon_rows)
    return metrics, paired, diag, by_horizon


def forecast_table(model_data: pd.DataFrame, scenario_path: pd.DataFrame) -> pd.DataFrame:
    """Pronóstico SARIMAX usando una trayectoria explícita de regresores por trimestre."""
    ordered = model_data.sort_values("_key").reset_index(drop=True)
    y = ordered[TARGET].astype(float).to_numpy()
    x = ordered[EXOG_COLS].astype(float).reset_index(drop=True)
    if scenario_path.empty or not set(EXOG_COLS).issubset(scenario_path.columns):
        raise ValueError("La tabla de supuestos debe contener cobre, plata y USD/PEN.")
    future_x = scenario_path[EXOG_COLS].astype(float).reset_index(drop=True)
    if future_x.isna().any().any() or (future_x <= 0).any().any():
        raise ValueError("Completa la tabla con valores positivos para todos los indicadores.")
    future_labels = scenario_path["trimestre"].astype(str).tolist()
    predictions = fit_sarimax_exog(y, x, future_x)
    rows = []
    previous = float(y[-1])
    for label, value, exog in zip(future_labels, predictions, future_x.to_dict("records")):
        key = quarter_key(label)
        year_ago = key_to_quarter(key - 4)
        year_ago_values = ordered.loc[ordered["trimestre"].eq(year_ago), TARGET]
        year_ago_value = float(year_ago_values.iloc[0]) if not year_ago_values.empty else np.nan
        rows.append({
            "trimestre": label,
            "pronostico_usd_millones": float(value),
            "cambio_vs_trimestre_anterior_usd_millones": float(value - previous),
            "variacion_vs_trimestre_anterior_pct": float((value / previous - 1.0) * 100.0) if previous else np.nan,
            "ingreso_trimestre_anterior_usd_millones": previous,
            "mismo_trimestre_anio_anterior_usd_millones": year_ago_value,
            "precio_cobre_escenario_usd_t": float(exog["precio_cobre_usd_t"]),
            "precio_plata_escenario_usd_oz": float(exog["precio_plata_usd_oz"]),
            "tipo_cambio_escenario_pen_usd": float(exog["tipo_cambio_pen_usd"]),
            "modelo": MODEL_SARIMAX_EXOG,
        })
        previous = float(value)
    return pd.DataFrame(rows)


# =====================================================================
# APLICACIÓN · navegación clara para usuarios no técnicos
# =====================================================================
def _apply_scenario() -> None:
    st.session_state["scenario_horizon"] = int(st.session_state["input_horizon"])
    for name in ("copper", "silver", "fx"):
        st.session_state[f"scenario_{name}_pct"] = float(st.session_state[f"input_{name}_pct"])


def _reset_scenario() -> None:
    st.session_state["scenario_horizon"] = 8
    st.session_state["input_horizon"] = 8
    for name in ("copper", "silver", "fx"):
        st.session_state[f"scenario_{name}_pct"] = 0.0
        st.session_state[f"input_{name}_pct"] = 0.0

sales_file = find_sales_file()
if sales_file is None:
    st.error(f"No encuentro **{SALES_FILENAME}**. Coloca la base auditada dentro de `data`. La aplicación no crea ni modifica bases.")
    st.stop()
try:
    sales = load_sales(str(sales_file))
    model_data = prepare_reference_data(sales)
    external_metadata = metadata_from_candidate(sales)
except Exception as exc:
    st.error(f"No se pudieron validar los datos de ingresos y sus variables externas: {exc}")
    st.stop()

try:
    model_metrics, backtest = cached_backtest(model_data)
except Exception as exc:
    st.error(f"No se pudo calcular la evaluación histórica del modelo: {exc}")
    st.stop()

selected_model_name = MODEL_SARIMAX_EXOG
selected_rows = model_metrics.loc[model_metrics["Modelo"].eq(selected_model_name)]
if selected_rows.empty:
    st.error("El SARIMAX seleccionado no produjo métricas válidas. No se mostrará un pronóstico alternativo como si fuera el modelo elegido.")
    st.stop()
selected_metrics = selected_rows.iloc[0]
selected_mae = float(selected_metrics["MAE (US$ M)"])
selected_rmse = float(selected_metrics["RMSE (US$ M)"])
last_sales = float(sales[TARGET].iloc[-1])
last_sales_quarter = str(sales["trimestre"].iloc[-1])
latest_driver = model_data.iloc[-1]

# El ajuste que se edita se separa del escenario aplicado a la gráfica.
st.session_state.setdefault("scenario_horizon", 8)
st.session_state.setdefault("input_horizon", st.session_state["scenario_horizon"])
for name in ("copper", "silver", "fx"):
    st.session_state.setdefault(f"scenario_{name}_pct", 0.0)
    st.session_state.setdefault(f"input_{name}_pct", st.session_state[f"scenario_{name}_pct"])
copper_reference = float(latest_driver["precio_cobre_usd_t"])
silver_reference = float(latest_driver["precio_plata_usd_oz"])
fx_reference = float(latest_driver["tipo_cambio_pen_usd"])

with st.sidebar:
    try:
        st.image(OFFICIAL_LOGO_URL, width=175)
    except Exception:
        st.markdown("## Compañía de Minas Buenaventura")
    st.image(str(APP_DIR / "assets" / "mining_sidebar.jpg"), width=235)
    st.caption("Minería peruana · inteligencia de ventas")
    page_choice = st.radio(
        "Secciones",
        ["Resumen", "Probar escenarios", "Cómo interpretar", "Calidad del modelo", "Datos y fuentes"],
        index=0,
        key="section_navigation_v3",
    )
    st.divider()
    st.markdown("**Modelo en uso**")
    st.caption("SARIMAX con cobre, plata y USD/PEN")
    st.caption("Herramienta analítica independiente; no es una proyección oficial de Buenaventura.")
    st.link_button("Ver portal de inversionistas", OFFICIAL_INVESTOR_URL, use_container_width=True)

# El resumen es la pantalla inicial; se muestra sin agregar un botón al menú.
page = page_choice or "Resumen"

# Al cambiar de sección, devolver la vista al comienzo del contenido.
previous_page = st.session_state.get("_previous_page_view")
if previous_page is not None and previous_page != page:
    st.components.v1.html(
        """
        <script>
        const parentWindow = window.parent;
        parentWindow.requestAnimationFrame(() => {
          const main = parentWindow.document.querySelector('[data-testid="stMain"]');
          const app = parentWindow.document.querySelector('[data-testid="stAppViewContainer"]');
          if (main) main.scrollTo({top: 0, behavior: "auto"});
          if (app) app.scrollTo({top: 0, behavior: "auto"});
          parentWindow.scrollTo(0, 0);
        });
        </script>
        """,
        height=0,
    )
st.session_state["_previous_page_view"] = page

if page == "Resumen":
    st.markdown(
        """
        <section class="bvn-hero">
          <div class="bvn-hero-kicker">Ingresos · planificación trimestral</div>
          <h1>Pronóstico de ingresos</h1>
          <p>Revisa los ingresos publicados, cambia los supuestos de cobre, plata y dólar, y observa cómo cambia la estimación.</p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.info("La proyección es una estimación condicionada a los supuestos; no es una cifra oficial ni una garantía de ingresos.")

# Los controles en edición actualizan el precio equivalente, pero la gráfica
# conserva el último escenario aplicado hasta pulsar el botón.
scenario_page = page == "Probar escenarios"
if scenario_page:
    st.markdown(
        '<section class="bvn-scenario-panel"><h3>Ajusta el escenario</h3>'
        '<div class="bvn-scenario-note">Mueve los porcentajes para ver el precio equivalente. La gráfica cambia solo al pulsar “Actualizar pronóstico y gráfica”. El ajuste se aplica una vez frente a la referencia y el nivel resultante se mantiene en el horizonte.</div></section>',
        unsafe_allow_html=True,
    )
    horizon_labels = {
        1: "1 trimestre", 2: "2 trimestres", 3: "3 trimestres",
        4: "1 año · 4 trimestres", 8: "2 años · 8 trimestres",
        12: "3 años · 12 trimestres (exploratorio)",
    }
    st.selectbox(
        "Horizonte del pronóstico",
        [1, 2, 3, 4, 8, 12],
        format_func=lambda value: horizon_labels[value],
        key="input_horizon",
    )
    ccol, scol, fcol = st.columns(3)
    with ccol:
        st.slider("Cobre · ajuste vs. referencia", -20.0, 20.0, 0.0, 1.0, format="%.0f%%", key="input_copper_pct")
        copper_preview = copper_reference * (1 + float(st.session_state["input_copper_pct"]) / 100)
        st.markdown(f'<div class="bvn-price-preview"><span>Referencia</span><b>$ {copper_reference:,.0f}/t</b><span>Con {float(st.session_state["input_copper_pct"]):+.0f}%</span><b>$ {copper_preview:,.0f}/t</b></div>', unsafe_allow_html=True)
    with scol:
        st.slider("Plata · ajuste vs. referencia", -20.0, 20.0, 0.0, 1.0, format="%.0f%%", key="input_silver_pct")
        silver_preview = silver_reference * (1 + float(st.session_state["input_silver_pct"]) / 100)
        st.markdown(f'<div class="bvn-price-preview"><span>Referencia</span><b>$ {silver_reference:,.2f}/oz</b><span>Con {float(st.session_state["input_silver_pct"]):+.0f}%</span><b>$ {silver_preview:,.2f}/oz</b></div>', unsafe_allow_html=True)
    with fcol:
        st.slider("USD/PEN · ajuste vs. referencia", -10.0, 10.0, 0.0, 0.5, format="%.1f%%", key="input_fx_pct")
        fx_preview = fx_reference * (1 + float(st.session_state["input_fx_pct"]) / 100)
        st.markdown(f'<div class="bvn-price-preview"><span>Referencia</span><b>S/ {fx_reference:,.3f}</b><span>Con {float(st.session_state["input_fx_pct"]):+.1f}%</span><b>S/ {fx_preview:,.3f}</b></div>', unsafe_allow_html=True)
    st.button("Actualizar pronóstico y gráfica", on_click=_apply_scenario, type="primary", use_container_width=True)
    reset_col, note_col = st.columns([1, 3])
    with reset_col:
        st.button("Volver a referencias (0%)", on_click=_reset_scenario)
    with note_col:
        st.caption("El modelo activo usa cobre, plata y USD/PEN. El oro no forma parte de esta especificación.")
    pending = (
        int(st.session_state["input_horizon"]) != int(st.session_state["scenario_horizon"])
        or any(float(st.session_state[f"input_{name}_pct"]) != float(st.session_state[f"scenario_{name}_pct"]) for name in ("copper", "silver", "fx"))
    )
    if pending:
        st.info("Vista previa: los valores bajo los controles ya cambiaron; la gráfica conserva el último escenario aplicado hasta que pulses actualizar.")


horizon = int(st.session_state.get("scenario_horizon", 8))
copper_pct = float(st.session_state["scenario_copper_pct"])
silver_pct = float(st.session_state["scenario_silver_pct"])
fx_pct = float(st.session_state["scenario_fx_pct"])
copper_reference = float(latest_driver["precio_cobre_usd_t"])
silver_reference = float(latest_driver["precio_plata_usd_oz"])
fx_reference = float(latest_driver["tipo_cambio_pen_usd"])
copper_level = copper_reference * (1 + copper_pct / 100)
silver_level = silver_reference * (1 + silver_pct / 100)
fx_level = fx_reference * (1 + fx_pct / 100)
next_quarter_label = next_quarter(last_sales_quarter)
scenario_quarters = [key_to_quarter(quarter_key(next_quarter_label) + i) for i in range(horizon)]
scenario_path = pd.DataFrame({
    "trimestre": scenario_quarters,
    "precio_cobre_usd_t": [copper_level] * horizon,
    "precio_plata_usd_oz": [silver_level] * horizon,
    "tipo_cambio_pen_usd": [fx_level] * horizon,
})
try:
    forecast = forecast_table(model_data, scenario_path)
except Exception as exc:
    st.error(f"No se pudo calcular el pronóstico con esos supuestos: {exc}")
    st.stop()

first_row = forecast.iloc[0]
first_value = float(first_row["pronostico_usd_millones"])
first_change_pct = float(first_row["variacion_vs_trimestre_anterior_pct"])
last_row = forecast.iloc[-1]
last_value = float(last_row["pronostico_usd_millones"])
if page == "Probar escenarios" and horizon > 4:
    st.caption(f"Nota: el escenario llega hasta {last_row['trimestre']}. Más allá de cuatro trimestres es exploratorio: el backtest no valida ese horizonte y la incertidumbre aumenta.")

if page in ("Resumen", "Probar escenarios"):
    st.subheader("Ingresos observados y estimación")
    st.caption(f"Datos observados hasta {last_sales_quarter} · {len(sales)} trimestres históricos")
    k1, k2, k3 = st.columns(3)
    k1.metric("Último ingreso publicado", f"$ {last_sales:,.1f} M", last_sales_quarter)
    k2.metric(f"Estimación para {first_row['trimestre']}", f"$ {first_value:,.1f} M", f"{first_change_pct:+.1f}% vs. trimestre previo")
    with k3:
        st.markdown(
            f'<div class="bvn-metric-card"><div class="bvn-metric-label">Error medio histórico (MAE)</div>'
            f'<div class="bvn-metric-value" translate="no">$ {selected_mae:,.1f} M</div>'
            '<div class="bvn-metric-note">Backtest; no es un margen garantizado</div></div>',
            unsafe_allow_html=True,
        )
    direction = "mayor" if first_change_pct > 0 else "menor" if first_change_pct < 0 else "igual"
    st.markdown(
        f'<div class="bvn-interpretation"><b>Lectura sencilla:</b> con estos supuestos, el ingreso estimado para {first_row["trimestre"]} es <b>$ {first_value:,.1f} M</b>, {abs(first_change_pct):.1f}% {direction} que el último trimestre publicado. '
        'El modelo activo usa cobre, plata y USD/PEN; el precio del oro no forma parte de esta especificación.</div>',
        unsafe_allow_html=True,
    )

    chart = go.Figure()
    chart.add_trace(go.Scatter(
        x=sales["trimestre"], y=sales[TARGET], mode="lines+markers", name="Ingresos publicados",
        line=dict(color="#174b37", width=3), marker=dict(size=6, color="#174b37"),
        hovertemplate="<b>%{x}</b><br>$ %{y:,.1f} M<extra>Ingresos publicados</extra>",
    ))
    chart.add_trace(go.Scatter(
        x=[last_sales_quarter] + forecast["trimestre"].tolist(),
        y=[last_sales] + forecast["pronostico_usd_millones"].astype(float).tolist(),
        mode="lines+markers", name="Pronóstico SARIMAX",
        line=dict(color="#70a83b", width=4, dash="dash"), marker=dict(size=8, color="#70a83b"),
        hovertemplate="<b>%{x}</b><br>$ %{y:,.1f} M<extra>Pronóstico</extra>",
    ))
    chart.add_vline(x=last_sales_quarter, line_dash="dot", line_color="#718679", line_width=2)
    quarter_labels = sales["trimestre"].astype(str).tolist() + forecast["trimestre"].astype(str).tolist()
    tick_labels = [label for idx, label in enumerate(quarter_labels) if idx % 2 == 0 or idx == len(quarter_labels) - 1]
    chart.update_layout(
        height=400, template="plotly_white", margin=dict(l=74, r=20, t=46, b=82),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="#ffffff", hovermode="x unified",
        font=dict(family="Arial, sans-serif", size=14, color="#20352b"),
        legend=dict(orientation="h", y=1.14, x=0, font=dict(size=13, color="#20352b")),
        xaxis=dict(
            title=dict(text="Trimestre", font=dict(size=14, color="#20352b")),
            tickmode="array", tickvals=tick_labels, ticktext=tick_labels, tickangle=-35,
            tickfont=dict(size=11, color="#405449"), showgrid=False, automargin=True,
            linecolor="#9aac9f", ticks="outside",
        ),
        yaxis=dict(
            title=dict(text="Ingresos trimestrales (US$ millones)", font=dict(size=14, color="#20352b")),
            tickfont=dict(size=12, color="#405449"), tickformat=",.0f",
            rangemode="tozero", gridcolor="#e2e9e4", zerolinecolor="#9aac9f",
            automargin=True, ticks="outside",
        ),
        hoverlabel=dict(bgcolor="#ffffff", bordercolor="#174b37", font=dict(color="#20352b", size=13)),
    )
    st.plotly_chart(chart, width="stretch", theme=None, config={"displaylogo": False, "responsive": True})
    st.caption("La línea oscura muestra ingresos publicados. La línea verde discontinua muestra el pronóstico del escenario elegido. Pasa el cursor sobre un punto para ver el valor exacto.")

    if page == "Resumen":
        st.caption(
            f"Referencias → escenario: cobre $ {copper_reference:,.0f}/t → $ {copper_level:,.0f}/t ({copper_pct:+.0f}%) · "
            f"plata $ {silver_reference:,.2f}/oz → $ {silver_level:,.2f}/oz ({silver_pct:+.0f}%) · "
            f"USD/PEN S/ {fx_reference:,.3f} → S/ {fx_level:,.3f} ({fx_pct:+.1f}%)."
        )
        st.markdown(
            '<div class="bvn-card"><h3>Cómo leerlo</h3><div class="bvn-muted">Compara la estimación con los ingresos observados. Los precios internacionales son referencias: pueden diferir de los precios realizados por la empresa y no incorporan por sí solos volúmenes, costos u otros factores.</div></div>',
            unsafe_allow_html=True,
        )
    else:
        st.subheader("Detalle del escenario")
        show = forecast[[
            "trimestre", "pronostico_usd_millones", "variacion_vs_trimestre_anterior_pct",
            "precio_cobre_escenario_usd_t", "precio_plata_escenario_usd_oz", "tipo_cambio_escenario_pen_usd",
        ]].copy()
        show.columns = [
            "Trimestre", "Ingresos estimados (US$ M)", "Cambio trimestral (%)",
            "Cobre asumido (US$/t)", "Plata asumida (US$/oz)", "USD/PEN asumido",
        ]
        show["Cambio cobre vs. referencia (%)"] = (show["Cobre asumido (US$/t)"] / float(latest_driver["precio_cobre_usd_t"]) - 1) * 100
        show["Cambio plata vs. referencia (%)"] = (show["Plata asumida (US$/oz)"] / float(latest_driver["precio_plata_usd_oz"]) - 1) * 100
        show["Cambio USD/PEN vs. referencia (%)"] = (show["USD/PEN asumido"] / float(latest_driver["tipo_cambio_pen_usd"]) - 1) * 100
        st.dataframe(
            show, hide_index=True, width="stretch",
            column_config={
                "Ingresos estimados (US$ M)": st.column_config.NumberColumn(format="US$ %.1f M"),
                "Cambio trimestral (%)": st.column_config.NumberColumn(format="%.1f%%"),
                "Cobre asumido (US$/t)": st.column_config.NumberColumn(format="%.0f"),
                "Plata asumida (US$/oz)": st.column_config.NumberColumn(format="%.2f"),
                "USD/PEN asumido": st.column_config.NumberColumn(format="%.3f"),
            },
        )
        st.download_button(
            "Descargar este escenario",
            forecast.to_csv(index=False).encode("utf-8-sig"),
            "escenario_ingresos_buenaventura.csv",
            "text/csv",
            type="primary",
        )
        st.caption("Los porcentajes son ajustes únicos frente a la última referencia observada; el nivel resultante se mantiene en cada trimestre proyectado.")

elif page == "Cómo interpretar":
    st.subheader("Cómo interpretar el pronóstico")
    st.markdown("### En una frase")
    st.success(f"Con los supuestos actuales, el modelo estima **$ {first_value:,.1f} M** para {first_row['trimestre']}. El último ingreso publicado fue **$ {last_sales:,.1f} M** en {last_sales_quarter}.")
    a, b, c = st.columns(3)
    with a:
        st.markdown('<div class="bvn-card"><h3>Es una estimación</h3><div class="bvn-muted">El resultado cambia cuando cambian los supuestos de cobre, plata o USD/PEN. No es un compromiso ni una cifra oficial.</div></div>', unsafe_allow_html=True)
    with b:
        st.markdown('<div class="bvn-card"><h3>No es utilidad</h3><div class="bvn-muted">El modelo estima ingresos totales. No estima EBITDA, utilidad neta, costos ni flujo de caja.</div></div>', unsafe_allow_html=True)
    with c:
        st.markdown('<div class="bvn-card"><h3>Los metales son referencias</h3><div class="bvn-muted">Los precios internacionales pueden diferir del precio realizado por Buenaventura, sus volúmenes vendidos y sus condiciones comerciales.</div></div>', unsafe_allow_html=True)
    st.markdown("### Cómo leer el error")
    e1, e2 = st.columns(2)
    e1.metric("MAE histórico", f"$ {selected_mae:,.1f} M", help="En promedio, los pronósticos históricos se apartaron esta cantidad de los ingresos reales. No es un intervalo de confianza.")
    e2.metric("RMSE histórico", f"$ {selected_rmse:,.1f} M", help="La raíz del error cuadrático medio da más peso a errores grandes; tampoco es un intervalo de confianza.")
    st.caption("Las métricas provienen de ocho orígenes históricos y 32 pronósticos; es una muestra pequeña. Úsalas como referencia, no como margen exacto para un trimestre.")

elif page == "Calidad del modelo":
    st.subheader("Qué tan bien funcionó en la evaluación histórica")
    st.markdown("El backtest usa únicamente información disponible hasta cada origen. Los regresores futuros se proyectan desde ese origen, sin usar valores observados después.")
    m1, m2, m3 = st.columns(3)
    m1.metric("Error medio (MAE)", f"$ {selected_mae:,.1f} M", help="Promedio de las diferencias absolutas entre estimación e ingreso real.")
    m2.metric("Penaliza más errores grandes (RMSE)", f"$ {selected_rmse:,.1f} M", help="La raíz del error cuadrático medio da más peso a errores grandes.")
    n_origins = int(backtest["origen"].nunique())
    n_predictions = int(backtest.loc[backtest["modelo"].eq(selected_model_name)].shape[0])
    m3.metric("Evaluación histórica", f"{n_predictions} pronósticos", f"{n_origins} orígenes")
    selected_backtest = backtest.loc[backtest["modelo"].eq(selected_model_name)].copy()
    details = selected_backtest.groupby("horizonte", as_index=False).agg(
        MAE=("error_absoluto", "mean"),
        RMSE=("error_cuadratico", lambda values: float(np.sqrt(values.mean()))),
        Pronósticos=("pronostico", "count"),
    )
    details["Horizonte"] = details["horizonte"].map(lambda value: f"{int(value)} trimestre" if int(value) == 1 else f"{int(value)} trimestres")
    details = details[["Horizonte", "Pronósticos", "MAE", "RMSE"]]
    st.dataframe(
        details, hide_index=True, width="stretch",
        column_config={
            "MAE": st.column_config.NumberColumn("MAE (US$ millones)", format="%.1f"),
            "RMSE": st.column_config.NumberColumn("RMSE (US$ millones)", format="%.1f"),
        },
    )
    st.warning("La historia es corta y los orígenes se solapan. Las métricas no garantizan que el siguiente pronóstico tenga un error similar.")
    with st.expander("Ver cada pronóstico evaluado"):
        detail = selected_backtest.sort_values(["origen", "horizonte"]).rename(columns={
            "origen": "Origen", "horizonte": "Horizonte", "trimestre": "Trimestre evaluado",
            "ventas_reales": "Ingreso real (US$ M)", "pronostico": "Pronóstico (US$ M)",
            "error_absoluto": "Error absoluto (US$ M)",
        })
        st.dataframe(
            detail[["Origen", "Horizonte", "Trimestre evaluado", "Ingreso real (US$ M)", "Pronóstico (US$ M)", "Error absoluto (US$ M)"]],
            hide_index=True, width="stretch", height=420,
        )

    st.markdown("---")
    st.subheader("Prueba experimental: ventas por metal y precio realizado")
    st.markdown(
        "Se compara el SARIMAX activo con una variante que añade **un solo indicador agregado**: "
        "volumen reportado × precio realizado para oro, plata, cobre, zinc y plomo. "
        "El indicador aproxima valor bruto de metal; no equivale a ingresos contables. "
        "Para oro se usa el precio realizado informado por Buenaventura, no la cotización internacional; esta prueba no cambia la especificación activa."
    )
    st.caption(
        "La base conserva el perímetro por metal. Desde 2025Q2, los totales reportados de cobre y plata "
        "incluyen Buenaventura Trading; sus componentes separados no se vuelven a sumar. "
        "Las cifras de asociadas quedan fuera del agregado."
    )
    try:
        metal_data = load_comparable_metal_base()
        with st.spinner("Evaluando la variable experimental con los mismos orígenes y horizontes…"):
            metal_metrics, metal_details, metal_diagnostics, metal_by_horizon = backtest_metal_proxy(
                sales, model_data, backtest, metal_data
            )
        candidate_name = "SARIMAX referencia + proxy de ventas por metal"
        reference_name = MODEL_SARIMAX_EXOG
        if metal_metrics.empty or candidate_name not in metal_metrics["Modelo"].tolist():
            st.warning("La variante con ventas por metal no pudo producir una comparación completa. Revisa los diagnósticos de convergencia y disponibilidad abajo.")
        else:
            reference_metric = metal_metrics.loc[metal_metrics["Modelo"].eq(reference_name)].iloc[0]
            candidate_metric = metal_metrics.loc[metal_metrics["Modelo"].eq(candidate_name)].iloc[0]
            if int(candidate_metric["Orígenes"]) < n_origins:
                st.warning(
                    f"La variante convergió en {int(candidate_metric['Orígenes'])} de {n_origins} orígenes. "
                    "Las métricas usan solo orígenes comunes y no representan la evaluación completa del modelo activo."
                )
            mae_change = (candidate_metric["MAE (US$ M)"] / reference_metric["MAE (US$ M)"] - 1) * 100
            rmse_change = (candidate_metric["RMSE (US$ M)"] / reference_metric["RMSE (US$ M)"] - 1) * 100
            mae_ref_col, mae_candidate_col = st.columns(2)
            mae_ref_col.metric("MAE · SARIMAX activo", f"$ {reference_metric['MAE (US$ M)']:,.1f} M")
            mae_candidate_col.metric(
                "MAE · con ventas por metal", f"$ {candidate_metric['MAE (US$ M)']:,.1f} M",
                f"{mae_change:+.1f}%", delta_color="inverse",
            )
            rmse_ref_col, rmse_candidate_col = st.columns(2)
            rmse_ref_col.metric("RMSE · SARIMAX activo", f"$ {reference_metric['RMSE (US$ M)']:,.1f} M")
            rmse_candidate_col.metric(
                "RMSE · con ventas por metal", f"$ {candidate_metric['RMSE (US$ M)']:,.1f} M",
                f"{rmse_change:+.1f}%", delta_color="inverse",
            )
            if mae_change < 0 and rmse_change < 0:
                st.success("En esta prueba retrospectiva, el indicador agregado reduce MAE y RMSE. Es evidencia inicial; no cambia automáticamente el modelo activo.")
            elif mae_change > 0 and rmse_change > 0:
                st.warning("En esta prueba retrospectiva, el indicador agregado aumenta MAE y RMSE. El SARIMAX activo sigue siendo la referencia; no se incorpora la variable al pronóstico.")
            else:
                st.info("Las métricas dan señales mixtas. La prueba no basta para afirmar que la variable mejore el pronóstico; el SARIMAX activo sigue siendo la referencia.")

            ref_horizon = metal_by_horizon.loc[
                metal_by_horizon["Modelo"].eq(reference_name),
                ["Horizonte", "MAE (US$ M)", "RMSE (US$ M)"],
            ].rename(columns={
                "MAE (US$ M)": "MAE activo",
                "RMSE (US$ M)": "RMSE activo",
            })
            candidate_horizon = metal_by_horizon.loc[
                metal_by_horizon["Modelo"].eq(candidate_name),
                ["Horizonte", "MAE (US$ M)", "RMSE (US$ M)"],
            ].rename(columns={
                "MAE (US$ M)": "MAE metales",
                "RMSE (US$ M)": "RMSE metales",
            })
            show_metrics = ref_horizon.merge(candidate_horizon, on="Horizonte", validate="one_to_one")
            show_metrics["Horizonte"] = show_metrics["Horizonte"].map(
                lambda value: f"{value} trimestre" if value == 1 else f"{value} trimestres"
            )
            st.markdown("#### Resultado por horizonte · mismos orígenes")
            horizon_lines = []
            for _, row in show_metrics.iterrows():
                horizon_lines.append(
                    f"- **{row['Horizonte']}** — MAE (activo / metales): "
                    f"$ {row['MAE activo']:.1f} / $ {row['MAE metales']:.1f} M · "
                    f"RMSE (activo / metales): $ {row['RMSE activo']:.1f} / $ {row['RMSE metales']:.1f} M"
                )
            st.markdown("\n".join(horizon_lines))
            st.caption(
                f"Comparación pareada: {int(candidate_metric['Orígenes'])} orígenes, "
                f"{int(candidate_metric['Pronósticos evaluados'])} pronósticos por modelo. "
                "En cada origen, los volúmenes futuros se mantienen en el último dato disponible y "
                "los precios realizados siguen su deriva histórica calculada hasta ese origen."
            )
        with st.expander("Revisar disponibilidad, perímetro y convergencia por origen"):
            if not metal_diagnostics.empty:
                st.dataframe(metal_diagnostics, hide_index=True, width="stretch")
            if not metal_details.empty:
                st.markdown("**Pronósticos del proxy por origen y trimestre**")
                candidate_detail = metal_details.loc[
                    metal_details["modelo"].eq("SARIMAX referencia + proxy de ventas por metal")
                ].rename(columns={
                    "origen": "Origen", "horizonte": "Horizonte", "trimestre": "Trimestre",
                    "ventas_reales": "Ingreso real (US$ M)", "pronostico": "Pronóstico (US$ M)",
                    "valor_proxy_futuro_usd_millones": "Proxy futuro (US$ M)",
                    "fecha_corte_informacion": "Fecha de corte de información",
                    "error_firmado": "Error con signo (US$ M)", "error_absoluto": "Error absoluto (US$ M)",
                })
                st.dataframe(candidate_detail, hide_index=True, width="stretch", height=360)
        long_download, proxy_download = st.columns(2)
        long_download.download_button(
            "Descargar base detallada por metal",
            metal_data.drop(columns=["_key"], errors="ignore").to_csv(index=False).encode("utf-8-sig"),
            METAL_BASE_FILENAME, "text/csv", type="primary",
        )
        proxy_path = APP_DIR / "data" / METAL_PROXY_FILENAME
        if proxy_path.is_file():
            proxy_download.download_button(
                "Descargar resumen trimestral de la prueba",
                proxy_path.read_bytes(), METAL_PROXY_FILENAME, "text/csv", type="primary",
            )
    except Exception as exc:
        st.error(f"No fue posible completar la prueba experimental de ventas por metal: {exc}")

else:
    st.subheader("Datos y fuentes")
    st.markdown(f"La base comprende **{len(sales)} trimestres**, desde {sales['trimestre'].iloc[0]} hasta {last_sales_quarter}. El ingreso más reciente está documentado con su fuente y fecha de publicación.")
    src1, src2 = st.columns(2)
    with src1:
        st.markdown("#### Ingresos")
        st.link_button("Consultar resultados oficiales de 2026Q3", Q3_2026_RESULTS_URL)
        st.caption("Cada observación del objetivo conserva su fuente dentro de la base.")
    with src2:
        st.markdown("#### Variables económicas")
        st.link_button("Banco Mundial · precios de metales", WORLD_BANK_PAGE)
        st.link_button("BCRP · tipo de cambio promedio", BCRP_FX_PAGE)
    st.markdown("#### Registro de las series")
    st.dataframe(external_metadata, hide_index=True, width="stretch")
    st.caption("Cobre y plata están expresados en US$ nominales; USD/PEN en soles por dólar. Los valores macroeconómicos históricos son revisados; no se dispone de vintages completas para reconstruir cada publicación pasada.")
    st.download_button("Descargar base utilizada", sales.to_csv(index=False).encode("utf-8-sig"), SALES_FILENAME, "text/csv")
    metal_path = APP_DIR / "data" / METAL_BASE_FILENAME
    if metal_path.is_file():
        st.download_button(
            "Descargar base de ventas por metal y perímetro",
            metal_path.read_bytes(), METAL_BASE_FILENAME, "text/csv",
        )
    proxy_path = APP_DIR / "data" / METAL_PROXY_FILENAME
    if proxy_path.is_file():
        st.download_button(
            "Descargar resumen trimestral del valor bruto por metal",
            proxy_path.read_bytes(), METAL_PROXY_FILENAME, "text/csv",
        )

st.markdown("<hr>", unsafe_allow_html=True)
st.caption(f"Buenaventura · Inteligencia de ventas | Modelo en uso: SARIMAX con cobre, plata y USD/PEN | Datos observados hasta {last_sales_quarter} | Base: {sales_file.name}")

