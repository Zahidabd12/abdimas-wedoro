import streamlit as st
import pandas as pd
import numpy as np
import math
from xgboost import XGBRegressor
from sklearn.model_selection import GroupShuffleSplit
import os
import plotly.graph_objects as go

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Tumbuh Kembang Anak",
    page_icon="👶",
    layout="wide",
)

# ─── CSS Custom ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #f8fafc;
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* 1. Global dark text override for main content area (light background) */
    div[data-testid="stMarkdownContainer"] p, 
    div[data-testid="stMarkdownContainer"] span, 
    div[data-testid="stRadio"] label p, 
    div[data-testid="stCheckbox"] label p, 
    label[data-testid="stWidgetLabel"] p {
        color: #1e293b !important;
    }
    
    /* 2. Specific sidebar light text overrides to override global rules for the dark sidebar background */
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] p, 
    section[data-testid="stSidebar"] div[data-testid="stMarkdownContainer"] span, 
    section[data-testid="stSidebar"] label[data-testid="stWidgetLabel"] p,
    section[data-testid="stSidebar"] div[data-testid="stCaptionContainer"] p,
    section[data-testid="stSidebar"] .stCaption p,
    section[data-testid="stSidebar"] small {
        color: #cbd5e1 !important;
    }
    
    section[data-testid="stSidebar"] h1, 
    section[data-testid="stSidebar"] h2, 
    section[data-testid="stSidebar"] h3, 
    section[data-testid="stSidebar"] h4 {
        color: #ffffff !important;
    }
    
    section[data-testid="stSidebar"] div[role="alert"] * {
        color: #ffffff !important;
    }
    
    /* Header Card */
    .header-card {
        background: linear-gradient(135deg, #4f46e5 0%, #3b82f6 100%);
        color: #ffffff !important;
        padding: 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(59, 130, 246, 0.3);
    }
    div[data-testid="stMarkdownContainer"] .header-card h1,
    div[data-testid="stMarkdownContainer"] .header-card p,
    div[data-testid="stMarkdownContainer"] .header-card b,
    div[data-testid="stMarkdownContainer"] .header-card span,
    .header-card h1,
    .header-card p,
    .header-card b,
    .header-card span {
        color: #ffffff !important;
    }
    .header-card h1 {
        font-weight: 800;
        margin-bottom: 0.5rem;
        font-size: 2.2rem;
        border: none;
    }
    .header-card p {
        font-size: 1.1rem;
        opacity: 0.9;
        margin: 0;
    }
    
    /* Segment headers */
    .section-header {
        font-size: 20px;
        font-weight: 700;
        color: #1e293b;
        border-left: 5px solid #3b82f6;
        padding-left: 12px;
        margin: 2rem 0 1rem;
    }
    
    /* Status Badges */
    .status-badge {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 30px;
        font-size: 12px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .status-normal   { background-color: #d1fae5; color: #065f46 !important; border: 1px solid #a7f3d0; }
    .status-warning  { background-color: #fef3c7; color: #92400e !important; border: 1px solid #fde68a; }
    .status-danger   { background-color: #fee2e2; color: #991b1b !important; border: 1px solid #fca5a5; }
    .status-info     { background-color: #e0f2fe; color: #0369a1 !important; border: 1px solid #bae6fd; }
    
    /* Tables */
    .modern-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        background: white;
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0,0,0,0.03);
        border: 1px solid #e2e8f0;
    }
    .modern-table th {
        background-color: #f8fafc;
        color: #64748b;
        font-weight: 600;
        text-align: center;
        padding: 14px;
        border-bottom: 1px solid #e2e8f0;
        font-size: 13px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .modern-table td {
        padding: 16px;
        border-bottom: 1px solid #f1f5f9;
        color: #334155;
        font-size: 14px;
        text-align: center;
    }
    .modern-table tr:last-child td {
        border-bottom: none;
    }
</style>
""", unsafe_allow_html=True)

# ─── Z-Score Functions ─────────────────────────────────────────────────────────
TABLE_COL_MAP = {
    "PBU_L": 0, "PBU_P": 5,
    "BBU_L": 10, "BBU_P": 15,
    "BBPB_L": 20, "BBPB_P": 25,
    "BBTB_L": 30, "BBTB_P": 35,
}

@st.cache_data
def load_who_tables(filepath):
    raw = pd.read_excel(filepath, sheet_name="Sheet1", header=None)
    tables = {}
    for name, col_start in TABLE_COL_MAP.items():
        chunk = raw.iloc[2:, col_start:col_start + 4].copy()
        chunk.columns = ["key", "L", "M", "S"]
        chunk = chunk.dropna(subset=["key", "L", "M", "S"])
        chunk = chunk.astype(float).reset_index(drop=True)
        chunk["key"] = chunk["key"].round(4)
        tables[name] = chunk
    return tables

def get_lms(tables, table_name, key_value):
    if table_name not in tables:
        return None
    tbl = tables[table_name]
    keys = tbl["key"].values
    if key_value < keys[0] or key_value > keys[-1]:
        return None
    exact = tbl[tbl["key"] == round(key_value, 4)]
    if not exact.empty:
        row = exact.iloc[0]
        return float(row["L"]), float(row["M"]), float(row["S"])
    idx_high = int(np.searchsorted(keys, key_value))
    idx_low = idx_high - 1
    t = (key_value - keys[idx_low]) / (keys[idx_high] - keys[idx_low])
    row0, row1 = tbl.iloc[idx_low], tbl.iloc[idx_high]
    L = float(row0["L"] + t * (row1["L"] - row0["L"]))
    M = float(row0["M"] + t * (row1["M"] - row0["M"]))
    S = float(row0["S"] + t * (row1["S"] - row0["S"]))
    return L, M, S

def _hitung_z(X, L, M, S):
    if X <= 0 or M <= 0:
        return None
    if abs(L) < 1e-6:
        return math.log(X / M) / S
    return ((X / M) ** L - 1) / (L * S)

def _pilih_tabel(indeks, jenis_kelamin):
    suffix = "L" if jenis_kelamin.upper() == "L" else "P"
    return f"{indeks}_{suffix}"

def zscore_BB_U(tables, bb_kg, umur_bulan, jenis_kelamin):
    if bb_kg is None or bb_kg <= 0 or umur_bulan is None or not (0 <= umur_bulan <= 60):
        return None
    lms = get_lms(tables, _pilih_tabel("BBU", jenis_kelamin), umur_bulan)
    return None if lms is None else _hitung_z(bb_kg, *lms)

def zscore_TB_U(tables, tb_cm, umur_bulan, jenis_kelamin):
    if tb_cm is None or tb_cm <= 0 or umur_bulan is None or not (0 <= umur_bulan <= 60):
        return None
    lms = get_lms(tables, _pilih_tabel("PBU", jenis_kelamin), umur_bulan)
    return None if lms is None else _hitung_z(tb_cm, *lms)

def zscore_BB_TB(tables, bb_kg, tb_cm, umur_bulan, jenis_kelamin):
    if bb_kg is None or bb_kg <= 0 or tb_cm is None or tb_cm <= 0 or umur_bulan is None:
        return None
    indeks = "BBPB" if umur_bulan <= 24 else "BBTB"
    lms = get_lms(tables, _pilih_tabel(indeks, jenis_kelamin), tb_cm)
    return None if lms is None else _hitung_z(bb_kg, *lms)

def status_BB_U(z):
    if z is None: return "N/A", "warning"
    if z < -3:    return "BB Sangat Kurang", "danger"
    if z < -2:    return "BB Kurang", "warning"
    if z <= 1:    return "BB Normal", "normal"
    return "Risiko Lebih", "info"

def status_TB_U(z):
    if z is None: return "N/A", "warning"
    if z < -3:    return "Sangat Pendek", "danger"
    if z < -2:    return "Pendek (Stunting)", "warning"
    if z <= 3:    return "Normal", "normal"
    return "Tinggi", "info"

def status_BB_TB(z):
    if z is None:  return "N/A", "warning"
    if z < -3:     return "Gizi Buruk", "danger"
    if z < -2:     return "Gizi Kurang", "warning"
    if z <= 1:     return "Gizi Baik", "normal"
    if z <= 2:     return "Risiko Gizi Lebih", "info"
    if z <= 3:     return "Gizi Lebih", "warning"
    return "Obesitas", "danger"

def badge(label, style):
    css = {"normal": "status-normal", "warning": "status-warning", "danger": "status-danger", "info": "status-info"}
    return f'<span class="status-badge {css.get(style, "status-warning")}">{label}</span>'

# Helper to calculate KMS curve
def get_z_curve(tables, table_name, z_val, keys_range):
    x_vals = []
    y_vals = []
    for k in keys_range:
        lms = get_lms(tables, table_name, k)
        if lms is not None:
            L, M, S = lms
            if abs(L) < 1e-6:
                X = M * math.exp(S * z_val)
            else:
                val = 1 + L * S * z_val
                if val > 0:
                    X = M * (val ** (1.0 / L))
                else:
                    X = None
            if X is not None:
                x_vals.append(k)
                y_vals.append(X)
    return x_vals, y_vals

# ─── Prepare Input ─────────────────────────────────────────────────────────────
def prepare_input(umur, sex, bb, tb, bb_prev=None, tb_prev=None):
    bb_diff = 0 if bb_prev is None else bb - bb_prev
    tb_diff = 0 if tb_prev is None else tb - tb_prev
    return pd.DataFrame({
        "umur": [umur], "sex": [sex],
        "BB": [bb], "TB": [tb],
        "bb_diff": [bb_diff], "tb_diff": [tb_diff]
    })

# ─── Train Model ───────────────────────────────────────────────────────────────
@st.cache_resource
def train_models(df_raw):
    df = df_raw.copy()
    df = df.loc[:, ~df.columns.str.contains("^Unnamed")]
    df = df.dropna(subset=["umur"])
    df["umur"] = pd.to_numeric(df["umur"], errors="coerce")
    df = df.dropna(subset=["umur"])
    df["tanggal_timbang"] = pd.to_datetime(df["tanggal_timbang"], format="%m/%d/%Y", errors="coerce")
    df = df.sort_values(by=["id", "tanggal_timbang"])
    df = df[df["umur"] <= 60]
    valid_id = df["id"].value_counts()
    valid_id = valid_id[valid_id >= 5].index
    df = df[df["id"].isin(valid_id)]
    df["sex"] = df["sex"].astype(str).str.strip()
    df = df[df["sex"].isin(["1", "2"])].copy()
    df["sex"] = df["sex"].map({"1": 1, "2": 0})
    df["BB"] = df["BB"].replace(0, np.nan)
    df["TB"] = df["TB"].replace(0, np.nan)
    df = df.dropna(subset=["BB", "TB"])
    df["bb_diff"] = df.groupby("id")["BB"].diff().fillna(0)
    df["tb_diff"] = df.groupby("id")["TB"].diff().fillna(0)
    df["bb_target_1m"] = df.groupby("id")["BB"].shift(-1)
    df["tb_target_1m"] = df.groupby("id")["TB"].shift(-1)
    df_clean = df.dropna(subset=["bb_target_1m", "tb_target_1m"]).copy()

    features = ["umur", "sex", "BB", "TB", "bb_diff", "tb_diff"]
    X = df_clean[features]
    y_bb = df_clean["bb_target_1m"]
    y_tb = df_clean["tb_target_1m"]

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, _ = next(gss.split(X, y_bb, groups=df_clean["id"]))
    X_train = X.iloc[train_idx]
    y_bb_train = y_bb.iloc[train_idx]
    y_tb_train = y_tb.iloc[train_idx]

    model_bb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model_bb.fit(X_train, y_bb_train)

    model_tb = XGBRegressor(n_estimators=200, learning_rate=0.05, max_depth=4, random_state=42)
    model_tb.fit(X_train, y_tb_train)

    return model_bb, model_tb, df_clean

# ─── Custom Styles for Premium Cards ──────────────────────────────────────────
def make_metric_card(title, value, unit, delta, delta_color, icon_type="weight", z_score=0.0):
    color_map = {
        "green": "#22c55e",
        "red": "#ef4444",
        "orange": "#f59e0b",
        "neutral": "#64748b"
    }
    col = color_map.get(delta_color, "#64748b")
    
    # Calculate progress bar percentage (Z-score from -3 to +3 mapped to 0-100%)
    z_score = 0.0 if z_score is None else float(z_score)
    percent = (z_score + 3) / 6 * 100
    percent = max(10.0, min(100.0, percent)) # clamp between 10% and 100% for visual aesthetics
    
    if icon_type == "weight":
        icon_bg = "#e0f2fe"
        icon_svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#0284c7" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <rect x="3" y="11" width="18" height="10" rx="2" ry="2"></rect>
            <path d="M12 2v9M8 5h8"></path>
        </svg>
        """
        bar_color = "#22c55e" # Green progress bar for weight
    else:
        icon_bg = "#fce7f3"
        icon_svg = """
        <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="28" height="28" fill="none" stroke="#db2777" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M5 3h14c1.1 0 2 .9 2 2v14c0 1.1-.9 2-2 2H5c-1.1 0-2-.9-2-2V5c0-1.1.9-2 2-2z"></path>
            <path d="M3 7h3M3 12h5M3 17h3"></path>
        </svg>
        """
        bar_color = "#7c3aed" # Purple/violet progress bar for height

    html_content = f"""
<div style="background: white; border-radius: 16px; padding: 24px; box-shadow: 0 4px 20px rgba(0,0,0,0.03); border: 1px solid #f1f5f9; display: flex; flex-direction: column; margin-bottom: 20px;">
<div style="display: flex; align-items: center; gap: 16px;">
<div style="width: 52px; height: 52px; border-radius: 12px; background-color: {icon_bg}; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">
{icon_svg}
</div>
<div style="display: flex; flex-direction: column; text-align: left;">
<div style="font-size: 14px; color: #64748b; font-weight: 500; line-height: 1.2;">{title}</div>
<div style="font-size: 26px; font-weight: 700; color: #0f172a; margin-top: 4px; display: flex; align-items: baseline; gap: 8px;">
{value} <span style="font-size: 16px; font-weight: 500; color: #64748b;">{unit}</span>
<span style="font-size: 14px; font-weight: 600; color: {col};">({delta})</span>
</div>
</div>
</div>
<div style="background-color: #f1f5f9; height: 6px; border-radius: 3px; overflow: hidden; margin-top: 16px; width: 100%;">
<div style="background-color: {bar_color}; width: {percent}%; height: 100%; border-radius: 3px;"></div>
</div>
</div>
"""
    cleaned_html = "\n".join([line.strip() for line in html_content.split("\n") if line.strip()])
    return cleaned_html

def get_recommendation_card(status_now, status_pred):
    st_bbu, st_tbu, st_bbtb = status_now
    st_bbu_p, st_tbu_p, st_bbtb_p = status_pred
    
    alerts = []
    recs = []
    
    if "Pendek" in st_tbu_p or "Sangat Pendek" in st_tbu_p:
        alerts.append("⚠️ <b>Peringatan Risiko Stunting (TB/U):</b> Tinggi badan anak saat ini atau prediksi bulan depan masuk kategori Pendek/Sangat Pendek.")
        recs.append("• <b>Intervensi Protein Hewani:</b> Berikan minimal 1 butir telur, susu, ikan terasi, atau daging setiap hari.")
        recs.append("• <b>Konsultasi Medis:</b> Rujuk ke puskesmas untuk mendapatkan skrining stunting dan suplemen mikronutrien.")
        
    if "Sangat Kurang" in st_bbu_p or "Kurang" in st_bbu_p:
        alerts.append("⚠️ <b>Peringatan Berat Badan Kurang (BB/U):</b> Berat badan anak saat ini atau prediksi bulan depan kurang dari standar seusianya.")
        recs.append("• <b>MPASI / Makanan Padat Kalori:</b> Tambahkan minyak sayur, keju, mentega, atau santan pada makanan anak untuk menambah kepadatan kalori.")
        recs.append("• <b>FMT (Feeding Rules):</b> Berikan jadwal makan teratur, batasi durasi makan max 30 menit, dan hindari distrasi (gadget/mainan) saat makan.")
        
    if "Gizi Lebih" in st_bbtb_p or "Obesitas" in st_bbtb_p:
        alerts.append("⚠️ <b>Peringatan Kelebihan Gizi (BB/TB):</b> Berat badan melebihi tinggi badan idealnya.")
        recs.append("• <b>Kurangi Gula & Camilan Olahan:</b> Hindari junk food, biskuit manis, jus instan kemasan, dan makanan berminyak.")
        recs.append("• <b>Aktivitas Fisik:</b> Perbanyak jalan kaki, merangkak, bermain aktif, dan kurangi waktu menatap layar (screen time).")
        
    if not alerts:
        alerts.append("✅ <b>Status Pertumbuhan Baik:</b> Status gizi saat ini dan prediksi pertumbuhan bulan depan dalam kondisi normal & optimal.")
        recs.append("• <b>Pertahankan Pola Makan Seimbang:</b> Lanjutkan pola gizi seimbang dengan porsi karbohidrat, lauk pauk, sayur, dan buah yang seimbang.")
        recs.append("• <b>Kunjungan Posyandu Berkala:</b> Tetap lakukan pengukuran rutin setiap bulan untuk mendeteksi dini jika terjadi penyimpangan pertumbuhan.")
        
    alert_html = "<br>".join(alerts)
    rec_html = "".join([f"<div style='margin-bottom:8px;'>{r}</div>" for r in recs])
    
    card_color = "#ef4444" if "Peringatan" in alert_html else "#10b981"
    bg_color = "#fef2f2" if "Peringatan" in alert_html else "#f0fdf4"
    border_color = "#fca5a5" if "Peringatan" in alert_html else "#bbf7d0"
    
    return f"""
    <div style="
        background: {bg_color};
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.02);
        border: 1px solid {border_color};
        border-left: 6px solid {card_color};
        margin-top: 25px;
    ">
        <div style="font-size: 16px; font-weight: 700; color: #1e293b; margin-bottom: 12px; text-transform: uppercase; letter-spacing: 0.5px;">💡 Rekomendasi Gizi & Tindakan Posyandu</div>
        <div style="font-size: 14px; color: #334155; margin-bottom: 15px; line-height: 1.4;">
            {alert_html}
        </div>
        <div style="border-top: 1px solid {border_color}; padding-top: 15px; font-size: 14px; color: #475569; line-height: 1.5;">
            {rec_html}
        </div>
    </div>
    """

# ─── Plotly Visualizations ───────────────────────────────────────────────────
def plot_growth_chart_trend(tables, table_type, sex_str, history_df, curr_umur, curr_val, pred_umur_list, pred_val_list, val_name, unit, limit_months):
    table_name = _pilih_tabel(table_type, sex_str)
    fig = go.Figure()
    
    # Process history
    if history_df is not None and len(history_df) > 0:
        df_sorted = history_df.sort_values("umur")
        df_hist = df_sorted[df_sorted["umur"] <= curr_umur].copy()
        if limit_months is not None:
            df_hist = df_hist.tail(limit_months)
    else:
        df_hist = pd.DataFrame()
        
    if df_hist.empty:
        df_hist = pd.DataFrame({"umur": [curr_umur], val_name: [curr_val]})
        
    # Ensure current point is accurately represented
    if df_hist["umur"].iloc[-1] == curr_umur:
        df_hist.iloc[-1, df_hist.columns.get_loc(val_name)] = curr_val
    else:
        new_row = df_hist.iloc[-1].copy()
        new_row["umur"] = curr_umur
        new_row[val_name] = curr_val
        df_hist = pd.concat([df_hist, pd.DataFrame([new_row])], ignore_index=True)
        
    # Combine history and prediction
    x_comb_ages = list(df_hist["umur"]) + pred_umur_list
    y_comb = list(df_hist[val_name]) + pred_val_list
    
    # Custom x-axis tick labels
    ticktext = []
    for age in df_hist["umur"]:
        if age == curr_umur:
            ticktext.append(f"{int(age)} Bln (Sekarang)")
        elif age == curr_umur - 1:
            ticktext.append(f"{int(age)} Bln (Lalu)")
        else:
            ticktext.append(f"{int(age)} Bln (U-{int(curr_umur - age)})")
            
    for i, p_age in enumerate(pred_umur_list):
        if i == len(pred_umur_list) - 1:
            ticktext.append(f"{int(p_age)} Bln (+3 Bln)")
        else:
            ticktext.append(f"{int(p_age)} Bln (+{i+1} Bln)")
            
    # Calculate WHO Median values corresponding to the exact ages in x_comb_ages
    ys_ref = []
    for age in x_comb_ages:
        lms = get_lms(tables, table_name, age)
        if lms is not None:
            ys_ref.append(lms[1]) # LMS M (Median) value
        else:
            ys_ref.append(None)
            
    # 1. Background WHO Median reference curve (dotted gray)
    fig.add_trace(go.Scatter(
        x=ticktext, y=ys_ref,
        mode='lines',
        line=dict(color='#cbd5e1', width=1.5, dash='dot'),
        name="Rujukan WHO (Median)",
        hoverinfo='skip'
    ))
    
    # 2. Main child curve (spline with filled area)
    line_color = '#2e7d32' if val_name == 'BB' else '#7c3aed'
    fill_color = 'rgba(46, 125, 80, 0.08)' if val_name == 'BB' else 'rgba(124, 92, 237, 0.08)'
    
    fig.add_trace(go.Scatter(
        x=ticktext, y=y_comb,
        mode='lines+markers',
        line=dict(color=line_color, width=3.5, shape='spline'),
        fill='tozeroy',
        fillcolor=fill_color,
        marker=dict(
            size=[8]*(len(y_comb) - 1) + [10],
            color=[line_color]*(len(y_comb) - 1) + ['#ef4444'], # Make prediction point stand out in red
            symbol=['circle']*(len(y_comb) - 1) + ['star']
        ),
        name="Tren Pertumbuhan"
    ))
    
    # 3. Add text annotations for "Hari Ini" and "Prediksi (+3 Bln)" using category string coordinates
    fig.add_annotation(
        x=f"{int(curr_umur)} Bln (Sekarang)",
        y=curr_val,
        text="<b>Hari Ini</b>",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-35,
        font=dict(size=11, color="#1e293b"),
        arrowcolor="#64748b"
    )
    
    fig.add_annotation(
        x=f"{int(pred_umur_list[-1])} Bln (+3 Bln)",
        y=pred_val_list[-1],
        text="<b>Prediksi (+3 Bln)</b>",
        showarrow=True,
        arrowhead=2,
        ax=0,
        ay=-35,
        font=dict(size=11, color=line_color),
        arrowcolor="#64748b"
    )
    
    title_lbl = f"Tren Pertumbuhan Berat Badan ({unit})" if val_name == "BB" else f"Tren Pertumbuhan Tinggi Badan ({unit})"
    
    fig.update_layout(
        title=dict(text=f"<b>{title_lbl}</b>", font=dict(size=14, color="#1e293b")),
        xaxis=dict(
            type='category',
            gridcolor="#e2e8f0",
            showgrid=True,
            tickfont=dict(color="#1e293b", size=9),
            tickangle=-45
        ),
        yaxis=dict(
            title=dict(
                text=f"{'Berat' if val_name == 'BB' else 'Tinggi'} ({unit})",
                font=dict(color="#1e293b", size=11)
            ),
            gridcolor="#e2e8f0",
            showgrid=True,
            tickfont=dict(color="#1e293b", size=10)
        ),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=65),
        height=380,
        showlegend=False,
        hovermode="x unified",
        font=dict(color="#1e293b")
    )
    
    return fig

def plot_growth_chart_kms(tables, table_type, sex_str, history_df, curr_umur, curr_val, pred_umur_list, pred_val_list, val_name, unit):
    table_name = _pilih_tabel(table_type, sex_str)
    fig = go.Figure()
    
    # Process history
    if history_df is not None and len(history_df) > 0:
        df_sorted = history_df.sort_values("umur")
        df_hist = df_sorted[df_sorted["umur"] <= curr_umur].copy()
    else:
        df_hist = pd.DataFrame()
        
    if df_hist.empty:
        df_hist = pd.DataFrame({"umur": [curr_umur], val_name: [curr_val]})
        
    # Combine history and prediction
    x_comb = list(df_hist["umur"]) + pred_umur_list
    y_comb = list(df_hist[val_name]) + pred_val_list
    
    # Define zoom bounds for age (X-axis)
    min_age = min(x_comb)
    max_age = max(x_comb)
    start_age = max(0, int(min_age) - 2)
    end_age = min(60, int(max_age) + 2)
    
    ages_kms = list(range(start_age, end_age + 1))
    
    # Plot the 7 Z-Score curves
    z_scores_config = [
        (-3, "-3 SD", "#800000", "dash"),
        (-2, "-2 SD", "#dc2626", "dash"),
        (-1, "-1 SD", "#eab308", "dash"),
        (0, "Median", "#22c55e", "solid"),
        (1, "+1 SD", "#eab308", "dash"),
        (2, "+2 SD", "#dc2626", "dash"),
        (3, "+3 SD", "#800000", "dash")
    ]
    
    for z_val, label, color, line_dash in z_scores_config:
        xs, ys = get_z_curve(tables, table_name, z_val, ages_kms)
        fig.add_trace(go.Scatter(
            x=xs, y=ys,
            mode='lines',
            name=label,
            line=dict(color=color, width=1.5, dash=line_dash),
            hoverinfo='skip'
        ))
        
    # Plot "Anak Anda" curve (history + predictions) in blue
    fig.add_trace(go.Scatter(
        x=x_comb, y=y_comb,
        mode='lines+markers',
        name="Anak Anda",
        line=dict(color="#2563eb", width=3),
        marker=dict(size=8, color="#2563eb", symbol="circle")
    ))
    
    title_lbl = f"Kurva Z-Score WHO: Berat Badan ({unit})" if val_name == "BB" else f"Kurva Z-Score WHO: Tinggi Badan ({unit})"
    
    fig.update_layout(
        title=dict(text=f"<b>{title_lbl}</b>", font=dict(size=14, color="#1e293b")),
        xaxis=dict(
            title=dict(
                text="Umur (bulan)",
                font=dict(color="#1e293b", size=11)
            ),
            range=[start_age, end_age],
            dtick=1,
            gridcolor="#e2e8f0",
            showgrid=True,
            tickfont=dict(color="#1e293b", size=9)
        ),
        yaxis=dict(
            title=dict(
                text=f"{'Berat' if val_name == 'BB' else 'Tinggi'} ({unit})",
                font=dict(color="#1e293b", size=11)
            ),
            gridcolor="#e2e8f0",
            showgrid=True,
            tickfont=dict(color="#1e293b", size=10)
        ),
        plot_bgcolor="white",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=40, r=40, t=50, b=80), # extra bottom margin for horizontal legend
        height=380,
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.22,
            xanchor="center",
            x=0.5,
            font=dict(size=10, color="#1e293b")
        ),
        hovermode="x unified",
        font=dict(color="#1e293b")
    )
    
    return fig

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="header-card">
    <h1>👶 Dashboard Prediksi Tumbuh Kembang Anak</h1>
    <p>Prediksi tumbuh kembang anak posyandu satu bulan ke depan dengan model kecerdasan buatan <b>XGBoost</b> dan klasifikasi standar <b>WHO</b></p>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — Upload file
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CSV = "trial_2025.csv"
DEMO_CSV = "demo_data.csv"
DEFAULT_WHO = "tabel_who.xlsx"

with st.sidebar:
    st.header("⚙️ Konfigurasi Data")

    st.subheader("1. Data Riwayat Anak (CSV)")
    uploaded_csv = st.file_uploader("Upload File CSV Baru", type=["csv"])
    if uploaded_csv is None:
        if os.path.exists(DEFAULT_CSV):
            st.caption(f"📁 Menggunakan data lokal: `{DEFAULT_CSV}`")
            df_source = DEFAULT_CSV
        elif os.path.exists(DEMO_CSV):
            st.caption(f"📁 Menggunakan data demo publik: `{DEMO_CSV}`")
            df_source = DEMO_CSV
        else:
            df_source = None
    else:
        df_source = uploaded_csv

    st.subheader("2. Tabel Referensi WHO (Excel)")
    uploaded_who = st.file_uploader("Upload File Excel Baru", type=["xltm", "xlsx", "xlsm"])
    if uploaded_who is None:
        if os.path.exists(DEFAULT_WHO):
            st.caption(f"🩺 Menggunakan tabel WHO lokal: `{DEFAULT_WHO}`")
            who_source = DEFAULT_WHO
        else:
            who_source = None
    else:
        who_source = uploaded_who

    st.divider()
    st.caption("Format CSV wajib memiliki kolom: `id`, `tanggal_timbang`, `umur`, `sex`, `BB`, `TB`")

# ── Validasi file ──────────────────────────────────────────────────────────────
if not df_source:
    st.info("👈 Silakan upload **data CSV** di sidebar untuk memulai.")
    st.stop()

if not who_source:
    st.info("👈 Silakan upload **tabel WHO** di sidebar atau letakkan `tabel_who.xlsx` di direktori aplikasi.")
    st.stop()

# ── Load data ──────────────────────────────────────────────────────────────────
with st.spinner("Memuat data & melatih model..."):
    # Read CSV
    if isinstance(df_source, str):
        df_raw = pd.read_csv(df_source)
    else:
        df_raw = pd.read_csv(df_source)

    # Read WHO
    if isinstance(who_source, str):
        who_path = who_source
        who_tables = load_who_tables(who_path)
    else:
        temp_dir = os.path.join(os.getcwd(), "temp")
        os.makedirs(temp_dir, exist_ok=True)
        who_path = os.path.join(temp_dir, "tabel_who_temp.xlsx")
        with open(who_path, "wb") as f:
            f.write(who_source.read())
        who_tables = load_who_tables(who_path)

    try:
        model_bb, model_tb, df_clean = train_models(df_raw)
        n_anak = len(df_clean["id"].unique())
        st.sidebar.success(f"✅ Model XGBoost terlatih ({n_anak} anak)")
    except Exception as e:
        st.error(f"Gagal melatih model: {e}")
        st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# INPUT DATA ANAK (Lookup DB vs Manual)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📋 Input Data Anak</div>', unsafe_allow_html=True)

mode_input = st.radio("Pilih Mode Penginputan Data", ["🔎 Cari dari Riwayat Posyandu (ID)", "✍️ Input Manual (Data Baru)"], horizontal=True)

# default values to collect
umur = 24
sex = 1
bb = 12.0
tb = 85.0
bb_prev = None
tb_prev = None
history_df = None

if "Cari dari Riwayat" in mode_input:
    col_sel1, col_sel2 = st.columns([1, 2])
    with col_sel1:
        child_ids = sorted(df_clean["id"].unique())
        selected_id = st.selectbox("Pilih ID Anak", options=child_ids)
    
    # Extract records for selected child
    df_child = df_clean[df_clean["id"] == selected_id].sort_values("umur")
    latest_record = df_child.iloc[-1]
    
    if len(df_child) > 1:
        prev_record = df_child.iloc[-2]
        bb_prev_val = float(prev_record["BB"])
        tb_prev_val = float(prev_record["TB"])
        has_prev = True
    else:
        bb_prev_val = float(latest_record["BB"])
        tb_prev_val = float(latest_record["TB"])
        has_prev = False
        
    sex_lbl = "Laki-laki" if latest_record["sex"] == 1 else "Perempuan"
    
    with col_sel2:
        st.markdown(f"""
        <div style="background-color: #f1f5f9; padding: 15px; border-radius: 12px; border: 1px solid #e2e8f0; margin-top: 10px; color: #1e293b;">
            <span style="font-weight: 700; color: #1e293b; font-size: 14px;">PROFIL ANAK TERPILIH (ID: {selected_id})</span><br>
            • Jenis Kelamin: <b>{sex_lbl}</b> | • Umur Terakhir: <b>{int(latest_record['umur'])} bulan</b><br>
            • BB Terakhir: <b>{latest_record['BB']:.2f} kg</b> | • TB Terakhir: <b>{latest_record['TB']:.2f} cm</b>
        </div>
        """, unsafe_allow_html=True)
        
    umur = int(latest_record["umur"])
    sex = int(latest_record["sex"])
    bb = float(latest_record["BB"])
    tb = float(latest_record["TB"])
    bb_prev = bb_prev_val if has_prev else None
    tb_prev = tb_prev_val if has_prev else None
    history_df = df_child
    
    # Optional override of latest measurements
    with st.expander("⚙️ Sesuaikan/Update Pengukuran Terakhir (Opsional)"):
        col_ovr1, col_ovr2 = st.columns(2)
        with col_ovr1:
            umur = st.number_input("Umur Terakhir (bulan)", min_value=0, max_value=60, value=umur, key="lk_umur")
            bb = st.number_input("Berat Badan / BB Terakhir (kg)", min_value=1.0, max_value=40.0, value=bb, key="lk_bb")
        with col_ovr2:
            tb = st.number_input("Tinggi Badan / TB Terakhir (cm)", min_value=40.0, max_value=130.0, value=tb, key="lk_tb")
            if has_prev:
                bb_prev = st.number_input("BB Bulan Sebelumnya (kg)", min_value=1.0, max_value=40.0, value=bb_prev, key="lk_bb_prev")
                tb_prev = st.number_input("TB Bulan Sebelumnya (cm)", min_value=40.0, max_value=130.0, value=tb_prev, key="lk_tb_prev")

else:
    col_man1, col_man2 = st.columns(2)
    with col_man1:
        umur = st.number_input("Umur Anak (bulan)", min_value=0, max_value=60, value=24, step=1)
        sex_label = st.selectbox("Jenis Kelamin", ["Laki-laki", "Perempuan"])
        sex = 1 if sex_label == "Laki-laki" else 0
        
        punya_riwayat = st.checkbox("Ada data bulan sebelumnya?", value=False)
        if punya_riwayat:
            bb_prev = st.number_input("BB Bulan Sebelumnya (kg)", min_value=1.0, max_value=40.0, value=11.5, step=0.1)
            tb_prev = st.number_input("TB Bulan Sebelumnya (cm)", min_value=40.0, max_value=130.0, value=84.0, step=0.1)
            
    with col_man2:
        bb = st.number_input("Berat Badan Sekarang / BB (kg)", min_value=1.0, max_value=40.0, value=12.0, step=0.1)
        tb = st.number_input("Tinggi Badan Sekarang / TB (cm)", min_value=40.0, max_value=130.0, value=85.0, step=0.1)

sex_str = "L" if sex == 1 else "P"

# Persist analysis state and reset on profile input changes
if "analyze_triggered" not in st.session_state:
    st.session_state.analyze_triggered = False

current_input_sig = (
    mode_input,
    selected_id if "Cari dari Riwayat" in mode_input else None,
    umur,
    sex,
    bb,
    tb,
    bb_prev,
    tb_prev
)

if "last_input_sig" not in st.session_state:
    st.session_state.last_input_sig = current_input_sig

if st.session_state.last_input_sig != current_input_sig:
    st.session_state.analyze_triggered = False
    st.session_state.last_input_sig = current_input_sig

st.divider()
predict_btn = st.button("🔮 ANALISIS & HITUNG PREDIKSI", type="primary", use_container_width=True)

if predict_btn:
    st.session_state.analyze_triggered = True

if not st.session_state.analyze_triggered:
    st.info("💡 Klik tombol **ANALISIS & HITUNG PREDIKSI** di atas untuk melihat status gizi dan prediksi kurva pertumbuhan.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════════
# PREDIKSI & Z-SCORE
# ══════════════════════════════════════════════════════════════════════════════
name_str = f"Arka" if "Cari dari Riwayat" in mode_input and selected_id == 11 else (f"Anak ID {selected_id}" if "Cari dari Riwayat" in mode_input else "Anak Baru")

# Multi-step recursive forecasting for +3 months
bb_pred_list = []
tb_pred_list = []
umur_pred_list = []

# Bulan +1
sample_1 = prepare_input(umur, sex, bb, tb, bb_prev, tb_prev)
bb_p1 = float(model_bb.predict(sample_1)[0])
tb_p1 = float(model_tb.predict(sample_1)[0])
bb_pred_list.append(bb_p1)
tb_pred_list.append(tb_p1)
umur_pred_list.append(umur + 1)

# Bulan +2
sample_2 = prepare_input(umur + 1, sex, bb_p1, tb_p1, bb, tb)
bb_p2 = float(model_bb.predict(sample_2)[0])
tb_p2 = float(model_tb.predict(sample_2)[0])
bb_pred_list.append(bb_p2)
tb_pred_list.append(tb_p2)
umur_pred_list.append(umur + 2)

# Bulan +3
sample_3 = prepare_input(umur + 2, sex, bb_p2, tb_p2, bb_p1, tb_p1)
bb_p3 = float(model_bb.predict(sample_3)[0])
tb_p3 = float(model_tb.predict(sample_3)[0])
bb_pred_list.append(bb_p3)
tb_pred_list.append(tb_p3)
umur_pred_list.append(umur + 3)

# For compatibility and single-month references
bb_pred = bb_p1
tb_pred = tb_p1
umur_pred = umur + 1

# Z-score saat ini
z_bbu_s  = zscore_BB_U(who_tables, bb, umur, sex_str)
z_tbu_s  = zscore_TB_U(who_tables, tb, umur, sex_str)
z_bbtb_s = zscore_BB_TB(who_tables, bb, tb, umur, sex_str)

# Z-score prediksi +1 bulan (for tables)
z_bbu_p  = zscore_BB_U(who_tables, bb_pred, umur_pred, sex_str)
z_tbu_p  = zscore_TB_U(who_tables, tb_pred, umur_pred, sex_str)
z_bbtb_p = zscore_BB_TB(who_tables, bb_pred, tb_pred, umur_pred, sex_str)

# Z-score prediksi +3 bulan (for metric cards)
z_bbu_p3  = zscore_BB_U(who_tables, bb_p3, umur + 3, sex_str)
z_tbu_p3  = zscore_TB_U(who_tables, tb_p3, umur + 3, sex_str)

# Status gizi saat ini
st_bbu_s,  cl_bbu_s  = status_BB_U(z_bbu_s)
st_tbu_s,  cl_tbu_s  = status_TB_U(z_tbu_s)
st_bbtb_s, cl_bbtb_s = status_BB_TB(z_bbtb_s)

# Status gizi prediksi +1 bulan
st_bbu_p,  cl_bbu_p  = status_BB_U(z_bbu_p)
st_tbu_p,  cl_tbu_p  = status_TB_U(z_tbu_p)
st_bbtb_p, cl_bbtb_p = status_BB_TB(z_bbtb_p)

# Status gizi prediksi +3 bulan
st_bbu_p3,  cl_bbu_p3  = status_BB_U(z_bbu_p3)
st_tbu_p3,  cl_tbu_p3  = status_TB_U(z_tbu_p3)
st_bbtb_p3, cl_bbtb_p3 = status_BB_TB(zscore_BB_TB(who_tables, bb_p3, tb_p3, umur + 3, sex_str))

# Dynamically determine the badge for outlook
if cl_bbu_p3 == "danger" or cl_tbu_p3 == "danger" or cl_bbtb_p3 == "danger":
    badge_style = "danger"
    badge_text = "RISIKO TINGGI"
elif cl_bbu_p3 == "warning" or cl_tbu_p3 == "warning" or cl_bbtb_p3 == "warning":
    badge_style = "warning"
    badge_text = "PERINGATAN"
else:
    badge_style = "normal"
    badge_text = "OPTIMAL"
badge_outlook = badge(badge_text, badge_style)

# ══════════════════════════════════════════════════════════════════════════════
# HASIL — Metric cards
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">📊 Hasil Pengukuran & Prediksi Target +3 Bulan</div>', unsafe_allow_html=True)

mc1, mc2 = st.columns(2)
with mc1:
    delta_bb = bb_p3 - bb
    c_bb = "green" if delta_bb > 0 else ("red" if delta_bb < 0 else "neutral")
    delta_sign = "+" if delta_bb > 0 else ""
    st.markdown(make_metric_card(
        title="Prediksi Berat Badan",
        value=f"{bb_p3:.1f}",
        unit="Kg",
        delta=f"{delta_sign}{delta_bb:.1f} Kg",
        delta_color=c_bb,
        icon_type="weight",
        z_score=z_bbu_p3
    ), unsafe_allow_html=True)

with mc2:
    delta_tb = tb_p3 - tb
    c_tb = "green" if delta_tb > 0 else ("red" if delta_tb < 0 else "neutral")
    delta_sign = "+" if delta_tb > 0 else ""
    st.markdown(make_metric_card(
        title="Prediksi Tinggi Badan",
        value=f"{tb_p3:.1f}",
        unit="Cm",
        delta=f"{delta_sign}{delta_tb:.1f} Cm",
        delta_color=c_tb,
        icon_type="height",
        z_score=z_tbu_p3
    ), unsafe_allow_html=True)

# ── Visualisasi Kurva Pertumbuhan ──────────────────────────────────────────────
st.markdown('<div class="section-header">📈 Kurva Pertumbuhan & Outlook Pertumbuhan Anak</div>', unsafe_allow_html=True)

# Add horizontal period filter at the top of the chart section
time_range = st.radio(
    "Filter Periode Riwayat Tren:",
    options=["3 Bulan", "6 Bulan", "12 Bulan", "Semua"],
    index=1, # Default to 6 Bulan as in user screenshot
    horizontal=True
)

limit_map = {
    "3 Bulan": 3,
    "6 Bulan": 6,
    "12 Bulan": 12,
    "Semua": None
}
limit_val = limit_map[time_range]

chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    tab_trend, tab_kms = st.tabs(["📈 Tren Pertumbuhan", "🩺 Kurva Z-Score (KMS)"])
    with tab_trend:
        # Display header like in Image 4
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 10px;">
            <div>
                <h4 style="margin: 0; font-size: 16px; font-weight: 700; color: #0f172a;">Outlook Pertumbuhan 3 Bulan</h4>
                <div style="margin: 2px 0 0 0; font-size: 12px; color: #64748b;">Estimasi berdasarkan tren data {time_range.lower()} terakhir {name_str}.</div>
            </div>
            <div>
                {badge_outlook}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig_bb = plot_growth_chart_trend(
            who_tables, "BBU", sex_str, history_df, 
            umur, bb, umur_pred_list, bb_pred_list, "BB", "kg", limit_val
        )
        st.plotly_chart(fig_bb, use_container_width=True)
        
    with tab_kms:
        fig_bb_kms = plot_growth_chart_kms(
            who_tables, "BBU", sex_str, history_df,
            umur, bb, umur_pred_list, bb_pred_list, "BB", "kg"
        )
        st.plotly_chart(fig_bb_kms, use_container_width=True)

with chart_col2:
    tab_trend, tab_kms = st.tabs(["📈 Tren Pertumbuhan", "🩺 Kurva Z-Score (KMS)"])
    with tab_trend:
        # Display header like in Image 4
        st.markdown(f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 10px; margin-bottom: 10px;">
            <div>
                <h4 style="margin: 0; font-size: 16px; font-weight: 700; color: #0f172a;">Outlook Pertumbuhan 3 Bulan</h4>
                <div style="margin: 2px 0 0 0; font-size: 12px; color: #64748b;">Estimasi berdasarkan tren data {time_range.lower()} terakhir {name_str}.</div>
            </div>
            <div>
                {badge_outlook}
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        fig_tb = plot_growth_chart_trend(
            who_tables, "PBU", sex_str, history_df,
            umur, tb, umur_pred_list, tb_pred_list, "TB", "cm", limit_val
        )
        st.plotly_chart(fig_tb, use_container_width=True)
        
    with tab_kms:
        fig_tb_kms = plot_growth_chart_kms(
            who_tables, "PBU", sex_str, history_df,
            umur, tb, umur_pred_list, tb_pred_list, "TB", "cm"
        )
        st.plotly_chart(fig_tb_kms, use_container_width=True)

# ── Tabel Z-score & Status Gizi ───────────────────────────────────────────────
st.markdown('<div class="section-header">🩺 Status Gizi (WHO Z-Score)</div>', unsafe_allow_html=True)

def z_fmt(z): return f"{z:+.2f}" if z is not None else "N/A"

table_data = {
    "Indikator": ["BB/U (Berat Badan / Umur)", "TB/U (Tinggi Badan / Umur)", "BB/TB (Berat / Tinggi)"],
    "Z-Score Sekarang": [z_fmt(z_bbu_s), z_fmt(z_tbu_s), z_fmt(z_bbtb_s)],
    "Status Sekarang":  [st_bbu_s, st_tbu_s, st_bbtb_s],
    "Z-Score +1 Bulan": [z_fmt(z_bbu_p), z_fmt(z_tbu_p), z_fmt(z_bbtb_p)],
    "Status +1 Bulan":  [st_bbu_p, st_tbu_p, st_bbtb_p],
}

df_tabel = pd.DataFrame(table_data)

# Render HTML tabel dengan badge warna
rows_html = ""
status_cols_now = [st_bbu_s, st_tbu_s, st_bbtb_s]
status_cols_pred = [st_bbu_p, st_tbu_p, st_bbtb_p]
class_now  = [cl_bbu_s, cl_tbu_s, cl_bbtb_s]
class_pred = [cl_bbu_p, cl_tbu_p, cl_bbtb_p]

for i in range(3):
    rows_html += f"""
    <tr>
        <td style="font-weight:600;padding:14px;text-align:left;">{df_tabel['Indikator'][i]}</td>
        <td style="padding:14px;">{df_tabel['Z-Score Sekarang'][i]}</td>
        <td style="padding:14px;">{badge(status_cols_now[i], class_now[i])}</td>
        <td style="padding:14px;">{df_tabel['Z-Score +1 Bulan'][i]}</td>
        <td style="padding:14px;">{badge(status_cols_pred[i], class_pred[i])}</td>
    </tr>"""

html_table = f"""
<table class="modern-table">
  <thead>
    <tr>
      <th style="text-align:left;">Indikator Pertumbuhan</th>
      <th>Z-Score Sekarang</th>
      <th>Status Sekarang</th>
      <th>Z-Score +1 Bulan</th>
      <th>Status +1 Bulan</th>
    </tr>
  </thead>
  <tbody>{rows_html}</tbody>
</table>"""

st.markdown(html_table, unsafe_allow_html=True)

# ── Panel Rekomendasi Gizi ────────────────────────────────────────────────────
rec_card = get_recommendation_card((st_bbu_s, st_tbu_s, st_bbtb_s), (st_bbu_p, st_tbu_p, st_bbtb_p))
st.markdown(rec_card, unsafe_allow_html=True)

# ── Keterangan Status ─────────────────────────────────────────────────────────
with st.expander("ℹ️ Keterangan Standar Z-Score Gizi WHO"):
    st.markdown("""
    | Indikator | Status Gizi | Rentang Z-Score |
    |-----------|-------------|-----------------|
    | **BB/U** | BB Sangat Kurang | < -3 SD |
    | | BB Kurang | -3 SD s.d. < -2 SD |
    | | BB Normal | -2 SD s.d. +1 SD |
    | | Risiko BB Lebih | > +1 SD |
    | **TB/U** | Sangat Pendek | < -3 SD |
    | | Pendek (Stunting) | -3 SD s.d. < -2 SD |
    | | Normal | -2 SD s.d. +3 SD |
    | | Tinggi | > +3 SD |
    | **BB/TB** | Gizi Buruk | < -3 SD |
    | | Gizi Kurang | -3 SD s.d. < -2 SD |
    | | Gizi Baik (Normal) | -2 SD s.d. +1 SD |
    | | Risiko Gizi Lebih | +1 SD s.d. +2 SD |
    | | Gizi Lebih | +2 SD s.d. +3 SD |
    | | Obesitas | > +3 SD |
    """)

st.divider()
st.caption("Model: XGBoost Regressor | Standar Referensi: WHO Child Growth Standards | Database lokal Posyandu")