import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Stroke Risk Dashboard",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    color: #E2E8F0;
}
.main { background-color: #080C14; }
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1422 0%, #0A1020 100%);
    border-right: 1px solid #1A2540;
}
#MainMenu, footer, header { visibility: hidden; }

/* ── Upload zone ── */
.upload-zone {
    background: linear-gradient(135deg, #0D1830 0%, #111E38 100%);
    border: 2px dashed #2A3F6F;
    border-radius: 20px;
    padding: 52px 40px;
    text-align: center;
    margin: 40px auto;
    max-width: 680px;
    transition: border-color 0.3s, background 0.3s;
}
.upload-zone:hover { border-color: #F87171; background: #111E38; }
.upload-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    background: linear-gradient(135deg, #F87171, #FB923C, #FBBF24);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 8px;
}
.upload-sub { color: #64748B; font-size: 0.92rem; margin-bottom: 24px; }

/* ── Page header ── */
.dash-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.4rem;
    background: linear-gradient(135deg, #F87171 0%, #FB923C 50%, #FBBF24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 2px;
}
.dash-sub {
    color: #475569;
    font-size: 0.82rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    font-weight: 500;
}

/* ── KPI cards ── */
.kpi-wrap {
    background: linear-gradient(145deg, #0D1830 0%, #111E38 100%);
    border: 1px solid #1A2D50;
    border-radius: 16px;
    padding: 20px 16px 16px;
    text-align: center;
    position: relative;
    overflow: hidden;
    transition: transform 0.2s, border-color 0.2s;
}
.kpi-wrap::before {
    content: '';
    position: absolute; top: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #F87171, #FBBF24);
    border-radius: 16px 16px 0 0;
}
.kpi-wrap:hover { transform: translateY(-2px); border-color: #F87171; }
.kpi-val  { font-family: 'Playfair Display', serif; font-size: 2rem; color: #F87171; line-height: 1; }
.kpi-lbl  { color: #64748B; font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.1em; margin-top: 6px; }
.kpi-note { color: #6EE7B7; font-size: 0.68rem; margin-top: 5px; }

/* ── Section headers ── */
.sec-hdr {
    font-family: 'Playfair Display', serif;
    font-size: 1.2rem; color: #E2E8F0;
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 12px;
}
.sec-hdr::before {
    content: '';
    display: inline-block;
    width: 4px; height: 22px;
    background: linear-gradient(180deg, #F87171, #FBBF24);
    border-radius: 2px;
    flex-shrink: 0;
}

/* ── Insight box ── */
.ibox {
    background: #0D1830;
    border: 1px solid #1A2D50;
    border-left: 4px solid #F87171;
    border-radius: 10px;
    padding: 14px 18px;
    margin: 12px 0;
    font-size: 0.86rem;
    color: #94A3B8;
    line-height: 1.7;
}

/* ── Divider ── */
.hr { border-top: 1px solid #1A2540; margin: 20px 0; }

/* ── Badge ── */
.badge {
    display: inline-block;
    background: #1A2D50;
    border: 1px solid #2A3F6F;
    border-radius: 20px;
    padding: 2px 12px;
    font-size: 0.75rem;
    color: #94A3B8;
    margin: 2px;
}
.badge-hot { border-color: #F87171; color: #F87171; }

/* ── Streamlit file uploader tweak ── */
[data-testid="stFileUploader"] {
    background: transparent !important;
}
[data-testid="stFileUploadDropzone"] {
    background: #0D1830 !important;
    border: 1.5px dashed #2A3F6F !important;
    border-radius: 12px !important;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ALTAIR DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
STROKE_COLOR    = "#F87171"
NO_STROKE_COLOR = "#60A5FA"
GOLD            = "#FBBF24"
TEAL            = "#34D399"
PURPLE          = "#A78BFA"

def dark_theme():
    return {
        "config": {
            "background": "#0D1422",
            "view": {"stroke": "transparent", "fill": "#0D1422"},
            "axis": {
                "domainColor": "#1E2D4A",
                "gridColor":   "#131F35",
                "labelColor":  "#64748B",
                "titleColor":  "#64748B",
                "tickColor":   "#1E2D4A",
                "labelFontSize": 11,
                "titleFontSize": 12,
            },
            "legend": {
                "labelColor": "#94A3B8",
                "titleColor": "#94A3B8",
                "fillColor":  "#0D1422",
                "strokeColor": "#1E2D4A",
            },
            "title": {"color": "#E2E8F0", "fontSize": 13, "fontWeight": 500, "anchor": "start"},
            "mark":  {"tooltip": True},
        }
    }

alt.themes.register("dark", dark_theme)
alt.themes.enable("dark")

STROKE_SCALE = alt.Scale(domain=["No Stroke","Stroke"], range=[NO_STROKE_COLOR, STROKE_COLOR])

# ─────────────────────────────────────────────────────────────────────────────
# SMOOTH CURVE HELPER  (rolling-window mean → dense Altair line)
# ─────────────────────────────────────────────────────────────────────────────
def smooth_line(df, x_col, y_col, window=5):
    """Return a DataFrame with a rolling-smoothed y column, sorted by x."""
    tmp = (df.groupby(x_col)[y_col]
             .mean()
             .rename(y_col)          # keep series name explicit before reset_index
             .reset_index(name=y_col)
             .sort_values(x_col))
    tmp[y_col] = tmp[y_col].rolling(window=window, center=True, min_periods=1).mean()
    return tmp

# ─────────────────────────────────────────────────────────────────────────────
# DATA CLEANING FUNCTION
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def clean_data(raw_bytes):
    import io
    df = pd.read_csv(io.BytesIO(raw_bytes))

    df["gender"]         = df["gender"].astype(str).str.strip()
    df["smoking_status"] = df["smoking_status"].astype(str).str.strip()
    df["work_type"]      = df["work_type"].astype(str).str.strip()

    df["gender"] = df["gender"].replace({"0":"Female","1":"Male",0:"Female",1:"Male"})
    df["hypertension"]  = df["hypertension"].replace({0:"No",  1:"Yes"})
    df["heart_disease"] = df["heart_disease"].replace({0:"No", 1:"Yes"})
    df["stroke"]        = df["stroke"].replace({0:"No Stroke", 1:"Stroke"})

    df["smoking_status"] = df["smoking_status"].replace({
        "never smoked":"Never Smoked", "formerly smoked":"Former Smoker",
        "smokes":"Smoker", "Unknown":"Unknown", "0":"Unknown", 0:"Unknown"
    })
    df["work_type"] = df["work_type"].replace({
        "Private":"Private Sector", "Self-employed":"Self Employed",
        "Govt_job":"Government Job", "children":"Children", "Never_worked":"Never Worked"
    })

    df["bmi"] = df["bmi"].fillna(df["bmi"].mean())

    df["high_glucose"] = (df["avg_glucose_level"] > 140).map(
        {True:"High (>140)", False:"Normal (≤140)"})
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0,18.5,25,30,200],
        labels=["Underweight","Normal","Overweight","Obese"])
    df["age_group"] = pd.cut(
        df["age"], bins=[0,30,45,60,75,200],
        labels=["0–30","30–45","45–60","60–75","75+"])
    df["cardio_risk"] = (
        df["hypertension"].map({"Yes":1,"No":0}) +
        df["heart_disease"].map({"Yes":1,"No":0})
    )
    return df

# ─────────────────────────────────────────────────────────────────────────────
# ▸ UPLOAD SCREEN  — shown when no file is loaded yet
# ─────────────────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    # Centered upload card
    _, mid, _ = st.columns([1, 2.4, 1])
    with mid:
        st.markdown("""
        <div class='upload-zone'>
            <div class='upload-title'>🧠 Stroke Risk Dashboard</div>
            <div class='upload-sub'>Drop your dataset to unlock all charts & insights</div>
        </div>
        """, unsafe_allow_html=True)

        uploaded = st.file_uploader(
            "Upload CSV dataset",
            type=["csv"],
            label_visibility="collapsed",
            help="Expected columns: age, gender, hypertension, heart_disease, "
                 "avg_glucose_level, bmi, smoking_status, work_type, stroke"
        )

        if uploaded is not None:
            with st.spinner("Cleaning & engineering features…"):
                st.session_state["df"] = clean_data(uploaded.read())
            st.rerun()

        st.markdown("""
        <div style='margin-top:18px; color:#334155; font-size:0.8rem; line-height:1.8'>
            Expected columns: <code>age</code> · <code>gender</code> · <code>hypertension</code>
            · <code>heart_disease</code> · <code>avg_glucose_level</code> · <code>bmi</code>
            · <code>smoking_status</code> · <code>work_type</code> · <code>stroke</code>
        </div>
        """, unsafe_allow_html=True)
    st.stop()

# ─────────────────────────────────────────────────────────────────────────────
# DATA IS LOADED — rest of app runs from here
# ─────────────────────────────────────────────────────────────────────────────
df = st.session_state["df"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)

    # Re-upload button
    if st.button("↩ Upload new file", use_container_width=True):
        del st.session_state["df"]
        st.rerun()

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 🔍 Filters")

    gender        = st.selectbox("Gender",    ["All"] + sorted(df["gender"].dropna().unique().tolist()))
    smoking       = st.selectbox("Smoking",   ["All"] + sorted(df["smoking_status"].dropna().unique().tolist()))
    work          = st.selectbox("Work Type", ["All"] + sorted(df["work_type"].dropna().unique().tolist()))
    stroke_filter = st.selectbox("Outcome",   ["All","Stroke","No Stroke"])

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

    age_range = st.slider("Age Range",
                           int(df.age.min()), int(df.age.max()), (20, 80))
    bmi_range = st.slider("BMI Range",
                           int(df.bmi.min()), int(df.bmi.max()), (15, 40))
    glu_range = st.slider("Glucose Range",
                           int(df.avg_glucose_level.min()),
                           int(df.avg_glucose_level.max()), (55, 272))

    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.caption(f"Dataset: {len(df):,} rows · {df.shape[1]} columns")

# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────
fdf = df.copy()
if gender        != "All": fdf = fdf[fdf["gender"]         == gender]
if smoking       != "All": fdf = fdf[fdf["smoking_status"] == smoking]
if work          != "All": fdf = fdf[fdf["work_type"]       == work]
if stroke_filter != "All": fdf = fdf[fdf["stroke"]          == stroke_filter]
fdf = fdf[
    fdf["age"].between(*age_range) &
    fdf["bmi"].between(*bmi_range) &
    fdf["avg_glucose_level"].between(*glu_range)
]

total        = len(fdf)
stroke_count = (fdf["stroke"] == "Stroke").sum()
stroke_rate  = round(stroke_count / total * 100, 2) if total > 0 else 0

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
lc, rc = st.columns([3, 1])
with lc:
    st.markdown("<div class='dash-title'>🧠 Stroke Risk Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-sub'>Exploratory &amp; Predictive Analysis</div>", unsafe_allow_html=True)
with rc:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:right;color:#334155;font-size:0.82rem'>"
        f"Showing <b style='color:#F87171'>{total:,}</b> of <b style='color:#60A5FA'>{len(df):,}</b> patients</div>",
        unsafe_allow_html=True
    )

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
hyp_pct = round((fdf["hypertension"]=="Yes").mean()*100,1) if total>0 else 0
hd_pct  = round((fdf["heart_disease"]=="Yes").mean()*100,1) if total>0 else 0

k1,k2,k3,k4,k5,k6,k7 = st.columns(7)
kpi_items = [
    (k1, f"{total:,}",                             "Patients",         ""),
    (k2, f"{fdf['age'].mean():.1f}",               "Avg Age",          "years"),
    (k3, f"{fdf['bmi'].mean():.1f}",               "Avg BMI",          ""),
    (k4, f"{fdf['avg_glucose_level'].mean():.1f}", "Avg Glucose",      "mg/dL"),
    (k5, f"{stroke_count:,}",                      "Stroke Cases",     ""),
    (k6, f"{stroke_rate}%",                        "Stroke Rate",
         "⚠ class imbalance" if stroke_rate < 10 else ""),
    (k7, f"{hyp_pct}%",                            "Hypertension",     "of cohort"),
]
for col, val, label, note in kpi_items:
    note_html = f"<div class='kpi-note'>{note}</div>" if note else ""
    col.markdown(f"""
    <div class='kpi-wrap'>
        <div class='kpi-val'>{val}</div>
        <div class='kpi-lbl'>{label}</div>
        {note_html}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊  Population",
    "❤️  Risk Factors",
    "📈  Distributions & Curves",
    "🔗  Correlations",
    "🤖  Model Results",
])

# ═══════════════════════════════════════════════════════════════════
# TAB 1 — POPULATION
# ═══════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='sec-hdr'>Demographic Breakdown</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    with c1:
        gdf = fdf["gender"].value_counts().reset_index()
        gdf.columns = ["Gender","Count"]
        ch = alt.Chart(gdf, title="Gender Split").mark_arc(
            innerRadius=60, padAngle=0.02, cornerRadius=4
        ).encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Gender:N",
                scale=alt.Scale(domain=["Female","Male"], range=["#F87171","#60A5FA"])),
            tooltip=["Gender","Count"]
        ).properties(height=230)
        st.altair_chart(ch, use_container_width=True)

    with c2:
        smk = fdf["smoking_status"].value_counts().reset_index()
        smk.columns = ["Status","Count"]
        ch = alt.Chart(smk, title="Smoking Status").mark_bar(
            cornerRadiusTopRight=5, cornerRadiusBottomRight=5,
            color=GOLD
        ).encode(
            x=alt.X("Count:Q", title="Patients"),
            y=alt.Y("Status:N", sort="-x", title=""),
            tooltip=["Status","Count"]
        ).properties(height=230)
        st.altair_chart(ch, use_container_width=True)

    with c3:
        wt = fdf["work_type"].value_counts().reset_index()
        wt.columns = ["Type","Count"]
        ch = alt.Chart(wt, title="Work Type").mark_bar(
            cornerRadiusTopRight=5, cornerRadiusBottomRight=5,
            color=PURPLE
        ).encode(
            x=alt.X("Count:Q", title="Patients"),
            y=alt.Y("Type:N", sort="-x", title=""),
            tooltip=["Type","Count"]
        ).properties(height=230)
        st.altair_chart(ch, use_container_width=True)

    # Stroke rate by age group — gradient bars
    st.markdown("<div class='sec-hdr'>Stroke Rate by Age Group</div>", unsafe_allow_html=True)
    age_s = (
        fdf.groupby("age_group", observed=True)
        .apply(lambda x: pd.Series({
            "Stroke Rate (%)": round((x["stroke"]=="Stroke").mean()*100, 2),
            "N": len(x)
        })).reset_index()
    )
    age_s["age_group"] = age_s["age_group"].astype(str)

    bars = alt.Chart(age_s, title="Stroke Prevalence Across Age Groups").mark_bar(
        cornerRadiusTopLeft=5, cornerRadiusTopRight=5
    ).encode(
        x=alt.X("age_group:N",
                sort=["0–30","30–45","45–60","60–75","75+"],
                title="Age Group",
                axis=alt.Axis(labelFontSize=12)),
        y=alt.Y("Stroke Rate (%):Q", title="Stroke Rate (%)"),
        color=alt.Color("Stroke Rate (%):Q",
                         scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=["age_group","Stroke Rate (%)","N"]
    ).properties(height=300)
    lbl = bars.mark_text(dy=-8, fontSize=11, color="#94A3B8").encode(
        text=alt.Text("Stroke Rate (%):Q", format=".1f")
    )
    st.altair_chart(bars + lbl, use_container_width=True)

    st.markdown("""<div class='ibox'>💡 <b>Key Insight:</b> Stroke risk climbs steeply after age 45.
    The 60–75 and 75+ groups account for the vast majority of cases.</div>""",
    unsafe_allow_html=True)

    c1b, c2b = st.columns(2)
    with c1b:
        bmi_s = (fdf.groupby(["bmi_category","stroke"], observed=True)
                    .size().reset_index(name="Count"))
        bmi_s["bmi_category"] = bmi_s["bmi_category"].astype(str)
        ch = alt.Chart(bmi_s, title="Stroke by BMI Category").mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X("bmi_category:N",
                    sort=["Underweight","Normal","Overweight","Obese"],
                    title="BMI Category"),
            y=alt.Y("Count:Q"),
            color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
            xOffset="stroke:N",
            tooltip=["bmi_category","stroke","Count"]
        ).properties(height=270)
        st.altair_chart(ch, use_container_width=True)

    with c2b:
        glu_s = (fdf.groupby(["high_glucose","stroke"], observed=True)
                    .size().reset_index(name="Count"))
        ch = alt.Chart(glu_s, title="Stroke by Glucose Level").mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X("high_glucose:N", title="Glucose Level"),
            y=alt.Y("Count:Q"),
            color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
            xOffset="stroke:N",
            tooltip=["high_glucose","stroke","Count"]
        ).properties(height=270)
        st.altair_chart(ch, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 2 — RISK FACTORS
# ═══════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='sec-hdr'>Clinical Risk Factors vs Stroke</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    def grouped_bar(col, title, container):
        tmp = fdf.groupby([col,"stroke"]).size().reset_index(name="Count")
        ch = alt.Chart(tmp, title=title).mark_bar(
            cornerRadiusTopLeft=4, cornerRadiusTopRight=4
        ).encode(
            x=alt.X(f"{col}:N", title=col),
            y=alt.Y("Count:Q"),
            color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
            xOffset="stroke:N",
            tooltip=[col,"stroke","Count"]
        ).properties(height=270)
        container.altair_chart(ch, use_container_width=True)

    grouped_bar("hypertension",  "Hypertension vs Stroke",  c1)
    grouped_bar("heart_disease", "Heart Disease vs Stroke",  c2)

    st.markdown("<div class='sec-hdr'>Stroke Rate by Risk Group</div>", unsafe_allow_html=True)

    rows = []
    for col, val, label in [
        ("hypertension",   "Yes",           "Hypertension"),
        ("hypertension",   "No",            "No Hypertension"),
        ("heart_disease",  "Yes",           "Heart Disease"),
        ("heart_disease",  "No",            "No Heart Disease"),
        ("high_glucose",   "High (>140)",   "High Glucose (>140)"),
        ("high_glucose",   "Normal (≤140)", "Normal Glucose (≤140)"),
        ("smoking_status", "Smoker",        "Active Smoker"),
        ("smoking_status", "Former Smoker", "Former Smoker"),
        ("smoking_status", "Never Smoked",  "Never Smoked"),
    ]:
        sub = fdf[fdf[col]==val]
        if len(sub) > 0:
            rows.append({
                "Group": label,
                "Stroke Rate": round((sub["stroke"]=="Stroke").mean()*100, 2),
                "N": len(sub)
            })

    risk_df = pd.DataFrame(rows)
    bars = alt.Chart(risk_df, title="Stroke Rate by Risk Group (%)").mark_bar(
        cornerRadiusTopRight=5, cornerRadiusBottomRight=5
    ).encode(
        x=alt.X("Stroke Rate:Q", title="Stroke Rate (%)"),
        y=alt.Y("Group:N", sort="-x", title=""),
        color=alt.Color("Stroke Rate:Q",
                         scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=["Group","Stroke Rate","N"]
    ).properties(height=310)
    lbl = bars.mark_text(align="left", dx=4, color="#64748B", fontSize=11).encode(
        text=alt.Text("Stroke Rate:Q", format=".1f")
    )
    st.altair_chart(bars + lbl, use_container_width=True)

    # Scatter
    st.markdown("<div class='sec-hdr'>Age × Glucose — Patient Cloud</div>", unsafe_allow_html=True)
    sample = fdf.sample(min(1200, len(fdf)), random_state=42)
    ch = alt.Chart(sample, title="Age vs Glucose Level (up to 1,200 patients)").mark_circle(
        opacity=0.5, size=32
    ).encode(
        x=alt.X("age:Q", title="Age"),
        y=alt.Y("avg_glucose_level:Q", title="Avg Glucose (mg/dL)"),
        color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
        tooltip=["age","avg_glucose_level","bmi","hypertension","heart_disease","stroke"]
    ).properties(height=360).interactive()
    st.altair_chart(ch, use_container_width=True)

    st.markdown("""<div class='ibox'>💡 <b>Key Insight:</b> Stroke patients (red) cluster toward
    the older, higher-glucose corner. Hypertension and heart disease both roughly double the stroke rate.
    High glucose (&gt;140 mg/dL) is a strong independent predictor.</div>""",
    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUTIONS & SMOOTH CURVES
# ═══════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='sec-hdr'>Health Metric Distributions</div>", unsafe_allow_html=True)

    metric  = st.radio("Metric", ["Age","BMI","Glucose Level"], horizontal=True)
    col_map = {"Age":"age","BMI":"bmi","Glucose Level":"avg_glucose_level"}
    chosen  = col_map[metric]

    # Histogram overlay
    ch = alt.Chart(fdf, title=f"{metric} Distribution by Outcome").mark_bar(
        opacity=0.65, binSpacing=0
    ).encode(
        x=alt.X(f"{chosen}:Q", bin=alt.Bin(maxbins=45), title=metric),
        y=alt.Y("count()", stack=None, title="Patients"),
        color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
        tooltip=[f"{chosen}:Q","count()","stroke:N"]
    ).properties(height=280)
    st.altair_chart(ch, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        ch_box = alt.Chart(fdf, title=f"{metric} — Box Plot").mark_boxplot(
            extent="min-max", outliers={"size":12,"opacity":0.25},
            median={"color":"white","size":2}
        ).encode(
            x=alt.X("stroke:N", title="Outcome"),
            y=alt.Y(f"{chosen}:Q", title=metric),
            color=alt.Color("stroke:N", scale=STROKE_SCALE, legend=None)
        ).properties(height=300)
        st.altair_chart(ch_box, use_container_width=True)

    with c2:
        # ── SMOOTH CURVE: rolling mean per age ──
        smoothed = smooth_line(fdf, "age", chosen, window=7)
        line = alt.Chart(smoothed, title=f"Smoothed Mean {metric} by Age").mark_line(
            strokeWidth=3, color=GOLD, interpolate="monotone"
        ).encode(
            x=alt.X("age:Q", title="Age"),
            y=alt.Y(chosen, title=f"Mean {metric}"),
            tooltip=["age", alt.Tooltip(chosen, format=".1f")]
        )
        # Confidence band — std envelope
        std_data = (fdf.groupby("age")[chosen]
                       .agg(["mean","std"])
                       .reset_index()
                       .rename(columns={"mean":"m","std":"s"}))
        std_data["lo"] = std_data["m"] - std_data["s"] * 0.5
        std_data["hi"] = std_data["m"] + std_data["s"] * 0.5
        std_data["lo"] = std_data["lo"].rolling(7, center=True, min_periods=1).mean()
        std_data["hi"] = std_data["hi"].rolling(7, center=True, min_periods=1).mean()

        band = alt.Chart(std_data).mark_area(
            opacity=0.15, color=GOLD, interpolate="monotone"
        ).encode(
            x="age:Q",
            y=alt.Y("lo:Q", title=""),
            y2="hi:Q"
        )

        if chosen == "avg_glucose_level":
            rule = alt.Chart(pd.DataFrame({"y":[140]})).mark_rule(
                color=STROKE_COLOR, strokeDash=[6,3], strokeWidth=1.5
            ).encode(y="y:Q")
            rule_lbl = alt.Chart(pd.DataFrame({"y":[143],"t":["Threshold: 140 mg/dL"]})).mark_text(
                align="left", dx=4, color=STROKE_COLOR, fontSize=10
            ).encode(y="y:Q", text="t:N", x=alt.value(8))
            st.altair_chart((band + line + rule + rule_lbl).properties(height=300),
                            use_container_width=True)
        else:
            st.altair_chart((band + line).properties(height=300),
                            use_container_width=True)

    # ── FULL-WIDTH GLUCOSE TREND (improved smooth curve) ──
    st.markdown("<div class='sec-hdr'>Glucose Trend by Age — Stroke vs No Stroke</div>",
                unsafe_allow_html=True)

    for_trend = fdf.copy()
    smooth_stroke    = smooth_line(for_trend[for_trend["stroke"]=="Stroke"],    "age","avg_glucose_level", window=7)
    smooth_no_stroke = smooth_line(for_trend[for_trend["stroke"]=="No Stroke"], "age","avg_glucose_level", window=7)
    smooth_stroke["Outcome"]    = "Stroke"
    smooth_no_stroke["Outcome"] = "No Stroke"
    trend_df = pd.concat([smooth_stroke, smooth_no_stroke], ignore_index=True)

    line_trend = alt.Chart(trend_df, title="Smoothed Mean Glucose by Age — Split by Outcome").mark_line(
        strokeWidth=2.8, interpolate="monotone"
    ).encode(
        x=alt.X("age:Q", title="Age"),
        y=alt.Y("avg_glucose_level:Q", title="Mean Glucose (mg/dL)"),
        color=alt.Color("Outcome:N",
                         scale=alt.Scale(domain=["No Stroke","Stroke"],
                                         range=[NO_STROKE_COLOR, STROKE_COLOR]),
                         title="Outcome"),
        tooltip=["age","Outcome",alt.Tooltip("avg_glucose_level:Q",format=".1f")]
    )
    threshold = alt.Chart(pd.DataFrame({"y":[140]})).mark_rule(
        color="#475569", strokeDash=[5,4], strokeWidth=1.5
    ).encode(y="y:Q")
    thr_lbl = alt.Chart(pd.DataFrame({"y":[143],"t":["High glucose threshold (140 mg/dL)"]})).mark_text(
        align="left", dx=4, color="#475569", fontSize=10
    ).encode(y="y:Q", text="t:N", x=alt.value(4))

    st.altair_chart(
        (line_trend + threshold + thr_lbl).properties(height=310),
        use_container_width=True
    )

    # ── STROKE RATE SMOOTH CURVE BY AGE ──
    st.markdown("<div class='sec-hdr'>Stroke Probability by Age — Smoothed</div>",
                unsafe_allow_html=True)

    rate_by_age = (fdf.groupby("age")
                      .apply(lambda x: (x["stroke"]=="Stroke").mean()*100)
                      .rename("Stroke Rate (%)")
                      .reset_index())
    rate_by_age["Stroke Rate (%)"] = (
        rate_by_age["Stroke Rate (%)"]
        .rolling(window=9, center=True, min_periods=1).mean()
    )

    area = alt.Chart(rate_by_age).mark_area(
        opacity=0.2, color=STROKE_COLOR, interpolate="monotone"
    ).encode(
        x=alt.X("age:Q", title="Age"),
        y=alt.Y("Stroke Rate (%):Q", title="Stroke Rate (%)")
    )
    line_rate = alt.Chart(rate_by_age).mark_line(
        strokeWidth=3, color=STROKE_COLOR, interpolate="monotone"
    ).encode(
        x="age:Q",
        y=alt.Y("Stroke Rate (%):Q"),
        tooltip=["age", alt.Tooltip("Stroke Rate (%):Q", format=".2f")]
    )
    st.altair_chart((area + line_rate).properties(height=290), use_container_width=True)

    st.markdown("""<div class='ibox'>📈 <b>Smooth curves</b> use a rolling-window mean
    (window = 7–9 age steps) to reduce noise while preserving the true trend.
    The shaded band in the upper-right chart shows ± ½ standard deviation.
    Stroke patients consistently show higher glucose at every age.</div>""",
    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════
# TAB 4 — CORRELATIONS
# ═══════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='sec-hdr'>Correlation Matrix</div>", unsafe_allow_html=True)

    num_df = fdf.select_dtypes(include=["int64","float64"]).copy()
    num_df["stroke_num"]        = (fdf["stroke"]        == "Stroke").astype(int)
    num_df["hypertension_num"]  = (fdf["hypertension"]  == "Yes").astype(int)
    num_df["heart_disease_num"] = (fdf["heart_disease"] == "Yes").astype(int)

    keep = [c for c in ["age","bmi","avg_glucose_level",
                         "hypertension_num","heart_disease_num",
                         "cardio_risk","stroke_num"] if c in num_df.columns]
    corr = num_df[keep].corr().reset_index().melt("index")
    corr.columns = ["var1","var2","correlation"]

    heatmap = alt.Chart(corr, title="Feature Correlation Heatmap").mark_rect(
        cornerRadius=3
    ).encode(
        x=alt.X("var1:N", title=""),
        y=alt.Y("var2:N", title=""),
        color=alt.Color("correlation:Q",
                         scale=alt.Scale(scheme="redblue", domain=[-1,1], reverse=True),
                         title="r"),
        tooltip=["var1","var2",alt.Tooltip("correlation:Q",format=".3f")]
    ).properties(height=380)
    text_layer = heatmap.mark_text(fontSize=11, fontWeight=600).encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.condition(
            alt.datum.correlation > 0.4,
            alt.value("white"), alt.value("#475569")
        )
    )
    st.altair_chart(heatmap + text_layer, use_container_width=True)

    st.markdown("""<div class='ibox'>💡 <b>Reading this:</b>
    The <b>stroke_num</b> column shows the strongest predictors.
    <b>age</b> has the highest positive correlation with stroke.
    Near ±1 = strong linear relationship; near 0 = weak.</div>""",
    unsafe_allow_html=True)

    # Scatter matrix
    st.markdown("<div class='sec-hdr'>Age × BMI × Glucose — Scatter Matrix</div>",
                unsafe_allow_html=True)
    smpl = fdf.sample(min(800, len(fdf)), random_state=1)[
        ["age","bmi","avg_glucose_level","stroke"]
    ]
    ch = alt.Chart(smpl).mark_circle(size=22, opacity=0.45).encode(
        alt.X(alt.repeat("column"), type="quantitative"),
        alt.Y(alt.repeat("row"),    type="quantitative"),
        color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome")
    ).repeat(
        row=["age","bmi","avg_glucose_level"],
        column=["avg_glucose_level","bmi","age"]
    ).properties(title="Pairwise Scatter Matrix (800 random patients)")
    st.altair_chart(ch, use_container_width=True)

    # Data preview
    st.markdown("<div class='sec-hdr'>Data Preview</div>", unsafe_allow_html=True)
    cols_show = ["age","gender","hypertension","heart_disease","bmi",
                 "avg_glucose_level","smoking_status","work_type","stroke"]
    st.dataframe(fdf[cols_show].head(30).reset_index(drop=True),
                 use_container_width=True, height=330)

# ═══════════════════════════════════════════════════════════════════
# TAB 5 — ML MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='sec-hdr'>ML Model Performance — Results from Notebook</div>",
                unsafe_allow_html=True)

    model_data = pd.DataFrame({
        "Model": [
            "Logistic Reg. (Base)",
            "Logistic Reg. (New Feats)",
            "Random Forest (Base)",
            "Random Forest (New Feats)",
            "Optimized Random Forest",
            "Decision Tree (Initial)",
            "Optimized Decision Tree",
            "XGBoost (Initial)",
            "Tuned XGBoost",
        ],
        "Accuracy":  [0.9393,0.9393,0.9403,0.9393,0.7573,0.7740,0.7535,0.7759,0.7632],
        "Precision": [0.0000,0.0000,0.3333,0.2000,0.2133,0.1606,0.1615,0.1700,0.1538],
        "Recall":    [0.0000,0.0000,0.0323,0.0161,0.5161,0.6452,0.7581,0.6935,0.6452],
        "F1-Score":  [0.0000,0.0000,0.0588,0.0299,0.3019,0.2572,0.2663,0.2730,0.2484],
    })

    best_idx  = int(model_data["Recall"].idxmax())
    best_name = model_data.loc[best_idx,"Model"]

    st.markdown(f"""<div class='ibox'>
    🏆 <b>Best for Medical Screening (highest Recall):</b>
    <b style='color:#F87171'>{best_name}</b>
    &nbsp;—&nbsp; Recall <b>{model_data.loc[best_idx,'Recall']:.1%}</b>
    &nbsp;·&nbsp; F1 <b>{model_data.loc[best_idx,'F1-Score']:.1%}</b><br><br>
    In stroke detection <b>Recall matters most</b> — missing a real stroke case is far more costly than
    a false alarm. Both Logistic Regression variants scored 0 on the stroke class due to severe class
    imbalance (~4.9% positive rate). XGBoost used <code>scale_pos_weight = 20.86</code> to partially compensate.
    </div>""", unsafe_allow_html=True)

    metric_choice = st.radio("Metric to display",
                              ["Precision","Recall","F1-Score"], horizontal=True)

    bar_data = model_data[["Model",metric_choice]].copy()
    bar_data["Highlight"] = bar_data["Model"] == best_name

    ch = alt.Chart(bar_data, title=f"{metric_choice} — Stroke Class (all models)").mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X("Model:N", sort=None, title="",
                axis=alt.Axis(labelAngle=-35, labelLimit=150)),
        y=alt.Y(f"{metric_choice}:Q", scale=alt.Scale(domain=[0,0.88])),
        color=alt.condition(
            alt.datum.Highlight,
            alt.value(STROKE_COLOR),
            alt.value("#1E3A5F")
        ),
        tooltip=["Model", alt.Tooltip(f"{metric_choice}:Q", format=".1%")]
    ).properties(height=320)
    lbl = ch.mark_text(dy=-8, color="#64748B", fontSize=11).encode(
        text=alt.Text(f"{metric_choice}:Q", format=".1%")
    )
    st.altair_chart(ch + lbl, use_container_width=True)

    # All 3 metrics
    st.markdown("<div class='sec-hdr'>All Metrics — Side by Side</div>", unsafe_allow_html=True)
    melted = model_data.melt(
        id_vars="Model", value_vars=["Precision","Recall","F1-Score"],
        var_name="Metric", value_name="Score"
    )
    ch_all = alt.Chart(melted, title="Model Comparison — Precision / Recall / F1").mark_bar(
        cornerRadiusTopLeft=3, cornerRadiusTopRight=3
    ).encode(
        x=alt.X("Model:N", sort=None, title="",
                axis=alt.Axis(labelAngle=-35, labelLimit=150)),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0,0.88])),
        color=alt.Color("Metric:N",
                         scale=alt.Scale(
                             domain=["Precision","Recall","F1-Score"],
                             range=[NO_STROKE_COLOR, STROKE_COLOR, GOLD]
                         ), title="Metric"),
        xOffset="Metric:N",
        tooltip=["Model","Metric",alt.Tooltip("Score:Q",format=".1%")]
    ).properties(height=340)
    st.altair_chart(ch_all, use_container_width=True)

    # Table
    st.markdown("<div class='sec-hdr'>Full Results Table</div>", unsafe_allow_html=True)
    disp = model_data.copy()
    for c in ["Accuracy","Precision","Recall","F1-Score"]:
        disp[c] = disp[c].map(lambda x: f"{x:.1%}")
    st.dataframe(disp.set_index("Model"), use_container_width=True, height=360)

    st.markdown("""<div class='ibox'>
    📌 <b>Feature Engineering Impact:</b> Adding clinical features (high-glucose flag, cardio risk,
    BMI category, age group) did <em>not</em> significantly improve simpler models.
    The bottleneck was class imbalance, not feature richness. <b>SMOTE</b> or cost-sensitive learning
    would likely yield larger gains.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.markdown("""<div style='text-align:center;color:#1E2D4A;font-size:0.78rem;padding:6px 0 18px'>
    Stroke Risk Dashboard · Streamlit + Altair · No external chart dependencies
</div>""", unsafe_allow_html=True)
