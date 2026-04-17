import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #E8EAF0;
}
.main { background-color: #0D1117; }
section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1F2937; }

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Page title ── */
.dash-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #F87171 0%, #FB923C 50%, #FBBF24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0;
    line-height: 1.1;
}
.dash-sub {
    color: #6B7280;
    font-size: 0.95rem;
    font-weight: 300;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* ── KPI cards ── */
.kpi-card {
    background: linear-gradient(135deg, #161D2E 0%, #1A2236 100%);
    border: 1px solid #1F2D40;
    border-radius: 14px;
    padding: 18px 22px;
    text-align: center;
    transition: border-color 0.25s;
}
.kpi-card:hover { border-color: #F87171; }
.kpi-value {
    font-family: 'DM Serif Display', serif;
    font-size: 2.1rem;
    color: #F87171;
    line-height: 1;
}
.kpi-label {
    color: #9CA3AF;
    font-size: 0.78rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-top: 6px;
}
.kpi-delta { font-size: 0.75rem; color: #6EE7B7; margin-top: 4px; }

/* ── Section headers ── */
.section-header {
    font-family: 'DM Serif Display', serif;
    font-size: 1.3rem;
    color: #F3F4F6;
    border-left: 3px solid #F87171;
    padding-left: 10px;
    margin: 24px 0 12px;
}

/* ── Alert / insight box ── */
.insight-box {
    background: #1C2333;
    border: 1px solid #2D3748;
    border-left: 4px solid #F87171;
    border-radius: 8px;
    padding: 14px 18px;
    margin: 10px 0;
    font-size: 0.88rem;
    color: #CBD5E1;
}

/* ── Model badge ── */
.model-badge {
    display: inline-block;
    background: #1E293B;
    border: 1px solid #334155;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.76rem;
    color: #94A3B8;
    margin: 3px;
}
.model-badge.best { border-color: #F87171; color: #F87171; }

/* ── Tab override ── */
button[data-baseweb="tab"] { font-family: 'DM Sans', sans-serif !important; }

/* ── Plotly chart containers ── */
.plot-container { border-radius: 12px; overflow: hidden; }

/* ── Divider ── */
.dash-divider { border-top: 1px solid #1F2937; margin: 20px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PLOTLY TEMPLATE  (dark, matching the palette)
# ─────────────────────────────────────────────────────────────────────────────
PLOT_BG  = "#0D1117"
PAPER_BG = "#111827"
GRID_CLR = "#1F2937"
FONT_CLR = "#9CA3AF"
ACCENT   = ["#F87171", "#FB923C", "#FBBF24", "#34D399", "#60A5FA",
            "#A78BFA", "#F472B6", "#22D3EE"]

LAYOUT_DEFAULTS = dict(
    paper_bgcolor=PAPER_BG,
    plot_bgcolor=PLOT_BG,
    font=dict(family="DM Sans", color=FONT_CLR, size=12),
    margin=dict(l=30, r=20, t=40, b=30),
    xaxis=dict(gridcolor=GRID_CLR, zeroline=False),
    yaxis=dict(gridcolor=GRID_CLR, zeroline=False),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
    colorway=ACCENT,
)

def apply_defaults(fig):
    fig.update_layout(**LAYOUT_DEFAULTS)
    return fig

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_DataSet_Stroke.csv")

    df["gender"]         = df["gender"].astype(str).str.strip()
    df["smoking_status"] = df["smoking_status"].astype(str).str.strip()
    df["work_type"]      = df["work_type"].astype(str).str.strip()

    df["gender"] = df["gender"].replace({"0": "Female","1": "Male", 0: "Female", 1: "Male"})

    df["hypertension"]  = df["hypertension"].replace({0: "No",  1: "Yes"})
    df["heart_disease"] = df["heart_disease"].replace({0: "No", 1: "Yes"})
    df["stroke"]        = df["stroke"].replace({0: "No Stroke", 1: "Stroke"})

    df["smoking_status"] = df["smoking_status"].replace({
        "never smoked": "Never Smoked", "formerly smoked": "Former Smoker",
        "smokes": "Smoker", "Unknown": "Unknown", "0": "Unknown", 0: "Unknown"
    })
    df["work_type"] = df["work_type"].replace({
        "Private": "Private Sector", "Self-employed": "Self Employed",
        "Govt_job": "Government Job", "children": "Children",
        "Never_worked": "Never Worked"
    })

    # Feature engineering (from notebook)
    df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
    df["high_glucose"]  = (df["avg_glucose_level"] > 140).map({True:"High",False:"Normal"})
    df["bmi_category"]  = pd.cut(df["bmi"], bins=[0,18.5,25,30,200],
                                  labels=["Underweight","Normal","Overweight","Obese"])
    df["age_group"]     = pd.cut(df["age"], bins=[0,30,45,60,75,200],
                                  labels=["0–30","30–45","45–60","60–75","75+"])
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Filters")
    st.markdown("<div class='dash-divider'></div>", unsafe_allow_html=True)

    gender  = st.selectbox("Gender",  ["All"] + sorted(df["gender"].dropna().unique().tolist()))
    smoking = st.selectbox("Smoking", ["All"] + sorted(df["smoking_status"].dropna().unique().tolist()))
    work    = st.selectbox("Work Type",["All"] + sorted(df["work_type"].dropna().unique().tolist()))
    stroke_filter = st.selectbox("Outcome", ["All", "Stroke", "No Stroke"])

    st.markdown("<div class='dash-divider'></div>", unsafe_allow_html=True)
    age_range = st.slider("Age Range", int(df.age.min()), int(df.age.max()), (20, 80))
    bmi_range = st.slider("BMI Range", int(df.bmi.min()), int(df.bmi.max()), (15, 40))
    glu_range = st.slider("Glucose Range",
                           int(df.avg_glucose_level.min()), int(df.avg_glucose_level.max()), (55, 272))

    st.markdown("<div class='dash-divider'></div>", unsafe_allow_html=True)
    st.caption("Stroke Risk Dashboard · CS Project")

# ─────────────────────────────────────────────────────────────────────────────
# FILTER
# ─────────────────────────────────────────────────────────────────────────────
fdf = df.copy()
if gender  != "All": fdf = fdf[fdf["gender"]         == gender]
if smoking != "All": fdf = fdf[fdf["smoking_status"]  == smoking]
if work    != "All": fdf = fdf[fdf["work_type"]        == work]
if stroke_filter != "All": fdf = fdf[fdf["stroke"]     == stroke_filter]
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
col_title, col_badge = st.columns([3,1])
with col_title:
    st.markdown("<div class='dash-title'>🧠 Stroke Risk Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-sub'>Exploratory & Predictive Analysis · 5,110 Patients</div>", unsafe_allow_html=True)
with col_badge:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""<div style='text-align:right;color:#6B7280;font-size:0.82rem'>
        Showing <b style='color:#F87171'>{total:,}</b> of {len(df):,} patients
    </div>""", unsafe_allow_html=True)

st.markdown("<div class='dash-divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1,k2,k3,k4,k5,k6 = st.columns(6)
cards = [
    (k1, f"{total:,}",    "Total Patients", ""),
    (k2, f"{fdf['age'].mean():.1f}",  "Avg Age (yrs)", ""),
    (k3, f"{fdf['bmi'].mean():.1f}",  "Avg BMI", ""),
    (k4, f"{fdf['avg_glucose_level'].mean():.1f}", "Avg Glucose", ""),
    (k5, f"{stroke_count:,}",  "Stroke Cases", ""),
    (k6, f"{stroke_rate}%", "Stroke Rate", "⚠️ class imbalance" if stroke_rate < 10 else ""),
]
for col, val, label, note in cards:
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-label'>{label}</div>
        {'<div class="kpi-delta">'+note+'</div>' if note else ''}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Population", "❤️ Risk Factors", "📈 Distributions",
    "🔗 Correlations", "🤖 Model Results"
])

# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — POPULATION OVERVIEW
# ════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Demographic Breakdown</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    # Gender donut
    with c1:
        gen_counts = fdf["gender"].value_counts().reset_index()
        gen_counts.columns = ["Gender","Count"]
        fig = px.pie(gen_counts, names="Gender", values="Count",
                     title="Gender Split", hole=0.55,
                     color_discrete_sequence=["#F87171","#60A5FA"])
        fig.update_traces(textposition="outside", textinfo="percent+label")
        apply_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Smoking status
    with c2:
        smk = fdf["smoking_status"].value_counts().reset_index()
        smk.columns = ["Status","Count"]
        fig = px.bar(smk, x="Count", y="Status", orientation="h",
                     title="Smoking Status", color="Count",
                     color_continuous_scale=["#1E293B","#F87171"])
        fig.update_coloraxes(showscale=False)
        apply_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Work type
    with c3:
        wt = fdf["work_type"].value_counts().reset_index()
        wt.columns = ["Type","Count"]
        fig = px.bar(wt, x="Count", y="Type", orientation="h",
                     title="Work Type", color="Count",
                     color_continuous_scale=["#1E293B","#FB923C"])
        fig.update_coloraxes(showscale=False)
        apply_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Stroke by age group
    st.markdown("<div class='section-header'>Stroke Rate by Age Group</div>", unsafe_allow_html=True)
    age_stroke = fdf.groupby("age_group", observed=True).apply(
        lambda x: (x["stroke"] == "Stroke").mean() * 100
    ).reset_index()
    age_stroke.columns = ["Age Group","Stroke Rate (%)"]
    fig = px.bar(age_stroke, x="Age Group", y="Stroke Rate (%)",
                 title="Stroke Prevalence Across Age Groups",
                 color="Stroke Rate (%)",
                 color_continuous_scale=["#1E293B","#FBBF24","#F87171"])
    fig.update_coloraxes(showscale=False)
    apply_defaults(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class='insight-box'>
    💡 <b>Key Insight:</b> Stroke risk increases dramatically after age 45, with the 60–75 and 75+ age
    groups showing the highest prevalence — consistent with known clinical epidemiology.
    </div>""", unsafe_allow_html=True)

    # Stroke by BMI category
    c1b, c2b = st.columns(2)
    with c1b:
        bmi_stroke = fdf.groupby("bmi_category", observed=True).apply(
            lambda x: pd.Series({
                "No Stroke": (x["stroke"]=="No Stroke").sum(),
                "Stroke":    (x["stroke"]=="Stroke").sum()
            })
        ).reset_index()
        fig = px.bar(bmi_stroke, x="bmi_category", y=["No Stroke","Stroke"],
                     barmode="group", title="Stroke by BMI Category",
                     color_discrete_map={"No Stroke":"#60A5FA","Stroke":"#F87171"},
                     labels={"bmi_category":"BMI Category","value":"Patients","variable":""})
        apply_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2b:
        glu_stroke = fdf.groupby("high_glucose").apply(
            lambda x: pd.Series({
                "No Stroke": (x["stroke"]=="No Stroke").sum(),
                "Stroke":    (x["stroke"]=="Stroke").sum()
            })
        ).reset_index()
        fig = px.bar(glu_stroke, x="high_glucose", y=["No Stroke","Stroke"],
                     barmode="group", title="Stroke by Glucose Level",
                     color_discrete_map={"No Stroke":"#60A5FA","Stroke":"#F87171"},
                     labels={"high_glucose":"Glucose Level","value":"Patients","variable":""})
        apply_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK FACTORS
# ════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Clinical Risk Factors vs Stroke</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    def risk_grouped_bar(col, title, c):
        counts = fdf.groupby([col,"stroke"]).size().reset_index(name="Count")
        fig = px.bar(counts, x=col, y="Count", color="stroke", barmode="group",
                     title=title, color_discrete_map={"No Stroke":"#60A5FA","Stroke":"#F87171"},
                     labels={"stroke":"Outcome",col:col})
        apply_defaults(fig)
        c.plotly_chart(fig, use_container_width=True)

    risk_grouped_bar("hypertension",  "Hypertension vs Stroke",  c1)
    risk_grouped_bar("heart_disease", "Heart Disease vs Stroke",  c2)

    # Stroke rate per risk factor — horizontal summary
    st.markdown("<div class='section-header'>Stroke Rate — Key Risk Comparisons</div>", unsafe_allow_html=True)

    risk_rows = []
    for col, val, label in [
        ("hypertension","Yes","Hypertension"),
        ("hypertension","No","No Hypertension"),
        ("heart_disease","Yes","Heart Disease"),
        ("heart_disease","No","No Heart Disease"),
        ("high_glucose","High","High Glucose (>140)"),
        ("high_glucose","Normal","Normal Glucose"),
        ("smoking_status","Smoker","Active Smoker"),
        ("smoking_status","Former Smoker","Former Smoker"),
        ("smoking_status","Never Smoked","Never Smoked"),
    ]:
        sub = fdf[fdf[col]==val]
        if len(sub) > 0:
            rate = (sub["stroke"]=="Stroke").mean()*100
            risk_rows.append({"Group":label,"Stroke Rate (%)":round(rate,2),"N":len(sub)})

    risk_df = pd.DataFrame(risk_rows).sort_values("Stroke Rate (%)", ascending=True)
    fig = px.bar(risk_df, x="Stroke Rate (%)", y="Group", orientation="h",
                 title="Stroke Rate by Risk Group (%)",
                 color="Stroke Rate (%)",
                 color_continuous_scale=["#1E293B","#FBBF24","#F87171"],
                 text="Stroke Rate (%)")
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
    fig.update_coloraxes(showscale=False)
    apply_defaults(fig)
    st.plotly_chart(fig, use_container_width=True)

    # Age vs Stroke scatter
    st.markdown("<div class='section-header'>Age & Glucose — Individual Patients</div>", unsafe_allow_html=True)
    sample = fdf.sample(min(1000, len(fdf)), random_state=42)
    fig = px.scatter(sample, x="age", y="avg_glucose_level", color="stroke",
                     color_discrete_map={"No Stroke":"#3B82F6","Stroke":"#F87171"},
                     opacity=0.6, title="Age vs Glucose Level (random 1000 sample)",
                     labels={"age":"Age","avg_glucose_level":"Avg Glucose Level"},
                     hover_data=["bmi","hypertension","heart_disease"])
    apply_defaults(fig)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class='insight-box'>
    💡 <b>Key Insight:</b> Stroke patients cluster in the older + high-glucose region. Hypertension
    and heart disease both roughly double the stroke rate, while high glucose (&gt;140 mg/dL) is 
    a strong independent predictor.
    </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUTIONS
# ════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Health Metric Distributions</div>", unsafe_allow_html=True)

    metric = st.radio("Select metric", ["Age","BMI","Glucose"], horizontal=True)
    col_map = {"Age":"age","BMI":"bmi","Glucose":"avg_glucose_level"}
    chosen  = col_map[metric]

    fig = px.histogram(fdf, x=chosen, color="stroke", nbins=40, barmode="overlay",
                       color_discrete_map={"No Stroke":"#3B82F6","Stroke":"#F87171"},
                       opacity=0.75, title=f"{metric} Distribution by Stroke Outcome",
                       labels={chosen:metric,"stroke":"Outcome"})
    apply_defaults(fig)
    st.plotly_chart(fig, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.box(fdf, x="stroke", y=chosen, color="stroke",
                     color_discrete_map={"No Stroke":"#3B82F6","Stroke":"#F87171"},
                     title=f"{metric} — Box Plot by Outcome",
                     labels={chosen:metric,"stroke":"Outcome"})
        apply_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.violin(fdf, x="stroke", y=chosen, color="stroke", box=True,
                        color_discrete_map={"No Stroke":"#3B82F6","Stroke":"#F87171"},
                        title=f"{metric} — Violin Plot by Outcome",
                        labels={chosen:metric,"stroke":"Outcome"})
        apply_defaults(fig)
        st.plotly_chart(fig, use_container_width=True)

    # Glucose trend by age
    st.markdown("<div class='section-header'>Avg Glucose Level by Age</div>", unsafe_allow_html=True)
    age_glu = fdf.groupby("age")["avg_glucose_level"].mean().reset_index()
    fig = px.line(age_glu, x="age", y="avg_glucose_level",
                  title="Mean Glucose Level Across Age",
                  labels={"age":"Age","avg_glucose_level":"Avg Glucose (mg/dL)"},
                  line_shape="spline")
    fig.update_traces(line_color="#FBBF24", line_width=2)
    fig.add_hline(y=140, line_dash="dash", line_color="#F87171",
                  annotation_text="High Glucose Threshold (140)", annotation_position="top right")
    apply_defaults(fig)
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — CORRELATIONS
# ════════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Correlation Matrix</div>", unsafe_allow_html=True)

    numeric_df = fdf.select_dtypes(include=["int64","float64"])
    # keep stroke as numeric for correlation
    stroke_num = (fdf["stroke"] == "Stroke").astype(int)
    numeric_df["stroke_num"] = stroke_num

    corr = numeric_df.corr()

    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                    zmin=-1, zmax=1, aspect="auto",
                    title="Feature Correlation Heatmap")
    apply_defaults(fig)
    fig.update_layout(height=500, coloraxis_colorbar=dict(title="r"))
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<div class='insight-box'>
    💡 <b>Reading this:</b> Look at the <b>stroke_num</b> row/column for direct predictors.
    <b>age</b> has the strongest positive correlation with stroke. 
    Features close to ±1 indicate strong linear relationships; near 0 means weak.
    </div>""", unsafe_allow_html=True)

    # Pairwise scatter matrix
    st.markdown("<div class='section-header'>Pairwise Scatter Matrix</div>", unsafe_allow_html=True)
    scatter_cols = ["age","bmi","avg_glucose_level"]
    fig = px.scatter_matrix(fdf.sample(min(800,len(fdf)), random_state=1),
                             dimensions=scatter_cols,
                             color="stroke",
                             color_discrete_map={"No Stroke":"#3B82F6","Stroke":"#F87171"},
                             opacity=0.5,
                             labels={"age":"Age","bmi":"BMI","avg_glucose_level":"Glucose"})
    fig.update_traces(diagonal_visible=False, marker=dict(size=3))
    apply_defaults(fig)
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

    # Data preview
    st.markdown("<div class='section-header'>Data Preview</div>", unsafe_allow_html=True)
    cols_show = ["age","gender","hypertension","heart_disease","bmi",
                 "avg_glucose_level","smoking_status","work_type","stroke"]
    st.dataframe(
        fdf[cols_show].head(30).reset_index(drop=True),
        use_container_width=True,
        height=320
    )

# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ML MODEL RESULTS  (from notebook)
# ════════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>ML Model Performance Comparison</div>", unsafe_allow_html=True)

    model_data = pd.DataFrame({
        "Model": [
            "Logistic Regression (Base)",
            "Logistic Regression (New Features)",
            "Random Forest (Base)",
            "Random Forest (New Features)",
            "Optimized Random Forest",
            "Decision Tree (Initial)",
            "Optimized Decision Tree",
            "XGBoost (Initial)",
            "Tuned XGBoost"
        ],
        "Accuracy": [0.9393, 0.9393, 0.9403, 0.9393, 0.7573, 0.7740, 0.7535, 0.7759, 0.7632],
        "Precision":[0.0000, 0.0000, 0.3333, 0.2000, 0.2133, 0.1606, 0.1615, 0.1700, 0.1538],
        "Recall":   [0.0000, 0.0000, 0.0323, 0.0161, 0.5161, 0.6452, 0.7581, 0.6935, 0.6452],
        "F1-Score": [0.0000, 0.0000, 0.0588, 0.0299, 0.3019, 0.2572, 0.2663, 0.2730, 0.2484],
    })

    # Best model annotation
    best_idx = model_data["Recall"].idxmax()  # recall matters most for medical screening
    st.markdown(f"""<div class='insight-box'>
    🏆 <b>Best for Medical Screening (highest Recall):</b>
    <span class='model-badge best'>{model_data.loc[best_idx,'Model']}</span>
    — Recall <b>{model_data.loc[best_idx,'Recall']:.1%}</b> · F1 <b>{model_data.loc[best_idx,'F1-Score']:.1%}</b><br><br>
    In stroke screening, <b>Recall</b> is the priority metric — missing a real stroke case is far more costly
    than a false alarm. All logistic regression models failed completely on the minority class due to
    class imbalance (~4.9% stroke rate).
    </div>""", unsafe_allow_html=True)

    # Grouped bar — Precision / Recall / F1
    fig = make_subplots(rows=1, cols=3,
                        subplot_titles=["Precision (Stroke Class)",
                                        "Recall (Stroke Class)",
                                        "F1-Score (Stroke Class)"])

    colors = [ACCENT[0] if i == best_idx else ACCENT[5] for i in range(len(model_data))]

    for i, metric in enumerate(["Precision","Recall","F1-Score"], start=1):
        fig.add_trace(go.Bar(
            x=model_data["Model"],
            y=model_data[metric],
            name=metric,
            marker_color=colors,
            text=[f"{v:.1%}" for v in model_data[metric]],
            textposition="outside",
            showlegend=False,
        ), row=1, col=i)

    fig.update_layout(
        height=420, title_text="Model Comparison — Stroke Detection Metrics",
        **{k:v for k,v in LAYOUT_DEFAULTS.items() if k not in ("xaxis","yaxis")},
        xaxis=dict(tickangle=-45, gridcolor=GRID_CLR),
        xaxis2=dict(tickangle=-45, gridcolor=GRID_CLR),
        xaxis3=dict(tickangle=-45, gridcolor=GRID_CLR),
        yaxis=dict(range=[0,0.9], gridcolor=GRID_CLR),
        yaxis2=dict(range=[0,0.9], gridcolor=GRID_CLR),
        yaxis3=dict(range=[0,0.9], gridcolor=GRID_CLR),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Radar chart
    st.markdown("<div class='section-header'>Radar — Multi-Metric Overview</div>", unsafe_allow_html=True)
    radar_models = ["Optimized Random Forest","Optimized Decision Tree","XGBoost (Initial)","Tuned XGBoost"]
    metrics_radar = ["Precision","Recall","F1-Score"]

    fig_radar = go.Figure()
    for m_name, color in zip(radar_models, ACCENT):
        row = model_data[model_data["Model"]==m_name].iloc[0]
        vals = [row[m] for m in metrics_radar]
        vals += vals[:1]
        fig_radar.add_trace(go.Scatterpolar(
            r=vals,
            theta=metrics_radar + [metrics_radar[0]],
            fill="toself",
            name=m_name,
            line_color=color,
            fillcolor=color.replace(")", ", 0.15)").replace("rgb","rgba") if color.startswith("rgb") else color,
            opacity=0.8,
        ))
    fig_radar.update_layout(
        polar=dict(
            bgcolor=PLOT_BG,
            radialaxis=dict(visible=True, range=[0,0.8], color=FONT_CLR, gridcolor=GRID_CLR),
            angularaxis=dict(color=FONT_CLR, gridcolor=GRID_CLR)
        ),
        **{k:v for k,v in LAYOUT_DEFAULTS.items() if k not in ("xaxis","yaxis")},
        height=400
    )
    st.plotly_chart(fig_radar, use_container_width=True)

    # Table
    st.markdown("<div class='section-header'>Full Results Table</div>", unsafe_allow_html=True)

    styled = model_data.copy()
    styled["Accuracy"]  = styled["Accuracy"].map(lambda x: f"{x:.1%}")
    styled["Precision"] = styled["Precision"].map(lambda x: f"{x:.1%}")
    styled["Recall"]    = styled["Recall"].map(lambda x: f"{x:.1%}")
    styled["F1-Score"]  = styled["F1-Score"].map(lambda x: f"{x:.1%}")
    st.dataframe(styled.set_index("Model"), use_container_width=True, height=360)

    st.markdown("""<div class='insight-box'>
    📌 <b>Feature Engineering Impact:</b> Adding clinical features (high glucose flag, cardio risk, BMI category, 
    age group) did not significantly improve Logistic Regression or basic Random Forest, suggesting 
    these models require better handling of class imbalance (SMOTE, cost-sensitive learning) rather 
    than more features alone. XGBoost with <code>scale_pos_weight=20.86</code> performed best overall.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='dash-divider'></div>", unsafe_allow_html=True)
st.markdown("""
<div style='text-align:center;color:#374151;font-size:0.78rem;padding:10px 0 20px'>
    Stroke Risk Dashboard · Built with Streamlit & Plotly · Data: 5,110 patients
</div>
""", unsafe_allow_html=True)
