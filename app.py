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
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; color: #E8EAF0; }
.main { background-color: #0D1117; }
section[data-testid="stSidebar"] { background-color: #111827; border-right: 1px solid #1F2937; }
#MainMenu, footer, header { visibility: hidden; }

.dash-title {
    font-family: 'DM Serif Display', serif;
    font-size: 2.6rem;
    background: linear-gradient(135deg, #F87171 0%, #FB923C 50%, #FBBF24 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0; line-height: 1.1;
}
.dash-sub {
    color: #6B7280; font-size: 0.9rem; font-weight: 300;
    letter-spacing: 0.06em; text-transform: uppercase; margin-top: 4px;
}
.kpi-card {
    background: linear-gradient(135deg, #161D2E 0%, #1A2236 100%);
    border: 1px solid #1F2D40; border-radius: 14px;
    padding: 18px 22px; text-align: center; transition: border-color 0.25s;
}
.kpi-card:hover { border-color: #F87171; }
.kpi-value { font-family: 'DM Serif Display', serif; font-size: 2.1rem; color: #F87171; line-height: 1; }
.kpi-label { color: #9CA3AF; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px; }
.kpi-note  { font-size: 0.72rem; color: #6EE7B7; margin-top: 4px; }

.section-header {
    font-family: 'DM Serif Display', serif; font-size: 1.25rem; color: #F3F4F6;
    border-left: 3px solid #F87171; padding-left: 10px; margin: 24px 0 10px;
}
.insight-box {
    background: #1C2333; border: 1px solid #2D3748;
    border-left: 4px solid #F87171; border-radius: 8px;
    padding: 14px 18px; margin: 10px 0; font-size: 0.88rem; color: #CBD5E1;
    line-height: 1.6;
}
.divider { border-top: 1px solid #1F2937; margin: 18px 0; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ALTAIR DARK THEME
# ─────────────────────────────────────────────────────────────────────────────
STROKE_COLOR    = "#F87171"
NO_STROKE_COLOR = "#60A5FA"

def dark_theme():
    return {
        "config": {
            "background": "#111827",
            "view": {"stroke": "transparent", "fill": "#111827"},
            "axis": {
                "domainColor": "#374151", "gridColor": "#1F2937",
                "labelColor": "#9CA3AF",  "titleColor": "#9CA3AF",
                "tickColor":  "#374151",  "labelFontSize": 11,
            },
            "legend": {"labelColor": "#9CA3AF", "titleColor": "#9CA3AF", "fillColor": "#111827"},
            "title":  {"color": "#E5E7EB", "fontSize": 13, "fontWeight": 500},
            "mark":   {"tooltip": True},
        }
    }

alt.themes.register("dark", dark_theme)
alt.themes.enable("dark")

STROKE_SCALE = alt.Scale(domain=["No Stroke", "Stroke"],
                          range=[NO_STROKE_COLOR, STROKE_COLOR])

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & CLEAN DATA
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data
def load_data():
    df = pd.read_csv("Cleaned_DataSet_Stroke.csv")

    df["gender"]         = df["gender"].astype(str).str.strip()
    df["smoking_status"] = df["smoking_status"].astype(str).str.strip()
    df["work_type"]      = df["work_type"].astype(str).str.strip()

    df["gender"] = df["gender"].replace({"0":"Female","1":"Male",0:"Female",1:"Male"})
    df["hypertension"]  = df["hypertension"].replace({0:"No", 1:"Yes"})
    df["heart_disease"] = df["heart_disease"].replace({0:"No", 1:"Yes"})
    df["stroke"]        = df["stroke"].replace({0:"No Stroke", 1:"Stroke"})

    df["smoking_status"] = df["smoking_status"].replace({
        "never smoked":"Never Smoked","formerly smoked":"Former Smoker",
        "smokes":"Smoker","Unknown":"Unknown","0":"Unknown",0:"Unknown"
    })
    df["work_type"] = df["work_type"].replace({
        "Private":"Private Sector","Self-employed":"Self Employed",
        "Govt_job":"Government Job","children":"Children","Never_worked":"Never Worked"
    })

    # Feature engineering (from notebook)
    df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
    df["high_glucose"] = (df["avg_glucose_level"] > 140).map(
        {True: "High (>140)", False: "Normal (≤140)"})
    df["bmi_category"] = pd.cut(
        df["bmi"], bins=[0, 18.5, 25, 30, 200],
        labels=["Underweight", "Normal", "Overweight", "Obese"]
    )
    df["age_group"] = pd.cut(
        df["age"], bins=[0, 30, 45, 60, 75, 200],
        labels=["0–30", "30–45", "45–60", "60–75", "75+"]
    )
    df["cardio_risk"] = (
        df["hypertension"].map({"Yes":1,"No":0}) +
        df["heart_disease"].map({"Yes":1,"No":0})
    )
    return df

df = load_data()

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR FILTERS
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🔍 Filters")
    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

    gender        = st.selectbox("Gender",    ["All"] + sorted(df["gender"].dropna().unique().tolist()))
    smoking       = st.selectbox("Smoking",   ["All"] + sorted(df["smoking_status"].dropna().unique().tolist()))
    work          = st.selectbox("Work Type", ["All"] + sorted(df["work_type"].dropna().unique().tolist()))
    stroke_filter = st.selectbox("Outcome",   ["All", "Stroke", "No Stroke"])

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    age_range = st.slider("Age Range",     int(df.age.min()), int(df.age.max()), (20, 80))
    bmi_range = st.slider("BMI Range",     int(df.bmi.min()), int(df.bmi.max()), (15, 40))
    glu_range = st.slider("Glucose Range",
                           int(df.avg_glucose_level.min()),
                           int(df.avg_glucose_level.max()), (55, 272))

    st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
    st.caption("Stroke Risk Dashboard · CS Project")

# ─────────────────────────────────────────────────────────────────────────────
# APPLY FILTERS
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
# PAGE HEADER
# ─────────────────────────────────────────────────────────────────────────────
lc, rc = st.columns([3, 1])
with lc:
    st.markdown("<div class='dash-title'>🧠 Stroke Risk Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-sub'>Exploratory & Predictive Analysis · 5,110 Patients</div>",
                unsafe_allow_html=True)
with rc:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(
        f"<div style='text-align:right;color:#6B7280;font-size:0.82rem'>"
        f"Showing <b style='color:#F87171'>{total:,}</b> of {len(df):,} patients</div>",
        unsafe_allow_html=True
    )

st.markdown("<div class='divider'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# KPI ROW
# ─────────────────────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5, k6 = st.columns(6)
kpi_data = [
    (k1, f"{total:,}",                              "Total Patients",  ""),
    (k2, f"{fdf['age'].mean():.1f}",                "Avg Age (yrs)",   ""),
    (k3, f"{fdf['bmi'].mean():.1f}",                "Avg BMI",         ""),
    (k4, f"{fdf['avg_glucose_level'].mean():.1f}",  "Avg Glucose",     ""),
    (k5, f"{stroke_count:,}",                       "Stroke Cases",    ""),
    (k6, f"{stroke_rate}%",                         "Stroke Rate",
         "⚠️ class imbalance" if stroke_rate < 10 else ""),
]
for col, val, label, note in kpi_data:
    note_html = f"<div class='kpi-note'>{note}</div>" if note else ""
    col.markdown(f"""
    <div class='kpi-card'>
        <div class='kpi-value'>{val}</div>
        <div class='kpi-label'>{label}</div>
        {note_html}
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Population", "❤️ Risk Factors",
    "📈 Distributions", "🔗 Correlations", "🤖 Model Results"
])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — POPULATION
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown("<div class='section-header'>Demographic Breakdown</div>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)

    with c1:
        gdf = fdf["gender"].value_counts().reset_index()
        gdf.columns = ["Gender", "Count"]
        ch = alt.Chart(gdf, title="Gender Split").mark_arc(innerRadius=55).encode(
            theta=alt.Theta("Count:Q"),
            color=alt.Color("Gender:N", scale=alt.Scale(
                domain=["Female", "Male"], range=["#F87171", "#60A5FA"])),
            tooltip=["Gender", "Count"]
        ).properties(height=220)
        st.altair_chart(ch, use_container_width=True)

    with c2:
        smk = fdf["smoking_status"].value_counts().reset_index()
        smk.columns = ["Status", "Count"]
        ch = alt.Chart(smk, title="Smoking Status").mark_bar(
            cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#FB923C"
        ).encode(
            x=alt.X("Count:Q", title="Patients"),
            y=alt.Y("Status:N", sort="-x", title=""),
            tooltip=["Status", "Count"]
        ).properties(height=220)
        st.altair_chart(ch, use_container_width=True)

    with c3:
        wt = fdf["work_type"].value_counts().reset_index()
        wt.columns = ["Type", "Count"]
        ch = alt.Chart(wt, title="Work Type").mark_bar(
            cornerRadiusTopRight=4, cornerRadiusBottomRight=4, color="#A78BFA"
        ).encode(
            x=alt.X("Count:Q", title="Patients"),
            y=alt.Y("Type:N", sort="-x", title=""),
            tooltip=["Type", "Count"]
        ).properties(height=220)
        st.altair_chart(ch, use_container_width=True)

    # Stroke rate by age group
    st.markdown("<div class='section-header'>Stroke Rate by Age Group</div>", unsafe_allow_html=True)
    age_s = (
        fdf.groupby("age_group", observed=True)
        .apply(lambda x: pd.Series({
            "Stroke Rate (%)": round((x["stroke"] == "Stroke").mean() * 100, 2),
            "N": len(x)
        }))
        .reset_index()
    )
    age_s["age_group"] = age_s["age_group"].astype(str)

    ch = alt.Chart(age_s, title="Stroke Prevalence Across Age Groups").mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X("age_group:N",
                sort=["0–30","30–45","45–60","60–75","75+"], title="Age Group"),
        y=alt.Y("Stroke Rate (%):Q"),
        color=alt.Color("Stroke Rate (%):Q",
                         scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=["age_group", "Stroke Rate (%)", "N"]
    ).properties(height=280)
    st.altair_chart(ch, use_container_width=True)

    st.markdown("""<div class='insight-box'>💡 <b>Key Insight:</b> Stroke risk climbs steeply after age 45.
    The 60–75 and 75+ groups account for the vast majority of cases — consistent with clinical epidemiology.
    </div>""", unsafe_allow_html=True)

    c1b, c2b = st.columns(2)

    with c1b:
        bmi_s = (fdf.groupby(["bmi_category","stroke"], observed=True)
                    .size().reset_index(name="Count"))
        bmi_s["bmi_category"] = bmi_s["bmi_category"].astype(str)
        ch = alt.Chart(bmi_s, title="Stroke by BMI Category").mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3
        ).encode(
            x=alt.X("bmi_category:N",
                    sort=["Underweight","Normal","Overweight","Obese"], title="BMI Category"),
            y=alt.Y("Count:Q"),
            color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
            xOffset="stroke:N",
            tooltip=["bmi_category","stroke","Count"]
        ).properties(height=260)
        st.altair_chart(ch, use_container_width=True)

    with c2b:
        glu_s = (fdf.groupby(["high_glucose","stroke"], observed=True)
                    .size().reset_index(name="Count"))
        ch = alt.Chart(glu_s, title="Stroke by Glucose Level").mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3
        ).encode(
            x=alt.X("high_glucose:N", title="Glucose Level"),
            y=alt.Y("Count:Q"),
            color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
            xOffset="stroke:N",
            tooltip=["high_glucose","stroke","Count"]
        ).properties(height=260)
        st.altair_chart(ch, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — RISK FACTORS
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("<div class='section-header'>Clinical Risk Factors vs Stroke</div>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    def grouped_bar(col, title, container):
        tmp = fdf.groupby([col,"stroke"]).size().reset_index(name="Count")
        ch = alt.Chart(tmp, title=title).mark_bar(
            cornerRadiusTopLeft=3, cornerRadiusTopRight=3
        ).encode(
            x=alt.X(f"{col}:N", title=col),
            y=alt.Y("Count:Q"),
            color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
            xOffset="stroke:N",
            tooltip=[col, "stroke", "Count"]
        ).properties(height=260)
        container.altair_chart(ch, use_container_width=True)

    grouped_bar("hypertension",  "Hypertension vs Stroke",  c1)
    grouped_bar("heart_disease", "Heart Disease vs Stroke",  c2)

    st.markdown("<div class='section-header'>Stroke Rate by Risk Group (%)</div>", unsafe_allow_html=True)

    rows = []
    for col, val, label in [
        ("hypertension","Yes","Hypertension"),
        ("hypertension","No","No Hypertension"),
        ("heart_disease","Yes","Heart Disease"),
        ("heart_disease","No","No Heart Disease"),
        ("high_glucose","High (>140)","High Glucose (>140)"),
        ("high_glucose","Normal (≤140)","Normal Glucose (≤140)"),
        ("smoking_status","Smoker","Active Smoker"),
        ("smoking_status","Former Smoker","Former Smoker"),
        ("smoking_status","Never Smoked","Never Smoked"),
    ]:
        sub = fdf[fdf[col] == val]
        if len(sub) > 0:
            rows.append({
                "Group": label,
                "Stroke Rate": round((sub["stroke"] == "Stroke").mean() * 100, 2),
                "N": len(sub)
            })

    risk_df = pd.DataFrame(rows)
    bars = alt.Chart(risk_df, title="Stroke Rate by Risk Group (%)").mark_bar(
        cornerRadiusTopRight=4, cornerRadiusBottomRight=4
    ).encode(
        x=alt.X("Stroke Rate:Q", title="Stroke Rate (%)"),
        y=alt.Y("Group:N", sort="-x", title=""),
        color=alt.Color("Stroke Rate:Q", scale=alt.Scale(scheme="reds"), legend=None),
        tooltip=["Group", "Stroke Rate", "N"]
    ).properties(height=300)
    labels = bars.mark_text(align="left", dx=4, color="#9CA3AF", fontSize=11).encode(
        text=alt.Text("Stroke Rate:Q", format=".1f")
    )
    st.altair_chart(bars + labels, use_container_width=True)

    # Scatter: age vs glucose
    st.markdown("<div class='section-header'>Age & Glucose — Individual Patients</div>",
                unsafe_allow_html=True)
    sample = fdf.sample(min(1000, len(fdf)), random_state=42)
    ch = alt.Chart(sample, title="Age vs Glucose Level (random 1,000 sample)").mark_circle(
        opacity=0.55, size=35
    ).encode(
        x=alt.X("age:Q", title="Age"),
        y=alt.Y("avg_glucose_level:Q", title="Avg Glucose (mg/dL)"),
        color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
        tooltip=["age","avg_glucose_level","bmi","hypertension","heart_disease","stroke"]
    ).properties(height=340).interactive()
    st.altair_chart(ch, use_container_width=True)

    st.markdown("""<div class='insight-box'>💡 <b>Key Insight:</b> Stroke patients cluster toward
    the older + higher-glucose region. Hypertension and heart disease both roughly double stroke rate.
    High glucose (&gt;140 mg/dL) is a strong independent predictor.</div>""",
    unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 3 — DISTRIBUTIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown("<div class='section-header'>Health Metric Distributions</div>", unsafe_allow_html=True)

    metric    = st.radio("Select metric", ["Age","BMI","Glucose Level"], horizontal=True)
    col_map   = {"Age":"age","BMI":"bmi","Glucose Level":"avg_glucose_level"}
    chosen    = col_map[metric]

    ch = alt.Chart(fdf, title=f"{metric} Distribution by Stroke Outcome").mark_bar(
        opacity=0.65, binSpacing=1
    ).encode(
        x=alt.X(f"{chosen}:Q", bin=alt.Bin(maxbins=40), title=metric),
        y=alt.Y("count()", stack=None, title="Patients"),
        color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome"),
        tooltip=[f"{chosen}:Q","count()","stroke:N"]
    ).properties(height=300)
    st.altair_chart(ch, use_container_width=True)

    c1, c2 = st.columns(2)

    with c1:
        ch_box = alt.Chart(fdf, title=f"{metric} — Box Plot").mark_boxplot(
            extent="min-max", outliers={"size":15,"opacity":0.3}
        ).encode(
            x=alt.X("stroke:N", title="Outcome"),
            y=alt.Y(f"{chosen}:Q", title=metric),
            color=alt.Color("stroke:N", scale=STROKE_SCALE, legend=None)
        ).properties(height=300)
        st.altair_chart(ch_box, use_container_width=True)

    with c2:
        age_mean = fdf.groupby("age")[chosen].mean().rename("mean_val").reset_index()
        line = alt.Chart(age_mean, title=f"Mean {metric} by Age").mark_line(
            strokeWidth=2.5, color="#FBBF24"
        ).encode(
            x=alt.X("age:Q", title="Age"),
            y=alt.Y("mean_val:Q", title=f"Mean {metric}"),
            tooltip=["age","mean_val"]
        ).properties(height=300)

        if chosen == "avg_glucose_level":
            rule = alt.Chart(pd.DataFrame({"y":[140]})).mark_rule(
                color="#F87171", strokeDash=[6,3], strokeWidth=1.5
            ).encode(y="y:Q")
            st.altair_chart(line + rule, use_container_width=True)
        else:
            st.altair_chart(line, use_container_width=True)

    # Glucose trend full-width
    st.markdown("<div class='section-header'>Mean Glucose Level Across Age</div>",
                unsafe_allow_html=True)
    agl = fdf.groupby("age")["avg_glucose_level"].mean().rename("Avg Glucose").reset_index()

    trend_line = alt.Chart(agl).mark_line(strokeWidth=2.5, color="#FBBF24").encode(
        x=alt.X("age:Q", title="Age"),
        y=alt.Y("Avg Glucose:Q", title="Mean Glucose (mg/dL)"),
        tooltip=["age","Avg Glucose"]
    )
    threshold = alt.Chart(pd.DataFrame({"y":[140]})).mark_rule(
        color="#F87171", strokeDash=[6,3]
    ).encode(y="y:Q")
    label = alt.Chart(pd.DataFrame({"y":[140],"t":["High Glucose Threshold (140 mg/dL)"]})).mark_text(
        align="right", dx=-8, dy=-8, color="#F87171", fontSize=11
    ).encode(y="y:Q", text="t:N")

    st.altair_chart(
        (trend_line + threshold + label).properties(height=280),
        use_container_width=True
    )

# ═══════════════════════════════════════════════════════════════════════════
# TAB 4 — CORRELATIONS
# ═══════════════════════════════════════════════════════════════════════════
with tab4:
    st.markdown("<div class='section-header'>Correlation Matrix</div>", unsafe_allow_html=True)

    num_df = fdf.select_dtypes(include=["int64","float64"]).copy()
    num_df["stroke_num"]        = (fdf["stroke"]        == "Stroke").astype(int)
    num_df["hypertension_num"]  = (fdf["hypertension"]  == "Yes").astype(int)
    num_df["heart_disease_num"] = (fdf["heart_disease"] == "Yes").astype(int)

    keep = [c for c in ["age","bmi","avg_glucose_level",
                         "hypertension_num","heart_disease_num",
                         "cardio_risk","stroke_num"] if c in num_df.columns]
    corr = num_df[keep].corr().reset_index().melt("index")
    corr.columns = ["var1","var2","correlation"]

    heatmap = alt.Chart(corr, title="Feature Correlation Heatmap").mark_rect().encode(
        x=alt.X("var1:N", title=""),
        y=alt.Y("var2:N", title=""),
        color=alt.Color("correlation:Q",
                         scale=alt.Scale(scheme="redblue", domain=[-1,1], reverse=True),
                         title="r"),
        tooltip=["var1","var2",alt.Tooltip("correlation:Q",format=".3f")]
    ).properties(height=360)
    text_layer = heatmap.mark_text(fontSize=11, fontWeight=500).encode(
        text=alt.Text("correlation:Q", format=".2f"),
        color=alt.condition(
            alt.datum.correlation > 0.4,
            alt.value("white"), alt.value("#9CA3AF")
        )
    )
    st.altair_chart(heatmap + text_layer, use_container_width=True)

    st.markdown("""<div class='insight-box'>💡 <b>Reading this:</b>
    The <b>stroke_num</b> row shows the strongest predictors of stroke.
    <b>age</b> has the highest positive correlation. Values near ±1 = strong linear relationship;
    near 0 = weak.</div>""", unsafe_allow_html=True)

    # Scatter matrix
    st.markdown("<div class='section-header'>Age × BMI × Glucose — Scatter Matrix</div>",
                unsafe_allow_html=True)
    smpl = fdf.sample(min(800, len(fdf)), random_state=1)[
        ["age","bmi","avg_glucose_level","stroke"]
    ]
    ch = alt.Chart(smpl).mark_circle(size=25, opacity=0.5).encode(
        alt.X(alt.repeat("column"), type="quantitative"),
        alt.Y(alt.repeat("row"),    type="quantitative"),
        color=alt.Color("stroke:N", scale=STROKE_SCALE, title="Outcome")
    ).repeat(
        row=["age","bmi","avg_glucose_level"],
        column=["avg_glucose_level","bmi","age"]
    ).properties(title="Pairwise Scatter Matrix (800 random patients)")
    st.altair_chart(ch, use_container_width=True)

    # Data preview
    st.markdown("<div class='section-header'>Data Preview</div>", unsafe_allow_html=True)
    cols_show = ["age","gender","hypertension","heart_disease","bmi",
                 "avg_glucose_level","smoking_status","work_type","stroke"]
    st.dataframe(fdf[cols_show].head(30).reset_index(drop=True),
                 use_container_width=True, height=320)

# ═══════════════════════════════════════════════════════════════════════════
# TAB 5 — ML MODEL RESULTS
# ═══════════════════════════════════════════════════════════════════════════
with tab5:
    st.markdown("<div class='section-header'>ML Model Performance — Results from Notebook</div>",
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
    best_name = model_data.loc[best_idx, "Model"]

    st.markdown(f"""<div class='insight-box'>
    🏆 <b>Best for Medical Screening (highest Recall):</b>
    <b style='color:#F87171'>{best_name}</b>
    &nbsp;—&nbsp; Recall <b>{model_data.loc[best_idx,'Recall']:.1%}</b>
    &nbsp;·&nbsp; F1 <b>{model_data.loc[best_idx,'F1-Score']:.1%}</b><br><br>
    In stroke detection, <b>Recall matters most</b> — missing a real stroke is far more costly than
    a false alarm. Both Logistic Regression variants scored 0 on the stroke class due to severe class
    imbalance (~4.9% positive rate). XGBoost used <code>scale_pos_weight = 20.86</code> to partially
    compensate.
    </div>""", unsafe_allow_html=True)

    metric_choice = st.radio(
        "Metric to display", ["Precision","Recall","F1-Score"], horizontal=True
    )

    bar_data = model_data[["Model", metric_choice]].copy()
    bar_data["Highlight"] = bar_data["Model"] == best_name

    ch = alt.Chart(bar_data, title=f"{metric_choice} — Stroke Class (all models)").mark_bar(
        cornerRadiusTopLeft=4, cornerRadiusTopRight=4
    ).encode(
        x=alt.X("Model:N", sort=None, title="",
                axis=alt.Axis(labelAngle=-35, labelLimit=140)),
        y=alt.Y(f"{metric_choice}:Q", title=metric_choice,
                scale=alt.Scale(domain=[0, 0.85])),
        color=alt.condition(
            alt.datum.Highlight,
            alt.value("#F87171"),
            alt.value("#3B4F6B")
        ),
        tooltip=["Model", alt.Tooltip(f"{metric_choice}:Q", format=".1%")]
    ).properties(height=320)
    text_lbl = ch.mark_text(dy=-8, color="#9CA3AF", fontSize=11).encode(
        text=alt.Text(f"{metric_choice}:Q", format=".1%")
    )
    st.altair_chart(ch + text_lbl, use_container_width=True)

    # All 3 metrics grouped
    st.markdown("<div class='section-header'>All Metrics — Side by Side</div>",
                unsafe_allow_html=True)
    melted = model_data.melt(
        id_vars="Model",
        value_vars=["Precision","Recall","F1-Score"],
        var_name="Metric", value_name="Score"
    )
    ch_all = alt.Chart(melted, title="Model Comparison — Precision / Recall / F1").mark_bar(
        cornerRadiusTopLeft=3, cornerRadiusTopRight=3
    ).encode(
        x=alt.X("Model:N", sort=None, title="",
                axis=alt.Axis(labelAngle=-35, labelLimit=140)),
        y=alt.Y("Score:Q", scale=alt.Scale(domain=[0, 0.85])),
        color=alt.Color("Metric:N",
                         scale=alt.Scale(
                             domain=["Precision","Recall","F1-Score"],
                             range=["#60A5FA","#F87171","#FBBF24"]
                         ), title="Metric"),
        xOffset="Metric:N",
        tooltip=["Model","Metric",alt.Tooltip("Score:Q",format=".1%")]
    ).properties(height=340)
    st.altair_chart(ch_all, use_container_width=True)

    # Full results table
    st.markdown("<div class='section-header'>Full Results Table</div>", unsafe_allow_html=True)
    disp = model_data.copy()
    for c in ["Accuracy","Precision","Recall","F1-Score"]:
        disp[c] = disp[c].map(lambda x: f"{x:.1%}")
    st.dataframe(disp.set_index("Model"), use_container_width=True, height=360)

    st.markdown("""<div class='insight-box'>
    📌 <b>Feature Engineering Impact:</b> Adding clinical features (high-glucose flag, cardio risk score,
    BMI category, age group) did <em>not</em> significantly improve simpler models. The bottleneck was
    class imbalance, not feature richness. Techniques like <b>SMOTE</b> or cost-sensitive learning
    would likely yield larger gains than additional features alone.
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("<div class='divider'></div>", unsafe_allow_html=True)
st.markdown("""<div style='text-align:center;color:#374151;font-size:0.78rem;padding:8px 0 20px'>
    Stroke Risk Dashboard · Streamlit + Altair · 5,110 patients
</div>""", unsafe_allow_html=True)
