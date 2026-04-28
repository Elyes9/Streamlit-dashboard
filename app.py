import streamlit as st
import pandas as pd
import numpy as np
import altair as alt

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Stroke Risk Dashboard", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

# ─────────────────────────────────────────────────────────────────────────────
# STYLE
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@600;700&family=Inter:wght@300;400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Inter',sans-serif;color:#E2E8F0;}
.main{background:#080C14;}
section[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1422,#0A1020);border-right:1px solid #1A2540;}
#MainMenu,footer,header{visibility:hidden;}

.upload-zone{background:linear-gradient(135deg,#0D1830,#111E38);border:2px dashed #2A3F6F;
  border-radius:20px;padding:52px 40px;text-align:center;margin:40px auto;max-width:680px;}
.upload-title{font-family:'Playfair Display',serif;font-size:2.2rem;
  background:linear-gradient(135deg,#F87171,#FB923C,#FBBF24);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:8px;}
.upload-sub{color:#64748B;font-size:.92rem;margin-bottom:24px;}

.dash-title{font-family:'Playfair Display',serif;font-size:2.4rem;
  background:linear-gradient(135deg,#F87171 0%,#FB923C 50%,#FBBF24 100%);
  -webkit-background-clip:text;-webkit-text-fill-color:transparent;line-height:1.15;margin-bottom:2px;}
.dash-sub{color:#475569;font-size:.82rem;letter-spacing:.1em;text-transform:uppercase;font-weight:500;}

.kpi-wrap{background:linear-gradient(145deg,#0D1830,#111E38);border:1px solid #1A2D50;
  border-radius:16px;padding:20px 16px 16px;text-align:center;position:relative;
  overflow:hidden;transition:transform .2s,border-color .2s;}
.kpi-wrap::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;
  background:linear-gradient(90deg,#F87171,#FBBF24);border-radius:16px 16px 0 0;}
.kpi-wrap:hover{transform:translateY(-2px);border-color:#F87171;}
.kpi-val{font-family:'Playfair Display',serif;font-size:2rem;color:#F87171;line-height:1;}
.kpi-lbl{color:#64748B;font-size:.7rem;text-transform:uppercase;letter-spacing:.1em;margin-top:6px;}
.kpi-note{color:#6EE7B7;font-size:.68rem;margin-top:5px;}

.sec-hdr{font-family:'Playfair Display',serif;font-size:1.2rem;color:#E2E8F0;
  display:flex;align-items:center;gap:10px;margin:28px 0 12px;}
.sec-hdr::before{content:'';display:inline-block;width:4px;height:22px;
  background:linear-gradient(180deg,#F87171,#FBBF24);border-radius:2px;flex-shrink:0;}

.ibox{background:#0D1830;border:1px solid #1A2D50;border-left:4px solid #F87171;
  border-radius:10px;padding:14px 18px;margin:12px 0;font-size:.86rem;color:#94A3B8;line-height:1.7;}
.ibox-teal{border-left-color:#34D399;}
.ibox-gold{border-left-color:#FBBF24;}
.ibox-blue{border-left-color:#60A5FA;}
.ibox-purple{border-left-color:#A78BFA;}

.model-best{background:#1A0F0F;border:1px solid #F87171;border-radius:8px;
  padding:12px 16px;margin:8px 0;font-size:.84rem;color:#FCA5A5;}

.hr{border-top:1px solid #1A2540;margin:20px 0;}
[data-testid="stFileUploadDropzone"]{background:#0D1830 !important;
  border:1.5px dashed #2A3F6F !important;border-radius:12px !important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# ALTAIR THEME
# ─────────────────────────────────────────────────────────────────────────────
RED    = "#F87171"; BLUE   = "#60A5FA"; GOLD   = "#FBBF24"
TEAL   = "#34D399"; PURPLE = "#A78BFA"; ORANGE = "#FB923C"

def dark_theme():
    return {"config":{
        "background":"#0D1422","view":{"stroke":"transparent","fill":"#0D1422"},
        "axis":{"domainColor":"#1E2D4A","gridColor":"#131F35","labelColor":"#64748B",
                "titleColor":"#64748B","tickColor":"#1E2D4A","labelFontSize":11,"titleFontSize":12},
        "legend":{"labelColor":"#94A3B8","titleColor":"#94A3B8","fillColor":"#0D1422","strokeColor":"#1E2D4A"},
        "title":{"color":"#E2E8F0","fontSize":13,"fontWeight":500,"anchor":"start"},
        "mark":{"tooltip":True},
    }}
alt.themes.register("dark", dark_theme)
alt.themes.enable("dark")
STROKE_SCALE = alt.Scale(domain=["No Stroke","Stroke"], range=[BLUE, RED])

# ─────────────────────────────────────────────────────────────────────────────
# LABEL MAPS  (exact encodings from notebook)
# gender: Male=0, Female=1
# work_type: Private=0, Self-employed=1, children=2, Govt_job=3, Never_worked=4
# smoking_status: never_smoked=0, formerly_smoked=1, smokes=2, Unknown=3
# Residence_type: Rural=0, Urban=1
# ever_married: No=0, Yes=1
# age_group: [0-30]=0,[30-45]=1,[45-60]=2,[60-75]=3,[75+]=4
# bmi_category: underweight=0,normal=1,overweight=2,obese=3
# ─────────────────────────────────────────────────────────────────────────────
GENDER_MAP    = {0:"Male",          1:"Female"}
WORK_MAP      = {0:"Private",       1:"Self-Employed", 2:"Children",
                 3:"Government",    4:"Never Worked"}
SMOKING_MAP   = {0:"Never Smoked",  1:"Former Smoker", 2:"Smoker",    3:"Unknown",
                 2.0:"Smoker",      3.0:"Unknown",     0.0:"Never Smoked", 1.0:"Former Smoker"}
RESIDENCE_MAP = {0:"Rural",         1:"Urban"}
MARRIED_MAP   = {0:"No",            1:"Yes"}
CARDIO_MAP    = {0:"None",          1:"One Condition", 2:"Both Conditions"}
AGE_GRP_MAP   = {0:"0–30",          1:"30–45",         2:"45–60", 3:"60–75", 4:"75+"}
BMI_CAT_MAP   = {0:"Underweight",   1:"Normal",        2:"Overweight",    3:"Obese"}
STROKE_MAP    = {0:"No Stroke",     1:"Stroke"}
HI_GLU_MAP    = {0:"Normal (≤140)", 1:"High (>140)"}

# ─────────────────────────────────────────────────────────────────────────────
# SAFE HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def gb_mean(df, x, y):
    g = df.groupby(x)[y].mean()
    return pd.DataFrame({x: g.index.tolist(), y: g.values.tolist()})

def smooth(df, x, y, w=7):
    t = gb_mean(df, x, y).sort_values(x)
    t[y] = t[y].rolling(w, center=True, min_periods=1).mean()
    return t.reset_index(drop=True)

def band(df, x, y, w=7):
    g = df.groupby(x)[y]
    m, s = g.mean(), g.std().fillna(0)
    t = pd.DataFrame({x: m.index.tolist(), "m": m.values.tolist(), "s": s.values.tolist()}).sort_values(x)
    t["lo"] = (t["m"]-t["s"]*.5).rolling(w, center=True, min_periods=1).mean()
    t["hi"] = (t["m"]+t["s"]*.5).rolling(w, center=True, min_periods=1).mean()
    return t[[x,"lo","hi"]].reset_index(drop=True)

# ─────────────────────────────────────────────────────────────────────────────
# LOAD & LABEL
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load(raw):
    import io
    df = pd.read_csv(io.BytesIO(raw))
    df["bmi"] = df["bmi"].fillna(df["bmi"].mean())
    df["gender_lbl"]   = df["gender"].map(GENDER_MAP).fillna("Unknown")
    df["work_lbl"]     = df["work_type"].map(WORK_MAP).fillna("Unknown")
    df["smoking_lbl"]  = df["smoking_status"].map(SMOKING_MAP).fillna("Unknown")
    df["res_lbl"]      = df["Residence_type"].map(RESIDENCE_MAP).fillna("Unknown")
    df["married_lbl"]  = df["ever_married"].map(MARRIED_MAP).fillna("Unknown")
    df["cardio_lbl"]   = df["cardio_risk"].map(CARDIO_MAP).fillna("Unknown")
    df["ag_lbl"]       = df["age_group"].map(AGE_GRP_MAP).fillna("Unknown")
    df["bmi_lbl"]      = df["bmi_category"].map(BMI_CAT_MAP).fillna("Unknown")
    df["stroke_lbl"]   = df["stroke"].map(STROKE_MAP).fillna("Unknown")
    df["glucose_lbl"]  = df["high_glucose"].map(HI_GLU_MAP).fillna("Unknown")
    df["hyp_lbl"]      = df["hypertension"].map({0:"No",1:"Yes"}).fillna("Unknown")
    df["hd_lbl"]       = df["heart_disease"].map({0:"No",1:"Yes"}).fillna("Unknown")
    return df

# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD SCREEN
# ─────────────────────────────────────────────────────────────────────────────
if "df" not in st.session_state:
    _, mid, _ = st.columns([1, 2.4, 1])
    with mid:
        st.markdown("""
        <div class='upload-zone'>
          <div class='upload-title'>🧠 Stroke Risk Dashboard</div>
          <div class='upload-sub'>Upload <b>Final_Cleaned_DataSet.csv</b> to unlock all charts</div>
        </div>""", unsafe_allow_html=True)
        up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
        if up:
            with st.spinner("Loading…"):
                st.session_state["df"] = load(up.read())
            st.rerun()
        st.markdown("""<div style='margin-top:14px;color:#334155;font-size:.8rem;line-height:1.9'>
          Columns used: <code>gender</code> · <code>age</code> · <code>hypertension</code> ·
          <code>heart_disease</code> · <code>ever_married</code> · <code>work_type</code> ·
          <code>Residence_type</code> · <code>avg_glucose_level</code> · <code>bmi</code> ·
          <code>smoking_status</code> · <code>stroke</code> · <code>high_glucose</code> ·
          <code>cardio_risk</code> · <code>age_group</code> · <code>bmi_category</code> ·
          <code>age_hypertension</code> · <code>age_glucose</code>
        </div>""", unsafe_allow_html=True)
    st.stop()

df = st.session_state["df"]

# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("↩  Upload new file", use_container_width=True):
        del st.session_state["df"]; st.rerun()
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.markdown("### 🔍 Filters")
    sg  = st.selectbox("Gender",          ["All"]+sorted(df["gender_lbl"].dropna().unique().tolist()))
    sw  = st.selectbox("Work Type",       ["All"]+sorted(df["work_lbl"].dropna().unique().tolist()))
    ssk = st.selectbox("Smoking Status",  ["All"]+sorted(df["smoking_lbl"].dropna().unique().tolist()))
    sr  = st.selectbox("Residence",       ["All"]+sorted(df["res_lbl"].dropna().unique().tolist()))
    so  = st.selectbox("Outcome",         ["All","Stroke","No Stroke"])
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    ar  = st.slider("Age",     int(df.age.min()),               int(df.age.max()),               (0,82))
    br  = st.slider("BMI",     int(df.bmi.min()),               int(df.bmi.max()),               (10,50))
    gr  = st.slider("Glucose", int(df.avg_glucose_level.min()), int(df.avg_glucose_level.max()), (55,272))
    st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
    st.caption(f"Dataset · {len(df):,} patients · 17 features")

# ── FILTER ────────────────────────────────────────────────────────────────────
fdf = df.copy()
if sg  != "All": fdf = fdf[fdf["gender_lbl"]  == sg]
if sw  != "All": fdf = fdf[fdf["work_lbl"]     == sw]
if ssk != "All": fdf = fdf[fdf["smoking_lbl"]  == ssk]
if sr  != "All": fdf = fdf[fdf["res_lbl"]      == sr]
if so  != "All": fdf = fdf[fdf["stroke_lbl"]   == so]
fdf = fdf[fdf["age"].between(*ar) & fdf["bmi"].between(*br) & fdf["avg_glucose_level"].between(*gr)]

N  = len(fdf)
SC = (fdf["stroke_lbl"]=="Stroke").sum()
SR = round(SC/N*100,2) if N>0 else 0

# ─────────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────────
hc, rc = st.columns([3,1])
with hc:
    st.markdown("<div class='dash-title'>🧠 Stroke Risk Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='dash-sub'>Exploratory & Predictive Analysis · 5,109 Patients · 17 Features</div>", unsafe_allow_html=True)
with rc:
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"<div style='text-align:right;color:#334155;font-size:.82rem'>"
                f"Showing <b style='color:#F87171'>{N:,}</b> of <b style='color:#60A5FA'>{len(df):,}</b></div>",
                unsafe_allow_html=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ── KPIs ──────────────────────────────────────────────────────────────────────
cols8 = st.columns(8)
kpis = [
    (f"{N:,}",                                    "Patients",       ""),
    (f"{fdf['age'].mean():.1f}",                  "Avg Age",        "yrs"),
    (f"{fdf['bmi'].mean():.1f}",                  "Avg BMI",        ""),
    (f"{fdf['avg_glucose_level'].mean():.1f}",    "Avg Glucose",    "mg/dL"),
    (f"{SC:,}",                                   "Stroke Cases",   ""),
    (f"{SR}%",                                    "Stroke Rate",    "⚠ imbalanced" if SR<10 else ""),
    (f"{round((fdf['hypertension']==1).mean()*100,1)}%", "Hypertension", "of cohort"),
    (f"{round((fdf['ever_married']==1).mean()*100,1)}%", "Ever Married",  "of cohort"),
]
for col, (v,l,n) in zip(cols8, kpis):
    nh = f"<div class='kpi-note'>{n}</div>" if n else ""
    col.markdown(f"<div class='kpi-wrap'><div class='kpi-val'>{v}</div><div class='kpi-lbl'>{l}</div>{nh}</div>",
                 unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────────
t1,t2,t3,t4,t5,t6 = st.tabs([
    "📊 Demographics", "❤️ Risk Factors",
    "📈 Curves & Distributions", "🔬 Feature Interactions",
    "🔗 Correlations", "🤖 Model Results"])

# ═══ TAB 1 — DEMOGRAPHICS ════════════════════════════════════════════════════
with t1:
    st.markdown("<div class='sec-hdr'>Population Overview</div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)

    def donut(col, vals, title, clrs, container):
        d = fdf[col].value_counts().reset_index(); d.columns=["x","Count"]
        ch = alt.Chart(d,title=title).mark_arc(innerRadius=55,padAngle=.02,cornerRadius=4).encode(
            theta="Count:Q",
            color=alt.Color("x:N",scale=alt.Scale(domain=list(vals.values()),range=clrs),title=""),
            tooltip=["x","Count"]
        ).properties(height=210)
        container.altair_chart(ch, use_container_width=True)

    donut("gender_lbl", GENDER_MAP,  "Gender",           [RED,  BLUE],           c1)
    donut("res_lbl",    RESIDENCE_MAP,"Residence Type",  [TEAL, PURPLE],         c2)
    donut("married_lbl",MARRIED_MAP, "Ever Married",     [GOLD, "#475569"],      c3)
    donut("glucose_lbl",HI_GLU_MAP,  "Glucose Level",   [BLUE, RED],            c4)

    cb1, cb2 = st.columns(2)
    with cb1:
        st.markdown("<div class='sec-hdr'>Work Type Distribution</div>", unsafe_allow_html=True)
        wt = fdf["work_lbl"].value_counts().reset_index(); wt.columns=["Type","Count"]
        st.altair_chart(
            alt.Chart(wt,title="Patients by Work Type").mark_bar(cornerRadiusTopRight=5,cornerRadiusBottomRight=5,color=PURPLE)
            .encode(x=alt.X("Count:Q",title="Patients"),y=alt.Y("Type:N",sort="-x",title=""),tooltip=["Type","Count"])
            .properties(height=230), use_container_width=True)

    with cb2:
        st.markdown("<div class='sec-hdr'>Smoking Status Distribution</div>", unsafe_allow_html=True)
        smk = fdf["smoking_lbl"].value_counts().reset_index(); smk.columns=["Status","Count"]
        st.altair_chart(
            alt.Chart(smk,title="Patients by Smoking Status").mark_bar(cornerRadiusTopRight=5,cornerRadiusBottomRight=5,color=ORANGE)
            .encode(x=alt.X("Count:Q",title="Patients"),y=alt.Y("Status:N",sort="-x",title=""),tooltip=["Status","Count"])
            .properties(height=230), use_container_width=True)

    st.markdown("<div class='sec-hdr'>Stroke Rate by Age Group</div>", unsafe_allow_html=True)
    agr = []
    for code,lbl in AGE_GRP_MAP.items():
        sub = fdf[fdf["age_group"]==code]
        if len(sub)>0:
            agr.append({"Age Group":lbl,
                        "Stroke Rate (%)":round((sub["stroke_lbl"]=="Stroke").mean()*100,2),
                        "N":len(sub)})
    ags = pd.DataFrame(agr)
    bars = alt.Chart(ags,title="Stroke Prevalence Across Age Groups").mark_bar(cornerRadiusTopLeft=5,cornerRadiusTopRight=5).encode(
        x=alt.X("Age Group:N",sort=["0–30","30–45","45–60","60–75","75+"],title=""),
        y="Stroke Rate (%):Q",
        color=alt.Color("Stroke Rate (%):Q",scale=alt.Scale(scheme="reds"),legend=None),
        tooltip=["Age Group","Stroke Rate (%)","N"]
    ).properties(height=280)
    lm = bars.mark_text(dy=-8,fontSize=11,color="#94A3B8").encode(text=alt.Text("Stroke Rate (%):Q",format=".1f"))
    st.altair_chart(bars+lm, use_container_width=True)

    st.markdown("""<div class='ibox'>
    💡 <b>Demographics:</b> The cohort is <b>Female-majority (58.6%)</b> with mean age <b>43.2 yrs</b>.
    Urban/rural split is roughly 50/50. <b>65.6%</b> of patients have ever been married.
    Stroke risk is near zero below age 30, then rises steeply — the 75+ group has the highest prevalence.
    </div>""", unsafe_allow_html=True)

# ═══ TAB 2 — RISK FACTORS ════════════════════════════════════════════════════
with t2:
    st.markdown("<div class='sec-hdr'>Clinical Risk Factors vs Stroke</div>", unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)

    def grp_bar(lbl_col, title, cont):
        tmp = fdf.groupby([lbl_col,"stroke_lbl"]).size().reset_index(name="Count")
        ch = alt.Chart(tmp,title=title).mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
            x=alt.X(f"{lbl_col}:N",title=""), y="Count:Q",
            color=alt.Color("stroke_lbl:N",scale=STROKE_SCALE,title="Outcome"),
            xOffset="stroke_lbl:N", tooltip=[lbl_col,"stroke_lbl","Count"]
        ).properties(height=250)
        cont.altair_chart(ch, use_container_width=True)

    grp_bar("hyp_lbl",    "Hypertension vs Stroke",     c1)
    grp_bar("hd_lbl",     "Heart Disease vs Stroke",    c2)
    grp_bar("cardio_lbl", "Cardio Risk Score vs Stroke",c3)

    st.markdown("<div class='sec-hdr'>Stroke Rate by Risk Group (%)</div>", unsafe_allow_html=True)
    rr = []
    for col,val,lbl in [
        ("hyp_lbl","Yes","Hypertension"),("hyp_lbl","No","No Hypertension"),
        ("hd_lbl","Yes","Heart Disease"),("hd_lbl","No","No Heart Disease"),
        ("cardio_lbl","Both Conditions","Both Cardio Conditions"),
        ("cardio_lbl","One Condition","One Cardio Condition"),
        ("cardio_lbl","None","No Cardio Condition"),
        ("glucose_lbl","High (>140)","High Glucose (>140)"),
        ("glucose_lbl","Normal (≤140)","Normal Glucose (≤140)"),
        ("married_lbl","Yes","Ever Married"),("married_lbl","No","Never Married"),
        ("res_lbl","Urban","Urban Residence"),("res_lbl","Rural","Rural Residence"),
        ("smoking_lbl","Former Smoker","Former Smoker"),("smoking_lbl","Never Smoked","Never Smoked"),
        ("smoking_lbl","Smoker","Current Smoker"),
    ]:
        sub = fdf[fdf[col]==val]
        if len(sub)>0:
            rr.append({"Group":lbl,"Stroke Rate":round((sub["stroke_lbl"]=="Stroke").mean()*100,2),"N":len(sub)})
    rdf = pd.DataFrame(rr).sort_values("Stroke Rate",ascending=True)
    b = alt.Chart(rdf,title="Stroke Rate by Risk Group (%)").mark_bar(cornerRadiusTopRight=5,cornerRadiusBottomRight=5).encode(
        x=alt.X("Stroke Rate:Q",title="Stroke Rate (%)"),
        y=alt.Y("Group:N",sort="-x",title=""),
        color=alt.Color("Stroke Rate:Q",scale=alt.Scale(scheme="reds"),legend=None),
        tooltip=["Group","Stroke Rate","N"]
    ).properties(height=430)
    lb = b.mark_text(align="left",dx=4,color="#64748B",fontSize=11).encode(text=alt.Text("Stroke Rate:Q",format=".1f"))
    st.altair_chart(b+lb, use_container_width=True)

    # Scatter
    st.markdown("<div class='sec-hdr'>Age × Glucose — Patient Cloud</div>", unsafe_allow_html=True)
    smp = fdf.sample(min(1500,N),random_state=42)
    ch = alt.Chart(smp,title="Age vs Avg Glucose Level (up to 1,500 patients)").mark_circle(opacity=.5,size=32).encode(
        x=alt.X("age:Q",title="Age"),
        y=alt.Y("avg_glucose_level:Q",title="Avg Glucose (mg/dL)"),
        color=alt.Color("stroke_lbl:N",scale=STROKE_SCALE,title="Outcome"),
        tooltip=["age","avg_glucose_level","bmi_lbl","hyp_lbl","hd_lbl","stroke_lbl"]
    ).properties(height=360).interactive()
    st.altair_chart(ch, use_container_width=True)

    st.markdown("""<div class='ibox ibox-teal'>
    💡 <b>Key Risk Findings:</b><br>
    • Patients with <b>both hypertension and heart disease</b> (cardio_risk=2) face the highest stroke rate.<br>
    • <b>High glucose (&gt;140 mg/dL)</b> nearly doubles stroke probability vs normal glucose.<br>
    • <b>Former smokers</b> show slightly higher stroke rates than never-smokers — a confound with age.<br>
    • Stroke patients cluster clearly in the <b>older + high-glucose</b> quadrant of the scatter.
    </div>""", unsafe_allow_html=True)

# ═══ TAB 3 — CURVES & DISTRIBUTIONS ══════════════════════════════════════════
with t3:
    st.markdown("<div class='sec-hdr'>Metric Distribution by Outcome</div>", unsafe_allow_html=True)
    met = st.radio("Metric", ["Age","BMI","Glucose Level","Age × Glucose (Feature)"], horizontal=True)
    cmap = {"Age":"age","BMI":"bmi","Glucose Level":"avg_glucose_level","Age × Glucose (Feature)":"age_glucose"}
    ch_col = cmap[met]

    ch = alt.Chart(fdf,title=f"{met} Distribution by Outcome").mark_bar(opacity=.65,binSpacing=0).encode(
        x=alt.X(f"{ch_col}:Q",bin=alt.Bin(maxbins=45),title=met),
        y=alt.Y("count()",stack=None,title="Patients"),
        color=alt.Color("stroke_lbl:N",scale=STROKE_SCALE,title="Outcome"),
        tooltip=[f"{ch_col}:Q","count()","stroke_lbl:N"]
    ).properties(height=280)
    st.altair_chart(ch, use_container_width=True)

    c1,c2 = st.columns(2)
    with c1:
        ch_b = alt.Chart(fdf,title=f"{met} — Box Plot by Outcome").mark_boxplot(
            extent="min-max",outliers={"size":10,"opacity":.2},median={"color":"white","size":2}
        ).encode(
            x=alt.X("stroke_lbl:N",title="Outcome"),
            y=alt.Y(f"{ch_col}:Q",title=met),
            color=alt.Color("stroke_lbl:N",scale=STROKE_SCALE,legend=None)
        ).properties(height=300)
        st.altair_chart(ch_b, use_container_width=True)

    with c2:
        sm   = smooth(fdf,"age",ch_col,7)
        bd   = band(fdf,"age",ch_col,7)
        line = alt.Chart(sm,title=f"Smoothed Mean {met} by Age").mark_line(strokeWidth=3,color=GOLD,interpolate="monotone").encode(
            x=alt.X("age:Q",title="Age"), y=alt.Y(f"{ch_col}:Q",title=f"Mean {met}"),
            tooltip=["age:Q",alt.Tooltip(f"{ch_col}:Q",format=".1f")])
        bnd  = alt.Chart(bd).mark_area(opacity=.15,color=GOLD,interpolate="monotone").encode(
            x="age:Q",y=alt.Y("lo:Q",title=""),y2="hi:Q")
        comp = bnd+line
        if ch_col=="avg_glucose_level":
            comp = comp + alt.Chart(pd.DataFrame({"y":[140]})).mark_rule(color=RED,strokeDash=[6,3],strokeWidth=1.5).encode(y="y:Q")
        st.altair_chart(comp.properties(height=300), use_container_width=True)

    # Glucose trend split
    st.markdown("<div class='sec-hdr'>Glucose Trend by Age — Stroke vs No Stroke</div>", unsafe_allow_html=True)
    ss  = smooth(fdf[fdf["stroke_lbl"]=="Stroke"],    "age","avg_glucose_level"); ss["Outcome"]="Stroke"
    ns  = smooth(fdf[fdf["stroke_lbl"]=="No Stroke"], "age","avg_glucose_level"); ns["Outcome"]="No Stroke"
    trd = pd.concat([ss,ns],ignore_index=True)
    lt = alt.Chart(trd,title="Smoothed Mean Glucose by Age").mark_line(strokeWidth=3,interpolate="monotone").encode(
        x="age:Q",y=alt.Y("avg_glucose_level:Q",title="Mean Glucose (mg/dL)"),
        color=alt.Color("Outcome:N",scale=alt.Scale(domain=["No Stroke","Stroke"],range=[BLUE,RED])),
        tooltip=["age:Q","Outcome:N",alt.Tooltip("avg_glucose_level:Q",format=".1f")])
    rl = alt.Chart(pd.DataFrame({"y":[140]})).mark_rule(color="#475569",strokeDash=[5,4],strokeWidth=1.5).encode(y="y:Q")
    st.altair_chart((lt+rl).properties(height=295), use_container_width=True)

    # Stroke probability curve
    st.markdown("<div class='sec-hdr'>Stroke Probability Curve by Age</div>", unsafe_allow_html=True)
    sf   = (fdf["stroke_lbl"]=="Stroke").astype(float)*100
    rtmp = pd.DataFrame({"age":fdf["age"].values,"r":sf.values})
    rg   = rtmp.groupby("age")["r"].mean()
    rd   = pd.DataFrame({"age":rg.index.tolist(),"Stroke Rate":rg.values.tolist()}).sort_values("age")
    rd["Stroke Rate"] = rd["Stroke Rate"].rolling(9,center=True,min_periods=1).mean()
    ac = alt.Chart(rd).mark_area(opacity=.18,color=RED,interpolate="monotone").encode(x="age:Q",y=alt.Y("Stroke Rate:Q",title="Stroke Rate (%)"))
    lc2 = alt.Chart(rd).mark_line(strokeWidth=3,color=RED,interpolate="monotone").encode(
        x="age:Q",y="Stroke Rate:Q",tooltip=["age:Q",alt.Tooltip("Stroke Rate:Q",format=".2f")])
    st.altair_chart((ac+lc2).properties(height=270), use_container_width=True)

    # BMI
    st.markdown("<div class='sec-hdr'>BMI Category vs Stroke</div>", unsafe_allow_html=True)
    bs = fdf.groupby(["bmi_lbl","stroke_lbl"]).size().reset_index(name="Count")
    ch = alt.Chart(bs,title="Stroke Outcome by BMI Category").mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
        x=alt.X("bmi_lbl:N",sort=["Underweight","Normal","Overweight","Obese"],title="BMI Category"),
        y="Count:Q",
        color=alt.Color("stroke_lbl:N",scale=STROKE_SCALE,title="Outcome"),
        xOffset="stroke_lbl:N",tooltip=["bmi_lbl","stroke_lbl","Count"]
    ).properties(height=260)
    st.altair_chart(ch, use_container_width=True)

    st.markdown("""<div class='ibox ibox-gold'>
    📈 <b>Curve Insights:</b><br>
    • The <b>age_glucose interaction</b> (age × avg_glucose_level) best separates stroke from no-stroke patients,
      confirming its importance as an engineered feature (ranked #2 in RF importance after age alone).<br>
    • Stroke probability is essentially 0% below age 30, then accelerates steeply after 55.<br>
    • Stroke patients maintain <b>consistently higher glucose at every age</b> — the two smoothed lines
      diverge clearly from ~age 40 onward.
    </div>""", unsafe_allow_html=True)

# ═══ TAB 4 — FEATURE INTERACTIONS ════════════════════════════════════════════
with t4:
    st.markdown("<div class='sec-hdr'>Engineered Feature Analysis</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)

    # age_hypertension
    with c1:
        ah_s  = smooth(fdf[fdf["stroke_lbl"]=="Stroke"],    "age","age_hypertension"); ah_s["Outcome"]="Stroke"
        ah_ns = smooth(fdf[fdf["stroke_lbl"]=="No Stroke"], "age","age_hypertension"); ah_ns["Outcome"]="No Stroke"
        ahd = pd.concat([ah_s,ah_ns],ignore_index=True)
        ch = alt.Chart(ahd,title="Age × Hypertension Feature by Age").mark_line(strokeWidth=2.8,interpolate="monotone").encode(
            x="age:Q",y=alt.Y("age_hypertension:Q",title="age_hypertension value"),
            color=alt.Color("Outcome:N",scale=alt.Scale(domain=["No Stroke","Stroke"],range=[BLUE,RED])),
            tooltip=["age","Outcome",alt.Tooltip("age_hypertension:Q",format=".1f")]
        ).properties(height=270)
        st.altair_chart(ch, use_container_width=True)

    # age_glucose
    with c2:
        ag_s  = smooth(fdf[fdf["stroke_lbl"]=="Stroke"],    "age","age_glucose"); ag_s["Outcome"]="Stroke"
        ag_ns = smooth(fdf[fdf["stroke_lbl"]=="No Stroke"], "age","age_glucose"); ag_ns["Outcome"]="No Stroke"
        agd = pd.concat([ag_s,ag_ns],ignore_index=True)
        ch = alt.Chart(agd,title="Age × Glucose Feature by Age").mark_line(strokeWidth=2.8,interpolate="monotone").encode(
            x="age:Q",y=alt.Y("age_glucose:Q",title="age_glucose value"),
            color=alt.Color("Outcome:N",scale=alt.Scale(domain=["No Stroke","Stroke"],range=[BLUE,RED])),
            tooltip=["age","Outcome",alt.Tooltip("age_glucose:Q",format=".0f")]
        ).properties(height=270)
        st.altair_chart(ch, use_container_width=True)

    # RF Feature Importance (from notebook output, exact values)
    st.markdown("<div class='sec-hdr'>Random Forest Feature Importance (from Notebook)</div>", unsafe_allow_html=True)
    fi = pd.DataFrame({
        "Feature":    ["age","avg_glucose_level","bmi","age_group","cardio_risk",
                       "bmi_category","high_glucose","hypertension","heart_disease"],
        "Importance": [0.3355, 0.2289, 0.2038, 0.1369, 0.0298,
                       0.0295, 0.0144, 0.0127, 0.0085]
    }).sort_values("Importance",ascending=True)
    bars = alt.Chart(fi,title="RF Feature Importance — Extended Model (9 features)").mark_bar(
        cornerRadiusTopRight=5,cornerRadiusBottomRight=5
    ).encode(
        x=alt.X("Importance:Q",title="Importance Score"),
        y=alt.Y("Feature:N",sort="-x",title=""),
        color=alt.Color("Importance:Q",scale=alt.Scale(scheme="reds"),legend=None),
        tooltip=["Feature",alt.Tooltip("Importance:Q",format=".4f")]
    ).properties(height=300)
    lm = bars.mark_text(align="left",dx=4,color="#64748B",fontSize=11).encode(text=alt.Text("Importance:Q",format=".4f"))
    st.altair_chart(bars+lm, use_container_width=True)

    # Heatmap: age group × cardio risk
    st.markdown("<div class='sec-hdr'>Stroke Rate Heatmap — Age Group × Cardio Risk</div>", unsafe_allow_html=True)
    hm_r = []
    for ag,al in AGE_GRP_MAP.items():
        for cr,cl in CARDIO_MAP.items():
            sub = fdf[(fdf["age_group"]==ag)&(fdf["cardio_risk"]==cr)]
            if len(sub)>=5:
                hm_r.append({"Age Group":al,"Cardio Risk":cl,
                              "Stroke Rate (%)":round((sub["stroke_lbl"]=="Stroke").mean()*100,2),"N":len(sub)})
    hm = pd.DataFrame(hm_r)
    h = alt.Chart(hm,title="Stroke Rate (%) — Age Group × Cardio Risk").mark_rect(cornerRadius=4).encode(
        x=alt.X("Cardio Risk:N",sort=["None","One Condition","Both Conditions"]),
        y=alt.Y("Age Group:N", sort=["0–30","30–45","45–60","60–75","75+"]),
        color=alt.Color("Stroke Rate (%):Q",scale=alt.Scale(scheme="reds")),
        tooltip=["Age Group","Cardio Risk","Stroke Rate (%)","N"]
    ).properties(height=270)
    ht = h.mark_text(fontSize=12,fontWeight=600).encode(
        text=alt.Text("Stroke Rate (%):Q",format=".1f"),
        color=alt.condition(alt.datum["Stroke Rate (%)"]>15,alt.value("white"),alt.value("#475569")))
    st.altair_chart(h+ht, use_container_width=True)

    st.markdown("""<div class='ibox ibox-blue'>
    🔬 <b>Feature Engineering Insights (from notebook):</b><br>
    • <b>age</b> is the single most important feature (0.3355) — dominating all models.<br>
    • <b>avg_glucose_level</b> (0.2289) and <b>bmi</b> (0.2038) are the next strongest raw features.<br>
    • The engineered <b>age_group</b> (0.1369) adds non-linear bucket information that individual age misses.<br>
    • <b>age_glucose</b> captures the joint effect of high age + high glucose — stroke patients have significantly
      higher values because both components are elevated simultaneously.<br>
    • The heatmap confirms that <b>elderly patients with both cardio conditions</b> face the highest risk.
    </div>""", unsafe_allow_html=True)

# ═══ TAB 5 — CORRELATIONS ════════════════════════════════════════════════════
with t5:
    st.markdown("<div class='sec-hdr'>Correlation Matrix — All Numeric Features</div>", unsafe_allow_html=True)
    num_cols = ["age","hypertension","heart_disease","ever_married","work_type","Residence_type",
                "avg_glucose_level","bmi","stroke","high_glucose","cardio_risk",
                "age_group","bmi_category","age_hypertension","age_glucose"]
    ndf = fdf[[c for c in num_cols if c in fdf.columns]].copy()
    ndf["smoking_status"] = fdf["smoking_status"].fillna(fdf["smoking_status"].median())
    cm  = ndf.corr()
    crw = []
    for r in cm.index:
        for c in cm.columns:
            crw.append({"F1":r,"F2":c,"r":round(cm.loc[r,c],3)})
    cdf = pd.DataFrame(crw)

    hm = alt.Chart(cdf,title="Pearson Correlation Matrix").mark_rect(cornerRadius=2).encode(
        x=alt.X("F1:N",title=""),y=alt.Y("F2:N",title=""),
        color=alt.Color("r:Q",scale=alt.Scale(scheme="redblue",domain=[-1,1],reverse=True),title="r"),
        tooltip=["F1","F2",alt.Tooltip("r:Q",format=".3f")]
    ).properties(height=430)
    tl = hm.mark_text(fontSize=9,fontWeight=600).encode(
        text=alt.Text("r:Q",format=".2f"),
        color=alt.condition((alt.datum.r>.45)|(alt.datum.r<-.45),alt.value("white"),alt.value("#475569")))
    st.altair_chart(hm+tl, use_container_width=True)

    # Top predictors of stroke
    st.markdown("<div class='sec-hdr'>Top Predictors of Stroke (Absolute Correlation)</div>", unsafe_allow_html=True)
    sc  = cm["stroke"].drop("stroke").abs().sort_values(ascending=False)
    scd = pd.DataFrame({"Feature":sc.index.tolist(),"Abs Corr":sc.values.tolist(),
                        "Dir":[("Positive" if cm.loc[f,"stroke"]>=0 else "Negative") for f in sc.index]})
    b = alt.Chart(scd,title="Feature Correlation with Stroke (|r|)").mark_bar(cornerRadiusTopRight=5,cornerRadiusBottomRight=5).encode(
        x=alt.X("Abs Corr:Q",title="| r with stroke |"),
        y=alt.Y("Feature:N",sort="-x",title=""),
        color=alt.Color("Dir:N",scale=alt.Scale(domain=["Positive","Negative"],range=[RED,BLUE])),
        tooltip=["Feature","Abs Corr","Dir"]
    ).properties(height=340)
    lb = b.mark_text(align="left",dx=4,color="#64748B",fontSize=11).encode(text=alt.Text("Abs Corr:Q",format=".3f"))
    st.altair_chart(b+lb, use_container_width=True)

    st.markdown("""<div class='ibox'>
    💡 <b>Strongest correlates with stroke:</b> <b>age</b> leads clearly, followed by
    <b>age_glucose</b> and <b>ever_married</b> (age confound), <b>avg_glucose_level</b>,
    <b>age_group</b>, and <b>hypertension</b>. The engineered features <b>age_glucose</b>
    and <b>age_hypertension</b> amplify signal that individual variables carry alone.
    </div>""", unsafe_allow_html=True)

    # Scatter matrix
    st.markdown("<div class='sec-hdr'>Pairwise Scatter — Key Features</div>", unsafe_allow_html=True)
    sp = fdf.sample(min(800,N),random_state=1)[["age","bmi","avg_glucose_level","age_glucose","stroke_lbl"]]
    ch = alt.Chart(sp).mark_circle(size=20,opacity=.4).encode(
        alt.X(alt.repeat("column"),type="quantitative"),
        alt.Y(alt.repeat("row"),type="quantitative"),
        color=alt.Color("stroke_lbl:N",scale=STROKE_SCALE,title="Outcome")
    ).repeat(row=["age","avg_glucose_level","age_glucose"],column=["age_glucose","avg_glucose_level","age"])
    st.altair_chart(ch, use_container_width=True)

    st.markdown("<div class='sec-hdr'>Data Preview (first 30 rows)</div>", unsafe_allow_html=True)
    pv = ["age","gender_lbl","hyp_lbl","hd_lbl","married_lbl","work_lbl","res_lbl",
          "avg_glucose_level","bmi","smoking_lbl","bmi_lbl","cardio_lbl","ag_lbl","stroke_lbl"]
    st.dataframe(fdf[pv].head(30).reset_index(drop=True), use_container_width=True, height=320)

# ═══ TAB 6 — MODEL RESULTS ═══════════════════════════════════════════════════
with t6:
    st.markdown("<div class='sec-hdr'>Phase 1 — Imbalanced Data (no SMOTE)</div>", unsafe_allow_html=True)

    # ── Exact values from notebook outputs ──────────────────────────────────
    phase1 = pd.DataFrame({
        "Model":      ["LR (Base)","LR (New Features)","RF (Base)","RF (New Features)",
                       "RF (Optimized)","DT (Initial)","DT (Optimized)",
                       "XGBoost (Initial)","XGBoost (Tuned)"],
        "Accuracy":   [0.9393,0.9393,0.9384,0.9393,0.8434,0.7661,0.7750,0.7661,0.8033],
        "Precision":  [0.0000,0.0000,0.4286,0.5000,0.1938,0.1502,0.1557,0.1502,0.1600],
        "Recall":     [0.0000,0.0000,0.0484,0.0484,0.5000,0.6129,0.6129,0.6129,0.5484],
        "F1-Score":   [0.0000,0.0000,0.0870,0.0882,0.2793,0.2413,0.2484,0.2413,0.2500],
    })

    phase2 = pd.DataFrame({
        "Model":     ["Logistic Regression","Random Forest","Decision Tree","KNN (k=7)"],
        "Accuracy":  [0.8313,0.9284,0.7572,0.8872],
        "Precision": [0.6203,0.8514,0.4467,0.7325],
        "Recall":    [0.4033,0.7778,0.8971,0.6872],
        "F1-Score":  [0.4888,0.8129,0.5964,0.7091],
    })

    st.markdown("""<div class='ibox ibox-purple'>
    ⚠️ <b>Class Imbalance Problem:</b> Only <b>4.87%</b> of patients had a stroke (248 of 5,109).
    This caused Logistic Regression to predict "No Stroke" for every patient, scoring 0 on the stroke class.
    Even Random Forest barely detected any strokes without SMOTE.
    XGBoost partially compensated using <code>scale_pos_weight = 20.86</code> (ratio of negatives to positives).
    </div>""", unsafe_allow_html=True)

    m1 = st.radio("Metric (Phase 1)", ["Precision","Recall","F1-Score","Accuracy"], horizontal=True, key="m1")
    p1 = phase1[["Model",m1]].copy(); p1["Best"] = p1["Model"]=="XGBoost (Tuned)"
    ch = alt.Chart(p1,title=f"{m1} — Imbalanced Data (Stroke Class)").mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
        x=alt.X("Model:N",sort=None,title="",axis=alt.Axis(labelAngle=-35,labelLimit=160)),
        y=alt.Y(f"{m1}:Q",scale=alt.Scale(domain=[0,1.0])),
        color=alt.condition(alt.datum.Best,alt.value(RED),alt.value("#1E3A5F")),
        tooltip=["Model",alt.Tooltip(f"{m1}:Q",format=".1%")]
    ).properties(height=300)
    lb = ch.mark_text(dy=-8,color="#64748B",fontSize=10).encode(text=alt.Text(f"{m1}:Q",format=".1%"))
    st.altair_chart(ch+lb, use_container_width=True)

    # ── Phase 2: SMOTE ────────────────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>Phase 2 — SMOTE Balanced Data</div>", unsafe_allow_html=True)

    st.markdown("""<div class='ibox ibox-teal'>
    ✅ <b>SMOTE Results:</b> After applying SMOTE (<code>sampling_strategy=0.25, k_neighbors=5</code>),
    all models detected stroke patients. <b>Random Forest</b> achieved the best overall balance with
    <b>Precision 85.1% · Recall 77.8% · F1 81.3%</b> — a dramatic improvement over the imbalanced baseline.
    </div>""", unsafe_allow_html=True)

    m2 = st.radio("Metric (Phase 2)", ["Precision","Recall","F1-Score","Accuracy"], horizontal=True, key="m2")
    p2 = phase2[["Model",m2]].copy(); p2["Best"] = p2["Model"]=="Random Forest"
    ch2 = alt.Chart(p2,title=f"{m2} — SMOTE-Balanced Data (Stroke Class)").mark_bar(cornerRadiusTopLeft=4,cornerRadiusTopRight=4).encode(
        x=alt.X("Model:N",sort=None,title="",axis=alt.Axis(labelAngle=-20)),
        y=alt.Y(f"{m2}:Q",scale=alt.Scale(domain=[0,1.0])),
        color=alt.condition(alt.datum.Best,alt.value(TEAL),alt.value("#1E3A5F")),
        tooltip=["Model",alt.Tooltip(f"{m2}:Q",format=".1%")]
    ).properties(height=280)
    lb2 = ch2.mark_text(dy=-8,color="#64748B",fontSize=11).encode(text=alt.Text(f"{m2}:Q",format=".1%"))
    st.altair_chart(ch2+lb2, use_container_width=True)

    # ── Before vs After SMOTE (RF) ────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>Before vs After SMOTE — Random Forest</div>", unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Phase":     ["Original RF (imbalanced)","SMOTE RF (balanced)",
                      "Original RF (imbalanced)","SMOTE RF (balanced)",
                      "Original RF (imbalanced)","SMOTE RF (balanced)"],
        "Metric":    ["Precision","Precision","Recall","Recall","F1-Score","F1-Score"],
        "Score":     [0.5000, 0.8514, 0.0484, 0.7778, 0.0882, 0.8129],
    })
    ch3 = alt.Chart(comp,title="RF Performance: Before vs After SMOTE (Stroke Class)").mark_bar(
        cornerRadiusTopLeft=4,cornerRadiusTopRight=4
    ).encode(
        x=alt.X("Metric:N",title=""),
        y=alt.Y("Score:Q",scale=alt.Scale(domain=[0,1.0])),
        color=alt.Color("Phase:N",scale=alt.Scale(domain=["Original RF (imbalanced)","SMOTE RF (balanced)"],
                                                    range=["#1E3A5F",TEAL])),
        xOffset="Phase:N",
        tooltip=["Metric","Phase",alt.Tooltip("Score:Q",format=".1%")]
    ).properties(height=300)
    lb3 = ch3.mark_text(dy=-8,color="#64748B",fontSize=11).encode(text=alt.Text("Score:Q",format=".1%"))
    st.altair_chart(ch3+lb3, use_container_width=True)

    # ── All metrics table ─────────────────────────────────────────────────────
    st.markdown("<div class='sec-hdr'>Full Results Tables</div>", unsafe_allow_html=True)
    c1,c2 = st.columns(2)
    with c1:
        st.caption("Phase 1 — Imbalanced")
        d1 = phase1.copy()
        for c in ["Accuracy","Precision","Recall","F1-Score"]: d1[c]=d1[c].map(lambda x:f"{x:.1%}")
        st.dataframe(d1.set_index("Model"), use_container_width=True, height=370)
    with c2:
        st.caption("Phase 2 — SMOTE Balanced")
        d2 = phase2.copy()
        for c in ["Accuracy","Precision","Recall","F1-Score"]: d2[c]=d2[c].map(lambda x:f"{x:.1%}")
        st.dataframe(d2.set_index("Model"), use_container_width=True, height=220)

    st.markdown("""<div class='ibox ibox-gold'>
    🏆 <b>Conclusions:</b><br>
    • <b>Best imbalanced model:</b> XGBoost (Tuned) — Recall 54.8%, F1 25.0% — best without resampling.<br>
    • <b>Best overall model:</b> Random Forest on SMOTE data — Precision 85.1%, Recall 77.8%, F1 81.3%.<br>
    • SMOTE improved RF Recall from <b>4.8% → 77.8%</b> and F1 from <b>8.8% → 81.3%</b>.<br>
    • High accuracy (~93%) in imbalanced models is <b>misleading</b> — they simply predict "No Stroke" every time.<br>
    • For medical screening, <b>Recall is the priority</b>: missing a real stroke is far more costly than a false alarm.
    </div>""", unsafe_allow_html=True)

# ── FOOTER ────────────────────────────────────────────────────────────────────
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)
st.markdown("""<div style='text-align:center;color:#1A2540;font-size:.78rem;padding:6px 0 16px'>
  Stroke Risk Dashboard · Streamlit + Altair · 5,109 patients · 17 features · LR · RF · DT · KNN · XGBoost · SMOTE
</div>""", unsafe_allow_html=True)
