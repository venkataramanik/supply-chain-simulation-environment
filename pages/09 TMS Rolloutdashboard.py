import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="TMS Executive Cockpit", page_icon="🚛", layout="wide")

# -----------------------------
# Data Generators (synthetic)
# -----------------------------
@st.cache_data
def gen_monthly_ops(seed=7):
    np.random.seed(seed)
    months = pd.date_range("2024-01-01", periods=18, freq="MS")
    regions = ["North America", "Europe", "APAC", "LATAM"]
    rows = []
    for m in months:
        for r in regions:
            base = np.random.normal(120, 7)
            tms = base - np.random.normal(14, 3)
            on_time = np.clip(np.random.normal(90, 4), 75, 99)
            adoption = np.clip(35 + months.get_loc(m)*4 + np.random.normal(0, 5), 15, 100)
            co2_b = np.random.normal(18, 2)
            co2_t = max(co2_b - np.random.normal(2.5, 0.8), 8)
            dwell_b = np.random.normal(5.8, 0.7)
            dwell_t = max(dwell_b - np.random.normal(1.2, 0.4), 1.2)
            rows.append({
                "Month": m, "Region": r,
                "Cost_Baseline": round(base,2), "Cost_TMS": round(tms,2),
                "On_Time_Percent": round(on_time,1),
                "Adoption_Rate": round(adoption,1),
                "CO2_Baseline": round(co2_b,2), "CO2_TMS": round(co2_t,2),
                "Dwell_Baseline_Hours": round(dwell_b,2), "Dwell_TMS_Hours": round(dwell_t,2)
            })
    df = pd.DataFrame(rows)
    dfm = (df.groupby("Month")
             .agg(Cost_Baseline=("Cost_Baseline","mean"),
                  Cost_TMS=("Cost_TMS","mean"),
                  On_Time_Percent=("On_Time_Percent","mean"),
                  Adoption_Rate=("Adoption_Rate","mean"),
                  CO2_Baseline=("CO2_Baseline","mean"),
                  CO2_TMS=("CO2_TMS","mean"),
                  Dwell_Baseline_Hours=("Dwell_Baseline_Hours","mean"),
                  Dwell_TMS_Hours=("Dwell_TMS_Hours","mean"))
             .reset_index())
    shipments = 3000
    dfm["Monthly_Savings"] = (dfm["Cost_Baseline"]-dfm["Cost_TMS"]).clip(lower=0)*shipments
    dfm["Cumulative_Savings"] = dfm["Monthly_Savings"].cumsum()
    dfm["CO2_Reduction"] = (dfm["CO2_Baseline"]-dfm["CO2_TMS"]).clip(lower=0)
    return df, dfm

@st.cache_data
def gen_rollout_status():
    countries = [
        ("United States","USA","Live"),
        ("Canada","CAN","Stabilizing"),
        ("Germany","DEU","Live"),
        ("France","FRA","Planned"),
        ("India","IND","Live"),
        ("China","CHN","Planned"),
        ("Mexico","MEX","Stabilizing"),
        ("Brazil","BRA","Live"),
        ("United Kingdom","GBR","Planned"),
        ("Japan","JPN","Planned"),
        ("Australia","AUS","Live"),
    ]
    df = pd.DataFrame(countries, columns=["country","iso3","Status"])
    code = {"Planned":0,"Stabilizing":1,"Live":2}
    df["Status_Code"] = df["Status"].map(code)
    return df

@st.cache_data
def gen_roadmap():
    start = datetime(2024,1,1)
    tasks = [
        ("Program Mobilization","Program", start, start+timedelta(days=60), "Done"),
        ("Global Design & Data Model","Program", start+timedelta(days=30), start+timedelta(days=150), "Done"),
        ("Core Build & Integration","Program", start+timedelta(days=120), start+timedelta(days=270), "In Progress"),
        ("Global UAT & Cutover Prep","Program", start+timedelta(days=250), start+timedelta(days=330), "Planned"),
        ("North America Go-Live","Rollout", start+timedelta(days=210), start+timedelta(days=300), "In Progress"),
        ("Europe Go-Live","Rollout", start+timedelta(days=280), start+timedelta(days=370), "Planned"),
        ("APAC Go-Live","Rollout", start+timedelta(days=330), start+timedelta(days=430), "Planned"),
        ("LATAM Go-Live","Rollout", start+timedelta(days=360), start+timedelta(days=460), "Planned"),
        ("Stabilization & Hypercare","Program", start+timedelta(days=300), start+timedelta(days=420), "Planned"),
    ]
    df = pd.DataFrame(tasks, columns=["Task","Workstream","Start","Finish","Status"])
    return df

@st.cache_data
def gen_raid():
    np.random.seed(22)
    risks = [
        ("Integration delays with legacy ERP","High","Medium","Mitigate","API mock + layered cutover; add test automation"),
        ("Carrier onboarding resistance","Medium","High","Mitigate","Tiered incentives + super-user carrier program"),
        ("Data quality issues in rate tables","High","High","Mitigate","Ownership + data SLAs + dual-control changes"),
        ("Change fatigue in plants","Medium","Medium","Mitigate","Weekly adoption huddles + visible KPIs + leadership notes"),
        ("Scope creep on regional variations","Medium","Medium","Accept","Pattern library + variance approval board"),
    ]
    issues = [
        ("Dwell events not posting to TMS in EU","High","Open","Ops/IT","Hotfix in testing; monitor backlog SLA"),
        ("Duplicate carrier codes in NA","Medium","Open","Master Data","Merge & lock; audit weekly"),
    ]
    actions = [
        ("Finalize canonical shipment status model","High","PMO","This Sprint"),
        ("Enable automated freight audit checks","Medium","Finance/IT","Next Sprint"),
        ("Roll out carrier scorecards in LATAM","Medium","Ops","This Month"),
    ]
    decisions = [
        ("Adopt global accessorial code set v1.2","Approved","SteerCo"),
        ("Keep NA tender window at 4h for Q4","Approved","Ops Lead"),
    ]
    df_risk = pd.DataFrame(risks, columns=["Risk","Impact","Probability","Strategy","Plan"])
    ip_map = {"Low":1,"Medium":2,"High":3}
    df_risk["Score"] = df_risk["Impact"].map(ip_map)*df_risk["Probability"].map(ip_map)
    df_issue = pd.DataFrame(issues, columns=["Issue","Impact","Status","Owner","Next Step"])
    df_action = pd.DataFrame(actions, columns=["Action","Priority","Owner","ETA"])
    df_decision = pd.DataFrame(decisions, columns=["Decision","Status","Owner"])
    return df_risk, df_issue, df_action, df_decision

# -----------------------------
# KPI calculations
# -----------------------------
def compute_kpis(df_month):
    latest = df_month.iloc[-1]
    kpi_cost_reduction = (1 - df_month["Cost_TMS"].mean()/df_month["Cost_Baseline"].mean())*100
    kpi_ontime = latest["On_Time_Percent"]
    kpi_adoption = latest["Adoption_Rate"]
    kpi_cumsave = df_month["Cumulative_Savings"].iloc[-1]
    kpi_co2 = (1 - df_month["CO2_TMS"].mean()/df_month["CO2_Baseline"].mean())*100
    kpi_dwell = (1 - df_month["Dwell_TMS_Hours"].mean()/df_month["Dwell_Baseline_Hours"].mean())*100
    return dict(
        cost_reduction_pct=float(kpi_cost_reduction),
        on_time=float(kpi_ontime),
        adoption=float(kpi_adoption),
        cumulative_savings=float(kpi_cumsave),
        co2_reduction_pct=float(kpi_co2),
        dwell_reduction_pct=float(kpi_dwell),
    )

# -----------------------------
# Load Data
# -----------------------------
df_ops, df_month = gen_monthly_ops()
df_roll = gen_rollout_status()
df_rmap = gen_roadmap()
df_risk, df_issue, df_action, df_decision = gen_raid()
k = compute_kpis(df_month)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")
date_min, date_max = df_month["Month"].min(), df_month["Month"].max()
date_rng = st.sidebar.slider("Date Range", min_value=date_min.to_pydatetime(),
                             max_value=date_max.to_pydatetime(),
                             value=(date_min.to_pydatetime(), date_max.to_pydatetime()))
region_opts = ["All"] + sorted(df_ops["Region"].unique().tolist())
region_sel = st.sidebar.selectbox("Region", region_opts, index=0)

def apply_filters(df_ops, df_month, region_sel, date_rng):
    m_start, m_end = pd.to_datetime(date_rng[0]), pd.to_datetime(date_rng[1])
    dfm = df_month[(df_month["Month"]>=m_start) & (df_month["Month"]<=m_end)].copy()
    if region_sel != "All":
        dfo = df_ops[(df_ops["Region"]==region_sel) & (df_ops["Month"]>=m_start) & (df_ops["Month"]<=m_end)].copy()
        dfm = (dfo.groupby("Month").agg(Cost_Baseline=("Cost_Baseline","mean"),
                                        Cost_TMS=("Cost_TMS","mean"),
                                        On_Time_Percent=("On_Time_Percent","mean"),
                                        Adoption_Rate=("Adoption_Rate","mean"),
                                        CO2_Baseline=("CO2_Baseline","mean"),
                                        CO2_TMS=("CO2_TMS","mean"),
                                        Dwell_Baseline_Hours=("Dwell_Baseline_Hours","mean"),
                                        Dwell_TMS_Hours=("Dwell_TMS_Hours","mean")).reset_index())
        shipments = 800  # region scale assumption
        dfm["Monthly_Savings"] = (dfm["Cost_Baseline"]-dfm["Cost_TMS"]).clip(lower=0)*shipments
        dfm["Cumulative_Savings"] = dfm["Monthly_Savings"].cumsum()
        dfm["CO2_Reduction"] = (dfm["CO2_Baseline"]-dfm["CO2_TMS"]).clip(lower=0)
    return dfm

dfm_f = apply_filters(df_ops, df_month, region_sel, date_rng)
k_f = compute_kpis(dfm_f)

# -----------------------------
# Header
# -----------------------------
st.title("TMS Executive Cockpit — Global Program")
st.caption("Roadmap • KPIs • Adoption • Risks (RAID) • Rollout • Cost/Service • ROI • Forecast")
st.markdown("> **Benchmark reference:** McKinsey Global Implementation Survey 2022")

# -----------------------------
# KPI Tiles
# -----------------------------
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Avg Cost Reduction", f"{k_f['cost_reduction_pct']:.1f}%", "vs. baseline")
c2.metric("On-Time Delivery", f"{k_f['on_time']:.1f}%", "latest")
c3.metric("TMS Adoption", f"{k_f['adoption']:.1f}%", "company-wide")
c4.metric("Cumulative Savings", f"${k_f['cumulative_savings']:,.0f}")
c5.metric("CO₂ | Dwell Reduction", f"{k_f['co2_reduction_pct']:.1f}% | {k_f['dwell_reduction_pct']:.1f}%", "vs. baseline")

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_roadmap, tab_rollout, tab_costsvc, tab_adopt, tab_susops, tab_raid, tab_forecast, tab_data = st.tabs(
    ["Overview","Roadmap","Rollout","Cost & Service","Adoption & Value","Sustainability & Ops","RAID","Forecast","Data"]
)

with tab_overview:
    colA, colB = st.columns(2)
    with colA:
        fig_cost = px.line(dfm_f, x="Month", y=["Cost_Baseline","Cost_TMS"], title="Freight Cost per Shipment (Baseline vs Post‑TMS)")
        fig_cost.update_layout(yaxis_title="USD per Shipment", legend_title_text="Series")
        st.plotly_chart(fig_cost, use_container_width=True)
    with colB:
        latest = df_ops["Month"].max()
        dfl = df_ops[df_ops["Month"]==latest].groupby("Region", as_index=False)["On_Time_Percent"].mean()
        st.plotly_chart(px.bar(dfl, x="Region", y="On_Time_Percent", title=f"On‑Time Delivery by Region ({latest.date()})"), use_container_width=True)

with tab_roadmap:
    fig_rmap = px.timeline(df_rmap, x_start="Start", x_end="Finish", y="Workstream", color="Status", text="Task",
                           title="Implementation Roadmap (Gantt)")
    fig_rmap.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_rmap, use_container_width=True)

with tab_rollout:
    fig_map = px.choropleth(df_roll, locations="iso3", color="Status_Code", hover_name="country",
                            color_continuous_scale=[(0,"#b0b0b0"),(0.5,"#F1C232"),(1,"#6AA84F")],
                            title="TMS Rollout Status by Country")
    fig_map.update_coloraxes(colorbar_title="Status", colorbar=dict(tickvals=[0,1,2], ticktext=["Planned","Stabilizing","Live"]))
    st.plotly_chart(fig_map, use_container_width=True)

with tab_costsvc:
    st.plotly_chart(fig_cost, use_container_width=True)
    st.markdown("---")
    st.subheader("On‑Time Distribution (Latest Month, by Region)")
    st.plotly_chart(px.box(df_ops[df_ops["Month"]==df_ops["Month"].max()], x="Region", y="On_Time_Percent", points="all"), use_container_width=True)

with tab_adopt:
    st.plotly_chart(px.area(dfm_f, x="Month", y="Adoption_Rate", title="TMS Adoption Trend (Company‑Wide)"), use_container_width=True)
    st.markdown("---")
    st.plotly_chart(px.line(dfm_f, x="Month", y="Cumulative_Savings", title="Value Realization (Cumulative Savings)"), use_container_width=True)

with tab_susops:
    k_all = compute_kpis(df_month)  # company-wide for gauges
    g1 = go.Figure(go.Indicator(mode="gauge+number", value=round(k_all["co2_reduction_pct"],1),
                                title={'text': "CO₂ Reduction vs Baseline (%)"}, gauge={'axis': {'range':[0,30]}}))
    g2 = go.Figure(go.Indicator(mode="gauge+number", value=round(k_all["dwell_reduction_pct"],1),
                                title={'text': "Dwell Time Reduction (%)"}, gauge={'axis': {'range':[0,40]}}))
    col1, col2 = st.columns(2)
    col1.plotly_chart(g1, use_container_width=True)
    col2.plotly_chart(g2, use_container_width=True)

with tab_raid:
    st.subheader("Risks (Prioritized)")
    st.dataframe(df_risk.sort_values("Score", ascending=False), use_container_width=True)
    st.subheader("Issues (Open)")
    st.dataframe(df_issue, use_container_width=True)
    st.subheader("Actions (In Flight)")
    st.dataframe(df_action, use_container_width=True)
    st.subheader("Decisions (Approved)")
    st.dataframe(df_decision, use_container_width=True)

with tab_forecast:
    df_fc = dfm_f.copy()
    df_fc["t"] = np.arange(len(df_fc))
    if len(df_fc) > 1:
        coef = np.polyfit(df_fc["t"], df_fc["Cost_TMS"], 1)
        trend = np.poly1d(coef)
        future = []
        for i in range(1,7):
            t = len(df_fc)+i-1
            m = df_fc["Month"].max() + pd.DateOffset(months=i)
            future.append({"Month": m, "Cost_TMS": trend(t)})
        df_future = pd.DataFrame(future)
        fig_forecast = go.Figure()
        fig_forecast.add_trace(go.Scatter(x=df_fc["Month"], y=df_fc["Cost_TMS"], mode="lines+markers", name="Actual (TMS)"))
        fig_forecast.add_trace(go.Scatter(x=df_future["Month"], y=df_future["Cost_TMS"], mode="lines+markers", name="Forecast (Next 6 mo)", line=dict(dash="dash")))
        fig_forecast.update_layout(title="Forecast: Cost per Shipment (Next 6 Months)", yaxis_title="USD per Shipment", xaxis_title="Month")
        st.plotly_chart(fig_forecast, use_container_width=True)
    else:
        st.info("Expand the date range to enable forecasting.")

with tab_data:
    st.write("**Monthly Company-Wide Metrics**")
    st.dataframe(df_month, use_container_width=True)
    st.write("**Operational (Region-Month)**")
    st.dataframe(df_ops, use_container_width=True)

st.caption("© {} — Example app with synthetic data. Benchmark: McKinsey Global Implementation Survey 2022.".format(datetime.now().year))
