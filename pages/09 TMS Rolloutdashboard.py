import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Global TMS Executive Cockpit", page_icon="🚚", layout="wide")

# =====================
# Synthetic Data
# =====================
@st.cache_data
def gen_base(seed=10):
    np.random.seed(seed)
    months = pd.date_range("2024-01-01", periods=18, freq="MS")
    regions = ["North America", "Europe", "APAC", "LATAM"]
    # Ops by region-month
    rows = []
    for m in months:
        for r in regions:
            base = np.random.normal(120, 7)
            tms = base - np.random.normal(14, 3)
            on_time = np.clip(np.random.normal(90, 4), 75, 99)
            adoption = np.clip(30 + months.get_loc(m)*4 + np.random.normal(0, 5), 10, 100)
            dq = np.clip(70 + months.get_loc(m)*2 + np.random.normal(0, 6), 40, 100)  # data quality score
            freshness = np.clip(24 - months.get_loc(m)*0.8 + np.random.normal(0,1.2), 4, 24)  # hours
            co2_b = np.random.normal(18, 2)
            co2_t = max(co2_b - np.random.normal(2.5, 0.8), 7.5)
            dwell_b = np.random.normal(5.8, 0.7)
            dwell_t = max(dwell_b - np.random.normal(1.2, 0.4), 1.1)
            inc = max(int(np.random.poisson(3) - months.get_loc(m)*0.1), 0)
            mttr = np.clip(np.random.normal(9,2) - months.get_loc(m)*0.2, 2, 12)  # hours
            iface_uptime = np.clip(97 + months.get_loc(m)*0.1 + np.random.normal(0,0.2), 96.5, 99.9)

            rows.append(dict(
                Month=m, Region=r,
                Cost_Baseline=round(base,2), Cost_TMS=round(tms,2),
                On_Time_Percent=round(on_time,1),
                Adoption_Rate=round(adoption,1),
                Data_Quality=round(dq,1), Data_Freshness_Hours=round(freshness,1),
                CO2_Baseline=round(co2_b,2), CO2_TMS=round(co2_t,2),
                Dwell_Baseline_Hours=round(dwell_b,2), Dwell_TMS_Hours=round(dwell_t,2),
                Incidents=inc, MTTR_Hours=round(mttr,1), Interface_Uptime=round(iface_uptime,2)
            ))
    df_ops = pd.DataFrame(rows)

    # Company month rollup
    dfm = (df_ops.groupby("Month")
           .agg(Cost_Baseline=("Cost_Baseline","mean"),
                Cost_TMS=("Cost_TMS","mean"),
                On_Time_Percent=("On_Time_Percent","mean"),
                Adoption_Rate=("Adoption_Rate","mean"),
                Data_Quality=("Data_Quality","mean"),
                Data_Freshness_Hours=("Data_Freshness_Hours","mean"),
                CO2_Baseline=("CO2_Baseline","mean"),
                CO2_TMS=("CO2_TMS","mean"),
                Dwell_Baseline_Hours=("Dwell_Baseline_Hours","mean"),
                Dwell_TMS_Hours=("Dwell_TMS_Hours","mean"),
                Incidents=("Incidents","sum"),
                MTTR_Hours=("MTTR_Hours","mean"),
                Interface_Uptime=("Interface_Uptime","mean")
                ).reset_index())
    shipments = 3200
    dfm["Monthly_Savings"] = (dfm["Cost_Baseline"]-dfm["Cost_TMS"]).clip(lower=0)*shipments
    dfm["Cumulative_Savings"] = dfm["Monthly_Savings"].cumsum()

    # Rollout countries and wave cohorts
    countries = [
        ("United States","USA","Wave 1","Live",0.95),
        ("Canada","CAN","Wave 1","Stabilizing",0.8),
        ("Germany","DEU","Wave 1","Live",0.92),
        ("France","FRA","Wave 2","Planned",0.3),
        ("India","IND","Wave 2","Live",0.9),
        ("China","CHN","Wave 3","Planned",0.2),
        ("Mexico","MEX","Wave 2","Stabilizing",0.7),
        ("Brazil","BRA","Wave 1","Live",0.9),
        ("United Kingdom","GBR","Wave 3","Planned",0.25),
        ("Japan","JPN","Wave 3","Planned",0.2),
        ("Australia","AUS","Wave 2","Live",0.88),
    ]
    df_roll = pd.DataFrame(countries, columns=["Country","ISO3","Wave","Status","Adoption"])
    code = {"Planned":0,"Stabilizing":1,"Live":2}
    df_roll["Status_Code"] = df_roll["Status"].map(code)

    # Program roadmap tasks
    start = datetime(2024,1,1)
    tasks = [
        ("Mobilization","Program", start, start+timedelta(days=50), "Done"),
        ("Global Design/Data","Program", start+timedelta(days=20), start+timedelta(days=150), "Done"),
        ("Build & Integrations","Program", start+timedelta(days=120), start+timedelta(days=290), "In Progress"),
        ("Cutover Prep","Program", start+timedelta(days=260), start+timedelta(days=340), "Planned"),
        ("NA Go‑Live","Rollout", start+timedelta(days=220), start+timedelta(days=300), "In Progress"),
        ("EU Go‑Live","Rollout", start+timedelta(days=300), start+timedelta(days=380), "Planned"),
        ("APAC Go‑Live","Rollout", start+timedelta(days=340), start+timedelta(days=440), "Planned"),
        ("LATAM Go‑Live","Rollout", start+timedelta(days=360), start+timedelta(days=460), "Planned"),
        ("Stabilization/Hypercare","Program", start+timedelta(days=300), start+timedelta(days=420), "Planned"),
    ]
    df_gantt = pd.DataFrame(tasks, columns=["Task","Workstream","Start","Finish","Status"])

    # RAID
    risks = [
        ("Legacy ERP interface latency","High","Medium","Mitigate","Add async queues + retries; perf test"),
        ("Carrier onboarding lag","Medium","High","Mitigate","Carrier pilot incentives + super-user program"),
        ("Rate table inaccuracies","High","High","Mitigate","Data owner + SLAs + dual-control changes"),
        ("Plant change fatigue","Medium","Medium","Mitigate","Weekly adoption huddles + visual KPIs"),
        ("Scope creep by region","Medium","Medium","Accept","Pattern library + variance board")
    ]
    df_risk = pd.DataFrame(risks, columns=["Risk","Impact","Probability","Strategy","Plan"])
    ip_map = {"Low":1,"Medium":2,"High":3}
    df_risk["Score"] = df_risk["Impact"].map(ip_map)*df_risk["Probability"].map(ip_map)
    # Simulated burndown by month (sum of scores)
    risk_burn = pd.DataFrame({
        "Month": months,
        "Total_Risk_Score": np.clip(50 - np.arange(len(months))*1.8 + np.random.normal(0,2,len(months)), 10, 55)
    })

    # Savings waterfall components (illustrative)
    waterfall = pd.DataFrame({
        "label": ["Baseline Spend","Mode/Route Opt","Auto Tender","Freight Audit","Dwell Reduction","Other","Post‑TMS Spend"],
        "value": [0, -8.5, -6.0, -3.2, -2.4, -1.0, 0],  # percentage points
        "type": ["total","relative","relative","relative","relative","relative","total"]
    })

    return df_ops, dfm, df_roll, df_gantt, df_risk, risk_burn, waterfall

df_ops, dfm, df_roll, df_gantt, df_risk, risk_burn, waterfall = gen_base()

# Optional CSV overrides
st.sidebar.header("Upload Overrides (Optional)")
up_ops = st.sidebar.file_uploader("Operational metrics CSV (region-month)", type=["csv"])
if up_ops is not None:
    try:
        df_ops = pd.read_csv(up_ops, parse_dates=["Month"])
        st.sidebar.success("Operational metrics loaded.")
    except Exception as e:
        st.sidebar.error(f"Failed to read ops CSV: {e}")

# =====================
# Helpers
# =====================
def kpis(dfm):
    latest = dfm.iloc[-1]
    cost_red = (1 - dfm["Cost_TMS"].mean()/dfm["Cost_Baseline"].mean())*100
    ontime = latest["On_Time_Percent"]
    adopt = latest["Adoption_Rate"]
    dq = latest["Data_Quality"]
    cumsave = dfm["Cumulative_Savings"].iloc[-1]
    co2 = (1 - dfm["CO2_TMS"].mean()/dfm["CO2_Baseline"].mean())*100
    dwell = (1 - dfm["Dwell_TMS_Hours"].mean()/dfm["Dwell_Baseline_Hours"].mean())*100
    mttr = latest["MTTR_Hours"]
    uptime = latest["Interface_Uptime"]
    return dict(cost_red=cost_red, ontime=ontime, adopt=adopt, dq=dq, cumsave=cumsave, co2=co2, dwell=dwell, mttr=mttr, uptime=uptime)

def health_color(v, green, amber):
    if v >= green: return "✅"
    if v >= amber: return "🟡"
    return "🔴"

# =====================
# Header
# =====================
st.title("Global TMS Executive Cockpit")
st.caption("One place for rollout, roadmap, KPIs, adoption, RAID, value, data quality, incidents & forecast")

# Global filters
date_min, date_max = dfm["Month"].min(), dfm["Month"].max()
colf1, colf2, colf3 = st.columns([2,2,3])
with colf1:
    dr = st.slider("Date range", min_value=date_min.to_pydatetime(), max_value=date_max.to_pydatetime(),
                   value=(date_min.to_pydatetime(), date_max.to_pydatetime()))
with colf2:
    region_opt = ["All"] + sorted(df_ops["Region"].unique().tolist())
    region = st.selectbox("Region", region_opt, index=0)
with colf3:
    st.info("Tip: Upload overrides on the left sidebar to demo with your data.")

def filter_dfm(df_ops, dfm, region, dr):
    m0, m1 = pd.to_datetime(dr[0]), pd.to_datetime(dr[1])
    if region == "All":
        return dfm[(dfm["Month"]>=m0) & (dfm["Month"]<=m1)].copy()
    # rebuild dfm for region
    dfo = df_ops[(df_ops["Region"]==region) & (df_ops["Month"]>=m0) & (df_ops["Month"]<=m1)].copy()
    dfmr = (dfo.groupby("Month").agg(Cost_Baseline=("Cost_Baseline","mean"),
                                     Cost_TMS=("Cost_TMS","mean"),
                                     On_Time_Percent=("On_Time_Percent","mean"),
                                     Adoption_Rate=("Adoption_Rate","mean"),
                                     Data_Quality=("Data_Quality","mean"),
                                     Data_Freshness_Hours=("Data_Freshness_Hours","mean"),
                                     CO2_Baseline=("CO2_Baseline","mean"),
                                     CO2_TMS=("CO2_TMS","mean"),
                                     Dwell_Baseline_Hours=("Dwell_Baseline_Hours","mean"),
                                     Dwell_TMS_Hours=("Dwell_TMS_Hours","mean"),
                                     Incidents=("Incidents","sum"),
                                     MTTR_Hours=("MTTR_Hours","mean"),
                                     Interface_Uptime=("Interface_Uptime","mean")).reset_index())
    shipments = 900
    dfmr["Monthly_Savings"] = (dfmr["Cost_Baseline"]-dfmr["Cost_TMS"]).clip(lower=0)*shipments
    dfmr["Cumulative_Savings"] = dfmr["Monthly_Savings"].cumsum()
    return dfmr

dfm_f = filter_dfm(df_ops, dfm, region, dr)
K = kpis(dfm_f)

# =====================
# Executive Overview
# =====================
st.subheader("Executive Overview")
t1,t2,t3,t4,t5,t6 = st.columns(6)
t1.metric("Avg Cost Reduction", f"{K['cost_red']:.1f}%")
t2.metric("On-Time Delivery", f"{K['ontime']:.1f}%")
t3.metric("TMS Adoption", f"{K['adopt']:.1f}%")
t4.metric("Data Quality", f"{K['dq']:.1f}")
t5.metric("Cumulative Savings", f"${K['cumsave']:,.0f}")
t6.metric("Interface Uptime", f"{K['uptime']:.2f}%")

h1,h2,h3 = st.columns(3)
with h1:
    # Overall health RAG based on thresholds
    cr = health_color(K['cost_red'], 10, 5)
    ot = health_color(K['ontime'], 92, 88)
    ad = health_color(K['adopt'], 80, 60)
    dq = health_color(K['dq'], 85, 75)
    st.markdown(f"**Program Health:** {cr} Cost | {ot} On‑Time | {ad} Adoption | {dq} Data Quality")
with h2:
    # Milestone donut (percent complete vs roadmap tasks)
    completed = (df_gantt["Status"]=="Done").sum()
    inprog = (df_gantt["Status"]=="In Progress").sum()
    planned = (df_gantt["Status"]=="Planned").sum()
    fig_donut = go.Figure(data=[go.Pie(labels=["Done","In Progress","Planned"], values=[completed,inprog,planned], hole=.55)])
    fig_donut.update_layout(title="Milestone Status")
    st.plotly_chart(fig_donut, use_container_width=True)
with h3:
    # Alerts
    latest = dfm_f.iloc[-1]
    alerts = []
    if K['ontime'] < 88: alerts.append("On‑time below target 88%.")
    if K['adopt'] < 60: alerts.append("Adoption below 60% in selected scope.")
    if K['dq'] < 75: alerts.append("Data quality below 75%.")
    if latest["Data_Freshness_Hours"] > 12: alerts.append("Data freshness > 12 hours.")
    if latest["MTTR_Hours"] > 8: alerts.append("MTTR > 8 hours.")
    if not alerts: alerts = ["All core indicators within thresholds."]
    st.markdown("**Program Alerts**")
    for a in alerts:
        st.write("• " + a)

st.markdown("---")

# =====================
# Roadmap & Waves
# =====================
st.subheader("Roadmap & Waves")
cA,cB = st.columns([2,1])
with cA:
    fig_gantt = px.timeline(df_gantt, x_start="Start", x_end="Finish", y="Workstream", color="Status", text="Task",
                            title="Implementation Roadmap (Gantt)")
    fig_gantt.update_yaxes(autorange="reversed")
    st.plotly_chart(fig_gantt, use_container_width=True)
with cB:
    wave_prog = df_roll.groupby("Wave").agg(Avg_Adoption=("Adoption","mean")).reset_index()
    st.plotly_chart(px.bar(wave_prog, x="Wave", y="Avg_Adoption", title="Wave Progress (Avg Adoption %)",
                           range_y=[0,1]).update_yaxes(tickformat=".0%"), use_container_width=True)
    st.dataframe(df_roll[["Country","Wave","Status","Adoption"]].assign(Adoption=lambda d: (d["Adoption"]*100).round(0).astype(int).astype(str)+"%"),
                 use_container_width=True)

st.markdown("---")

# =====================
# Cost & Service
# =====================
st.subheader("Cost & Service")
c1,c2 = st.columns(2)
with c1:
    fig_cost = px.line(dfm_f, x="Month", y=["Cost_Baseline","Cost_TMS"], title="Freight Cost per Shipment (Baseline vs TMS)")
    fig_cost.update_layout(yaxis_title="USD per Shipment")
    st.plotly_chart(fig_cost, use_container_width=True)
with c2:
    latest_m = df_ops["Month"].max()
    dfl = df_ops[df_ops["Month"]==latest_m].groupby("Region", as_index=False)["On_Time_Percent"].mean()
    st.plotly_chart(px.bar(dfl, x="Region", y="On_Time_Percent", title=f"On‑Time by Region ({latest_m.date()})"),
                    use_container_width=True)

# =====================
# Value Realization
# =====================
st.subheader("Value Realization")
v1,v2 = st.columns(2)
with v1:
    st.plotly_chart(px.line(dfm_f, x="Month", y="Cumulative_Savings", title="Cumulative Savings"),
                    use_container_width=True)
with v2:
    fig_wf = go.Figure(go.Waterfall(
        name="Savings",
        orientation="v",
        measure=waterfall["type"],
        x=waterfall["label"],
        text=[f"{v:+.1f}%" if i not in [0,len(waterfall)-1] else "" for i,v in enumerate(waterfall["value"])],
        y=[0 if t=="total" else v for t,v in zip(waterfall["type"], waterfall["value"])],
        connector={"line":{"width":1}}
    ))
    fig_wf.update_layout(title="Savings Waterfall (Illustrative % of Baseline Spend)")
    st.plotly_chart(fig_wf, use_container_width=True)

st.markdown("---")

# =====================
# Adoption & Data Quality
# =====================
st.subheader("Adoption & Data Quality")
a1,a2,a3 = st.columns([1.2,1.2,0.8])
with a1:
    st.plotly_chart(px.area(dfm_f, x="Month", y="Adoption_Rate", title="Adoption Trend"), use_container_width=True)
with a2:
    st.plotly_chart(px.line(dfm_f, x="Month", y="Data_Quality", title="Data Quality Score"), use_container_width=True)
with a3:
    st.plotly_chart(px.line(dfm_f, x="Month", y="Data_Freshness_Hours", title="Data Freshness (Hours)"),
                    use_container_width=True)

st.markdown("---")

# =====================
# RAID
# =====================
st.subheader("RAID — Risks, Issues, Actions, Decisions")
r1,r2 = st.columns([1.4,1])
with r1:
    order = ["Low","Medium","High"]
    pivot = (df_risk.assign(cnt=1)
             .pivot_table(index="Impact", columns="Probability", values="cnt", aggfunc="sum", fill_value=0)
             .reindex(index=order, columns=order))
    fig_heat = px.imshow(pivot.values, x=pivot.columns, y=pivot.index, text_auto=True,
                         title="Risk Heatmap (Impact × Probability)")
    fig_heat.update_xaxes(title="Probability")
    fig_heat.update_yaxes(title="Impact")
    st.plotly_chart(fig_heat, use_container_width=True)
with r2:
    st.plotly_chart(px.line(risk_burn, x="Month", y="Total_Risk_Score", title="Risk Burndown (Total Score)"),
                    use_container_width=True)

st.dataframe(df_risk.sort_values("Score", ascending=False), use_container_width=True)

st.markdown("---")

# =====================
# Exceptions & SLA
# =====================
st.subheader("Exceptions & SLA")
e1,e2,e3 = st.columns(3)
with e1:
    st.plotly_chart(px.line(dfm_f, x="Month", y="Incidents", title="Incident Count"), use_container_width=True)
with e2:
    st.plotly_chart(px.line(dfm_f, x="Month", y="MTTR_Hours", title="Mean Time to Resolve (Hours)"),
                    use_container_width=True)
with e3:
    st.plotly_chart(px.line(dfm_f, x="Month", y="Interface_Uptime", title="Interface Uptime (%)"),
                    use_container_width=True)

# =====================
# Forecast & Scenarios
# =====================
st.subheader("Forecast & Scenarios")
s1, s2 = st.columns([1,1])
with s1:
    # Scenario sliders
    st.markdown("**Assumptions**")
    delta_route = st.slider("Route optimization impact (USD/shipment)", 0.0, 10.0, 3.0, 0.5)
    delta_tender = st.slider("Auto-tender impact (USD/shipment)", 0.0, 8.0, 2.0, 0.5)
    delta_dwell = st.slider("Dwell reduction impact (USD/shipment)", 0.0, 6.0, 1.5, 0.5)
    base_last = dfm_f["Cost_TMS"].iloc[-1] if len(dfm_f) else 100
    projected = base_last - (delta_route + delta_tender + delta_dwell)
    st.metric("Projected Cost per Shipment (Next Q)", f"${projected:,.2f}")
with s2:
    # Simple 6-month projection from last point + scenario deltas
    if len(dfm_f) > 0:
        future_months = [dfm_f["Month"].max() + pd.DateOffset(months=i) for i in range(1,7)]
        proj_vals = [max(projected + np.random.normal(0,0.8), 70) for _ in future_months]
        fig_proj = go.Figure()
        fig_proj.add_trace(go.Scatter(x=dfm_f["Month"], y=dfm_f["Cost_TMS"], mode="lines+markers", name="Actual TMS"))
        fig_proj.add_trace(go.Scatter(x=future_months, y=proj_vals, mode="lines+markers", name="Projection", line=dict(dash="dash")))
        fig_proj.update_layout(title="6-Month Cost Projection (Scenario)", yaxis_title="USD per Shipment", xaxis_title="Month")
        st.plotly_chart(fig_proj, use_container_width=True)

st.caption(f"© {datetime.now().year} — Example executive cockpit with synthetic data.")
