import sys
from pathlib import Path
import pandas as pd
import streamlit as st
import miniscotSupplychainsimulation as runner  # 👈 uses your existing simulation script

st.set_page_config(page_title="miniSCOT Supply Chain Simulator", layout="wide")
st.title("miniSCOT: Supply Chain Simulation Dashboard")

# Sidebar parameters
st.sidebar.header("Simulation Parameters")
sim_days = st.sidebar.slider("Simulation horizon (days)", 7, 60, 14)
avg_demand = st.sidebar.number_input("Avg daily demand", 10, 5000, 120, 10)
safety_stock = st.sidebar.number_input("Safety stock", 0, 20000, 500, 50)
lead_mu = st.sidebar.number_input("Mean lead time (days)", 0.1, 30.0, 2.0, 0.1)
inv_hold = st.sidebar.number_input("Inventory holding $/unit/day", 0.0, 10.0, 0.02, 0.01)
trans_cost = st.sidebar.number_input("Transport $/unit", 0.0, 100.0, 1.2, 0.1)

cfg = {
    "sim_days": sim_days,
    "avg_daily_demand": avg_demand,
    "safety_stock": safety_stock,
    "lead_time_mean": lead_mu,
    "inventory_holding_per_unit": inv_hold,
    "transport_cost_per_unit": trans_cost,
}

st.subheader("Config")
st.json(cfg)

if st.button("Run simulation", type="primary"):
    # Run the mock simulator from your Python script
    df = runner.simulate_mock(cfg)
    kpis = runner.compute_kpis(df)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Avg Fill Rate", f"{kpis['avg_fill_rate']:.2%}")
    c2.metric("Total Stockouts", f"{kpis['total_stockouts']}")
    c3.metric("Total Cost", f"${kpis['total_cost']:,.0f}")
    c4.metric("Avg Lead Time", f"{kpis['avg_lead_time']:.2f} days")

    st.markdown("### Results Table")
    st.dataframe(df, use_container_width=True)

    st.markdown("### Cost over Time")
    st.line_chart(df.set_index("period")[["transport_cost", "inventory_cost", "total_cost"]])

    st.markdown("### Fill Rate")
    st.line_chart(df.set_index("period")[["fill_rate"]])

    st.download_button(
        "Download CSV",
        df.to_csv(index=False).encode("utf-8"),
        file_name="results.csv",
        mime="text/csv",
    )
