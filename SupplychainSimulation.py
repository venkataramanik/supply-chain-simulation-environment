import sys
from pathlib import Path
import pandas as pd
import streamlit as st

# Allow import of your simulation logic file
REPO_ROOT = Path(__file__).resolve().parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import miniscotSupplychainsimulation as runner  # uses your mock simulator

st.set_page_config(page_title="miniSCOT Supply Chain Simulator", layout="wide")
st.title("miniSCOT: Supply Chain Simulation")

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

# Tabs
tab_sim, tab_about, tab_builder = st.tabs(["Simulator", "About miniSCOT", "Network Builder"])

# ------------------------
# Simulator Tab
# ------------------------
with tab_sim:
    st.subheader("Config")
    st.json(cfg)

    if st.button("Run simulation", type="primary"):
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

# ------------------------
# About miniSCOT Tab
# ------------------------
with tab_about:
    st.subheader("About miniSCOT")
    st.markdown("""
**miniSCOT** (Supply Chain Optimization Technologies) is an open-source, Python-based simulation environment developed by Amazon.  
It helps model, test, and optimize supply-chain networks using configurable modules.

### Key Benefits
- **Modular Design:** Mix and match modules for demand, inventory, transportation, and fulfillment.
- **Scenario Testing:** Explore "what-if" analyses such as demand spikes, delays, or new warehouse additions.
- **Algorithm Evaluation:** Compare rule-based versus ML-based inventory and routing policies.
- **Scalability:** Simulate complex multi-echelon networks efficiently.
- **Decision Support:** Quantify cost-service trade-offs before real-world changes.

### How It Works
1. **Network Setup:** Define suppliers, warehouses, and customers connected by transport lanes.
2. **Demand Generation:** Specify stochastic demand at downstream nodes.
3. **Policies:** Choose inventory and transportation policies.
4. **Simulation Engine:** Discrete-event simulation processes receipts, shipments, and orders.
5. **Performance Metrics:** Fill rate, stockouts, total cost, lead time, and service level.

### Example Questions miniSCOT Can Answer
- How much safety stock is required to reach a 98% service level?
- What’s the cost impact of a 1-day increase in supplier lead time?
- Does adding a regional warehouse reduce total logistics cost?
- How does mode switching (truck ↔ air) change service and cost?

**Typical Elements Modeled**
- **Suppliers:** MOQ, reliability, lead-time variability.
- **Warehouses (DCs):** Reorder logic, safety stock, throughput.
- **Customers:** Demand profiles, delivery targets.
- **Transportation:** Cost per unit, mode, and transit time.
- **Inventory Policies:** Base-stock, (s, S), reorder point, or ML-driven rules.

---
**Common Metrics**
- Fill Rate — % of demand fulfilled immediately  
- Stockouts — Count of unmet demand  
- Lead Time — Time from order to receipt  
- Inventory Cost — $ per unit per day  
- Transport Cost — $ per unit shipped  
- Total Cost — Transport + Inventory + Penalties  
- Service Level — % of orders on time or in full
""")

# ------------------------
# Network Builder Tab
# ------------------------
with tab_builder:
    st.subheader("Network Builder")
    st.caption("Define suppliers, warehouses, customers, and transport lanes. Export as YAML or CSV.")

    suppliers_df = st.session_state.get("suppliers_df", pd.DataFrame([
        {"name": "Supplier_A", "lead_time_mean": 3.0, "lead_time_sd": 0.5, "moq": 100},
    ]))
    warehouses_df = st.session_state.get("warehouses_df", pd.DataFrame([
        {"name": "DC_Atlanta", "safety_stock": 500, "reorder_point": 800},
    ]))
    customers_df = st.session_state.get("customers_df", pd.DataFrame([
        {"name": "Store_1", "avg_daily_demand": 120},
    ]))
    lanes_df = st.session_state.get("lanes_df", pd.DataFrame([
        {"origin": "Supplier_A", "destination": "DC_Atlanta", "mode": "TL", "cost_per_unit": 1.2, "lt_mean": 2.0},
        {"origin": "DC_Atlanta", "destination": "Store_1", "mode": "Van", "cost_per_unit": 0.8, "lt_mean": 1.0},
    ]))

    st.markdown("Suppliers")
    suppliers_df = st.data_editor(suppliers_df, num_rows="dynamic", use_container_width=True)

    st.markdown("Warehouses")
    warehouses_df = st.data_editor(warehouses_df, num_rows="dynamic", use_container_width=True)

    st.markdown("Customers")
    customers_df = st.data_editor(customers_df, num_rows="dynamic", use_container_width=True)

    st.markdown("Transport Lanes")
    st.caption("Define arcs between nodes with mode, cost, and lead time.")
    lanes_df = st.data_editor(lanes_df, num_rows="dynamic", use_container_width=True)

    st.session_state["suppliers_df"] = suppliers_df
    st.session_state["warehouses_df"] = warehouses_df
    st.session_state["customers_df"] = customers_df
    st.session_state["lanes_df"] = lanes_df

    col1, col2 = st.columns(2)
    col1.download_button(
        "Download CSV (lanes)",
        lanes_df.to_csv(index=False).encode("utf-8"),
        file_name="lanes.csv",
        mime="text/csv",
    )

    try:
        import yaml as _yaml
        net_cfg = {
            "suppliers": suppliers_df.to_dict(orient="records"),
            "warehouses": warehouses_df.to_dict(orient="records"),
            "customers": customers_df.to_dict(orient="records"),
            "lanes": lanes_df.to_dict(orient="records"),
        }
        yaml_str = _yaml.safe_dump(net_cfg, sort_keys=False)
        col2.download_button(
            "Download YAML (network)",
            yaml_str.encode("utf-8"),
            file_name="network.yaml",
            mime="text/plain",
        )
        st.code(yaml_str, language="yaml")
    except Exception:
        st.info("PyYAML not found. Add 'PyYAML' to requirements.txt to enable YAML export.")
