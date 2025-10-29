import streamlit as st, pandas as pd
from sim_core import run_simpy_warehouse
from viz import build_plotly_animation, warehouse_layout

st.set_page_config(page_title="SimPy Warehouse", layout="wide")
st.title("Warehouse Simulation (SimPy)")

# Sidebar inputs
with st.sidebar:
    st.header("Parameters")
    sim_hours = st.number_input("Simulation hours", 1, 72, 8)
    num_dock_doors = st.number_input("Dock doors", 1, 20, 2)
    num_forklifts = st.number_input("Forklifts", 1, 20, 2)
    num_pickers = st.number_input("Pickers", 1, 50, 3)
    inbound_trucks_per_hour = st.number_input("Inbound trucks/hour", 0, 60, 4)
    orders_per_hour = st.number_input("Orders/hour", 0, 1000, 60)
    unload_mean_min = st.number_input("Unload mean (min)", 1.0, 240.0, 25.0, 1.0)
    unload_sd_min = st.number_input("Unload SD (min)", 0.0, 120.0, 5.0, 0.5)
    putaway_minutes_per_truck = st.number_input("Put-away (min/truck)", 0.0, 120.0, 10.0, 0.5)
    lines_per_order_mean = st.number_input("Mean lines/order (exp)", 0.5, 50.0, 3.0, 0.5)
    pick_time_per_line_min = st.number_input("Pick time/line (min)", 0.1, 30.0, 1.5, 0.1)
    pack_time_min = st.number_input("Pack time (min)", 0.0, 60.0, 2.0, 0.5)
    order_sla_minutes = st.number_input("Order SLA (minutes)", 10.0, 600.0, 120.0, 5.0)

cfg = {
    "sim_hours": sim_hours,
    "num_dock_doors": num_dock_doors,
    "num_forklifts": num_forklifts,
    "num_pickers": num_pickers,
    "inbound_trucks_per_hour": inbound_trucks_per_hour,
    "orders_per_hour": orders_per_hour,
    "unload_mean_min": unload_mean_min,
    "unload_sd_min": unload_sd_min,
    "putaway_minutes_per_truck": putaway_minutes_per_truck,
    "lines_per_order_mean": lines_per_order_mean,
    "pick_time_per_line_min": pick_time_per_line_min,
    "pack_time_min": pack_time_min,
    "order_sla_minutes": order_sla_minutes,
}

col1, col2 = st.columns([1,1])

if col1.button("Run warehouse sim"):
    metrics, orders_df, trucks_df = run_simpy_warehouse(cfg)
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Orders", metrics["throughput_orders"])
    k2.metric("Avg cycle (min)", f"{metrics['avg_order_cycle_time_min']:.1f}")
    k3.metric("SLA hit rate", f"{metrics['sla_hit_rate']:.1%}")
    k4.metric("Trucks", metrics["throughput_trucks"])
    u1, u2, u3 = st.columns(3)
    u1.metric("Dock util.", f"{metrics['dock_utilization']:.1%}")
    u2.metric("Forklift util.", f"{metrics['forklift_utilization']:.1%}")
    u3.metric("Picker util.", f"{metrics['picker_utilization']:.1%}")

    st.subheader("Orders")
    st.dataframe(orders_df, use_container_width=True, height=260)
    st.subheader("Inbound trucks")
    st.dataframe(trucks_df, use_container_width=True, height=260)

if col2.button("Animate layout"):
    # quick lightweight animation using synthetic frames:
    # If you want SimPy-driven animation, integrate your logged frames (we can add next).
    shapes, L = warehouse_layout(n_docks=num_dock_doors)
    # Minimal placeholder: a single empty frame to show layout
    import pandas as pd
    frames = [pd.DataFrame(columns=["t","agent","type","x","y"]).assign(t=0)]
    fig = build_plotly_animation(frames, shapes, L)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
