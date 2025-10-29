# pages/02_SimPy_Warehouse.py
# Full Streamlit page for Warehouse (SimPy) with:
#  - Styled blurb explaining SimPy & discrete-event simulation (no AI icons)
#  - KPI simulation
#  - SimPy-driven animation

import streamlit as st
import pandas as pd

from sim_core import run_simpy_warehouse, run_simpy_warehouse_animated
from viz import build_plotly_animation, warehouse_layout

st.set_page_config(page_title="Warehouse Simulation (SimPy)", layout="wide")

# =========================
# Styled Blurb (No Icons)
# =========================
BLURB_HTML = """
<div style="font-family: Inter, system-ui, -apple-system, Segoe UI, Roboto, Helvetica, Arial, sans-serif;
            background:#f8fafc; border:1px solid #e5e7eb; border-radius:16px; padding:22px 24px; margin-bottom:18px;">
  <h1 style="font-size:28px; line-height:1.25; margin:0 0 8px 0; font-weight:750;">
    Warehouse Simulation with SimPy
  </h1>
  <p style="font-size:16px; color:#334155; margin:0 0 14px 0;">
    <strong>SimPy</strong> is a lightweight, Python-based <em>discrete-event simulation</em> framework.
    Instead of simulating every second in real time, it advances from one event to the next
    (e.g., a truck arriving, a dock becoming free, or a picker completing a job).
    This makes it ideal for analyzing complex operations quickly and repeatably.
  </p>
  <div style="display:grid; grid-template-columns: repeat(3, minmax(0,1fr)); gap:12px;">
    <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px;">
      <div style="font-size:14px; font-weight:700; color:#111827; margin-bottom:6px;">What this models</div>
      <ul style="padding-left:18px; margin:0; font-size:14px; color:#374151;">
        <li>Inbound: trucks ⇒ dock queue ⇒ unload (forklifts) ⇒ put-away</li>
        <li>Outbound: orders ⇒ pick (pickers) ⇒ pack ⇒ complete (SLA tracked)</li>
        <li>Finite resources: dock doors, forklifts, pickers</li>
      </ul>
    </div>
    <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px;">
      <div style="font-size:14px; font-weight:700; color:#111827; margin-bottom:6px;">Why discrete-event?</div>
      <ul style="padding-left:18px; margin:0; font-size:14px; color:#374151;">
        <li>Fast &amp; scalable for “what-if” scenarios</li>
        <li>Captures real queueing dynamics &amp; contention</li>
        <li>Transparent logic, reproducible results</li>
      </ul>
    </div>
    <div style="background:#ffffff; border:1px solid #e5e7eb; border-radius:12px; padding:12px 14px;">
      <div style="font-size:14px; font-weight:700; color:#111827; margin-bottom:6px;">Business questions answered</div>
      <ul style="padding-left:18px; margin:0; font-size:14px; color:#374151;">
        <li>How many doors/forklifts/pickers to meet SLA?</li>
        <li>Impact of demand spikes on cycle time &amp; backlog</li>
        <li>Where are bottlenecks (utilization &amp; queue time)?</li>
      </ul>
    </div>
  </div>
  <div style="margin-top:12px; font-size:13px; color:#475569;">
    Key KPIs: Throughput, average cycle time, SLA hit rate, resource utilization, truck queue waits.
  </div>
</div>
"""

st.markdown(BLURB_HTML, unsafe_allow_html=True)

st.title("Warehouse Simulation (SimPy)")

# =========================
# Sidebar parameters
# =========================
with st.sidebar:
    st.header("Parameters")
    sim_hours = st.number_input("Simulation hours", 1, 72, 8)
    num_dock_doors = st.number_input("Dock doors", 1, 20, 2)
    num_forklifts = st.number_input("Forklifts", 1, 20, 2)
    num_pickers = st.number_input("Pickers", 1, 50, 3)

    st.markdown("---")
    inbound_trucks_per_hour = st.number_input("Inbound trucks/hour", 0, 120, 4)
    orders_per_hour = st.number_input("Orders/hour", 0, 2000, 60)

    st.markdown("---")
    unload_mean_min = st.number_input("Unload mean (min)", 1.0, 240.0, 25.0, 1.0)
    unload_sd_min = st.number_input("Unload SD (min)", 0.0, 120.0, 5.0, 0.5)
    putaway_minutes_per_truck = st.number_input("Put-away (min/truck)", 0.0, 120.0, 10.0, 0.5)

    st.markdown("---")
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

# =========================
# Tabs
# =========================
tab1, tab2 = st.tabs(["KPIs & Tables", "Animated Layout"])

# ---- Tab 1: KPI run ----
with tab1:
    st.subheader("Run KPI Simulation")
    if st.button("Run warehouse sim", key="run_kpi"):
        metrics, orders_df, trucks_df = run_simpy_warehouse(cfg)

        # KPIs
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Orders", metrics["throughput_orders"])
        k2.metric("Avg cycle (min)", f"{metrics['avg_order_cycle_time_min']:.1f}")
        k3.metric("SLA hit rate", f"{metrics['sla_hit_rate']:.1%}")
        k4.metric("Trucks", metrics["throughput_trucks"])

        u1, u2, u3 = st.columns(3)
        u1.metric("Dock util.", f"{metrics['dock_utilization']:.1%}")
        u2.metric("Forklift util.", f"{metrics['forklift_utilization']:.1%}")
        u3.metric("Picker util.", f"{metrics['picker_utilization']:.1%}")

        st.markdown("### Orders")
        st.dataframe(orders_df, use_container_width=True, height=260)
        st.download_button(
            "Download orders CSV",
            orders_df.to_csv(index=False).encode("utf-8"),
            file_name="orders.csv",
            mime="text/csv",
        )

        st.markdown("### Inbound trucks")
        st.dataframe(trucks_df, use_container_width=True, height=260)
        st.download_button(
            "Download trucks CSV",
            trucks_df.to_csv(index=False).encode("utf-8"),
            file_name="trucks.csv",
            mime="text/csv",
        )

        # Quick charts (if data present)
        if not orders_df.empty:
            st.markdown("### Order cycle time (min)")
            st.line_chart(orders_df.set_index("order_id")[["cycle_time_min"]])

            st.markdown("### SLA cumulative")
            o2 = orders_df.sort_values("completion_min").copy()
            o2["cum_sla"] = o2["met_sla"].expanding().mean()
            st.line_chart(o2.set_index("completion_min")[["cum_sla"]])

        if not trucks_df.empty:
            st.markdown("### Truck queue wait (min)")
            st.line_chart(trucks_df.set_index("truck_id")[["queue_wait_min"]])

# ---- Tab 2: Animated layout (SimPy-driven) ----
with tab2:
    st.subheader("SimPy-driven animation")
    st.caption("Agents (trucks, forklifts, pickers) move based on real SimPy events — not a mock path.")
    st.markdown("Click to run the simulation and build the animation frames.")
    if st.button("Animate with SimPy", key="run_anim"):
        frames, shapes, labels, m2, orders_df_anim, trucks_df_anim = run_simpy_warehouse_animated(cfg, step_seconds=5)
        fig = build_plotly_animation(frames, shapes, labels)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        # Optional: key animation KPIs
        a1, a2, a3 = st.columns(3)
        a1.metric("Animated orders", m2["orders"])
        a2.metric("Avg cycle (min)", f"{m2['avg_cycle_min']:.1f}")
        a3.metric("SLA rate", f"{m2['sla_rate']:.1%}")

    # Quick static layout preview (no sim), useful for a design glance
    with st.expander("Show empty layout (no simulation)"):
        shapes, labels = warehouse_layout(n_docks=num_dock_doors)
        from plotly.graph_objects import Figure
        fig = Figure()
        fig.update_layout(
            shapes=shapes,
            xaxis=dict(range=[0, labels["dims"][0]], visible=False),
            yaxis=dict(range=[0, labels["dims"][1]], visible=False),
            width=900, height=540, margin=dict(l=10, r=10, t=30, b=10),
        )
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
