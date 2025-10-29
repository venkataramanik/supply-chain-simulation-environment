# pages/07_Order_to_Delivery.py
# End-to-end order -> pick -> pack -> ship pipeline with SLA

import math, random
import numpy as np
import pandas as pd
import simpy
import streamlit as st

# Must be first:
st.set_page_config(page_title="Order-to-Delivery (SimPy)", layout="wide")

# Sidebar scrollbar fix
st.markdown(
    "<style>section[data-testid='stSidebar'] > div {height:100vh; overflow-y:auto;}</style>",
    unsafe_allow_html=True,
)

st.title("Order-to-Delivery Pipeline (SimPy)")

with st.expander("What this solves & how to run", expanded=True):
    st.markdown("""
**Objective.** Model an end-to-end **order pipeline** with three stations: **Pickers → Packers → Carriers**.
Orders arrive randomly, consume resources at each stage, and complete when shipped. Track **cycle time**, **SLA hit**,
and **station utilizations**.

**Good for:**  
• Staffing each stage to meet SLA.  
• Understanding how bottlenecks shift with demand.

**Run:** set parameters, click **Run simulation**.
""")

with st.sidebar:
    st.header("Parameters")
    sim_hours = st.number_input("Simulation hours", 1, 168, 12)
    orders_per_hour = st.number_input("Orders/hour", 0, 10000, 300)
    num_pickers = st.number_input("Pickers", 1, 1000, 20)
    num_packers = st.number_input("Packers", 1, 1000, 12)
    num_carriers = st.number_input("Carrier docks", 1, 1000, 8)
    pick_time_mean = st.number_input("Pick time mean (min)", 0.1, 240.0, 1.5, 0.1)
    pack_time_mean = st.number_input("Pack time mean (min)", 0.1, 240.0, 1.0, 0.1)
    ship_time_mean = st.number_input("Ship time mean (min)", 0.1, 240.0, 3.0, 0.1)
    sla_minutes = st.number_input("Order SLA (minutes)", 10.0, 1440.0, 120.0, 5.0)
    seed = st.number_input("Random seed", 0, 999999, 42)

def pos_exp(mean):
    lam = 1.0/max(1e-6, mean)
    return np.random.exponential(1.0/lam)

def run_pipeline(cfg):
    np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    sim_min = int(cfg["sim_hours"] * 60)
    env = simpy.Environment()
    pick = simpy.Resource(env, capacity=cfg["num_pickers"])
    pack = simpy.Resource(env, capacity=cfg["num_packers"])
    ship = simpy.Resource(env, capacity=cfg["num_carriers"])

    busy = {"pick":0,"pack":0,"ship":0}
    last = {"pick":0.0,"pack":0.0,"ship":0.0}
    BT = {"pick":0.0,"pack":0.0,"ship":0.0}

    def note(res):
        now = env.now; BT[res] += busy[res]*(now-last[res]); last[res]=now
    def acquire(res): note(res); busy[res]+=1
    def release(res): note(res); busy[res]-=1

    lam = cfg["orders_per_hour"]/60.0
    orders = []

    def gen():
        oid = 0
        while env.now < sim_min:
            ia = np.random.exponential(1.0/lam) if lam>0 else math.inf
            if env.now + ia > sim_min: break
            yield env.timeout(ia)
            oid += 1
            env.process(handle(oid, env.now))

    def handle(oid, t0):
        # pick
        with pick.request() as r1:
            q1 = env.now; yield r1; acquire("pick")
            w1 = env.now - q1; yield env.timeout(pos_exp(cfg["pick_time_mean"])); release("pick")
        # pack
        with pack.request() as r2:
            q2 = env.now; yield r2; acquire("pack")
            w2 = env.now - q2; yield env.timeout(pos_exp(cfg["pack_time_mean"])); release("pack")
        # ship
        with ship.request() as r3:
            q3 = env.now; yield r3; acquire("ship")
            w3 = env.now - q3; yield env.timeout(pos_exp(cfg["ship_time_mean"])); release("ship")

        sojourn = env.now - t0
        orders.append({
            "order_id": oid,
            "wait_pick": w1, "wait_pack": w2, "wait_ship": w3,
            "cycle_time_min": sojourn,
            "met_sla": 1 if sojourn <= cfg["sla_minutes"] else 0
        })

    env.process(gen())
    env.run(until=sim_min)
    for k in busy: note(k)

    df = pd.DataFrame(orders)
    metrics = {
        "throughput": int(len(df)),
        "avg_cycle_min": float(df["cycle_time_min"].mean()) if len(df) else 0.0,
        "sla_rate": float(df["met_sla"].mean()) if len(df) else 0.0,
        "pick_util": BT["pick"]/(cfg["num_pickers"]*sim_min) if sim_min>0 else 0.0,
        "pack_util": BT["pack"]/(cfg["num_packers"]*sim_min) if sim_min>0 else 0.0,
        "ship_util": BT["ship"]/(cfg["num_carriers"]*sim_min) if sim_min>0 else 0.0,
    }
    return metrics, df

cfg = dict(sim_hours=sim_hours, orders_per_hour=orders_per_hour,
           num_pickers=num_pickers, num_packers=num_packers, num_carriers=num_carriers,
           pick_time_mean=pick_time_mean, pack_time_mean=pack_time_mean, ship_time_mean=ship_time_mean,
           sla_minutes=sla_minutes, seed=seed)

if st.button("Run simulation", type="primary"):
    m, df = run_pipeline(cfg)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Orders", m["throughput"])
    c2.metric("Avg cycle (min)", f"{m['avg_cycle_min']:.1f}")
    c3.metric("SLA rate", f"{m['sla_rate']:.1%}")
    c4.metric("Pick util.", f"{m['pick_util']:.1%}")
    c5.metric("Pack util.", f"{m['pack_util']:.1%}")
    st.subheader("Order log")
    st.dataframe(df, use_container_width=True, height=380)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="o2d_orders.csv", mime="text/csv")
