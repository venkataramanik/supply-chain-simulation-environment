# pages/05_Inventory_Replenishment.py
# Single-echelon continuous review (s, S) policy with stochastic demand & lead time

import math, random
import numpy as np
import pandas as pd
import simpy
import streamlit as st

# Must be first:
st.set_page_config(page_title="Inventory Replenishment (SimPy)", layout="wide")

# Sidebar scrollbar fix
st.markdown(
    "<style>section[data-testid='stSidebar'] > div {height:100vh; overflow-y:auto;}</style>",
    unsafe_allow_html=True,
)

st.title("Inventory & Replenishment (SimPy) — (s, S) Policy")

with st.expander("What this solves & how to run", expanded=True):
    st.markdown("""
**Objective.** Simulate a single-echelon item with continuous review **(s, S)** policy.
Daily demand is stochastic; when **on-hand ≤ s**, place an order to raise inventory to **S**.
Lead time is random. Track **service level**, **stockouts**, **avg inventory**, **orders placed**.

**Good for:**  
• Choosing **s** and **S** for a target service level.  
• Understanding impact of demand & lead-time variability.

**Run:** set parameters, click **Run simulation**.
""")

with st.sidebar:
    st.header("Parameters")
    sim_days = st.number_input("Simulation days", 7, 3650, 365)
    demand_mean = st.number_input("Mean daily demand (Poisson λ)", 0.0, 1e6, 20.0, 1.0)
    s_level = st.number_input("Reorder point s", 0, 1000000, 50, 1)
    S_level = st.number_input("Order-up-to S", 0, 1000000, 150, 1)
    lead_time_mean = st.number_input("Lead time mean (days)", 0.0, 365.0, 5.0, 0.5)
    lead_time_sd = st.number_input("Lead time SD (days)", 0.0, 365.0, 1.0, 0.5)
    init_inventory = st.number_input("Initial on-hand", 0, 1000000, 150, 1)
    seed = st.number_input("Random seed", 0, 999999, 42)

def normal_pos(mean, sd):
    x = np.random.normal(mean, sd)
    while x <= 0: x = np.random.normal(mean, sd)
    return x

def run_inventory(cfg):
    np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    env = simpy.Environment()

    on_hand = cfg["init_inventory"]
    pipeline = []  # list of (arrival_day, qty)
    backorders = 0
    rec = []

    def day_process():
        nonlocal on_hand, backorders, pipeline
        for d in range(int(cfg["sim_days"])):
            # receive arrivals
            arrivals = [q for (t,q) in pipeline if t == d]
            if arrivals:
                on_hand += sum(arrivals)
                pipeline = [(t,q) for (t,q) in pipeline if t != d]

            # demand
            dem = np.random.poisson(cfg["demand_mean"])
            shipped = min(on_hand, dem)
            on_hand -= shipped
            unfilled = dem - shipped
            backorders += unfilled

            # policy
            if on_hand <= cfg["s_level"]:
                order_qty = max(0, cfg["S_level"] - on_hand)
                if order_qty > 0:
                    lt = int(round(normal_pos(cfg["lead_time_mean"], cfg["lead_time_sd"])))
                    arr_day = d + max(1, lt)
                    pipeline.append((arr_day, order_qty))

            rec.append({
                "day": d,
                "demand": dem,
                "shipped": shipped,
                "unfilled": unfilled,
                "on_hand_end": on_hand,
                "pipeline_qty": sum(q for _,q in pipeline)
            })
            yield env.timeout(1)

    env.process(day_process())
    env.run(until=int(cfg["sim_days"]))

    df = pd.DataFrame(rec)
    service_level = 1.0 - (df["unfilled"].sum() / max(1, df["demand"].sum()))
    orders_placed = int(((df["on_hand_end"].shift(1) > cfg["s_level"]) &
                         (df["on_hand_end"] <= cfg["s_level"])).sum())

    metrics = {
        "service_level": float(service_level),
        "stockout_days": int((df["unfilled"] > 0).sum()),
        "avg_on_hand": float(df["on_hand_end"].mean()),
        "orders_placed": orders_placed,
        "total_unfilled": int(df["unfilled"].sum()),
    }
    return metrics, df

cfg = dict(sim_days=sim_days, demand_mean=demand_mean, s_level=s_level, S_level=S_level,
           lead_time_mean=lead_time_mean, lead_time_sd=lead_time_sd, init_inventory=init_inventory, seed=seed)

if st.button("Run simulation", type="primary"):
    m, df = run_inventory(cfg)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Service level", f"{m['service_level']:.1%}")
    c2.metric("Stockout days", m["stockout_days"])
    c3.metric("Avg on-hand", f"{m['avg_on_hand']:.1f}")
    c4.metric("Orders placed", m["orders_placed"])
    c5.metric("Total unfilled", m["total_unfilled"])

    st.subheader("Daily record")
    st.dataframe(df, use_container_width=True, height=380)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="inventory_daily.csv", mime="text/csv")
