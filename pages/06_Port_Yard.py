# pages/06_Port_Yard.py
# Port / Yard operations with berths & cranes (resource contention & turnaround)

import math, random
import numpy as np
import pandas as pd
import simpy
import streamlit as st

# Must be first:
st.set_page_config(page_title="Port / Yard (SimPy)", layout="wide")

# Sidebar scrollbar fix
st.markdown(
    "<style>section[data-testid='stSidebar'] > div {height:100vh; overflow-y:auto;}</style>",
    unsafe_allow_html=True,
)

st.title("Port / Yard / Cross-Dock (SimPy)")

with st.expander("What this solves & how to run", expanded=True):
    st.markdown("""
**Objective.** Simulate vessels/trailers arriving to a limited number of **berths/doors** and being served by
limited **cranes/forklifts**. Measures **queue time**, **turnaround**, **utilization** of resources.

**Good for:**  
• How many berths/cranes are needed at peak?  
• What policies/throughput are feasible under variability?

**Run:** set parameters, click **Run simulation**.
""")

with st.sidebar:
    st.header("Parameters")
    sim_hours = st.number_input("Simulation hours", 1, 168, 48)
    arrivals_per_hour = st.number_input("Arrivals/hour", 0, 1000, 6)
    num_berths = st.number_input("Berths/Doors", 1, 200, 4)
    num_cranes = st.number_input("Cranes/Forklifts", 1, 500, 6)
    moves_mean = st.number_input("Mean moves per vessel", 1.0, 1e6, 500.0, 1.0)
    moves_sd = st.number_input("SD moves per vessel", 0.0, 1e6, 200.0, 1.0)
    move_rate_per_crane = st.number_input("Moves/min per crane", 0.01, 1000.0, 1.2, 0.01)
    seed = st.number_input("Random seed", 0, 999999, 42)

def pos_normal(mu, sd):
    x = np.random.normal(mu, sd)
    while x <= 0: x = np.random.normal(mu, sd)
    return x

def run_port(cfg):
    np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    sim_min = int(cfg["sim_hours"] * 60)
    env = simpy.Environment()
    berths = simpy.Resource(env, capacity=cfg["num_berths"])
    cranes = simpy.Resource(env, capacity=cfg["num_cranes"])
    lam = cfg["arrivals_per_hour"] / 60.0

    util_b_busy = 0.0; util_c_busy = 0.0
    last_t_b = 0.0; last_t_c = 0.0
    busy_b = 0; busy_c = 0
    def note_b():
        nonlocal last_t_b, util_b_busy
        now = env.now; util_b_busy += busy_b * (now - last_t_b); last_t_b = now
    def note_c():
        nonlocal last_t_c, util_c_busy
        now = env.now; util_c_busy += busy_c * (now - last_t_c); last_t_c = now
    def acquire_b(): 
        nonlocal busy_b; note_b(); busy_b += 1
    def release_b():
        nonlocal busy_b; note_b(); busy_b -= 1
    def acquire_c():
        nonlocal busy_c; note_c(); busy_c += 1
    def release_c():
        nonlocal busy_c; note_c(); busy_c -= 1

    vessels = []

    def generator():
        vid = 0
        while env.now < sim_min:
            ia = np.random.exponential(1.0/lam) if lam>0 else math.inf
            if env.now + ia > sim_min: break
            yield env.timeout(ia)
            vid += 1
            env.process(handle(vid, env.now))

    def handle(vid, arr_min):
        with berths.request() as bq:
            start_q = env.now
            yield bq
            acquire_b()
            q_wait = env.now - start_q

            # one crane per vessel for simplicity
            with cranes.request() as cq:
                yield cq
                acquire_c()
                moves = pos_normal(cfg["moves_mean"], cfg["moves_sd"])
                service_min = moves / max(1e-6, cfg["move_rate_per_crane"])
                yield env.timeout(service_min)
                release_c()

            release_b()
        vessels.append({
            "vessel_id": vid,
            "arrival_min": arr_min,
            "queue_wait_min": q_wait,
            "service_min": service_min,
            "turnaround_min": (env.now - arr_min)
        })

    env.process(generator())
    env.run(until=sim_min)
    note_b(); note_c()

    df = pd.DataFrame(vessels)
    metrics = {
        "throughput": int(len(df)),
        "avg_queue_min": float(df["queue_wait_min"].mean()) if len(df) else 0.0,
        "avg_turnaround_min": float(df["turnaround_min"].mean()) if len(df) else 0.0,
        "berth_utilization": util_b_busy / (cfg["num_berths"] * sim_min) if sim_min>0 else 0.0,
        "crane_utilization": util_c_busy / (cfg["num_cranes"] * sim_min) if sim_min>0 else 0.0,
    }
    return metrics, df

cfg = dict(sim_hours=sim_hours, arrivals_per_hour=arrivals_per_hour,
           num_berths=num_berths, num_cranes=num_cranes, moves_mean=moves_mean,
           moves_sd=moves_sd, move_rate_per_crane=move_rate_per_crane, seed=seed)

if st.button("Run simulation", type="primary"):
    m, df = run_port(cfg)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Vessels", m["throughput"])
    c2.metric("Avg queue (min)", f"{m['avg_queue_min']:.1f}")
    c3.metric("Avg turnaround (min)", f"{m['avg_turnaround_min']:.1f}")
    c4.metric("Berth util.", f"{m['berth_utilization']:.1%}")
    c5.metric("Crane util.", f"{m['crane_utilization']:.1%}")
    st.subheader("Vessel log")
    st.dataframe(df, use_container_width=True, height=380)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="port_vessels.csv", mime="text/csv")
