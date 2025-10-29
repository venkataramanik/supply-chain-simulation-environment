# pages/04_Production_Scheduling.py
# Production & Assembly Line with machine breakdowns & a single repairman (preemptive)

import math, random
import numpy as np
import pandas as pd
import simpy
import streamlit as st

# ❗ Must be the first Streamlit call:
st.set_page_config(page_title="Production & Assembly (SimPy)", layout="wide")

# (CSS fix for sidebar scroll)
st.markdown(
    "<style>section[data-testid='stSidebar'] > div {height:100vh; overflow-y:auto;}</style>",
    unsafe_allow_html=True,
)

st.title("Production & Assembly Line (SimPy)")

with st.expander("What this solves & how to run", expanded=True):
    st.markdown("""
**Objective.** Model a small shop of identical machines that produce parts. Machines **break randomly** and
a single **repairman** (PreemptiveResource) fixes them. Measures **throughput**, **downtime**, **WIP delays**, and
**repairman utilization**.

**Good for:**  
• What’s the impact of MTTF/repair-time on throughput?  
• Do we need a second repair resource?  

**Run:** set parameters on the left, click **Run simulation**.
""")

with st.sidebar:
    st.header("Parameters")
    sim_hours = st.number_input("Simulation hours", 1, 168, 24)
    num_machines = st.number_input("Machines", 1, 200, 10)
    pt_mean = st.number_input("Proc. time mean (min)", 0.1, 600.0, 10.0, 0.1)
    pt_sigma = st.number_input("Proc. time SD (min)", 0.0, 300.0, 2.0, 0.1)
    mttf_min = st.number_input("Mean time to failure (min)", 10.0, 100000.0, 300.0, 1.0)
    repair_min = st.number_input("Repair time (min)", 1.0, 1440.0, 30.0, 1.0)
    other_job_min = st.number_input("Other task dur (min)", 1.0, 240.0, 30.0, 1.0)
    seed = st.number_input("Random seed", 0, 999999, 42)

def time_per_part(mu, sd):
    x = np.random.normal(mu, sd)
    while x <= 0: x = np.random.normal(mu, sd)
    return x

def run_shop(cfg):
    np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    sim_min = int(cfg["sim_hours"] * 60)
    env = simpy.Environment()
    repairman = simpy.PreemptiveResource(env, capacity=1)

    parts_made = [0]*cfg["num_machines"]
    broken = [False]*cfg["num_machines"]

    util_busy_time = 0.0; last_t = 0.0; busy = 0
    def note_change():
        nonlocal last_t, util_busy_time
        now = env.now
        util_busy_time += busy * (now - last_t)
        last_t = now
    def acquire():
        nonlocal busy; note_change(); busy += 1
    def release():
        nonlocal busy; note_change(); busy -= 1

    def machine(i):
        while True:
            done_in = time_per_part(cfg["pt_mean"], cfg["pt_sigma"])
            while done_in > 0:
                try:
                    start = env.now
                    yield env.timeout(done_in)
                    done_in = 0
                except simpy.Interrupt:
                    broken[i] = True
                    done_in -= env.now - start
                    with repairman.request(priority=1) as req:
                        yield req
                        acquire()
                        yield env.timeout(cfg["repair_min"])
                        release()
                    broken[i] = False
            parts_made[i] += 1

    def break_machine(i):
        lam = 1.0 / cfg["mttf_min"]
        while True:
            ttf = np.random.exponential(1.0/lam)
            yield env.timeout(ttf)
            if not broken[i]:
                procs[i].interrupt()

    def other_job():
        while True:
            with repairman.request(priority=2) as req:
                yield req
                acquire()
                try:
                    yield env.timeout(cfg["other_job_min"])
                except simpy.Interrupt:
                    pass
                finally:
                    release()

    procs = []
    for i in range(cfg["num_machines"]):
        p = env.process(machine(i))
        procs.append(p)
        env.process(break_machine(i))
    env.process(other_job())
    env.run(until=sim_min)
    note_change()

    df = pd.DataFrame({"machine": [f"M{i}" for i in range(cfg["num_machines"])],
                       "parts_made": parts_made})
    metrics = {
        "throughput_parts": int(sum(parts_made)),
        "avg_parts_per_machine": float(np.mean(parts_made)) if parts_made else 0.0,
        "repair_utilization": util_busy_time / sim_min if sim_min>0 else 0.0,
    }
    return metrics, df

cfg = dict(sim_hours=sim_hours, num_machines=num_machines, pt_mean=pt_mean, pt_sigma=pt_sigma,
           mttf_min=mttf_min, repair_min=repair_min, other_job_min=other_job_min, seed=seed)

if st.button("Run simulation", type="primary"):
    m, df = run_shop(cfg)
    c1,c2,c3 = st.columns(3)
    c1.metric("Total parts", m["throughput_parts"])
    c2.metric("Avg parts/machine", f"{m['avg_parts_per_machine']:.1f}")
    c3.metric("Repair util.", f"{m['repair_utilization']:.1%}")
    st.subheader("Per-machine output")
    st.dataframe(df, use_container_width=True, height=380)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="production_parts.csv", mime="text/csv")
