# pages/03_Fleet_Dispatch.py
# Transportation & Fleet Flow simulation (fleet dispatch & on-time performance)
import math, random
import numpy as np
import pandas as pd
import simpy
import streamlit as st

# --- Sidebar scrollbar fix ---
st.markdown(
    "<style>section[data-testid='stSidebar'] > div {height:100vh; overflow-y:auto;}</style>",
    unsafe_allow_html=True,
)

st.set_page_config(page_title="Fleet Dispatch (SimPy)", layout="wide")
st.title("Transportation & Fleet Flow (SimPy)")

with st.expander("What this solves & how to run", expanded=True):
    st.markdown("""
**Objective.** Simulate a delivery fleet serving jobs that arrive stochastically. Each job needs a vehicle for
travel + service time. We measure **on-time rate**, **lateness**, **throughput**, and **vehicle utilization**.

**Good for answering:**  
• How many vehicles do we need to achieve ≥X% on-time?  
• What’s the impact of travel-time variability or demand spikes?  

**How to run:** set parameters on the left, click **Run simulation**.
""")

with st.sidebar:
    st.header("Parameters")
    sim_hours = st.number_input("Simulation hours", 1, 72, 8)
    num_vehicles = st.number_input("Vehicles", 1, 500, 25)
    jobs_per_hour = st.number_input("Jobs per hour (avg)", 0, 5000, 120)
    base_speed_kmph = st.number_input("Avg speed (km/h)", 1.0, 200.0, 35.0, 0.5)
    km_per_job_mean = st.number_input("Mean distance/job (km)", 1.0, 500.0, 12.0, 0.5)
    km_per_job_sd = st.number_input("SD distance/job (km)", 0.0, 300.0, 6.0, 0.5)
    onsite_service_min = st.number_input("Onsite service mean (min)", 0.0, 240.0, 10.0, 1.0)
    sla_minutes = st.number_input("Promised SLA (minutes)", 10.0, 1440.0, 120.0, 5.0)
    seed = st.number_input("Random seed", 0, 999999, 42)

def normal_pos(mean, sd):
    x = np.random.normal(mean, sd)
    while x <= 0: x = np.random.normal(mean, sd)
    return x

def run_sim(cfg):
    np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    sim_min = int(cfg["sim_hours"] * 60)
    env = simpy.Environment()
    fleet = simpy.Resource(env, capacity=cfg["num_vehicles"])

    jobs = []
    util_busy_time = 0.0
    last_t = 0.0
    busy = 0

    def note_change():
        nonlocal last_t, util_busy_time
        now = env.now
        util_busy_time += busy * (now - last_t)
        last_t = now

    def acquire():
        nonlocal busy
        note_change(); busy += 1

    def release():
        nonlocal busy
        note_change(); busy -= 1

    lam = cfg["jobs_per_hour"] / 60.0  # per minute

    def job_generator():
        jid = 0
        while env.now < sim_min:
            ia = np.random.exponential(1.0/lam) if lam > 0 else math.inf
            if env.now + ia > sim_min: break
            yield env.timeout(ia)
            jid += 1
            env.process(handle_job(jid, env.now))

    def handle_job(jid, arrival_min):
        nonlocal busy
        with fleet.request() as req:
            start_q = env.now
            yield req
            acquire()
            q_wait = env.now - start_q

            dist_km = normal_pos(cfg["km_per_job_mean"], cfg["km_per_job_sd"])
            travel_h = dist_km / max(1e-6, cfg["base_speed_kmph"])
            travel_min = travel_h * 60.0
            service_min = cfg["onsite_service_min"]

            total_time = travel_min + service_min
            yield env.timeout(total_time)
            release()

        sojourn = (env.now - arrival_min)
        jobs.append({
            "job_id": jid,
            "arrival_min": arrival_min,
            "queue_wait_min": q_wait,
            "distance_km": dist_km,
            "travel_min": travel_min,
            "service_min": service_min,
            "sojourn_min": sojourn,
            "met_sla": 1 if sojourn <= cfg["sla_minutes"] else 0
        })

    env.process(job_generator())
    env.run(until=sim_min)
    # close utilization window
    note_change()

    df = pd.DataFrame(jobs)
    metrics = {
        "throughput_jobs": int(len(df)),
        "avg_queue_wait_min": float(df["queue_wait_min"].mean()) if len(df) else 0.0,
        "avg_sojourn_min": float(df["sojourn_min"].mean()) if len(df) else 0.0,
        "on_time_rate": float(df["met_sla"].mean()) if len(df) else 0.0,
        "vehicle_utilization": util_busy_time / (cfg["num_vehicles"] * sim_min) if sim_min > 0 else 0.0,
    }
    return metrics, df

cfg = dict(sim_hours=sim_hours, num_vehicles=num_vehicles, jobs_per_hour=jobs_per_hour,
           base_speed_kmph=base_speed_kmph, km_per_job_mean=km_per_job_mean, km_per_job_sd=km_per_job_sd,
           onsite_service_min=onsite_service_min, sla_minutes=sla_minutes, seed=seed)

if st.button("Run simulation", type="primary"):
    m, df = run_sim(cfg)
    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("Jobs", m["throughput_jobs"])
    c2.metric("On-time rate", f"{m['on_time_rate']:.1%}")
    c3.metric("Avg queue (min)", f"{m['avg_queue_wait_min']:.1f}")
    c4.metric("Avg sojourn (min)", f"{m['avg_sojourn_min']:.1f}")
    c5.metric("Fleet util.", f"{m['vehicle_utilization']:.1%}")

    st.subheader("Job log")
    st.dataframe(df, use_container_width=True, height=380)
    st.download_button("Download CSV", df.to_csv(index=False).encode("utf-8"),
                       file_name="fleet_dispatch_jobs.csv", mime="text/csv")
