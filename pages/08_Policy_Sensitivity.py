# pages/08_Policy_Sensitivity.py
# Batch "what-if" runner: vary parameters and compare KPIs (no visuals)

import math, random
import itertools as it
import numpy as np
import pandas as pd
import simpy
import streamlit as st

# Must be first:
st.set_page_config(page_title="Policy & What-If (SimPy)", layout="wide")

# Sidebar scrollbar fix
st.markdown(
    "<style>section[data-testid='stSidebar'] > div {height:100vh; overflow-y:auto;}</style>",
    unsafe_allow_html=True,
)

st.title("Policy & “What-If” Evaluation (SimPy)")

with st.expander("What this solves & how to run", expanded=True):
    st.markdown("""
**Objective.** Run **many simulations** over a grid of parameters and compare KPIs. This lets you do quick
**sensitivity analysis** or pick an **optimal policy** (e.g., staffing levels to hit SLA).

**Model used:** a simple 2-stage queue (Stage A then Stage B), both exponential service, finite servers.  
**KPIs:** throughput, average cycle time, SLA hit rate, stage utilizations.

**Run:** define ranges on the left → click **Run batch**. The table shows one row per scenario.
""")

with st.sidebar:
    st.header("Base Parameters")
    sim_hours = st.number_input("Simulation hours", 1, 168, 8)
    orders_per_hour = st.number_input("Orders/hour", 0, 10000, 300)
    a_service_mean = st.number_input("Stage A mean service (min)", 0.1, 240.0, 1.5, 0.1)
    b_service_mean = st.number_input("Stage B mean service (min)", 0.1, 240.0, 2.0, 0.1)
    sla_minutes = st.number_input("SLA (minutes)", 10.0, 1440.0, 120.0, 5.0)
    seed = st.number_input("Random seed", 0, 999999, 42)

    st.markdown("---")
    st.subheader("Grid (comma-sep)")
    a_servers_grid = st.text_input("Stage A servers", value="10,12,14")
    b_servers_grid = st.text_input("Stage B servers", value="6,8,10")
    demand_mult_grid = st.text_input("Demand multipliers", value="0.8,1.0,1.2")

def pos_exp(mean):
    lam = 1.0/max(1e-6, mean)
    return np.random.exponential(1.0/lam)

def run_two_stage(cfg):
    np.random.seed(cfg["seed"]); random.seed(cfg["seed"])
    sim_min = int(cfg["sim_hours"]*60)
    env = simpy.Environment()
    A = simpy.Resource(env, capacity=cfg["a_servers"])
    B = simpy.Resource(env, capacity=cfg["b_servers"])

    busy = {"A":0,"B":0}; last={"A":0.0,"B":0.0}; BT={"A":0.0,"B":0.0}
    def note(x): now=env.now; BT[x]+=busy[x]*(now-last[x]); last[x]=now
    def acquire(x): note(x); busy[x]+=1
    def release(x): note(x); busy[x]-=1

    lam = cfg["orders_per_hour"]/60.0
    orders = []

    def gen():
        oid=0
        while env.now < sim_min:
            ia = np.random.exponential(1.0/lam) if lam>0 else math.inf
            if env.now + ia > sim_min: break
            yield env.timeout(ia)
            oid+=1
            env.process(handle(oid, env.now))

    def handle(oid, t0):
        with A.request() as r1:
            q1=env.now; yield r1; acquire("A")
            w1=env.now-q1; yield env.timeout(pos_exp(cfg["a_service_mean"])); release("A")
        with B.request() as r2:
            q2=env.now; yield r2; acquire("B")
            w2=env.now-q2; yield env.timeout(pos_exp(cfg["b_service_mean"])); release("B")
        soj=env.now-t0
        orders.append({"order_id":oid,"wA":w1,"wB":w2,"cycle_min":soj,"met_sla":1 if soj<=cfg["sla_minutes"] else 0})

    env.process(gen()); env.run(until=sim_min)
    for k in busy: note(k)

    df = pd.DataFrame(orders)
    metrics = {
        "throughput": int(len(df)),
        "avg_cycle_min": float(df["cycle_min"].mean()) if len(df) else 0.0,
        "sla_rate": float(df["met_sla"].mean()) if len(df) else 0.0,
        "A_util": BT["A"]/(cfg["a_servers"]*sim_min) if sim_min>0 else 0.0,
        "B_util": BT["B"]/(cfg["b_servers"]*sim_min) if sim_min>0 else 0.0,
    }
    return metrics, df

def parse_list(txt):
    vals=[]
    for t in txt.split(","):
        t=t.strip()
        if not t: continue
        vals.append(float(t))
    return vals

if st.button("Run batch", type="primary"):
    Agrid = [int(x) for x in parse_list(a_servers_grid)]
    Bgrid = [int(x) for x in parse_list(b_servers_grid)]
    Mgrid = parse_list(demand_mult_grid)

    rows=[]
    for a,b,mult in it.product(Agrid, Bgrid, Mgrid):
        cfg = dict(sim_hours=sim_hours,
                   orders_per_hour=orders_per_hour*mult,
                   a_service_mean=a_service_mean, b_service_mean=b_service_mean,
                   a_servers=a, b_servers=b, sla_minutes=sla_minutes, seed=seed)
        met, _ = run_two_stage(cfg)
        rows.append({
            "A_servers": a, "B_servers": b, "demand_mult": mult,
            "throughput": met["throughput"],
            "avg_cycle_min": round(met["avg_cycle_min"], 2),
            "sla_rate": round(met["sla_rate"], 4),
            "A_util": round(met["A_util"], 4),
            "B_util": round(met["B_util"], 4),
        })
    out = pd.DataFrame(rows).sort_values(["sla_rate","throughput"], ascending=[False,False])
    st.subheader("Scenario comparison")
    st.dataframe(out, use_container_width=True, height=440)
    st.download_button("Download CSV", out.to_csv(index=False).encode("utf-8"),
                       file_name="policy_sensitivity.csv", mime="text/csv")
