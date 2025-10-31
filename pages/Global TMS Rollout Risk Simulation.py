
import streamlit as st
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

# ===============================================================
# A. Non-technical project blurb (no emojis/icons)
# ===============================================================

PROJECT_BLURB = """
### Project Goal: Predicting Rollout Success (or Failure)

We are running a virtual simulation of a five-year global Transportation Management System (TMS) rollout.

Rather than using a single optimistic plan, we run thousands of Monte Carlo trials. Each trial varies key risks
(e.g., carrier resistance, currency swings, integration delays) within sensible ranges.

**Objectives**
1) Estimate a risk-adjusted finish date and budget (e.g., P90) to size realistic contingency.
2) Quantify probability of missing the time/budget/ROI targets.
3) Test mitigations: adjust inputs (e.g., team capacity, compliance risk) and see real impact on schedule, cost, and ROI.
"""

# ===============================================================
# B. Sidebar controls and cache management
# ===============================================================

# Global constants
SIM_DURATION = 5 * 365  # 5 years in days
TARGET_COST_MAX = 3_000_000
TARGET_ROI_MIN = 0.50

# Session nonce for cache key & reseeding
if "nonce" not in st.session_state:
    st.session_state.nonce = 0

def apply_and_recompute():
    st.session_state.nonce += 1
    st.rerun()

def clear_simulation_cache():
    st.cache_data.clear()

with st.sidebar:
    st.title("Simulation Controls")

    st.subheader("Project & Resource Capacity")
    NUM_SIMULATIONS = st.number_input(
        "Monte Carlo Runs", min_value=100, max_value=20000, value=5000, step=500,
        key="num_sims", on_change=clear_simulation_cache
    )
    GIT_TEAM_CAPACITY = st.number_input(
        "Global Integration Team Size", min_value=1, max_value=20, value=4, step=1,
        key="git_cap", on_change=clear_simulation_cache
    )

    st.subheader("Risk Mitigation Levers")
    PROB_CARRIER_NON_COMPLIANCE = st.slider(
        "Carrier Non-Compliance Probability", min_value=0.0, max_value=0.5, value=0.2, step=0.05,
        format="%.2f", key="carrier_prob", on_change=clear_simulation_cache
    )
    INTEGRATION_DIFFICULTY_FACTOR_MAX = st.slider(
        "Max Integration Difficulty Multiplier", min_value=1.0, max_value=2.0, value=1.5, step=0.1,
        key="integration_max", on_change=clear_simulation_cache
    )

    st.markdown("---")
    col_a, col_b = st.columns(2)
    with col_a:
        st.button("Apply & recompute", on_click=apply_and_recompute, use_container_width=True)
    with col_b:
        if st.button("Clear cache", use_container_width=True):
            clear_simulation_cache()
            st.rerun()

# ===============================================================
# C. Risk parameter distributions (fixed)
# ===============================================================

T1_DURATION = (60, 90, 120)              # Global mobilization/design triangle (days)
ROLLOUT_TIME_BASE = (90, 150, 240)       # Base per-region rollout (days)
COST_OVERRUN_DIST = (0.0, 0.15, 0.50)    # Cost overrun factor
CURRENCY_FLUCTUATION_DIST = (-0.10, 0.0, 0.20)  # FX impact factor
RESISTANCE_DELAY_FACTOR = (1.0, 1.15, 1.40)     # Org resistance multiplies duration
DATA_DELAY_DAYS = (5, 10, 30)            # Data prep/quality delay
DATA_CLEANUP_COST = (20000, 50000, 100000)
TARGET_ANNUAL_SAVINGS = 800000
RATE_VOLATILITY_FACTOR = (0.9, 1.0, 1.2) # Freight rate volatility
CARRIER_PENALTY_DAYS = (10, 20, 50)      # Penalty for compliance issues

REGIONS = {
    'NA':    {'complexity': 1.0, 'compliance_risk': (10, 20, 40)},
    'EMEA':  {'complexity': 1.4, 'compliance_risk': (30, 60, 100)},
    'APAC':  {'complexity': 1.2, 'compliance_risk': (20, 40, 70)},
    'LATAM': {'complexity': 1.1, 'compliance_risk': (15, 30, 60)},
}
ROLLOUT_ORDER = ['NA', 'EMEA', 'APAC', 'LATAM']

# ===============================================================
# D. SimPy rollout model
# ===============================================================

class TMSRollout:
    def __init__(self, env, git_capacity, carrier_prob, integration_max, fixed_params):
        self.env = env
        self.total_cost_usd = 0.0
        self.git_team = simpy.Resource(env, capacity=git_capacity)
        self.carrier_prob = carrier_prob
        self.integration_dist = (1.0, 1.2, integration_max)
        self.fixed = fixed_params
        self.env.process(self.run_project())

    def run_project(self):
        base_cost_setup = 500000
        setup_time = np.random.triangular(*self.fixed['T1_DURATION'])
        yield self.env.timeout(setup_time)
        cost_factor = np.random.triangular(*self.fixed['COST_OVERRUN_DIST'])
        self.total_cost_usd += base_cost_setup * (1 + cost_factor)
        for region_key in self.fixed['ROLLOUT_ORDER']:
            yield self.env.process(self.regional_rollout(region_key))

    def regional_rollout(self, region_key):
        region_data = self.fixed['REGIONS'][region_key]
        rollout_time = np.random.triangular(*self.fixed['ROLLOUT_TIME_BASE'])

        # Integration complexity/time
        integration_factor = np.random.triangular(*self.integration_dist)
        data_delay = np.random.triangular(*self.fixed['DATA_DELAY_DAYS'])
        rollout_time = rollout_time * integration_factor + data_delay

        # Resistance and compliance delay
        resistance_factor = np.random.triangular(*self.fixed['RESISTANCE_DELAY_FACTOR'])
        compliance_delay = np.random.triangular(*region_data['compliance_risk'])
        rollout_time = rollout_time * resistance_factor + compliance_delay

        # Data cleanup cost
        data_cost = np.random.triangular(*self.fixed['DATA_CLEANUP_COST'])
        self.total_cost_usd += data_cost

        # Execute using constrained team
        with self.git_team.request() as req:
            yield req
            yield self.env.timeout(rollout_time)
            # Carrier non-compliance penalty
            if random.random() < self.carrier_prob:
                penalty_time = np.random.triangular(*self.fixed['CARRIER_PENALTY_DAYS'])
                yield self.env.timeout(penalty_time)

        # Regional cost with FX and overrun
        base_cost_region = 200000 * region_data['complexity']
        cost_overrun = np.random.triangular(*self.fixed['COST_OVERRUN_DIST'])
        currency_factor = np.random.triangular(*self.fixed['CURRENCY_FLUCTUATION_DIST'])
        final_regional_cost = base_cost_region * (1 + cost_overrun) * (1 + currency_factor)
        self.total_cost_usd += final_regional_cost

# ===============================================================
# E. Monte Carlo driver (nonce-based caching)
# ===============================================================

FIXED_PARAMS = {
    'T1_DURATION': T1_DURATION,
    'ROLLOUT_TIME_BASE': ROLLOUT_TIME_BASE,
    'COST_OVERRUN_DIST': COST_OVERRUN_DIST,
    'CURRENCY_FLUCTUATION_DIST': CURRENCY_FLUCTUATION_DIST,
    'RESISTANCE_DELAY_FACTOR': RESISTANCE_DELAY_FACTOR,
    'DATA_DELAY_DAYS': DATA_DELAY_DAYS,
    'DATA_CLEANUP_COST': DATA_CLEANUP_COST,
    'RATE_VOLATILITY_FACTOR': RATE_VOLATILITY_FACTOR,
    'CARRIER_PENALTY_DAYS': CARRIER_PENALTY_DAYS,
    'REGIONS': REGIONS,
    'ROLLOUT_ORDER': ROLLOUT_ORDER,
    'TARGET_ANNUAL_SAVINGS': TARGET_ANNUAL_SAVINGS,
}

def _seed_from_inputs(num_runs, git_capacity, carrier_prob, integration_max, nonce:int):
    # Build a deterministic seed from inputs + nonce
    base = 41 * num_runs + 97 * git_capacity + 389 * int(carrier_prob * 100) + 761 * int(integration_max * 10) + 1543 * nonce
    base = int(base % (2**31 - 1))
    if base <= 0:
        base = 12345 + nonce
    return base

@st.cache_data(show_spinner=False)
def run_monte_carlo_simulation(num_runs, git_capacity, carrier_prob, integration_max, fixed_params, _nonce:int):
    seed = _seed_from_inputs(num_runs, git_capacity, carrier_prob, integration_max, _nonce)
    np.random.seed(seed)
    random.seed(seed)

    durations = []
    costs = []
    rois = []

    PROJECT_LIFE_YEARS = 5

    for _ in range(num_runs):
        env = simpy.Environment()
        project = TMSRollout(env, git_capacity, carrier_prob, integration_max, fixed_params)
        env.run(until=SIM_DURATION + 1000)

        durations.append(env.now)
        costs.append(project.total_cost_usd)

        rate_factor = np.random.triangular(*fixed_params['RATE_VOLATILITY_FACTOR'])
        realized_savings = fixed_params['TARGET_ANNUAL_SAVINGS'] / rate_factor
        total_savings = realized_savings * PROJECT_LIFE_YEARS
        total_project_cost = project.total_cost_usd
        roi = (total_savings - total_project_cost) / max(total_project_cost, 1e-9)
        rois.append(roi)

    DURATIONS = np.array(durations)
    COSTS = np.array(costs)
    ROIS = np.array(rois)

    P_SUCCESS_TIME = float(np.sum(DURATIONS <= SIM_DURATION) / num_runs)
    P_SUCCESS_COST = float(np.sum(COSTS <= TARGET_COST_MAX) / num_runs)
    P_SUCCESS_ROI = float(np.sum(ROIS >= TARGET_ROI_MIN) / num_runs)

    P90_DURATION = float(np.percentile(DURATIONS, 90))
    P90_COST = float(np.percentile(COSTS, 90))
    P10_ROI = float(np.percentile(ROIS, 10))

    return DURATIONS, COSTS, ROIS, P90_DURATION, P90_COST, P10_ROI, P_SUCCESS_TIME, P_SUCCESS_COST, P_SUCCESS_ROI

DURATIONS, COSTS, ROIS, P90_DURATION, P90_COST, P10_ROI, P_SUCCESS_TIME, P_SUCCESS_COST, P_SUCCESS_ROI = run_monte_carlo_simulation(
    NUM_SIMULATIONS,
    GIT_TEAM_CAPACITY,
    PROB_CARRIER_NON_COMPLIANCE,
    INTEGRATION_DIFFICULTY_FACTOR_MAX,
    FIXED_PARAMS,
    st.session_state.nonce
)

# ===============================================================
# F. Dashboard output
# ===============================================================

st.title("Global TMS Rollout Risk Analysis")
st.markdown(PROJECT_BLURB)
st.markdown(f"Running {NUM_SIMULATIONS} Monte Carlo scenarios. Adjust controls in the sidebar and click 'Apply & recompute'.")

# Key metrics
st.header("1. Critical Risk Metrics")
col1, col2, col3 = st.columns(3)

col1.subheader("Schedule Risk")
col1.metric("90th Percentile Duration", f"{P90_DURATION/365:.2f} years", f"{100 * (1 - P_SUCCESS_TIME):.2f}% risk of overrun")
col1.caption(f"Target: {SIM_DURATION/365:.1f} years")

col2.subheader("Cost Risk")
col2.metric("90th Percentile Cost", f"${P90_COST/1_000_000:.2f}M", f"{100 * (1 - P_SUCCESS_COST):.2f}% risk of overrun")
col2.caption(f"Target: ${TARGET_COST_MAX/1_000_000:.1f}M")

col3.subheader("Financial Risk")
col3.metric("10th Percentile ROI", f"{P10_ROI:.2f}", f"{100 * (1 - P_SUCCESS_ROI):.2f}% risk of failure")
col3.caption(f"Target Min ROI: {TARGET_ROI_MIN:.2f}")

st.markdown("---")

# Charts
st.header("2. Risk Distribution Visuals")
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

# Duration histogram
axs[0].hist(DURATIONS/365, bins=30, color='skyblue', edgecolor='black')
axs[0].axvline(SIM_DURATION/365, color='red', linestyle='dashed', linewidth=2, label='Target 5 Years')
axs[0].axvline(P90_DURATION/365, color='orange', linestyle='dashed', linewidth=2, label='P90 Duration')
axs[0].set_title('Project Duration Distribution (Years)')
axs[0].set_xlabel('Duration (Years)')
axs[0].set_ylabel('Frequency')
axs[0].legend()

# Cost histogram
axs[1].hist(COSTS/1_000_000, bins=30, color='lightcoral', edgecolor='black')
axs[1].axvline(TARGET_COST_MAX/1_000_000, color='red', linestyle='dashed', linewidth=2, label='Target Cost')
axs[1].axvline(P90_COST/1_000_000, color='orange', linestyle='dashed', linewidth=2, label='P90 Cost')
axs[1].set_title('Project Cost Distribution (Millions USD)')
axs[1].set_xlabel('Cost (Millions USD)')
axs[1].set_ylabel('Frequency')
axs[1].legend()

# Cost vs. Duration scatter with ROI color
scatter = axs[2].scatter(DURATIONS/365, COSTS/1_000_000, alpha=0.5, s=15, c=ROIS, cmap='viridis')
axs[2].set_title('Cost vs. Duration (Colored by ROI)')
axs[2].set_xlabel('Duration (Years)')
axs[2].set_ylabel('Cost (Millions USD)')
cbar = fig.colorbar(scatter, ax=axs[2])
cbar.set_label('ROI')

plt.tight_layout()
st.pyplot(fig)

st.markdown("---")

st.header("3. Interpretation and Value of This Simulation")
st.markdown("""
How this helps leadership  
- Risk-adjusted targets: P90 duration and cost provide realistic contingency.  
- Mitigation ROI: Raising team capacity or lowering compliance risk shows measurable impact on schedule and cost risk.  
- Focus: Sensitivity of outcomes highlights which risks deserve attention first.
""")

with st.expander("Detailed graph explanations"):
    st.markdown("""
Duration histogram: area to the right of the red line is probability of time overrun; orange line is P90.  
Cost histogram: area to the right of the red line is probability of cost overrun; orange line is P90.  
Scatter: top-right cluster indicates late and expensive scenarios; color shows ROI (darker is worse).
""")
