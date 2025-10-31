import streamlit as st
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# A. NON-TECHNICAL PROJECT BLURB
# ====================================================================

PROJECT_BLURB = """
### 🚀 Project Goal: Predicting Rollout Success (or Failure)

We are running a **virtual simulation** of our five-year global Transportation Management System (TMS) rollout project.

Instead of relying on a single, optimistic plan, we use the **Monte Carlo method** to run the project thousands of times. Each run uses random factors (like high carrier resistance, sudden currency swings, or integration delays) based on real-world risks.

**What we are trying to do:**
1.  **Find the Risk-Adjusted Finish Date and Budget:** Determine the date and cost we are **90% sure** we won't exceed, giving us a realistic contingency buffer.
2.  **Quantify Failure Probability:** Calculate the chance that we will miss our 5-year deadline, exceed our budget, or fail to achieve our target Return on Investment (ROI).
3.  **Test Risk Mitigation:** By adjusting the sliders (e.g., increasing team size or reducing compliance risk), we can see the direct, quantifiable impact of these actions on our schedule, budget, and ROI.
"""

# ====================================================================
# B. SIMULATION CONTROLS (THE STREAMLIT SIDEBAR)
# ====================================================================

# Define constants
SIM_DURATION = 5 * 365 # 5 years in days
TARGET_COST_MAX = 6_000_000 # Max budget for cost overrun analysis (USD)
TARGET_ROI_MIN = 0.50 # Minimum acceptable ROI (50%)

# Set up the interactive controls in the sidebar
with st.sidebar:
    st.title("⚙️ Simulation Controls")
    
    # --- 1. Project Parameters & Resources (Widgets) ---
    st.subheader("Project & Resource Capacity")
    NUM_SIMULATIONS = st.number_input("Monte Carlo Runs", min_value=100, max_value=10000, value=5000, step=1000)
    GIT_TEAM_CAPACITY = st.number_input("Global Integration Team Size", min_value=1, max_value=10, value=4, step=1)
    
    # --- 2. Risk Controls (Widgets) ---
    st.subheader("Risk Mitigation Levers")
    PROB_CARRIER_NON_COMPLIANCE = st.slider("Carrier Non-Compliance Probability", min_value=0.0, max_value=0.5, value=0.2, step=0.05, format="%.2f")
    INTEGRATION_DIFFICULTY_FACTOR_MAX = st.slider("Max Integration Difficulty Multiplier", min_value=1.0, max_value=2.0, value=1.5, step=0.1)
    st.markdown("---")


# ====================================================================
# C. MONTE CARLO INPUT DISTRIBUTIONS (RISK PARAMETERS)
# ====================================================================

# These are the statistical distributions for the risks (Triangular: Min, Mode, Max)
T1_DURATION = (60, 90, 120)       
ROLLOUT_TIME_BASE = (90, 150, 240)
COST_OVERRUN_DIST = (0.0, 0.15, 0.50)
CURRENCY_FLUCTUATION_DIST = (-0.10, 0.0, 0.20)
RESISTANCE_DELAY_FACTOR = (1.0, 1.15, 1.40)
DATA_DELAY_DAYS = (5, 10, 30)
DATA_CLEANUP_COST = (20000, 50000, 100000)
TARGET_ANNUAL_SAVINGS = 800000
RATE_VOLATILITY_FACTOR = (0.9, 1.0, 1.2)
CARRIER_PENALTY_DAYS = (10, 20, 50) 
REGIONS = {
    'NA':    {'complexity': 1.0, 'compliance_risk': (10, 20, 40)},
    'EMEA':  {'complexity': 1.4, 'compliance_risk': (30, 60, 100)},
    'APAC':  {'complexity': 1.2, 'compliance_risk': (20, 40, 70)},
    'LATAM': {'complexity': 1.1, 'compliance_risk': (15, 30, 60)},
}
ROLLOUT_ORDER = ['NA', 'EMEA', 'APAC', 'LATAM']


# ====================================================================
# D. SIMPY MODEL: TMS ROLLOUT CLASS 
# ====================================================================

class TMSRollout:
    def __init__(self, env, git_capacity, carrier_prob, integration_max):
        self.env = env
        self.total_cost_usd = 0.0
        self.git_team = simpy.Resource(env, capacity=git_capacity)
        self.carrier_prob = carrier_prob
        self.integration_dist = (1.0, 1.2, integration_max)
        self.env.process(self.run_project())

    def run_project(self):
        base_cost_setup = 500000 
        setup_time = np.random.triangular(*T1_DURATION)
        yield self.env.timeout(setup_time)
        cost_factor = np.random.triangular(*COST_OVERRUN_DIST)
        self.total_cost_usd += base_cost_setup * (1 + cost_factor)
        for region_key in ROLLOUT_ORDER:
            yield self.env.process(self.regional_rollout(region_key))

    def regional_rollout(self, region_key):
        region_data = REGIONS[region_key]
        rollout_time = np.random.triangular(*ROLLOUT_TIME_BASE)
        
        # Operational Risk: Integration and Data Challenges
        integration_factor = np.random.triangular(*self.integration_dist)
        data_delay = np.random.triangular(*DATA_DELAY_DAYS)
        rollout_time = rollout_time * integration_factor + data_delay
        
        # Multinational Risk: Resistance
        resistance_factor = np.random.triangular(*RESISTANCE_DELAY_FACTOR)
        compliance_delay = np.random.triangular(*region_data['compliance_risk'])
        rollout_time = rollout_time * resistance_factor + compliance_delay

        data_cost = np.random.triangular(*DATA_CLEANUP_COST)
        self.total_cost_usd += data_cost

        with self.git_team.request() as req:
            yield req
            yield self.env.timeout(rollout_time)
            # Carrier Non-Compliance check
            if random.random() < self.carrier_prob:
                penalty_time = np.random.triangular(*CARRIER_PENALTY_DAYS)
                yield self.env.timeout(penalty_time)

        # Regional Cost Calculation
        base_cost_region = 200000 * region_data['complexity']
        cost_overrun = np.random.triangular(*COST_OVERRUN_DIST)
        currency_factor = np.random.triangular(*CURRENCY_FLUCTUATION_DIST) 
        final_regional_cost = base_cost_region * (1 + cost_overrun) * (1 + currency_factor)
        self.total_cost_usd += final_regional_cost


# ====================================================================
# E. MONTE CARLO DRIVER AND ANALYSIS - CACHED
# ====================================================================

# This function is cached: it only reruns when inputs change, saving time.
@st.cache_data
def run_monte_carlo_simulation(num_runs, git_capacity, carrier_prob, integration_max):
    ALL_PROJECT_DURATIONS = []
    ALL_PROJECT_COSTS = []
    ANNUAL_SAVINGS_RISKED = []

    for i in range(num_runs):
        env = simpy.Environment()
        # Initialize project with user inputs
        project = TMSRollout(env, git_capacity, carrier_prob, integration_max)
        env.run(until=SIM_DURATION + 1000)
        
        ALL_PROJECT_DURATIONS.append(env.now)
        ALL_PROJECT_COSTS.append(project.total_cost_usd)
        
        # Financial Risk: Calculate Realized Savings
        rate_factor = np.random.triangular(*RATE_VOLATILITY_FACTOR)
        realized_savings = TARGET_ANNUAL_SAVINGS / rate_factor
        ANNUAL_SAVINGS_RISKED.append(realized_savings)

    # --- Metrics Calculation ---
    DURATIONS = np.array(ALL_PROJECT_DURATIONS)
    COSTS = np.array(ALL_PROJECT_COSTS)
    SAVINGS = np.array(ANNUAL_SAVINGS_RISKED)

    PROJECT_LIFE_YEARS = 5 
    ALL_ROIS = []
    for i in range(num_runs):
        total_savings = SAVINGS[i] * PROJECT_LIFE_YEARS
        total_project_cost = COSTS[i]
        roi = (total_savings - total_project_cost) / total_project_cost
        ALL_ROIS.append(roi)
    ROIS = np.array(ALL_ROIS)

    P_SUCCESS_TIME = np.sum(DURATIONS <= SIM_DURATION) / num_runs
    P_SUCCESS_COST = np.sum(COSTS <= TARGET_COST_MAX) / num_runs
    P_SUCCESS_ROI = np.sum(ROIS >= TARGET_ROI_MIN) / num_runs

    P90_DURATION = np.percentile(DURATIONS, 90) 
    P90_COST = np.percentile(COSTS, 90)       
    P10_ROI = np.percentile(ROIS, 10) 

    return DURATIONS, COSTS, ROIS, P90_DURATION, P90_COST, P10_ROI, P_SUCCESS_TIME, P_SUCCESS_COST, P_SUCCESS_ROI

# Run the simulation with sidebar controls
DURATIONS, COSTS, ROIS, P90_DURATION, P90_COST, P10_ROI, P_SUCCESS_TIME, P_SUCCESS_COST, P_SUCCESS_ROI = run_monte_carlo_simulation(
    NUM_SIMULATIONS, GIT_TEAM_CAPACITY, PROB_CARRIER_NON_COMPLIANCE, INTEGRATION_DIFFICULTY_FACTOR_MAX
)

# ====================================================================
# F. STREAMLIT DASHBOARD OUTPUT
# ====================================================================

st.title("🌍 Global TMS Rollout Risk Analysis")
st.markdown(PROJECT_BLURB) # DISPLAY THE NON-TECHNICAL BLURB HERE
st.markdown(f"**Running {NUM_SIMULATIONS} Monte Carlo scenarios.** Adjust controls in the sidebar to test mitigation strategies.")

# --- 1. Display Key Metrics (st.metric) ---

st.header("1. Critical Risk Metrics")
col1, col2, col3 = st.columns(3)

# Schedule Risk Card
col1.subheader("Schedule Risk")
col1.metric("90th Percentile Duration", f"{P90_DURATION/365:.2f} years", f"{100 * (1 - P_SUCCESS_TIME):.2f}% risk of overrun")
col1.caption(f"Target: {SIM_DURATION/365:.1f} years (5 years)")

# Cost Risk Card
col2.subheader("Cost Risk")
col2.metric("90th Percentile Cost", f"${P90_COST/1000000:.2f}M", f"{100 * (1 - P_SUCCESS_COST):.2f}% risk of overrun")
col2.caption(f"Target: ${TARGET_COST_MAX/1000000:.1f}M")

# Financial Risk Card
col3.subheader("Financial Risk")
col3.metric("10th Percentile ROI", f"{P10_ROI:.2f}", f"{100 * (1 - P_SUCCESS_ROI):.2f}% risk of failure")
col3.caption(f"Target Min ROI: {TARGET_ROI_MIN:.0f}")

st.markdown("---")

# --- 2. Chart the Results (st.pyplot) ---

st.header("2. Risk Distribution Visuals")
# Create a single Matplotlib figure object
fig, axs = plt.subplots(1, 3, figsize=(18, 5)) 


# 1. Duration Histogram (axs[0])
axs[0].hist(DURATIONS/365, bins=30, color='skyblue', edgecolor='black')
axs[0].axvline(SIM_DURATION/365, color='red', linestyle='dashed', linewidth=2, label='Target 5 Yrs')
axs[0].axvline(P90_DURATION/365, color='orange', linestyle='dashed', linewidth=2, label='P90 Duration')
axs[0].set_title('Project Duration Distribution (Years)')
axs[0].set_xlabel('Duration (Years)')
axs[0].set_ylabel('Frequency')
axs[0].legend()

# 2. Cost Histogram (axs[1])
axs[1].hist(COSTS/1000000, bins=30, color='lightcoral', edgecolor='black')
axs[1].axvline(TARGET_COST_MAX/1000000, color='red', linestyle='dashed', linewidth=2, label='Target Cost')
axs[1].axvline(P90_COST/1000000, color='orange', linestyle='dashed', linewidth=2, label='P90 Cost')
axs[1].set_title('Project Cost Distribution (Millions USD)')
axs[1].set_xlabel('Cost (Millions USD)')
axs[1].set_ylabel('Frequency')
axs[1].legend()

# 3. Cost vs. Duration Scatter Plot (axs[2])
scatter = axs[2].scatter(DURATIONS/365, COSTS/1000000, alpha=0.5, s=15, c=ROIS, cmap='viridis')
axs[2].set_title('Cost vs. Duration (Colored by ROI)')
axs[2].set_xlabel('Duration (Years)')
axs[2].set_ylabel('Cost (Millions USD)')
cbar = fig.colorbar(scatter, ax=axs[2])
cbar.set_label('ROI')

plt.tight_layout()

# Pass the entire figure object to Streamlit to display it
st.pyplot(fig)
