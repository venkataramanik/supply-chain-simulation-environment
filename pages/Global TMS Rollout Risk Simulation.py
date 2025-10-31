import streamlit as st
import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# A. NON-TECHNICAL PROJECT BLURB
# ====================================================================

PROJECT_BLURB = """
### Project Goal: Predicting Rollout Success (or Failure)

We are running a **virtual simulation** of our five-year global Transportation Management System (TMS) rollout project.

Instead of relying on a single, optimistic plan, we use the **Monte Carlo method** to run the project thousands of times. Each run uses random factors (like high carrier resistance, sudden currency swings, or integration delays) based on real-world risks.

**What we are trying to do:**
1.  **Find the Risk-Adjusted Finish Date and Budget:** Determine the date and cost we are **90% sure** we won't exceed, giving us a realistic contingency buffer.
2.  **Quantify Failure Probability:** Calculate the chance that we will miss our 5-year deadline, exceed our budget, or fail to achieve our target Return on Investment (ROI).
3.  **Test Risk Mitigation:** By adjusting the sliders (e.g., increasing team size or reducing compliance risk), we can see the direct, quantifiable impact of these actions on our schedule, budget, and ROI.
"""

# ====================================================================
# B. SIMULATION CONTROLS (THE STREAMLIT SIDEBAR) - WIDGETS
# ====================================================================

# Define constants
SIM_DURATION = 5 * 365 # 5 years in days
TARGET_COST_MAX = 6_000_000 # Max budget for cost overrun analysis (USD)
TARGET_ROI_MIN = 0.50 # Minimum acceptable ROI (50%)

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
# C. MONTE CARLO INPUT DISTRIBUTIONS (FIXED RISK PARAMETERS)
# ====================================================================

# These fixed distributions do not rely on sidebar widgets
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
    # Class accepts and uses all dynamic parameters
    def __init__(self, env, git_capacity, carrier_prob, integration_max):
        self.env = env
        self.total_cost_usd = 0.0
        self.git_team = simpy.Resource(env, capacity=git_capacity)
        self.carrier_prob = carrier_prob
        self.integration_dist = (1.0, 1.2, integration_max) # Uses dynamic max
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
# E. MONTE CARLO DRIVER AND ANALYSIS - CACHED FIX APPLIED
# ====================================================================

# ALL interactive widget values MUST be passed as arguments here.
@st.cache_data
def run_monte_carlo_simulation(num_runs, git_capacity, carrier_prob, integration_max):
    ALL_PROJECT_DURATIONS = []
    ALL_PROJECT_COSTS = []
    ANNUAL_SAVINGS_RISKED = []

    for i in range(num_runs):
        env = simpy.Environment()
        # Pass all user inputs to the project initialization
        project = TMSRollout(env, git_capacity, carrier_prob, integration_max)
        env.run(until=SIM_DURATION + 1000)
        
        ALL_PROJECT_DURATIONS.append(env.now)
        ALL_PROJECT_COSTS.append(project.total_cost_usd)
        
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

# Call the function, passing ALL widget outputs. This is the dependency list for caching.
DURATIONS, COSTS, ROIS, P90_DURATION, P90_COST, P10_ROI, P_SUCCESS_TIME, P_SUCCESS_COST, P_SUCCESS_ROI = run_monte_carlo_simulation(
    NUM_SIMULATIONS, 
    GIT_TEAM_CAPACITY, 
    PROB_CARRIER_NON_COMPLIANCE, 
    INTEGRATION_DIFFICULTY_FACTOR_MAX
)

# ====================================================================
# F. STREAMLIT DASHBOARD OUTPUT
# ====================================================================

st.title("🌍 Global TMS Rollout Risk Analysis")
st.markdown(PROJECT_BLURB) 
st.markdown(f"**Running {NUM_SIMULATIONS} Monte Carlo scenarios.** Adjust controls in the sidebar to test mitigation strategies.")

# --- 1. Display Key Metrics (st.metric) ---

st.header("1. Critical Risk Metrics")
col1, col2, col3 = st.columns(3)

col1.subheader("Schedule Risk")
# Risk value update should now be instantaneous when a sidebar value changes
col1.metric("90th Percentile Duration", f"{P90_DURATION/365:.2f} years", f"{100 * (1 - P_SUCCESS_TIME):.2f}% risk of overrun")
col1.caption(f"Target: {SIM_DURATION/365:.1f} years (5 years)")

col2.subheader("Cost Risk")
col2.metric("90th Percentile Cost", f"${P90_COST/1000000:.2f}M", f"{100 * (1 - P_SUCCESS_COST):.2f}% risk of overrun")
col2.caption(f"Target: ${TARGET_COST_MAX/1000000:.1f}M")

col3.subheader("Financial Risk")
col3.metric("10th Percentile ROI", f"{P10_ROI:.2f}", f"{100 * (1 - P_SUCCESS_ROI):.2f}% risk of failure")
col3.caption(f"Target Min ROI: {TARGET_ROI_MIN:.0f}")

st.markdown("---")

# ====================================================================
# G. CHARTING AND EXPLANATION
# ====================================================================

st.header("2. Risk Distribution Visuals")
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
st.pyplot(fig)

st.markdown("---")

st.header("3. Interpretation and Value of This Simulation")

st.markdown("### How This Simulation Helps Project Leadership")
st.markdown("""
This model transforms project management from relying on optimistic estimates (the 'best case') to making **risk-informed decisions**. It addresses the common pitfalls of multinational rollouts: resource conflicts, regulatory compliance, and currency volatility.

* **Setting Realistic Expectations:** The **90th Percentile (P90)** metrics for Duration and Cost provide the **contingency budget and schedule** required to finish the project 9 out of 10 times. This protects against stakeholder disappointment and budget blowouts.
* **Quantifying Mitigation ROI:** By changing the sliders (e.g., increasing **GIT Team Capacity**), project leaders can instantly see the **financial return** of that investment in terms of reduced project duration and lower probability of cost overrun.
* **Focusing Mitigation Efforts:** The model clearly identifies the largest risks (e.g., is **Integration Difficulty** a bigger threat than **Carrier Non-Compliance**?) allowing resources to be focused where they have the biggest impact on ROI.
""")

st.markdown("### Explanation of the Graphs")
with st.expander("Click to view detailed graph explanations"):
    st.markdown("""
    #### 1. Project Duration Distribution (Histogram)
    * **What it shows:** The frequency of various project completion times. A wide spread indicates high **Schedule Volatility**.
    * **Key Lines:**
        * **Red Line (Target 5 Yrs):** The official deadline. The area of the bars to the **right** of this line represents the **Probability of Time Overrun**.
        * **Orange Line (P90 Duration):** The **Risk-Adjusted Schedule**. This is the date you should plan for to be 90% certain you'll meet the deadline.
    

    #### 2. Project Cost Distribution (Histogram)
    * **What it shows:** The frequency of various final project costs. This captures risks like **Currency Fluctuation** and **Cost Overrun**.
    * **Key Lines:**
        * **Red Line (Target Cost):** The initial maximum budget. The bars to the **right** of this line represent the **Probability of Cost Overrun**.
        * **Orange Line (P90 Cost):** The **Risk-Adjusted Budget** required to cover realistic contingencies 90% of the time.
    

    #### 3. Cost vs. Duration (Scatter Plot)
    * **What it shows:** The relationship between all three critical metrics: **Cost**, **Duration**, and **ROI** (represented by color).
    * **Interpretation:**
        * **High Risk Zone:** Look at the points clustered in the **top-right corner** (high cost and high duration).
        * **ROI Color:** The color of the points represents the ROI. Scenarios clustered in the top-right often have a **low ROI (darker colors)**, confirming that late, expensive projects destroy value.
    
    """)
