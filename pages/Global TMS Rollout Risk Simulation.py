import simpy
import random
import numpy as np
import matplotlib.pyplot as plt

# ====================================================================
# A. SIMULATION CONTROLS (THE "SIDEBAR")
# ====================================================================

# 1. Project Parameters
SIM_DURATION = 5 * 365 # 5 years in days
NUM_SIMULATIONS = 5000 # Monte Carlo Loop Size
TARGET_COST_MAX = 6_000_000 # Max budget for cost overrun analysis (USD)
TARGET_ROI_MIN = 0.50 # Minimum acceptable ROI (50%)

# 2. Resource Controls
GIT_TEAM_CAPACITY = 4 # Global Integration Team size (Core resource)

# 3. Risk Controls (Adjust these to test mitigation strategies)
# Carrier Risk: Chance of a major non-compliance event per region (0.0 to 1.0)
PROB_CARRIER_NON_COMPLIANCE = 0.20 
# Carrier Risk: Time penalty if non-compliance occurs (min, likely, max days)
CARRIER_PENALTY_DAYS = (10, 20, 50) 
# Data Risk: Multiplier on time due to integration complexity
INTEGRATION_DIFFICULTY_FACTOR = (1.0, 1.2, 1.5) 

# ====================================================================
# B. MONTE CARLO INPUT DISTRIBUTIONS (RISK PARAMETERS)
# ====================================================================

# Project Execution Risks
T1_DURATION = (60, 90, 120)       # Core Setup Duration (days)
ROLLOUT_TIME_BASE = (90, 150, 240) # Regional Rollout Duration (days)
COST_OVERRUN_DIST = (0.0, 0.15, 0.50) # Cost overrun factor (0% to 50%)

# Multinational Risks
CURRENCY_FLUCTUATION_DIST = (-0.10, 0.0, 0.20) # Cost factor (-10% to +20%)
RESISTANCE_DELAY_FACTOR = (1.0, 1.15, 1.40)     # User Resistance Rollout Time Multiplier

# Data Risks
DATA_DELAY_DAYS = (5, 10, 30)                    # Delay in days due to data cleanup
DATA_CLEANUP_COST = (20000, 50000, 100000)       # USD cost for data remediation

# ROI/Financial Risks
TARGET_ANNUAL_SAVINGS = 800000                   # Annual projected savings (USD)
RATE_VOLATILITY_FACTOR = (0.9, 1.0, 1.2)         # Impact on realized savings (multiplier)

# Regional Profiles (Base Cost Multiplier, Compliance Difficulty days)
REGIONS = {
    'NA':    {'complexity': 1.0, 'compliance_risk': (10, 20, 40)},
    'EMEA':  {'complexity': 1.4, 'compliance_risk': (30, 60, 100)},
    'APAC':  {'complexity': 1.2, 'compliance_risk': (20, 40, 70)},
    'LATAM': {'complexity': 1.1, 'compliance_risk': (15, 30, 60)},
}
ROLLOUT_ORDER = ['NA', 'EMEA', 'APAC', 'LATAM'] # Defined project sequence

# ====================================================================
# C. SIMPY MODEL: TMS ROLLOUT CLASS (DISCRETE EVENT SIMULATION)
# ====================================================================

class TMSRollout:
    def __init__(self, env):
        self.env = env
        self.total_cost_usd = 0.0
        # The key resource that dictates sequential flow: the Global Integration Team
        self.git_team = simpy.Resource(env, capacity=GIT_TEAM_CAPACITY)
        self.env.process(self.run_project())

    def run_project(self):
        # --- PHASE 1: Core System Setup ---
        base_cost_setup = 500000 
        setup_time = np.random.triangular(*T1_DURATION)
        yield self.env.timeout(setup_time)
        
        cost_factor = np.random.triangular(*COST_OVERRUN_DIST)
        self.total_cost_usd += base_cost_setup * (1 + cost_factor)
        
        # --- PHASE 2: Staged Regional Rollouts ---
        for region_key in ROLLOUT_ORDER:
            # Sequentially process each region
            yield self.env.process(self.regional_rollout(region_key))

    def regional_rollout(self, region_key):
        region_data = REGIONS[region_key]

        # 1. Calculate Risk-Adjusted Duration (Monte Carlo Sampling)
        rollout_time = np.random.triangular(*ROLLOUT_TIME_BASE)
        
        # Operational Risk: Integration and Data Challenges
        integration_factor = np.random.triangular(*INTEGRATION_DIFFICULTY_FACTOR)
        data_delay = np.random.triangular(*DATA_DELAY_DAYS)
        rollout_time = rollout_time * integration_factor + data_delay
        
        # Multinational Risk: Compliance and Resistance
        resistance_factor = np.random.triangular(*RESISTANCE_DELAY_FACTOR)
        compliance_delay = np.random.triangular(*region_data['compliance_risk'])
        rollout_time = rollout_time * resistance_factor + compliance_delay

        # Add Data Cleanup Cost
        data_cost = np.random.triangular(*DATA_CLEANUP_COST)
        self.total_cost_usd += data_cost

        # 2. Acquire Global Integration Team (GIT) Resource
        with self.git_team.request() as req:
            yield req # Wait for the team to be available
            
            # Execute Rollout
            yield self.env.timeout(rollout_time)

            # 3. Transportation Risk: Carrier Non-Compliance Check (Stochastic Event)
            if random.random() < PROB_CARRIER_NON_COMPLIANCE:
                penalty_time = np.random.triangular(*CARRIER_PENALTY_DAYS)
                yield self.env.timeout(penalty_time) # Non-compliance forces a delay

        # 4. Regional Cost Calculation (Including Multinational Currency Risk)
        base_cost_region = 200000 * region_data['complexity']
        cost_overrun = np.random.triangular(*COST_OVERRUN_DIST)
        currency_factor = np.random.triangular(*CURRENCY_FLUCTUATION_DIST) 
        
        final_regional_cost = base_cost_region * (1 + cost_overrun) * (1 + currency_factor)
        self.total_cost_usd += final_regional_cost


# ====================================================================
# D. MONTE CARLO DRIVER AND ANALYSIS
# ====================================================================

ALL_PROJECT_DURATIONS = []
ALL_PROJECT_COSTS = []
ANNUAL_SAVINGS_RISKED = []

for i in range(NUM_SIMULATIONS):
    env = simpy.Environment()
    project = TMSRollout(env)
    env.run(until=SIM_DURATION + 1000)
    
    # Collect results
    ALL_PROJECT_DURATIONS.append(env.now)
    ALL_PROJECT_COSTS.append(project.total_cost_usd)
    
    # Financial Risk: Calculate Realized Savings
    rate_factor = np.random.triangular(*RATE_VOLATILITY_FACTOR)
    realized_savings = TARGET_ANNUAL_SAVINGS / rate_factor
    ANNUAL_SAVINGS_RISKED.append(realized_savings)


# --- 1. Final Calculations ---
DURATIONS = np.array(ALL_PROJECT_DURATIONS)
COSTS = np.array(ALL_PROJECT_COSTS)
SAVINGS = np.array(ANNUAL_SAVINGS_RISKED)

# Calculate ROI for each run
PROJECT_LIFE_YEARS = 5 
ALL_ROIS = []
for i in range(NUM_SIMULATIONS):
    total_savings = SAVINGS[i] * PROJECT_LIFE_YEARS
    total_project_cost = COSTS[i]
    # Simplified ROI: (Total Savings - Total Cost) / Total Cost
    roi = (total_savings - total_project_cost) / total_project_cost
    ALL_ROIS.append(roi)
ROIS = np.array(ALL_ROIS)


# --- 2. Risk Metrics ---
P_SUCCESS_TIME = np.sum(DURATIONS <= SIM_DURATION) / NUM_SIMULATIONS
P_SUCCESS_COST = np.sum(COSTS <= TARGET_COST_MAX) / NUM_SIMULATIONS
P_SUCCESS_ROI = np.sum(ROIS >= TARGET_ROI_MIN) / NUM_SIMULATIONS

# P90 (90th percentile) is the risk measure for Duration and Cost
P90_DURATION = np.percentile(DURATIONS, 90) 
P90_COST = np.percentile(COSTS, 90)       
# P10 (10th percentile) is the risk measure for ROI (the low outcome)
P10_ROI = np.percentile(ROIS, 10) 

# ====================================================================
# E. RESULTS OUTPUT AND CHARTING
# ====================================================================

print("="*80)
print(f"GLOBAL TMS ROLLOUT MONTE CARLO RISK ANALYSIS RESULTS ({NUM_SIMULATIONS} Runs)")
print("="*80)
print(f"Simulation Controls: GIT Team Capacity={GIT_TEAM_CAPACITY} | Carrier Non-Compliance P={PROB_CARRIER_NON_COMPLIANCE * 100:.0f}%")
print("-" * 80)

# --- SCHEDULE RISK (Duration) ---
print("\n--- SCHEDULE RISK (Duration) ---")
print(f"Target Duration: {SIM_DURATION/365:.1f} years (1825 days)")
print(f"Probability of Time Overrun (> 5 years): {100 * (1 - P_SUCCESS_TIME):.2f}%")
print(f"**90th Percentile Duration:** {P90_DURATION/365:.2f} years (Risk-Adjusted Finish Date)")

# --- COST RISK (Budget) ---
print("\n--- COST RISK (Budget) ---")
print(f"Target Budget: ${TARGET_COST_MAX:,.0f}")
print(f"Probability of Cost Overrun (> ${TARGET_COST_MAX/1000000:.1f}M): {100 * (1 - P_SUCCESS_COST):.2f}%")
print(f"**90th Percentile Cost:** ${P90_COST:,.0f} (Risk-Adjusted Contingency Budget)")

# --- FINANCIAL RISK (ROI) ---
print("\n--- FINANCIAL RISK (ROI) ---")
print(f"Target Minimum ROI: {TARGET_ROI_MIN * 100:.0f}%")
print(f"Probability of ROI Failure (< {TARGET_ROI_MIN * 100:.0f}%): {100 * (1 - P_SUCCESS_ROI):.2f}%")
print(f"**10th Percentile ROI:** {P10_ROI:.2f} (Worst-Case Scenario ROI)")


# --- Chart the Results ---
plt.figure(figsize=(15, 5))

# 1. Duration Histogram
plt.subplot(1, 3, 1)
plt.hist(DURATIONS/365, bins=30, color='skyblue', edgecolor='black')
plt.axvline(SIM_DURATION/365, color='red', linestyle='dashed', linewidth=1, label='Target 5 Yrs')
plt.axvline(P90_DURATION/365, color='orange', linestyle='dashed', linewidth=1, label='P90 Duration')
plt.title('Project Duration Distribution (Years)')
plt.xlabel('Duration (Years)')
plt.ylabel('Frequency')
plt.legend()

# 2. Cost Histogram
plt.subplot(1, 3, 2)
plt.hist(COSTS/1000000, bins=30, color='lightcoral', edgecolor='black')
plt.axvline(TARGET_COST_MAX/1000000, color='red', linestyle='dashed', linewidth=1, label='Target Cost')
plt.axvline(P90_COST/1000000, color='orange', linestyle='dashed', linewidth=1, label='P90 Cost')
plt.title('Project Cost Distribution (Millions USD)')
plt.xlabel('Cost (Millions USD)')
plt.ylabel('Frequency')
plt.legend()

# 3. Cost vs. Duration (Risk Scatter Plot)
plt.subplot(1, 3, 3)
# Scatter plot where color represents the ROI outcome
plt.scatter(DURATIONS/365, COSTS/1000000, alpha=0.5, s=10, c=ROIS, cmap='viridis')
plt.title('Cost vs. Duration (Colored by ROI)')
plt.xlabel('Duration (Years)')
plt.ylabel('Cost (Millions USD)')
cbar = plt.colorbar()
cbar.set_label('ROI')

plt.tight_layout()
plt.show()
