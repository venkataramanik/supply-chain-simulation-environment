🛻 03_Fleet_Dispatch.py — Transportation & Fleet Flow Simulation

About:
Simulates a stochastic fleet-dispatch environment where vehicles handle randomly arriving delivery jobs. Tracks on-time performance, waiting times, and vehicle utilization.

Parameters:

Simulation hours, number of vehicles, jobs/hour

Average vehicle speed, mean & SD of distance/job

Onsite service time, SLA minutes, random seed

KPIs:

Jobs completed

On-time rate (%)

Average queue time (min)

Average sojourn time (min)

Fleet utilization (%)

How to Run:

Open the app → “Fleet Dispatch (SimPy)” page.

Adjust parameters in the sidebar.

Click Run Simulation to compute metrics and export results as CSV.

⚙️ 04_Production_Scheduling.py — Production & Assembly Line

About:
Models a production line of machines that produce parts but fail randomly. A single preemptive repairman fixes them. Helps identify throughput and repair capacity requirements.

Parameters:

Simulation hours, machines count

Processing time mean & SD

Mean time to failure (MTTF), repair duration

Other (secondary) job duration, random seed

KPIs:

Total parts produced

Average parts per machine

Repairman utilization (%)

How to Run:

Open the Production & Assembly Line (SimPy) page.

Adjust time, number of machines, and MTTF/repair parameters.

Click Run Simulation → review metrics table and CSV.

📦 05_Inventory_Replenishment.py — Continuous Review (s, S) Policy

About:
Simulates a single-echelon inventory system following an (s, S) reorder policy with stochastic daily demand and random lead times.

Parameters:

Simulation days

Mean daily demand (Poisson λ)

Reorder point (s) and order-up-to level (S)

Lead-time mean & SD, initial inventory, random seed

KPIs:

Service level (%)

Stockout days

Average on-hand inventory

Orders placed

Total unfilled demand

How to Run:

Open Inventory & Replenishment (SimPy).

Enter (s, S), demand, and lead-time values.

Run → analyze daily inventory table and download CSV.

⚓ 06_Port_Yard.py — Port / Yard / Cross-Dock Operations

About:
Models port or cross-dock operations where vessels/trailers queue for limited berths/doors and cranes/forklifts. Captures resource contention and turnaround delays.

Parameters:

Simulation hours

Arrivals/hour

Number of berths & cranes

Mean and SD of moves per vessel

Move rate per crane (moves/min), random seed

KPIs:

Throughput (vessels handled)

Average queue time (min)

Average turnaround (min)

Berth utilization (%)

Crane utilization (%)

How to Run:

Open Port / Yard (SimPy) page.

Set resource counts and arrival rates.

Run simulation and export the vessel log to CSV.

📬 07_Order_to_Delivery.py — End-to-End Fulfillment Pipeline

About:
Simulates a multi-stage order-to-delivery flow (Pick → Pack → Ship).
Evaluates staffing requirements and SLA performance.

Parameters:

Simulation hours

Orders/hour

Number of pickers, packers, and carriers

Mean processing times per stage

SLA threshold (minutes), random seed

KPIs:

Orders processed

Average total cycle time (min)

SLA compliance rate (%)

Picker, Packer, and Carrier utilization (%)

How to Run:

Open Order-to-Delivery (SimPy).

Input stage capacities and SLA.

Run and inspect the order log table or CSV output.

📊 08_Policy_Sensitivity.py — “What-If” Scenario & Policy Testing

About:
Runs batch experiments across multiple configurations to assess how resource levels or demand multipliers impact throughput and SLA compliance.

Parameters:

Base order arrival rate, SLA, service times

Server counts for Stage A and Stage B (comma-separated)

Demand multipliers (comma-separated)

Simulation time, random seed

KPIs:

Throughput (orders)

Average cycle time (min)

SLA success rate (%)

Utilization of Stage A & B (%)

How to Run:

Open Policy & What-If (SimPy).

Enter comma-separated lists for servers and demand multipliers.

Click Run Batch → compare all scenarios in one table.
