# sim_core.py
import math, random
import simpy
import numpy as np
import pandas as pd

class UtilizationMonitor:
    def __init__(self, env, resource, capacity):
        self.env = env
        self.capacity = capacity
        self.busy = 0
        self.last_t = 0.0
        self.busy_time = 0.0
    def note_change(self):
        now = self.env.now
        dt = now - self.last_t
        self.busy_time += self.busy * dt
        self.last_t = now
    def acquire(self):
        self.note_change(); self.busy += 1
    def release(self):
        self.note_change(); self.busy -= 1
    def utilization(self, sim_time):
        if sim_time <= 0: return 0.0
        return min(1.0, max(0.0, self.busy_time / (self.capacity * sim_time)))

def exp_interarrival(rate_per_hour):
    lam = rate_per_hour / 60.0
    def _draw(): return np.random.exponential(1.0/lam) if lam>0 else math.inf
    return _draw

def normal_positive(mean, sd):
    x = np.random.normal(mean, sd)
    while x <= 0: x = np.random.normal(mean, sd)
    return x

def run_simpy_warehouse(cfg, seed=42):
    np.random.seed(seed); random.seed(seed)
    sim_minutes = int(cfg.get("sim_hours", 8) * 60)

    # Params
    num_docks = int(cfg.get("num_dock_doors", 2))
    num_forklifts = int(cfg.get("num_forklifts", 2))
    num_pickers = int(cfg.get("num_pickers", 3))
    inbound_rate_h = float(cfg.get("inbound_trucks_per_hour", 4))
    order_rate_h = float(cfg.get("orders_per_hour", 60))
    unload_mean = float(cfg.get("unload_mean_min", 25.0))
    unload_sd = float(cfg.get("unload_sd_min", 5.0))
    putaway_per_truck_min = float(cfg.get("putaway_minutes_per_truck", 10.0))
    lines_per_order_mean = float(cfg.get("lines_per_order_mean", 3.0))
    pick_time_per_line_min = float(cfg.get("pick_time_per_line_min", 1.5))
    pack_time_min = float(cfg.get("pack_time_min", 2.0))
    sla_minutes = float(cfg.get("order_sla_minutes", 120.0))

    env = simpy.Environment()
    dock_doors = simpy.Resource(env, capacity=num_docks)
    forklifts = simpy.Resource(env, capacity=num_forklifts)
    pickers = simpy.Resource(env, capacity=num_pickers)

    mon_docks = UtilizationMonitor(env, dock_doors, num_docks)
    mon_forks = UtilizationMonitor(env, forklifts, num_forklifts)
    mon_picks = UtilizationMonitor(env, pickers, num_pickers)

    trucks, orders = [], []

    def truck_generator():
        inter = exp_interarrival(inbound_rate_h)
        t = 0.0; i = 0
        while t < sim_minutes:
            ia = inter()
            if t + ia > sim_minutes: break
            t += ia; i += 1
            env.process(handle_truck(i, t))

    def handle_truck(idx, arrival_min):
        yield env.timeout(arrival_min - env.now)
        start_q = env.now
        with dock_doors.request() as req:
            yield req; mon_docks.acquire()
            q_wait = env.now - start_q

            # Unload
            with forklifts.request() as fkr:
                yield fkr; mon_forks.acquire()
                unload_t = normal_positive(unload_mean, unload_sd)
                yield env.timeout(unload_t)
                mon_forks.release()

            # Put-away
            with forklifts.request() as fkr2:
                yield fkr2; mon_forks.acquire()
                yield env.timeout(putaway_per_truck_min)
                mon_forks.release()

            dock_doors.release(req); mon_docks.release()
            trucks.append({
                "truck_id": idx,
                "arrival_min": arrival_min,
                "queue_wait_min": q_wait,
                "unload_min": unload_t,
                "putaway_min": putaway_per_truck_min,
                "departure_min": env.now
            })

    def order_generator():
        inter = exp_interarrival(order_rate_h)
        t = 0.0; oid = 0
        while t < sim_minutes:
            ia = inter()
            if t + ia > sim_minutes: break
            t += ia; oid += 1
            env.process(handle_order(oid, t))

    def handle_order(oid, arrival_min):
        yield env.timeout(arrival_min - env.now)
        with pickers.request() as req:
            start_q = env.now
            yield req; mon_picks.acquire()
            lines = max(1, int(round(np.random.exponential(lines_per_order_mean))))
            pick_time = lines * pick_time_per_line_min
            yield env.timeout(pick_time)
            yield env.timeout(pack_time_min)
            mon_picks.release()
        sojourn = env.now - arrival_min
        orders.append({
            "order_id": oid,
            "arrival_min": arrival_min,
            "lines": lines,
            "completion_min": env.now,
            "cycle_time_min": sojourn,
            "met_sla": 1 if sojourn <= sla_minutes else 0
        })

    env.process(truck_generator())
    env.process(order_generator())
    env.run(until=sim_minutes)

    trucks_df = pd.DataFrame(trucks)
    orders_df = pd.DataFrame(orders)

    metrics = {
        "sim_hours": cfg.get("sim_hours", 8),
        "throughput_trucks": len(trucks_df),
        "avg_truck_queue_wait_min": float(trucks_df["queue_wait_min"].mean()) if len(trucks_df) else 0.0,
        "throughput_orders": len(orders_df),
        "avg_order_cycle_time_min": float(orders_df["cycle_time_min"].mean()) if len(orders_df) else 0.0,
        "sla_hit_rate": float(orders_df["met_sla"].mean()) if len(orders_df) else 0.0,
        "dock_utilization": mon_docks.utilization(sim_minutes),
        "forklift_utilization": mon_forks.utilization(sim_minutes),
        "picker_utilization": mon_picks.utilization(sim_minutes),
    }
    return metrics, orders_df, trucks_df
