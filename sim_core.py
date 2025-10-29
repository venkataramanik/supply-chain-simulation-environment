# sim_core.py
# Complete SimPy warehouse core: KPI run + animated run with position logging.

import math
import random
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import simpy


# -----------------------------
# Utilities & monitors
# -----------------------------
class UtilizationMonitor:
    """Time-weighted utilization for a SimPy Resource."""
    def __init__(self, env: simpy.Environment, resource: simpy.Resource, capacity: int):
        self.env = env
        self.capacity = capacity
        self._busy = 0
        self._last_t = 0.0
        self._busy_time = 0.0

    def _note(self):
        now = self.env.now
        dt = now - self._last_t
        self._busy_time += self._busy * dt
        self._last_t = now

    def acquire(self):
        self._note()
        self._busy += 1

    def release(self):
        self._note()
        self._busy -= 1

    def utilization(self, sim_time: float) -> float:
        if sim_time <= 0:
            return 0.0
        # Close out the timeline
        self._note()
        return min(1.0, max(0.0, self._busy_time / (self.capacity * sim_time)))


def exp_interarrival(rate_per_hour: float):
    """Factory: exponential interarrival (minutes) for a given per-hour rate."""
    lam = rate_per_hour / 60.0  # per minute
    def _draw():
        return np.random.exponential(1.0 / lam) if lam > 0 else math.inf
    return _draw


def normal_positive(mean: float, sd: float) -> float:
    """Truncated normal (>0) sample in minutes."""
    x = np.random.normal(mean, sd)
    while x <= 0:
        x = np.random.normal(mean, sd)
    return x


# -----------------------------
# Core KPI simulation (no animation)
# -----------------------------
def run_simpy_warehouse(cfg: Dict, seed: int = 42) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Discrete-event warehouse simulation:
      - Inbound: trucks -> dock doors (queue) -> unload (forklifts) -> put-away (forklifts)
      - Outbound: orders -> pick (pickers) -> pack -> complete (SLA)
    Returns:
      metrics (dict), orders_df, trucks_df
    """
    np.random.seed(seed)
    random.seed(seed)

    sim_minutes = int(cfg.get("sim_hours", 8) * 60)

    # Parameters
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

    trucks: List[Dict] = []
    orders: List[Dict] = []

    def truck_generator():
        """Generate inbound trucks via exponential interarrival and start their processes."""
        inter = exp_interarrival(inbound_rate_h)
        i = 0
        while env.now < sim_minutes:
            ia = inter()
            if env.now + ia > sim_minutes:
                break
            # wait to next arrival
            yield env.timeout(ia)
            i += 1
            env.process(handle_truck(i, env.now))

    def handle_truck(idx: int, arrival_min: float):
        # Request a dock
        start_q = env.now
        with dock_doors.request() as req:
            yield req
            mon_docks.acquire()
            q_wait = env.now - start_q

            # Unload (forklift used)
            with forklifts.request() as fkr:
                yield fkr
                mon_forks.acquire()
                unload_t = normal_positive(unload_mean, unload_sd)
                yield env.timeout(unload_t)
                mon_forks.release()

            # Put-away (forklift again)
            with forklifts.request() as fkr2:
                yield fkr2
                mon_forks.acquire()
                yield env.timeout(putaway_per_truck_min)
                mon_forks.release()

            dock_doors.release(req)
            mon_docks.release()

            trucks.append({
                "truck_id": idx,
                "arrival_min": arrival_min,
                "queue_wait_min": q_wait,
                "unload_min": unload_t,
                "putaway_min": putaway_per_truck_min,
                "departure_min": env.now
            })

    def order_generator():
        """Generate customer orders via exponential interarrival and start their processes."""
        inter = exp_interarrival(order_rate_h)
        oid = 0
        while env.now < sim_minutes:
            ia = inter()
            if env.now + ia > sim_minutes:
                break
            yield env.timeout(ia)
            oid += 1
            env.process(handle_order(oid, env.now))

    def handle_order(oid: int, arrival_min: float):
        with pickers.request() as req:
            start_q = env.now
            yield req
            mon_picks.acquire()
            # Random order size (exp)
            lines = max(1, int(round(np.random.exponential(lines_per_order_mean))))
            pick_time = lines * pick_time_per_line_min
            # pick + pack
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

    trucks_df = pd.DataFrame(trucks) if trucks else pd.DataFrame(
        columns=["truck_id", "arrival_min", "queue_wait_min", "unload_min", "putaway_min", "departure_min"]
    )
    orders_df = pd.DataFrame(orders) if orders else pd.DataFrame(
        columns=["order_id", "arrival_min", "lines", "completion_min", "cycle_time_min", "met_sla"]
    )

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


# -----------------------------
# Animated simulation (position logging)
# -----------------------------
def run_simpy_warehouse_animated(cfg: Dict, step_seconds: int = 5, seed: int = 42):
    """
    Like run_simpy_warehouse, but logs agent positions every `step_seconds` for animation.
    Returns:
      frames (List[pd.DataFrame]), shapes (list), labels (dict),
      metrics (dict), orders_df, trucks_df
    """
    np.random.seed(seed)
    random.seed(seed)

    # Minutes ↔ seconds
    sim_minutes = int(cfg.get("sim_hours", 8) * 60)
    sim_seconds = sim_minutes * 60

    # Parameters
    n_docks = int(cfg.get("num_dock_doors", 3))
    n_forklifts = int(cfg.get("num_forklifts", 2))
    n_pickers = int(cfg.get("num_pickers", 3))
    inbound_rate_h = float(cfg.get("inbound_trucks_per_hour", 4))
    order_rate_h = float(cfg.get("orders_per_hour", 60))

    unload_mean = float(cfg.get("unload_mean_min", 25.0))
    unload_sd = float(cfg.get("unload_sd_min", 5.0))
    putaway_per_truck_min = float(cfg.get("putaway_minutes_per_truck", 10.0))

    lines_per_order_mean = float(cfg.get("lines_per_order_mean", 3.0))
    pick_time_per_line_min = float(cfg.get("pick_time_per_line_min", 1.5))
    pack_time_min = float(cfg.get("pack_time_min", 2.0))
    sla_minutes = float(cfg.get("order_sla_minutes", 120.0))

    # Static layout (used for animation)
    shapes, labels = _warehouse_layout(n_docks=n_docks)
    W, H = labels["dims"]
    docks_xy = labels["docks_xy"]
    racks_xy = labels["racks_xy"]
    pack_xy = labels["pack_xy"]

    env = simpy.Environment()
    dock_res = simpy.Resource(env, capacity=n_docks)
    fork_res = simpy.Resource(env, capacity=n_forklifts)
    pick_res = simpy.Resource(env, capacity=n_pickers)

    mon_docks = UtilizationMonitor(env, dock_res, n_docks)
    mon_forks = UtilizationMonitor(env, fork_res, n_forklifts)
    mon_picks = UtilizationMonitor(env, pick_res, n_pickers)

    # Mutable agent positions
    forklifts = [(20, H * (i + 1) / (n_forklifts + 1)) for i in range(n_forklifts)]
    pickers = [(50, racks_xy[i % len(racks_xy)][1]) for i in range(n_pickers)]
    dock_slots: List[Tuple[int, float] | None] = [None] * n_docks  # slot -> (truck_id, depart_time_s)
    next_truck_id = 1

    trucks: List[Dict] = []
    orders: List[Dict] = []
    traces: List[pd.DataFrame] = []

    # --- Helpers for motion & logging ---
    def _move_step(p0, p1, step):
        x0, y0 = p0
        x1, y1 = p1
        dx, dy = x1 - x0, y1 - y0
        dist = math.hypot(dx, dy)
        if dist <= step:
            return (x1, y1)
        return (x0 + dx / dist * step, y0 + dy / dist * step)

    def record_frame():
        t = env.now  # seconds
        rows = []
        for idx, slot in enumerate(dock_slots):
            if slot is not None:
                x, y = docks_xy[idx]
                rows.append([t, f"Truck-{slot[0]}", "truck", x, y])
        for i, (x, y) in enumerate(forklifts, 1):
            rows.append([t, f"Fork-{i}", "forklift", x, y])
        for i, (x, y) in enumerate(pickers, 1):
            rows.append([t, f"Picker-{i}", "picker", x, y])
        if rows:
            traces.append(pd.DataFrame(rows, columns=["t", "agent", "type", "x", "y"]))

    def move_agent(agent_list, idx, target_xy, speed_units_per_min):
        """Coroutine to move an agent over time toward a target."""
        speed = speed_units_per_min / 60.0  # units/sec
        while True:
            p0 = agent_list[idx]
            p1 = target_xy
            step = speed * step_seconds
            p_next = _move_step(p0, p1, step)
            agent_list[idx] = p_next
            if p_next == p1:
                break
            yield env.timeout(step_seconds)

    def heartbeat():
        """Global sampler for animation frames."""
        while env.now < sim_seconds:
            record_frame()
            yield env.timeout(step_seconds)
        record_frame()

    # --- Processes ---
    def truck_generator():
        lam = inbound_rate_h / 60.0  # per minute
        while env.now < sim_seconds:
            ia_min = np.random.exponential(1.0 / lam) if lam > 0 else math.inf
            ia_sec = ia_min * 60.0
            if env.now + ia_sec > sim_seconds:
                break
            yield env.timeout(ia_sec)
            env.process(handle_truck(env.now))  # arrival at current time (seconds)

    def handle_truck(arrival_s: float):
        nonlocal next_truck_id
        # Request dock
        start_q = env.now
        with dock_res.request() as req:
            yield req
            mon_docks.acquire()
            q_wait = (env.now - start_q) / 60.0  # minutes

            # Pick a free slot
            slot_idx = None
            for i in range(n_docks):
                if dock_slots[i] is None:
                    slot_idx = i
                    break
            if slot_idx is None:
                slot_idx = 0  # defensive fallback

            truck_id = next_truck_id
            next_truck_id += 1

            # Unload using forklift
            with fork_res.request() as frq:
                yield frq
                mon_forks.acquire()
                # send nearest forklift to dock
                fk_idx = int(np.argmin([
                    math.hypot(forklifts[k][0] - docks_xy[slot_idx][0],
                               forklifts[k][1] - docks_xy[slot_idx][1])
                    for k in range(len(forklifts))
                ]))
                env.process(move_agent(forklifts, fk_idx, docks_xy[slot_idx], speed_units_per_min=60 * 1.2))

                unload_t_min = normal_positive(unload_mean, unload_sd)
                unload_t_s = unload_t_min * 60.0

                depart_time_s = env.now + unload_t_s + putaway_per_truck_min * 60.0
                dock_slots[slot_idx] = (truck_id, depart_time_s)

                yield env.timeout(unload_t_s)

                # Put-away run to a rack and pack area
                rack_xy = racks_xy[truck_id % len(racks_xy)]
                env.process(move_agent(forklifts, fk_idx, rack_xy, speed_units_per_min=60 * 1.2))
                yield env.timeout(putaway_per_truck_min * 60.0)
                # park back at rack
                env.process(move_agent(forklifts, fk_idx, rack_xy, speed_units_per_min=60 * 1.2))

                mon_forks.release()

            dock_res.release(req)
            mon_docks.release()

            def clear_slot():
                yield env.timeout(max(0.0, depart_time_s - env.now))
                dock_slots[slot_idx] = None
            env.process(clear_slot())

            trucks.append({
                "truck_id": truck_id,
                "arrival_min": arrival_s / 60.0,
                "queue_wait_min": q_wait,
                "unload_min": unload_t_min,
                "putaway_min": putaway_per_truck_min,
                "departure_min": depart_time_s / 60.0
            })

    def order_generator():
        lam = order_rate_h / 60.0
        oid = 0
        while env.now < sim_seconds:
            ia_min = np.random.exponential(1.0 / lam) if lam > 0 else math.inf
            ia_sec = ia_min * 60.0
            if env.now + ia_sec > sim_seconds:
                break
            yield env.timeout(ia_sec)
            oid += 1
            env.process(handle_order(oid, env.now))

    def handle_order(oid: int, arrival_s: float):
        with pick_res.request() as req:
            start_q = env.now
            yield req
            mon_picks.acquire()
            pk_idx = (oid - 1) % n_pickers
            rack_xy = racks_xy[oid % len(racks_xy)]
            # move to rack
            env.process(move_agent(pickers, pk_idx, rack_xy, speed_units_per_min=60 * 0.8))

            lines = max(1, int(round(np.random.exponential(lines_per_order_mean))))
            pick_t_s = lines * pick_time_per_line_min * 60.0
            yield env.timeout(pick_t_s)

            # to pack
            env.process(move_agent(pickers, pk_idx, pack_xy, speed_units_per_min=60 * 0.8))
            yield env.timeout(pack_time_min * 60.0)

            # return toward racks
            env.process(move_agent(pickers, pk_idx, rack_xy, speed_units_per_min=60 * 0.8))

            mon_picks.release()

        sojourn_min = (env.now - arrival_s) / 60.0
        orders.append({
            "order_id": oid,
            "arrival_min": arrival_s / 60.0,
            "lines": lines,
            "completion_min": env.now / 60.0,
            "cycle_time_min": sojourn_min,
            "met_sla": 1 if sojourn_min <= sla_minutes else 0
        })

    # Run
    env.process(heartbeat())
    env.process(truck_generator())
    env.process(order_generator())
    env.run(until=sim_seconds)

    # Build per-minute frames for the slider
    if not traces:
        # ensure at least one frame exists
        traces.append(pd.DataFrame([{"t": 0, "agent": "Picker-1", "type": "picker", "x": 50, "y": H / 2}]))

    frames_map: Dict[int, List[pd.DataFrame]] = {}
    for df in traces:
        minute = int(round(df["t"].iloc[0] / 60.0))
        frames_map.setdefault(minute, [])
        frames_map[minute].append(df.assign(t=minute))

    frames = [pd.concat(v, ignore_index=True) for k, v in sorted(frames_map.items())]

    trucks_df = pd.DataFrame(trucks) if trucks else pd.DataFrame(
        columns=["truck_id", "arrival_min", "queue_wait_min", "unload_min", "putaway_min", "departure_min"]
    )
    orders_df = pd.DataFrame(orders) if orders else pd.DataFrame(
        columns=["order_id", "arrival_min", "lines", "completion_min", "cycle_time_min", "met_sla"]
    )
    metrics = {
        "orders": len(orders_df),
        "avg_cycle_min": float(orders_df["cycle_time_min"].mean()) if len(orders_df) else 0.0,
        "sla_rate": float(orders_df["met_sla"].mean()) if len(orders_df) else 0.0,
        "trucks": len(trucks_df),
    }
    return frames, shapes, labels, metrics, orders_df, trucks_df


# -----------------------------
# Layout helper (shared with viz)
# -----------------------------
def _warehouse_layout(width: int = 100, height: int = 60, n_docks: int = 3, n_rack_rows: int = 3):
    """Static layout used by the animated run."""
    shapes = []
    docks_xy = []
    dock_h = height / (n_docks + 1)
    dock_w = 8
    for i in range(n_docks):
        y_center = (i + 1) * dock_h
        x0, x1 = 2, 2 + dock_w
        y0, y1 = y_center - dock_h / 3, y_center + dock_h / 3
        shapes.append(dict(type="rect", x0=x0, y0=y0, x1=x1, y1=y1,
                           line=dict(color="black"), fillcolor="lightgray"))
        docks_xy.append(((x0 + x1) / 2, (y0 + y1) / 2))

    racks_xy = []
    rack_zone_x0, rack_zone_x1 = 30, 70
    rack_gap = (height - 10) / max(1, n_rack_rows)
    for r in range(n_rack_rows):
        y0 = 5 + r * rack_gap
        y1 = y0 + rack_gap * 0.5
        shapes.append(dict(type="rect", x0=rack_zone_x0, y0=y0, x1=rack_zone_x1, y1=y1,
                           line=dict(color="black"), fillcolor="#E8F0FE"))
        racks_xy.append(((rack_zone_x0 + rack_zone_x1) / 2, (y0 + y1) / 2))

    pack_x0, pack_x1 = 78, 95
    pack_y0, pack_y1 = height * 0.25, height * 0.75
    shapes.append(dict(type="rect", x0=pack_x0, y0=pack_y0, x1=pack_x1, y1=pack_y1,
                       line=dict(color="black"), fillcolor="#FFE8CC"))

    labels = {
        "docks_xy": docks_xy,
        "racks_xy": racks_xy,
        "pack_xy": ((pack_x0 + pack_x1) / 2, (pack_y0 + pack_y1) / 2),
        "dims": (width, height),
    }
    return shapes, labels
