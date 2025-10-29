#!/usr/bin/env python3
"""
Single-file runner for miniSCOT-style simulations.

Usage:
  python miniscot_single.py --config configs/baseline.yaml --out results.csv
  python miniscot_single.py --mock --out results.csv  # quick dry run w/o engine

What you must edit:
  - Update the two import lines marked "# >>> EDIT THIS IMPORT <<<"
    to match your fork's module path.

Examples (pick one that matches your repo):
  from scse.simulator import Simulator
  from scse.config import load_config

  or

  from supply_chain_sim.engine import Simulator
  from supply_chain_sim.io import load_config
"""

import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# -----------------------------
# Try to import your engine API
# -----------------------------
ENGINE_OK = True
try:
    # >>> EDIT THIS IMPORT <<< (example A)
    # from scse.simulator import Simulator
    # from scse.config import load_config

    # >>> EDIT THIS IMPORT <<< (example B)
    # from supply_chain_sim.engine import Simulator
    # from supply_chain_sim.io import load_config

    # If your repo isn't a package, add repo root to sys.path, then import:
    REPO_ROOT = Path(__file__).resolve().parent
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    # If your repo provides helpers in examples/, you can import from there as needed.
    # For now we define thin adapters below; set ENGINE_OK=False to use --mock.
    raise ImportError("Point the imports above to your fork, then remove this line.")
except Exception:
    ENGINE_OK = False


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def tiny_default_config() -> dict:
    """A very small config so runs finish fast on any laptop/CI."""
    return {
        "sim_days": 14,
        "avg_daily_demand": 120,
        "safety_stock": 500,
        "lead_time_mean": 2.0,
        "inventory_holding_per_unit": 0.02,
        "transport_cost_per_unit": 1.2,
        # Add any engine-specific fields here if needed, e.g. seeds, replications, etc.
    }


# -----------------------------
# MOCK SIM (optional quick test)
# -----------------------------
def simulate_mock(cfg: dict) -> pd.DataFrame:
    """
    Stand-in for the real engine so you can validate IO & KPIs.
    Returns a tidy dataframe with period-level metrics.
    """
    np.random.seed(42)
    periods = int(cfg.get("sim_days", 14))
    periods = max(7, min(60, periods))

    demand = np.random.poisson(cfg.get("avg_daily_demand", 120), periods)
    safety_stock = cfg.get("safety_stock", 500)
    lead_mu = cfg.get("lead_time_mean", 2.0)
    transport_cost_per_unit = cfg.get("transport_cost_per_unit", 1.2)
    inv_cost_per_unit = cfg.get("inventory_holding_per_unit", 0.02)

    fill_rate = np.clip(0.85 + (safety_stock / 5000.0) + np.random.normal(0, 0.01, periods), 0.85, 0.999)
    stockouts = np.maximum(0, (1.0 - fill_rate) * demand).astype(int)
    lead_time_mean = np.clip(np.random.normal(lead_mu, 0.3, periods), 0.5, None)
    transport_cost = demand * transport_cost_per_unit * (1.0 + 0.02 * (lead_time_mean - lead_mu))
    inventory_cost = (safety_stock * inv_cost_per_unit) * (1 + 0.005 * np.arange(periods))
    total_cost = transport_cost + inventory_cost

    df = pd.DataFrame({
        "period": np.arange(1, periods + 1),
        "demand": demand,
        "fill_rate": fill_rate,
        "stockouts": stockouts,
        "lead_time_mean": lead_time_mean,
        "transport_cost": transport_cost,
        "inventory_cost": inventory_cost,
        "total_cost": total_cost,
    })
    return df


# -----------------------------
# REAL ENGINE ADAPTER (edit me)
# -----------------------------
def simulate_with_engine(cfg: dict) -> pd.DataFrame:
    """
    Adapt this to your engine’s API.
    Typical pattern:
        sim = Simulator(cfg)
        results = sim.run()
        df = results.to_dataframe()   # or build from logs
    Return a tidy DataFrame with at least: period, fill_rate, stockouts, total_cost, ...
    """
    # Example skeleton—replace with actual calls from your fork:
    raise NotImplementedError(
        "Wire this function to your repo's Simulator (see the import section)."
    )


# -----------------------------
# KPI helpers
# -----------------------------
def compute_kpis(df: pd.DataFrame) -> dict:
    return {
        "avg_fill_rate": float(df["fill_rate"].mean()),
        "total_stockouts": int(df["stockouts"].sum()),
        "avg_lead_time": float(df["lead_time_mean"].mean()),
        "total_cost": float(df["total_cost"].sum()),
        "avg_daily_demand": float(df["demand"].mean()) if "demand" in df else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Single-file miniSCOT runner")
    ap.add_argument("--config", type=str, help="Path to YAML config")
    ap.add_argument("--out", type=str, default="results.csv", help="CSV to write")
    ap.add_argument("--mock", action="store_true", help="Use mock simulator (no engine)")
    args = ap.parse_args()

    if args.config:
        cfg_path = Path(args.config)
        if not cfg_path.exists():
            print(f"[error] config not found: {cfg_path}")
            sys.exit(2)
        cfg = load_yaml(cfg_path)
    else:
        cfg = tiny_default_config()

    # Keep small for quick iteration
    cfg.setdefault("sim_days", min(int(cfg.get("sim_days", 14)), 30))

    # Run
    if args.mock or not ENGINE_OK:
        if not args.mock:
            print("[info] Engine imports not wired yet; falling back to --mock.")
        df = simulate_mock(cfg)
    else:
        df = simulate_with_engine(cfg)

    # Output
    out_path = Path(args.out)
    df.to_csv(out_path, index=False)

    kpis = compute_kpis(df)
    print("\n=== Simulation KPIs ===")
    print(json.dumps(kpis, indent=2))
    print(f"\n[ok] wrote {len(df)} rows to {out_path.resolve()}")


if __name__ == "__main__":
    main()
