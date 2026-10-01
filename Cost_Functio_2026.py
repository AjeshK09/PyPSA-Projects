"""
tkwp_pipe_functions.py

DIRECT ANSWER to the assigned task:
    "Create a Python function that reads data points for costs and DN,
    and builds a cost function from it. [...] The Python function should
    depend on: Heat Network level (110C/70C/LT), delta T, paved/unpaved."

This file contains ONLY that -- two function-BUILDERS (build_cost_function
and build_capacity_function) and the real data they read. There is
deliberately NO optimizer, NO PyPSA network, NO SOS2/MILP machinery here:
none of that was asked for by the task above. This file answers "given a
diameter, what does it cost / what can it carry" -- a pure lookup and
interpolation tool, nothing more.

(An SOS2-based pipe-SIZING optimizer that uses these same functions to
automatically choose a diameter for a given heat demand exists separately,
in dh_pipe_tkwp_sos2.py / TKWP_SOS2.py, as an additional, optional
extension -- not part of what was asked for here.)

DATA SOURCE -- ONE catalogue edition only
-------------------------------------------
Every number below comes from a SINGLE edition of the German TKWP
(Technikkatalog Wärmeplanung): KWW-Technikkatalog-Waermeplanung_08-2026_
Version_1_1.xlsx (August 2026). That edition happens to contain all three
required network levels:
    Tab 36 -> "Wärmenetze 'konventionell' 110°C" (KMR)     -> "110C", DN20-DN500
    Tab 37 -> "Wärmenetze 'innovativ' 70°C" (PMR)           -> "70C",  DN20-DN160
    Tab 38 -> "Wärmenetze 'kalte Nahwärme'" (LT/unisolated) -> "LT",   DN50-DN250
No other edition or third-party source (nPro etc.) is used anywhere in
this file.

WHAT WAS DELIBERATELY EXCLUDED, per the task's own instructions
-------------------------------------------------------------------
- Building/house-connection ("Hausanschluss") costs: excluded. Only
  MAIN PIPE figures (material + trenching) are used.
- Every catalogue tab's own footnote states its prices are PER SINGLE
  PIPE. A real network needs a supply+return PAIR, so every cost figure
  here is DOUBLED (x2) before being returned.
"""

from __future__ import annotations

from typing import Callable, Literal

import numpy as np
from scipy.interpolate import PchipInterpolator

NetworkLevel = Literal["110C", "70C", "LT"]
Terrain = Literal["unpaved", "paved"]
TerrainStat = Literal["min", "max", "mean"]


# DATA Points from TWKP August-2026 edition


TKWP_DATA: dict[NetworkLevel, dict] = {
    "110C": {
        "label": "Wärmenetze 'konventionell' 110°C (KMR) -- Tab 36",
        "DN_mm":               [20,  25,  32,  40,  50,  65,  80,  100,  125,  150,  200,  250,  300,  350,  400,  450,  500],
        "material_eur_per_m":  [71,  87,  111, 137, 169, 217, 265, 328,  406,  483,  636,  787,  937,  1086, 1233, 1380, 1526],
        "trench_unpaved_min":  [161, 196, 244, 297, 362, 457, 549, 669,  815,  958,  1236, 1506, 1769, 2028, 2283, 2534, 2782],
        "trench_unpaved_max":  [317, 381, 469, 564, 680, 846, 1006, 1212, 1460, 1700, 2161, 2603, 3031, 3447, 3853, 4250, 4641],
        "trench_paved_min":    [252, 308, 386, 473, 580, 736, 889, 1090, 1335, 1576, 2048, 2509, 2962, 3409, 3849, 4285, 4716],
        "trench_paved_max":    [591, 711, 872, 1050, 1263, 1570, 1865, 2244, 2701, 3142, 3989, 4800, 5584, 6345, 7089, 7816, 8530],
    },
    "70C": {
        "label": "Wärmenetze 'innovativ' 70°C (PMR) -- Tab 37",
        "DN_mm":               [20,  25,  32,  40,  50,  63,  75,  90,   110,  125,  160],
        "material_eur_per_m":  [52,  67,  90,  117, 152, 200, 246, 305,  387,  450,  602],
        "trench_unpaved_min":  [67,  73,  81,  88,  96,  105, 112, 120,  130,  136,  150],
        "trench_unpaved_max":  [172, 191, 214, 237, 262, 292, 316, 344,  377,  400,  448],
        "trench_paved_min":    [152, 167, 186, 204, 224, 248, 267, 289,  314,  332,  369],
        "trench_paved_max":    [294, 377, 497, 638, 818, 1058, 1286, 1576, 1972, 2274, 2995],
    },
    "LT": {
        "label": "Wärmenetze Kalte Nahwärme (LT-Netz) -- Tab 38",
        "DN_mm":               [50,  75,  90,  100, 110, 125, 140, 160, 180, 200, 225, 250],
        "material_eur_per_m":  [11,  21,  28,  33,  38,  47,  56,  69,  84,  99,  119, 141],
        "trench_unpaved_min":  [125, 158, 176, 187, 197, 213, 227, 245, 263, 279, 299, 318],
        "trench_unpaved_max":  [125, 158, 176, 187, 197, 213, 227, 245, 263, 279, 299, 318],
        "trench_paved_min":    [265, 324, 355, 374, 392, 418, 442, 472, 501, 528, 560, 590],
        "trench_paved_max":    [265, 324, 355, 374, 392, 418, 442, 472, 501, 528, 560, 590],
    },
}


# ============================================================================
# STEP 2 -- turn the raw table into (DN, cost) data points for one choice
#           of network level / terrain
# ============================================================================

def get_pipe_cost_datapoints(
    network_level: NetworkLevel,
    terrain: Terrain = "unpaved",
    terrain_stat: TerrainStat = "mean",
    pair_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (DN_mm[], cost_eur_per_m[]): MAIN PIPE cost (material +
    trenching) at every catalogue diameter for one network level and
    terrain, DOUBLED (pair_factor, default 2.0) to go from the catalogue's
    single-pipe prices to a real supply+return pair.
    """
    data = TKWP_DATA[network_level]
    dn = np.array(data["DN_mm"], dtype=float)
    material = np.array(data["material_eur_per_m"], dtype=float)
    lo = np.array(data[f"trench_{terrain}_min"], dtype=float)
    hi = np.array(data[f"trench_{terrain}_max"], dtype=float)
    trench = {"min": lo, "max": hi, "mean": (lo + hi) / 2.0}[terrain_stat]
    cost_pair = (material + trench) * pair_factor
    return dn, cost_pair


# ============================================================================
# STEP 3 -- THE COST FUNCTION BUILDER (the core deliverable)
# ============================================================================

def build_cost_function(
    network_level: NetworkLevel,
    terrain: Terrain = "unpaved",
    terrain_stat: TerrainStat = "mean",
    beyond_data_source: Literal["linear", "flat"] = "linear",
) -> Callable[[float], float]:
    """Build and return a CALLABLE cost_per_m(diameter_mm) -> EUR/m.

    Interpolates (PCHIP: monotonic piecewise cubic Hermite) exactly
    through the real catalogue data points for the chosen network level
    and terrain -- works at ANY diameter, including ones the catalogue
    never tabulated (e.g. cost_per_m(73.4)).

    Rebuildable by design: this reads current values out of TKWP_DATA
    every time it is called, so editing that dictionary (e.g. pasting in
    a newer catalogue edition, or different data entirely) immediately
    changes what this function returns -- no other code needs to change.

    Diameters beyond the catalogue's own maximum are extrapolated using
    the slope of the last known segment (default) or held flat -- treat
    either as a rough placeholder, since the catalogue's own stated focus
    is on small/medium networks.
    """
    dn, cost = get_pipe_cost_datapoints(network_level, terrain, terrain_stat)

    dn_full = np.concatenate([[0.0], dn])      # (0mm, 0 EUR/m): "no pipe"
    cost_full = np.concatenate([[0.0], cost])

    pchip = PchipInterpolator(dn_full, cost_full, extrapolate=False)
    dn_max = float(dn_full.max())
    cost_at_max = float(cost_full[-1])
    slope_last_segment = (
        cost_full[-1] - cost_full[-2]) / (dn_full[-1] - dn_full[-2])

    def cost_per_m(diameter_mm: float) -> float:
        if diameter_mm < 0:
            raise ValueError("diameter_mm must be >= 0")
        if diameter_mm <= dn_max:
            return float(pchip(diameter_mm))
        if beyond_data_source == "flat":
            return cost_at_max
        return cost_at_max + slope_last_segment * (diameter_mm - dn_max)

    cost_per_m.network_level = network_level      # type: ignore[attr-defined]
    # type: ignore[attr-defined]
    cost_per_m.terrain = terrain
    # type: ignore[attr-defined]
    cost_per_m.terrain_stat = terrain_stat
    # type: ignore[attr-defined]
    cost_per_m.dn_data_max_mm = dn_max
    return cost_per_m


# STEP 4 -- THE CAPACITY FUNCTION BUILDER (the delta_T dependency)

RHO_WATER = 1000.0       # kg/m3
CP_WATER = 4185.0        # J/(kg K)
K_ROUGHNESS_MM = 0.1     # typical new steel/PMR district-heating pipe
MU_WATER_PAS = 4.0e-4    # dynamic viscosity, ~65 degC mean pipe temperature


def _nikuradse_friction_factor(reynolds: float, k_mm: float, d_m: float) -> float:
    re = max(reynolds, 1e-6)
    return 64.0 / re + 1.0 / (-2.0 * np.log10((k_mm / 1000.0) / (3.71 * d_m))) ** 2


def _max_velocity_pressure_limited(diameter_mm: float, r_max_pa_per_m: float,
                                   v_max_absolute: float) -> float:
    """Iteratively solve Darcy-Weisbach + Nikuradse for the velocity that
    produces exactly r_max_pa_per_m of pressure gradient, capped at
    v_max_absolute.
    """
    if diameter_mm <= 0:
        return 0.0
    d_m = diameter_mm / 1000.0
    v = 1.0
    for _ in range(100):
        re = RHO_WATER * v * d_m / MU_WATER_PAS
        lam = _nikuradse_friction_factor(re, K_ROUGHNESS_MM, d_m)
        v_new = np.sqrt(2.0 * d_m * r_max_pa_per_m / (lam * RHO_WATER))
        if abs(v_new - v) < 1e-8:
            v = v_new
            break
        v = v_new
    return min(v, v_max_absolute)


def build_capacity_function(
    network_level: NetworkLevel,
    delta_t_k: float = 30.0,
    r_max_pa_per_m: float = 100.0,
    v_max_absolute_m_s: float = 3.0,
) -> Callable[[float], float]:
    """Build and return a CALLABLE capacity_mw(diameter_mm) -> MW.

    Velocity is derived per diameter from a maximum pressure-gradient
    design criterion (Darcy-Weisbach + Nikuradse friction factor), capped
    at v_max_absolute_m_s -- NOT a single fixed velocity for every
    diameter, since that would make capacity scale simply as D^2 and
    understate real economies of scale.

    Capacity is directly proportional to delta_t_k (Q = mdot * cp * dT),
    with mdot set by that pressure-gradient-limited velocity.
    """
    def capacity_mw(diameter_mm: float) -> float:
        if diameter_mm <= 0:
            return 0.0
        d_m = diameter_mm / 1000.0
        area = np.pi / 4.0 * d_m ** 2
        v = _max_velocity_pressure_limited(
            diameter_mm, r_max_pa_per_m, v_max_absolute_m_s)
        return RHO_WATER * CP_WATER * v * area * delta_t_k / 1e6

    capacity_mw.network_level = network_level    # type: ignore[attr-defined]
    capacity_mw.delta_t_k = delta_t_k              # type: ignore[attr-defined]
    return capacity_mw


# ============================================================================
# STEP 5 -- DEMO: exactly what the task asked for, nothing more
# ============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("Cost function: depends on network level, terrain not on delta_T")
    print("=" * 78)
    for level in ["110C", "70C", "LT"]:
        for terrain in ["unpaved", "paved"]:
            f = build_cost_function(
                level, terrain=terrain, terrain_stat="mean")
            print(f"{level:5s} at {terrain:7s}  DN100 the cost is {f(100):9.2f} EUR/m   "
                  f"Give Diameter Here -> {f(62):9.2f} EUR/m")
    print("Capacity function depends on network level and delta_T")
    for dT in [20, 30, 45, 60]:
        cap = build_capacity_function("110C", delta_t_k=dT)
        print(
            f"  dT={dT:3.0f} K:     Calculated capacity with given diameter {cap(28):.3f} MW   DN300 -> {cap(64):.3f} MW")

    print("Both together: cost and capacity at the same diameter (110C, unpaved)")
    cost_fn = build_cost_function("110C", terrain="unpaved")
    cap_fn = build_capacity_function("110C", delta_t_k=30.0)
    for d in [50, 100, 172, 111, 34, 200, 400, 198]:
        print(
            f"  D={d:4d}mm:  cost={cost_fn(d):9.2f} EUR/m   capacity={cap_fn(d):7.3f} MW")
f = build_cost_function("110C", terrain="unpaved")
