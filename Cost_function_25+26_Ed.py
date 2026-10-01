"""
tkwp_cost_functions.py

Reads primary TKWP (Technikkatalog Wärmeplanung) pipe cost/DN data points
and builds a CALLABLE cost function from them -- rather than hardcoding a
single fixed set of breakpoint costs, so the function can be rebuilt
whenever the underlying data changes or a different data source is used.

DATA SOURCES
------------
Primary (this file's uploaded workbook, KWW-Technikkatalog-Waermeplanung,
Dec-2025, version "12-2025"):
  - Tab 33: "Wärmenetze 'konventionell' - 110°C Vorlauftemperatur;
             Kunststoffmantelrohre (KMR)"           -> network_level="110C"
  - Tab 34: "Wärmenetze 'innovativ' - 70°C Vorlauftemperatur;
             Polymere Mediumrohre (PMR)"             -> network_level="70C"
  - Tab 35: "Wärmenetze Kalte Nahwärme"               -> network_level="LT"

Each tab's own annotation ("GL": "Es werden nur die Preise für Einzelrohre
dargestellt" -- "only single-pipe prices are shown") means every cost
figure in these tables is for ONE pipe. A real district-heating pipe
network needs a supply AND a return pipe (a pair), so -- exactly per your
instruction -- all costs read from these tables are DOUBLED (x2) before
being used anywhere in this module.

EXTENSION beyond the primary tables' own DN range (all three tabs stop at
DN150-DN250, i.e. small/medium network scale by the catalog's own stated
focus -- Tab 33's footnote "DM": "Bau neuer großer Fernwärmetrassen (DN
400/500) soll nicht mit abgedeckt werden [...]"):
  - For network_level="110C" specifically, we extend using Tab 36 of the
    PREVIOUSLY uploaded TKWP workbook (Aug-2026, version "1.1"), which
    documents the SAME technology/network level ("Wärmenetze
    'konventionell' - 110°C [...] KMR") but with data points running from
    DN20 up to DN500 -- i.e. it is the natural extension of this file's
    Tab 33 to larger diameters. NOTE: this splices two different EDITIONS
    of the TKWP (Dec-2025 vs Aug-2026); both describe the same technology,
    but the underlying figures may have been revised slightly between
    versions -- flagged explicitly wherever this extension is used.
  - Beyond DN500 (still only relevant for "110C"), we fall back to nPro's
    published polynomial fit (see dh_pipe_npro_validation.py in this same
    project), itself derived from Logstor's TCO tool + TKWP trenching
    data -- a genuinely different, independent source, used here only to
    reach diameters neither TKWP table covers at all.
  - "70C" (PMR) and "LT" (Kalte Nahwärme) are NOT extended: PMR pipes are
    explicitly a small/medium-network technology (footnote "DV": large
    trunk lines DN400/500 out of scope), and Kalte Nahwärme networks are
    inherently small-diameter by design philosophy, so extending either
    beyond the catalog's own tested range would not be physically
    meaningful without a genuinely different technology's data.

WHY delta_T DOES NOT CHANGE THE €/m COST CURVE ITSELF
------------------------------------------------------
Tab 33's own footnote "DP" explains how these DN-indexed costs were
derived: the original source data (Danish catalogue / KEA-BW) gives cost
as a function of THERMAL POWER, and TKWP converted this to a function of
DIAMETER using  Q = mdot_max * cp * dT  with a REFERENCE dT = 30 K and the
pipe's maximum mass flow rate from manufacturer datasheets. The price of a
physical pipe of a given diameter does not itself depend on which dT you
plan to operate it at -- what changes with dT is how much thermal
CAPACITY that same physical pipe can deliver. So in this module:
  - build_cost_function(...)     -> EUR/m as a function of DIAMETER only
                                     (delta_T has no effect here, by design)
  - build_capacity_function(...) -> MW as a function of DIAMETER, and here
                                     delta_T matters directly (capacity is
                                     proportional to dT, for a fixed
                                     maximum flow velocity -- same physics
                                     used throughout this project's models)
Both functions share the same diameter axis, so chaining them (as the
SOS2 pipe-sizing models in this project already do) gives you cost as a
function of capacity at whatever dT you choose, without ever needing to
invent a dT-dependence in the cost data that the source itself doesn't
have.
"""

from __future__ import annotations

from typing import Callable, Literal
import numpy as np
from scipy.interpolate import PchipInterpolator

# ============================================================================
# 1) RAW DATA POINTS -- kept as plain, easily-editable dictionaries so the
#    cost function can be rebuilt the moment new/different data is available.
#    All costs are €/2025 (Tabs 33-35) or €/2026 (extension, Tab 36 of the
#    Aug-2026 file) real, excl. VAT, PER SINGLE PIPE (doubled later).
# ============================================================================

TKWP_DATA = {
    # --- Tab 33: "konventionell" 110°C, KMR (Kunststoffmantelrohr) -----
    "110C": {
        "label": "Wärmenetze 'konventionell' 110°C (KMR)",
        "source": "KWW-Technikkatalog-Waermeplanung_12-2025, Tab 33",
        "DN_mm": [20, 25, 32, 40, 50, 65, 80, 100, 125, 150],
        "heat_loss_w_per_m_single": [12, 14, 14, 16, 18, 20, 22, 22, 25, 30],
        "lifetime_years": 30,
        "trench_unpaved_min": [200, 315, 400, 417, 417, 518, 518, 697, 722, 722],
        "trench_unpaved_max": [600, 650, 700, 750, 800, 850, 900, 1050, 1150, 1250],
        "trench_paved_min": [378, 378, 480, 500, 500, 623, 623, 867, 867, 867],
        "trench_paved_max": [1500, 1500, 1500, 1500, 1500, 2000, 2000, 2500, 2500, 2500],
        "material_eur_per_m": [70, 90, 100, 140, 200, 240, 280, 390, 560, 1000],
        # cost-vs-DN in this table derived from Danish-catalogue cost-vs-power
        # data using Q = mdot_max * cp * dT with reference dT_ref = 30 K
        "dT_reference_K": 30.0,
    },
    # --- Tab 34: "innovativ" 70°C, PMR (Polymere Mediumrohre) -----------
    "70C": {
        "label": "Wärmenetze 'innovativ' 70°C (PMR)",
        "source": "KWW-Technikkatalog-Waermeplanung_12-2025, Tab 34",
        "DN_mm": [25, 32, 40, 50, 63, 75, 90, 110, 125, 160],
        "heat_loss_w_per_m_single": [10, 10, 14, 16, 18, 20, 22, 24, 26, 30],  # @100 degC ref (footnote DW)
        "lifetime_years": 50,
        "trench_unpaved_min": [60, 60, 60, 60, 60, 80, 80, 100, 100, 100],
        "trench_unpaved_max": [200, 200, 200, 200, 200, 250, 250, 300, 300, 300],
        "trench_paved_min": [150, 150, 150, 150, 150, 200, 200, 250, 250, 250],
        "trench_paved_max": [1500, 1500, 1500, 1500, 1500, 2000, 2000, 2500, 2500, 2500],
        "material_eur_per_m": [44, 52, 67, 92, 121, 136, 174, 208, 283, 498],
        "dT_reference_K": 30.0,  # not separately stated for Tab 34; assumed same convention as Tab 33
    },
    # --- Tab 35: "Kalte Nahwärme" (cold local heating / LT network) -----
    "LT": {
        "label": "Wärmenetze Kalte Nahwärme (LT-Netz)",
        "source": "KWW-Technikkatalog-Waermeplanung_12-2025, Tab 35",
        "DN_mm": [50, 75, 90, 100, 110, 125, 140, 160, 180, 200, 225, 250],
        # NOTE: this is a HEAT GAIN (not loss) for cold networks -- stored
        # as a negative "loss" so downstream code can treat it uniformly.
        "heat_gain_pct_of_demand": 20.0,  # flat 20% for all DN (footnote EB)
        "lifetime_years": 50,
        # Tab 35 gives only ONE trenching figure per terrain (no min/max range)
        "trench_unpaved": [130, 154, 168, 179, 189, 203, 218, 237, 256, 276, 300, 324],
        "trench_paved": [276, 317, 341, 357, 374, 398, 422, 456, 489, 521, 562, 603],
        "material_eur_per_m": [7, 15, 21, 26, 30, 38, 47, 63, 78, 96, 121, 147],
        "dT_reference_K": None,  # not applicable -- LT networks are demand-led, not dT-power derived
    },
}

# ---------------------------------------------------------------------------
# Extension data for "110C" beyond DN150, from Tab 36 of the PREVIOUSLY
# uploaded TKWP file (Aug-2026 edition) -- same technology, larger range.
# Costs here are ALSO per single pipe (same "GL"-type annotation), so also
# get doubled. Only the DN200+ portion is used (DN20-150 already covered,
# more precisely, by Tab 33 of the Dec-2025 edition used above).
# ---------------------------------------------------------------------------

TKWP_EXTENSION_110C_AUG2026 = {
    "label": "Wärmenetze 'konventionell' 110°C (KMR) -- Aug-2026 edition, Tab 36",
    "source": "KWW-Technikkatalog-Waermeplanung_08-2026_Version_1_1, Tab 36",
    "DN_mm":                [200,  250,  300,  350,  400,  450,  500],
    "trench_unpaved_min":   [1236, 1506, 1769, 2028, 2283, 2534, 2782],
    "trench_unpaved_max":   [2161, 2603, 3031, 3447, 3853, 4250, 4641],
    "trench_paved_min":     [2048, 2509, 2962, 3409, 3849, 4285, 4716],
    "trench_paved_max":     [3989, 4800, 5584, 6345, 7089, 7816, 8530],
    "material_eur_per_m":   [636,  787,  937,  1086, 1233, 1380, 1526],
}


# ============================================================================
# 2) BUILDING (DN, cost) DATA-POINT ARRAYS FROM THE RAW TABLES
# ============================================================================

Terrain = Literal["unpaved", "paved"]
TerrainStat = Literal["min", "max", "mean"]


def _trench_cost_per_m(data: dict, terrain: Terrain, terrain_stat: TerrainStat) -> np.ndarray:
    """Pick the requested terrain/statistic from a network level's raw data."""
    if f"trench_{terrain}" in data:
        # Tab 35 style: single value per terrain, no min/max
        return np.array(data[f"trench_{terrain}"], dtype=float)

    lo = np.array(data[f"trench_{terrain}_min"], dtype=float)
    hi = np.array(data[f"trench_{terrain}_max"], dtype=float)
    if terrain_stat == "min":
        return lo
    elif terrain_stat == "max":
        return hi
    elif terrain_stat == "mean":
        return (lo + hi) / 2.0
    else:
        raise ValueError(f"Unknown terrain_stat: {terrain_stat!r}")


def get_pipe_cost_datapoints(
    network_level: Literal["110C", "70C", "LT"],
    terrain: Terrain = "unpaved",
    terrain_stat: TerrainStat = "mean",
    extend_beyond_catalog: bool = True,
    pair_factor: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Read the primary TKWP data points for one network level and return
    (DN_mm[], cost_eur_per_m[]) -- MAIN PIPE ONLY (material + trenching;
    building/house connection costs are deliberately excluded), already
    DOUBLED (pair_factor, default 2.0) to go from the tables' single-pipe
    prices to a full supply+return pipe pair, and optionally extended to
    larger diameters using a second data source (only implemented for
    network_level="110C" -- see module docstring).
    """
    if network_level not in TKWP_DATA:
        raise ValueError(f"Unknown network_level {network_level!r}; choose from {list(TKWP_DATA)}")

    data = TKWP_DATA[network_level]
    dn = np.array(data["DN_mm"], dtype=float)
    material = np.array(data["material_eur_per_m"], dtype=float)
    trench = _trench_cost_per_m(data, terrain, terrain_stat)

    cost_single_pipe = material + trench
    cost_pair = cost_single_pipe * pair_factor

    if extend_beyond_catalog and network_level == "110C":
        ext = TKWP_EXTENSION_110C_AUG2026
        dn_ext = np.array(ext["DN_mm"], dtype=float)
        material_ext = np.array(ext["material_eur_per_m"], dtype=float)
        trench_ext = _trench_cost_per_m(ext, terrain, terrain_stat)
        cost_ext_pair = (material_ext + trench_ext) * pair_factor

        # drop any DN already covered by the primary table to avoid duplicates
        mask = dn_ext > dn.max()
        dn = np.concatenate([dn, dn_ext[mask]])
        cost_pair = np.concatenate([cost_pair, cost_ext_pair[mask]])

    order = np.argsort(dn)
    return dn[order], cost_pair[order]


# ============================================================================
# 3) THE COST FUNCTION BUILDER (the core deliverable)
# ============================================================================

def build_cost_function(
    network_level: Literal["110C", "70C", "LT"],
    terrain: Terrain = "unpaved",
    terrain_stat: TerrainStat = "mean",
    extend_beyond_catalog: bool = True,
    beyond_extension_source: Literal["linear", "npro_scaled", "flat"] = "linear",
) -> Callable[[float], float]:
    """Build and return a CALLABLE cost_per_m(diameter_mm) -> EUR/m,
    constructed from the TKWP data points for the chosen network level and
    terrain. Interpolation uses PCHIP (monotonic piecewise cubic Hermite),
    which passes exactly through the given data points without the
    oscillation a high-degree polynomial fit can introduce, and stays
    monotonically increasing -- appropriate since pipe cost should never
    decrease with diameter.

    Diameters beyond the available data's own maximum (up to DN500 for
    "110C" via the Aug-2026 extension; the primary table's own max for
    "70C"/"LT") are handled by beyond_extension_source:
      - "linear" (default): extrapolate using the SLOPE of the last known
        data segment. Simple, and guarantees the curve stays continuous
        and monotonically increasing at the boundary -- the safest default.
      - "npro_scaled": use nPro's independently-sourced polynomial SHAPE
        (curvature/economies-of-scale behaviour) beyond the data, but
        RESCALED so it matches TKWP's value exactly at the boundary
        diameter. This matters because nPro's raw formula turned out to be
        roughly 4x LOWER than TKWP's boundary value in absolute terms (the
        two sources evidently assume different trenching complexity/scope)
        -- using it unscaled would make cost visibly DROP as diameter
        increases past the boundary, which is physically impossible and
        was caught by testing this function before shipping it. Scaling
        preserves nPro's curve shape while staying consistent with TKWP's
        cost level.
      - "flat": hold the last known value constant. Most conservative
        (never overstates cost) but least realistic for a genuinely much
        larger pipe.
    Any extrapolation beyond the data's real range should be treated as a
    rough placeholder, not a citable figure -- TKWP's own stated focus is
    explicitly on small/medium networks.

    Rebuildable by design: this function reads current values out of
    TKWP_DATA / TKWP_EXTENSION_110C_AUG2026 every time it's called, so
    editing those dictionaries (or replacing them wholesale with a newer
    catalog edition or a different data source) immediately changes what
    build_cost_function() returns, with no other code changes required.
    """
    dn, cost = get_pipe_cost_datapoints(
        network_level, terrain=terrain, terrain_stat=terrain_stat,
        extend_beyond_catalog=extend_beyond_catalog,
    )

    # prepend a (0mm, 0 EUR/m) point so the interpolation is well-defined
    # down to "no pipe" -- consistent with the D0 breakpoint used in this
    # project's SOS2 formulations
    dn_full = np.concatenate([[0.0], dn])
    cost_full = np.concatenate([[0.0], cost])

    pchip = PchipInterpolator(dn_full, cost_full, extrapolate=False)
    dn_max = dn_full.max()
    cost_at_max = float(cost_full[-1])

    # slope of the LAST data segment, for "linear" extrapolation
    slope_last_segment = (cost_full[-1] - cost_full[-2]) / (dn_full[-1] - dn_full[-2])

    def npro_raw(dn_mm: float) -> float:
        return (0.0033 * dn_mm**2 + 0.6843 * dn_mm + 54.033) * 2.0  # nPro is per-pipe too -> x2

    npro_scale = cost_at_max / npro_raw(dn_max) if network_level == "110C" else None

    def cost_per_m(diameter_mm: float) -> float:
        if diameter_mm <= dn_max:
            return float(pchip(diameter_mm))
        if beyond_extension_source == "flat":
            return cost_at_max
        if beyond_extension_source == "linear":
            return cost_at_max + slope_last_segment * (diameter_mm - dn_max)
        if beyond_extension_source == "npro_scaled":
            if npro_scale is None:
                raise ValueError('"npro_scaled" extrapolation is only defined for network_level="110C"')
            return npro_raw(diameter_mm) * npro_scale
        raise ValueError(f"Unknown beyond_extension_source: {beyond_extension_source!r}")

    cost_per_m.network_level = network_level          # type: ignore[attr-defined]
    cost_per_m.terrain = terrain                        # type: ignore[attr-defined]
    cost_per_m.terrain_stat = terrain_stat               # type: ignore[attr-defined]
    cost_per_m.dn_data_max_mm = dn_max                    # type: ignore[attr-defined]
    cost_per_m.beyond_extension_source = beyond_extension_source  # type: ignore[attr-defined]
    return cost_per_m


# ============================================================================
# 4) CAPACITY FUNCTION -- the piece that DOES depend on delta_T
# ============================================================================

RHO_WATER = 1000.0       # kg/m3
CP_WATER = 4185.0        # J/(kg K)
K_ROUGHNESS_MM = 0.1     # typical new steel/PMR DH pipe
MU_WATER_PAS = 4.0e-4    # dynamic viscosity, ~65 degC mean pipe temperature


def _nikuradse_friction_factor(reynolds: float, k_mm: float, d_m: float) -> float:
    re = max(reynolds, 1e-6)
    return 64.0 / re + 1.0 / (-2.0 * np.log10((k_mm / 1000.0) / (3.71 * d_m))) ** 2


def _max_velocity_pressure_limited(diameter_mm: float, r_max_pa_per_m: float,
                                    v_max_absolute: float) -> float:
    """Same pressure-gradient-limited hydraulic sizing used elsewhere in
    this project (see dh_pipe_capacity_pressure_limited.py): solve
    Darcy-Weisbach + Nikuradse iteratively for the velocity that produces
    exactly r_max_pa_per_m, capped at v_max_absolute.
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
    network_level: Literal["110C", "70C", "LT"],
    delta_t_k: float = 30.0,
    r_max_pa_per_m: float = 100.0,
    v_max_absolute_m_s: float = 3.0,
) -> Callable[[float], float]:
    """Build and return a callable capacity_mw(diameter_mm) -> MW, using
    the maximum-pressure-gradient hydraulic sizing method (see module
    docstring for why capacity, not cost, is where delta_T enters).
    Capacity is directly proportional to delta_t_k, matching
    Q = mdot * cp * dT with mdot set by the pressure-gradient-limited
    maximum velocity, independent of network_level for now (all three use
    the same hydraulic design criteria here) -- network_level is accepted
    for interface symmetry with build_cost_function() and to allow
    per-technology hydraulic limits to be introduced later without
    changing the calling code.
    """
    def capacity_mw(diameter_mm: float) -> float:
        if diameter_mm <= 0:
            return 0.0
        d_m = diameter_mm / 1000.0
        area = np.pi / 4.0 * d_m ** 2
        v = _max_velocity_pressure_limited(diameter_mm, r_max_pa_per_m, v_max_absolute_m_s)
        power_w = RHO_WATER * CP_WATER * v * area * delta_t_k
        return power_w / 1e6

    capacity_mw.network_level = network_level   # type: ignore[attr-defined]
    capacity_mw.delta_t_k = delta_t_k             # type: ignore[attr-defined]
    return capacity_mw


# ============================================================================
# 5) DEMO / VALIDATION
# ============================================================================

if __name__ == "__main__":
    print("=" * 78)
    print("TKWP pipe cost functions -- built from primary data points")
    print("=" * 78)

    for level in ["110C", "70C", "LT"]:
        dn, cost = get_pipe_cost_datapoints(level, terrain="unpaved", terrain_stat="mean")
        print(f"\n[{level}] {TKWP_DATA[level]['label']}")
        print(f"  data points (DN, EUR/m pair, unpaved-mean): "
              f"{list(zip(dn.astype(int), cost.round(1)))}")

    print("\n" + "=" * 78)
    print("Cost function evaluated at a few diameters (unpaved, mean terrain)")
    print("=" * 78)
    for level in ["110C", "70C", "LT"]:
        f = build_cost_function(level, terrain="unpaved", terrain_stat="mean")
        test_diams = [d for d in [30, 60, 100, 150, 250, 400, 600] if d <= 700]
        print(f"\n[{level}]  (data covers up to DN{f.dn_data_max_mm:.0f}, "
              f"beyond that = '{f.beyond_extension_source}' extrapolation)")
        for d in test_diams:
            try:
                print(f"  D={d:4d}mm -> {f(d):9.2f} EUR/m")
            except Exception as e:
                print(f"  D={d:4d}mm -> n/a ({e})")

    print("\n" + "=" * 78)
    print("Continuity check across the DN500 boundary (110C) -- and why the")
    print("default extrapolation method matters")
    print("=" * 78)
    for method in ["linear", "npro_scaled", "flat"]:
        f = build_cost_function("110C", beyond_extension_source=method)
        vals = [f(d) for d in [450, 500, 550, 600, 700]]
        monotonic = all(vals[i] <= vals[i+1] for i in range(len(vals)-1))
        print(f"  {method:12s}: D=450->{vals[0]:8.1f}  D=500->{vals[1]:8.1f}  "
              f"D=550->{vals[2]:8.1f}  D=600->{vals[3]:8.1f}  D=700->{vals[4]:8.1f}   "
              f"monotonic={'YES' if monotonic else 'NO -- BROKEN'}")

    print("\n" + "=" * 78)
    print("Capacity function: same diameter, different delta_T (network_level='110C')")
    print("=" * 78)
    for dT in [20, 30, 45, 60]:
        cap_fn = build_capacity_function("110C", delta_t_k=dT)
        print(f"  dT={dT:3.0f} K: D=100mm -> {cap_fn(100):.3f} MW,  D=300mm -> {cap_fn(300):.3f} MW")

    print("\n" + "=" * 78)
    print("Terrain sensitivity (110C, DN100, extend_beyond_catalog=True)")
    print("=" * 78)
    for terrain in ["unpaved", "paved"]:
        for stat in ["min", "max", "mean"]:
            f = build_cost_function("110C", terrain=terrain, terrain_stat=stat)
            print(f"  terrain={terrain:8s} stat={stat:5s} -> DN100 cost = {f(100):8.2f} EUR/m")

    print("\nDone. Edit TKWP_DATA / TKWP_EXTENSION_110C_AUG2026 above and re-run "
          "to rebuild every function with new data -- no other code changes needed.")
