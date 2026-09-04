"""
Minimal PyPSA test model: a single district-heating pipe (Link) whose
diameter/capacity/cost is chosen from a discrete set {DN50, DN100, DN200}
via a piecewise-linear (SOS2) interpolation, plus a binary "is the pipe
built at all" decision that carries a fixed (length-based) investment cost.

Physics
-------
Thermal capacity of a pipe for a given nominal diameter is estimated from

    P = rho * cp * v * A * dT

with
    rho : density of water                      [kg/m3]
    cp  : specific heat capacity of water        [J/(kg K)]
    v   : design flow velocity                   [m/s]
    A   : cross-sectional (bore) area            [m2]
    dT  : temperature spread forward/return flow [K]

We use a single design velocity (typical DH design value) for all
diameters -- this is obviously a simplification (in reality allowed
velocity grows with diameter to limit noise/erosion), but it is a
reasonable starting point for a *test* model and can be refined later
without touching the SOS2 / MILP machinery below.

Losses are neglected in this first version (efficiency = 1).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
import pypsa


# --------------------------------------------------------------------------
# 1) Physical / techno-economic input data for the discrete diameter choices
# --------------------------------------------------------------------------

RHO = 971.0          # kg/m3, density of water at ~ 80 degC
CP = 4185.0          # J/(kg K), specific heat capacity of water at ~ 80 degC
V_DESIGN = 1.5        # m/s, assumed design flow velocity (typical DH value)
DELTA_T = 30.0        # K, temperature spread forward/return flow
PIPE_LENGTH = 100.0   # m

# Nominal diameters and their (approximate) bore diameter in metres.
# DN roughly equals the bore diameter in mm for the steel pipes used here.
DIAMETERS_M = {
    "DN50": 0.050,
    "DN100": 0.100,
    "DN200": 0.200,
}

# Cost per metre of pipe, as given in the task [EUR/m]
COST_PER_M = {
    "DN50": 300.0,
    "DN100": 450.0,
    "DN200": 700.0,
}

FIXED_COST_PER_M = 400.0  # EUR/m, paid once if the pipe is built at all,
                           # independent of which diameter is chosen


def thermal_capacity_mw(diameter_m: float) -> float:
    """Steady-state thermal transport capacity of a pipe [MW]."""
    area = np.pi / 4.0 * diameter_m**2
    volumetric_flow = V_DESIGN * area          # m3/s
    power_w = RHO * CP * volumetric_flow * DELTA_T  # W
    return power_w / 1e6                       # MW

DIAM_NAMES = list(DIAMETERS_M.keys())
CAPACITY_MW = {d: thermal_capacity_mw(DIAMETERS_M[d]) for d in DIAM_NAMES}
VAR_COST_EUR = {d: COST_PER_M[d] * PIPE_LENGTH for d in DIAM_NAMES}
FIXED_COST_EUR = FIXED_COST_PER_M * PIPE_LENGTH

print("Diameter breakpoints used for the SOS2 interpolation:")
for d in DIAM_NAMES:
    print(
        f"  {d:>6s}: capacity = {CAPACITY_MW[d]:6.3f} MW_th,"
        f"  variable cost = {VAR_COST_EUR[d]:8.0f} EUR"
    )
print(f"  fixed cost if built (any diameter): {FIXED_COST_EUR:8.0f} EUR")


# --------------------------------------------------------------------------
# 2) Build the minimal PyPSA network
# --------------------------------------------------------------------------

def build_network(demand_mw: float, snapshots: pd.Index | None = None) -> pypsa.Network:
    """Build the minimal heat-source -> pipe -> heat-demand network.

    Parameters
    ----------
    demand_mw : float
        Constant heat demand at the far end of the pipe [MW].
    snapshots : optional index of snapshots; defaults to a single snapshot.
    """
    n = pypsa.Network()
    if snapshots is None:
        snapshots = pd.Index(["now"], name="snapshot")
    n.set_snapshots(snapshots)

    n.add("Carrier", "heat")

    n.add("Bus", "heat_source", carrier="heat")
    n.add("Bus", "heat_demand", carrier="heat")

    # "infinite" heat source: very large capacity, (near) zero marginal cost.
    # A tiny non-zero marginal cost (1e-6 EUR/MWh) is used only so that
    # pypsa's own objective assembly has at least one non-zero cost term
    # to work with (it errors out if literally everything is free) -- it
    # is negligible compared to the pipe investment costs and does not
    # influence the diameter choice.
    n.add(
        "Generator",
        "source",
        bus="heat_source",
        carrier="heat",
        p_nom=1e6,          # MW, effectively unlimited for this test model
        marginal_cost=1e-6,
    )

    # simple heat demand as an inflexible load
    n.add(
        "Load",
        "demand",
        bus="heat_demand",
        carrier="heat",
        p_set=demand_mw,
    )

    # the district-heating pipe itself, modelled as a Link.
    # p_nom is extendable, but its *investment cost* is NOT the standard
    # linear pypsa capital_cost -- it is replaced below by the custom
    # SOS2 / binary cost terms, so capital_cost is set to 0 here.
    n.add(
        "Link",
        "pipe",
        bus0="heat_source",
        bus1="heat_demand",
        carrier="heat",
        p_nom_extendable=True,
        p_min_pu=0.0,
        efficiency=1.0,      # no heat losses in this first version
        capital_cost=0.0,
        marginal_cost=0.0,
    )

    return n


# --------------------------------------------------------------------------
# 3) Add the SOS2 diameter-selection + binary built-decision to the model
# --------------------------------------------------------------------------

def add_pipe_investment_formulation(n: pypsa.Network) -> None:
    """Augment the linopy model created by ``n.optimize.create_model()``
    with:

      * continuous SOS2 weights ``lambda[diam]`` in [0, 1] per diameter,
        interpolating capacity and variable cost between neighbouring
        diameter breakpoints,
      * a binary ``built`` variable that
          - forces sum(lambda) == built  (so lambda are all 0 if the pipe
            is not built, and form a standard convex SOS2 combination
            summing to 1 if it is built),
          - carries the fixed investment cost ``FIXED_COST_EUR`` in the
            objective,
      * a constraint tying the Link's ``p_nom`` decision variable to the
        SOS2-interpolated capacity, and
      * the combined (fixed + interpolated variable) investment cost term
        added to the objective.

    Must be called *after* ``n.optimize.create_model()`` and before
    ``n.optimize.solve_model()``.
    """
    m = n.model

    capacity = xr.DataArray(
        [CAPACITY_MW[d] for d in DIAM_NAMES], coords=[DIAM_NAMES], dims=["diam"]
    )
    var_cost = xr.DataArray(
        [VAR_COST_EUR[d] for d in DIAM_NAMES], coords=[DIAM_NAMES], dims=["diam"]
    )

    # SOS2 interpolation weights, one set per pipe (here: only "pipe")
    lam = m.add_variables(
        lower=0, upper=1, coords=[DIAM_NAMES], dims=["diam"], name="pipe-lambda"
    )
    # binary "is the pipe built at all" decision
    built = m.add_variables(binary=True, name="pipe-built")

    # weights sum to 1 if built, to 0 if not built
    m.add_constraints(lam.sum("diam") == built, name="pipe-lambda-sum-built")

    # SOS2: at most two *adjacent* (in the declared order DN50-DN100-DN200)
    # weights may be non-zero -> linear interpolation between neighbouring
    # diameters only, never e.g. a mix of DN50 and DN200 directly.
    m.add_sos_constraints(lam, sos_type=2, sos_dim="diam")

    # interpolated capacity from the SOS2 weights
    interpolated_capacity = (lam * capacity).sum("diam")

    # tie the Link's p_nom (used by pypsa's flow/capacity constraints,
    # p0 <= p_nom etc.) to the interpolated discrete-diameter capacity
    link_p_nom = m.variables["Link-p_nom"].loc["pipe"]
    m.add_constraints(
        link_p_nom == interpolated_capacity, name="pipe-p_nom-from-sos2"
    )

    # combined investment cost: fixed cost if built at all, plus the
    # SOS2-interpolated variable (diameter-dependent) cost
    investment_cost = built * FIXED_COST_EUR + (lam * var_cost).sum("diam")

    # add to the objective (pypsa's own objective already contains
    # marginal costs etc.; the Link's capital_cost is 0, so there is no
    # double counting of the investment cost)
    m.objective += investment_cost


# --------------------------------------------------------------------------
# 4) Convenience: solve one scenario and report the result
# --------------------------------------------------------------------------

def solve_scenario(demand_mw: float, solver_name: str = "highs", verbose: bool = False) -> dict:
    n = build_network(demand_mw)
    n.optimize.create_model()
    add_pipe_investment_formulation(n)
    solver_options = {} if verbose else {"output_flag": False}
    status, condition = n.optimize.solve_model(
        solver_name=solver_name,
        reformulate_sos=True,
        solver_options=solver_options,
    )

    result = {
        "demand_mw": demand_mw,
        "status": status,
        "condition": condition,
    }

    if status != "ok":
        result["feasible"] = False
        return result

    m = n.model
    lam_sol = m.variables["pipe-lambda"].solution.to_pandas()
    built_sol = float(m.variables["pipe-built"].solution.item())
    p_nom_sol = float(n.links.at["pipe", "p_nom_opt"])
    p0_sol = float(n.links_t.p0.at["now", "pipe"])
    total_obj = float(m.objective.value)

    result.update(
        {
            "feasible": True,
            "built": built_sol,
            "lambda": lam_sol.to_dict(),
            "link_p_nom_opt_MW": p_nom_sol,
            "link_flow_MW": p0_sol,
            "objective_EUR": total_obj,
        }
    )
    return result


def pretty_print(result: dict) -> None:
    print(f"\n--- demand = {result['demand_mw']:.3f} MW ---")
    print(f"status/condition: {result['status']} / {result['condition']}")
    if not result.get("feasible", False):
        print("  infeasible (demand cannot be met by any available diameter)")
        return
    print(f"  built                 : {result['built']:.0f}")
    lam_str = ", ".join(f"{d}={v:6.3f}" for d, v in result["lambda"].items())
    print(f"  SOS2 weights (lambda) : {lam_str}")
    print(f"  Link p_nom_opt        : {result['link_p_nom_opt_MW']:.4f} MW")
    print(f"  Link flow (p0)        : {result['link_flow_MW']:.4f} MW")
    print(f"  total objective       : {result['objective_EUR']:.2f} EUR")


if __name__ == "__main__":
    # Single base-case scenario
    base_result = solve_scenario(demand_mw=1.0)
    pretty_print(base_result)
