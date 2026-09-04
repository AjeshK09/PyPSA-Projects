import numpy as np
import pandas as pd
import xarray as xr
import pypsa


# ============================================================
# 1. PARAMETERS
# ============================================================

L = 100.0                 # pipe length [m]
DELTA_T = 30.0            # temperature difference [K]

RHO = 1000.0              # water density [kg/m3]
CP = 4180.0               # specific heat [J/(kg K)]
VELOCITY = 1.0            # assumed water velocity [m/s]

# Diameter breakpoints
diameters = np.array([
    0.05,   # DN50
    0.10,   # DN100
    0.20    # DN200
])

diameter_names = [
    "DN50",
    "DN100",
    "DN200"
]

# Given investment cost [EUR/m]
cost_per_m = np.array([
    300.0,
    450.0,
    700.0
])

# Fixed cost [EUR/m]
FIXED_COST_PER_M = 400.0

# Test heat demand [MW]
DEMAND = 0.50


# ============================================================
# 2. CALCULATE CAPACITY FOR EACH DIAMETER
# ============================================================

area = np.pi * diameters**2 / 4.0

capacity_MW = (
    RHO
    * CP
    * DELTA_T
    * VELOCITY
    * area
    / 1e6
)

print("\nDiameter data")
print("-" * 60)

for name, d, q, c in zip(
    diameter_names,
    diameters,
    capacity_MW,
    cost_per_m
):
    print(
        f"{name:5s} | "
        f"D = {d:.3f} m | "
        f"Capacity = {q:.4f} MW | "
        f"Cost = {c:.0f} EUR/m"
    )


# ============================================================
# 3. CREATE PYPSA NETWORK
# ============================================================

n = pypsa.Network()

n.set_snapshots(["now"])

# Carrier
n.add(
    "Carrier",
    "heat"
)

# Source bus
n.add(
    "Bus",
    "source",
    carrier="heat"
)

# Demand bus
n.add(
    "Bus",
    "demand",
    carrier="heat"
)

# Infinite heat source
n.add(
    "Generator",
    "heat_source",
    bus="source",
    carrier="heat",
    p_nom=1000.0,
    marginal_cost=0.0
)

# Heat demand
n.add(
    "Load",
    "heat_demand",
    bus="demand",
    carrier="heat",
    p_set=DEMAND
)

# Pipe
n.add(
    "Link",
    "pipe",
    bus0="source",
    bus1="demand",
    carrier="heat",
    efficiency=1.0,

    # We will determine p_nom ourselves
    p_nom_extendable=True,

    # Maximum possible capacity = DN200
    p_nom_max=capacity_MW[-1],

    # Cost is handled manually below
    capital_cost=0.0
)


# ============================================================
# 4. CREATE PYPSA / LINOPY MODEL
# ============================================================

m = n.optimize.create_model()

print("\nPyPSA model created successfully.")


# ============================================================
# 5. GET PIPE CAPACITY VARIABLE
# ============================================================

pipe_p_nom = m.variables["Link-p_nom"].sel(
    name="pipe"
)


# ============================================================
# 6. BINARY VARIABLE: IS PIPE BUILT?
# ============================================================

pipe_built = m.add_variables(
    lower=0,
    upper=1,
    binary=True,
    name="pipe_built"
)


# ============================================================
# 7. SOS2 LAMBDA VARIABLES
# ============================================================

# IMPORTANT:
# The SOS dimension must have numeric/ordered coordinates.

bp = pd.Index(
    [0, 1, 2],
    name="breakpoint"
)

lam = m.add_variables(
    lower=0,
    upper=1,
    coords=[bp],
    name="pipe_lambda"
)

print("SOS2 lambda variables created.")


# ============================================================
# 8. ADD SOS2 CONSTRAINT
# ============================================================

m.add_sos_constraints(
    lam,
    sos_type=2,
    sos_dim="breakpoint"
)

print("SOS2 constraint added.")


# ============================================================
# 9. LAMBDA SUM = BUILD BINARY
# ============================================================

m.add_constraints(
    lam.sum() == pipe_built,
    name="pipe_build_lambda"
)


# ============================================================
# 10. CAPACITY INTERPOLATION
# ============================================================

capacity_values = xr.DataArray(
    capacity_MW,
    coords={"breakpoint": bp}
)

m.add_constraints(
    pipe_p_nom
    == (lam * capacity_values).sum(),
    name="pipe_capacity_interpolation"
)


# ============================================================
# 11. VARIABLE COST INTERPOLATION
# ============================================================

variable_cost_values = xr.DataArray(
    cost_per_m * L,
    coords={"breakpoint": bp}
)

variable_cost = (
    lam * variable_cost_values
).sum()


# ============================================================
# 12. FIXED COST
# ============================================================

fixed_cost = (
    pipe_built
    * FIXED_COST_PER_M
    * L
)


# ============================================================
# 13. TOTAL INVESTMENT COST
# ============================================================

total_cost = (
    fixed_cost
    + variable_cost
)


# ============================================================
# 14. REPLACE OBJECTIVE
# ============================================================

m.add_objective(
    total_cost,
    overwrite=True
)


# ============================================================
# 15. SOLVE
# ============================================================

print("\nSolving...")

status, condition = n.optimize.solve_model(
    solver_name="highs",
    log_to_console=True,
    reformulate_sos=True
)

print("\nSolver status:", status)
print("Solver condition:", condition)


# ============================================================
# 16. READ RESULTS
# ============================================================

if condition != "optimal":
    raise RuntimeError(
        f"Optimization failed: {condition}"
    )


built = float(pipe_built.solution)

capacity = float(
    pipe_p_nom.solution
)

lambdas = lam.solution.to_pandas()

fixed = float(
    fixed_cost.solution
)

variable = float(
    variable_cost.solution
)

total = float(
    total_cost.solution
)


# ============================================================
# 17. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("RESULTS")
print("=" * 60)

print(f"Demand              : {DEMAND:.4f} MW")
print(f"Pipe built          : {built:.0f}")
print(f"Pipe capacity       : {capacity:.4f} MW")

print("\nSOS2 weights:")

for name, value in zip(
    diameter_names,
    lambdas.values
):
    print(
        f"  {name:5s} = {value:.6f}"
    )

print("\nCosts:")
print(f"Fixed cost          : {fixed:.2f} EUR")
print(f"Variable cost       : {variable:.2f} EUR")
print(f"Total investment    : {total:.2f} EUR")


# ============================================================
# 18. CHECK LINK DISPATCH
# ============================================================

dispatch = float(
    n.links_t.p1.loc[
        "now",
        "pipe"
    ]
)

print("\nLink dispatch:")
print(f"Pipe p1             : {dispatch:.4f} MW")


# ============================================================
# 19. VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("VALIDATION")
print("=" * 60)

# ------------------------------------------------------------
# Check 1: Pipe is built
# ------------------------------------------------------------

assert built > 0.5

print("PASS: pipe is built")


# ------------------------------------------------------------
# Check 2: Capacity >= demand
# ------------------------------------------------------------

assert capacity >= DEMAND - 1e-6

print("PASS: pipe capacity >= demand")


# ------------------------------------------------------------
# Check 3: Dispatch equals demand
# ------------------------------------------------------------

assert abs(
    abs(dispatch) - DEMAND
) < 1e-6

print("PASS: pipe dispatch = demand")


# ------------------------------------------------------------
# Check 4: SOS2
# ------------------------------------------------------------

active = np.where(
    lambdas.values > 1e-6
)[0]

print(
    f"Active SOS2 breakpoints: {active.tolist()}"
)

assert len(active) <= 2

if len(active) == 2:

    assert (
        active[1]
        == active[0] + 1
    )

print("PASS: SOS2 neighboring condition")


# ------------------------------------------------------------
# Final
# ------------------------------------------------------------

print("\nALL TESTS PASSED")
