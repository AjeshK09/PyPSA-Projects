import pypsa
import numpy as np
import pandas as pd
import xarray as xr


# ============================================================
# 1. PARAMETERS
# ============================================================

L = 100.0                  # pipe length [m]
delta_T = 30.0             # temperature difference [K]

rho = 1000.0               # kg/m3
cp = 4180.0                # J/(kg K)
velocity = 1.0              # m/s

demand = 0.50              # MW

# Diameter breakpoints
diameters = np.array([
    0.05,
    0.10,
    0.20
])

diameter_names = [
    "DN50",
    "DN100",
    "DN200"
]

# Investment cost [EUR/m]
cost_per_m = np.array([
    300.0,
    450.0,
    700.0
])

# Fixed cost [EUR/m]
fixed_cost_per_m = 400.0


# ============================================================
# 2. CALCULATE CAPACITY
# ============================================================

area = np.pi * diameters**2 / 4

capacity_MW = (
    rho
    * cp
    * delta_T
    * velocity
    * area
    / 1e6
)

print("\nPipe breakpoint data")

for name, d, q, c in zip(
    diameter_names,
    diameters,
    capacity_MW,
    cost_per_m
):
    print(
        f"{name}: "
        f"D={d:.3f} m, "
        f"capacity={q:.4f} MW, "
        f"cost={c:.0f} EUR/m"
    )


# ============================================================
# 3. CREATE NETWORK
# ============================================================

n = pypsa.Network()

n.set_snapshots(["now"])

n.add(
    "Carrier",
    "heat"
)

n.add(
    "Bus",
    "source",
    carrier="heat"
)

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
    p_nom=1000,
    marginal_cost=0
)

# Heat demand
n.add(
    "Load",
    "heat_demand",
    bus="demand",
    carrier="heat",
    p_set=demand
)

# Pipe
n.add(
    "Link",
    "pipe",
    bus0="source",
    bus1="demand",
    carrier="heat",
    efficiency=1.0,
    p_nom_extendable=True,
    p_nom_max=capacity_MW[-1],

    # PyPSA needs a capital_cost so that its
    # objective is not empty.
    capital_cost=0.0
)


# ============================================================
# 4. CREATE PYPSA OPTIMIZATION MODEL
# ============================================================

m = n.optimize.create_model()

print("\nPyPSA model created.")


# ============================================================
# 5. PIPE CAPACITY VARIABLE
# ============================================================

pipe_capacity = m.variables["Link-p_nom"].sel(
    name="pipe"
)


# ============================================================
# 6. BINARY VARIABLE
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


# ============================================================
# 8. SOS2 CONSTRAINT
# ============================================================

m.add_sos_constraints(
    lam,
    sos_type=2,
    sos_dim="breakpoint"
)


# ============================================================
# 9. LINK BINARY TO SOS2
# ============================================================

m.add_constraints(
    lam.sum() == pipe_built,
    name="pipe_build"
)


# ============================================================
# 10. CAPACITY INTERPOLATION
# ============================================================

capacity_da = xr.DataArray(
    capacity_MW,
    coords={
        "breakpoint": bp
    }
)

m.add_constraints(
    pipe_capacity
    == (lam * capacity_da).sum(),
    name="pipe_capacity_sos2"
)


# ============================================================
# 11. COST
# ============================================================

cost_da = xr.DataArray(
    cost_per_m * L,
    coords={
        "breakpoint": bp
    }
)

variable_cost = (
    lam * cost_da
).sum()

fixed_cost = (
    pipe_built
    * fixed_cost_per_m
    * L
)

pipe_cost = (
    variable_cost
    + fixed_cost
)


# ============================================================
# 12. ADD COST TO EXISTING OBJECTIVE
# ============================================================

# IMPORTANT:
# We do NOT call m.add_objective().
#
# PyPSA already created the objective when
# create_model() was called.
#
# We modify it here.

m.objective += pipe_cost


# ============================================================
# 13. SOLVE
# ============================================================

print("\nSolving...")

status, condition = n.optimize.solve_model(
    solver_name="highs",
    reformulate_sos=True,
    log_to_console=True
)

print("\nStatus:", status)
print("Condition:", condition)


# ============================================================
# 14. RESULTS
# ============================================================

if condition != "optimal":
    raise RuntimeError(
        f"Optimization failed: {condition}"
    )


built = float(
    pipe_built.solution
)

pipe_capacity_result = float(
    pipe_capacity.solution
)

lambda_result = (
    lam.solution
    .to_pandas()
)

variable_result = float(
    variable_cost.solution
)

fixed_result = float(
    fixed_cost.solution
)

total_pipe_cost = (
    variable_result
    + fixed_result
)

dispatch = float(
    n.links_t.p1.loc[
        "now",
        "pipe"
    ]
)


# ============================================================
# 15. PRINT RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("PIPE OPTIMIZATION RESULT")
print("=" * 60)

print(
    f"Demand             = {demand:.4f} MW"
)

print(
    f"Pipe built         = {built:.0f}"
)

print(
    f"Pipe capacity      = "
    f"{pipe_capacity_result:.4f} MW"
)

print("\nSOS2 weights:")

for name, value in zip(
    diameter_names,
    lambda_result.values
):
    print(
        f"  {name:5s} = {value:.6f}"
    )

print("\nCosts:")

print(
    f"Fixed cost         = "
    f"{fixed_result:.2f} EUR"
)

print(
    f"Variable cost      = "
    f"{variable_result:.2f} EUR"
)

print(
    f"Total pipe cost    = "
    f"{total_pipe_cost:.2f} EUR"
)

print("\nFlow:")

print(
    f"Pipe flow          = "
    f"{dispatch:.4f} MW"
)


# ============================================================
# 16. VALIDATION
# ============================================================

print("\n")
print("=" * 60)
print("VALIDATION")
print("=" * 60)


# Capacity must cover demand
assert (
    pipe_capacity_result
    >= demand - 1e-6
)

print("✓ Capacity covers demand")


# Pipe should be built
assert built > 0.5

print("✓ Pipe is built")


# Flow must equal demand
assert (
    abs(
        abs(dispatch) - demand
    ) < 1e-6
)

print("✓ Pipe flow equals demand")


# SOS2 check
active = np.where(
    lambda_result.values > 1e-6
)[0]

print(
    "Active breakpoints:",
    active.tolist()
)

assert len(active) <= 2

if len(active) == 2:
    assert (
        active[1]
        == active[0] + 1
    )

print("✓ SOS2 uses neighboring breakpoints")

print("\nALL VALIDATIONS PASSED")
