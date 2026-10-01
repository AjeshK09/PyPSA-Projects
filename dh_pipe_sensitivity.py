
from __future__ import annotations

import pandas as pd

from dh_pipe_model import (
    DIAM_NAMES,
    CAPACITY_MW,
    solve_scenario,
)

TOL = 1e-6


def check_sos2_adjacency(lam: dict) -> tuple[bool, str]:
    """Return (ok, description) for the SOS2 adjacency property."""
    nonzero = [d for d, v in lam.items() if v > 1e-6]
    if len(nonzero) <= 1:
        return True, f"single breakpoint active: {nonzero}"
    if len(nonzero) > 2:
        return False, f"MORE THAN 2 breakpoints active: {nonzero}"
    i0 = DIAM_NAMES.index(nonzero[0])
    i1 = DIAM_NAMES.index(nonzero[1])
    if abs(i0 - i1) == 1:
        return True, f"adjacent breakpoints active: {nonzero}"
    return False, f"NON-ADJACENT breakpoints active: {nonzero}"


def run_sensitivity(demands: list[float]) -> pd.DataFrame:
    rows = []
    for d in demands:
        res = solve_scenario(d)
        row = {"demand_MW": d,
               "status": res["status"], "condition": res["condition"]}
        if res.get("feasible", False):
            ok_adj, desc_adj = check_sos2_adjacency(res["lambda"])
            cap = res["link_p_nom_opt_MW"]
            flow = res["link_flow_MW"]
            row.update(
                {
                    "feasible": True,
                    "built": res["built"],
                    "lambda_DN50": res["lambda"]["DN50"],
                    "lambda_DN100": res["lambda"]["DN100"],
                    "lambda_DN200": res["lambda"]["DN200"],
                    "sos2_adjacency_ok": ok_adj,
                    "sos2_note": desc_adj,
                    "link_p_nom_opt_MW": cap,
                    "link_flow_MW": flow,
                    "capacity_covers_demand": cap >= d - TOL,
                    "flow_matches_demand": abs(flow - d) < 1e-4,
                    "objective_EUR": res["objective_EUR"],
                }
            )
        else:
            row.update({"feasible": False})
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    cap50, cap100, cap200 = (CAPACITY_MW[d] for d in DIAM_NAMES)

    demands = [
        0.0,                    # no demand at all -> pipe should not be built
        0.1,                    # below DN50 capacity
        cap50,                  # exactly at DN50 breakpoint
        (cap50 + cap100) / 2,   # strictly between DN50 and DN100
        cap100,                 # exactly at DN100 breakpoint
        (cap100 + cap200) / 2,  # strictly between DN100 and DN200
        cap200,                 # exactly at DN200 breakpoint
        cap200 + 1.0,           # above DN200 capacity -> infeasible
    ]

    df = run_sensitivity(demands)

    pd.set_option("display.width", 160)
    pd.set_option("display.max_columns", 20)
    pd.set_option("display.float_format", lambda x: f"{x:9.4f}")

    print("\n================ SENSITIVITY OVER DEMAND ================\n")
    cols_main = [
        "demand_MW", "feasible", "built",
        "lambda_DN50", "lambda_DN100", "lambda_DN200",
        "link_p_nom_opt_MW", "link_flow_MW", "objective_EUR",
    ]
    print(df[cols_main].to_string(index=False))

    print("\n================ VALIDATION CHECKS =======================\n")
    all_ok = True
    for _, r in df.iterrows():
        if not r["feasible"]:
            note = "infeasible as expected" if r["demand_MW"] > cap200 else "UNEXPECTED infeasibility"
            ok = r["demand_MW"] > cap200
            print(f"demand={r['demand_MW']:7.3f} MW : {note}")
            all_ok &= ok
            continue

        adj_ok = bool(r["sos2_adjacency_ok"])
        cov_ok = bool(r["capacity_covers_demand"])
        flow_ok = bool(r["flow_matches_demand"])
        built_expected = 0.0 if r["demand_MW"] == 0.0 else 1.0
        built_ok = abs(r["built"] - built_expected) < 1e-6

        status = "OK" if (
            adj_ok and cov_ok and flow_ok and built_ok) else "FAIL"
        print(
            f"demand={r['demand_MW']:7.3f} MW : {status:4s} | "
            f"SOS2 adjacency: {r['sos2_note']:38s} | "
            f"capacity>=demand: {cov_ok} | flow==demand: {flow_ok} | "
            f"built as expected: {built_ok}"
        )
        all_ok &= adj_ok and cov_ok and flow_ok and built_ok

    print("\nALL CHECKS PASSED" if all_ok else "\nSOME CHECKS FAILED")
