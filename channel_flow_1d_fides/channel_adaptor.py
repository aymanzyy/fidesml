"""
channel_adaptor.py — Catalyst V2 adaptor for the channel flow solver.

IMPORTANT: set_external() stores a raw pointer — the caller must keep
the numpy array alive until after catalyst.execute() returns. All arrays
passed via set_external are stored in a local list 'refs' for the
duration of the call. Do NOT use set_external with temporaries like
arr.astype(...) directly — assign to a named variable first.
"""
import sys
import numpy as np
import catalyst_conduit as conduit
import catalyst
import time
_mode   = "inference"
_Re_tau = 180

cat_exec_timers = []
cat_results_timers = []

def initialize(mode="inference", Re_tau=180):
    global _mode, _Re_tau
    _mode, _Re_tau = mode, Re_tau

    node = conduit.Node()
    count = 0
    for arg in sys.argv[1:]:
        if arg.endswith(".py"):
            node[f"catalyst/scripts/script{count}"] = arg
            count += 1

    node["catalyst_load/implementation"]                    = "paraview"
    node["catalyst_params/catalyst/ml_coupling/mode"]       = mode
    node["catalyst_params/catalyst/ml_coupling/redis"]      = "localhost:6379"
    node["catalyst_params/catalyst/ml_coupling/channel"]    = "ml_closure"
    node["catalyst_params/catalyst/ml_coupling/Re_tau"]     = str(Re_tau)

    catalyst.initialize(node)
    print(f"[adaptor] initialized mode={mode} Re_tau={Re_tau}", flush=True)


def coprocess(step, time_, y, u, nu_t, du_dy):
    """Push channel state via Catalyst. Returns delta_nu_t or None.

    All arrays passed via set_external are kept in 'refs' to prevent
    garbage collection before catalyst.execute() reads them.
    """
    refs = []   # keep all set_external arrays alive until execute() returns

    def ext(arr):
        """Ensure contiguous float64, store reference, return it."""
        a = np.ascontiguousarray(arr, dtype=np.float64)
        refs.append(a)
        return a

    N = len(y)
    y64       = ext(y)
    u64       = ext(u)
    nu_t64    = ext(nu_t)
    du_dy64   = ext(du_dy)
    zeros     = ext(np.zeros(N))
    zeros_two     = ext(np.zeros(N))

    conn      = np.ascontiguousarray(
                    np.arange(N-1).repeat(2).reshape(-1,2) +
                    np.array([0,1]), dtype=np.int64)

    refs.append(conn)

    node = conduit.Node()
    node["catalyst/state/timestep"]      = step
    node["catalyst/state/time"]          = time_
    node["catalyst/state/pipelines/0"]   = "script0"
    node["catalyst/channels/grid/type"]  = "mesh"

    # Include ML coupling params every step.
    # info.catalyst_params during catalyst_execute returns the per-step node,
    # not the init node, so init params must be re-passed here.
    node["catalyst_params/catalyst/ml_coupling/mode"]    = _mode
    node["catalyst_params/catalyst/ml_coupling/redis"]   = "localhost:6379"
    node["catalyst_params/catalyst/ml_coupling/channel"] = "ml_closure"
    node["catalyst_params/catalyst/ml_coupling/Re_tau"]  = str(_Re_tau)

    mesh = node["catalyst/channels/grid/data"]

    mesh["coordsets/coordinates/type"]                = "explicit"
    mesh["coordsets/coordinates/values/x"].set_external(y64)
    mesh["coordsets/coordinates/values/y"].set_external(zeros)
    mesh["coordsets/coordinates/values/z"].set_external(zeros_two)
    mesh["topologies/mesh/type"]                 = "unstructured"
    mesh["topologies/mesh/coordset"]             = "coordinates"
    mesh["topologies/mesh/elements/shape"]       = "line"
    mesh["topologies/mesh/elements/connectivity"].set_external(conn.flatten())

    for name, arr64 in [("u_plus", u64), ("du_dy", du_dy64),
                        ("nu_t", nu_t64), ("y_plus", y64)]:
        mesh[f"fields/{name}/association"] = "vertex"
        mesh[f"fields/{name}/topology"]    = "mesh"
        mesh[f"fields/{name}/values"].set_external(arr64)

    cat_exec_begin = time.monotonic()

    try:
        catalyst.execute(node)

    except Exception as e:
        sys.stderr.write(f"[adaptor] catalyst.execute failed at step {step}: {e}\n")
        return None

    cat_exec_diff = time.monotonic() - cat_exec_begin
    cat_exec_timers.append(cat_exec_diff)
    # Read correction back via the proper Catalyst V2 results protocol.
    # catalyst_results(info) in the script writes into info.catalyst_params;
    # catalyst.results(result_node) retrieves it here.
    cat_res_begin = time.monotonic()
    result_node = conduit.Node()
    catalyst.results(result_node)
    cat_res_diff = time.monotonic() - cat_res_begin
    cat_results_timers.append(cat_res_diff)
    if result_node.has_path("delta_nu_t"):
        return np.array(result_node["delta_nu_t"])
    return 1, cat_exec_timers, cat_results_timers


def finalize():
    catalyst.finalize(conduit.Node())
    print("[adaptor] finalized", flush=True)
