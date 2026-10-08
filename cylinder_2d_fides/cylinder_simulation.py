#!/usr/bin/env python3
"""
cylinder_simulation.py - Coarse cylinder solver with catalyst-ml + async Catalyst.

Architecture:
  1. Coarse LBM solver runs on GPU
  2. Pushes state to Redis via catalyst-ml transport (BLOCKING - needs result)
  3. Inference service returns FNO-corrected fields via Redis
  4. Both coarse and ROM fields pushed to Catalyst (ASYNC - doesn't block solver)
  5. Catalyst renders PNG images on a worker thread
"""
import sys, os
# CATALYST_IMPLEMENTATION_PATHS should be set in the environment

from mpi4py import MPI
import numpy as np
import torch
import time

from cylinder_solver import CylinderSolver
from catalyst_ml.transport import RedisTransport

# Catalyst (optional)
HAS_CATALYST = False
try:
    import catalyst_conduit as conduit
    import catalyst
    HAS_CATALYST = True
except ImportError:
    pass

def build_catalyst_node_alt(exec_node, cycle, t, nx, ny, fields_dict):
    """Build Conduit mesh (uniform quad)."""
    refs = []
    def ext(arr, dtype=np.float64):
        a = np.ascontiguousarray(arr, dtype=dtype)
        refs.append(a)
        return a

    exec_node['catalyst/state/timestep'] = cycle
    exec_node['catalyst/state/time'] = float(t)
    exec_node['catalyst/channels/grid/type'] = 'mesh'
    mesh = exec_node['catalyst/channels/grid/data']


    mesh['coordsets/coords/type'] = 'uniform'
    mesh['coordsets/coords/dims/i'].set(nx)
    mesh['coordsets/coords/dims/j'].set(ny)
    mesh['coordsets/coords/dims/k'].set(1)

    mesh['coordsets/coords/origin/x'].set(0.0)
    mesh['coordsets/coords/origin/y'].set(0.0)
    mesh['coordsets/coords/origin/z'].set(0.0)

    mesh['coordsets/coords/spacing/dx'].set(1.0)
    mesh['coordsets/coords/spacing/dy'].set(1.0)
    mesh['coordsets/coords/spacing/dz'].set(1.0)

    mesh["topologies/mesh/type"].set("uniform");
    mesh["topologies/mesh/coordset"].set("coords");

    for name, arr in fields_dict.items():
        mesh[f'fields/{name}/association'] = 'vertex'
        mesh[f'fields/{name}/topology'] = 'mesh'
        mesh[f'fields/{name}/values'].set_external(ext(arr.flatten()))

    return refs

def build_catalyst_node(exec_node, cycle, t, nx, ny, fields_dict, ux, uy):
    """Build Conduit mesh (explicit unstructured quad)."""
    refs = []
    def ext(arr, dtype=np.float64):
        a = np.ascontiguousarray(arr, dtype=dtype)
        refs.append(a)
        return a

    exec_node['catalyst/state/timestep'] = cycle
    exec_node['catalyst/state/time'] = float(t)
    exec_node['catalyst/channels/grid/type'] = 'mesh'
    mesh = exec_node['catalyst/channels/grid/data']

    x = np.linspace(0, nx - 1, nx).astype("float64")
    y = np.linspace(0, ny - 1, ny).astype("float64")
    X, Y = np.meshgrid(x, y, indexing='ij')

    mesh['coordsets/coordinates/type'] = 'explicit'
    mesh['coordsets/coordinates/values/x'].set_external(ext(X.flatten()))
    mesh['coordsets/coordinates/values/y'].set_external(ext(Y.flatten()))
    mesh['coordsets/coordinates/values/z'].set_external(ext(np.zeros(nx * ny)))

    i_idx, j_idx = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij')
    p0 = i_idx * ny + j_idx
    p1 = (i_idx + 1) * ny + j_idx
    p2 = (i_idx + 1) * ny + (j_idx + 1)
    p3 = i_idx * ny + (j_idx + 1)
    conn = ext(np.stack([p0, p1, p2, p3], axis=-1).flatten(), dtype=np.int64)

    mesh['topologies/mesh/type'] = 'unstructured'
    mesh['topologies/mesh/coordset'] = 'coordinates'
    mesh['topologies/mesh/elements/shape'] = 'quad'
    mesh['topologies/mesh/elements/connectivity'].set_external(conn)

    for name, arr in fields_dict.items():
        mesh[f'fields/{name}/association'] = 'vertex'
        mesh[f'fields/{name}/topology'] = 'mesh'
        mesh[f'fields/{name}/values'].set_external(ext(arr.flatten()))
    ## Need to pass ux, uy. Going to do the same thing we do for X coords
    ## Expand from 150 to 7500, slice the 150 

    #ux_x, Y = np.meshgrid(ux, y, indexing='ij')
    #uy_x, Y = np.meshgrid(uy, y, indexing='ij')

    mesh['fields/ux/association'] = 'vertex'
    mesh['fields/ux/topology'] = 'mesh'
    mesh['fields/ux/values'].set_external(ext(ux.flatten()))

    mesh['fields/uy/association'] = 'vertex'
    mesh['fields/uy/topology'] = 'mesh'
    mesh['fields/uy/values'].set_external(ext(uy.flatten()))

    return refs


def main():
    global HAS_CATALYST
    print('[SIM] === Cylinder: catalyst-ml (blocking) ===', flush=True)

    comm = MPI.COMM_WORLD

    # Get total number of processes (world size)
    world_size = comm.Get_size()

    # Get the rank of the current process
    mpi_rank = comm.Get_rank()

    print('[SIM] === Cylinder: Rank {}/{} (blocking) ==='.format(mpi_rank+1, world_size), flush=True)

    device = 'cpu'
    NX, NY = 150, 50
    RE = 100.0
    U_INF = 0.04
    R = 5

    loss_threshold = .001

    train_or_infer = -1

    if len(sys.argv) < 2:
        print("Must pass a real Catalyst pipelining script")
        sys.exit()
    else:
        cat_script = sys.argv[1]
        train_or_infer = sys.argv[2]

    if HAS_CATALYST:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    str(cat_script))
        init_node = conduit.Node()
        init_node['catalyst_load/implementation'] = 'paraview'
        init_node['catalyst_load/search_paths/p0'] = os.environ.get(
            'CATALYST_IMPLEMENTATION_PATHS', '/usr/local/lib/catalyst')
        init_node['catalyst/scripts/script0'] = script_path


        try:
            catalyst.initialize(init_node)
            print('[SIM] Catalyst initialized', flush=True)
        except Exception as e:
            print(f'[SIM] Catalyst init failed: {e}', flush=True)
            HAS_CATALYST = False

    # Solver
    s = CylinderSolver(nx=NX, ny=NY, Re=RE, U_inf=U_INF, R=R, device=device)
    noise = 0.001 * torch.randn(NX, NY, dtype=torch.float64, device=device)
    s.f[:, :, 2] += noise
    s.f[:, :, 4] -= noise
    print(f'[SIM] Coarse solver: {NX}x{NY}, R={R}, Re={RE}', flush=True)

    # Spinup
    SPINUP = 10000
    print(f'[SIM] Spinup: {SPINUP} steps...', flush=True)
    s.step(n=SPINUP)

    N_OUTPUTS = 55
    STEPS_BETWEEN = 50

    print(f'[SIM] Running {N_OUTPUTS} outputs ({STEPS_BETWEEN} LBM steps each)', flush=True)
    print(f'[SIM]   Inference: BLOCKING (solver waits for FNO result)', flush=True)
    print('[SIM]', flush=True)

    t_total_model_query = 0.0
    t_total_catalyst = 0.0

    for step in range(N_OUTPUTS):
        print(" ====================== STEP {} [SPINUP Step: {}] ======================".format(step, SPINUP + STEPS_BETWEEN))

        s.step(n=STEPS_BETWEEN)

        omega = s.omega_field.astype(np.float64)
        ux = s.ux_field.astype(np.float64)
        uy = s.uy_field.astype(np.float64)

        t0 = time.monotonic()
        features = {'omega': omega, 'ux': ux, 'uy': uy}
        exec_node = conduit.Node() 

        cat_fields = {
            'omega_coarse': omega.astype(np.float64)
        }
        refs = build_catalyst_node(exec_node, step, float(s.step_count),
                                        NX, NY, cat_fields, ux, uy)

        t0 = time.monotonic()
        catalyst.execute(exec_node) # Should do the inference and training data push all at once

        t_infer = time.monotonic() - t0
        t_total_model_query += t_infer

        result_node = conduit.Node()

        catalyst.results(result_node) 
        t_cat = time.monotonic() - t0
        t_total_catalyst += t_cat
 
        model_loss = 1000
        del refs

        if result_node.has_path("train_loss"):
            model_loss = np.array(result_node["train_loss"]) 

        if model_loss < loss_threshold:
            print("REPORTED MODEL LOSS IS BELOW THRESHOLD, SWITCH TO INFERENCE")
            sys.exit()

        if exec_node["omega_rom"] is not None:
            omega_rom = np.array(exec_node["omega_rom"])

        if step % 10 == 0:
            if omega_rom is None:
                print(f'[SIM] step={s.step_count:6d}  '
                    f'omega [{omega.min():.4f},{omega.max():.4f}]  '
                    f'infer={t_infer*1000:.0f}ms  ', flush=True)
            else: 
                if omega_rom is None:
                    print(f'[SIM] step={s.step_count:6d}  '
                        f'omega [{omega.min():.4f},{omega.max():.4f}]  '
                        f'rom [{omega_rom.min():.4f},{omega_rom.max():.4f}]  '
                        f'infer={t_infer*1000:.0f}ms  ', flush=True)
    if HAS_CATALYST:
        catalyst.finalize(conduit.Node())  # waits for all pending async work
        print('[SIM] Catalyst finalized (async work completed)', flush=True)

    print('[SIM]', flush=True)
    if train_or_infer == "train":
        print(f'[SIM] Total train time: {t_total_model_query:.2f}s '
          f'({t_total_model_query/N_OUTPUTS*1000:.1f}ms avg)', flush=True)
    elif train_or_infer == "inference": 
        print(f'[SIM] Total inference time: {t_total_model_query:.2f}s '
          f'({t_total_model_query/N_OUTPUTS*1000:.1f}ms avg)', flush=True)
    if HAS_CATALYST:
        print(f'[SIM] Total Catalyst time: {t_total_catalyst:.2f}s '
              f'({t_total_catalyst/N_OUTPUTS*1000:.1f}ms avg)', flush=True)
        print(f'[SIM] With sync Catalyst, total would be ~{t_total_catalyst:.1f}s longer', flush=True)
    print('[SIM] Done.', flush=True)


if __name__ == '__main__':
    main()
