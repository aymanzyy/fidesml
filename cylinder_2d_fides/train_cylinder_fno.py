#!/usr/bin/env python3
"""
train_cylinder_fno.py - Train FNO for cylinder flow with Catalyst visualization.

At each epoch, pushes the FNO prediction through async Catalyst so you can
watch the model learn the vortex structure from scratch.

Epoch 1:   noise
Epoch 50:  vortex shape appearing
Epoch 200: matches fine solution

Catalyst runs ASYNC: training loop doesn't block on rendering.
"""
import sys, os
# CATALYST_IMPLEMENTATION_PATHS should be set in the environment

from mpi4py import MPI
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from fno2d_cylinder import CylinderFNO

# Catalyst (optional)
HAS_CATALYST = False
try:
    import catalyst_conduit as conduit
    import catalyst
    HAS_CATALYST = True
except ImportError:
    pass


def build_catalyst_node(exec_node, cycle, t, nx, ny, fields_dict):
    refs = []
    def ext(arr, dtype=np.float64):
        a = np.ascontiguousarray(arr, dtype=dtype)
        refs.append(a)
        return a

    exec_node['catalyst/state/timestep'] = cycle
    exec_node['catalyst/state/time'] = float(t)
    exec_node['catalyst/channels/grid/type'] = 'mesh'
    mesh = exec_node['catalyst/channels/grid/data']

    x = np.linspace(0, nx - 1, nx)
    y = np.linspace(0, ny - 1, ny)
    X, Y = np.meshgrid(x, y, indexing='ij')

    mesh['coordsets/coords/type'] = 'explicit'
    mesh['coordsets/coords/values/x'].set_external(ext(X.flatten()))
    mesh['coordsets/coords/values/y'].set_external(ext(Y.flatten()))
    mesh['coordsets/coords/values/z'].set_external(ext(np.zeros(nx * ny)))

    i_idx, j_idx = np.meshgrid(np.arange(nx - 1), np.arange(ny - 1), indexing='ij')
    p0 = i_idx * ny + j_idx
    p1 = (i_idx + 1) * ny + j_idx
    p2 = (i_idx + 1) * ny + (j_idx + 1)
    p3 = i_idx * ny + (j_idx + 1)
    conn = ext(np.stack([p0, p1, p2, p3], axis=-1).flatten(), dtype=np.int32)

    mesh['topologies/mesh/type'] = 'unstructured'
    mesh['topologies/mesh/coordset'] = 'coords'
    mesh['topologies/mesh/elements/shape'] = 'quad'
    mesh['topologies/mesh/elements/connectivity'].set_external(conn)

    for name, arr in fields_dict.items():
        mesh[f'fields/{name}/association'] = 'vertex'
        mesh[f'fields/{name}/topology'] = 'mesh'
        mesh[f'fields/{name}/values'].set_external(ext(arr.flatten()))

    return refs


def main():
    global HAS_CATALYST
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print('[train] Loading cylinder training data...', flush=True)
    data = np.load('datasets/cylinder_training_data.npz')
    coarse = data['coarse_states']   # (N, 3, 150, 50)
    fine = data['fine_states']       # (N, 3, 150, 50)
    N = len(coarse)
    NX, NY = 150, 50
    print(f'[train] {N} snapshots, {NX}x{NY}', flush=True)

    # Per-sample per-channel normalization (combined range)
    coarse_norm = np.zeros_like(coarse)
    fine_norm = np.zeros_like(fine)
    norm_params = []

    for i in range(N):
        sample_params = []
        for c in range(3):
            lo = min(np.percentile(coarse[i, c], 1), np.percentile(fine[i, c], 1))
            hi = max(np.percentile(coarse[i, c], 99), np.percentile(fine[i, c], 99))
            if hi - lo < 1e-10:
                lo, hi = -1e-5, 1e-5
            coarse_norm[i, c] = 2.0 * (np.clip(coarse[i, c], lo, hi) - lo) / (hi - lo) - 1.0
            fine_norm[i, c] = 2.0 * (np.clip(fine[i, c], lo, hi) - lo) / (hi - lo) - 1.0
            sample_params.extend([lo, hi])
        norm_params.append(sample_params)

    norm_params = np.array(norm_params)

    # Split
    n_train = int(0.8 * N)
    idx = np.random.RandomState(42).permutation(N)
    tr, te = idx[:n_train], idx[n_train:]

    train_ds = TensorDataset(
        torch.from_numpy(coarse_norm[tr]).float().to(device),
        torch.from_numpy(fine_norm[tr]).float().to(device)
    )
    test_ds = TensorDataset(
        torch.from_numpy(coarse_norm[te]).float().to(device),
        torch.from_numpy(fine_norm[te]).float().to(device)
    )
    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # Model
    model = CylinderFNO(ic=3, oc=3, w=32, nb=4, mx=12, my=8).to(device)
    nparams = sum(p.numel() for p in model.parameters())
    print(f'[train] CylinderFNO: {nparams:,} params', flush=True)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)
    criterion = nn.MSELoss()
    best_loss = float('inf')

    # Catalyst for training visualization (async)
    if HAS_CATALYST:
        script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    'catalyst_pipeline_training_cyl.py')
        if not os.path.exists(script_path):
            # Fall back to the inference pipeline
            script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                        'catalyst_pipeline_cylinder.py')

        init_node = conduit.Node()
        init_node['catalyst_load/implementation'] = 'paraview'
        init_node['catalyst_load/search_paths/p0'] = os.environ.get(
            'CATALYST_IMPLEMENTATION_PATHS', '/usr/local/lib/catalyst')
        init_node['catalyst/scripts/script0'] = script_path
        # init_node['catalyst/async/enabled'] = 1  # disabled: Python GIL conflict
        # init_node['catalyst/async/queue_depth'] = 2

        try:
            catalyst.initialize(init_node)
            print('[train] Catalyst initialized (sync, async disabled for Python GIL)', flush=True)
        except Exception as e:
            print(f'[train] Catalyst init failed: {e}', flush=True)
            HAS_CATALYST = False

    # Pick a reference test sample for visualization
    viz_idx = te[0]
    viz_coarse = coarse_norm[viz_idx]  # (3, 150, 50) normalized
    viz_coarse_raw = coarse[viz_idx]   # (3, 150, 50) physical
    viz_fine_raw = fine[viz_idx]        # (3, 150, 50) physical
    viz_params = norm_params[viz_idx]   # (6,)
    viz_x = torch.from_numpy(viz_coarse).unsqueeze(0).float().to(device)

    print(f'[train] Training 300 epochs (Catalyst viz on test sample {viz_idx})...', flush=True)
    print('[train]', flush=True)
    viz_cycle = 0

    for epoch in range(1, 301):
        model.train()
        tloss = 0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
        tloss /= len(train_loader)
        scheduler.step()

        model.eval()
        vloss = 0
        with torch.no_grad():
            for xb, yb in test_loader:
                vloss += criterion(model(xb), yb).item()
        vloss /= len(test_loader)

        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), 'models/cylinder_fno.pt')
            marker = ' *'
        else:
            marker = ''

        # Visualize every epoch through Catalyst (async, doesn't block)
        if HAS_CATALYST:
            model.eval()
            with torch.no_grad():
                pred_norm = model(viz_x)[0].cpu().numpy()  # (3, 150, 50)

            # Denormalize using this sample's ranges
            omega_pred = (pred_norm[0] + 1.0) / 2.0 * (viz_params[1] - viz_params[0]) + viz_params[0]

            exec_node = conduit.Node()
            cat_fields = {
                'omega_coarse': viz_coarse_raw[0].astype(np.float64),
                'omega_rom': omega_pred.astype(np.float64),
            }
            refs = build_catalyst_node(exec_node, viz_cycle, float(epoch),
                                        NX, NY, cat_fields)
            try:
                catalyst.execute(exec_node)
            except Exception as e:
                pass
            del refs
            viz_cycle += 1

        if epoch <= 5 or epoch % 20 == 0:
            print(f'[train] Epoch {epoch:3d}  train={tloss:.6f}  test={vloss:.6f}{marker}', flush=True)

    if HAS_CATALYST:
        catalyst.finalize(conduit.Node())
        print('[train] Catalyst finalized', flush=True)

    os.makedirs('models', exist_ok=True)
    np.savez('models/cylinder_norm_params.npz', norm_params=norm_params)

    print(f'[train] Best test loss: {best_loss:.6f}', flush=True)
    print(f'[train] Model: models/cylinder_fno.pt', flush=True)


if __name__ == '__main__':
    main()
