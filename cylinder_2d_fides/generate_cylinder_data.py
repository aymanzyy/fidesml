#!/usr/bin/env python3
"""
generate_cylinder_data.py - Generate training data for cylinder FNO.

Runs fine-grid (600x200) and coarse-grid (150x50) simulations simultaneously.
Saves paired snapshots: (coarse_state, fine_state_restricted_to_coarse_grid)

The FNO learns: coarse snapshot -> fine snapshot (at each timestep).
This is the time-dependent version of the cavity ROM approach.
"""
import sys

import numpy as np
import torch
import time
from scipy.interpolate import RectBivariateSpline
from cylinder_solver import CylinderSolver


def restrict_field(field_fine, nx_fine, ny_fine, nx_coarse, ny_coarse):
    """Interpolate fine field to coarse grid points."""
    x_f = np.linspace(0, 1, nx_fine)
    y_f = np.linspace(0, 1, ny_fine)
    x_c = np.linspace(0, 1, nx_coarse)
    y_c = np.linspace(0, 1, ny_coarse)
    return RectBivariateSpline(x_f, y_f, field_fine)(x_c, y_c)


def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Grid sizes
    NX_FINE, NY_FINE = 600, 200
    NX_COARSE, NY_COARSE = 150, 50
    RE = 100.0
    U_INF = 0.04

    # Cylinder radius scales with grid
    R_FINE = 20
    R_COARSE = 5  # same physical size, fewer grid points

    # Create solvers
    print(f'Fine solver: {NX_FINE}x{NY_FINE}, R={R_FINE}', flush=True)
    s_fine = CylinderSolver(nx=NX_FINE, ny=NY_FINE, Re=RE, U_inf=U_INF,
                             R=R_FINE, device=device)

    print(f'Coarse solver: {NX_COARSE}x{NY_COARSE}, R={R_COARSE}', flush=True)
    s_coarse = CylinderSolver(nx=NX_COARSE, ny=NY_COARSE, Re=RE, U_inf=U_INF,
                               R=R_COARSE, device=device)

    # Add same perturbation (scaled) to both
    noise_fine = 0.001 * torch.randn(NX_FINE, NY_FINE, dtype=torch.float64, device=device)
    s_fine.f[:, :, 2] += noise_fine
    s_fine.f[:, :, 4] -= noise_fine

    noise_coarse = 0.001 * torch.randn(NX_COARSE, NY_COARSE, dtype=torch.float64, device=device)
    s_coarse.f[:, :, 2] += noise_coarse
    s_coarse.f[:, :, 4] -= noise_coarse

    # Spinup: run both solvers until shedding develops
    SPINUP_STEPS = 30000
    STEP_RATIO = NX_FINE // NX_COARSE  # coarse takes fewer steps per physical time

    print(f'Spinup: {SPINUP_STEPS} fine steps...', flush=True)
    t0 = time.monotonic()
    s_fine.step(n=SPINUP_STEPS)
    s_coarse.step(n=SPINUP_STEPS // STEP_RATIO)
    dt_spinup = time.monotonic() - t0
    print(f'Spinup done in {dt_spinup:.1f}s', flush=True)

    # Collect snapshots
    N_SNAPSHOTS = 200
    STEPS_BETWEEN = 100  # fine steps between snapshots
    coarse_steps_between = STEPS_BETWEEN // STEP_RATIO

    coarse_states = []
    fine_states = []

    print(f'Collecting {N_SNAPSHOTS} snapshots ({STEPS_BETWEEN} fine steps apart)...', flush=True)
    t0 = time.monotonic()

    for i in range(N_SNAPSHOTS):
        s_fine.step(n=STEPS_BETWEEN)
        s_coarse.step(n=max(1, coarse_steps_between))

        # Get fields
        omega_fine = s_fine.omega_field
        ux_fine = s_fine.ux_field
        uy_fine = s_fine.uy_field

        omega_coarse = s_coarse.omega_field
        ux_coarse = s_coarse.ux_field
        uy_coarse = s_coarse.uy_field

        # Restrict fine fields to coarse grid
        omega_fine_r = restrict_field(omega_fine, NX_FINE, NY_FINE, NX_COARSE, NY_COARSE)
        ux_fine_r = restrict_field(ux_fine, NX_FINE, NY_FINE, NX_COARSE, NY_COARSE)
        uy_fine_r = restrict_field(uy_fine, NX_FINE, NY_FINE, NX_COARSE, NY_COARSE)

        # Stack channels: [omega, ux, uy]
        coarse_state = np.stack([omega_coarse, ux_coarse, uy_coarse], axis=0)  # (3, 150, 50)
        fine_state = np.stack([omega_fine_r, ux_fine_r, uy_fine_r], axis=0)     # (3, 150, 50)

        coarse_states.append(coarse_state.astype(np.float32))
        fine_states.append(fine_state.astype(np.float32))

        if (i + 1) % 20 == 0:
            f_fine = s_fine.compute_forces()
            f_coarse = s_coarse.compute_forces()
            print(f'  [{i+1}/{N_SNAPSHOTS}] fine Cl={f_fine["Cl"]:+.4f} '
                  f'coarse Cl={f_coarse["Cl"]:+.4f}', flush=True)

    dt_collect = time.monotonic() - t0
    print(f'Collection done in {dt_collect:.1f}s', flush=True)

    # Save
    coarse_arr = np.stack(coarse_states, axis=0)  # (200, 3, 150, 50)
    fine_arr = np.stack(fine_states, axis=0)        # (200, 3, 150, 50)

    outpath = 'datasets/cylinder_training_data.npz'
    import os
    os.makedirs('datasets', exist_ok=True)
    np.savez(outpath, coarse_states=coarse_arr, fine_states=fine_arr,
             Re=RE, nx_fine=NX_FINE, ny_fine=NY_FINE,
             nx_coarse=NX_COARSE, ny_coarse=NY_COARSE)

    print(f'\nSaved {outpath}:', flush=True)
    print(f'  coarse: {coarse_arr.shape}', flush=True)
    print(f'  fine: {fine_arr.shape}', flush=True)
    print(f'  Total: {coarse_arr.nbytes / 1e6:.1f} MB', flush=True)


if __name__ == '__main__':
    main()
