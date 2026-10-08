"""
reference.py — Van Driest channel flow reference data generator.

Solves 1D channel flow ODE with simple Prandtl mixing length (baseline)
and Van Driest damped mixing length (reference/truth). Saves
delta_nu_t = nu_t_ref - nu_t_baseline as the ML training target.

Run with pvpython or plain Python:
    pvpython reference.py --output channel_ref.npz
"""
import numpy as np
import argparse

KAPPA   = 0.41   # von Karman constant
A_PLUS  = 26.0   # Van Driest damping
RE_TAU  = 180    # friction Reynolds number
N       = 180    # wall-normal grid points


def solve_channel(nu_t_model, Re_tau=RE_TAU, N=N, n_iter=500):
    """
    Solve fully-developed channel flow by fixed-point iteration.

        d/dy+[(1 + nu_t+) du+/dy+] = -(1 - y+/Re_tau)

    nu_t_model: callable(y+, du+/dy+) -> nu_t+
    Returns (y+, u+, nu_t+, du+/dy+).
    """
    y  = np.linspace(0, Re_tau, N+1)[1:]
    dy = y[1] - y[0]
    u  = y * 0.1
    nu_t = np.zeros(N)

    for _ in range(n_iter):
        du_dy         = np.gradient(u, y)
        nu_t          = nu_t_model(y, du_dy)
        total_stress  = 1.0 - y / Re_tau
        du_dy_new     = total_stress / (1.0 + nu_t)
        u_new         = np.zeros(N)
        u_new[0]      = du_dy_new[0] * y[0]
        for i in range(1, N):
            u_new[i]  = u_new[i-1] + du_dy_new[i] * dy
        if np.max(np.abs(u_new - u)) < 1e-10:
            u = u_new
            break
        u = u_new

    du_dy = np.gradient(u, y)
    nu_t  = nu_t_model(y, du_dy)
    return y, u, nu_t, du_dy


def mixing_length_simple(y_plus, du_dy):
    """Prandtl mixing length — no wall damping. BASELINE."""
    return (KAPPA * y_plus) ** 2 * np.abs(du_dy)


def mixing_length_van_driest(y_plus, du_dy):
    """Van Driest damped mixing length. REFERENCE (truth)."""
    l = KAPPA * y_plus * (1.0 - np.exp(-y_plus / A_PLUS))
    return l ** 2 * np.abs(du_dy)


def generate_reference(output="channel_ref.npz", Re_tau=RE_TAU):
    print(f"Solving baseline (simple mixing length, Re_tau={Re_tau})...")
    y, u_base, nt_base, du_base = solve_channel(mixing_length_simple, Re_tau)

    print(f"Solving reference (Van Driest, Re_tau={Re_tau})...")
    y, u_ref,  nt_ref,  du_ref  = solve_channel(mixing_length_van_driest, Re_tau)

    correction = nt_ref - nt_base   # delta_nu_t — ML target
    coords = np.stack([y, np.zeros_like(y)], axis=1).astype(np.float64)

    np.savez(output,
             coords     = coords,
             correction = correction,
             nu_t_ref   = nt_ref,
             nu_t_base  = nt_base,
             u_ref      = u_ref,
             u_base     = u_base,
             y_plus     = y,
             Re_tau     = Re_tau)

    rmse = np.sqrt(np.mean((u_ref - u_base)**2))
    print(f"Saved {output}: {len(y)} points")
    print(f"  baseline u+_max={u_base.max():.3f}")
    print(f"  reference u+_max={u_ref.max():.3f}  (expect ~21)")
    print(f"  u+ RMSE={rmse:.4f}  max|delta_nu_t|={np.max(np.abs(correction)):.4f}")
    return output


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--output",  default="channel_ref.npz")
    ap.add_argument("--Re_tau",  type=float, default=RE_TAU)
    args = ap.parse_args()
    generate_reference(args.output, args.Re_tau)
