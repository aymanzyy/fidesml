"""
channel_sim.py — 1D turbulent channel flow solver, simulation side.

Inference mode runs in two phases:
  1. Converge the baseline mixing length solver (no ML correction)
  2. Apply ML correction once (via Catalyst), re-solve to convergence
  3. Report improvement vs baseline and reference

This mirrors how RANS closure corrections are applied: the baseline solver
converges first, then the ML correction to nu_t+ is applied and the solver
re-runs. Applying the correction at every iteration step before convergence
causes distribution shift (the model sees out-of-distribution features).

Training mode calls Catalyst at every iteration for diverse feature coverage.

Run with pvpython:
    pvpython channel_sim.py catalyst_channel_ml.py [training|inference]
"""
import sys
import numpy as np
import channel_adaptor
import time
KAPPA  = 0.41
RE_TAU = 180
N      = 10000
N_ITER = 2000


def mixing_length_simple(y_plus, du_dy):
    return (KAPPA * y_plus) ** 2 * np.abs(du_dy)


def solve_to_convergence(y, delta_nu_t=None, max_iter=500, tol=1e-10):
    """Run the channel flow ODE to convergence.
    
    delta_nu_t: optional additive correction to mixing length, held fixed
    throughout the solve. This is what the ML model returns — the solver
    then converges the velocity field given the corrected closure.
    """
    dy = y[1] - y[0]
    u  = y * 0.1

    for i in range(max_iter):
        du_dy = np.gradient(u, y)
        nu_t  = mixing_length_simple(y, du_dy)
        if delta_nu_t is not None:
            nu_t = np.maximum(nu_t + delta_nu_t, 0.0)

        total_stress = 1.0 - y / RE_TAU
        du_dy_new    = total_stress / (1.0 + nu_t)
        u_new        = np.zeros(N)
        u_new[0]     = du_dy_new[0] * y[0]
        for j in range(1, N):
            u_new[j] = u_new[j-1] + du_dy_new[j] * dy

        residual = np.max(np.abs(u_new - u))
        u = u_new
        if residual < tol and i > 5:
            return u, nu_t, du_dy
    return u, nu_t, du_dy


def main():
    mode = "inference"
    for arg in sys.argv[2:]:
        if arg in ("training", "inference"):
            mode = arg

    y  = np.linspace(0, RE_TAU, N+1)[1:]
    dy = y[1] - y[0]
    u  = y * 0.1

    loop_with_init = time.monotonic()

    channel_adaptor.initialize(mode=mode, Re_tau=RE_TAU)

    loop_timers = []

    if mode == "training":
        # Training: call Catalyst every step for diverse feature coverage
        for step in range(N_ITER):
            loop_begin = time.monotonic()

            du_dy = np.gradient(u, y)
            nu_t  = mixing_length_simple(y, du_dy)

            _, cat_execs, cat_ress = channel_adaptor.coprocess(step, float(step), y, u, nu_t, du_dy)

            total_stress = 1.0 - y / RE_TAU
            du_dy_new    = total_stress / (1.0 + nu_t)
            u_new        = np.zeros(N)
            u_new[0]     = du_dy_new[0] * y[0]
            for i in range(1, N):
                u_new[i] = u_new[i-1] + du_dy_new[i] * dy

            residual = np.max(np.abs(u_new - u))
            u = u_new

            if step % 50 == 0:
                print(f"[sim] step={step:4d}  u+_max={u.max():.4f}  "
                      f"residual={residual:.2e}", flush=True)

            loop_end = time.monotonic() - loop_begin

            loop_timers.append(loop_end)
    else:
        # Inference: two-phase approach
        # Phase 1: converge baseline (no Catalyst, no correction)
        print("[sim] Phase 1: converging baseline...", flush=True)
        u_base, nu_t_base, du_base = solve_to_convergence(y)
        print(f"[sim] Baseline converged: u+_max={u_base.max():.4f}", flush=True)

        print("[sim] Phase 2: requesting ML correction...", flush=True)
        delta_nu_t = channel_adaptor.coprocess(
            0, 0.0, y, u_base, nu_t_base, du_base)

        if delta_nu_t is not None:
            print(f"[sim] Correction received: max|Δνt+|={np.max(np.abs(delta_nu_t)):.4f}",
                  flush=True)
            # Phase 3: re-solve holding delta_nu_t fixed at every iteration
            u_corr, _, _ = solve_to_convergence(y, delta_nu_t=delta_nu_t)
            print(f"[sim] Corrected solution: u+_max={u_corr.max():.4f}", flush=True)
        else:
            print("[sim] No correction received, reporting baseline", flush=True)
            u_corr = u_base

        print(f"\n[sim] Results:")
        print(f"  baseline u+_max  = {u_base.max():.4f}  (simple mixing length)")
        print(f"  corrected u+_max = {u_corr.max():.4f}  (with ML correction)")
        print(f"  reference u+_max = ~15.79              (Van Driest)")
        print(f"  improvement      = {u_corr.max() - u_base.max():+.4f} u+ units")

    channel_adaptor.finalize()
    avg_loop_timer = np.mean(np.array(loop_timers))
    avg_exces_timer = np.mean(np.array(cat_execs))
    avg_results_timer = np.mean(np.array(cat_ress))

    print(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++")
    print("Avg Loop Time: {} secs".format(avg_loop_timer))
    print("Avg Catalyst Execute Time: {} secs".format(avg_exces_timer))
    print("Avg Catalyst Results Time: {} secs".format(avg_results_timer))
    print(" ++++++++++++++++++++++++++++++++++++++++++++++++++++++")

if __name__ == "__main__":
    main()
