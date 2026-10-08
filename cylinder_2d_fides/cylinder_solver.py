"""
cylinder_solver.py - 2D LBM (D2Q9 BGK) for flow past a cylinder on GPU.

Uses PyTorch tensors on CUDA. Same physics as the numpy version,
but runs 50-100x faster on GPU.

Based on the validated Ceyron/JAX implementation pattern.
"""

import numpy as np
import torch


# D2Q9 lattice constants
EX = torch.tensor([0,  1,  0, -1,  0,  1, -1, -1,  1], dtype=torch.long)
EY = torch.tensor([0,  0,  1,  0, -1,  1,  1, -1, -1], dtype=torch.long)
W  = torch.tensor([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36], dtype=torch.float64)
OPP = torch.tensor([0, 3, 4, 1, 2, 7, 8, 5, 6], dtype=torch.long)

RIGHT_V = torch.tensor([1, 5, 8], dtype=torch.long)
LEFT_V  = torch.tensor([3, 6, 7], dtype=torch.long)
VERT_V  = torch.tensor([0, 2, 4], dtype=torch.long)


class CylinderSolver:

    def __init__(self, nx=600, ny=200, Re=100.0, U_inf=0.04,
                 cx=None, cy=None, R=None, device='cuda'):
        self.nx = nx
        self.ny = ny
        self.Re = Re
        self.U_inf = U_inf
        self.device = device

        self.R = R if R is not None else ny // 10
        self.cx = cx if cx is not None else nx // 5
        self.cy = cy if cy is not None else ny // 2
        self.D = 2 * self.R

        self.nu = U_inf * self.D / Re  # Re based on DIAMETER (standard)
        self.tau = 3.0 * self.nu + 0.5
        self.omega_lbm = 1.0 / self.tau

        # Move lattice constants to device
        self._ex = EX.to(device)
        self._ey = EY.to(device)
        self._w = W.to(device)
        self._opp = OPP.to(device)
        self._right_v = RIGHT_V.to(device)
        self._left_v = LEFT_V.to(device)
        self._vert_v = VERT_V.to(device)

        # Obstacle mask
        Y, X = torch.meshgrid(torch.arange(ny, device=device),
                               torch.arange(nx, device=device), indexing='ij')
        X = X.T  # (nx, ny)
        Y = Y.T
        self.obstacle = ((X - self.cx)**2 + (Y - self.cy)**2) < self.R**2

        # Lattice velocities as (2, 9) float for einsum
        self._lv = torch.tensor([
            [0,  1,  0, -1,  0,  1, -1, -1,  1],
            [0,  0,  1,  0, -1,  1,  1, -1, -1],
        ], dtype=torch.float64, device=device)

        # Initial velocity
        vel = torch.zeros(nx, ny, 2, dtype=torch.float64, device=device)
        vel[:, :, 0] = U_inf

        # Initialize at equilibrium
        rho0 = torch.ones(nx, ny, dtype=torch.float64, device=device)
        self.f = self._equilibrium(vel, rho0)

        self.step_count = 0

    def _equilibrium(self, u, rho):
        """Compute equilibrium distribution."""
        # eu[nx, ny, 9] = sum_d lv[d, q] * u[nx, ny, d]
        eu = torch.einsum('dq,NMd->NMq', self._lv, u)
        u_sq = torch.sum(u**2, dim=-1)  # (nx, ny)
        feq = (rho.unsqueeze(-1) * self._w.unsqueeze(0).unsqueeze(0) *
               (1.0 + 3.0 * eu + 4.5 * eu**2 - 1.5 * u_sq.unsqueeze(-1)))
        return feq

    def _macroscopic(self, f):
        """Compute density and velocity from distribution."""
        rho = torch.sum(f, dim=-1)
        u = torch.einsum('NMq,dq->NMd', f, self._lv) / rho.unsqueeze(-1)
        return rho, u

    def step(self, n=1):
        """Advance n LBM timesteps on GPU."""
        f = self.f
        obstacle = self.obstacle
        omega = self.omega_lbm
        U_inf = self.U_inf

        for _ in range(n):
            # (1) Outflow BC (right boundary): copy left-moving distributions
            f[-1, :, self._left_v] = f[-2, :, self._left_v]

            # (2) Macroscopic quantities
            rho, u = self._macroscopic(f)

            # (3) Inlet BC: Zou-He (left boundary)
            u[0, 1:-1, 0] = U_inf
            u[0, 1:-1, 1] = 0.0
            rho[0, :] = (
                torch.sum(f[0][:, self._vert_v], dim=-1) +
                2.0 * torch.sum(f[0][:, self._left_v], dim=-1)
            ) / (1.0 - U_inf)

            # (4) Equilibrium
            feq = self._equilibrium(u, rho)

            # Zou-He: set right-moving at inlet to equilibrium
            f[0, :, self._right_v] = feq[0, :, self._right_v]

            # (5) Collision (BGK)
            f_post = f - omega * (f - feq)

            # (6) Bounce-back (uses pre-collision f)
            for i in range(9):
                f_post[:, :, i][obstacle] = f[:, :, self._opp[i]][obstacle]

            # (7) Streaming (roll each velocity component)
            for i in range(9):
                f_post[:, :, i] = torch.roll(
                    torch.roll(f_post[:, :, i], int(self._ex[i]), dims=0),
                    int(self._ey[i]), dims=1
                )

            f = f_post
            self.step_count += 1

        self.f = f

    @property
    def omega_field(self):
        """Vorticity field (on CPU as numpy)."""
        rho, u = self._macroscopic(self.f)
        ux = u[:, :, 0]
        uy = u[:, :, 1]
        # Gradient via central differences
        duy_dx = torch.roll(uy, -1, 0) - torch.roll(uy, 1, 0)
        dux_dy = torch.roll(ux, -1, 1) - torch.roll(ux, 1, 1)
        w = 0.5 * (duy_dx - dux_dy)
        w[self.obstacle] = 0.0
        return w.cpu().numpy()

    @property
    def velocity_magnitude(self):
        """Velocity magnitude (on CPU as numpy)."""
        rho, u = self._macroscopic(self.f)
        mag = torch.sqrt(u[:, :, 0]**2 + u[:, :, 1]**2)
        mag[self.obstacle] = 0.0
        return mag.cpu().numpy()

    @property
    def ux_field(self):
        rho, u = self._macroscopic(self.f)
        ux = u[:, :, 0].clone()
        ux[self.obstacle] = 0.0
        return ux.cpu().numpy()

    @property
    def uy_field(self):
        rho, u = self._macroscopic(self.f)
        uy = u[:, :, 1].clone()
        uy[self.obstacle] = 0.0
        return uy.cpu().numpy()

    @property
    def rho_field(self):
        return torch.sum(self.f, dim=-1).cpu().numpy()

    def compute_forces(self):
        """Compute Cd and Cl via control volume momentum flux."""
        rho, u = self._macroscopic(self.f)
        ux = u[:, :, 0]
        uy = u[:, :, 1]
        p = rho / 3.0

        # Control volume faces: upstream and downstream of cylinder
        x1 = max(0, int(self.cx - 2.5 * self.R))
        x2 = min(self.nx - 1, int(self.cx + 4 * self.R))

        # Drag: net x-momentum flux through left/right faces
        # F_x = integral(rho*ux^2 + p) at x1 - integral(rho*ux^2 + p) at x2
        #      + integral(rho*ux*uy) at top - integral(rho*ux*uy) at bottom
        left = torch.sum(rho[x1, :] * ux[x1, :]**2 + p[x1, :]).item()
        right = torch.sum(rho[x2, :] * ux[x2, :]**2 + p[x2, :]).item()
        fx = left - right

        # Lift: net y-momentum flux
        left_y = torch.sum(rho[x1, :] * ux[x1, :] * uy[x1, :]).item()
        right_y = torch.sum(rho[x2, :] * ux[x2, :] * uy[x2, :]).item()
        fy = left_y - right_y

        rho_mean = torch.mean(rho).item()
        denom = 0.5 * rho_mean * self.U_inf**2 * self.D
        return {'Cd': fx / denom, 'Cl': fy / denom}


    def get_state(self):
        return {
            'omega': self.omega_field,
            'ux': self.ux_field,
            'uy': self.uy_field,
            'rho': self.rho_field,
            'step': self.step_count,
            'Re': self.Re,
        }


if __name__ == '__main__':
    import time

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f'LBM GPU cylinder solver on {device}', flush=True)

    s = CylinderSolver(nx=600, ny=200, Re=100, U_inf=0.04, R=20, device=device)  # Re_D=100
    print(f'Grid: {s.nx}x{s.ny}, R={s.R}, D={s.D}', flush=True)
    print(f'Re={s.Re}, tau={s.tau:.4f}, nu={s.nu:.6f}', flush=True)
    print(f'', flush=True)

    # Warmup
    s.step(n=10)
    torch.cuda.synchronize() if device == 'cuda' else None

    # Benchmark
    t0 = time.monotonic()
    s.step(n=1000)
    torch.cuda.synchronize() if device == 'cuda' else None
    dt = time.monotonic() - t0
    print(f'Benchmark: 1000 steps in {dt:.2f}s ({1000/dt:.0f} steps/s)', flush=True)
    print(f'', flush=True)

    # Run to develop shedding
    cl_history = []
    print('Running 20000 steps...', flush=True)
    for i in range(200):
        s.step(n=100)
        f = s.compute_forces()
        cl_history.append(f['Cl'])
        if i % 20 == 0:
            print(f'  step={s.step_count:6d}  Cd={f["Cd"]:7.3f}  Cl={f["Cl"]:+8.4f}', flush=True)

    print(f'\nDone: {s.step_count} steps', flush=True)
    print(f'Cl range: [{min(cl_history):.4f}, {max(cl_history):.4f}]', flush=True)

    # Plot
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    w = s.omega_field
    fig, axes = plt.subplots(2, 1, figsize=(16, 8))

    ax = axes[0]
    X = np.arange(s.nx)
    Y = np.arange(s.ny)
    cf = ax.contourf(X, Y, w.T, levels=np.linspace(-0.02, 0.02, 40),
                      cmap='RdBu_r', extend='both')
    ax.add_patch(plt.Circle((s.cx, s.cy), s.R, color='gray'))
    ax.set_title(f'Vorticity at step {s.step_count}, Re={s.Re}')
    ax.set_aspect('equal')
    plt.colorbar(cf, ax=ax)

    ax = axes[1]
    ax.plot(np.arange(len(cl_history)) * 100, cl_history, 'b-', linewidth=0.5)
    ax.set_xlabel('Step')
    ax.set_ylabel('Cl')
    ax.set_title('Lift coefficient history')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('cylinder_gpu.png', dpi=100)
    print('Saved cylinder_gpu.png', flush=True)
