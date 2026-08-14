#!/usr/bin/env python3
"""
1D heat equation with a FUNCTION-VALUED input parameter.

    u_t = kappa u_xx + g(t) s(x),   u(0,t)=u(L,t)=0,   u(x,0)=0

The "input parameter" is the entire forcing schedule g(t), drawn randomly
per trajectory. The IC is zero for every trajectory, so the ONLY thing that
distinguishes one trajectory from another is its forcing function g(t).

This is the case interpolation-based LaSDI cannot easily handle
"""

import os, sys, time
import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import solve_ivp
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
import matplotlib
import matplotlib.pyplot as plt


from lasdi.latent_dynamics.gatedsindy import TauGatedSINDy

from matplotlib import rc
rc('font', **{'family': 'serif', 'serif': ['Computer Modern']})
rc('text', usetex=True)

plt.rcParams['font.size'] = 20
plt.rcParams['axes.linewidth'] = 1
plt.rcParams['lines.linewidth'] = 3
plt.rcParams['lines.markersize'] = 10
plt.rcParams['axes.spines.right'] = True
plt.rcParams['axes.spines.top'] = True
plt.rcParams['legend.frameon'] = True
plt.rcParams['legend.fontsize'] = 20
# plt.rcParams['suptitle.fontsize'] = 20
plt.rcParams['axes.labelsize'] = 28
plt.rcParams['figure.dpi'] = 150

torch.manual_seed(0)
np.random.seed(0)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("device:", device)


# %% Autoencoder
class Autoencoder(torch.nn.Module):
    def __init__(self, space_dim, hidden_units, n_z):
        super().__init__()
        n_layers = len(hidden_units)
        self.n_layers = n_layers
        self.fc1_e = torch.nn.Linear(space_dim, hidden_units[0])
        
        if n_layers > 1:
            for i in range(n_layers - 1):
                setattr(self, 'fc' + str(i + 2) + '_e',
                        torch.nn.Linear(hidden_units[i], hidden_units[i + 1]))
        setattr(self, 'fc' + str(n_layers + 1) + '_e',
                torch.nn.Linear(hidden_units[-1], n_z))
        
        # self.g_e = torch.nn.Softplus()
        self.g_e = torch.nn.GELU()
        
        self.fc1_d = torch.nn.Linear(n_z, hidden_units[-1])
        
        if n_layers > 1:
            for i in range(n_layers - 1, 0, -1):
                setattr(self, 'fc' + str(n_layers - i + 1) + '_d',
                        torch.nn.Linear(hidden_units[i], hidden_units[i - 1]))
        setattr(self, 'fc' + str(n_layers + 1) + '_d',
                torch.nn.Linear(hidden_units[0], space_dim))
        
        self.n_z = n_z

    def encoder(self, x):
        for i in range(1, self.n_layers + 1):
            x = self.g_e(getattr(self, 'fc' + str(i) + '_e')(x))
        return getattr(self, 'fc' + str(self.n_layers + 1) + '_e')(x)

    def decoder(self, x):
        for i in range(1, self.n_layers + 1):
            x = self.g_e(getattr(self, 'fc' + str(i) + '_d')(x))
        return getattr(self, 'fc' + str(self.n_layers + 1) + '_d')(x)

    def forward(self, x):
        return self.decoder(self.encoder(x))


# %% Problem setup
L      = 1.0
nx     = 64                     
kappa  = 0.10
T_end  = 2.0
# nt     = 161
nt     = 201
t_grid = np.linspace(0.0, T_end, nt)
dt     = float(t_grid[1] - t_grid[0])

dx     = L / (nx + 1)
x_int  = np.linspace(dx, L - dx, nx)

# Laplacian (Dirichlet) and forcing spatial profile (localized bump).
main = -2.0 * np.ones(nx); off = np.ones(nx - 1)
D2 = (np.diag(main) + np.diag(off, 1) + np.diag(off, -1)) / dx**2
A_heat = kappa * D2
# s_x = np.exp(-((x_int - 0.5) / 0.12) ** 2)
s_x = np.exp(-((x_int - 0.5) / 0.1) ** 2)


# %% Forcing functions
N_MODES = 3

#random sines and cosines
def make_forcing(rng):
    a = rng.uniform(-1.0, 1.0, N_MODES)
    b = rng.uniform(-1.0, 1.0, N_MODES)
    a0 = rng.uniform(-0.3, 0.3)
    def g(t):
        val = a0 * np.ones_like(np.atleast_1d(t), dtype=float)
        for m in range(1, N_MODES + 1):
            val = val + a[m-1]*np.sin(2*np.pi*m*t/T_end) + b[m-1]*np.cos(2*np.pi*m*t/T_end)
        return val if np.ndim(t) else float(val[0])
    return g

#out of distribution test
def ramp_forcing(t):                              
    t = np.atleast_1d(t).astype(float)
    # val = 0.8 * (np.tanh((t - 0.7) / 0.12) - np.tanh((t - 1.4) / 0.12))
    val = np.exp(-((t - 1) / 0.5) ** 2)
    return val if np.ndim(t) else float(val[0])

# %% build train / test sets
def solve_heat(gfun):
    def rhs(t, u):
        return A_heat @ u + gfun(t) * s_x
    sol = solve_ivp(rhs, (t_grid[0], t_grid[-1]), np.zeros(nx),
                    t_eval=t_grid, method='BDF', jac=lambda t, u: A_heat,
                    rtol=1e-8, atol=1e-10)
    return sol.y.T                                # [nt, nx]

N_train, N_test = 16, 8
rng = np.random.default_rng(1)
train_g = [make_forcing(rng) for _ in range(N_train)]
test_g  = [make_forcing(rng) for _ in range(N_test)]

print("solving full-order heat equation ...")
tic = time.time()
U_train = np.stack([solve_heat(g) for g in train_g])      # [B,T,nx]
U_test  = np.stack([solve_heat(g) for g in test_g])
U_ood   = solve_heat(ramp_forcing)[None]                  # [1,T,nx]
print(f"  FOM solves done in {time.time()-tic:.1f}s")

# forcing values on the time grid
G_train = np.stack([g(t_grid) for g in train_g])          # [B,T]
G_test  = np.stack([g(t_grid) for g in test_g])
G_ood   = ramp_forcing(t_grid)[None]

# normalization for the gate's forcing channel (from training only)
GMIN, GMAX = G_train.min(), G_train.max()
def gnorm(G): return (G - GMIN) / (GMAX - GMIN)

# data scale for AE
USCALE = np.abs(U_train).max()
U_train_s = U_train / USCALE
U_test_s  = U_test  / USCALE
U_ood_s   = U_ood   / USCALE


#%% write to tensors for training
def tt(a): 
    return torch.tensor(a, dtype=torch.float32, device=device)

U_tr  = tt(U_train_s)                                     # [B,T,nx]
P_tr  = tt(gnorm(G_train))[..., None]                     # [B,T,1] forcing channel
tau_np = (t_grid - t_grid[0]) / (t_grid[-1] - t_grid[0])
Tau_tr = tt(tau_np).view(1, nt, 1).expand(N_train, nt, 1).contiguous()


# %% models

#also set n_tau_freqs = 2, K = 20, Thresh 5e-3, lr 1E-3 for this one
# n_z = 10
# hidden_units = [200, 20]

n_z = 10
hidden_units = [32, 16]
# hidden_units = [32, 16, 16]
# hidden_units = [20, 10]
ae = Autoencoder(nx, hidden_units, n_z).to(device)

gsindy = TauGatedSINDy(
    n_z=n_z, K=20, poly_order=1,        # LINEAR experts
    n_p=1,                              # one forcing feature g(t)
    n_tau_freqs=0, n_param_freqs=0,
    use_t_in_gate=True,
    use_param_in_gate=True,             # gate reads the forcing schedule
    use_z_in_gate=False,                # no state, usually do this
    use_param_in_lib=False,
    gate_hidden=64, sparsity='l2', use_mask=False,
).to(device)

print(f"TauGatedSINDy: gate_in_dim={gsindy.gate_in_dim}, L={gsindy.L}, K={gsindy.K}, "
      f"state-blind gate (L_w=0): {not gsindy.use_z_in_gate}")

params = list(ae.parameters()) + list(gsindy.parameters())
opt = torch.optim.Adam(params, lr=1e-3)
MSE = nn.MSELoss()

# ae_w, sindy_w, coef_w = 1.0, 1e-1, 1e-3
ae_w, sindy_w, coef_w = 1.0, 1e-1, 1e-4
SINDY_WARMUP = 1500
WARMUP, RETHRESH, THRESH = 3000, 1500, 1e-3
n_iter = 6000
# n_iter = 10000


# %% Train

print("training (AE + gated MoE) ...")
tic = time.time()
for it in range(n_iter):
    opt.zero_grad()
    Z = ae.encoder(U_tr)
    Upred = ae.decoder(Z)
    loss_ae = MSE(U_tr, Upred)
    loss_sindy, loss_coef, loss_bal, w = gsindy.losses(Z, Tau_tr, dt, P=P_tr, fd='central')

    if it == WARMUP:
        na, nt_ = gsindy.update_mask(THRESH, 'absolute')
        print(f"  [STLS] iter {it}: {na}/{nt_} active ({100*na/nt_:.1f}%)")
    elif it > WARMUP and (it - WARMUP) % RETHRESH == 0:
        na, nt_ = gsindy.update_mask(THRESH, 'absolute')
        print(f"  [STLS] iter {it}: {na}/{nt_} active")

    ramp = min(1.0, it / SINDY_WARMUP)
    loss = ae_w*loss_ae + sindy_w*ramp*loss_sindy + coef_w*ramp*loss_coef
    loss.backward()
    opt.step()
    if it % 100 == 0 or it == n_iter - 1:
        print(f"  it {it:5d}  loss {loss.item():.3e}  ae {loss_ae.item():.3e}  "
              f"sindy {loss_sindy.item():.3e}  coeff {loss_coef.item():.3e}")
print(f" Training finished in {time.time()-tic:.1f}s")


# %% Rollout fine-tune. We don't actually need for this problem
# Freeze AE, fine-tune dynamics on a detached multi-step rollout
def rollout_finetune(n_roll=300, H=10, lr=1e-4):
    for p in ae.parameters():
        p.requires_grad_(False)
    opt2 = torch.optim.Adam(gsindy.parameters(), lr=lr)
    span = float(Tau_tr[0, -1, 0] - Tau_tr[0, 0, 0])
    with torch.no_grad():
        Z_full = ae.encoder(U_tr)                         # [B,T,n_z]
    print("rollout fine-tuning ...")
    tic = time.time()
    for it in range(n_roll):
        opt2.zero_grad()
        n_starts = nt - H
        z = Z_full[:, :n_starts, :]
        total = 0.0
        for h in range(H):
            Ta = Tau_tr[:, h:h+n_starts, :]
            Tb = Ta + 0.5 * dt / span
            Tc = Tau_tr[:, h+1:h+1+n_starts, :]
            Pa = P_tr[:, h:h+n_starts, :]
            Pc = P_tr[:, h+1:h+1+n_starts, :]
            Pb = 0.5 * (Pa + Pc)
            k1, _ = gsindy.predict_dzdt(z,              Ta, Pa)
            k2, _ = gsindy.predict_dzdt(z + 0.5*dt*k1,  Tb, Pb)
            k3, _ = gsindy.predict_dzdt(z + 0.5*dt*k2,  Tb, Pb)
            k4, _ = gsindy.predict_dzdt(z +     dt*k3,  Tc, Pc)
            z_next = z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
            total = total + ((z_next - Z_full[:, h+1:h+1+n_starts, :])**2).mean()
            z = z_next.detach()
        _, lc, _, _ = gsindy.losses(Z_full, Tau_tr, dt, P=P_tr, fd='central')
        loss = total / H + 1e-3 * lc
        loss.backward()
        opt2.step()
        if it % 200 == 0 or it == n_roll - 1:
            print(f"  ft {it:4d}  rollout {(total/H).item():.3e}")
    print(f"  fine-tune done in {time.time()-tic:.1f}s")

# rollout_finetune()

# %% Eval
ae.eval(); gsindy.eval()


with torch.no_grad():
    z0_np = ae.encoder(U_tr).cpu().numpy()[:, 0, :].mean(0)   # (kept for reference)

# ============================================================ (B) gated MoE rollout
@torch.no_grad()
def moe_predict(gfun):
    span = float(t_grid[-1] - t_grid[0])
    z = tt(z0_np).view(1, 1, n_z)
    def gp_(t):                                            # normalized forcing tensor
        return tt(np.array([[[(gfun(t) - GMIN) / (GMAX - GMIN)]]], dtype=np.float32)
                  ).view(1, 1, 1)
    def tau_(t):
        return tt(np.array([[[(t - t_grid[0]) / span]]], dtype=np.float32)).view(1, 1, 1)
    def f(zz, t):
        dz, _ = gsindy.predict_dzdt(zz, tau_(t), gp_(t))
        return dz
    zs = [z]
    for i in range(nt - 1):
        tn = float(t_grid[i])
        k1 = f(z,             tn)
        k2 = f(z + 0.5*dt*k1, tn + 0.5*dt)
        k3 = f(z + 0.5*dt*k2, tn + 0.5*dt)
        k4 = f(z +     dt*k3, tn + dt)
        z = z + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        zs.append(z)
    Z = torch.cat(zs, dim=1)                               # [1,T,n_z]
    U = ae.decoder(Z).cpu().numpy()[0]
    return U * USCALE


# ============================================================ evaluate
def rel_err_time(pred, true):
    num = np.linalg.norm(pred - true, axis=1)
    den = np.linalg.norm(true, axis=1).max() + 1e-12
    # den = np.linalg.norm(true, axis=1) + 1e-12
    return num / den                                       # per-time relative error

print("\nevaluating on held-out forcing functions ...")
errs_moe = []
moe_preds = []
for i, g in enumerate(test_g):
    Um = moe_predict(g);
    moe_preds.append(Um);
    em = np.nanmean(rel_err_time(Um, U_test[i]))
    errs_moe.append(em);
    print(f"  test {i}:  MoE {em:6.3%} ")

Um_ood = moe_predict(ramp_forcing); 
e_ood_m = np.nanmean(rel_err_time(Um_ood, U_ood[0]))
print(f"  OOD ramp: MoE {e_ood_m:6.3%} ")
print(f"\nMEAN held-out:  MoE {np.mean(errs_moe):.3%}   ")
print(f"n_active experts after STLS: "
      f"{int((gsindy.Xi_effective.abs().sum(dim=(1,2))>1e-8).sum().item())}/{gsindy.K}")


# ============================================================ figure

fig, axes = plt.subplots(nrows=2, ncols=3, sharex=True, sharey=False,constrained_layout=True)
fig.set_size_inches(12, 6)
idx = 0
# forcing functions
ax = axes[0,0]
for i in range(N_train):
    ax.plot(t_grid, G_train[i], color='0.8', lw=0.8)
ax.plot(t_grid, G_test[0], 'C0', label='Held-out')
# ax.plot(t_grid, G_ood[0], 'C3', label='OOD $g(t)$')
ax.plot([], [], color='0.8', lw=0.8, label='Training')
#ax.set_title('(a) input is a function g(t)')
ax.set_xlabel('$t$'); ax.set_ylabel('$g(t)$'); 
ax.set_title('Forcing Functions')
# ax.legend()

cmin = min(np.min(U_test[idx]), np.min(moe_preds[idx]))
cmax = max(np.max(U_test[idx]), np.max(moe_preds[idx]))

ax = axes[0,1]
im = ax.imshow(U_test[idx].T, aspect='auto', origin='lower',
               extent=[0, T_end, 0, L], cmap='magma', vmin = cmin, vmax = cmax)
ax.set_title('True'); ax.set_xlabel('$t$'); ax.set_ylabel('$x$')

ax = axes[0,2]
im = ax.imshow(moe_preds[idx].T, aspect='auto', origin='lower',
               extent=[0, T_end, 0, L], cmap='magma', vmin = cmin, vmax = cmax)
ax.set_title('Gated MoE'); ax.set_xlabel('$t$'); ax.set_ylabel('$x$')
fig.colorbar(im, ax=axes[0,1:])

# forcing functions
ax = axes[1,0]
for i in range(N_train):
    ax.plot(t_grid, G_train[i], color='0.8', lw=0.8)
# ax.plot(t_grid, G_test[0], 'C0', label='Held-out $g(t)$')
ax.plot(t_grid, G_ood[0], 'C3', label='OOD')
ax.plot([], [], color='0.8', lw=0.8, label='Training')
#ax.set_title('(a) input is a function g(t)')
ax.set_xlabel('$t$'); ax.set_ylabel('$g(t)$'); 
ax.set_title('Forcing Functions')
# ax.legend()

idx = 0
cmin = min(np.min(U_ood[idx]), np.min(Um_ood))
cmax = max(np.max(U_ood[idx]), np.max(Um_ood))

ax = axes[1,1]
im = ax.imshow(U_ood[idx].T, aspect='auto', origin='lower',
               extent=[0, T_end, 0, L], cmap='magma', vmin = cmin, vmax = cmax)
ax.set_title('True'); ax.set_xlabel('$t$'); ax.set_ylabel('$x$')

ax = axes[1,2]
im = ax.imshow(Um_ood.T, aspect='auto', origin='lower',
               extent=[0, T_end, 0, L], cmap='magma', vmin = cmin, vmax = cmax)
ax.set_title('Gated MoE'); ax.set_xlabel('$t$'); ax.set_ylabel('$x$')
fig.colorbar(im, ax=axes[1,1:])
