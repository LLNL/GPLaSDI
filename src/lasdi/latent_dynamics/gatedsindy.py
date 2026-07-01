"""
Mixture-of-experts SINDy 
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#Don't need this
def fourier_features(s, num_freqs):
    """
    Fourier feature encoding applied per-component.

    s         : [..., D], each component expected roughly in [0, 1].
    num_freqs : K. Frequencies are pi * 2^k for k = 0..K-1 (same schedule
                as the UNet operator script; lowest frequency is half a
                cycle across [0, 1] so it is monotonic, no period-1 alias).

    Returns [..., D * (1 + 2*K)]:
        [s, sin(w_0 s), cos(w_0 s), sin(w_1 s), cos(w_1 s), ...]
    where each sin/cos is applied element-wise so each input component
    gets its own Fourier expansion.
    """
    if num_freqs <= 0:
        return s
    feats = [s]
    for k in range(num_freqs):
        w = (2.0 ** k) * math.pi
        feats.append(torch.sin(w * s))
        feats.append(torch.cos(w * s))
    return torch.cat(feats, dim=-1)


class TauGatedSINDy(nn.Module):
    """
    Parameters
    ----------
    n_z : int
        Latent dimension.
    K : int
        Number of expert coefficient matrices.
    poly_order : int
        Polynomial order of the library in z (1 or 2).
    include_bias : bool
        Include a constant term in the library.
    n_p : int
        Number of scalar parameter features 
    n_tau_freqs : int
        Number of Fourier-feature frequencies for tau.
    n_param_freqs : int
        Number of Fourier-feature frequencies for each component of p.
        Used only when use_param_in_gate=True.
    use_z_in_gate : bool
        Default False. If True, raw z is concatenated into the gate input.
    use_z0_in_gate : bool
        Default False. If True, the per-trajectory initial latent z0 is
        concatenated (after optional embedding) into the gate input.
        z0 stays constant across time within a trajectory
    z0_embed_dim : int or None
        If not None and use_z0_in_gate=True, z0 is first projected by a
        learned linear layer to this dimension before being fed to the
        gate. Useful when n_z is large relative to gate_hidden.
    use_param_in_gate : bool
        Default True. If True and n_p > 0, the gate sees Fourier features
        of p.
    use_param_in_lib : bool
        Default False. If True and n_p > 0, the library is augmented with
        p and p_i * base_lib. Probably never use this
    gate_hidden : int
        Hidden width of the gate MLP.
    sparsity, elastic_alpha, use_mask, xi_init_scale : see
        SparseGatedSINDy.
    """

    def __init__(self,
                 n_z,
                 K=5,
                 poly_order=1,
                 include_bias=True,
                 n_p=2,
                 n_tau_freqs=0,
                 n_param_freqs=0,
                 use_t_in_gate=True,
                 use_z_in_gate=False,
                 use_z0_in_gate=False,
                 z0_embed_dim=None,
                 use_param_in_gate=False,
                 use_param_in_lib=False,
                 gate_hidden=64,
                 xi_init_scale=1e-2,
                 sparsity='l2',
                 elastic_alpha=0.5,
                 use_mask=False):
        super().__init__()

        valid = ('l2', 'l1', 'group_row', 'shared', 'elastic')
        if sparsity not in valid:
            raise ValueError(
                f"sparsity must be one of {valid}, got {sparsity!r}"
            )

        self.n_z = n_z
        self.n_p = int(n_p)
        self.K = K
        self.poly_order = poly_order
        self.include_bias = include_bias
        self.n_tau_freqs = int(n_tau_freqs)
        self.n_param_freqs = int(n_param_freqs)
        self.use_t_in_gate = use_t_in_gate
        self.use_z_in_gate = use_z_in_gate
        self.use_z0_in_gate = use_z0_in_gate
        self.z0_embed_dim = z0_embed_dim
        self.use_param_in_gate = use_param_in_gate and (self.n_p > 0)
        self.use_param_in_lib = use_param_in_lib and (self.n_p > 0)
        self.sparsity = sparsity
        self.elastic_alpha = float(elastic_alpha)
        self.use_mask = use_mask

        # Base library (in z only).
        L_base = (1 if include_bias else 0) + n_z
        if poly_order >= 2:
            L_base += n_z * (n_z + 1) // 2
        if poly_order >= 3:
            raise NotImplementedError("poly_order >= 3 not supported")
        self.L_base = L_base

        # Full library after parameter augmentation.
        # Append: p (n_p terms) and p_i * base for each i (n_p * L_base terms).
        if self.use_param_in_lib:
            L = L_base + self.n_p + self.n_p * L_base
        else:
            L = L_base
        self.L = L

        # Coefficient tensor.
        self.Xi = nn.Parameter(torch.randn(K, L, n_z) * xi_init_scale)

        # Hard mask buffer (1 = active, 0 = killed).
        self.register_buffer('mask', torch.ones(K, L, n_z))

        # Optional z0 embedding. If z0_embed_dim is None we feed raw z0.
        if self.use_z0_in_gate and self.z0_embed_dim is not None:
            self.z0_embed = nn.Linear(n_z, int(self.z0_embed_dim))
            z0_dim_out = int(self.z0_embed_dim)
        else:
            self.z0_embed = None
            z0_dim_out = n_z

        # Gate input dimension.
        # gate_in = 1 + 2 * self.n_tau_freqs                     # tau features
        gate_in = 0
        if self.use_t_in_gate:
            gate_in += 1 + 2 * self.n_tau_freqs
        if self.use_param_in_gate:                          # tau features
            gate_in += self.n_p * (1 + 2 * self.n_param_freqs) # p features
        if self.use_z_in_gate:
            gate_in += n_z                                     # raw z(t)
        if self.use_z0_in_gate:
            gate_in += z0_dim_out                              # z0 (maybe embedded)
        self.gate_in_dim = gate_in

        self.gate = nn.Sequential(
            nn.Linear(gate_in, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, gate_hidden), nn.Tanh(),
            nn.Linear(gate_hidden, K),
        )

    # ------------------------------------------------------------------ Xi access

    @property
    def Xi_effective(self):
        if self.use_mask:
            return self.Xi * self.mask
        return self.Xi

    # ------------------------------------------------------------------ library

    def _base_library(self, z):
        feats = []
        if self.include_bias:
            feats.append(
                torch.ones(*z.shape[:-1], 1, device=z.device, dtype=z.dtype)
            )
        feats.append(z)
        if self.poly_order >= 2:
            quads = [(z[..., i] * z[..., j]).unsqueeze(-1)
                     for i in range(self.n_z) for j in range(i, self.n_z)]
            feats.append(torch.cat(quads, dim=-1))
        return torch.cat(feats, dim=-1)  # [B, T, L_base]

    def library(self, z, p=None):
        base = self._base_library(z)
        if not self.use_param_in_lib:
            return base
        if p is None:
            raise ValueError("use_param_in_lib=True but p was not provided.")
        feats = [base, p]
        for ip in range(self.n_p):
            feats.append(p[..., ip:ip + 1] * base)
        return torch.cat(feats, dim=-1)  # [B, T, L]

    # ------------------------------------------------------------------ gating

    def _gate_input(self, tau, p=None, z=None, z0=None):
        # feats = [fourier_features(tau, self.n_tau_freqs)]
        feats = []
        if self.use_t_in_gate:                # was unconditional
            feats.append(fourier_features(tau, self.n_tau_freqs))
        if self.use_param_in_gate:
            if p is None:
                raise ValueError(
                    "use_param_in_gate=True but p was not provided."
                )
            feats.append(fourier_features(p, self.n_param_freqs))
        if self.use_z_in_gate:
            if z is None:
                raise ValueError(
                    "use_z_in_gate=True but z was not provided."
                )
            feats.append(z)
        if self.use_z0_in_gate:
            if z0 is None:
                raise ValueError(
                    "use_z0_in_gate=True but z0 was not provided."
                )
            # Broadcast z0 to the same leading shape as tau if needed.
            # Accept z0 of shape [..., n_z]; if its time dim is 1 or absent,
            # broadcast against tau's shape.
            if z0.dim() == tau.dim() - 1:
                # e.g. tau is [B, T, 1] and z0 is [B, n_z]
                z0 = z0.unsqueeze(-2)
            if z0.shape[-2] == 1 and tau.shape[-2] != 1:
                z0 = z0.expand(*tau.shape[:-1], z0.shape[-1])
            z0_use = z0 if self.z0_embed is None else self.z0_embed(z0)
            feats.append(z0_use)
        return torch.cat(feats, dim=-1)

    def mix_Xi(self, tau, p=None, z=None, z0=None):
        gate_input = self._gate_input(tau, p, z, z0)
        logits = self.gate(gate_input)
        w = F.softmax(logits, dim=-1)
        Xi_use = self.Xi_effective
        Xi_eff = torch.einsum('btk,klm->btlm', w, Xi_use)
        return Xi_eff, w

    def predict_dzdt(self, z, tau, p=None, z0=None):
        Theta = self.library(z, p)
        z_for_gate = z if self.use_z_in_gate else None
        Xi_eff, w = self.mix_Xi(tau, p, z_for_gate, z0)
        dz_pred = torch.einsum('btl,btlm->btm', Theta, Xi_eff)
        return dz_pred, w

    # ------------------------------------------------------------------ losses

    @staticmethod
    def central_diff(z, dt):
        return (z[:, 2:] - z[:, :-2]) / (2.0 * dt)

    def coef_penalty(self):
        Xi = self.Xi_effective
        eps = 1e-12

        if self.sparsity == 'l2':
            return torch.mean(Xi ** 2)

        if self.sparsity == 'l1':
            # Skip the linear-in-z block of the base library so identification
            # of linear modes is unpenalized. With parameter augmentation,
            # only the first linear-in-z block is freed; p*z cross-terms are
            # still penalized.
            lin_end = (1 if self.include_bias else 0) + self.n_z
            return Xi[:, lin_end:, :].abs().sum()

        if self.sparsity == 'elastic':
            a = self.elastic_alpha
            return a * torch.mean(Xi.abs()) + (1.0 - a) * torch.mean(Xi ** 2)

        if self.sparsity == 'group_row':
            row_norm = torch.sqrt((Xi ** 2).sum(dim=-1) + eps)
            return row_norm.mean()

        if self.sparsity == 'shared':
            shared_norm = torch.sqrt((Xi ** 2).sum(dim=0) + eps)
            return shared_norm.mean()

        raise RuntimeError(f"unknown sparsity {self.sparsity}")

    def losses(self, Z, Tau, dt, P=None, fd='central', Z0=None):
        """
        Z   : [B, T, n_z]
        Tau : [B, T, 1]    -- normalized time in [0, 1].
        P   : [B, T, n_p]  -- normalized parameter values; required if
                              n_p > 0 and either use_param_in_gate or
                              use_param_in_lib is True.
        Z0  : [B, n_z] or [B, 1, n_z]  -- per-trajectory initial latent.
                              Required if use_z0_in_gate=True. If left
                              None and use_z0_in_gate=True, defaults to
                              Z[:, 0:1, :] (gradient still flows through
                              the encoder; pass Z[:, 0:1, :].detach() if
                              you want to block that).
        """
        # Resolve z0 once; will be broadcast inside _gate_input.
        if self.use_z0_in_gate:
            z0_use = Z[:, 0:1, :] if Z0 is None else Z0
        else:
            z0_use = None

        if fd == 'central':
            dz_fd = self.central_diff(Z, dt)
            z_mid = Z[:, 1:-1]
            tau_mid = Tau[:, 1:-1]
            p_mid = P[:, 1:-1] if P is not None else None
        else:
            dz_fd = (Z[:, 1:] - Z[:, :-1]) / dt
            z_mid = Z[:, :-1]
            tau_mid = Tau[:, :-1]
            p_mid = P[:, :-1] if P is not None else None

        dz_pred, w = self.predict_dzdt(z_mid, tau_mid, p_mid, z0=z0_use)
        loss_sindy = torch.mean((dz_fd - dz_pred) ** 2)
        loss_coef = self.coef_penalty()

        mean_w = w.mean(dim=(0, 1))
        loss_balance = ((mean_w - 1.0 / self.K) ** 2).sum()
        return loss_sindy, loss_coef, loss_balance, w

    # ------------------------------------------------------------------ STLS hooks

    @torch.no_grad()
    def update_mask(self, threshold, mode='absolute'):
        Xi_curr = self.Xi_effective.abs()
        if mode == 'absolute':
            keep = Xi_curr >= threshold
        elif mode == 'relative':
            scale = Xi_curr.amax(dim=(1, 2), keepdim=True)
            keep = Xi_curr >= threshold * scale
        else:
            raise ValueError(
                f"mode must be 'absolute' or 'relative', got {mode!r}"
            )
        self.mask = (self.mask * keep.float()).contiguous()
        self.Xi.data.mul_(self.mask)
        self.use_mask = True
        n_active = int(self.mask.sum().item())
        n_total = int(self.mask.numel())
        return n_active, n_total

    @torch.no_grad()
    def reset_mask(self):
        self.mask = torch.ones_like(self.mask)

    @torch.no_grad()
    def sparsity_summary(self):
        Xi = self.Xi_effective
        active = (Xi.abs() > 0).float()
        per_expert = active.sum(dim=(1, 2)).long().tolist()
        row_active_per_expert = (active.sum(dim=-1) > 0).long().tolist()
        return {
            'per_expert_active': per_expert,
            'row_active_per_expert': row_active_per_expert,
            'total_active': int(active.sum().item()),
            'total_possible': int(Xi.numel()),
            'fraction_active': float(active.mean().item()),
        }

    # ------------------------------------------------------------------ pretty print

    @torch.no_grad()
    def print_equations(self, var='z', pvar='p', tol=0.0):
        Xi = self.Xi_effective.cpu()
        base_names = []
        if self.include_bias:
            base_names.append('1')
        base_names += [f'{var}{i}' for i in range(self.n_z)]
        if self.poly_order >= 2:
            for i in range(self.n_z):
                for j in range(i, self.n_z):
                    if i == j:
                        base_names.append(f'{var}{i}^2')
                    else:
                        base_names.append(f'{var}{i}*{var}{j}')
        names = list(base_names)
        if self.use_param_in_lib:
            names += [f'{pvar}{i}' for i in range(self.n_p)]
            for ip in range(self.n_p):
                names += [f'{pvar}{ip}*{b}' for b in base_names]
        assert len(names) == self.L

        for k in range(self.K):
            print(f"--- expert {k} ---")
            for m in range(self.n_z):
                terms = []
                for l in range(self.L):
                    c = float(Xi[k, l, m])
                    if abs(c) > tol:
                        terms.append(f"{c:+.4f}*{names[l]}")
                rhs = " ".join(terms) if terms else "0"
                print(f"  d{var}{m}/dt = {rhs}")
