"""
ODERNNDynamic
-------------
Drop-in replacement for IrregularGRU that uses an ODE to evolve the hidden state between observations instead of exponential decay.

Architecture:

Between observations k-1 and k (real time gap Δt_k hours):

    dz/dt = f_θ(z)        (autonomous ODE — learned MLP)

integrated for Δt_k real hours via a single-step RK4 (pure tensor ops), then a GRU update at the observation:

    h_k  = h_path.query(t_k)
    z_k  = GRUCell([y_k | m_k | h_k], z_ode)

The ODE uses the "normalised-time" trick to batch-integrate different Δt:
instead of integrating from 0 to Δt_k (variable per sample) we integrate
from 0 to 1 and scale the ODE output by Δt_k:

    dz/ds = f_θ(z) · Δt_k     (s ∈ [0, 1])

with the solution evaluated via a single RK4 step (h=1 over [0,1]).
This is pure PyTorch tensors — no torchdiffeq overhead per step.

Interface:
Identical to IrregularGRU: forward() takes and returns the same tensors
so DualStreamSSM can swap between the two with a single config flag.
"""

import torch
import torch.nn as nn

from model.intervention_mamba import InterpolatedPath


class _ODEFunc(nn.Module):
    """Autonomous ODE function  dz/dt = MLP(z)."""

    def __init__(self, d_z: int, hidden: int, n_layers: int):
        super().__init__()
        layers: list[nn.Module] = [nn.Linear(d_z, hidden), nn.Tanh()]
        for _ in range(n_layers - 1):
            layers += [nn.Linear(hidden, hidden), nn.Tanh()]
        layers += [nn.Linear(hidden, d_z)]
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class ODERNNDynamic(nn.Module):
    """
    ODE-RNN dynamic stream with Mamba intervention conditioning.

    Parameters

    n_dyn      : number of dynamic input channels
    d_h        : Mamba hidden dimension (context injected at each step)
    d_z        : ODE / GRU hidden dimension
    ode_hidden : MLP hidden width inside ODE function  (default 64)
    ode_layers : MLP depth inside ODE function, not counting output (default 2)
    """

    def __init__(
        self,
        n_dyn:      int = 20,
        d_h:        int = 64,
        d_z:        int = 128,
        ode_hidden: int = 64,
        ode_layers: int = 2,
    ):
        super().__init__()
        input_dim      = 2 * n_dyn + d_h            # [y | m | h(t)]
        self.gru_cell  = nn.GRUCell(input_size=input_dim, hidden_size=d_z)
        self._ode_func = _ODEFunc(d_z, ode_hidden, ode_layers)
        self.d_z = d_z
        self.d_h = d_h

    def _evolve(self, z: torch.Tensor, dt: torch.Tensor) -> torch.Tensor:
        """
        Single-step RK4: integrates dz/ds = f_θ(z) · dt  over s in [0, 1].
        Pure tensor ops: no external solver overhead.

        # Parameters
        z  : (B, d_z)  current hidden state
        dt : (B,)      real time gaps in hours

        # Returns
        z_ode : (B, d_z)
        """
        dt = dt.unsqueeze(1)          # (B, 1) — broadcast over d_z

        def f(z_: torch.Tensor) -> torch.Tensor:
            return self._ode_func(z_) * dt

        k1 = f(z)
        k2 = f(z + 0.5 * k1)
        k3 = f(z + 0.5 * k2)
        k4 = f(z + k3)
        return z + (k1 + 2.0 * k2 + 2.0 * k3 + k4) / 6.0

    def sample_uniform_states(
        self,
        Z_obs:      torch.Tensor,   # (B, T_max, d_z) post-GRU states at observation times
        t_dyn:      torch.Tensor,   # (B, T_max)      observation timestamps (hours)
        dyn_lens:   torch.Tensor,   # (B,)            valid observation counts
        n_samples:  int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Sample the continuous ODE-RNN latent state at uniformly-spaced times.

        For each patient b:
          - Build a uniform grid over [t_dyn[b,0], t_dyn[b,last_valid]]
          - For each grid time t_q, find the latest observation index j with
            t_dyn[b,j] <= t_q
          - Start from z_j (post-GRU state at j) and evolve forward by
            dt = t_q - t_dyn[b,j] using the ODE to obtain z(t_q)

        Returns
        -------
        Z_uniform : (B, n_samples, d_z)
        t_uniform : (B, n_samples)
        """
        B, T_max, d_z = Z_obs.shape
        device = Z_obs.device
        dtype  = Z_obs.dtype

        last_idx = (dyn_lens - 1).clamp(min=0)                                # (B,)
        t_start  = t_dyn[:, 0]                                                 # (B,)
        t_end    = t_dyn[torch.arange(B, device=device), last_idx]             # (B,)

        alpha = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)   # (n_samples,)
        t_uniform = t_start.unsqueeze(1) + alpha.unsqueeze(0) * (t_end - t_start).unsqueeze(1)

        valid = (torch.arange(T_max, device=device).unsqueeze(0) < dyn_lens.unsqueeze(1)).unsqueeze(2)
        at_or_before = (t_dyn.unsqueeze(2) <= t_uniform.unsqueeze(1)) & valid      # (B, T_max, n_samples)

        k_idx = torch.arange(T_max, device=device, dtype=dtype).view(1, T_max, 1)
        sel_idx = torch.where(at_or_before, k_idx, torch.full_like(k_idx, -1.0))
        anchor_idx = sel_idx.max(dim=1).values.long().clamp(min=0)                 # (B, n_samples)

        gather_idx = anchor_idx.unsqueeze(2).expand(B, n_samples, d_z)
        z_anchor = Z_obs.gather(1, gather_idx)                                      # (B, n_samples, d_z)

        t_anchor = t_dyn.gather(1, anchor_idx)                                      # (B, n_samples)
        dt = (t_uniform - t_anchor).clamp(min=0.0)                                  # (B, n_samples)

        z_anchor_flat = z_anchor.reshape(B * n_samples, d_z)
        dt_flat = dt.reshape(B * n_samples)
        z_uniform_flat = self._evolve(z_anchor_flat, dt_flat)
        Z_uniform = z_uniform_flat.reshape(B, n_samples, d_z)

        return Z_uniform, t_uniform


    def forward(
        self,
        z0:                torch.Tensor,              # (B, d_z)
        t_dyn:             torch.Tensor,              # (B, T_max)   actual timestamps (raw hours)
        Y_dyn:             torch.Tensor,              # (B, T_max, n_dyn)
        M_dyn:             torch.Tensor,              # (B, T_max, n_dyn)
        dyn_lens:          torch.Tensor,              # (B,)
        h_path:            InterpolatedPath,          # Mamba hidden-state path
        intervention_mask: torch.Tensor | None = None,  # (B, T_max) bool — None -> continuous
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        z_final : (B, d_z)
        Z_traj  : (B, T_max, d_z)
        """
        B, T_max = t_dyn.shape
        device   = z0.device
        dtype    = z0.dtype

        z      = z0
        t_prev = t_dyn[:, 0]

        Z_list: list[torch.Tensor] = []

        for k in range(T_max):
            t_k = t_dyn[:, k]                               # (B,)
            dt  = (t_k - t_prev).clamp(min=0.0)             # (B,)
            t_prev = t_k

            # ODE evolution over real time gap (skip at first step)
            z_ode = z if k == 0 else self._evolve(z, dt)    # (B, d_z)

            # Query Mamba path at actual observation timestamp.
            # In gated mode: skip the query entirely if no sample in the batch has an intervention at this step (avoids all ZOH lookup work).
            if intervention_mask is not None:
                gate = intervention_mask[:, k]                      # (B,) bool
                if gate.any():
                    h_k = h_path.query(t_k).to(device)              # (B, d_h)
                    h_k = h_k * gate.to(dtype).unsqueeze(1)         # zero non-event
                else:
                    h_k = torch.zeros(B, self.d_h, device=device, dtype=dtype)
            else:
                h_k = h_path.query(t_k).to(device)                  # (B, d_h)

            # GRU update
            inp   = torch.cat([Y_dyn[:, k], M_dyn[:, k], h_k], dim=-1)
            z_new = self.gru_cell(inp, z_ode)                # (B, d_z)

            # Mask: hold ODE-evolved state for padding positions
            active = (k < dyn_lens).to(dtype).unsqueeze(1)  # (B, 1)
            z      = z_new * active + z_ode * (1.0 - active)

            Z_list.append(z)

        Z_traj = torch.stack(Z_list, dim=1)   # (B, T_max, d_z)
        return z, Z_traj
