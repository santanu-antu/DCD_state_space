"""
IrregularGRU
------------
GRU with exponential time-decay for truly irregular time series.

Why this replaces the spline-CDE
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
The previous design built a natural cubic spline assuming knots at
0, 1, 2, ..., T_max-1 (uniform index-time), then integrated a CDE over
that index-time.  Two patients with the same feature values but very different
real time-gaps between observations produced identical splines and therefore
identical CDE trajectories — only distinguishable via the appended t_norm
channel.  This is not a principled irregular-time model.

This module processes observations at their ACTUAL timestamps.  At each step
the hidden state is exponentially decayed by the real elapsed time before
being updated with the new observation.  The Mamba intervention path is also
queried at the exact observation timestamp.

Model equations
~~~~~~~~~~~~~~~
For observation k at actual time t_k:

    Δt_k     = t_k  − t_{k−1}                 (real elapsed time, hours)
    γ        = softplus(log_γ)                 (learnable per-dim decay rate)
    z̃_k     = z_{k−1} ⊙ exp(−γ · Δt_k)      (exponential decay toward zero)
    h_k      = h_path.query(t_k)              (Mamba state at actual t_k)
    inp_k    = [ y_k | m_k | h_k ]            (observed values | mask | control)
    z_k      = GRUCell(inp_k , z̃_k)

The observation mask m_k is passed as input so the GRU can learn to
discount carry-forward-imputed values versus genuinely observed ones.

Outputs
~~~~~~~
  z_final : (B, d_z)           last valid hidden state (for each sample)
  Z_traj  : (B, T_max, d_z)   hidden state at every timestep

The Z_traj (variable T_max each batch) is pooled by ReadoutHead using masked
temporal attention so padding positions are ignored.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.intervention_mamba import InterpolatedPath


class IrregularGRU(nn.Module):
    """
    Parameters
    ----------
    n_dyn  : number of dynamic feature channels
    d_h    : Mamba hidden-state dimension (for conditioning)
    d_z    : GRU hidden-state dimension (output latent dimension)
    """

    def __init__(self, n_dyn: int, d_h: int, d_z: int):
        super().__init__()
        # Input to GRU cell: [y_k (n_dyn) | m_k (n_dyn) | h(t_k) (d_h)]
        input_dim = 2 * n_dyn + d_h
        self.gru_cell = nn.GRUCell(input_size=input_dim, hidden_size=d_z)

        # Per-dimension log decay rate.
        # softplus(0) = log(2) ≈ 0.69 → effective γ ≈ 0.69 hr⁻¹ at init
        # (half-life ≈ 1 hour, a reasonable physiological prior).
        self.log_gamma = nn.Parameter(torch.zeros(d_z))

        self.d_z = d_z
        self.d_h = d_h

    def forward(
        self,
        z0:       torch.Tensor,       # (B, d_z)   initial latent from StaticEncoder
        t_dyn:    torch.Tensor,       # (B, T_max) actual timestamps (raw hours)
        Y_dyn:    torch.Tensor,       # (B, T_max, n_dyn)
        M_dyn:    torch.Tensor,       # (B, T_max, n_dyn)  1=observed, 0=imputed
        dyn_lens: torch.Tensor,       # (B,)  actual sequence lengths
        h_path:   InterpolatedPath,   # Mamba hidden-state path (raw-hour time axis)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns
        -------
        z_final : (B, d_z)
        Z_traj  : (B, T_max, d_z)
        """
        B, T_max = t_dyn.shape
        device   = z0.device
        dtype    = z0.dtype

        gamma = F.softplus(self.log_gamma)   # (d_z,)  positive decay rates

        z      = z0                          # (B, d_z)
        t_prev = t_dyn[:, 0]                 # (B,) seed with first observation time

        Z_list: list[torch.Tensor] = []

        for k in range(T_max):
            t_k = t_dyn[:, k]                                      # (B,)

            # Real elapsed time — clamp >= 0 (observations are sorted ascending)
            dt = (t_k - t_prev).clamp(min=0.0)                     # (B,)
            t_prev = t_k

            # Exponential decay of hidden state
            # z̃ ∈ (0, z_{k-1}]  when dt > 0;  z̃ = z_{k-1} when dt == 0
            decay    = torch.exp(-gamma * dt.unsqueeze(1))          # (B, d_z)
            z_decayed = z * decay                                   # (B, d_z)

            # Query Mamba intervention path at the actual observation timestamp
            h_k = h_path.query(t_k).to(device)                     # (B, d_h)

            # GRU update
            inp   = torch.cat([Y_dyn[:, k], M_dyn[:, k], h_k], dim=-1)  # (B, 2n+d_h)
            z_new = self.gru_cell(inp, z_decayed)                   # (B, d_z)

            # Mask: hold decayed state (not GRU update) for padding positions
            active = (k < dyn_lens).to(dtype).unsqueeze(1)         # (B, 1)
            z      = z_new * active + z_decayed * (1.0 - active)

            Z_list.append(z)

        Z_traj = torch.stack(Z_list, dim=1)   # (B, T_max, d_z)
        return z, Z_traj
