"""
ReadoutHead
------------
Aggregates the GRU trajectory Z_traj ∈ R^{T_max x d_z} and the final Mamba
hidden state h_final ∈ R^{d_h} into class logits.

Architecture
~~~~~~~~~~~~
1. Resample Z_traj onto a uniform time grid of n_samples points (ZOH) → Z_rs ∈ R^{n_samples x d_z}
    All n_samples points are valid for every patient, so no padding mask is needed.
2. Single-head temporal attention over Z_rs → z_agg ∈ R^{d_z}
3. Concatenate [z_agg ; h_final] ∈ R^{d_z + d_h}
4. MLP: (d_z+d_h) → 128 → ReLU → 64 → ReLU → n_classes
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def resample_trajectory(
    Z_traj:    torch.Tensor,   # (B, T_max, d_z)
    t_dyn:     torch.Tensor,   # (B, T_max)  raw timestamps (hours)
    dyn_lens:  torch.Tensor,   # (B,)        actual sequence lengths
    n_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Resample every patient's hidden-state trajectory onto a uniform time grid.

    For patient b the grid spans [t_dyn[b, 0], t_dyn[b, dyn_lens[b]-1]] with
    n_samples evenly-spaced points.  The hidden state at each grid point is
    taken via zero-order hold (ZOH): the last observed state at or before that
    time.

    Because the grid is patient-specific but always has exactly n_samples
    points, the returned tensor has no padding and no mask is required for
    downstream attention.

    Parameters
    ----------
    Z_traj    : (B, T_max, d_z)
    t_dyn     : (B, T_max)   raw observation timestamps
    dyn_lens  : (B,)         number of real (non-padded) observations
    n_samples : int          number of uniformly-spaced grid points to produce

    Returns
    -------
    Z_rs : (B, n_samples, d_z)   resampled trajectory
    t_rs : (B, n_samples)        corresponding grid times
    """
    B, T_max, d_z = Z_traj.shape
    device = Z_traj.device
    dtype  = Z_traj.dtype

    # --- time range per patient -------------------------------------------
    last_idx = (dyn_lens - 1).clamp(min=0)                            # (B,)
    t_start  = t_dyn[:, 0]                                            # (B,)
    t_end    = t_dyn[torch.arange(B, device=device), last_idx]        # (B,)

    # --- uniform grid -------------------------------------------------------
    alpha = torch.linspace(0.0, 1.0, n_samples, device=device, dtype=dtype)   # (n_samples,)
    t_rs  = t_start.unsqueeze(1) + alpha.unsqueeze(0) * (t_end - t_start).unsqueeze(1)
    # t_rs : (B, n_samples)

    # --- validity mask for real observations --------------------------------
    # valid[b, k] = True  iff  k < dyn_lens[b]
    valid = (torch.arange(T_max, device=device).unsqueeze(0)          # (1, T_max)
             < dyn_lens.unsqueeze(1)                                   # (B, 1)
             ).unsqueeze(2)                                            # (B, T_max, 1)

    # --- ZOH lookup: for each grid point n find the latest valid k
    #     such that t_dyn[b, k] <= t_rs[b, n]  -------------------------
    # t_dyn expanded: (B, T_max, 1), t_rs expanded: (B, 1, n_samples)
    at_or_before = (t_dyn.unsqueeze(2) <= t_rs.unsqueeze(1)) & valid  # (B, T_max, n_samples)

    # Assign each True position its observation index; False positions get -1
    k_idx   = torch.arange(T_max, device=device, dtype=dtype).view(1, T_max, 1)
    sel_idx = torch.where(at_or_before, k_idx, torch.full_like(k_idx, -1.0))
    best_k  = sel_idx.max(dim=1).values.long().clamp(min=0)           # (B, n_samples)

    # Gather hidden states
    best_k_exp = best_k.unsqueeze(2).expand(B, n_samples, d_z)        # (B, n_samples, d_z)
    Z_rs       = Z_traj.gather(1, best_k_exp)                         # (B, n_samples, d_z)

    return Z_rs, t_rs


class TemporalAttentionPool(nn.Module):
    """Single-head dot-product attention pooling over a time axis."""

    def __init__(self, d: int):
        super().__init__()
        self.d = d
        self.W_q = nn.Linear(d, d, bias=False)
        self.W_k = nn.Linear(d, d, bias=False)
        self.W_v = nn.Linear(d, d, bias=False)
        # Learnable global query vector (emulates CLS token)
        self.global_query = nn.Parameter(torch.randn(1, 1, d) * math.sqrt(1.0 / d))

    def forward(self, Z: torch.Tensor, seq_lens: torch.Tensor | None = None) -> torch.Tensor:
        """
        Parameters
        ----------
        Z        : (B, T, d)
        seq_lens : (B,)  optional — padded positions masked when provided.
                         Not needed when Z has already been resampled to a
                         uniform grid (all positions are valid).

        Returns
        -------
        out : (B, d)   pooled representation
        """
        B, T, d = Z.shape
        q = self.W_q(self.global_query.expand(B, 1, d))          # (B, 1, d)
        k = self.W_k(Z)                                           # (B, T, d)
        v = self.W_v(Z)                                           # (B, T, d)

        attn = torch.bmm(q, k.transpose(1, 2)) / math.sqrt(d)    # (B, 1, T)

        if seq_lens is not None:
            # Build mask: True for positions that should be ignored (beyond seq len)
            idxs = torch.arange(T, device=Z.device).unsqueeze(0)   # (1, T)
            pad_mask = idxs >= seq_lens.unsqueeze(1)                # (B, T)
            attn = attn.masked_fill(pad_mask.unsqueeze(1), float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out  = torch.bmm(attn, v).squeeze(1)                      # (B, d)
        return out


class ReadoutHead(nn.Module):
    def __init__(self, d_z: int, d_h: int, n_classes: int = 4, dropout: float = 0.2,
                 n_samples: int = 10):
        """
        Parameters
        ----------
        d_z       : GRU latent dimension
        d_h       : Mamba hidden-state dimension
        n_classes : number of output classes (default 4)
        dropout   : dropout rate in the MLP (default 0.2)
        n_samples : number of uniformly-spaced time points to resample Z_traj
                    onto before attention pooling (default 10)
        """
        super().__init__()
        self.n_samples = n_samples
        self.pool = TemporalAttentionPool(d_z)

        in_dim = d_z + d_h
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, n_classes),
        )

    def forward(
        self,
        Z_traj:  torch.Tensor,                    # (B, T_max, d_z)
        h_final: torch.Tensor,                    # (B, d_h)
        seq_lens: torch.Tensor | None = None,     # (B,)  actual lengths (required for resampling)
        t_dyn:    torch.Tensor | None = None,     # (B, T_max)  raw timestamps (required for resampling)
    ) -> torch.Tensor:                            # (B, n_classes)
        # Resample the trajectory onto a uniform time grid, then attend.
        # seq_lens and t_dyn are required for resampling; fall back to plain
        # masked pooling only if either is absent (e.g. unit tests).
        if t_dyn is not None and seq_lens is not None:
            Z_rs, _ = resample_trajectory(Z_traj, t_dyn, seq_lens, self.n_samples)
            z_agg   = self.pool(Z_rs)                        # no padding mask needed
        else:
            z_agg   = self.pool(Z_traj, seq_lens=seq_lens)   # original behaviour
        feat  = torch.cat([z_agg, h_final], dim=-1)          # (B, d_z+d_h)
        return self.mlp(feat)                                 # (B, n_classes)
