"""
ReadoutHead
-----------
Aggregates the GRU trajectory Z_traj ∈ ℝ^{T_max × d_z} and the final Mamba
hidden state h_final ∈ ℝ^{d_h} into class logits.

Architecture
~~~~~~~~~~~~
1. Single-head temporal attention over Z_traj (with padding mask) → z_agg ∈ ℝ^{d_z}
2. Concatenate [z_agg ; h_final] ∈ ℝ^{d_z + d_h}
3. MLP: (d_z+d_h) → 128 → ReLU → 64 → ReLU → n_classes
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


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
        seq_lens : (B,)  actual sequence lengths; padded positions are masked

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
    def __init__(self, d_z: int, d_h: int, n_classes: int = 4, dropout: float = 0.2):
        """
        Parameters
        ----------
        d_z       : GRU latent dimension
        d_h       : Mamba hidden-state dimension
        n_classes : number of output classes (default 4)
        dropout   : dropout rate in the MLP (default 0.2)
        """
        super().__init__()
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
        seq_lens: torch.Tensor | None = None,     # (B,)  actual lengths for masking
    ) -> torch.Tensor:                            # (B, n_classes)
        z_agg = self.pool(Z_traj, seq_lens=seq_lens)         # (B, d_z)
        feat  = torch.cat([z_agg, h_final], dim=-1)          # (B, d_z+d_h)
        return self.mlp(feat)                                 # (B, n_classes)
