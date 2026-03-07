"""
StaticEncoder

Maps the static patient features S ∈ R^{n_static} into two initial states:

  h₀ ∈ R^{d_h}   — seed for the intervention Mamba stream
  z₀ ∈ R^{d_z}   — seed for the dynamic CDE stream
"""

import torch
import torch.nn as nn


class StaticEncoder(nn.Module):
    def __init__(self, n_static: int, d_h: int, d_z: int):
        """
        Parameters:
        n_static : input dimension (number of static features, typically 5)
        d_h      : Mamba hidden-state dimension
        d_z      : CDE latent dimension
        """
        super().__init__()

        hidden = max(64, (n_static + d_h) // 2 * 2)   # at least 64

        # Branch → h₀
        self.h_branch = nn.Sequential(
            nn.Linear(n_static, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_h),
            nn.Tanh(),  # bounded init keeps SSM stable
        )

        # Branch → z₀
        self.z_branch = nn.Sequential(
            nn.Linear(n_static, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_z),
            nn.Tanh(),
        )

    def forward(self, S: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Parameters
        S : (B, n_static)

        Returns
        h0 : (B, d_h)
        z0 : (B, d_z)
        """
        return self.h_branch(S), self.z_branch(S)
