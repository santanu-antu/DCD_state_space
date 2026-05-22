"""
StaticEncoder: Maps the static patient features S in R^{n_static} into two initial states:
  h_theta in R^{d_h}: seed for the intervention Mamba stream
  z_theta in R^{d_z}: seed for the dynamic stream
"""

import torch
import torch.nn as nn


class StaticEncoder(nn.Module):
    def __init__(self, n_static: int, d_h: int, d_z: int):
        """
        Parameters:
        n_static : input dimension (number of static features, typically 5)
        d_h      : Mamba hidden-state dimension
        d_z      : ODE-RNN/IrregularGRU latent dimension
        """
        super().__init__()

        hidden = max(64, (n_static + d_h) // 2 * 2)   # at least 64

        # Branch -> h_theta
        self.h_branch = nn.Sequential(
            nn.Linear(n_static, hidden),
            nn.LayerNorm(hidden),
            nn.ReLU(),
            nn.Linear(hidden, d_h),
            nn.Tanh(),  # bounded init keeps SSM stable
        )

        # Branch -> z_theta
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
        h_theta : (B, d_h)
        z_theta : (B, d_z)
        """
        return self.h_branch(S), self.z_branch(S)
