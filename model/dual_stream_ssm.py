"""
DualStreamSSM
-------------
Top-level model wiring together:

  StaticEncoder  →  (h₀, z₀)
  InterventionMamba(h₀, medications)            →  (h_final, h_path)
  IrregularGRU *or* ODERNNDynamic (z₀, physiology, h_path, t_dyn) → (z_final, Z_traj)
  ReadoutHead(Z_traj, h_final, dyn_lens)        →  logits (B, 4)

Forward signature
~~~~~~~~~~~~~~~~~
  logits = model(S, t_dyn, Y_dyn, M_dyn, t_int, U_int, dyn_lens, int_lens)

Inputs (all batched, B = batch size)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
  S         : (B, n_static)
  t_dyn     : (B, T_max)          actual ICU timestamps (raw hours)
  Y_dyn     : (B, T_max, n_dyn)   dynamic features
  M_dyn     : (B, T_max, n_dyn)   observation mask (1=observed, 0=imputed)
  t_int     : (B, K_max)          medication event times (raw hours, padded)
  U_int     : (B, K_max, n_int)   medication doses (padded)
  dyn_lens  : (B,)                actual dynamic sequence lengths
  int_lens  : (B,)                actual number of medication events
"""

import torch
import torch.nn as nn

from model.static_encoder     import StaticEncoder
from model.intervention_mamba import InterventionMamba
from model.irregular_gru      import IrregularGRU
from model.ode_rnn_dynamic    import ODERNNDynamic
from model.readout             import ReadoutHead


class DualStreamSSM(nn.Module):
    def __init__(
        self,
        n_static:       int   = 5,
        n_dyn:          int   = 20,
        n_int:          int   = 5,
        d_h:            int   = 64,
        d_u:            int   = 32,
        d_z:            int   = 128,
        n_classes:      int   = 4,
        dropout:        float = 0.2,
        dynamic_module: str   = "irr_gru",   # "irr_gru" | "ode_rnn"
        ode_hidden:     int   = 64,
        ode_layers:     int   = 2,
        # kept as kwargs so old configs don't break
        **kwargs,
    ):
        super().__init__()
        self.static_enc  = StaticEncoder(n_static, d_h, d_z)
        self.int_mamba   = InterventionMamba(n_int, d_u, d_h)

        if dynamic_module == "irr_gru":
            self.dyn_module = IrregularGRU(n_dyn=n_dyn, d_h=d_h, d_z=d_z)
        elif dynamic_module == "ode_rnn":
            self.dyn_module = ODERNNDynamic(
                n_dyn=n_dyn, d_h=d_h, d_z=d_z,
                ode_hidden=ode_hidden, ode_layers=ode_layers,
            )
        else:
            raise ValueError(f"Unknown dynamic_module: {dynamic_module!r}. "
                             "Choose 'irr_gru' or 'ode_rnn'.")

        self.readout     = ReadoutHead(d_z, d_h, n_classes, dropout=dropout)

    # ─────────────────────────────────────────────────────────────────────────
    def forward(
        self,
        S:        torch.Tensor,   # (B, n_static)
        t_dyn:    torch.Tensor,   # (B, T_max)   raw hours
        Y_dyn:    torch.Tensor,   # (B, T_max, n_dyn)
        M_dyn:    torch.Tensor,   # (B, T_max, n_dyn)
        t_int:    torch.Tensor,   # (B, K_max)   raw hours
        U_int:    torch.Tensor,   # (B, K_max, n_int)
        dyn_lens: torch.Tensor,   # (B,)
        int_lens: torch.Tensor,   # (B,)
    ) -> torch.Tensor:            # (B, n_classes)

        # 1. Static features → initial states
        h0, z0 = self.static_enc(S)                                  # (B,d_h), (B,d_z)

        # 2. Intervention stream: time-aware Mamba over medication events
        h_final, h_path = self.int_mamba(h0, t_int, U_int, int_lens)

        # 3. Dynamic stream: irregular-time module conditioned on h(t)
        _z_final, Z_traj = self.dyn_module(z0, t_dyn, Y_dyn, M_dyn, dyn_lens, h_path)

        # 4. Readout: masked-attention pool over trajectory + final Mamba state
        logits = self.readout(Z_traj, h_final, seq_lens=dyn_lens)    # (B, n_classes)

        return logits

    # ─────────────────────────────────────────────────────────────────────────
    @staticmethod
    def from_config(cfg: dict) -> "DualStreamSSM":
        m = cfg["model"]
        # solver section is optional (kept for backward compat with old configs)
        s = cfg.get("solver", {})
        return DualStreamSSM(
            n_static       = m.get("n_static", 5),
            n_dyn          = m.get("n_dyn",    20),
            n_int          = m.get("n_int",     5),
            d_h            = m["d_h"],
            d_u            = m["d_u"],
            d_z            = m["d_z"],
            n_classes      = m["n_classes"],
            dropout        = m.get("dropout",        0.2),
            dynamic_module = m.get("dynamic_module", "irr_gru"),
            ode_hidden     = m.get("ode_hidden",     64),
            ode_layers     = m.get("ode_layers",     2),
        )
