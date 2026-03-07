"""
Time-aware Mamba-style SSM that evolves a hidden state h(t) driven by
sparse, irregularly-timed medication events.

Architecture:

For each medication event at time t_k with dose u_k ∈ R^{n_int}:

  1. Project dose:   ũ_k = LayerNorm(Linear(n_int → d_u))
  2. Compute input-dependent parameters (selectivity, à la Mamba):
       ΔB_k, ΔC_k, δ_k = Linear(d_u → d_h + d_h + 1)
  3. Effective SSM parameters for this step:
       Â   = diag(exp(A_log))          base (learnable, kept negative)
       B̂_k = B + ΔB_k                  input-dependent input matrix (d_h × d_u)
       Ĉ_k = C + ΔC_k                  input-dependent output vector (d_h,)
  4. ZOH discretization with Δt_k = t_k − t_{k−1}:
       Ā_k = exp(diag(Â) · Δt_k · softplus(δ_k))   (d_h,)
       B̄_k = (Ā_k − 1) / diag(Â) ⊙ B̂_k · ũ_k      (d_h,)
  5. State update:
       h_k = Ā_k ⊙ h_{k−1} + B̄_k

After processing all K events the module stores {(t_k, h_k)} and exposes
`query(t)` for linear interpolation at arbitrary query times.

For patients with no medication events the hidden state stays at h₀.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class InterventionMamba(nn.Module):
    def __init__(self, n_int: int, d_u: int, d_h: int):
        """
        Parameters
        ----------
        n_int : number of medication channels (e.g. 5)
        d_u   : internal medication projection dimension
        d_h   : hidden state (SSM state) dimension
        """
        super().__init__()
        self.d_h  = d_h
        self.d_u  = d_u
        self.n_int = n_int

        # Input projection 
        self.input_proj = nn.Sequential(
            nn.Linear(n_int, d_u),
            nn.LayerNorm(d_u),
        )

        # Base SSM parameters
        # A: diagonal, kept negative via log parameterisation
        self.A_log = nn.Parameter(torch.zeros(d_h))          # log|A|, init -> A=-1
        nn.init.uniform_(self.A_log, -1.0, -0.1)

        # B: (d_h, d_u)  base input matrix
        self.B = nn.Parameter(torch.empty(d_h, d_u))
        nn.init.xavier_uniform_(self.B)

        # C: (d_h,)  base output probe vector
        self.C = nn.Parameter(torch.zeros(d_h))

        # D: scalar skip connection
        self.D = nn.Parameter(torch.ones(1))

        # Input-dependent (selective) corrections 
        # maps ũ_k → [ΔB (d_h*d_u), ΔC (d_h), δ (1)]
        self.selective = nn.Linear(d_u, d_h * d_u + d_h + 1, bias=True)

    
    def _zoh_step(
        self,
        h: torch.Tensor,     # (B, d_h)
        u_proj: torch.Tensor, # (B, d_u)  projected dose
        dt: torch.Tensor,    # (B,)       time gap
    ) -> torch.Tensor:       # (B, d_h)  new hidden state
        """Single ZOH SSM step."""
        B_sz  = h.shape[0]
        d_h   = self.d_h
        d_u   = self.d_u

        # Selective corrections
        sel   = self.selective(u_proj)                       # (B, d_h*d_u + d_h + 1)
        dB    = sel[:, :d_h * d_u].reshape(B_sz, d_h, d_u)  # (B, d_h, d_u)
        dC    = sel[:, d_h * d_u: d_h * d_u + d_h]          # (B, d_h)
        delta = F.softplus(sel[:, -1])                       # (B,)  > 0

        # Effective per-sample B and C
        B_eff = self.B.unsqueeze(0) + dB                     # (B, d_h, d_u)
        # C_eff = self.C + dC                                # unused in state update

        # Base diagonal A (negative)
        A = -torch.exp(self.A_log)                            # (d_h,) negative

        # Time-scaled Δt with input-dependent multiplier δ
        dt_scaled = dt * delta                                # (B,)
        # Ā_k = exp(A * dt_scaled)  element-wise: (B, d_h)
        A_bar = torch.exp(A.unsqueeze(0) * dt_scaled.unsqueeze(1))  # (B, d_h)

        # B̄_k = (Ā_k - 1) / A * B_eff * ũ
        # For diagonal A: (Ā - 1)/A  ∈ (B, d_h)
        inv_A  = 1.0 / A.unsqueeze(0)                        # (B, d_h)
        scale  = (A_bar - 1.0) * inv_A                       # (B, d_h)
        Bu     = torch.einsum("bhi,bi->bh", B_eff, u_proj)   # (B, d_h)
        B_bar  = scale * Bu                                   # (B, d_h)

        h_new  = A_bar * h + B_bar                           # (B, d_h)
        return h_new


    def forward(
        self,
        h0:       torch.Tensor,   # (B, d_h)
        t_int:    torch.Tensor,   # (B, K_max)  padded medication times
        U_int:    torch.Tensor,   # (B, K_max, n_int)
        int_lens: torch.Tensor,   # (B,)  actual number of events
    ) -> tuple[torch.Tensor, list]:
        """
        Process all medication events and return:
          h_final  : (B, d_h)   hidden state after last event
          path     : list of (t_k_tensor, h_k) for interpolation  [length K_max+1]
                     where index 0 is the initial state before any event.
        """
        B     = h0.shape[0]
        K_max = t_int.shape[1]

        # Store (time, hidden) path for later interpolation by the CDE module.
        # We use lists of tensors; will be converted to padded tensors after.
        # path[i] = (scalar time, (B, d_h) hidden)
        # We accumulate all (t, h) pairs; entries beyond int_lens[b] are masked.

        h = h0.clone()           # (B, d_h)

        # Initial anchor: we don't have a well-defined t_0 for the intervention
        # path — use the first event time (or a sentinel if no events).
        # Callers that query before the first event get h0 by clamping.
        all_t = [None]   # placeholder for t_0; filled below
        all_h = [h0]     # h before any event

        for k in range(K_max):
            t_k  = t_int[:, k]                     # (B,)
            u_k  = U_int[:, k, :]                  # (B, n_int)

            # Projected dose
            u_proj = self.input_proj(u_k)           # (B, d_u)

            # Time gap: use difference from previous event (or 0 for first event)
            if k == 0:
                dt = torch.zeros(B, device=h.device, dtype=h.dtype)
            else:
                dt = (t_k - t_int[:, k - 1]).clamp(min=0.0)   # (B,)

            # New hidden state
            h_new = self._zoh_step(h, u_proj, dt)  # (B, d_h)

            # Mask: only apply update for samples that have >= k+1 events
            active = (k < int_lens).float().unsqueeze(1)        # (B, 1)
            h = h_new * active + h * (1.0 - active)

            all_t.append(t_k)
            all_h.append(h.clone())

        # Set sentinel t_0 = first event time (or zeros if no events)
        if K_max > 0:
            all_t[0] = t_int[:, 0]   # anchor at first event time
        else:
            all_t[0] = torch.zeros(B, device=h0.device)

        # Build InterpolatedPath object
        path = InterpolatedPath(all_t, all_h)

        return h, path



class InterpolatedPath:
    """
    Piecewise-linear interpolation of (time, hidden-state) sequence.

    query(t)  → (B, d_h)  linearly interpolated hidden state at scalar time t.
    Clamps to the first/last recorded state outside the observation window.
    """

    def __init__(
        self,
        times:  list[torch.Tensor | None],   # length K+1, each (B,)
        states: list[torch.Tensor],           # length K+1, each (B, d_h)
    ):
        # Filter out None entries (shouldn't happen after forward, but safe)
        valid = [(t, s) for t, s in zip(times, states) if t is not None]
        # Stack into contiguous tensors so query() is a single vectorised op
        # with no Python-level loop — eliminates per-segment kernel-launch overhead.
        self.times_mat  = torch.stack([v[0] for v in valid], dim=0)   # (K+1, B)
        self.states_mat = torch.stack([v[1] for v in valid], dim=0)   # (K+1, B, d_h)

    @property
    def states(self):   # kept for adjoint_params compatibility
        return list(self.states_mat.unbind(0))

    def query(self, t: float | torch.Tensor) -> torch.Tensor:
        """
        Vectorised piecewise-linear interpolation — no Python loop.

        Parameters
        ----------
        t : scalar float or 0-d/1-d tensor

        Returns
        -------
        (B, d_h)
        """
        dev   = self.states_mat.device
        dtype = self.states_mat.dtype
        B     = self.states_mat.shape[1]

        # Materialise query as (B,)
        if not isinstance(t, torch.Tensor):
            t_q = torch.full((B,), float(t), device=dev, dtype=dtype)
        else:
            t_q = t.to(dev).to(dtype)
            if t_q.ndim == 0:
                t_q = t_q.expand(B)

        if self.times_mat.shape[0] == 1:
            return self.states_mat[0]

        t_lo  = self.times_mat[:-1]    # (K, B)
        t_hi  = self.times_mat[1:]     # (K, B)
        h_lo  = self.states_mat[:-1]   # (K, B, d_h)
        h_hi  = self.states_mat[1:]    # (K, B, d_h)

        t_q_e = t_q.unsqueeze(0)                                        # (1, B)
        span  = (t_hi - t_lo).clamp(min=1e-8)                          # (K, B)
        alpha = ((t_q_e - t_lo) / span).clamp(0.0, 1.0).unsqueeze(-1) # (K, B, 1)
        h_seg = h_lo + alpha * (h_hi - h_lo)                           # (K, B, d_h)

        # Active segment: t_lo[k] <= t_q < t_hi[k].
        # Last segment also catches t_q >= t_hi[-1] (right extrapolation).
        in_seg = (t_q_e >= t_lo) & (t_q_e < t_hi)    # (K, B)
        in_seg[-1] = in_seg[-1] | (t_q >= t_hi[-1])  # right boundary

        # Weighted sum — each column has exactly one True entry.
        h_out = (in_seg.float().unsqueeze(-1) * h_seg).sum(0)  # (B, d_h)

        # Left extrapolation: t_q < first t_lo
        left_mask = (t_q < t_lo[0]).unsqueeze(-1)   # (B, 1)
        h_out = torch.where(left_mask, h_lo[0], h_out)

        return h_out
