"""
collate_fn for ICUStreamsDataset.

Pads variable-length sequences to the maximum length in the batch and
stacks everything into batched tensors.

All timestamps (t_dyn, t_int) are kept in their original raw-hour units
so that the IrregularGRU and InterventionMamba can use physically meaningful
Δt values.

max_seq_len (optional): if provided, each patient's dynamic sequence is
truncated to the *last* max_seq_len observations before padding. This
keeps the observations closest to extubation (most predictive) and prevents
outlier patients with thousands of steps from dominating batch T_max.
"""

import torch
from functools import partial


def collate_fn(batch: list[dict], max_seq_len: int | None = None) -> dict:
    """Collate a list of ICUStreamsDataset samples into batched tensors.

    Parameters
    ----------
    batch       : list of dicts from ICUStreamsDataset.__getitem__
    max_seq_len : if set, truncates each patient's dynamic sequence to the
                  last max_seq_len timesteps (closest to extubation).

    Returns
    dict with keys:
        pid        : list[str]
        S          : (B, n_static)
        # dynamic stream
        t_dyn      : (B, T_max)          actual timestamps (raw hours, padded)
        Y_dyn      : (B, T_max, n_dyn)
        M_dyn      : (B, T_max, n_dyn)
        dyn_lens   : (B,)                actual sequence length per sample
        # intervention stream
        t_int      : (B, K_max)          actual timestamps (raw hours, padded)
        U_int      : (B, K_max, n_int)
        int_lens   : (B,)
        # targets
        y          : (B,)   continuous target
        y_cls      : (B,)   integer class label
    """
    # Apply max_seq_len truncation (keep last N steps = closest to extubation)
    if max_seq_len is not None:
        truncated = []
        for b in batch:
            T = b["t_dyn"].shape[0]
            if T > max_seq_len:
                start = T - max_seq_len
                b = dict(b)   # shallow copy so we don't mutate dataset cache
                b["t_dyn"] = b["t_dyn"][start:]
                b["Y_dyn"] = b["Y_dyn"][start:]
                b["M_dyn"] = b["M_dyn"][start:]
            truncated.append(b)
        batch = truncated

    B = len(batch)

    # gather lengths
    dyn_lens = torch.tensor([b["t_dyn"].shape[0] for b in batch], dtype=torch.long)
    int_lens = torch.tensor([b["t_int"].shape[0] for b in batch], dtype=torch.long)
    T_max = max(int(dyn_lens.max().item()), 2)  # torchcde requires >= 2 knots
    K_max = int(int_lens.max().item()) if int_lens.max() > 0 else 0

    n_dyn = batch[0]["Y_dyn"].shape[1]
    # U_int is always 2D (K, n_int) from dataset.py, even when K==0.
    # Using numel() as a guard was wrong for empty tensors like (0, 5).
    n_int = batch[0]["U_int"].shape[-1]

    # allocate padded tensors 
    S     = torch.stack([b["S"]     for b in batch])              # (B, n_static)
    y     = torch.stack([b["y"]     for b in batch])              # (B,)
    y_cls = torch.stack([b["y_cls"] for b in batch])              # (B,)

    t_dyn = torch.zeros(B, T_max,         dtype=torch.float32)
    Y_dyn = torch.zeros(B, T_max, n_dyn,  dtype=torch.float32)
    M_dyn = torch.zeros(B, T_max, n_dyn,  dtype=torch.float32)

    t_int = torch.zeros(B, K_max,         dtype=torch.float32) if K_max > 0 else torch.zeros(B, 1)
    U_int = torch.zeros(B, K_max, n_int,  dtype=torch.float32) if K_max > 0 else torch.zeros(B, 1, n_int)

    for i, b in enumerate(batch):
        T = dyn_lens[i].item()
        t_dyn[i, :T]    = b["t_dyn"]
        Y_dyn[i, :T]    = b["Y_dyn"]
        M_dyn[i, :T]    = b["M_dyn"]

        K = int_lens[i].item()
        if K > 0:
            t_int[i, :K]    = b["t_int"]
            U_int[i, :K]    = b["U_int"]

    pids = [b["pid"] for b in batch]

    return {
        "pid":      pids,
        "S":        S,
        "t_dyn":    t_dyn,
        "Y_dyn":    Y_dyn,
        "M_dyn":    M_dyn,
        "dyn_lens": dyn_lens,
        "t_int":    t_int,
        "U_int":    U_int,
        "int_lens": int_lens,
        "y":        y,
        "y_cls":    y_cls,
    }
