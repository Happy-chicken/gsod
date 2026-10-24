"""Checkpoint interface, same file layout as uORF's BaseModel.

Files written into save_dir for a given suffix (e.g. 'latest' or 'iter_1000'):
    {suffix}_net_{name}.pth          one per network
    {suffix}_optimizer_{i}.pth       one per optimizer
    {suffix}_lr_scheduler_{i}.pth    one per scheduler
"""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import torch
from torch import nn


def save_networks(
    save_dir: str,
    suffix: str,
    networks: Mapping[str, nn.Module],
    optimizers: Sequence[torch.optim.Optimizer] = (),
    schedulers: Sequence = (),
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    for name, net in networks.items():
        torch.save(net.state_dict(), os.path.join(save_dir, f"{suffix}_net_{name}.pth"))
    for i, opt in enumerate(optimizers):
        torch.save(
            opt.state_dict(), os.path.join(save_dir, f"{suffix}_optimizer_{i}.pth")
        )
    for i, sch in enumerate(schedulers):
        torch.save(
            sch.state_dict(), os.path.join(save_dir, f"{suffix}_lr_scheduler_{i}.pth")
        )


def load_networks(
    save_dir: str,
    suffix: str,
    networks: Mapping[str, nn.Module],
    optimizers: Sequence[torch.optim.Optimizer] = (),
    schedulers: Sequence = (),
    device: str | torch.device = "cuda",
) -> dict[str, object]:
    """Load weights with strict=False, as uORF does. Returns the load result per network.

    Missing files are reported and skipped; optimizer and scheduler state is
    restored only when the file exists.
    """
    result: dict[str, object] = {}
    for name, net in networks.items():
        path = os.path.join(save_dir, f"{suffix}_net_{name}.pth")
        if not os.path.isfile(path):
            print(f"not found: {path}, skip {name}")
            continue
        state = torch.load(path, map_location=device)
        result[name] = net.load_state_dict(state, strict=False)
    for i, opt in enumerate(optimizers):
        path = os.path.join(save_dir, f"{suffix}_optimizer_{i}.pth")
        if os.path.isfile(path):
            opt.load_state_dict(torch.load(path, map_location=device))
    for i, sch in enumerate(schedulers):
        path = os.path.join(save_dir, f"{suffix}_lr_scheduler_{i}.pth")
        if os.path.isfile(path):
            sch.load_state_dict(torch.load(path, map_location=device))
    return result
