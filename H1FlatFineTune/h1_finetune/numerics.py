"""CPU-testable numerical guards; no simulator imports."""
from collections.abc import Mapping
import torch
from tensordict import TensorDictBase


def finite(*values):
    """Reject NaN/Inf in dense tensors, scalar metrics and nested containers."""
    for value in values:
        # Dense tensors expose .values(), but that method requires sparse layout.
        if isinstance(value, torch.Tensor):
            tensor = value
        elif isinstance(value, (Mapping, TensorDictBase)):
            finite(*value.values())
            continue
        elif isinstance(value, (list, tuple)):
            finite(*value)
            continue
        else:
            tensor = torch.as_tensor(value)
        if not torch.isfinite(tensor).all():
            raise FloatingPointError("NaN/Inf in rollout, action, reward, loss or model")
