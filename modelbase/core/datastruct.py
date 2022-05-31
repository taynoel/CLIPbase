from types import SimpleNamespace
from typing import Dict
from torch import Tensor


class IterData(SimpleNamespace):
    data: Dict[str, Tensor]
    output: Dict[str, Tensor]
    loss: Dict[str, Tensor]
    loss_for_iter: Tensor


class EpochData(SimpleNamespace):
    loss: dict
    loss_for_epoch: float
