from mindspore import nn
from mindspore.common.api import _pynative_executor


def wrap_recompute_cell(network: nn.Cell):
    if hasattr(network, "_recompute_cell"):
        network._mindone_wrapped_recompute_cell = network._recompute_cell

    for _, child in network.name_cells().items():
        wrap_recompute_cell(child)


def _recompute_cell(self):
    if _pynative_executor.requires_grad():
        return getattr(self, "_mindone_wrapped_recompute_cell", None)
    return None


def monkey_patch_recompute_cell():
    setattr(nn.Cell, "_recompute_cell", property(_recompute_cell))
