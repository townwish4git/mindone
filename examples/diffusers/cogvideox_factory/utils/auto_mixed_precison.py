from mindspore import nn
from mindspore.common import dtype as mstype
from mindspore.train.amp import AMP_BLACK_LIST, _OutputTo16

from mindone.diffusers.models.normalization import LayerNorm

OUR_FP32_WHITE_LIST = AMP_BLACK_LIST + [LayerNorm]


def auto_mixed_precision_rewrite(network, model_dtype, amp_dtype, white_list=OUR_FP32_WHITE_LIST):
    """
    Retain the data type (dtype) of most subcells, and perform auto-mixed-precision computation
    on the subcells corresponding to categories in the `white_list` according to `amp_dtype`.

    Parameters:
        - network (mindspore.nn.Cell): The neural network model.
        - model_dtype (mindspore.Type): The data type for the majority of subcells.
        - amp_dtype (mindspore.Type): The data type for automatic mixed precision computation.
        - white_list (List[mindspore.nn.Cell]): A list of categories for which subcells will use amp_dtype for mixed precision.
    """
    if model_dtype in (mstype.bfloat16, mstype.float16) or model_dtype == amp_dtype or not white_list:
        return network

    cells = network.name_cells()
    change = False
    for name in cells:
        subcell = cells[name]
        if subcell == network:
            continue
        if isinstance(subcell, tuple(white_list)):
            network._cells[name] = _OutputTo16(subcell.to_float(amp_dtype), model_dtype)
            change = True
        else:
            auto_mixed_precision_rewrite(subcell, model_dtype, amp_dtype, white_list)
    if isinstance(network, nn.SequentialCell) and change:
        network.cell_list = list(network.cells())
    return network
