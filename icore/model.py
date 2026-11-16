"""Optional named array container for adjacent-stage affine models."""

from typing import NamedTuple, Any


class ChainQP(NamedTuple):
    left: Any
    right: Any
    offset: Any
    pair_left: Any
    pair_right: Any
    pair_offset: Any
    local_matrix: Any
    local_offset: Any
    center: Any
    hessian: Any
    linear: Any
    l1_weight: Any
    free: Any
    fixed: Any
