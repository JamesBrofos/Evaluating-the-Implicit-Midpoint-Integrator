from typing import Any, Callable, Sequence, Tuple, Union

import numpy as np


def while_loop(cond: Callable, step: Callable, val: Any):
    while cond(val):
        val = step(val)
    return val

def for_loop(lower: int, upper: int, step: Callable, val: Any):
    for it in range(lower, upper):
        val = step(it, val)
    return val
