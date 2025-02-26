import numpy as np
from functools import wraps

from BitHub.quantizers.ap_fixed import *


def _partial(func, *args, **kwargs):
    @wraps(func)
    def wrapper(x):
        if "DataFrame" in str(type(x)):
            res = {col: func(x[col].values, *args, **kwargs) for col in x.columns}
        elif isinstance(x, dict):
            res = {col: func(x[col], *args, **kwargs) for col in x}
        elif isinstance(x, np.ndarray):
            res = func(x, *args, **kwargs)
        else:
            try:
                res = func(x, *args, **kwargs)
            except Exception as _:
                raise ValueError(f"Unsupported type: {type(x)}")
        return res
    return wrapper

def get_converter(typ, *args, **kwargs):
    func = eval(typ)
    return _partial(func, *args, **kwargs)