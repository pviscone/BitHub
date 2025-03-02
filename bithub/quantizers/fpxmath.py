# %%
from fxpmath import Fxp
import pandas as pd
import numpy as np

_q_modes = {
    "AP_RND_ZERO": "around",
    "AP_TRN": "floor",
    "AP_TRN_ZERO": "trunc",
}

_o_modes = {
    "AP_SAT": "saturate",
    "AP_WRAP": "wrap",
}


def get_q_mode(q_mode):
    q_mode = q_mode.upper()
    if q_mode not in _q_modes:
        raise ValueError(f"Quantization mode {q_mode} not supported")
    return _q_modes[q_mode]


def get_o_mode(o_mode):
    o_mode = o_mode.upper()
    if o_mode not in _o_modes:
        raise ValueError(f"Saturation mode {o_mode} not supported")
    return _o_modes[o_mode]


def _partial(typ, *args, **kwargs):
    def wrapper(x):
        if isinstance(x, pd.DataFrame):
            x = {col: x[col].values for col in x.columns}
        if isinstance(x, list | tuple):
            x = np.array(x)

        if isinstance(x, dict):
            res = {}
            for k, v in x.items():
                res[k] = typ(*args, **kwargs)(v)
        else:
            res = typ(*args, **kwargs)(x)

        return res

    return wrapper


def ap_fixed(nbits, int_bits, q_mode="AP_RND_ZERO", o_mode="AP_SAT"):
    quant_mode = get_q_mode(q_mode)
    overflow_mode = get_o_mode(o_mode)
    return _partial(
        Fxp,
        signed=True,
        n_word=nbits,
        n_frac=nbits - int_bits,
        rounding=quant_mode,
        overflow=overflow_mode,
    )


def ap_ufixed(nbits, int_bits, q_mode="AP_RND_ZERO", o_mode="AP_SAT"):
    quant_mode = get_q_mode(q_mode)
    overflow_mode = get_o_mode(o_mode)
    return _partial(
        Fxp,
        signed=False,
        n_word=nbits,
        n_frac=nbits - int_bits,
        rounding=quant_mode,
        overflow=overflow_mode,
    )


def ap_int(nbits):
    return _partial(
        Fxp,
        signed=True,
        n_word=nbits,
        n_frac=0,
        overflow="wrap",
    )


def ap_uint(nbits):
    return _partial(
        Fxp,
        signed=False,
        n_word=nbits,
        n_frac=0,
        overflow="wrap",
    )


def _convert(x, typ):
    if typ == "double":
        typ = "float"
    if typ == "str" or typ == "string" or typ == "bin":
        return x.bin(frac_dot=True)
    elif typ == "hex":
        return x.hex()
    elif typ.startswith("base_"):
        base = int(typ.split("_")[1])
        return x.base_repr(base)
    else:
        return eval(f"x.astype({typ})")


def convert(x, typ):
    if isinstance(x, dict):
        res = pd.DataFrame({k: _convert(v, typ) for k, v in x.items()})
    else:
        res = _convert(x, typ)
    return res


# %%
