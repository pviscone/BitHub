# %%
import os

import numpy as np
import pandas as pd
import ROOT
import array

from numbers import Number


ROOT.gInterpreter.Declare("""
template <typename T>
ROOT::VecOps::RVec<T> to_rvec(float *x, const int size_v) {
    ROOT::VecOps::RVec<T> v(size_v);
    for (int i = 0; i < size_v; i++) {
        T val = x[i];
        v[i] = val;
    }
    return v;
}
""")

hls_include_path = os.path.join(
    os.path.dirname(__file__), "../include/XilinxHeaders/include"
)
ROOT.gInterpreter.AddIncludePath(hls_include_path)
ROOT.gInterpreter.Declare("#include <ap_fixed.h>")
ROOT.gInterpreter.Declare("#include <ap_int.h>")

def AP_FIXED(nbits, int_bits, q_mode="AP_RND_CONV", o_mode="AP_SAT", N=0):
    quant_mode = getattr(ROOT, q_mode)
    overflow_mode = getattr(ROOT, o_mode)
    return ROOT.ap_fixed[nbits, int_bits, quant_mode, overflow_mode, N]

def AP_UFIXED(nbits, int_bits, q_mode="AP_RND_CONV", o_mode="AP_SAT", N=0):
    quant_mode = getattr(ROOT, q_mode)
    overflow_mode = getattr(ROOT, o_mode)
    return ROOT.ap_fixed[nbits, int_bits, quant_mode, overflow_mode, N]

def AP_INT(nbits, int_bits):
    return ROOT.ap_int[nbits, int_bits]

def AP_UINT(nbits, int_bits):
    return ROOT.ap_uint[nbits, int_bits]

def _partial(typ, *args):
    def wrapper(x):
        if isinstance(x, pd.DataFrame):
            x={col: x[col].values.tolist() for col in x.columns}
        if isinstance(x, np.ndarray):
            x=x.tolist()
        if isinstance(x, tuple):
            x=list(x)

        if isinstance(x, dict):
            res = {}
            for k, v in x.items():
                arr=array.array("f",v)
                res[k]=ROOT.to_rvec[typ[*args]](arr, len(v))
        elif isinstance(x, list):
            arr=array.array("f",x)
            res=ROOT.to_rvec[typ[*args]](arr, len(x))
        elif isinstance(x, Number):
            res = typ[*args](x)
        else:
            raise ValueError(f"Unsupported type {type(x)}")
        return res
    return wrapper

def ap_fixed(nbits, int_bits, q_mode="AP_RND_CONV", o_mode="AP_SAT", N=0):
    quant_mode = getattr(ROOT, q_mode)
    overflow_mode = getattr(ROOT, o_mode)
    return _partial(ROOT.ap_fixed, nbits, int_bits, quant_mode, overflow_mode, N)

def ap_ufixed(nbits, int_bits, q_mode="AP_RND_CONV", o_mode="AP_SAT", N=0):
    quant_mode = getattr(ROOT, q_mode)
    overflow_mode = getattr(ROOT, o_mode)
    return _partial(ROOT.ap_fixed, nbits, int_bits, quant_mode, overflow_mode, N=0)

def ap_int(nbits, int_bits):
    return _partial(ROOT.ap_int, nbits, int_bits)

def ap_uint(nbits, int_bits):
    return _partial(ROOT.ap_uint, nbits, int_bits)

_hashed_func=set({})
def convert(x, typ):
    if typ=="str":
        typ="string"
    cpp_func="""
    template <typename T>
    ROOT::VecOps::RVec<$typ> to_$typ(const ROOT::VecOps::RVec<T> &v) {
        ROOT::VecOps::RVec<$typ> $typ_v(v.size());
        for (size_t i = 0; i < v.size(); ++i) {
            $typ_v[i]=v[i].to_$typ();
        }
        return $typ_v;
    }
    """.replace("$typ", typ)
    if hash(cpp_func) not in _hashed_func:
        ROOT.gInterpreter.Declare(cpp_func)
        _hashed_func.add(hash(cpp_func))

    if isinstance(x, Number):
        return getattr(ROOT, f"to_{typ}")(x)
    elif isinstance(x, dict):
        return {k: np.asarray(getattr(ROOT, f"to_{typ}")(v)) for k, v in x.items()}
    else:
        return np.asarray(getattr(ROOT, f"to_{typ}")(x))
