# %%
import os

from numbers import Number
import ROOT

current_path = os.path.dirname(__file__)
hls_include_path = os.path.join(current_path, "../include/XilinxHeaders/include")
ROOT.gInterpreter.AddIncludePath(current_path)
ROOT.gInterpreter.AddIncludePath(hls_include_path)
ROOT.gInterpreter.Declare('#include <inverse_lut.cpp>')


def lut_ratio(x, in_t, table_t, N=256):
    if isinstance(x, Number):
        return ROOT.invert_with_shift[in_t, table_t, N](x)
    else:
        return ROOT.invert_with_shift_v[in_t, table_t, N](x)
