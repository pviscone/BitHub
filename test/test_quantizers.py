from BitHub.bithub.quantizers import fxpmath
from bithub.quantizers import xilinx

import numpy as np
import pytest
x = np.linspace(-100,100,1000)
xp = np.linspace(0,100,1000)

@pytest.mark.parametrize("nbits", [4, 8, 16, 24])
@pytest.mark.parametrize("int_bits", [2, 6, 12])
@pytest.mark.parametrize("q_mode", fxpmath._q_modes)
@pytest.mark.parametrize("o_mode", fxpmath._o_modes)
def test_str_fxpmath_apfixed(nbits, int_bits, q_mode, o_mode):
    if int_bits >= nbits:
        return
    xilinx_ap_fixed = xilinx.ap_fixed(nbits, int_bits, q_mode, o_mode)(x)
    xilinx_ap_fixed = xilinx.convert(xilinx_ap_fixed, "string")


    fpxmath_ap_fixed = fxpmath.ap_fixed(nbits, int_bits, q_mode, o_mode)(x)
    fpxmath_ap_fixed = fxpmath.convert(fpxmath_ap_fixed, "string")

    res1=[]
    res2=[]
    for idx, val in enumerate(xilinx_ap_fixed):
        if "." not in val:
            continue
        res1.append(val.split(".")[1])
        res2.append(fpxmath_ap_fixed[idx].split(".")[1])

    res1=np.array(res1)
    res2=np.array(res2)

    #decimal=int(np.ceil(np.log10(2**(nbits-int_bits))))
    #np.testing.assert_almost_equal(xilinx_ap_fixed, fpxmath_ap_fixed, decimal=decimal)
    np.testing.assert_equal(res1,res2)

@pytest.mark.parametrize("nbits", [4, 8, 16, 24])
@pytest.mark.parametrize("int_bits", [2, 6, 12])
@pytest.mark.parametrize("q_mode", fxpmath._q_modes)
@pytest.mark.parametrize("o_mode", fxpmath._o_modes)
def test_fxpmath_apfixed(nbits, int_bits, q_mode, o_mode):
    if int_bits >= nbits:
        return
    xilinx_ap_fixed = xilinx.ap_fixed(nbits, int_bits, q_mode, o_mode)(x)
    xilinx_ap_fixed = xilinx.convert(xilinx_ap_fixed, "double")


    fpxmath_ap_fixed = fxpmath.ap_fixed(nbits, int_bits, q_mode, o_mode)(x)
    fpxmath_ap_fixed = fxpmath.convert(fpxmath_ap_fixed, "double")

    decimal=int(np.ceil(np.log10(2**(nbits-int_bits))))
    np.testing.assert_almost_equal(xilinx_ap_fixed, fpxmath_ap_fixed, decimal=decimal)


@pytest.mark.parametrize("nbits", [4, 8, 16, 24])
@pytest.mark.parametrize("int_bits", [2, 6, 12])
@pytest.mark.parametrize("q_mode", fxpmath._q_modes)
@pytest.mark.parametrize("o_mode", fxpmath._o_modes)
def test_fxpmath_apufixed(nbits, int_bits, q_mode, o_mode):
    if int_bits >= nbits:
        return
    xilinx_ap_ufixed = xilinx.ap_ufixed(nbits, int_bits, q_mode, o_mode)(xp)
    xilinx_ap_ufixed = xilinx.convert(xilinx_ap_ufixed, "double")

    fpxmath_ap_ufixed = fxpmath.ap_ufixed(nbits, int_bits, q_mode, o_mode)(xp)
    fpxmath_ap_ufixed = fxpmath.convert(fpxmath_ap_ufixed, "double")

    decimal=int(np.ceil(np.log10(2**(nbits-int_bits))))
    np.testing.assert_almost_equal(xilinx_ap_ufixed, fpxmath_ap_ufixed, decimal=decimal)


@pytest.mark.parametrize("nbits", [4, 8, 16, 24])
def test_fxpmath_apint(nbits):
    xilinx_ap_int = xilinx.ap_int(nbits)(x)
    xilinx_ap_int = xilinx.convert(xilinx_ap_int, "int")

    fpxmath_ap_int = fxpmath.ap_int(nbits)(x)
    fpxmath_ap_int = fxpmath.convert(fpxmath_ap_int, "int")

    np.testing.assert_almost_equal(xilinx_ap_int, fpxmath_ap_int)


@pytest.mark.parametrize("nbits", [4, 8, 16, 24])
def test_fxpmath_apuint(nbits):
    xilinx_ap_uint = xilinx.ap_uint(nbits)(xp)
    xilinx_ap_uint = xilinx.convert(xilinx_ap_uint, "int")

    fpxmath_ap_uint = fxpmath.ap_uint(nbits)(xp)
    fpxmath_ap_uint = fxpmath.convert(fpxmath_ap_uint, "int")

    np.testing.assert_almost_equal(xilinx_ap_uint, fpxmath_ap_uint)