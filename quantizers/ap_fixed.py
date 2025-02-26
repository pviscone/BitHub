import apytypes
from apytypes import APyFixedArray

_ap_fixed_quantization_modes = {
    "AP_RND" : 5,         #Round to plus infinity (our default)
    "AP_RND_ZERO" : 6,    #Round to zero
    "AP_RND_MIN_INF" : 8, #Round to minus infinity
    "AP_RND_INF" : 7,     #Round to infinity
    "AP_RND_CONV" : 9,    #Convergent rounding
    "AP_TRN" : 0,         #Truncate (xilinx ap_fixed default)
    "AP_TRN_ZERO": 2      #Truncate to zero
}

_ap_fixed_overflow_modes = {
    "AP_SAT" : 1,          #Saturate (our default)
    "AP_SAT_ZERO" : -1,    #Saturate to zero
    "AP_SAT_SYM" : -1,     #Symmetric saturation
    "AP_WRAP" : 0,         #Wrap around (xilinx ap_fixed default)
    "AP_WRAP_SM" : -1,     #Wrap around with self-modulus
}

def ap_fixed(x, n_bits, frac_bits, quantization_mode="AP_RND", overflow_mode="AP_SAT"):
    if quantization_mode not in _ap_fixed_quantization_modes:
        raise ValueError(f"Invalid quantization mode: {quantization_mode}")
    if overflow_mode not in _ap_fixed_overflow_modes:
        raise ValueError(f"Invalid overflow mode: {overflow_mode}")

    if _ap_fixed_overflow_modes[overflow_mode] == -1:
        raise NotImplementedError(f"Overflow mode {overflow_mode} not implemented yet")

    apytypes.OverflowMode(_ap_fixed_overflow_modes[overflow_mode])
    apytypes.set_float_quantization_mode(_ap_fixed_quantization_modes[quantization_mode])

    int_bits = n_bits - frac_bits  #including sign bit

    return APyFixedArray.from_array(x, int_bits=int_bits, frac_bits=frac_bits)
