"""Exact rational sparse maps delegated to the target int32 schedule."""
from dataclasses import dataclass


@dataclass
class Src:
    name: str
    shape: tuple
    scale: object
    unsigned: bool = False


def kernel_for(op, lm, srcs, s_out):
    from .exact_linear import kernel
    return kernel(op, lm, srcs, s_out)


def copy_kernel(op, lm, srcs, s_out):
    return kernel_for(op,lm,srcs,s_out)


def sum_kernel(op,lm,src,s_out,*,weights=None,rho=None):
    # lm already contains exact rational source coefficients. No synthesis,
    # approximate integer weights or FP32 readout is necessary.
    return kernel_for(op,lm,{0:src},s_out)
