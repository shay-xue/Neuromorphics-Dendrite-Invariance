"""Channel and current models.

The parameter values are coming from the HW2 model.
"""

from __future__ import annotations
import numpy as np

# Equillibrium potentials
E_NMDA = 1.0
E_AMPA = 1.0
E_KIR = -1.0

# Parameters VHA and VS, fit from real data
VHA_NMDA = 71 / 150
VS_NMDA = 5 / 18
VHA_KIR = -7 / 9
VS_KIR = -2 / 9

# Normalization constants for NMDA and KIR activation curves
A_NMDA = 1 + np.exp(-(E_NMDA - VHA_NMDA) / VS_NMDA)
A_KIR = 1 + np.exp(-(E_KIR - VHA_KIR) / VS_KIR)


def sigma(v: np.ndarray | float, a: float, vha: float, vs: float) -> np.ndarray | float:
    """Sigmoid activation curve"""
    return a / (1 + np.exp(-(v - vha) / vs))


def sigma_NMDA(v: np.ndarray | float) -> np.ndarray | float:
    """NMDA activation"""
    return sigma(v, A_NMDA, VHA_NMDA, VS_NMDA)


def sigma_KIR(v: np.ndarray | float) -> np.ndarray | float:
    """KIR activation"""
    return sigma(v, A_KIR, VHA_KIR, VS_KIR)


def I_channels(v: np.ndarray, G_NMDA: float, G_KIR: float) -> np.ndarray:
    """"NMDA + KIR current for each dendrite segment"""
    return (
        G_NMDA * sigma_NMDA(v) * (E_NMDA - v)
        + G_KIR * sigma_KIR(v) * (E_KIR - v)
    )


def I_axial(v: np.ndarray, G_ax: float) -> np.ndarray:
    """Current between neighboring dendrite segments"""
    Iax = np.zeros_like(v)
    if len(v) == 1:
        return Iax
    Iax[0] = G_ax * (v[1] - v[0])
    Iax[-1] = G_ax * (v[-2] - v[-1])
    Iax[1:-1] = G_ax * (v[:-2] + v[2:] - 2 * v[1:-1])
    return Iax
