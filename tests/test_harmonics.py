from harmonic_ratio.harmonics import compute_harmonics
import numpy as np

def test_compute_harmonics_output():
    sig = np.sin(2 * np.pi * 1 * np.linspace(0, 1, 1000))
    amps = compute_harmonics(sig, fs=100)
    assert isinstance(amps, list)
    assert amps[0] >= 0