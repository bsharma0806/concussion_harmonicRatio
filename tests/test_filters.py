# tests/test_filters.py
from harmonic_ratio.filters import bandpass_filter
import numpy as np

def test_bandpass_filter_shape():
    x = np.random.randn(1000)
    y = bandpass_filter(x, 0.5, 15, fs=30.1, order=3)
    assert len(x) == len(y)