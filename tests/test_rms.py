import os
from harmonic_ratio.rms import process_rms_folder

def test_rms_output_structure():
    df = process_rms_folder("synthetic_data")
    assert "X_rms" in df.columns
    assert len(df) > 0