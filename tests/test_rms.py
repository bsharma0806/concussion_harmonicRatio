import os
from harmonic_ratio.rms import process_rms_file

def test_rms_output_structure():
    path = "synthetic_data/control01.csv"  # replace with any existing test file
    minute_df, summary_df = process_rms_file(path)
    assert "X_rms" in minute_df.columns
    assert "subject" in summary_df.columns
    assert len(minute_df) > 0