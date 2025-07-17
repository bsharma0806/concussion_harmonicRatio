# setup.py

from setuptools import setup, find_packages

setup(
    name="harmonic_ratio",
    version="0.1.0",
    description="RMS and harmonic-ratio analysis for sensor time-series",
    author="Your Name",
    python_requires=">=3.7",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # Numeric and data handling
        "numpy>=1.20",
        "pandas>=1.3",
        # Signal processing
        "scipy>=1.7",
        # Plotting
        "matplotlib>=3.4",
        "seaborn>=0.11",
        # Statistical modeling (if you use it)
        "statsmodels>=0.13"
    ],
    entry_points={
        "console_scripts": [
            # RMS processing & 3D animation
            "hr-rms=harmonic_ratio.rms:plot_rms_3d_motion",
            # Harmonic time-series animation
            "hr-harmonics=harmonic_ratio.harmonics:plot_harmonic_waveforms",
            # Static plots of harmonic ratios
            "hr-plot-hratio=harmonic_ratio.harmonics:plot_harmonic_ratios",
            "hr-plot-avg-hratio=harmonic_ratio.harmonics:plot_average_harmonic_ratio_per_axis",
            # Stacked sine-wave reconstruction
            "hr-plot-reconstruct=harmonic_ratio.harmonics:plot_reconstructed_harmonics",
        ]
    },
)