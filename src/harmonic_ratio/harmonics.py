# src/harmonic_ratio/harmonics.py

import os
import glob
import numpy as np
import pandas as pd
from .filters import bandpass_filter
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import seaborn as sns

# calculating and plotting harmonic ratios per axis
def compute_harmonics(signal, fs):
    """
    Compute harmonic amplitudes for a 1D signal via FFT.

    Parameters
    ----------
    signal : array-like
        Time-domain signal values.
    fs : float
        Sampling frequency in Hz.

    Returns
    -------
    amps : list of float
        Amplitudes for each harmonic component (including DC).
    """
    N = len(signal)
    vals = np.fft.rfft(signal)
    # Normalize and double (except DC) to get true amplitude
    amps = (2.0 / N) * np.abs(vals)
    return amps.tolist()

def process_harmonics_folder(folder, window='1Min', axes=('X','Y','Z')):
    """
    Compute time-series of mean harmonic amplitudes across subjects.

    For each CSV in `folder`, resample each axis into time windows and compute
    the first 5 harmonic amplitudes, then average across subjects by group.

    Parameters
    ----------
    folder : str
        Path to directory of CSV files (each with Timestamp and axis columns).
    window : str
        Pandas resample frequency string, e.g. '1Min', '10S'.
    axes : tuple of str
        Column names for the axes to process (e.g. ('X','Y','Z')).

    Returns
    -------
    harmonics_ts_mean : pandas.DataFrame
        Columns: ['Timeframe','group','Axis','Harmonic','Amplitude']
    """
    records = []
    for filepath in glob.glob(os.path.join(folder, '*.csv')):
        fname = os.path.basename(filepath)
        group = 'concussion' if 'concussion' in fname.lower() else 'control'
        df = pd.read_csv(filepath)
        # Convert timestamp seconds to datetime index for resampling
        df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
        start = pd.Timestamp("2025-01-01")
        df['Datetime'] = start + pd.to_timedelta(df['Timestamp'], unit='s')
        df.set_index('Datetime', inplace=True)

        for axis in axes:
            # Apply bandpass filter to clean signal
            sig = bandpass_filter(df[axis], lowcut=0.5, highcut=15, fs=30.1, order=3)
            # Resample into windows
            for ts, chunk in pd.Series(sig, index=df.index).resample(window):
                vals = chunk.values
                if len(vals) < 2:
                    continue
                amps = compute_harmonics(vals, fs=30.1)
                # Record first 5 harmonics
                for hnum, amp in enumerate(amps[1:6], start=1):  # skip DC (index 0)
                    records.append({
                        'Timeframe': ts,
                        'group': group,
                        'Axis': axis,
                        'Harmonic': hnum,
                        'Amplitude': amp
                    })

    harmonics_time_df = pd.DataFrame(records)
    # Group-mean across subjects by timeframe, group, axis, harmonic
    harmonics_ts_mean = (
        harmonics_time_df
        .groupby(['Timeframe','group','Axis','Harmonic'], as_index=False)['Amplitude']
        .mean()
    )
    return harmonics_ts_mean

def plot_harmonic_ratios(harmonics_df, axis='Z', output_path=None):
    """
    Static bar chart: mean amplitude for first 5 harmonics, by group, for one axis.

    Parameters
    ----------
    harmonics_df : pandas.DataFrame
        Output of process_harmonics_folder().
    axis : str
        Axis to plot (e.g. 'X', 'Y', 'Z').
    output_path : str or None
        If given, save figure to this path.
    """
    df = harmonics_df[
        (harmonics_df['Axis'] == axis) &
        (harmonics_df['Harmonic'] <= 5)
    ].copy()
    df['group'] = df['group'].str.capitalize()

    sns.set(style="whitegrid")
    plt.figure(figsize=(6,4))
    ax = sns.barplot(
        data=df,
        x='Harmonic', y='Amplitude', hue='group',
        palette={'Control':'blue','Concussion':'red'}
    )
    ax.set_title(f"{axis}-Axis Harmonic Ratios")
    ax.set_ylabel("Mean Amplitude")
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.show()

def plot_average_harmonic_ratio_per_axis(harmonics_df, output_path=None):
    """
    Static bar chart: average amplitude of first 5 harmonics per axis, by group.

    Parameters
    ----------
    harmonics_df : pandas.DataFrame
        Output of process_harmonics_folder().
    output_path : str or None
        If given, save figure to this path.
    """
    df = harmonics_df[harmonics_df['Harmonic'] <= 5].copy()
    df['group'] = df['group'].str.capitalize()
    avg = (
        df
        .groupby(['group','Axis'], as_index=False)['Amplitude']
        .mean()
    )

    sns.set(style="whitegrid")
    plt.figure(figsize=(6,4))
    ax = sns.barplot(
        data=avg,
        x='Axis', y='Amplitude', hue='group',
        palette={'Control':'blue','Concussion':'red'}
    )
    ax.set_title("Average Harmonic Ratio per Axis")
    ax.set_ylabel("Mean Amplitude")
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.show()

def plot_reconstructed_harmonics(harmonics_df, axis='Z', output_path=None):
    """
    Plot stacked time-domain sine waves reconstructed from the first 5 harmonics,
    showing fundamental and higher frequencies for each group.

    Parameters
    ----------
    harmonics_df : pandas.DataFrame
        Output of process_harmonics_folder().
    axis : str
        Axis to reconstruct (e.g. 'X', 'Y', 'Z').
    output_path : str or None
        If given, save figure to this path.
    """
    # Compute overall mean amplitude per harmonic, per group
    df = harmonics_df[
        (harmonics_df['Axis'] == axis) &
        (harmonics_df['Harmonic'] <= 5)
    ]
    meanamps = (
        df.groupby(['group','Harmonic'])['Amplitude']
          .mean()
          .reset_index()
    )

    # Extract control and concussion amplitude arrays
    ctrl = meanamps[meanamps['group']=='control'] \
                .sort_values('Harmonic')['Amplitude'].values
    conc = meanamps[meanamps['group']=='concussion'] \
                .sort_values('Harmonic')['Amplitude'].values

    t = np.linspace(0, 1, 1000)
    f0 = 1.0  # fundamental frequency (Hz)
    maxamp = max(ctrl.max(), conc.max())
    offset = maxamp * 1.5

    # Build stacked sine waves
    ctrl_waves = [A * np.sin(2*np.pi*(n+1)*f0 * t) + n * offset
                  for n, A in enumerate(ctrl)]
    conc_waves = [A * np.sin(2*np.pi*(n+1)*f0 * t) + n * offset
                  for n, A in enumerate(conc)]

    # Plot side-by-side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    for n, wave in enumerate(ctrl_waves, start=1):
        ax1.plot(t, wave, label=f"Harmonic {n}")
    for n, wave in enumerate(conc_waves, start=1):
        ax2.plot(t, wave, label=f"Harmonic {n}")

    ax1.set_title("Control Group")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude + offset")
    ax1.legend(loc="upper right")
    ax2.set_title("Concussion Group")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right")

    plt.suptitle(f"{axis}-Axis: First 5 Harmonic Waveforms (stacked)", fontsize=16)
    plt.tight_layout(rect=[0,0,1,0.95])
    if output_path:
        os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
        plt.savefig(output_path, dpi=300)
    plt.show()

def plot_harmonic_waveforms(harmonics_ts_mean, axis='Z', output_path='harmonics_waveforms.gif', fps=2):
    """
    Animate side-by-side reconstructed waveforms for the first 5 harmonics
    over time windows for control vs concussion.

    Parameters
    ----------
    harmonics_ts_mean : pandas.DataFrame
        Output of process_harmonics_folder().
    axis : str
        Axis to animate (e.g. 'X', 'Y', 'Z').
    output_path : str
        Path to save the GIF.
    fps : int
        Frames per second for the animation.
    """
    df = harmonics_ts_mean[
        (harmonics_ts_mean['Axis'] == axis) &
        (harmonics_ts_mean['Harmonic'] <= 5)
    ].copy()
    df.sort_values('Timeframe', inplace=True)

    # Pivot into time × harmonic matrices
    ctrl = df[df['group']=='control'].pivot(index='Timeframe', columns='Harmonic', values='Amplitude')
    conc = df[df['group']=='concussion'].pivot(index='Timeframe', columns='Harmonic', values='Amplitude')
    times = ctrl.index.intersection(conc.index)
    ctrl, conc = ctrl.loc[times], conc.loc[times]
    n = len(times)

    # Precompute wave reconstructions
    t = np.linspace(0, 1, 500)
    f0 = 1.0
    ctrl_waves = [sum(ctrl.iloc[i].values[n] * np.sin(2*np.pi*(n+1)*f0*t)
                      for n in range(5)) for i in range(n)]
    conc_waves = [sum(conc.iloc[i].values[n] * np.sin(2*np.pi*(n+1)*f0*t)
                      for n in range(5)) for i in range(n)]

    # Determine y-limits
    all_w = np.hstack(ctrl_waves + conc_waves)
    ymin, ymax = all_w.min(), all_w.max()
    pad = 0.1 * (ymax - ymin)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, title in zip([ax1, ax2], ['Control', 'Concussion']):
        ax.set_xlim(0, 1)
        ax.set_ylim(ymin - pad, ymax + pad)
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.set_title(title)

    line1, = ax1.plot([], [], 'b-', lw=2)
    line2, = ax2.plot([], [], 'r-', lw=2)

    def update(i):
        line1.set_data(t, ctrl_waves[i])
        line2.set_data(t, conc_waves[i])
        fig.suptitle(f"{axis}-Axis Reconstruction • Window {i+1}/{n}", fontsize=14)
        return line1, line2

    ani = FuncAnimation(fig, update, frames=n, interval=400, blit=False)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"✅ Harmonic waveforms animation saved to {output_path}")