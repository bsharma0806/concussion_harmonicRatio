import os
import numpy as np
import pandas as pd
from .filters import bandpass_filter
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.animation import FuncAnimation

# calculate root mean square and plot it as an animation
def calculate_rms(series):
    """Compute root-mean-square of a 1D array or pd.Series."""
    return np.sqrt(np.mean(series ** 2))


def process_rms(filepath, window='1Min'):
    """
    Process one CSV of X, Y, Z, Timestamp columns into per-window RMS.
    
    Returns a DataFrame with columns ['Timeframe','subject','X_rms','Y_rms','Z_rms','VM_rms'].
    """
    df = pd.read_csv(filepath)
    # treat Timestamp as seconds from start
    df['Timestamp'] = pd.to_numeric(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['Timestamp'])
    # convert to datetime index for resampling
    start = pd.Timestamp("2025-01-01")
    df['Datetime'] = start + pd.to_timedelta(df['Timestamp'], unit='s')
    df.set_index('Datetime', inplace=True)
    
    # bandpass filter each axis
    for col in ['X', 'Y', 'Z']:
        df[col] = bandpass_filter(df[col], lowcut=0.5, highcut=15, fs=30.1, order=3)
    # compute vector magnitude
    df['VM'] = np.sqrt(df['X']**2 + df['Y']**2 + df['Z']**2)
    
    records = []
    for ts, chunk in df.resample(window):
        rec = {'Timeframe': ts, 'subject': os.path.basename(filepath).split('.')[0]}
        rec['X_rms'] = calculate_rms(chunk['X'])
        rec['Y_rms'] = calculate_rms(chunk['Y'])
        rec['Z_rms'] = calculate_rms(chunk['Z'])
        rec['VM_rms'] = calculate_rms(chunk['VM'])
        records.append(rec)
    
    return pd.DataFrame(records)

def process_rms_folder(folder, window='1Min'):
    """
    Loop over all .csv in `folder`, process each, and tag group.
    Returns a single concatenated DataFrame.
    """
    dfs = []
    for fname in os.listdir(folder):
        if not fname.lower().endswith('.csv'):
            continue
        df = process_rms(os.path.join(folder, fname), window=window)
        df['group'] = 'concussion' if 'concussion' in fname.lower() else 'control'
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)

def plot_rms_3d_motion(rms_all, output_path="rms_3d_motion.gif", fps=3):
    """
    Animate the RMS (X_rms, Y_rms, Z_rms) as a moving 3D point with trail,
    side-by-side for control vs concussion.
    """
    # prepare group data
    grouped = rms_all.groupby(['Timeframe','group'])[['X_rms','Y_rms','Z_rms']].mean().reset_index()
    ctrl = grouped[grouped['group']=='control'].sort_values('Timeframe').reset_index(drop=True)
    conc = grouped[grouped['group']=='concussion'].sort_values('Timeframe').reset_index(drop=True)
    # sync frame count
    n = min(len(ctrl), len(conc))
    ctrl, conc = ctrl.iloc[:n], conc.iloc[:n]
    
    # global axis limits
    combined = pd.concat([ctrl, conc])
    mins = combined[['X_rms','Y_rms','Z_rms']].min()
    maxs = combined[['X_rms','Y_rms','Z_rms']].max()
    rng = maxs - mins
    margin = 0.05
    lims = {
        'x': (mins['X_rms'] - margin*rng['X_rms'], maxs['X_rms'] + margin*rng['X_rms']),
        'y': (mins['Y_rms'] - margin*rng['Y_rms'], maxs['Y_rms'] + margin*rng['Y_rms']),
        'z': (mins['Z_rms'] - margin*rng['Z_rms'], maxs['Z_rms'] + margin*rng['Z_rms']),
    }
    
    fig = plt.figure(figsize=(12,6))
    ax1 = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122, projection='3d')
    for ax,title in zip([ax1,ax2], ['Control','Concussion']):
        ax.set_xlim(lims['x']); ax.set_ylim(lims['y']); ax.set_zlim(lims['z'])
        ax.set_xlabel('X RMS'); ax.set_ylabel('Y RMS'); ax.set_zlabel('Z RMS')
        ax.set_title(title)
    
    # plot handles
    dot1, = ax1.plot([],[],[],'bo'); trail1, = ax1.plot([],[],[],'b--',alpha=0.6)
    dot2, = ax2.plot([],[],[],'ro'); trail2, = ax2.plot([],[],[],'r--',alpha=0.6)
    trail1_x, trail1_y, trail1_z = [],[],[]
    trail2_x, trail2_y, trail2_z = [],[],[]
    
    def update(i):
        for grp, dot, trail, tx, ty, tz, data in zip(
            ['control','concussion'],
            [dot1,dot2],
            [trail1,trail2],
            [trail1_x,trail2_x],
            [trail1_y,trail2_y],
            [trail1_z,trail2_z],
            [ctrl, conc]
        ):
            row = data.iloc[i]
            x,y,z = row['X_rms'], row['Y_rms'], row['Z_rms']
            tx.append(x); ty.append(y); tz.append(z)
            dot.set_data([x],[y]); dot.set_3d_properties([z])
            trail.set_data(tx,ty); trail.set_3d_properties(tz)
        fig.suptitle(f"Mean RMS – Window {i+1}/{n}", fontsize=14)
        return dot1, trail1, dot2, trail2
    
    ani = FuncAnimation(fig, update, frames=n, interval=300, blit=False)
    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    ani.save(output_path, writer='pillow', fps=fps)
    plt.close(fig)
    print(f"✅ RMS 3D animation saved to {output_path}")