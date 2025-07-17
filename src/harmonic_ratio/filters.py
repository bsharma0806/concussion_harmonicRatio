from scipy.signal import butter, filtfilt

# bandpass filtering the accelerometer data
def bandpass_filter(data, lowcut, highcut, fs, order):
    """
    Apply a zero-phase Butterworth bandpass filter.
    
    Parameters
    ----------
    data : array-like
        1D signal to filter.
    lowcut : float
        Low cutoff frequency (Hz).
    highcut : float
        High cutoff frequency (Hz).
    fs : float
        Sampling rate (Hz).
    order : int
        Filter order.
    
    Returns
    -------
    filtered : ndarray
        Bandpass-filtered signal.
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype='band')
    return filtfilt(b, a, data)