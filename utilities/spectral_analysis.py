"""
Spectral and Wavelet Analysis Module

This module provides tools for frequency domain analysis of time series:
- FFT and Periodogram
- Power Spectral Density
- Wavelet Analysis (Continuous and Discrete)
- Time-Frequency plots
- Seasonal component identification
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Optional, Union
from scipy import signal
from scipy.fft import fft, fftfreq
import warnings

try:
    import pywt # type: ignore
    PYWT_AVAILABLE = True
except ImportError:
    PYWT_AVAILABLE = False
    warnings.warn("pywt not available. Install with: pip install PyWavelets")

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


def compute_fft(timeseries: np.ndarray,
                sampling_rate: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Fast Fourier Transform

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    sampling_rate : float
        Sampling rate (1.0 for daily data)

    Returns:
    --------
    tuple: (frequencies, power)
    """
    n = len(timeseries)

    # Remove mean
    timeseries_centered = timeseries - np.mean(timeseries)

    # Compute FFT
    fft_values = fft(timeseries_centered)
    power = np.abs(fft_values) ** 2

    # Get positive frequencies only
    frequencies = fftfreq(n, d=1/sampling_rate)

    # Keep only positive frequencies
    positive_freq_idx = frequencies > 0
    frequencies = frequencies[positive_freq_idx]
    power = power[positive_freq_idx]

    return frequencies, power


def compute_periodogram(timeseries: np.ndarray,
                       sampling_rate: float = 1.0,
                       window: str = 'hann') -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute periodogram using Welch's method

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    sampling_rate : float
        Sampling rate
    window : str
        Window function ('hann', 'hamming', 'blackman', etc.)

    Returns:
    --------
    tuple: (frequencies, power_density)
    """
    frequencies, power = signal.periodogram(
        timeseries,
        fs=sampling_rate,
        window=window,
        scaling='density'
    )

    return frequencies, power


def compute_welch_psd(timeseries: np.ndarray,
                     sampling_rate: float = 1.0,
                     nperseg: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute Power Spectral Density using Welch's method

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    sampling_rate : float
        Sampling rate
    nperseg : int, optional
        Length of each segment (default: 256 or len(timeseries)//8)

    Returns:
    --------
    tuple: (frequencies, psd)
    """
    if nperseg is None:
        nperseg = min(256, len(timeseries) // 8)

    frequencies, psd = signal.welch(
        timeseries,
        fs=sampling_rate,
        nperseg=nperseg,
        scaling='density'
    )

    return frequencies, psd


def plot_frequency_spectrum(timeseries: np.ndarray,
                           sampling_rate: float = 1.0,
                           title: str = "Frequency Spectrum",
                           highlight_peaks: bool = True,
                           n_peaks: int = 5) -> plt.Figure:
    """
    Plot frequency spectrum with peak detection

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    sampling_rate : float
        Sampling rate (1.0 for daily data)
    title : str
        Plot title
    highlight_peaks : bool
        Whether to highlight spectral peaks
    n_peaks : int
        Number of peaks to highlight

    Returns:
    --------
    matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # FFT
    frequencies_fft, power_fft = compute_fft(timeseries, sampling_rate)

    # Periodogram
    frequencies_per, power_per = compute_periodogram(timeseries, sampling_rate)

    # Plot FFT
    ax1 = axes[0]
    ax1.plot(frequencies_fft, power_fft, linewidth=0.8, color='darkblue', alpha=0.7)

    if highlight_peaks:
        # Find peaks
        peaks, properties = signal.find_peaks(power_fft, height=np.percentile(power_fft, 90))
        if len(peaks) > 0:
            # Get top n_peaks
            peak_heights = power_fft[peaks]
            top_peak_indices = peaks[np.argsort(peak_heights)[-n_peaks:]]

            ax1.plot(frequencies_fft[top_peak_indices],
                    power_fft[top_peak_indices],
                    'ro', markersize=8, label='Top Peaks')

            # Annotate peaks with periods
            for peak_idx in top_peak_indices[:min(3, len(top_peak_indices))]:
                freq = frequencies_fft[peak_idx]
                if freq > 0:
                    period = 1 / freq
                    ax1.annotate(f'Period: {period:.1f}',
                               xy=(freq, power_fft[peak_idx]),
                               xytext=(10, 10), textcoords='offset points',
                               fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    ax1.set_xlabel('Frequency (cycles per unit time)', fontsize=12)
    ax1.set_ylabel('Power', fontsize=12)
    ax1.set_title(f'{title} - FFT Power Spectrum', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    if highlight_peaks:
        ax1.legend()

    # Plot Periodogram
    ax2 = axes[1]
    ax2.semilogy(frequencies_per, power_per, linewidth=0.8, color='darkgreen', alpha=0.7)
    ax2.set_xlabel('Frequency (cycles per unit time)', fontsize=12)
    ax2.set_ylabel('Power Spectral Density (log scale)', fontsize=12)
    ax2.set_title(f'{title} - Periodogram', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_psd(timeseries: np.ndarray,
            sampling_rate: float = 1.0,
            title: str = "Power Spectral Density") -> plt.Figure:
    """
    Plot Power Spectral Density using Welch's method

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    sampling_rate : float
        Sampling rate
    title : str
        Plot title

    Returns:
    --------
    matplotlib Figure
    """
    frequencies, psd = compute_welch_psd(timeseries, sampling_rate)

    fig, ax = plt.subplots(figsize=(16, 7))

    ax.semilogy(frequencies, psd, linewidth=1.5, color='darkblue', alpha=0.8)
    ax.set_xlabel('Frequency (cycles per unit time)', fontsize=13)
    ax.set_ylabel('Power Spectral Density (log scale)', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def identify_seasonal_periods(timeseries: np.ndarray,
                              sampling_rate: float = 1.0,
                              n_periods: int = 5) -> pd.DataFrame:
    """
    Identify dominant seasonal periods from frequency spectrum

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    sampling_rate : float
        Sampling rate (1.0 for daily data)
    n_periods : int
        Number of top periods to return

    Returns:
    --------
    DataFrame with period information
    """
    frequencies, power = compute_fft(timeseries, sampling_rate)

    # Find peaks
    peaks, properties = signal.find_peaks(power, height=np.percentile(power, 80))

    if len(peaks) == 0:
        print("No significant peaks found in frequency spectrum")
        return pd.DataFrame()

    # Get peak information
    peak_freqs = frequencies[peaks]
    peak_powers = power[peaks]
    peak_periods = 1 / peak_freqs

    # Sort by power
    sorted_indices = np.argsort(peak_powers)[::-1]

    # Create results dataframe
    results = [] # type: ignore
    for i in sorted_indices[:n_periods]:
        results.append({
            'rank': len(results) + 1,
            'period': peak_periods[i],
            'frequency': peak_freqs[i],
            'power': peak_powers[i],
            'interpretation': _interpret_period(peak_periods[i])
        })

    df = pd.DataFrame(results)

    print(f"\n{'='*70}")
    print("DOMINANT SEASONAL PERIODS")
    print(f"{'='*70}")
    print(df.to_string(index=False))
    print(f"{'='*70}\n")

    return df


def _interpret_period(period: float) -> str:
    """Interpret period in context of time series"""
    if 6.5 <= period <= 7.5:
        return "Weekly"
    elif 27 <= period <= 32:
        return "Monthly"
    elif 89 <= period <= 93:
        return "Quarterly"
    elif 182 <= period <= 184:
        return "Semi-annual"
    elif 360 <= period <= 370:
        return "Annual"
    else:
        return f"{period:.1f} days"


def continuous_wavelet_transform(timeseries: np.ndarray,
                                 scales: Optional[np.ndarray] = None,
                                 wavelet: str = 'morl',
                                 sampling_rate: float = 1.0,
                                 detrend: bool = True) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Continuous Wavelet Transform

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    scales : array-like, optional
        Scales to use for CWT
    wavelet : str
        Wavelet to use ('morl', 'cmor', 'gaus', etc.)
    sampling_rate : float
        Sampling rate
    detrend : bool
        Whether to detrend the signal before CWT

    Returns:
    --------
    tuple: (coefficients, frequencies, scales)
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")

    # Detrend if requested
    ts_data = timeseries.copy()
    if detrend:
        from scipy.signal import detrend as scipy_detrend
        ts_data = scipy_detrend(ts_data)

    if scales is None:
        # Improved scales for time series analysis
        # Focus on periods from 2 days to 2 years
        # More samples in the important range (weekly to annual)
        max_scale = min(len(timeseries)//2, 730)  # Up to 2 years

        # Create non-uniform scales: dense sampling around important periods
        scales_low = np.linspace(2, 20, 30)      # 2-20 days (weekly, biweekly)
        scales_mid = np.linspace(20, 100, 40)    # 20-100 days (monthly, quarterly)
        scales_high = np.geomspace(100, max_scale, 50)  # 100-730 days (semi-annual, annual)

        scales = np.unique(np.concatenate([scales_low, scales_mid, scales_high]))

    # Compute CWT
    coefficients, frequencies = pywt.cwt(ts_data, scales, wavelet, sampling_period=1/sampling_rate)

    return coefficients, frequencies, scales


def plot_wavelet_scalogram(timeseries: np.ndarray,
                           datetime_index: Optional[pd.DatetimeIndex] = None,
                           scales: Optional[np.ndarray] = None,
                           wavelet: str = 'morl',
                           sampling_rate: float = 1.0,
                           title: str = "Wavelet Scalogram",
                           detrend: bool = True,
                           normalize: str = 'log') -> plt.Figure:
    """
    Plot wavelet scalogram (time-frequency representation)

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    datetime_index : DatetimeIndex, optional
        Datetime index for x-axis
    scales : array-like, optional
        Scales for CWT
    wavelet : str
        Wavelet to use
    sampling_rate : float
        Sampling rate
    title : str
        Plot title
    detrend : bool
        Whether to detrend the signal before CWT
    normalize : str
        Power normalization method: 'log', 'scale', or None

    Returns:
    --------
    matplotlib Figure
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")

    # Compute CWT
    coefficients, frequencies, scales_used = continuous_wavelet_transform(
        timeseries, scales, wavelet, sampling_rate, detrend=detrend
    )

    # Compute power
    power = np.abs(coefficients) ** 2

    # Normalize power for better visualization
    if normalize == 'log':
        # Logarithmic scaling - makes weak patterns visible
        power = np.log10(power + 1e-12)  # Add small value to avoid log(0)
    elif normalize == 'scale':
        # Row-wise normalization - normalize each scale separately
        power = power / (power.max(axis=1, keepdims=True) + 1e-12)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(18, 10))

    # Plot original signal
    ax1 = axes[0]
    if datetime_index is not None:
        ax1.plot(datetime_index, timeseries, linewidth=0.8, color='darkblue', alpha=0.8)
        ax1.set_xlabel('Date', fontsize=12)
    else:
        ax1.plot(timeseries, linewidth=0.8, color='darkblue', alpha=0.8)
        ax1.set_xlabel('Time', fontsize=12)

    ax1.set_ylabel('Value', fontsize=12)
    signal_title = 'Detrended Signal' if detrend else 'Original Signal'
    ax1.set_title(f'{title} - {signal_title}', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)

    # Plot scalogram
    ax2 = axes[1]

    # Convert frequencies to periods
    periods = 1 / frequencies

    if datetime_index is not None:
        extent = [datetime_index.iloc[0], datetime_index.iloc[-1], periods.min(), periods.max()] # type: ignore
        im = ax2.imshow(power, extent=extent, aspect='auto', cmap='jet',
                       origin='lower', interpolation='bilinear')
    else:
        im = ax2.imshow(power, extent=[0, len(timeseries), periods.min(), periods.max()],
                       aspect='auto', cmap='jet', origin='lower', interpolation='bilinear')

    # Add reference lines for important periods
    important_periods = [7, 30, 91, 183, 365]  # Weekly, monthly, quarterly, semi-annual, annual
    for period in important_periods:
        if periods.min() <= period <= periods.max():
            ax2.axhline(y=period, color='white', linestyle='--', linewidth=1.5, alpha=0.6)
            ax2.text(ax2.get_xlim()[1] if datetime_index is None else extent[1],
                    period, f' {period}d', color='white',
                    fontsize=9, va='center', ha='right', weight='bold',
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

    ax2.set_ylabel('Period (days)', fontsize=12)
    ax2.set_xlabel('Date' if datetime_index is not None else 'Time', fontsize=12)
    power_label = 'Log Power' if normalize == 'log' else 'Normalized Power' if normalize == 'scale' else 'Power'
    ax2.set_title(f'{title} - Wavelet {power_label} Spectrum', fontsize=14, fontweight='bold')
    ax2.set_yscale('log')
    ax2.set_ylim([periods.min(), periods.max()])

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2)
    cbar.set_label(power_label, fontsize=11)

    plt.tight_layout()
    return fig


def discrete_wavelet_transform(timeseries: np.ndarray,
                               wavelet: str = 'db4',
                               level: Optional[int] = None) -> Tuple:
    """
    Compute Discrete Wavelet Transform

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    wavelet : str
        Wavelet to use ('db4', 'sym5', 'coif3', etc.)
    level : int, optional
        Decomposition level

    Returns:
    --------
    tuple of wavelet coefficients
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")

    if level is None:
        level = min(pywt.dwt_max_level(len(timeseries), wavelet), 6)

    coefficients = pywt.wavedec(timeseries, wavelet, level=level)

    return coefficients


def plot_dwt_decomposition(timeseries: np.ndarray,
                          wavelet: str = 'db4',
                          level: Optional[int] = None,
                          title: str = "DWT Decomposition") -> plt.Figure:
    """
    Plot Discrete Wavelet Transform decomposition

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    wavelet : str
        Wavelet to use
    level : int, optional
        Decomposition level
    title : str
        Plot title

    Returns:
    --------
    matplotlib Figure
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")

    coefficients = discrete_wavelet_transform(timeseries, wavelet, level)

    n_levels = len(coefficients)
    fig, axes = plt.subplots(n_levels, 1, figsize=(16, 3*n_levels))

    if n_levels == 1:
        axes = [axes]

    # Plot approximation coefficients
    axes[0].plot(coefficients[0], linewidth=0.8, color='darkblue', alpha=0.8)
    axes[0].set_title(f'{title} - Approximation Coefficients (Level {n_levels-1})',
                     fontsize=12, fontweight='bold')
    axes[0].grid(True, alpha=0.3)

    # Plot detail coefficients
    for i in range(1, n_levels):
        level_num = n_levels - i
        axes[i].plot(coefficients[i], linewidth=0.8, color='darkgreen', alpha=0.8)
        axes[i].set_title(f'Detail Coefficients (Level {level_num})',
                         fontsize=12, fontweight='bold')
        axes[i].grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time', fontsize=12)
    plt.tight_layout()
    return fig


def denoise_with_wavelet(timeseries: np.ndarray,
                        wavelet: str = 'db4',
                        level: Optional[int] = None,
                        threshold_type: str = 'soft') -> np.ndarray:
    """
    Denoise time series using wavelet thresholding

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    wavelet : str
        Wavelet to use
    level : int, optional
        Decomposition level
    threshold_type : str
        'soft' or 'hard' thresholding

    Returns:
    --------
    array: Denoised time series
    """
    if not PYWT_AVAILABLE:
        raise ImportError("PyWavelets is required. Install with: pip install PyWavelets")

    # Decompose
    coefficients = discrete_wavelet_transform(timeseries, wavelet, level)

    # Calculate threshold (universal threshold)
    sigma = np.median(np.abs(coefficients[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(timeseries)))

    # Threshold detail coefficients
    new_coefficients = [coefficients[0]]  # Keep approximation
    for detail in coefficients[1:]:
        if threshold_type == 'soft':
            thresholded = pywt.threshold(detail, threshold, mode='soft')
        else:
            thresholded = pywt.threshold(detail, threshold, mode='hard')
        new_coefficients.append(thresholded)

    # Reconstruct
    denoised = pywt.waverec(new_coefficients, wavelet)

    # Ensure same length as original
    denoised = denoised[:len(timeseries)]

    return denoised


if __name__ == "__main__":
    print("Spectral and Wavelet Analysis Module")
    print("\nFunctions available:")
    print("  - compute_fft(timeseries, sampling_rate)")
    print("  - compute_periodogram(timeseries, sampling_rate)")
    print("  - compute_welch_psd(timeseries, sampling_rate)")
    print("  - plot_frequency_spectrum(timeseries, sampling_rate, title)")
    print("  - plot_psd(timeseries, sampling_rate, title)")
    print("  - identify_seasonal_periods(timeseries, sampling_rate)")
    print("  - continuous_wavelet_transform(timeseries, scales, wavelet)")
    print("  - plot_wavelet_scalogram(timeseries, datetime_index, scales, wavelet)")
    print("  - discrete_wavelet_transform(timeseries, wavelet, level)")
    print("  - plot_dwt_decomposition(timeseries, wavelet, level)")
    print("  - denoise_with_wavelet(timeseries, wavelet, level)")
