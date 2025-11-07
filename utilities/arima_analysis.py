"""
ARIMA Time Series Analysis Module

This module provides tools for ARIMA modeling of time series data:
- ACF/PACF visualization
- Stationarity testing
- ARIMA model grid search
- Residual diagnostics
- Forecasting with confidence intervals
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from typing import Dict, List, Tuple, Optional, Union, Any
import warnings

# Type checking imports
try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf  # type: ignore
    from statsmodels.tsa.stattools import adfuller  # type: ignore
    from statsmodels.tsa.arima.model import ARIMA  # type: ignore
    from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore
except ImportError as e:
    raise ImportError(
        "statsmodels is required for ARIMA analysis. "
        "Install with: pip install statsmodels"
    ) from e

from scipy import stats
from itertools import product

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


def plot_acf_pacf(timeseries: np.ndarray,
                  lags: int = 40,
                  title_prefix: str = "") -> plt.Figure:
    """
    Plot ACF and PACF for a time series

    Parameters:
    -----------
    timeseries : array-like
        The time series data
    lags : int
        Number of lags to show
    title_prefix : str
        Prefix for plot titles

    Returns:
    --------
    matplotlib Figure
    """
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # ACF plot
    plot_acf(timeseries, lags=lags, ax=axes[0])
    axes[0].set_title(f'{title_prefix}Autocorrelation Function (ACF)',
                      fontsize=14, fontweight='bold', pad=15)
    axes[0].set_xlabel('Lag', fontsize=12)
    axes[0].set_ylabel('ACF', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # PACF plot
    plot_pacf(timeseries, lags=lags, ax=axes[1], method='ywm')
    axes[1].set_title(f'{title_prefix}Partial Autocorrelation Function (PACF)',
                      fontsize=14, fontweight='bold', pad=15)
    axes[1].set_xlabel('Lag', fontsize=12)
    axes[1].set_ylabel('PACF', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def check_stationarity(timeseries: np.ndarray,
                       name: str = "Time Series") -> Tuple:
    """
    Perform Augmented Dickey-Fuller test for stationarity

    Parameters:
    -----------
    timeseries : array-like
        The time series data
    name : str
        Name of the series for display

    Returns:
    --------
    tuple : ADF test results
    """
    print(f"\nAugmented Dickey-Fuller Test for {name}")
    print("-" * 60)

    result = adfuller(timeseries, autolag='AIC')

    print(f'ADF Statistic: {result[0]:.6f}')
    print(f'p-value: {result[1]:.6f}')
    print(f'Number of Lags Used: {result[2]}')
    print(f'Number of Observations: {result[3]}')

    print('\nCritical Values:')
    for key, value in result[4].items():
        print(f'  {key}: {value:.3f}')

    # Interpretation
    if result[1] <= 0.05:
        print(f"\n✓ Result: Series is STATIONARY (p-value = {result[1]:.6f} <= 0.05)")
        print("  → Reject null hypothesis (series has no unit root)")
    else:
        print(f"\n✗ Result: Series is NON-STATIONARY (p-value = {result[1]:.6f} > 0.05)")
        print("  → Fail to reject null hypothesis (series has unit root)")
        print("  → Consider differencing the series")

    return result


def arima_grid_search(timeseries: np.ndarray,
                     p_range: range,
                     d_range: range,
                     q_range: range,
                     verbose: bool = True) -> pd.DataFrame:
    """
    Perform grid search to find best ARIMA model

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    p_range : range
        Range of p values to try (AR order)
    d_range : range
        Range of d values to try (differencing order)
    q_range : range
        Range of q values to try (MA order)
    verbose : bool
        Print progress

    Returns:
    --------
    DataFrame with results sorted by AIC
    """
    results = []
    total_models = len(p_range) * len(d_range) * len(q_range)

    if verbose:
        print(f"Testing {total_models} ARIMA models...")
        print("=" * 70)

    model_count = 0

    for p, d, q in product(p_range, d_range, q_range):
        model_count += 1

        try:
            model = ARIMA(timeseries, order=(p, d, q))
            fitted_model = model.fit()

            results.append({
                'p': p,
                'd': d,
                'q': q,
                'AIC': fitted_model.aic,
                'BIC': fitted_model.bic,
                'HQIC': fitted_model.hqic,
                'log_likelihood': fitted_model.llf
            })

            if verbose and model_count % 10 == 0:
                print(f"Progress: {model_count}/{total_models} models tested...")

        except Exception as e:
            if verbose and model_count % 20 == 0:
                print(f"  Skipped ARIMA({p},{d},{q})")
            continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AIC').reset_index(drop=True)

    if verbose:
        print("\n" + "=" * 70)
        print(f"✓ Grid search complete! Tested {len(results_df)} successful models")
        print("=" * 70)

    return results_df


def plot_residual_diagnostics(fitted_model,
                              title: str = "ARIMA Model") -> plt.Figure:
    """
    Plot diagnostic plots for ARIMA model residuals

    Parameters:
    -----------
    fitted_model : fitted ARIMA model
        The fitted model with residuals
    title : str
        Title for the plots

    Returns:
    --------
    matplotlib Figure
    """
    residuals = fitted_model.resid

    fig = plt.figure(figsize=(16, 12))

    ax1 = plt.subplot(2, 2, 1)
    ax2 = plt.subplot(2, 2, 2)
    ax3 = plt.subplot(2, 2, 3)
    ax4 = plt.subplot(2, 2, 4)

    # 1. Residuals over time
    ax1.plot(residuals, linewidth=0.5, color='darkblue', alpha=0.7)
    ax1.axhline(y=0, color='red', linestyle='--', alpha=0.5)
    ax1.set_title(f'{title} - Residuals Over Time',
                  fontsize=12, fontweight='bold')
    ax1.set_xlabel('Observation', fontsize=10)
    ax1.set_ylabel('Residual', fontsize=10)
    ax1.grid(True, alpha=0.3)

    # 2. Histogram
    ax2.hist(residuals, bins=50, density=True, alpha=0.7,
             color='steelblue', edgecolor='black')
    mu, sigma = residuals.mean(), residuals.std()
    x = np.linspace(residuals.min(), residuals.max(), 100)
    ax2.plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2),
            'r-', linewidth=2, label=f'Normal(μ={mu:.1f}, σ={sigma:.1f})')
    ax2.set_title(f'{title} - Residuals Distribution',
                  fontsize=12, fontweight='bold')
    ax2.set_xlabel('Residual', fontsize=10)
    ax2.set_ylabel('Density', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y')

    # 3. Q-Q plot
    stats.probplot(residuals, dist="norm", plot=ax3)    # type: ignore
    ax3.set_title(f'{title} - Q-Q Plot', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)

    # 4. ACF of residuals
    plot_acf(residuals, lags=40, ax=ax4, alpha=0.05)
    ax4.set_title(f'{title} - Residuals ACF', fontsize=12, fontweight='bold')
    ax4.set_xlabel('Lag', fontsize=10)
    ax4.set_ylabel('ACF', fontsize=10)
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def print_residual_diagnostics(fitted_model) -> None:
    """
    Print statistical diagnostics for model residuals

    Parameters:
    -----------
    fitted_model : fitted ARIMA model
        The fitted model with residuals
    """
    print("\n" + "=" * 70)
    print("RESIDUAL DIAGNOSTICS")
    print("=" * 70)

    residuals = fitted_model.resid

    print(f"\nResidual Statistics:")
    print(f"  Mean: {residuals.mean():.6f} (should be close to 0)")
    print(f"  Std Dev: {residuals.std():.2f}")
    print(f"  Min: {residuals.min():.2f}")
    print(f"  Max: {residuals.max():.2f}")

    # Ljung-Box test
    lb_test = acorr_ljungbox(residuals, lags=[10, 20, 30], return_df=True)
    print(f"\nLjung-Box Test (tests for residual autocorrelation):")
    print(lb_test)
    print("\n  → If p-values > 0.05, residuals are white noise (good!)")


def plot_forecast(original_data: np.ndarray,
                 fitted_model,
                 forecast_steps: int = 365,
                 datetime_index: Optional[pd.DatetimeIndex] = None,
                 title: str = "") -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """
    Plot original data, fitted values, and forecast

    Parameters:
    -----------
    original_data : array-like
        Original time series
    fitted_model : fitted ARIMA model
        The fitted model
    forecast_steps : int
        Number of steps to forecast
    datetime_index : DatetimeIndex, optional
        Datetime index for x-axis
    title : str
        Plot title

    Returns:
    --------
    tuple: (figure, forecast_values, confidence_intervals)
    """
    fitted_values = fitted_model.fittedvalues
    forecast_result = fitted_model.forecast(steps=forecast_steps)

    forecast_df = fitted_model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast_df.conf_int()

    fig, ax = plt.subplots(figsize=(18, 7))

    # Get differencing order
    d_order = fitted_model.model.order[1]

    if datetime_index is not None:
        x_original: Union[pd.DatetimeIndex, np.ndarray] = datetime_index
        x_fitted: Union[pd.DatetimeIndex, np.ndarray] = datetime_index[d_order:]
        # Slice fitted_values to match x_fitted length
        fitted_values = fitted_values[d_order:]
        # Use [-1] instead of .iloc[-1] for DatetimeIndex
        last_date = datetime_index.iloc[-1] # type: ignore
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=forecast_steps, freq='D')
        x_forecast: Union[pd.DatetimeIndex, np.ndarray] = forecast_dates
    else:
        x_original = np.arange(len(original_data))
        x_fitted = np.arange(d_order, len(original_data))
        # Slice fitted_values to match x_fitted length
        fitted_values = fitted_values[d_order:]
        x_forecast = np.arange(len(original_data),
                              len(original_data) + forecast_steps)

    # Plot
    ax.plot(x_original, original_data, linewidth=1,
            color='darkgreen', alpha=0.6, label='Original Data')
    ax.plot(x_fitted, fitted_values, linewidth=1.5,
            color='blue', alpha=0.8, label='Fitted Values')
    ax.plot(x_forecast, forecast_result, linewidth=2,
            color='red', label=f'{forecast_steps}-Day Forecast')
    ax.fill_between(x_forecast,
                    forecast_ci.iloc[:, 0],
                    forecast_ci.iloc[:, 1],
                    alpha=0.3, color='red', label='95% Confidence Interval')

    ax.set_xlabel('Date' if datetime_index is not None else 'Time', fontsize=13)
    ax.set_ylabel('Number of Births', fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    if datetime_index is not None:
        plt.xticks(rotation=45)

    plt.tight_layout()
    return fig, forecast_result, forecast_ci


def compare_top_models(timeseries: np.ndarray,
                      results_df: pd.DataFrame,
                      top_n: int = 5) -> plt.Figure:
    """
    Fit and visually compare top N models from grid search

    Parameters:
    -----------
    timeseries : array-like
        The time series data
    results_df : DataFrame
        Results from arima_grid_search()
    top_n : int
        Number of top models to compare

    Returns:
    --------
    matplotlib Figure
    """
    print(f"\nComparing top {top_n} ARIMA models...")
    print("=" * 70)

    fig, axes = plt.subplots(top_n, 1, figsize=(18, 5*top_n))

    if top_n == 1:
        axes = [axes]

    for idx in range(top_n):
        model_params = results_df.iloc[idx]
        p, d, q = int(model_params['p']), int(model_params['d']), int(model_params['q'])

        # Fit model
        model = ARIMA(timeseries, order=(p, d, q))
        fitted = model.fit()

        # Plot
        ax = axes[idx]
        ax.plot(timeseries, linewidth=0.8, alpha=0.6,
                color='darkgreen', label='Original')
        ax.plot(fitted.fittedvalues, linewidth=1.5, alpha=0.8,
                color='blue', label='Fitted')

        model_info = f"ARIMA({p},{d},{q}) | AIC: {model_params['AIC']:.2f} | BIC: {model_params['BIC']:.2f}"
        ax.set_title(f"Model {idx+1}: {model_info}",
                    fontsize=13, fontweight='bold', pad=10)
        ax.set_ylabel('Births', fontsize=11)
        ax.legend(loc='best', fontsize=10)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel('Time', fontsize=12)
    plt.tight_layout()
    return fig


def plot_differenced_series(original_series: pd.Series,
                           datetime_index: pd.DatetimeIndex,
                           order: int = 1) -> Tuple[plt.Figure, pd.Series]:
    """
    Plot original and differenced series side by side

    Parameters:
    -----------
    original_series : Series
        The original time series
    datetime_index : DatetimeIndex
        Datetime index for plotting
    order : int
        Order of differencing (1 for first difference, 2 for second, etc.)

    Returns:
    --------
    tuple: (figure, differenced_series)
    """
    # Apply differencing
    diff_series = original_series.diff(order).dropna()

    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    # Original series
    axes[0].plot(datetime_index, original_series,
                 linewidth=0.8, color='darkgreen', alpha=0.7)
    axes[0].set_title('Original Series',
                      fontsize=14, fontweight='bold', pad=15)
    axes[0].set_ylabel('Value', fontsize=12)
    axes[0].grid(True, alpha=0.3)

    # Differenced series
    axes[1].plot(datetime_index[order:], diff_series,
                 linewidth=0.8, color='darkblue', alpha=0.7)
    axes[1].set_title(f'Differenced Series (order={order})',
                      fontsize=14, fontweight='bold', pad=15)
    axes[1].set_ylabel('Difference', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig, diff_series


# Example usage
if __name__ == "__main__":
    print("ARIMA Analysis Module")
    print("\nFunctions available:")
    print("  - plot_acf_pacf(timeseries, lags)")
    print("  - check_stationarity(timeseries, name)")
    print("  - arima_grid_search(timeseries, p_range, d_range, q_range)")
    print("  - plot_residual_diagnostics(fitted_model, title)")
    print("  - print_residual_diagnostics(fitted_model)")
    print("  - plot_forecast(data, model, forecast_steps, datetime_index, title)")
    print("  - compare_top_models(timeseries, results_df, top_n)")
    print("  - plot_differenced_series(series, datetime_index, order)")
