"""
SARIMA Time Series Analysis Module

This module provides tools for Seasonal ARIMA (SARIMA) modeling:
- Seasonal decomposition
- ACF/PACF with seasonal lags
- SARIMA model grid search
- Seasonal diagnostics
- Forecasting with seasonal patterns
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
    from statsmodels.tsa.seasonal import seasonal_decompose  # type: ignore
    from statsmodels.tsa.statespace.sarimax import SARIMAX  # type: ignore
    from statsmodels.stats.diagnostic import acorr_ljungbox  # type: ignore
    from statsmodels.tsa.stattools import adfuller  # type: ignore
except ImportError as e:
    raise ImportError(
        "statsmodels is required for SARIMA analysis. "
        "Install with: pip install statsmodels"
    ) from e

from scipy import stats
from itertools import product

warnings.filterwarnings('ignore')
sns.set_style("whitegrid")


def plot_seasonal_decomposition(timeseries: pd.Series,
                                period: int,
                                model: str = 'additive',
                                title_prefix: str = "") -> plt.Figure:
    """
    Decompose time series into trend, seasonal, and residual components

    Parameters:
    -----------
    timeseries : Series
        Time series data
    period : int
        Period of seasonality (e.g., 7 for weekly, 365 for yearly)
    model : str
        'additive' or 'multiplicative'
    title_prefix : str
        Prefix for plot titles

    Returns:
    --------
    matplotlib Figure
    """
    decomposition = seasonal_decompose(timeseries, model=model, period=period, extrapolate_trend='freq')

    fig, axes = plt.subplots(4, 1, figsize=(16, 12))

    # Original
    axes[0].plot(decomposition.observed, linewidth=0.8, color='black', alpha=0.8)
    axes[0].set_title(f'{title_prefix}Original Time Series', fontsize=13, fontweight='bold')
    axes[0].set_ylabel('Value', fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Trend
    axes[1].plot(decomposition.trend, linewidth=1.2, color='darkblue', alpha=0.8)
    axes[1].set_title('Trend Component', fontsize=13, fontweight='bold')
    axes[1].set_ylabel('Trend', fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # Seasonal
    axes[2].plot(decomposition.seasonal, linewidth=0.8, color='darkgreen', alpha=0.8)
    axes[2].set_title(f'Seasonal Component (Period={period})', fontsize=13, fontweight='bold')
    axes[2].set_ylabel('Seasonal', fontsize=11)
    axes[2].grid(True, alpha=0.3)

    # Residual
    axes[3].plot(decomposition.resid, linewidth=0.6, color='darkred', alpha=0.7)
    axes[3].axhline(y=0, color='black', linestyle='--', alpha=0.5)
    axes[3].set_title('Residual Component', fontsize=13, fontweight='bold')
    axes[3].set_ylabel('Residual', fontsize=11)
    axes[3].set_xlabel('Time', fontsize=11)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def plot_seasonal_acf_pacf(timeseries: np.ndarray,
                           seasonal_lags: int,
                           non_seasonal_lags: int = 40,
                           title_prefix: str = "") -> plt.Figure:
    """
    Plot ACF and PACF with focus on seasonal lags

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    seasonal_lags : int
        Seasonal period (e.g., 7 for weekly, 365 for yearly)
    non_seasonal_lags : int
        Number of non-seasonal lags to show
    title_prefix : str
        Prefix for plot titles

    Returns:
    --------
    matplotlib Figure
    """
    # Calculate total lags to show (at least 2x seasonal period)
    total_lags = max(non_seasonal_lags, 2 * seasonal_lags)

    fig, axes = plt.subplots(2, 2, figsize=(18, 10))

    # ACF - non-seasonal focus
    plot_acf(timeseries, lags=non_seasonal_lags, ax=axes[0, 0])
    axes[0, 0].set_title(f'{title_prefix}ACF (Non-seasonal Lags)',
                         fontsize=13, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)

    # ACF - seasonal focus
    plot_acf(timeseries, lags=total_lags, ax=axes[0, 1])
    axes[0, 1].set_title(f'{title_prefix}ACF (Including Seasonal Lags)',
                         fontsize=13, fontweight='bold')
    # Highlight seasonal lags
    for i in range(1, total_lags // seasonal_lags + 1):
        axes[0, 1].axvline(x=i*seasonal_lags, color='red', linestyle='--', alpha=0.3)
    axes[0, 1].grid(True, alpha=0.3)

    # PACF - non-seasonal focus
    plot_pacf(timeseries, lags=non_seasonal_lags, ax=axes[1, 0], method='ywm')
    axes[1, 0].set_title(f'{title_prefix}PACF (Non-seasonal Lags)',
                         fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # PACF - seasonal focus
    plot_pacf(timeseries, lags=total_lags, ax=axes[1, 1], method='ywm')
    axes[1, 1].set_title(f'{title_prefix}PACF (Including Seasonal Lags)',
                         fontsize=13, fontweight='bold')
    # Highlight seasonal lags
    for i in range(1, total_lags // seasonal_lags + 1):
        axes[1, 1].axvline(x=i*seasonal_lags, color='red', linestyle='--', alpha=0.3)
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def sarima_grid_search(timeseries: np.ndarray,
                      p_range: range,
                      d_range: range,
                      q_range: range,
                      P_range: range,
                      D_range: range,
                      Q_range: range,
                      s: int,
                      verbose: bool = True) -> pd.DataFrame:
    """
    Perform grid search to find best SARIMA model

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    p_range : range
        Non-seasonal AR order range
    d_range : range
        Non-seasonal differencing order range
    q_range : range
        Non-seasonal MA order range
    P_range : range
        Seasonal AR order range
    D_range : range
        Seasonal differencing order range
    Q_range : range
        Seasonal MA order range
    s : int
        Seasonal period (e.g., 7 for weekly, 365 for yearly)
    verbose : bool
        Print progress

    Returns:
    --------
    DataFrame with results sorted by AIC
    """
    results = []

    # Generate all combinations
    pdq = list(product(p_range, d_range, q_range))
    PDQs = list(product(P_range, D_range, Q_range))

    total_models = len(pdq) * len(PDQs)

    if verbose:
        print(f"Testing {total_models} SARIMA models...")
        print(f"Seasonal period: {s}")
        print("=" * 70)

    model_count = 0

    for order in pdq:
        for seasonal_order in PDQs:
            model_count += 1

            try:
                model = SARIMAX(timeseries,
                              order=order,
                              seasonal_order=seasonal_order + (s,),
                              enforce_stationarity=False,
                              enforce_invertibility=False)
                fitted_model = model.fit(disp=False, maxiter=200)

                results.append({
                    'p': order[0], 'd': order[1], 'q': order[2],
                    'P': seasonal_order[0], 'D': seasonal_order[1], 'Q': seasonal_order[2],
                    's': s,
                    'AIC': fitted_model.aic,
                    'BIC': fitted_model.bic,
                    'HQIC': fitted_model.hqic,
                    'log_likelihood': fitted_model.llf
                })

                if verbose and model_count % 20 == 0:
                    print(f"Progress: {model_count}/{total_models} models tested...")

            except Exception as e:
                if verbose and model_count % 50 == 0:
                    print(f"  Skipped SARIMA{order}x{seasonal_order}[{s}]")
                continue

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('AIC').reset_index(drop=True)

    if verbose:
        print("\n" + "=" * 70)
        print(f"✓ Grid search complete! Tested {len(results_df)} successful models")
        print("=" * 70)

    return results_df


def fit_sarima_model(timeseries: np.ndarray,
                    order: Tuple[int, int, int],
                    seasonal_order: Tuple[int, int, int, int],
                    verbose: bool = True):
    """
    Fit a SARIMA model with given parameters

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    order : tuple
        (p, d, q) - non-seasonal order
    seasonal_order : tuple
        (P, D, Q, s) - seasonal order
    verbose : bool
        Print model summary

    Returns:
    --------
    Fitted SARIMAX model
    """
    model = SARIMAX(timeseries,
                   order=order,
                   seasonal_order=seasonal_order,
                   enforce_stationarity=False,
                   enforce_invertibility=False)

    fitted_model = model.fit(disp=False, maxiter=200)

    if verbose:
        print(f"\n{'='*70}")
        print(f"SARIMA{order}x{seasonal_order} Model Summary")
        print(f"{'='*70}")
        print(fitted_model.summary())

    return fitted_model


def plot_sarima_diagnostics(fitted_model,
                           title: str = "SARIMA Model") -> plt.Figure:
    """
    Plot diagnostic plots for SARIMA model residuals

    Parameters:
    -----------
    fitted_model : fitted SARIMAX model
        The fitted model
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
    stats.probplot(residuals, dist="norm", plot=ax3)  # type: ignore
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


def plot_sarima_forecast(original_data: np.ndarray,
                        fitted_model,
                        forecast_steps: int = 365,
                        datetime_index: Optional[pd.DatetimeIndex] = None,
                        title: str = "") -> Tuple[plt.Figure, np.ndarray, pd.DataFrame]:
    """
    Plot original data, fitted values, and SARIMA forecast

    Parameters:
    -----------
    original_data : array-like
        Original time series
    fitted_model : fitted SARIMAX model
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

    forecast_obj = fitted_model.get_forecast(steps=forecast_steps)
    forecast_ci = forecast_obj.conf_int()

    # Ensure forecast_ci is a DataFrame (handles different statsmodels versions)
    if isinstance(forecast_ci, np.ndarray):
        forecast_ci = pd.DataFrame(forecast_ci, columns=['lower', 'upper'])

    fig, ax = plt.subplots(figsize=(18, 7))

    # Get differencing order (both regular and seasonal)
    d_order = fitted_model.model.order[1]
    D_order = fitted_model.model.seasonal_order[1]
    s_period = fitted_model.model.seasonal_order[3]

    # Total initial observations to skip
    skip = d_order + (D_order * s_period)

    if datetime_index is not None:
        x_original: Union[pd.DatetimeIndex, np.ndarray] = datetime_index
        x_fitted: Union[pd.DatetimeIndex, np.ndarray] = datetime_index[skip:]
        # Slice fitted_values to match x_fitted length
        fitted_values = fitted_values[skip:]

        last_date = datetime_index.iloc[-1] # type: ignore
        forecast_dates = pd.date_range(start=last_date + pd.Timedelta(days=1),
                                       periods=forecast_steps, freq='D')
        x_forecast: Union[pd.DatetimeIndex, np.ndarray] = forecast_dates
    else:
        x_original = np.arange(len(original_data))
        x_fitted = np.arange(skip, len(original_data))
        # Slice fitted_values to match x_fitted length
        fitted_values = fitted_values[skip:]
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


def print_sarima_diagnostics(fitted_model) -> None:
    """
    Print statistical diagnostics for SARIMA model residuals

    Parameters:
    -----------
    fitted_model : fitted SARIMAX model
        The fitted model
    """
    print("\n" + "=" * 70)
    print("SARIMA RESIDUAL DIAGNOSTICS")
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


def compare_arima_sarima(timeseries: np.ndarray,
                        arima_order: Tuple[int, int, int],
                        sarima_order: Tuple[int, int, int],
                        seasonal_order: Tuple[int, int, int, int],
                        datetime_index: Optional[pd.DatetimeIndex] = None) -> plt.Figure:
    """
    Compare ARIMA and SARIMA model fits

    Parameters:
    -----------
    timeseries : array-like
        Time series data
    arima_order : tuple
        (p, d, q) for ARIMA
    sarima_order : tuple
        (p, d, q) for SARIMA
    seasonal_order : tuple
        (P, D, Q, s) for SARIMA
    datetime_index : DatetimeIndex, optional
        Datetime index for plotting

    Returns:
    --------
    matplotlib Figure
    """
    from statsmodels.tsa.arima.model import ARIMA

    # Fit ARIMA
    arima_model = ARIMA(timeseries, order=arima_order)
    arima_fit = arima_model.fit()

    # Fit SARIMA
    sarima_model = SARIMAX(timeseries, order=sarima_order, seasonal_order=seasonal_order)
    sarima_fit = sarima_model.fit(disp=False)

    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(18, 12))

    # ARIMA plot
    ax1 = axes[0]
    if datetime_index is not None:
        x_data = datetime_index
    else:
        x_data = np.arange(len(timeseries)) # type: ignore

    ax1.plot(x_data, timeseries, linewidth=0.8, alpha=0.5,
            color='gray', label='Original Data')

    arima_skip = arima_order[1]
    ax1.plot(x_data[arima_skip:], arima_fit.fittedvalues[arima_skip:],
            linewidth=1.5, color='blue', alpha=0.8, label='ARIMA Fitted')

    ax1.set_title(f'ARIMA{arima_order} | AIC: {arima_fit.aic:.2f}',
                 fontsize=14, fontweight='bold')
    ax1.set_ylabel('Value', fontsize=12)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # SARIMA plot
    ax2 = axes[1]
    ax2.plot(x_data, timeseries, linewidth=0.8, alpha=0.5,
            color='gray', label='Original Data')

    sarima_skip = sarima_order[1] + (seasonal_order[1] * seasonal_order[3])
    ax2.plot(x_data[sarima_skip:], sarima_fit.fittedvalues[sarima_skip:],
            linewidth=1.5, color='darkgreen', alpha=0.8, label='SARIMA Fitted')

    ax2.set_title(f'SARIMA{sarima_order}x{seasonal_order} | AIC: {sarima_fit.aic:.2f}',
                 fontsize=14, fontweight='bold')
    ax2.set_ylabel('Value', fontsize=12)
    ax2.set_xlabel('Date' if datetime_index is not None else 'Time', fontsize=12)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Print comparison
    print("\n" + "=" * 70)
    print("ARIMA vs SARIMA COMPARISON")
    print("=" * 70)
    print(f"ARIMA{arima_order}:")
    print(f"  AIC: {arima_fit.aic:.2f}")
    print(f"  BIC: {arima_fit.bic:.2f}")
    print(f"\nSARIMA{sarima_order}x{seasonal_order}:")
    print(f"  AIC: {sarima_fit.aic:.2f}")
    print(f"  BIC: {sarima_fit.bic:.2f}")

    if sarima_fit.aic < arima_fit.aic:
        print(f"\n✓ SARIMA is better (lower AIC by {arima_fit.aic - sarima_fit.aic:.2f})")
    else:
        print(f"\n✓ ARIMA is better (lower AIC by {sarima_fit.aic - arima_fit.aic:.2f})")
    print("=" * 70)

    return fig


if __name__ == "__main__":
    print("SARIMA Time Series Analysis Module")
    print("\nFunctions available:")
    print("  - plot_seasonal_decomposition(timeseries, period, model)")
    print("  - plot_seasonal_acf_pacf(timeseries, seasonal_lags)")
    print("  - sarima_grid_search(timeseries, p_range, d_range, q_range, P_range, D_range, Q_range, s)")
    print("  - fit_sarima_model(timeseries, order, seasonal_order)")
    print("  - plot_sarima_diagnostics(fitted_model, title)")
    print("  - plot_sarima_forecast(data, model, forecast_steps, datetime_index, title)")
    print("  - print_sarima_diagnostics(fitted_model)")
    print("  - compare_arima_sarima(timeseries, arima_order, sarima_order, seasonal_order)")
