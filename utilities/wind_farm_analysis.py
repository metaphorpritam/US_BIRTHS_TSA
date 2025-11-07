import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta

sns.set_style("whitegrid")


def detect_time_interval(row):
    """
    Auto-detect the actual time interval between data points

    Parameters:
    -----------
    row : pandas Series
        Row with 'start', 'end', 'target' columns

    Returns:
    --------
    dict with interval information
    """
    start = pd.to_datetime(row['start'])
    end = pd.to_datetime(row['end'])
    n_points = len(row['target'])

    # Calculate interval
    total_seconds = (end - start).total_seconds()
    interval_seconds = total_seconds / (n_points - 1)

    # Determine best frequency representation
    if interval_seconds < 1:
        freq_str = f"{interval_seconds*1000:.0f}ms"
        freq_pandas = f"{int(interval_seconds*1000)}ms"
    elif interval_seconds < 60:
        freq_str = f"{interval_seconds:.0f}s"
        freq_pandas = f"{int(interval_seconds)}s"
    elif interval_seconds < 3600:
        freq_str = f"{interval_seconds/60:.1f}min"
        freq_pandas = f"{int(interval_seconds/60)}min"
    else:
        freq_str = f"{interval_seconds/3600:.1f}h"
        freq_pandas = f"{int(interval_seconds/3600)}H"

    return {
        'start': start,
        'end': end,
        'n_points': n_points,
        'total_duration': end - start,
        'interval_seconds': interval_seconds,
        'interval_str': freq_str,
        'freq_pandas': freq_pandas,
        'freq_column': row['freq']
    }


def expand_wind_timeseries(row, freq=None):
    """
    Expand wind farm time series into DataFrame format

    Parameters:
    -----------
    row : pandas Series
        Row with 'start', 'end', 'target' columns
    freq : str, optional
        Pandas frequency string. If None, auto-detect

    Returns:
    --------
    pandas DataFrame with datetime and power columns
    """
    start = pd.to_datetime(row['start'])
    end = pd.to_datetime(row['end'])
    target_values = row['target']
    n_points = len(target_values)

    # Auto-detect frequency if not provided
    if freq is None:
        interval_info = detect_time_interval(row)
        freq = interval_info['freq_pandas']

    # Create datetime index
    date_range = pd.date_range(start=start, end=end, periods=n_points)

    # Create DataFrame
    df = pd.DataFrame({
        'datetime': date_range,
        'power': target_values
    })

    return df


def aggregate_to_daily(wind_df):
    """
    Aggregate wind power data to daily statistics

    Parameters:
    -----------
    wind_df : DataFrame
        Expanded wind time series with 'datetime' and 'power' columns

    Returns:
    --------
    DataFrame with daily statistics
    """
    # Add date column
    wind_df['date'] = wind_df['datetime'].dt.date

    # Calculate daily statistics
    daily_stats = wind_df.groupby('date')['power'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std'),
        ('p25', lambda x: np.percentile(x, 25)),
        ('p75', lambda x: np.percentile(x, 75)),
        ('p10', lambda x: np.percentile(x, 10)),
        ('p90', lambda x: np.percentile(x, 90)),
        ('sum', 'sum'),  # Total daily production
        ('count', 'count')  # Number of readings
    ]).reset_index()

    daily_stats['date'] = pd.to_datetime(daily_stats['date'])

    return daily_stats


def plot_full_timeseries(wind_df, title="Wind Power Production", max_points=10000):
    """
    Plot the complete time series with automatic downsampling for large datasets

    Parameters:
    -----------
    wind_df : DataFrame
        Time series with 'datetime' and 'power' columns
    max_points : int
        Maximum points to plot (for performance)
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    # Downsample if necessary
    if len(wind_df) > max_points:
        # Use every nth point
        step = len(wind_df) // max_points
        plot_df = wind_df.iloc[::step]
        title_suffix = f" (showing 1 in {step} points)"
    else:
        plot_df = wind_df
        title_suffix = ""

    # Plot
    ax.plot(plot_df['datetime'], plot_df['power'],
            linewidth=0.5, color='steelblue', alpha=0.7)

    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Power Production (MW)', fontsize=13)
    ax.set_title(f'{title}{title_suffix}\n{wind_df["datetime"].min().date()} to {wind_df["datetime"].max().date()}',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_daily_statistics(daily_stats):
    """
    Plot daily mean/median/percentiles of wind power production

    Parameters:
    -----------
    daily_stats : DataFrame
        Daily statistics from aggregate_to_daily()
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot mean and median
    ax.plot(daily_stats['date'], daily_stats['mean'],
            linewidth=2.5, color='darkblue', label='Daily Mean', zorder=4)
    ax.plot(daily_stats['date'], daily_stats['median'],
            linewidth=2, color='coral', label='Daily Median',
            linestyle='--', alpha=0.8, zorder=3)

    # Add percentile bands
    ax.fill_between(daily_stats['date'],
                    daily_stats['p25'], daily_stats['p75'],
                    alpha=0.4, color='steelblue', label='P25-P75 (IQR)', zorder=2)
    ax.fill_between(daily_stats['date'],
                    daily_stats['p10'], daily_stats['p90'],
                    alpha=0.2, color='lightblue', label='P10-P90', zorder=1)

    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Power Production (MW)', fontsize=13)
    ax.set_title('Daily Wind Power Production Statistics\n(Mean, Median, and Percentiles)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add statistics box
    stats_text = f"Overall Statistics:\n"
    stats_text += f"Avg Daily Mean: {daily_stats['mean'].mean():.2f} MW\n"
    stats_text += f"Avg Daily Median: {daily_stats['median'].mean():.2f} MW\n"
    stats_text += f"Peak Daily Mean: {daily_stats['mean'].max():.2f} MW\n"
    stats_text += f"Min Daily Mean: {daily_stats['mean'].min():.2f} MW"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    return fig


def plot_multi_resolution_view(wind_df, date_ranges=None):
    """
    Create multi-resolution view showing data at different granularities

    Parameters:
    -----------
    wind_df : DataFrame
        Full time series
    date_ranges : list of tuples
        [(start, end, title), ...] for different zoom levels
    """
    if date_ranges is None:
        # Auto-generate zoom levels
        start = wind_df['datetime'].min()
        end = wind_df['datetime'].max()
        total_days = (end - start).days

        # Define zoom levels
        date_ranges = [
            (start, end, "Full Range"),
            (start, start + timedelta(days=min(30, total_days)), "First Month"),
            (start, start + timedelta(days=min(7, total_days)), "First Week"),
            (start, start + timedelta(days=1), "First Day")
        ]

    n_plots = len(date_ranges)
    fig, axes = plt.subplots(n_plots, 1, figsize=(18, 5*n_plots))

    if n_plots == 1:
        axes = [axes]

    for ax, (start_date, end_date, title) in zip(axes, date_ranges):
        # Filter data for this range
        mask = (wind_df['datetime'] >= start_date) & (wind_df['datetime'] <= end_date)
        subset = wind_df[mask]

        if len(subset) == 0:
            ax.text(0.5, 0.5, 'No data in this range',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Downsample if too many points
        max_points = 10000
        if len(subset) > max_points:
            step = len(subset) // max_points
            plot_subset = subset.iloc[::step]
            title_suffix = f" (1 in {step} points)"
        else:
            plot_subset = subset
            title_suffix = ""

        # Plot
        ax.plot(plot_subset['datetime'], plot_subset['power'],
               linewidth=0.8, color='steelblue', alpha=0.8)
        ax.fill_between(plot_subset['datetime'], plot_subset['power'],
                       alpha=0.3, color='lightblue')

        ax.set_ylabel('Power (MW)', fontsize=12)
        ax.set_title(f'{title}{title_suffix}\n{subset["datetime"].min()} to {subset["datetime"].max()}',
                    fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')

    axes[-1].set_xlabel('DateTime', fontsize=12)
    plt.tight_layout()

    return fig


def plot_hourly_patterns(wind_df):
    """
    Plot average power production by hour of day

    Parameters:
    -----------
    wind_df : DataFrame
        Wind time series with datetime and power columns
    """
    # Add hour column
    wind_df['hour'] = wind_df['datetime'].dt.hour

    # Calculate hourly statistics
    hourly_stats = wind_df.groupby('hour')['power'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median')
    ]).reset_index()

    fig, ax = plt.subplots(figsize=(14, 6))

    # Plot mean with error bars
    ax.plot(hourly_stats['hour'], hourly_stats['mean'],
           marker='o', linewidth=2.5, markersize=8,
           color='darkblue', label='Mean')
    ax.fill_between(hourly_stats['hour'],
                    hourly_stats['mean'] - hourly_stats['std'],
                    hourly_stats['mean'] + hourly_stats['std'],
                    alpha=0.3, color='lightblue', label='±1 Std Dev')

    ax.plot(hourly_stats['hour'], hourly_stats['median'],
           marker='s', linewidth=2, markersize=6,
           color='coral', linestyle='--', label='Median')

    ax.set_xlabel('Hour of Day', fontsize=13)
    ax.set_ylabel('Average Power Production (MW)', fontsize=13)
    ax.set_title('Wind Power Production by Hour of Day',
                fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(range(24))
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_daily_production_heatmap(wind_df):
    """
    Create a heatmap showing power production patterns over time

    Parameters:
    -----------
    wind_df : DataFrame
        Wind time series
    """
    # Create date and hour columns
    wind_df['date'] = wind_df['datetime'].dt.date
    wind_df['hour'] = wind_df['datetime'].dt.hour

    # Aggregate by date and hour
    heatmap_data = wind_df.groupby(['date', 'hour'])['power'].mean().reset_index()

    # Pivot for heatmap
    pivot = heatmap_data.pivot(index='hour', columns='date', values='power')

    # Create plot
    fig, ax = plt.subplots(figsize=(18, 8))

    sns.heatmap(pivot, cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Power Production (MW)'},
                xticklabels=False)  # Too many dates

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Hour of Day', fontsize=12)
    ax.set_title('Wind Power Production Heatmap\n(Hourly Patterns Across Days)',
                 fontsize=15, fontweight='bold', pad=20)

    # Add date range on x-axis
    dates = sorted(heatmap_data['date'].unique())
    n_ticks = 10
    tick_positions = np.linspace(0, len(dates)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[i] for i in tick_positions], rotation=45, ha='right')

    plt.tight_layout()

    return fig


def analyze_wind_data(row):
    """
    Complete analysis pipeline for wind farm data

    Parameters:
    -----------
    row : pandas Series
        Wind farm data row

    Returns:
    --------
    dict with expanded_df, daily_stats, and info
    """
    print("="*70)
    print("WIND FARM DATA ANALYSIS")
    print("="*70)

    # Step 1: Detect time interval
    print("\nStep 1: Detecting time interval...")
    interval_info = detect_time_interval(row)

    print(f"\nTime Series Information:")
    print(f"  Start: {interval_info['start']}")
    print(f"  End: {interval_info['end']}")
    print(f"  Duration: {interval_info['total_duration']}")
    print(f"  Data points: {interval_info['n_points']:,}")
    print(f"  Interval: {interval_info['interval_str']}")
    print(f"  Frequency (pandas): {interval_info['freq_pandas']}")
    print(f"  Frequency (dataset): '{interval_info['freq_column']}'")

    # Step 2: Expand time series
    print("\nStep 2: Expanding time series...")
    wind_df = expand_wind_timeseries(row)
    print(f"  Created DataFrame with {len(wind_df):,} rows")

    # Step 3: Calculate daily statistics
    print("\nStep 3: Calculating daily statistics...")
    daily_stats = aggregate_to_daily(wind_df)
    print(f"  Generated statistics for {len(daily_stats)} days")

    # Step 4: Basic statistics
    print("\nOverall Statistics:")
    print(f"  Mean power: {wind_df['power'].mean():.2f} MW")
    print(f"  Median power: {wind_df['power'].median():.2f} MW")
    print(f"  Std dev: {wind_df['power'].std():.2f} MW")
    print(f"  Min power: {wind_df['power'].min():.2f} MW")
    print(f"  Max power: {wind_df['power'].max():.2f} MW")

    print("\nDaily Statistics:")
    print(f"  Avg daily mean: {daily_stats['mean'].mean():.2f} MW")
    print(f"  Avg daily total: {daily_stats['sum'].mean():.2f} MWh")
    print(f"  Peak daily mean: {daily_stats['mean'].max():.2f} MW")
    print(f"  Lowest daily mean: {daily_stats['mean'].min():.2f} MW")

    print("="*70)

    return {
        'wind_df': wind_df,
        'daily_stats': daily_stats,
        'interval_info': interval_info
    }


# Example usage
if __name__ == "__main__":
    print("Wind Farm Analysis Module")
    print("\nFunctions available:")
    print("  - detect_time_interval(row)")
    print("  - expand_wind_timeseries(row)")
    print("  - aggregate_to_daily(wind_df)")
    print("  - plot_full_timeseries(wind_df)")
    print("  - plot_daily_statistics(daily_stats)")
    print("  - plot_multi_resolution_view(wind_df)")
    print("  - plot_hourly_patterns(wind_df)")
    print("  - plot_daily_production_heatmap(wind_df)")
    print("  - analyze_wind_data(row)  # Complete pipeline")
