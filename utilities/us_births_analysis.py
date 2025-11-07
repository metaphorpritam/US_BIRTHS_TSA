import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import datetime, timedelta
import calendar

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
    elif interval_seconds < 86400:
        freq_str = f"{interval_seconds/3600:.1f}h"
        freq_pandas = f"{int(interval_seconds/3600)}H"
    else:
        freq_str = f"{interval_seconds/86400:.1f}d"
        freq_pandas = f"{int(interval_seconds/86400)}D"

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


def expand_births_timeseries(row, freq=None):
    """
    Expand US births time series into DataFrame format

    Parameters:
    -----------
    row : pandas Series
        Row with 'start', 'end', 'target' columns
    freq : str, optional
        Pandas frequency string. If None, auto-detect

    Returns:
    --------
    pandas DataFrame with datetime and births columns
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
        'births': target_values
    })

    # Add useful date components
    df['date'] = df['datetime'].dt.date
    df['year'] = df['datetime'].dt.year
    df['month'] = df['datetime'].dt.month
    df['day'] = df['datetime'].dt.day
    df['dayofweek'] = df['datetime'].dt.dayofweek
    df['dayofweek_name'] = df['datetime'].dt.day_name()
    df['month_name'] = df['datetime'].dt.month_name()
    df['is_weekend'] = df['dayofweek'].isin([5, 6])

    return df


def aggregate_to_monthly(births_df):
    """
    Aggregate births data to monthly statistics

    Parameters:
    -----------
    births_df : DataFrame
        Expanded births time series with 'datetime' and 'births' columns

    Returns:
    --------
    DataFrame with monthly statistics
    """
    # Add year-month column
    births_df['year_month'] = births_df['datetime'].dt.to_period('M')

    # Calculate monthly statistics
    monthly_stats = births_df.groupby('year_month')['births'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std'),
        ('sum', 'sum'),  # Total monthly births
        ('count', 'count')  # Number of days
    ]).reset_index()

    monthly_stats['year_month'] = monthly_stats['year_month'].dt.to_timestamp()

    return monthly_stats


def aggregate_to_yearly(births_df):
    """
    Aggregate births data to yearly statistics

    Parameters:
    -----------
    births_df : DataFrame
        Expanded births time series

    Returns:
    --------
    DataFrame with yearly statistics
    """
    yearly_stats = births_df.groupby('year')['births'].agg([
        ('mean', 'mean'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('std', 'std'),
        ('sum', 'sum'),  # Total yearly births
        ('count', 'count')  # Number of days
    ]).reset_index()

    return yearly_stats


def plot_full_timeseries(births_df, title="US Daily Births"):
    """
    Plot the complete time series of births

    Parameters:
    -----------
    births_df : DataFrame
        Time series with 'datetime' and 'births' columns
    title : str
        Plot title
    """
    fig, ax = plt.subplots(figsize=(18, 6))

    # Plot
    ax.plot(births_df['datetime'], births_df['births'],
            linewidth=0.8, color='darkgreen', alpha=0.7)
    ax.fill_between(births_df['datetime'], births_df['births'],
                     alpha=0.3, color='lightgreen')

    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Number of Births', fontsize=13)
    ax.set_title(f'{title}\n{births_df["datetime"].min().date()} to {births_df["datetime"].max().date()}',
                 fontsize=15, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_monthly_statistics(monthly_stats):
    """
    Plot monthly mean/median/sum of births

    Parameters:
    -----------
    monthly_stats : DataFrame
        Monthly statistics from aggregate_to_monthly()
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

    # Plot 1: Monthly average births
    ax1.plot(monthly_stats['year_month'], monthly_stats['mean'],
             linewidth=2.5, color='darkblue', label='Monthly Mean', marker='o')
    ax1.plot(monthly_stats['year_month'], monthly_stats['median'],
             linewidth=2, color='coral', label='Monthly Median',
             linestyle='--', alpha=0.8, marker='s')

    ax1.fill_between(monthly_stats['year_month'],
                     monthly_stats['min'], monthly_stats['max'],
                     alpha=0.2, color='lightblue', label='Min-Max Range')

    ax1.set_xlabel('Date', fontsize=13)
    ax1.set_ylabel('Average Daily Births', fontsize=13)
    ax1.set_title('Monthly Average Daily Births\n(Mean, Median, and Range)',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total monthly births
    ax2.bar(monthly_stats['year_month'], monthly_stats['sum'],
            width=20, color='darkgreen', alpha=0.7, edgecolor='black')

    ax2.set_xlabel('Date', fontsize=13)
    ax2.set_ylabel('Total Monthly Births', fontsize=13)
    ax2.set_title('Total Births Per Month',
                  fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    plt.tight_layout()

    return fig


def plot_yearly_trends(yearly_stats):
    """
    Plot yearly trends in births

    Parameters:
    -----------
    yearly_stats : DataFrame
        Yearly statistics from aggregate_to_yearly()
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))

    # Plot 1: Average daily births by year
    ax1.plot(yearly_stats['year'], yearly_stats['mean'],
             linewidth=3, color='darkblue', marker='o', markersize=8)
    ax1.fill_between(yearly_stats['year'], yearly_stats['mean'],
                     alpha=0.3, color='lightblue')

    ax1.set_xlabel('Year', fontsize=13)
    ax1.set_ylabel('Average Daily Births', fontsize=13)
    ax1.set_title('Average Daily Births by Year',
                  fontsize=15, fontweight='bold', pad=20)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Total yearly births
    ax2.bar(yearly_stats['year'], yearly_stats['sum'],
            color='darkgreen', alpha=0.7, edgecolor='black')

    ax2.set_xlabel('Year', fontsize=13)
    ax2.set_ylabel('Total Yearly Births', fontsize=13)
    ax2.set_title('Total Births by Year',
                  fontsize=15, fontweight='bold', pad=20)
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()

    return fig


def plot_day_of_week_patterns(births_df):
    """
    Analyze and plot births by day of week

    Parameters:
    -----------
    births_df : DataFrame
        Births time series with day of week information
    """
    # Calculate statistics by day of week
    dow_stats = births_df.groupby(['dayofweek', 'dayofweek_name'])['births'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('count', 'count')
    ]).reset_index()

    # Sort by day of week (Monday=0, Sunday=6)
    dow_stats = dow_stats.sort_values('dayofweek')

    fig, ax = plt.subplots(figsize=(14, 7))

    # Create bar plot with error bars
    x_pos = np.arange(len(dow_stats))
    bars = ax.bar(x_pos, dow_stats['mean'], yerr=dow_stats['std'],
                   capsize=5, alpha=0.7, edgecolor='black')

    # Color weekends differently
    for i, (idx, row) in enumerate(dow_stats.iterrows()):
        if row['dayofweek'] in [5, 6]:  # Saturday, Sunday
            bars[i].set_color('coral')
        else:
            bars[i].set_color('steelblue')

    ax.set_xlabel('Day of Week', fontsize=13)
    ax.set_ylabel('Average Number of Births', fontsize=13)
    ax.set_title('Average Births by Day of Week\n(With Standard Deviation)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(dow_stats['dayofweek_name'], rotation=45, ha='right')
    ax.grid(True, alpha=0.3, axis='y')

    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='steelblue', alpha=0.7, label='Weekday'),
        Patch(facecolor='coral', alpha=0.7, label='Weekend')
    ]
    ax.legend(handles=legend_elements, loc='best', fontsize=11)

    plt.tight_layout()

    return fig


def plot_monthly_seasonal_patterns(births_df):
    """
    Analyze seasonal patterns by month of year

    Parameters:
    -----------
    births_df : DataFrame
        Births time series
    """
    # Calculate statistics by month
    monthly_pattern = births_df.groupby(['month', 'month_name'])['births'].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median')
    ]).reset_index()

    # Sort by month number
    monthly_pattern = monthly_pattern.sort_values('month')

    fig, ax = plt.subplots(figsize=(14, 7))

    # Plot with error bands
    x_pos = np.arange(len(monthly_pattern))
    ax.plot(x_pos, monthly_pattern['mean'],
            marker='o', linewidth=2.5, markersize=10,
            color='darkgreen', label='Mean')
    ax.fill_between(x_pos,
                     monthly_pattern['mean'] - monthly_pattern['std'],
                     monthly_pattern['mean'] + monthly_pattern['std'],
                     alpha=0.3, color='lightgreen', label='±1 Std Dev')

    ax.plot(x_pos, monthly_pattern['median'],
            marker='s', linewidth=2, markersize=7,
            color='orange', linestyle='--', label='Median')

    ax.set_xlabel('Month', fontsize=13)
    ax.set_ylabel('Average Number of Births', fontsize=13)
    ax.set_title('Seasonal Pattern: Average Births by Month of Year',
                 fontsize=15, fontweight='bold', pad=20)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(monthly_pattern['month_name'], rotation=45, ha='right')
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def plot_births_heatmap(births_df):
    """
    Create a heatmap showing births by year and month

    Parameters:
    -----------
    births_df : DataFrame
        Births time series
    """
    # Aggregate by year and month
    heatmap_data = births_df.groupby(['year', 'month'])['births'].mean().reset_index()

    # Pivot for heatmap
    pivot = heatmap_data.pivot(index='month', columns='year', values='births')

    # Create plot
    fig, ax = plt.subplots(figsize=(16, 8))

    sns.heatmap(pivot, cmap='YlGnBu', ax=ax, annot=False,
                cbar_kws={'label': 'Average Daily Births'},
                fmt='.0f')

    ax.set_xlabel('Year', fontsize=13)
    ax.set_ylabel('Month', fontsize=13)
    ax.set_title('Average Daily Births Heatmap\n(By Year and Month)',
                 fontsize=15, fontweight='bold', pad=20)

    # Set month labels
    month_labels = [calendar.month_abbr[i] for i in range(1, 13)]
    ax.set_yticklabels(month_labels, rotation=0)

    plt.tight_layout()

    return fig


def plot_weekday_weekend_comparison(births_df):
    """
    Compare births on weekdays vs weekends

    Parameters:
    -----------
    births_df : DataFrame
        Births time series with is_weekend column
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

    # Statistics
    weekday_births = births_df[~births_df['is_weekend']]['births']
    weekend_births = births_df[births_df['is_weekend']]['births']

    # Plot 1: Distribution comparison
    ax1.hist(weekday_births, bins=50, alpha=0.6, label='Weekday',
             color='steelblue', edgecolor='black')
    ax1.hist(weekend_births, bins=50, alpha=0.6, label='Weekend',
             color='coral', edgecolor='black')

    ax1.set_xlabel('Number of Births', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Distribution of Daily Births:\nWeekday vs Weekend',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best', fontsize=11)
    ax1.grid(True, alpha=0.3, axis='y')

    # Plot 2: Box plot comparison
    data_to_plot = [weekday_births, weekend_births]
    bp = ax2.boxplot(data_to_plot, labels=['Weekday', 'Weekend'],
                     patch_artist=True, showmeans=True)

    # Color the boxes
    bp['boxes'][0].set_facecolor('steelblue')
    bp['boxes'][0].set_alpha(0.6)
    bp['boxes'][1].set_facecolor('coral')
    bp['boxes'][1].set_alpha(0.6)

    ax2.set_ylabel('Number of Births', fontsize=12)
    ax2.set_title('Births Distribution:\nWeekday vs Weekend',
                  fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    # Add statistics text
    stats_text = f"Weekday: μ={weekday_births.mean():.0f}, σ={weekday_births.std():.0f}\n"
    stats_text += f"Weekend: μ={weekend_births.mean():.0f}, σ={weekend_births.std():.0f}\n"
    stats_text += f"Difference: {weekday_births.mean() - weekend_births.mean():.0f} births/day"

    ax2.text(0.5, 0.02, stats_text, transform=ax2.transAxes,
             fontsize=10, verticalalignment='bottom', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()

    return fig


def plot_multi_year_comparison(births_df, years=None):
    """
    Compare birth patterns across different years

    Parameters:
    -----------
    births_df : DataFrame
        Births time series
    years : list, optional
        Specific years to compare. If None, use all years
    """
    if years is None:
        years = sorted(births_df['year'].unique())

    # Limit to reasonable number of years for visualization
    if len(years) > 10:
        print(f"Too many years ({len(years)}). Showing first 10.")
        years = years[:10]

    fig, ax = plt.subplots(figsize=(16, 8))

    # Plot each year
    colors = plt.cm.viridis(np.linspace(0, 1, len(years)))

    for year, color in zip(years, colors):
        year_data = births_df[births_df['year'] == year].copy()
        year_data['day_of_year'] = year_data['datetime'].dt.dayofyear

        ax.plot(year_data['day_of_year'], year_data['births'],
                linewidth=1.5, alpha=0.7, label=str(year), color=color)

    ax.set_xlabel('Day of Year', fontsize=13)
    ax.set_ylabel('Number of Births', fontsize=13)
    ax.set_title('Daily Births Comparison Across Years',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=10, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    return fig


def analyze_births_data(row):
    """
    Complete analysis pipeline for US births data

    Parameters:
    -----------
    row : pandas Series
        US births data row

    Returns:
    --------
    dict with expanded_df, monthly_stats, yearly_stats, and info
    """
    print("="*70)
    print("US BIRTHS DATA ANALYSIS")
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
    births_df = expand_births_timeseries(row)
    print(f"  Created DataFrame with {len(births_df):,} rows")

    # Step 3: Calculate statistics
    print("\nStep 3: Calculating statistics...")
    monthly_stats = aggregate_to_monthly(births_df)
    yearly_stats = aggregate_to_yearly(births_df)
    print(f"  Generated monthly statistics for {len(monthly_stats)} months")
    print(f"  Generated yearly statistics for {len(yearly_stats)} years")

    # Step 4: Overall statistics
    print("\nOverall Statistics:")
    print(f"  Mean births/day: {births_df['births'].mean():.2f}")
    print(f"  Median births/day: {births_df['births'].median():.2f}")
    print(f"  Std dev: {births_df['births'].std():.2f}")
    print(f"  Min births/day: {births_df['births'].min():.0f}")
    print(f"  Max births/day: {births_df['births'].max():.0f}")
    print(f"  Total births: {births_df['births'].sum():,.0f}")

    # Step 5: Day of week analysis
    print("\nDay of Week Analysis:")
    dow_stats = births_df.groupby('dayofweek_name')['births'].mean()
    dow_stats = dow_stats.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday',
                                   'Friday', 'Saturday', 'Sunday'])
    for day, avg in dow_stats.items():
        print(f"  {day}: {avg:.0f} avg births")

    # Step 6: Weekend vs Weekday
    weekday_avg = births_df[~births_df['is_weekend']]['births'].mean()
    weekend_avg = births_df[births_df['is_weekend']]['births'].mean()
    print(f"\nWeekday vs Weekend:")
    print(f"  Weekday average: {weekday_avg:.0f}")
    print(f"  Weekend average: {weekend_avg:.0f}")
    print(f"  Difference: {weekday_avg - weekend_avg:.0f} ({(weekday_avg-weekend_avg)/weekend_avg*100:.1f}%)")

    # Step 7: Monthly patterns
    print("\nMonthly Patterns:")
    month_stats = births_df.groupby('month_name')['births'].mean()
    month_order = ['January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']
    month_stats = month_stats.reindex(month_order)
    max_month = month_stats.idxmax()
    min_month = month_stats.idxmin()
    print(f"  Highest: {max_month} ({month_stats[max_month]:.0f} avg births)")
    print(f"  Lowest: {min_month} ({month_stats[min_month]:.0f} avg births)")

    print("="*70)

    return {
        'births_df': births_df,
        'monthly_stats': monthly_stats,
        'yearly_stats': yearly_stats,
        'interval_info': interval_info
    }


# Example usage
if __name__ == "__main__":
    print("US Births Analysis Module")
    print("\nFunctions available:")
    print("  - detect_time_interval(row)")
    print("  - expand_births_timeseries(row)")
    print("  - aggregate_to_monthly(births_df)")
    print("  - aggregate_to_yearly(births_df)")
    print("  - plot_full_timeseries(births_df)")
    print("  - plot_monthly_statistics(monthly_stats)")
    print("  - plot_yearly_trends(yearly_stats)")
    print("  - plot_day_of_week_patterns(births_df)")
    print("  - plot_monthly_seasonal_patterns(births_df)")
    print("  - plot_births_heatmap(births_df)")
    print("  - plot_weekday_weekend_comparison(births_df)")
    print("  - plot_multi_year_comparison(births_df)")
    print("  - analyze_births_data(row)  # Complete pipeline")
