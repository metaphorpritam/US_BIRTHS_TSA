import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tqdm import tqdm

sns.set_style("whitegrid")


def expand_customer_timeseries(smart_meter_df, freq='30min'):
    """
    Expand all customer time series into a long-format DataFrame

    Parameters:
    -----------
    smart_meter_df : pandas DataFrame
        DataFrame with columns: item_id, start, end, freq, target
    freq : str
        Frequency string (e.g., '30min', '1H')

    Returns:
    --------
    pandas DataFrame with columns: datetime, customer_id, energy_consumption
    """
    all_data = []

    print(f"Expanding time series for {len(smart_meter_df)} customers...")

    for idx, row in tqdm(smart_meter_df.iterrows(), total=len(smart_meter_df)):
        try:
            # Parse dates
            start = pd.to_datetime(row['start'])
            end = pd.to_datetime(row['end'])

            # Create datetime range
            date_range = pd.date_range(start=start, end=end, freq=freq)
            target_values = row['target']

            # Ensure lengths match
            min_len = min(len(date_range), len(target_values))

            # Create customer DataFrame
            customer_df = pd.DataFrame({
                'datetime': date_range[:min_len],
                'customer_id': row['item_id'],
                'energy_consumption': target_values[:min_len]
            })

            all_data.append(customer_df)
        except Exception as e:
            print(f"Error processing row {idx}: {e}")
            continue

    # Concatenate all customer data
    combined_df = pd.concat(all_data, ignore_index=True)
    print(f"Total data points: {len(combined_df):,}")

    return combined_df


def aggregate_to_daily(expanded_df):
    """
    Aggregate expanded time series to daily level

    Parameters:
    -----------
    expanded_df : pandas DataFrame
        Long-format DataFrame with datetime, customer_id, energy_consumption

    Returns:
    --------
    pandas DataFrame with daily aggregations per customer
    """
    print("Aggregating to daily level...")

    # Create date column
    expanded_df['date'] = expanded_df['datetime'].dt.date

    # Aggregate by customer and date
    daily_customer = expanded_df.groupby(['customer_id', 'date'])['energy_consumption'].agg([
        ('mean', 'mean'),
        ('sum', 'sum'),
        ('count', 'count'),
        ('min', 'min'),
        ('max', 'max')
    ]).reset_index()

    # Convert date back to datetime
    daily_customer['date'] = pd.to_datetime(daily_customer['date'])

    return daily_customer


def calculate_daily_statistics(daily_customer_df):
    """
    Calculate daily statistics across all customers

    Returns:
    --------
    pandas DataFrame with date, mean_consumption, median_consumption,
    customer_count, total_consumption
    """
    print("Calculating daily statistics across all customers...")

    daily_stats = daily_customer_df.groupby('date').agg({
        'mean': ['mean', 'median', 'std'],
        'sum': 'sum',
        'customer_id': 'count'
    }).reset_index()

    # Flatten column names
    daily_stats.columns = ['date', 'avg_mean_consumption', 'median_mean_consumption',
                           'std_mean_consumption', 'total_consumption', 'customer_count']

    return daily_stats


def plot_daily_average_consumption(daily_stats):
    """
    Plot daily average consumption across all customers
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    # Plot mean with confidence band (std)
    ax.plot(daily_stats['date'], daily_stats['avg_mean_consumption'],
            linewidth=2, color='steelblue', label='Mean')
    ax.plot(daily_stats['date'], daily_stats['median_mean_consumption'],
            linewidth=2, color='coral', label='Median', alpha=0.7)

    # Add standard deviation band
    ax.fill_between(daily_stats['date'],
                    daily_stats['avg_mean_consumption'] - daily_stats['std_mean_consumption'],
                    daily_stats['avg_mean_consumption'] + daily_stats['std_mean_consumption'],
                    alpha=0.3, color='steelblue', label='±1 Std Dev')

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Average Energy Consumption (kWh)', fontsize=12)
    ax.set_title('Daily Average Energy Consumption Across All Customers',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_customer_coverage(daily_stats):
    """
    Plot the number of customers with data for each day
    """
    fig, ax = plt.subplots(figsize=(16, 6))

    ax.bar(daily_stats['date'], daily_stats['customer_count'],
           width=0.8, alpha=0.7, color='forestgreen', edgecolor='black', linewidth=0.5)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Number of Customers', fontsize=12)
    ax.set_title('Customer Coverage Over Time', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add statistics
    max_customers = daily_stats['customer_count'].max()
    min_customers = daily_stats['customer_count'].min()
    avg_customers = daily_stats['customer_count'].mean()

    stats_text = f"Max: {max_customers:,}\nMin: {min_customers:,}\nAvg: {avg_customers:.0f}"
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    return fig


def plot_combined_view(daily_stats):
    """
    Plot both average consumption and customer count in subplots
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

    # Top plot: Average consumption
    ax1.plot(daily_stats['date'], daily_stats['avg_mean_consumption'],
             linewidth=2, color='steelblue', label='Mean')
    ax1.fill_between(daily_stats['date'],
                     daily_stats['avg_mean_consumption'] - daily_stats['std_mean_consumption'],
                     daily_stats['avg_mean_consumption'] + daily_stats['std_mean_consumption'],
                     alpha=0.3, color='steelblue', label='±1 Std Dev')
    ax1.set_ylabel('Avg Energy Consumption (kWh)', fontsize=12)
    ax1.set_title('Daily Average Energy Consumption Across All Customers',
                  fontsize=14, fontweight='bold')
    ax1.legend(loc='best')
    ax1.grid(True, alpha=0.3)

    # Bottom plot: Customer count
    ax2.bar(daily_stats['date'], daily_stats['customer_count'],
            width=0.8, alpha=0.7, color='forestgreen', edgecolor='black', linewidth=0.5)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Number of Customers', fontsize=12)
    ax2.set_title('Customer Coverage Over Time', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_heatmap_customers_by_date(daily_customer_df, sample_size=50):
    """
    Create a heatmap showing customer consumption patterns over time

    Parameters:
    -----------
    daily_customer_df : DataFrame with customer_id, date, mean columns
    sample_size : int, number of customers to sample for visualization
    """
    # Sample customers if there are too many
    unique_customers = daily_customer_df['customer_id'].unique()
    if len(unique_customers) > sample_size:
        sampled_customers = np.random.choice(unique_customers, sample_size, replace=False)
        plot_df = daily_customer_df[daily_customer_df['customer_id'].isin(sampled_customers)]
    else:
        plot_df = daily_customer_df

    # Pivot for heatmap
    pivot = plot_df.pivot_table(values='mean', index='customer_id', columns='date')

    fig, ax = plt.subplots(figsize=(18, 10))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Daily Mean Energy Consumption (kWh)'},
                xticklabels=True, yticklabels=False)

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Customer ID', fontsize=12)
    ax.set_title(f'Customer Energy Consumption Patterns Over Time (Sample of {len(pivot)} customers)',
                 fontsize=14, fontweight='bold')

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
    plt.tight_layout()

    return fig


def analyze_time_coverage(smart_meter_df):
    """
    Analyze the time coverage of the dataset
    """
    print("\n" + "="*60)
    print("TIME COVERAGE ANALYSIS")
    print("="*60)

    # Parse all dates
    starts = pd.to_datetime(smart_meter_df['start'])
    ends = pd.to_datetime(smart_meter_df['end'])

    # Overall date range
    overall_start = starts.min()
    overall_end = ends.max()

    print(f"\nOverall date range:")
    print(f"  Earliest start: {overall_start}")
    print(f"  Latest end: {overall_end}")
    print(f"  Total span: {(overall_end - overall_start).days} days")

    print(f"\nCustomer statistics:")
    print(f"  Total customers: {len(smart_meter_df)}")

    # Calculate duration for each customer
    durations = (ends - starts).dt.days
    print(f"  Mean duration: {durations.mean():.1f} days")
    print(f"  Median duration: {durations.median():.1f} days")
    print(f"  Min duration: {durations.min()} days")
    print(f"  Max duration: {durations.max()} days")

    # Distribution of start dates
    print(f"\nStart date distribution:")
    print(f"  Earliest: {starts.min()}")
    print(f"  Latest: {starts.max()}")

    print(f"\nEnd date distribution:")
    print(f"  Earliest: {ends.min()}")
    print(f"  Latest: {ends.max()}")

    print("="*60 + "\n")

    return {
        'overall_start': overall_start,
        'overall_end': overall_end,
        'customer_count': len(smart_meter_df),
        'mean_duration': durations.mean(),
        'median_duration': durations.median()
    }


# Complete pipeline function
def complete_aggregation_pipeline(smart_meter_df, freq='30min'):
    """
    Run the complete aggregation and visualization pipeline

    Parameters:
    -----------
    smart_meter_df : pandas DataFrame
        Your smart meter dataset
    freq : str
        Frequency of the time series data

    Returns:
    --------
    dict with expanded_df, daily_customer_df, and daily_stats
    """
    # Step 1: Analyze time coverage
    coverage_info = analyze_time_coverage(smart_meter_df)

    # Step 2: Expand all time series
    expanded_df = expand_customer_timeseries(smart_meter_df, freq=freq)

    # Step 3: Aggregate to daily per customer
    daily_customer_df = aggregate_to_daily(expanded_df)

    # Step 4: Calculate daily statistics across customers
    daily_stats = calculate_daily_statistics(daily_customer_df)

    # Print summary
    print("\n" + "="*60)
    print("DAILY AGGREGATION SUMMARY")
    print("="*60)
    print(f"Date range: {daily_stats['date'].min()} to {daily_stats['date'].max()}")
    print(f"Total days: {len(daily_stats)}")
    print(f"Average customers per day: {daily_stats['customer_count'].mean():.0f}")
    print(f"Max customers on any day: {daily_stats['customer_count'].max()}")
    print(f"Min customers on any day: {daily_stats['customer_count'].min()}")
    print(f"\nAverage daily consumption:")
    print(f"  Mean: {daily_stats['avg_mean_consumption'].mean():.3f} kWh")
    print(f"  Median: {daily_stats['median_mean_consumption'].mean():.3f} kWh")
    print("="*60 + "\n")

    return {
        'expanded_df': expanded_df,
        'daily_customer_df': daily_customer_df,
        'daily_stats': daily_stats,
        'coverage_info': coverage_info
    }


if __name__ == "__main__":
    print("To use this module:")
    print("\n1. Run complete pipeline:")
    print("   results = complete_aggregation_pipeline(smart_meter_df)")
    print("\n2. Visualize:")
    print("   plot_daily_average_consumption(results['daily_stats'])")
    print("   plot_customer_coverage(results['daily_stats'])")
    print("   plot_combined_view(results['daily_stats'])")
