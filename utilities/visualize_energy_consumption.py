import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set style for better-looking plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (15, 6)

def visualize_single_series(row, title=None):
    """
    Visualize a single time series from the dataset

    Parameters:
    -----------
    row : pandas Series or dict
        A row from the smart_meter_df with 'start', 'end', 'target', and 'freq'
    title : str, optional
        Custom title for the plot
    """
    # Parse start and end dates
    start = pd.to_datetime(row['start'])
    end = pd.to_datetime(row['end'])

    # Create datetime index based on frequency
    # 'M' typically means minutes, adjust based on actual data
    freq_map = {
        'M': '30min',  # Based on the timestamps shown (30-minute intervals)
        'H': '1H',
        'D': '1D'
    }
    freq = freq_map.get(row['freq'], '30min')

    # Generate datetime range
    date_range = pd.date_range(start=start, end=end, freq=freq)

    # Get target values
    target_values = row['target']

    # Ensure lengths match (sometimes there might be slight mismatches)
    min_len = min(len(date_range), len(target_values))
    date_range = date_range[:min_len]
    target_values = target_values[:min_len]

    # Create DataFrame for easier manipulation
    df = pd.DataFrame({
        'datetime': date_range,
        'energy_consumption': target_values
    })

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df['datetime'], df['energy_consumption'], linewidth=0.8, alpha=0.8)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)

    if title:
        ax.set_title(title, fontsize=14, fontweight='bold')
    else:
        ax.set_title(f'Energy Consumption Time Series\n{start.date()} to {end.date()}',
                     fontsize=14, fontweight='bold')

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, df


def visualize_daily_aggregated(row, agg_func='mean'):
    """
    Visualize daily aggregated energy consumption

    Parameters:
    -----------
    row : pandas Series or dict
        A row from the smart_meter_df
    agg_func : str
        Aggregation function: 'mean', 'sum', 'max', 'min'
    """
    # Parse dates and create time series
    start = pd.to_datetime(row['start'])
    end = pd.to_datetime(row['end'])
    date_range = pd.date_range(start=start, end=end, freq='30min')
    target_values = row['target']

    min_len = min(len(date_range), len(target_values))

    df = pd.DataFrame({
        'datetime': date_range[:min_len],
        'energy_consumption': target_values[:min_len]
    })

    # Aggregate to daily level
    df['date'] = df['datetime'].dt.date
    daily_df = df.groupby('date')['energy_consumption'].agg(agg_func).reset_index()
    daily_df['date'] = pd.to_datetime(daily_df['date'])

    # Plot
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.bar(daily_df['date'], daily_df['energy_consumption'],
           width=0.8, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'Daily Energy Consumption ({agg_func.capitalize()}, kWh)', fontsize=12)
    ax.set_title(f'Daily {agg_func.capitalize()} Energy Consumption\n{start.date()} to {end.date()}',
                 fontsize=14, fontweight='bold')

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig, daily_df


def visualize_multiple_series(df, n_series=5, sample='first'):
    """
    Visualize multiple time series on the same plot

    Parameters:
    -----------
    df : pandas DataFrame
        The smart_meter_df dataframe
    n_series : int
        Number of series to plot
    sample : str
        'first', 'last', or 'random'
    """
    if sample == 'first':
        subset = df.head(n_series)
    elif sample == 'last':
        subset = df.tail(n_series)
    else:  # random
        subset = df.sample(n_series)

    fig, ax = plt.subplots(figsize=(15, 8))

    for idx, row in subset.iterrows():
        start = pd.to_datetime(row['start'])
        end = pd.to_datetime(row['end'])
        date_range = pd.date_range(start=start, end=end, freq='30min')
        target_values = row['target']

        min_len = min(len(date_range), len(target_values))

        ax.plot(date_range[:min_len], target_values[:min_len],
                linewidth=0.7, alpha=0.6, label=f"Series {idx}")

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Energy Consumption (kWh)', fontsize=12)
    ax.set_title(f'Multiple Energy Consumption Time Series (n={n_series})',
                 fontsize=14, fontweight='bold')
    ax.legend(loc='best', fontsize=10)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def create_heatmap_daily_patterns(row):
    """
    Create a heatmap showing daily patterns (hour of day vs day)
    """
    start = pd.to_datetime(row['start'])
    end = pd.to_datetime(row['end'])
    date_range = pd.date_range(start=start, end=end, freq='30min')
    target_values = row['target']

    min_len = min(len(date_range), len(target_values))

    df = pd.DataFrame({
        'datetime': date_range[:min_len],
        'energy_consumption': target_values[:min_len]
    })

    df['date'] = df['datetime'].dt.date
    df['hour'] = df['datetime'].dt.hour

    # Create pivot table
    pivot = df.pivot_table(values='energy_consumption',
                           index='hour',
                           columns='date',
                           aggfunc='mean')

    # Plot heatmap
    fig, ax = plt.subplots(figsize=(16, 8))
    sns.heatmap(pivot, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Energy Consumption (kWh)'})
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Hour of Day', fontsize=12)
    ax.set_title('Energy Consumption Heatmap: Daily Patterns', fontsize=14, fontweight='bold')
    plt.tight_layout()

    return fig


# Example usage:
if __name__ == "__main__":
    # Assuming you have smart_meter_df loaded
    print("To use these functions with your data:")
    print("\n1. Visualize a single series:")
    print("   fig, df = visualize_single_series(smart_meter_df.iloc[0])")
    print("   plt.show()")
    print("\n2. Visualize daily aggregated data:")
    print("   fig, daily_df = visualize_daily_aggregated(smart_meter_df.iloc[0], agg_func='mean')")
    print("   plt.show()")
    print("\n3. Visualize multiple series:")
    print("   fig = visualize_multiple_series(smart_meter_df, n_series=3)")
    print("   plt.show()")
    print("\n4. Create a heatmap:")
    print("   fig = create_heatmap_daily_patterns(smart_meter_df.iloc[0])")
    print("   plt.show()")
