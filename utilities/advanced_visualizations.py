import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Import your aggregate functions
import utilities.aggregate_customers as agg

sns.set_style("whitegrid")


def plot_total_consumption_with_customer_count(daily_stats):
    """
    Plot total energy consumption with customer count on secondary y-axis

    Parameters:
    -----------
    daily_stats : DataFrame
        Output from calculate_daily_statistics()
    """
    fig, ax1 = plt.subplots(figsize=(16, 7))

    # Primary axis: Total consumption
    color_consumption = 'steelblue'
    ax1.set_xlabel('Date', fontsize=13)
    ax1.set_ylabel('Total Energy Consumption (kWh)', fontsize=13, color=color_consumption)

    line1 = ax1.plot(daily_stats['date'], daily_stats['total_consumption'],
                     color=color_consumption, linewidth=2, label='Total Consumption')
    ax1.fill_between(daily_stats['date'], daily_stats['total_consumption'],
                     alpha=0.3, color=color_consumption)
    ax1.tick_params(axis='y', labelcolor=color_consumption)
    ax1.grid(True, alpha=0.3)

    # Secondary axis: Customer count
    ax2 = ax1.twinx()
    color_customers = 'darkgreen'
    ax2.set_ylabel('Number of Customers', fontsize=13, color=color_customers)

    line2 = ax2.plot(daily_stats['date'], daily_stats['customer_count'],
                     color=color_customers, linewidth=2, linestyle='--',
                     label='Customer Count', alpha=0.8)
    ax2.tick_params(axis='y', labelcolor=color_customers)

    # Title
    plt.title('Total Energy Consumption vs Customer Coverage\n(Dual Axis)',
              fontsize=15, fontweight='bold', pad=20)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper left', fontsize=11)

    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def plot_average_consumption_over_time_range(daily_stats):
    """
    Plot average energy consumption (mean across all customers) over entire time range
    with confidence intervals

    Parameters:
    -----------
    daily_stats : DataFrame
        Output from calculate_daily_statistics()
    """
    fig, ax = plt.subplots(figsize=(16, 7))

    # Plot mean consumption
    ax.plot(daily_stats['date'], daily_stats['avg_mean_consumption'],
            linewidth=2.5, color='darkblue', label='Mean', zorder=3)

    # Plot median consumption
    ax.plot(daily_stats['date'], daily_stats['median_mean_consumption'],
            linewidth=2, color='coral', label='Median', alpha=0.8,
            linestyle='--', zorder=2)

    # Add confidence band (±1 std)
    ax.fill_between(daily_stats['date'],
                    daily_stats['avg_mean_consumption'] - daily_stats['std_mean_consumption'],
                    daily_stats['avg_mean_consumption'] + daily_stats['std_mean_consumption'],
                    alpha=0.3, color='lightblue', label='±1 Std Dev', zorder=1)

    # Add 2 std dev band (95% confidence)
    ax.fill_between(daily_stats['date'],
                    daily_stats['avg_mean_consumption'] - 2*daily_stats['std_mean_consumption'],
                    daily_stats['avg_mean_consumption'] + 2*daily_stats['std_mean_consumption'],
                    alpha=0.15, color='lightblue', label='±2 Std Dev (95% CI)', zorder=0)

    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel('Average Energy Consumption per Customer (kWh)', fontsize=13)
    ax.set_title('Average Energy Consumption Over Time\n(Aggregated Across All Customers)',
                 fontsize=15, fontweight='bold', pad=20)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)

    plt.xticks(rotation=45)
    plt.tight_layout()

    # Add statistics box
    mean_avg = daily_stats['avg_mean_consumption'].mean()
    median_avg = daily_stats['median_mean_consumption'].mean()
    std_avg = daily_stats['std_mean_consumption'].mean()

    stats_text = f"Overall Statistics:\n"
    stats_text += f"Mean: {mean_avg:.3f} kWh\n"
    stats_text += f"Median: {median_avg:.3f} kWh\n"
    stats_text += f"Avg Std Dev: {std_avg:.3f} kWh"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            fontsize=10, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    return fig


def plot_daily_consumption_with_spread(daily_customer_df):
    """
    Plot daily average consumption showing individual customer variability
    Uses box plots or violin plots to show distribution

    Parameters:
    -----------
    daily_customer_df : DataFrame
        Output from aggregate_to_daily() - has customer_id, date, mean columns
    """
    # Sample dates for visualization (too many dates makes it cluttered)
    dates = sorted(daily_customer_df['date'].unique())

    # Option 1: If too many dates, sample evenly
    if len(dates) > 50:
        sample_indices = np.linspace(0, len(dates)-1, 50, dtype=int)
        sampled_dates = [dates[i] for i in sample_indices]
    else:
        sampled_dates = dates

    # Filter data
    plot_data = daily_customer_df[daily_customer_df['date'].isin(sampled_dates)].copy()
    plot_data['date_str'] = plot_data['date'].dt.strftime('%Y-%m-%d')

    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(18, 12))

    # ===== SUBPLOT 1: Box Plot =====
    # Prepare data for box plot
    box_data = [plot_data[plot_data['date'] == d]['mean'].values
                for d in sampled_dates]

    positions = range(len(sampled_dates))
    bp = ax1.boxplot(box_data, positions=positions,
                     widths=0.6, patch_artist=True,
                     boxprops=dict(facecolor='lightblue', alpha=0.7),
                     medianprops=dict(color='red', linewidth=2),
                     whiskerprops=dict(linewidth=1.5),
                     capprops=dict(linewidth=1.5))

    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Daily Mean Energy Consumption (kWh)', fontsize=12)
    ax1.set_title('Daily Energy Consumption Distribution Across Customers\n(Box Plot - Shows Quartiles and Outliers)',
                  fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')

    # Set x-axis labels (sample every nth label to avoid crowding)
    label_step = max(1, len(sampled_dates) // 20)
    ax1.set_xticks(positions[::label_step])
    ax1.set_xticklabels([sampled_dates[i].strftime('%Y-%m-%d')
                         for i in range(0, len(sampled_dates), label_step)],
                        rotation=45, ha='right')

    # ===== SUBPLOT 2: Percentile Bands =====
    # Calculate percentiles for each date
    percentile_data = plot_data.groupby('date')['mean'].agg([
        ('p10', lambda x: np.percentile(x, 10)),
        ('p25', lambda x: np.percentile(x, 25)),
        ('p50', lambda x: np.percentile(x, 50)),
        ('p75', lambda x: np.percentile(x, 75)),
        ('p90', lambda x: np.percentile(x, 90)),
        ('mean', 'mean')
    ]).reset_index()

    # Plot median and mean
    ax2.plot(percentile_data['date'], percentile_data['p50'],
             linewidth=2.5, color='darkblue', label='Median (P50)', zorder=4)
    ax2.plot(percentile_data['date'], percentile_data['mean'],
             linewidth=2, color='red', label='Mean', linestyle='--',
             alpha=0.8, zorder=3)

    # Fill between percentiles
    ax2.fill_between(percentile_data['date'],
                     percentile_data['p25'], percentile_data['p75'],
                     alpha=0.4, color='steelblue', label='P25-P75 (IQR)', zorder=2)
    ax2.fill_between(percentile_data['date'],
                     percentile_data['p10'], percentile_data['p90'],
                     alpha=0.2, color='lightblue', label='P10-P90', zorder=1)

    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Daily Mean Energy Consumption (kWh)', fontsize=12)
    ax2.set_title('Daily Energy Consumption Distribution Across Customers\n(Percentile Bands)',
                  fontsize=14, fontweight='bold')
    ax2.legend(loc='best', fontsize=11)
    ax2.grid(True, alpha=0.3)

    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')

    plt.tight_layout()

    return fig


def plot_customer_variability_heatmap(daily_customer_df, n_customers=100):
    """
    Create a heatmap showing individual customer consumption patterns

    Parameters:
    -----------
    daily_customer_df : DataFrame
        Output from aggregate_to_daily()
    n_customers : int
        Number of customers to show (randomly sampled if more exist)
    """
    # Sample customers
    unique_customers = daily_customer_df['customer_id'].unique()
    if len(unique_customers) > n_customers:
        sampled_customers = np.random.choice(unique_customers, n_customers, replace=False)
    else:
        sampled_customers = unique_customers

    # Filter and pivot
    plot_data = daily_customer_df[daily_customer_df['customer_id'].isin(sampled_customers)]
    pivot_data = plot_data.pivot_table(values='mean',
                                       index='customer_id',
                                       columns='date')

    # Create heatmap
    fig, ax = plt.subplots(figsize=(18, 12))

    sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Daily Mean Energy Consumption (kWh)'},
                xticklabels=False,  # Too many dates to show all
                yticklabels=False)  # Too many customers to show all

    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel(f'Customer ID (n={len(pivot_data)})', fontsize=12)
    ax.set_title(f'Individual Customer Energy Consumption Patterns Over Time\n(Sample of {len(sampled_customers)} customers)',
                 fontsize=14, fontweight='bold')

    # Add date range on x-axis
    dates = sorted(plot_data['date'].unique())
    n_ticks = 10
    tick_positions = np.linspace(0, len(dates)-1, n_ticks, dtype=int)
    ax.set_xticks(tick_positions)
    ax.set_xticklabels([dates[i].strftime('%Y-%m-%d') for i in tick_positions],
                       rotation=45, ha='right')

    plt.tight_layout()

    return fig


def create_all_visualizations(smart_meter_df):
    """
    Complete pipeline to create all visualizations

    Parameters:
    -----------
    smart_meter_df : DataFrame
        Your smart meter dataset

    Returns:
    --------
    dict with all results and figures
    """
    print("Running aggregation pipeline...")
    results = agg.complete_aggregation_pipeline(smart_meter_df)

    daily_stats = results['daily_stats']
    daily_customer_df = results['daily_customer_df']

    print("\nCreating visualizations...")

    # Plot 1: Dual axis - Total consumption + customer count
    print("  1. Total consumption with customer count (dual axis)...")
    fig1 = plot_total_consumption_with_customer_count(daily_stats)
    plt.show()

    # Plot 2: Average consumption over time
    print("  2. Average consumption over time range...")
    fig2 = plot_average_consumption_over_time_range(daily_stats)
    plt.show()

    # Plot 3: Daily consumption with spread
    print("  3. Daily consumption with customer variability...")
    fig3 = plot_daily_consumption_with_spread(daily_customer_df)
    plt.show()

    # Plot 4: Customer heatmap (optional)
    print("  4. Individual customer heatmap...")
    fig4 = plot_customer_variability_heatmap(daily_customer_df, n_customers=100)
    plt.show()

    print("\n✅ All visualizations created!")

    return {
        'results': results,
        'figures': {
            'dual_axis': fig1,
            'avg_consumption': fig2,
            'spread': fig3,
            'heatmap': fig4
        }
    }


# Example usage
if __name__ == "__main__":
    print("To use these functions:")
    print("\n# Option 1: Create all visualizations at once")
    print("viz_results = create_all_visualizations(smart_meter_df)")
    print("\n# Option 2: Create individual plots")
    print("results = agg.complete_aggregation_pipeline(smart_meter_df)")
    print("fig1 = plot_total_consumption_with_customer_count(results['daily_stats'])")
    print("fig2 = plot_average_consumption_over_time_range(results['daily_stats'])")
    print("fig3 = plot_daily_consumption_with_spread(results['daily_customer_df'])")
