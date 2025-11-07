"""
Energy Consumption Analysis Utilities

This package provides tools for:
- Aggregating multiple customer time series with non-uniform ranges
- Visualizing energy consumption patterns
- Analyzing daily consumption statistics
- Wind farm power production analysis
- US births data analysis
"""

# Import from aggregate_customers
from .aggregate_customers import (
    expand_customer_timeseries,
    aggregate_to_daily,
    calculate_daily_statistics,
    plot_daily_average_consumption,
    plot_customer_coverage,
    plot_combined_view,
    analyze_time_coverage,
    complete_aggregation_pipeline
)

# Import from visualize_energy_consumption
from .visualize_energy_consumption import (
    visualize_single_series,
    visualize_daily_aggregated,
    visualize_multiple_series,
    create_heatmap_daily_patterns
)

# Import from advanced_visualizations
from .advanced_visualizations import (
    plot_total_consumption_with_customer_count,
    plot_average_consumption_over_time_range,
    plot_daily_consumption_with_spread,
    plot_customer_variability_heatmap,
    create_all_visualizations
)

# Import from wind_farm_analysis
from .wind_farm_analysis import (
    detect_time_interval,
    expand_wind_timeseries,
    plot_full_timeseries,
    plot_daily_statistics,
    plot_multi_resolution_view,
    plot_hourly_patterns,
    plot_daily_production_heatmap,
    analyze_wind_data
)

# Import from us_births_analysis
from .us_births_analysis import (
    detect_time_interval as detect_births_time_interval,
    expand_births_timeseries,
    aggregate_to_monthly,
    aggregate_to_yearly,
    plot_full_timeseries as plot_births_timeseries,
    plot_monthly_statistics,
    plot_yearly_trends,
    plot_day_of_week_patterns,
    plot_monthly_seasonal_patterns,
    plot_births_heatmap,
    plot_weekday_weekend_comparison,
    plot_multi_year_comparison,
    analyze_births_data
)

# Import from arima_analysis
from .arima_analysis import (
    plot_acf_pacf,
    check_stationarity,
    arima_grid_search,
    plot_residual_diagnostics,
    print_residual_diagnostics,
    plot_forecast,
    compare_top_models,
    plot_differenced_series
)

__version__ = '0.1.0'

__all__ = [
    # Aggregation functions
    'expand_customer_timeseries',
    'aggregate_to_daily',
    'calculate_daily_statistics',
    'plot_daily_average_consumption',
    'plot_customer_coverage',
    'plot_combined_view',
    'analyze_time_coverage',
    'complete_aggregation_pipeline',

    # Basic visualization functions
    'visualize_single_series',
    'visualize_daily_aggregated',
    'visualize_multiple_series',
    'create_heatmap_daily_patterns',

    # Advanced visualization functions
    'plot_total_consumption_with_customer_count',
    'plot_average_consumption_over_time_range',
    'plot_daily_consumption_with_spread',
    'plot_customer_variability_heatmap',
    'create_all_visualizations',

    # Wind farm analysis functions
    'detect_time_interval',
    'expand_wind_timeseries',
    'plot_full_timeseries',
    'plot_daily_statistics',
    'plot_hourly_patterns',
    'plot_daily_production_heatmap',
    'plot_multi_resolution_view',
    'analyze_wind_data',

    # US births analysis functions
    'detect_births_time_interval',
    'expand_births_timeseries',
    'aggregate_to_monthly',
    'aggregate_to_yearly',
    'plot_births_timeseries',
    'plot_monthly_statistics',
    'plot_yearly_trends',
    'plot_day_of_week_patterns',
    'plot_monthly_seasonal_patterns',
    'plot_births_heatmap',
    'plot_weekday_weekend_comparison',
    'plot_multi_year_comparison',
    'analyze_births_data',

    # ARIMA analysis functions
    'plot_acf_pacf',
    'check_stationarity',
    'arima_grid_search',
    'plot_residual_diagnostics',
    'print_residual_diagnostics',
    'plot_forecast',
    'compare_top_models',
    'plot_differenced_series',
]
