"""
Time Series Subsetting Utilities

This module provides tools for extracting and working with subsets of time series data:
- Date range extraction
- Validation functions
- Splitting utilities
- Subset statistics
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Union
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


def extract_subset(data: pd.DataFrame,
                   datetime_col: str,
                   start_date: Optional[Union[str, datetime]] = None,
                   end_date: Optional[Union[str, datetime]] = None,
                   validate: bool = True) -> pd.DataFrame:
    """
    Extract a subset of time series data based on date range

    Parameters:
    -----------
    data : DataFrame
        The full time series dataframe
    datetime_col : str
        Name of the datetime column
    start_date : str or datetime, optional
        Start date (inclusive). If None, starts from beginning
    end_date : str or datetime, optional
        End date (inclusive). If None, goes to end
    validate : bool
        Whether to validate the subset

    Returns:
    --------
    DataFrame with subset of data
    """
    df = data.copy()

    # Ensure datetime column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(df[datetime_col]):
        df[datetime_col] = pd.to_datetime(df[datetime_col])

    # Sort by datetime
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # Apply filters
    if start_date is not None:
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        df = df[df[datetime_col] >= start_date]

    if end_date is not None:
        if isinstance(end_date, str):
            end_date = pd.to_datetime(end_date)
        df = df[df[datetime_col] <= end_date]

    # Reset index
    df = df.reset_index(drop=True)

    # Validate if requested
    if validate:
        _validate_subset(df, datetime_col, start_date, end_date)

    return df


def _validate_subset(df: pd.DataFrame,
                     datetime_col: str,
                     start_date: Optional[datetime],
                     end_date: Optional[datetime]) -> None:
    """
    Validate a time series subset

    Parameters:
    -----------
    df : DataFrame
        The subset dataframe
    datetime_col : str
        Name of datetime column
    start_date : datetime or None
        Expected start date
    end_date : datetime or None
        Expected end date
    """
    if len(df) == 0:
        raise ValueError("Subset is empty! Check your date range.")

    actual_start = df[datetime_col].min()
    actual_end = df[datetime_col].max()

    print(f"\n{'='*60}")
    print("TIME SUBSET VALIDATION")
    print(f"{'='*60}")
    print(f"Requested start: {start_date if start_date else 'Beginning'}")
    print(f"Actual start:    {actual_start}")
    print(f"Requested end:   {end_date if end_date else 'End'}")
    print(f"Actual end:      {actual_end}")
    print(f"Total records:   {len(df)}")
    print(f"Date range:      {(actual_end - actual_start).days} days")

    # Check for gaps
    gaps = check_time_gaps(df, datetime_col)
    if gaps is not None and len(gaps) > 0:
        print(f"⚠ Warning: Found {len(gaps)} gaps in the time series")
    else:
        print("✓ No gaps detected in time series")

    print(f"{'='*60}\n")


def check_time_gaps(df: pd.DataFrame,
                    datetime_col: str,
                    expected_freq: str = 'D') -> Optional[pd.DataFrame]:
    """
    Check for gaps in time series data

    Parameters:
    -----------
    df : DataFrame
        Time series dataframe
    datetime_col : str
        Name of datetime column
    expected_freq : str
        Expected frequency ('D' for daily, 'H' for hourly, etc.)

    Returns:
    --------
    DataFrame with gap information or None if no gaps
    """
    df = df.sort_values(datetime_col).reset_index(drop=True)
    dates = df[datetime_col]

    # Create expected date range
    expected_range = pd.date_range(start=dates.min(), end=dates.max(), freq=expected_freq)

    # Find missing dates
    missing_dates = expected_range.difference(dates) # type: ignore

    if len(missing_dates) == 0:
        return None

    # Create gap information
    gaps = []
    for missing_date in missing_dates:
        # Find surrounding dates
        before = dates[dates < missing_date].max() if any(dates < missing_date) else None
        after = dates[dates > missing_date].min() if any(dates > missing_date) else None

        gaps.append({
            'missing_date': missing_date,
            'date_before': before,
            'date_after': after
        })

    return pd.DataFrame(gaps)


def split_train_test(data: pd.DataFrame,
                     datetime_col: str,
                     test_size: Union[int, float] = 0.2,
                     by_date: bool = False,
                     split_date: Optional[Union[str, datetime]] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split time series data into train and test sets

    Parameters:
    -----------
    data : DataFrame
        Full time series dataframe
    datetime_col : str
        Name of datetime column
    test_size : int or float
        If float (0-1): proportion of data for testing
        If int: number of observations for testing
    by_date : bool
        If True, use split_date to split
    split_date : str or datetime
        Date to split on (only used if by_date=True)

    Returns:
    --------
    tuple: (train_df, test_df)
    """
    df = data.sort_values(datetime_col).reset_index(drop=True)

    if by_date:
        if split_date is None:
            raise ValueError("split_date must be provided when by_date=True")

        if isinstance(split_date, str):
            split_date = pd.to_datetime(split_date)

        train = df[df[datetime_col] < split_date].reset_index(drop=True)
        test = df[df[datetime_col] >= split_date].reset_index(drop=True)
    else:
        if isinstance(test_size, float):
            test_size = int(len(df) * test_size)

        split_idx = len(df) - test_size
        train = df.iloc[:split_idx].reset_index(drop=True)
        test = df.iloc[split_idx:].reset_index(drop=True)

    print(f"\n{'='*60}")
    print("TRAIN/TEST SPLIT")
    print(f"{'='*60}")
    print(f"Train set: {len(train)} observations")
    print(f"  Start: {train[datetime_col].min()}")
    print(f"  End:   {train[datetime_col].max()}")
    print(f"\nTest set: {len(test)} observations")
    print(f"  Start: {test[datetime_col].min()}")
    print(f"  End:   {test[datetime_col].max()}")
    print(f"{'='*60}\n")

    return train, test


def get_subset_statistics(data: pd.DataFrame,
                          value_col: str,
                          datetime_col: str) -> dict:
    """
    Calculate statistics for a time series subset

    Parameters:
    -----------
    data : DataFrame
        Time series dataframe
    value_col : str
        Name of value column
    datetime_col : str
        Name of datetime column

    Returns:
    --------
    dict with statistics
    """
    values = data[value_col].values
    dates = data[datetime_col]

    stats = {
        'count': len(values),
        'mean': np.mean(values), # type: ignore
        'std': np.std(values), # type: ignore
        'min': np.min(values), # type: ignore
        'max': np.max(values), # type: ignore
        'median': np.median(values), # type: ignore
        'q25': np.percentile(values, 25), # type: ignore
        'q75': np.percentile(values, 75), # type: ignore
        'start_date': dates.min(),
        'end_date': dates.max(),
        'days': (dates.max() - dates.min()).days,
        'missing_values': data[value_col].isna().sum()
    }

    return stats


def print_subset_statistics(stats: dict, title: str = "Time Series Statistics") -> None:
    """
    Print formatted statistics

    Parameters:
    -----------
    stats : dict
        Statistics dictionary from get_subset_statistics
    title : str
        Title for the output
    """
    print(f"\n{'='*60}")
    print(f"{title}")
    print(f"{'='*60}")
    print(f"Observations:    {stats['count']}")
    print(f"Date range:      {stats['start_date'].date()} to {stats['end_date'].date()}")
    print(f"Duration:        {stats['days']} days")
    print(f"\nValue Statistics:")
    print(f"  Mean:          {stats['mean']:.2f}")
    print(f"  Std Dev:       {stats['std']:.2f}")
    print(f"  Min:           {stats['min']:.2f}")
    print(f"  Q25:           {stats['q25']:.2f}")
    print(f"  Median:        {stats['median']:.2f}")
    print(f"  Q75:           {stats['q75']:.2f}")
    print(f"  Max:           {stats['max']:.2f}")
    print(f"  Missing:       {stats['missing_values']}")
    print(f"{'='*60}\n")


def plot_subset_comparison(full_data: pd.DataFrame,
                           subset_data: pd.DataFrame,
                           datetime_col: str,
                           value_col: str,
                           title: str = "Full Data vs Subset") -> plt.Figure:
    """
    Plot full data with subset highlighted

    Parameters:
    -----------
    full_data : DataFrame
        Full time series
    subset_data : DataFrame
        Subset to highlight
    datetime_col : str
        Name of datetime column
    value_col : str
        Name of value column
    title : str
        Plot title

    Returns:
    --------
    matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=(18, 7))

    # Plot full data
    ax.plot(full_data[datetime_col], full_data[value_col],
            linewidth=0.8, alpha=0.4, color='gray', label='Full Data')

    # Highlight subset
    ax.plot(subset_data[datetime_col], subset_data[value_col],
            linewidth=1.5, alpha=0.9, color='darkblue', label='Subset')

    # Add shaded region
    subset_start = subset_data[datetime_col].min()
    subset_end = subset_data[datetime_col].max()
    ax.axvspan(subset_start, subset_end, alpha=0.1, color='blue')

    ax.set_xlabel('Date', fontsize=13)
    ax.set_ylabel(value_col.capitalize(), fontsize=13)
    ax.set_title(title, fontsize=15, fontweight='bold', pad=20)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=45)
    plt.tight_layout()

    return fig


def create_rolling_subsets(data: pd.DataFrame,
                           datetime_col: str,
                           window_size: int,
                           step_size: int = 1) -> list:
    """
    Create rolling window subsets of time series

    Parameters:
    -----------
    data : DataFrame
        Full time series
    datetime_col : str
        Name of datetime column
    window_size : int
        Size of each window (in observations)
    step_size : int
        Step size between windows

    Returns:
    --------
    list of DataFrames
    """
    df = data.sort_values(datetime_col).reset_index(drop=True)
    subsets = []

    for i in range(0, len(df) - window_size + 1, step_size):
        subset = df.iloc[i:i+window_size].reset_index(drop=True)
        subsets.append(subset)

    print(f"Created {len(subsets)} rolling subsets")
    print(f"  Window size: {window_size} observations")
    print(f"  Step size: {step_size} observations")

    return subsets


if __name__ == "__main__":
    print("Time Series Subsetting Utilities Module")
    print("\nFunctions available:")
    print("  - extract_subset(data, datetime_col, start_date, end_date)")
    print("  - check_time_gaps(df, datetime_col, expected_freq)")
    print("  - split_train_test(data, datetime_col, test_size)")
    print("  - get_subset_statistics(data, value_col, datetime_col)")
    print("  - plot_subset_comparison(full_data, subset_data, datetime_col, value_col)")
    print("  - create_rolling_subsets(data, datetime_col, window_size, step_size)")
