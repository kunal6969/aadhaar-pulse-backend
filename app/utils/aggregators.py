"""
Data aggregation utility functions.
"""
import pandas as pd
from typing import Literal, List, Optional
from datetime import date


def aggregate_by_view_mode(
    df: pd.DataFrame,
    view_mode: Literal["daily", "monthly", "quarterly"],
    date_column: str = 'date',
    value_columns: List[str] = None,
    group_columns: List[str] = None
) -> pd.DataFrame:
    """
    Aggregate DataFrame by view mode (daily/monthly/quarterly).
    
    Args:
        df: Input DataFrame with date column
        view_mode: Aggregation level
        date_column: Name of date column
        value_columns: Columns to aggregate (sum). If None, aggregates all numeric columns.
        group_columns: Additional columns to group by (e.g., state, district)
        
    Returns:
        Aggregated DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    df[date_column] = pd.to_datetime(df[date_column])
    
    # Determine value columns if not specified
    if value_columns is None:
        value_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        if date_column in value_columns:
            value_columns.remove(date_column)
    
    # Build groupby columns
    if group_columns is None:
        group_columns = []
    
    if view_mode == "monthly":
        df['period'] = df[date_column].dt.to_period('M')
        groupby_cols = ['period'] + group_columns
        agg_df = df.groupby(groupby_cols)[value_columns].sum().reset_index()
        agg_df[date_column] = agg_df['period'].dt.to_timestamp()
        agg_df = agg_df.drop('period', axis=1)
    
    elif view_mode == "quarterly":
        df['period'] = df[date_column].dt.to_period('Q')
        groupby_cols = ['period'] + group_columns
        agg_df = df.groupby(groupby_cols)[value_columns].sum().reset_index()
        agg_df[date_column] = agg_df['period'].dt.to_timestamp()
        agg_df = agg_df.drop('period', axis=1)
    
    else:  # daily
        groupby_cols = [date_column] + group_columns
        agg_df = df.groupby(groupby_cols)[value_columns].sum().reset_index()
    
    return agg_df


def aggregate_by_geography(
    df: pd.DataFrame,
    level: Literal["state", "district", "pincode"],
    value_columns: List[str] = None
) -> pd.DataFrame:
    """
    Aggregate DataFrame by geographic level.
    
    Args:
        df: Input DataFrame with state/district/pincode columns
        level: Geographic aggregation level
        value_columns: Columns to aggregate (sum)
        
    Returns:
        Aggregated DataFrame
    """
    if df.empty:
        return df
    
    df = df.copy()
    
    # Determine value columns if not specified
    if value_columns is None:
        value_columns = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    if level == "state":
        group_cols = ['state']
    elif level == "district":
        group_cols = ['state', 'district']
    else:  # pincode
        group_cols = ['state', 'district', 'pincode']
    
    # Filter to only existing columns
    group_cols = [c for c in group_cols if c in df.columns]
    value_columns = [c for c in value_columns if c in df.columns]
    
    agg_df = df.groupby(group_cols)[value_columns].sum().reset_index()
    
    return agg_df


def calculate_rolling_stats(
    df: pd.DataFrame,
    value_column: str,
    window: int = 7,
    date_column: str = 'date'
) -> pd.DataFrame:
    """
    Calculate rolling statistics (mean, std, min, max).
    
    Args:
        df: Input DataFrame sorted by date
        value_column: Column to calculate stats for
        window: Rolling window size
        date_column: Date column for sorting
        
    Returns:
        DataFrame with additional rolling stat columns
    """
    df = df.copy()
    df = df.sort_values(date_column)
    
    df[f'{value_column}_rolling_mean'] = df[value_column].rolling(window, min_periods=1).mean()
    df[f'{value_column}_rolling_std'] = df[value_column].rolling(window, min_periods=1).std()
    df[f'{value_column}_rolling_min'] = df[value_column].rolling(window, min_periods=1).min()
    df[f'{value_column}_rolling_max'] = df[value_column].rolling(window, min_periods=1).max()
    
    return df


def calculate_growth_rate(
    current_value: int,
    previous_value: int
) -> float:
    """
    Calculate percentage growth rate.
    
    Args:
        current_value: Current period value
        previous_value: Previous period value
        
    Returns:
        Growth rate as percentage
    """
    if previous_value == 0:
        return 100.0 if current_value > 0 else 0.0
    
    return round(((current_value - previous_value) / previous_value) * 100, 2)


def normalize_intensity(values: List[float], max_val: float = 100) -> List[float]:
    """
    Normalize values to 0-max_val scale for heatmap intensity.
    
    Args:
        values: List of values to normalize
        max_val: Maximum value in output scale
        
    Returns:
        Normalized values
    """
    if not values:
        return []
    
    min_v = min(values)
    max_v = max(values)
    
    if max_v == min_v:
        return [max_val / 2] * len(values)
    
    return [
        round((v - min_v) / (max_v - min_v) * max_val, 2)
        for v in values
    ]
