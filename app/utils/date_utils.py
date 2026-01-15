"""
Date utility functions for simulation date handling.
"""
from datetime import date, timedelta
from typing import List, Optional, Tuple
import json
import os
from app.config import settings


def load_available_dates() -> List[date]:
    """
    Load available dates from JSON file.
    
    Returns:
        List of dates that have data in the dataset.
    """
    dates_file = os.path.join(settings.PROCESSED_DATA_DIR, "available_dates.json")
    
    if not os.path.exists(dates_file):
        return []
    
    with open(dates_file, 'r') as f:
        data = json.load(f)
    
    return [date.fromisoformat(d) for d in data.get('dates', [])]


def snap_to_nearest_available_date(
    requested_date: date, 
    available_dates: List[date]
) -> date:
    """
    Snap requested date to nearest available date.
    If requested_date has no data, return nearest available date.
    
    Args:
        requested_date: The date requested by user
        available_dates: List of dates with available data
        
    Returns:
        Nearest available date
    """
    if not available_dates:
        return requested_date
    
    if requested_date in available_dates:
        return requested_date
    
    # Find dates before and after
    past_dates = [d for d in available_dates if d <= requested_date]
    future_dates = [d for d in available_dates if d > requested_date]
    
    # Prefer past date (return data up to that point)
    if past_dates:
        return max(past_dates)
    
    # If no past dates, return first available
    return min(future_dates)


def validate_simulation_date(simulation_date: date) -> Tuple[bool, str]:
    """
    Validate if date is within simulation period.
    
    Args:
        simulation_date: Date to validate
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    start = date.fromisoformat(settings.SIMULATION_START_DATE)
    end = date.fromisoformat(settings.SIMULATION_END_DATE)
    
    if simulation_date < start:
        return False, f"Simulation date {simulation_date} is before start date {start}"
    
    if simulation_date > end:
        return False, f"Simulation date {simulation_date} is after end date {end}"
    
    return True, ""


def get_date_range_for_view_mode(
    simulation_date: date, 
    view_mode: str,
    lookback_days: int = 90
) -> Tuple[date, date]:
    """
    Get start and end dates based on view mode.
    
    Args:
        simulation_date: The simulated current date
        view_mode: daily, monthly, or quarterly
        lookback_days: Default days to look back (for daily mode)
        
    Returns:
        Tuple of (start_date, end_date)
    """
    if view_mode == "daily":
        start_date = simulation_date - timedelta(days=lookback_days)
        end_date = simulation_date
        
    elif view_mode == "monthly":
        # Get first day of month 3 months ago
        month = simulation_date.month
        year = simulation_date.year
        
        # Go back 3 months
        for _ in range(3):
            month -= 1
            if month < 1:
                month = 12
                year -= 1
        
        start_date = date(year, month, 1)
        end_date = simulation_date
        
    elif view_mode == "quarterly":
        # Get start of current quarter and 2 previous quarters
        current_quarter = (simulation_date.month - 1) // 3 + 1
        year = simulation_date.year
        
        # Go back 2 quarters
        start_quarter = current_quarter - 2
        if start_quarter < 1:
            start_quarter += 4
            year -= 1
        
        start_month = (start_quarter - 1) * 3 + 1
        start_date = date(year, start_month, 1)
        end_date = simulation_date
    
    else:
        # Default to daily
        start_date = simulation_date - timedelta(days=lookback_days)
        end_date = simulation_date
    
    return start_date, end_date


def parse_date_string(date_str: str, format: str = "%d-%m-%Y") -> Optional[date]:
    """
    Parse date string to date object.
    Handles multiple formats.
    
    Args:
        date_str: Date string to parse
        format: Expected format
        
    Returns:
        Parsed date or None if invalid
    """
    from datetime import datetime
    
    # Try multiple formats
    formats = [
        "%d-%m-%Y",  # DD-MM-YYYY (CSV format)
        "%Y-%m-%d",  # YYYY-MM-DD (ISO format)
        "%d/%m/%Y",  # DD/MM/YYYY
        "%Y/%m/%d",  # YYYY/MM/DD
    ]
    
    for fmt in formats:
        try:
            return datetime.strptime(date_str.strip(), fmt).date()
        except ValueError:
            continue
    
    return None


def get_month_date_range(simulation_date: date) -> Tuple[date, date]:
    """
    Get first and last day of month containing simulation_date.
    """
    import calendar
    
    first_day = simulation_date.replace(day=1)
    last_day_num = calendar.monthrange(simulation_date.year, simulation_date.month)[1]
    last_day = simulation_date.replace(day=last_day_num)
    
    return first_day, last_day


def get_quarter_date_range(simulation_date: date) -> Tuple[date, date]:
    """
    Get first and last day of quarter containing simulation_date.
    """
    import calendar
    
    quarter = (simulation_date.month - 1) // 3 + 1
    start_month = (quarter - 1) * 3 + 1
    end_month = quarter * 3
    
    first_day = date(simulation_date.year, start_month, 1)
    last_day_num = calendar.monthrange(simulation_date.year, end_month)[1]
    last_day = date(simulation_date.year, end_month, last_day_num)
    
    return first_day, last_day


def days_between(start: date, end: date) -> int:
    """Calculate number of days between two dates."""
    return (end - start).days
