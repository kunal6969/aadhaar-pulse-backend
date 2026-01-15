"""
Demographic Update Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date


class DemographicRecord(BaseModel):
    """Single demographic update record."""
    date: date
    state: str
    district: str
    pincode: Optional[str] = None
    demo_age_5_17: int
    demo_age_17_plus: int
    total: int
    
    class Config:
        from_attributes = True


class DemographicTrendDataPoint(BaseModel):
    """Single data point in demographic trend."""
    date: date
    total: int
    demo_age_5_17: Optional[int] = None
    demo_age_17_plus: Optional[int] = None


class DemographicTrendsResponse(BaseModel):
    """Response with demographic trend data."""
    data: List[DemographicTrendDataPoint]
    total_records: int
    aggregation_level: str
    requested_date: date
    actual_date_range: Dict[str, str]


class StateDemographicSummary(BaseModel):
    """Summary for a single state."""
    state: str
    count: int
    percentage: float


class DemographicSummaryResponse(BaseModel):
    """KPI summary for demographic updates."""
    total_updates_30d: int
    total_updates_7d: int
    total_updates_today: int
    active_districts: int
    active_states: int
    top_states: List[StateDemographicSummary]
    simulation_date: date
    
    # Age group breakdown
    age_5_17_total: int
    age_17_plus_total: int


class DemographicByDistrictResponse(BaseModel):
    """Demographic data grouped by district."""
    district: str
    state: str
    total: int
    demo_age_5_17: int
    demo_age_17_plus: int
    daily_average: float
