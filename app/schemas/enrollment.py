"""
Enrollment Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date


class EnrollmentRecord(BaseModel):
    """Single enrollment record."""
    date: date
    state: str
    district: str
    pincode: Optional[str] = None
    age_0_5: int
    age_5_17: int
    age_18_plus: int
    total: int
    
    class Config:
        from_attributes = True


class EnrollmentTrendDataPoint(BaseModel):
    """Single data point in enrollment trend."""
    date: date
    total: int
    age_0_5: Optional[int] = None
    age_5_17: Optional[int] = None
    age_18_plus: Optional[int] = None


class EnrollmentTrendsRequest(BaseModel):
    """Request for enrollment trends."""
    simulation_date: date
    view_mode: str = Field("daily", pattern="^(daily|monthly|quarterly)$")
    chart_start_date: Optional[date] = None
    chart_end_date: Optional[date] = None
    state: Optional[str] = None
    district: Optional[str] = None
    age_group: Optional[str] = Field(None, pattern="^(0-5|5-17|18\\+|all)$")


class EnrollmentTrendsResponse(BaseModel):
    """Response with enrollment trend data."""
    data: List[EnrollmentTrendDataPoint]
    total_records: int
    aggregation_level: str
    requested_date: date
    actual_date_range: Dict[str, str]


class StateEnrollmentSummary(BaseModel):
    """Summary for a single state."""
    state: str
    count: int
    percentage: float


class EnrollmentSummaryResponse(BaseModel):
    """KPI summary for overview page."""
    total_enrollments_30d: int
    total_enrollments_7d: int
    total_enrollments_today: int
    active_districts: int
    active_states: int
    top_states: List[StateEnrollmentSummary]
    simulation_date: date
    
    # Age group breakdown
    age_0_5_total: int
    age_5_17_total: int
    age_18_plus_total: int


class EnrollmentByDistrictResponse(BaseModel):
    """Enrollment data grouped by district."""
    district: str
    state: str
    total: int
    age_0_5: int
    age_5_17: int
    age_18_plus: int
    daily_average: float


class EnrollmentComparisonResponse(BaseModel):
    """Compare enrollment between two periods."""
    period_1: Dict[str, int]
    period_2: Dict[str, int]
    change_absolute: int
    change_percentage: float
