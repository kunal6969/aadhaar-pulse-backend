"""
Biometric Update Pydantic schemas for request/response validation.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date


class BiometricRecord(BaseModel):
    """Single biometric update record."""
    date: date
    state: str
    district: str
    pincode: Optional[str] = None
    bio_age_5_17: int
    bio_age_17_plus: int
    total: int
    
    class Config:
        from_attributes = True


class BiometricTrendDataPoint(BaseModel):
    """Single data point in biometric trend."""
    date: date
    total: int
    bio_age_5_17: Optional[int] = None
    bio_age_17_plus: Optional[int] = None


class BiometricTrendsResponse(BaseModel):
    """Response with biometric trend data."""
    data: List[BiometricTrendDataPoint]
    total_records: int
    aggregation_level: str
    requested_date: date
    actual_date_range: Dict[str, str]


class StateBiometricSummary(BaseModel):
    """Summary for a single state."""
    state: str
    count: int
    percentage: float


class BiometricSummaryResponse(BaseModel):
    """KPI summary for biometric updates."""
    total_updates_30d: int
    total_updates_7d: int
    total_updates_today: int
    active_districts: int
    active_states: int
    top_states: List[StateBiometricSummary]
    simulation_date: date
    
    # Age group breakdown (for MBU analysis)
    age_5_17_total: int
    age_17_plus_total: int
    
    # MBU specific metrics
    pending_mbu_estimate: int  # Estimated children approaching 15
    mbu_completion_rate: float  # Percentage of expected MBUs completed


class BiometricByDistrictResponse(BaseModel):
    """Biometric data grouped by district."""
    district: str
    state: str
    total: int
    bio_age_5_17: int
    bio_age_17_plus: int
    daily_average: float
    mbu_risk_level: str  # LOW, MEDIUM, HIGH
