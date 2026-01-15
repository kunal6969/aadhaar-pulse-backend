"""
Anomaly Detection Pydantic schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date


class AnomalyRecord(BaseModel):
    """Single anomaly detection record."""
    id: str
    pincode: Optional[str]
    district: str
    state: str
    date: date
    data_type: str  # enrollment, demographic, biometric
    observed_count: int
    baseline_mean: float
    deviation_ratio: float
    severity: Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    anomaly_score: float
    message: str
    
    class Config:
        from_attributes = True


class AnomalyDetectionRequest(BaseModel):
    """Request for anomaly detection."""
    simulation_date: date
    time_window: Literal["7d", "30d", "90d", "all"] = "30d"
    severity: Optional[Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]] = None
    state: Optional[str] = None
    district: Optional[str] = None
    data_type: Optional[Literal["enrollment", "demographic", "biometric", "all"]] = "all"


class AnomalyDetectionResponse(BaseModel):
    """Response for anomaly detection."""
    anomalies: List[AnomalyRecord]
    total_count: int
    time_window: str
    severity_breakdown: dict
    simulation_date: date


class AnomalyAlert(BaseModel):
    """Simplified anomaly for alert feed."""
    id: str
    district: str
    state: str
    date: date
    severity: str
    message: str
    data_type: str


class RecentAnomaliesResponse(BaseModel):
    """Response for recent anomalies endpoint."""
    alerts: List[AnomalyAlert]
    count: int
    as_of_date: date


class AnomalyTrend(BaseModel):
    """Trend of anomalies over time."""
    date: date
    critical_count: int
    high_count: int
    medium_count: int
    low_count: int
    total_count: int


class AnomalyTrendResponse(BaseModel):
    """Response for anomaly trend over time."""
    trends: List[AnomalyTrend]
    peak_date: date
    peak_count: int
