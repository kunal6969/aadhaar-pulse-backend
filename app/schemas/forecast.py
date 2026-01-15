"""
Forecast Pydantic schemas for ML predictions.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date


class ForecastRequest(BaseModel):
    """Request for MBU demand forecast."""
    district: str = Field(..., description="District name for forecast")
    forecast_from: date = Field(..., description="Start forecast from this date (simulation_date)")
    horizon_days: int = Field(180, ge=7, le=365, description="Days ahead to predict")


class ForecastDataPoint(BaseModel):
    """Single forecast data point."""
    date: date
    predicted: float
    lower_bound: float
    upper_bound: float


class HistoricalDataPoint(BaseModel):
    """Single historical data point."""
    date: date
    actual: int


class RiskAssessment(BaseModel):
    """Risk assessment for a district."""
    risk_level: str  # LOW, MEDIUM, HIGH, CRITICAL
    risk_score: float
    message: str
    expected_mbu_6months: int
    actual_updates_30days: int
    current_5_17_population: int


class MBUForecastResponse(BaseModel):
    """Response for MBU forecast endpoint."""
    district: str
    forecast_from: date
    horizon_days: int
    historical: List[HistoricalDataPoint]
    forecast: List[ForecastDataPoint]
    risk_assessment: RiskAssessment
    model_info: Dict[str, str]


class TrendForecastResponse(BaseModel):
    """Response for general trend forecast."""
    data_type: str  # enrollment, demographic, biometric
    aggregation_level: str  # state, district
    aggregation_value: str  # State/District name
    historical: List[HistoricalDataPoint]
    forecast: List[ForecastDataPoint]
    confidence_level: float


class SeasonalComponent(BaseModel):
    """Seasonal component of forecast."""
    name: str
    period: str
    peak_month: int
    trough_month: int
    amplitude: float


class ForecastDecomposition(BaseModel):
    """Decomposed forecast components."""
    trend: List[Dict[str, float]]
    seasonal: List[SeasonalComponent]
    residual_std: float
