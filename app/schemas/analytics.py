"""
Analytics and KPI Pydantic schemas.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
from datetime import date


class TopState(BaseModel):
    """Top state by activity."""
    state: str
    enrollment_count: int
    demographic_count: int
    biometric_count: int
    total_activity: int


class KPIResponse(BaseModel):
    """Dashboard KPIs response."""
    simulation_date: date
    
    # Enrollment KPIs
    total_enrollments_30d: int
    total_enrollments_7d: int
    total_enrollments_today: int
    enrollment_growth_rate: float  # % change from previous period
    
    # Demographic Update KPIs
    total_demo_updates_30d: int
    total_demo_updates_7d: int
    total_demo_updates_today: int
    demo_growth_rate: float
    
    # Biometric Update KPIs
    total_bio_updates_30d: int
    total_bio_updates_7d: int
    total_bio_updates_today: int
    bio_growth_rate: float
    
    # Geographic KPIs
    active_districts: int
    active_states: int
    
    # MBU specific
    pending_mbu_count: int
    mbu_completion_percentage: float
    
    # Top performers
    top_5_states: List[TopState]


class DistrictBurden(BaseModel):
    """Update burden for a single district."""
    district: str
    state: str
    burden_score: float
    demographic_updates: int
    biometric_updates: int
    population_estimate: int
    burden_level: str  # LOW, MEDIUM, HIGH, CRITICAL


class UpdateBurdenResponse(BaseModel):
    """Update burden index response."""
    districts: List[DistrictBurden]
    total_districts: int
    high_burden_count: int
    simulation_date: date


class DistrictReadiness(BaseModel):
    """Digital readiness for a single district."""
    district: str
    state: str
    readiness_score: float  # 0-100
    mobile_update_frequency: float
    stability_score: float
    recommendation: str


class DigitalReadinessResponse(BaseModel):
    """Digital readiness scores response."""
    districts: List[DistrictReadiness]
    total_districts: int
    high_readiness_count: int
    low_readiness_count: int
    simulation_date: date


class MigrationCorridor(BaseModel):
    """Migration corridor between regions."""
    source_district: str
    source_state: str
    destination_district: str
    destination_state: str
    migration_intensity: int
    percentage_of_total: float


class MigrationCorridorsResponse(BaseModel):
    """Migration corridors response."""
    corridors: List[MigrationCorridor]
    total_count: int
    as_of_date: date
    view_mode: str


class SeasonalPattern(BaseModel):
    """Seasonal migration pattern."""
    month: int
    month_name: str
    migration_intensity: int
    is_peak: bool
    notes: str


class SeasonalPatternsResponse(BaseModel):
    """Seasonal patterns response."""
    patterns: List[SeasonalPattern]
    peak_months: List[int]
    low_months: List[int]
