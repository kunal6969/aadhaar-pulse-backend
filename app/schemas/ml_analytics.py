"""
Pydantic schemas for ML Analytics API responses.
"""
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
from datetime import date
from enum import Enum


# ==================== ENUMS ====================

class DataType(str, Enum):
    ENROLLMENT = "enrollment"
    DEMOGRAPHIC = "demographic"
    BIOMETRIC = "biometric"


class RiskLevel(str, Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class FraudRiskLevel(str, Enum):
    CLEAN = "CLEAN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class HotspotLevel(str, Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


class DistrictSegment(str, Enum):
    STABLE = "A_STABLE"
    HIGH_DEMOGRAPHIC_CHURN = "B_HIGH_DEMOGRAPHIC_CHURN"
    HIGH_BIOMETRIC_RETRY = "C_HIGH_BIOMETRIC_RETRY"
    ENROLLMENT_HOTSPOT = "D_ENROLLMENT_HOTSPOT"
    SUSPICIOUS_PATTERN = "E_SUSPICIOUS_PATTERN"


class DemandUrgency(str, Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class InterventionType(str, Enum):
    MOBILE_UNIT = "mobile_unit"
    CAMP = "awareness_camp"
    EXTRA_CENTER = "additional_center"
    NO_ACTION = "no_action_needed"
    PRIORITY_TARGETING = "priority_targeting"


# ==================== FORECASTING SCHEMAS ====================

class ForecastPoint(BaseModel):
    """Single forecast data point."""
    date: str
    predicted: float
    lower_bound: float
    upper_bound: float
    trend: Optional[float] = None
    seasonal: Optional[float] = None


class SeasonalityPattern(BaseModel):
    """Seasonality pattern analysis."""
    day_of_week: Dict[str, float]
    monthly: Dict[str, float]
    quarterly: Dict[str, float]
    weekend_reduction: str
    peak_days: List[int]
    low_days: List[int]
    recommendations: List[str]


class DemandForecastResponse(BaseModel):
    """Response from demand forecasting endpoint."""
    location: Dict[str, Optional[str]]
    data_type: str
    forecast_from: str
    horizon_days: int
    historical_summary: Dict[str, Any]
    forecasts: List[ForecastPoint]
    seasonality_calendar: SeasonalityPattern
    model_info: Dict[str, Any]


# ==================== CAPACITY PLANNING SCHEMAS ====================

class CapacityRequirement(BaseModel):
    """Capacity requirement for a district."""
    district: str
    state: str
    forecasted_demand_daily: float
    service_rate_per_operator: float
    operators_required: int
    current_operators: Optional[int] = None
    operator_gap: Optional[int] = None
    utilization_rate: float
    expected_wait_time_hours: float
    expected_queue_length: float
    sla_risk_level: RiskLevel
    sla_risk_score: float
    weekday_operators: int
    weekend_operators: int


class CapacityPlanningResponse(BaseModel):
    """Response from capacity planning endpoint."""
    state: str
    simulation_date: str
    summary: Dict[str, Any]
    district_requirements: List[CapacityRequirement]
    model_info: Dict[str, Any]


class YearlyCapacityPlan(BaseModel):
    """Yearly capacity plan for a district."""
    district: str
    state: str
    monthly_requirements: List[Dict[str, Any]]
    annual_peak_operators: int
    annual_avg_operators: int
    recommended_permanent_staff: int
    recommended_seasonal_staff: int
    total_annual_demand: int
    capacity_utilization_yearly: float


# ==================== UNDERSERVED SCORING SCHEMAS ====================

class UnderservedScore(BaseModel):
    """Underserved score for a district."""
    district: str
    state: str
    overall_score: float
    component_scores: Dict[str, float]
    recommended_intervention: InterventionType
    intervention_priority: int
    estimated_roi: float
    action_required: bool


class MobileUnitPlacement(BaseModel):
    """Mobile unit placement recommendation."""
    district: str
    state: str
    placement_score: float
    estimated_daily_demand: float
    coverage_gap_pct: float
    population_density_factor: float
    recommended_days_per_month: int
    expected_monthly_enrollments: int
    roi_score: float


class UnderservedResponse(BaseModel):
    """Response from underserved scoring endpoint."""
    state: str
    simulation_date: str
    summary: Dict[str, Any]
    district_scores: List[UnderservedScore]
    model_info: Dict[str, Any]


# ==================== FRAUD DETECTION SCHEMAS ====================

class DigitDistribution(BaseModel):
    """Distribution of digits."""
    leading_digits: Dict[str, float]
    last_digits: Dict[str, float]
    round_number_pct: float
    sample_size: int


class FraudScore(BaseModel):
    """Fraud suspicion score for a district."""
    district: str
    state: str
    overall_fraud_score: float
    component_scores: Dict[str, float]
    risk_level: FraudRiskLevel
    reason_codes: List[str]
    recommendation: str
    digit_distribution: DigitDistribution
    requires_audit: bool


class AuditListItem(BaseModel):
    """District requiring audit."""
    district: str
    state: str
    risk_level: str
    fraud_score: float
    reason_codes: List[str]
    recommendation: str
    priority: str


class FraudAnalysisResponse(BaseModel):
    """Response from fraud analysis endpoint."""
    state: str
    analysis_period: str
    summary: Dict[str, Any]
    audit_list: List[AuditListItem]
    all_scores: List[FraudScore]
    model_info: Dict[str, Any]


# ==================== CLUSTERING SCHEMAS ====================

class SegmentProfile(BaseModel):
    """Profile of a segment."""
    segment_code: str
    segment_name: str
    description: str
    recommended_action: str
    characteristics: Dict[str, str]


class ClusterResult(BaseModel):
    """Clustering result for a district."""
    district: str
    state: str
    cluster_id: int
    segment: DistrictSegment
    anomaly_score: float
    is_infiltration_zone: bool
    features: Dict[str, float]
    profile: SegmentProfile


class InfiltrationZone(BaseModel):
    """Potential infiltration zone."""
    district: str
    state: str
    segment: str
    anomaly_score: float
    reason: str
    recommended_action: str


class InterventionRecommendation(BaseModel):
    """Intervention recommendation by segment."""
    priority: int
    segment: str
    segment_name: str
    district_count: int
    action: str
    top_districts: List[Dict[str, str]]


class ClusteringResponse(BaseModel):
    """Response from clustering endpoint."""
    state: str
    n_clusters: int
    segment_summary: Dict[str, Any]
    infiltration_zones: List[InfiltrationZone]
    intervention_recommendations: List[InterventionRecommendation]
    district_assignments: List[ClusterResult]
    model_info: Dict[str, Any]


# ==================== HOTSPOT DETECTION SCHEMAS ====================

class ControlLimits(BaseModel):
    """EWMA control chart limits."""
    center_line: float
    upper_warning: float
    upper_control: float
    lower_warning: float
    lower_control: float


class HotspotResult(BaseModel):
    """Hotspot detection result."""
    district: str
    state: str
    current_value: float
    ewma_value: float
    control_limits: ControlLimits
    deviation_sigma: float
    rolling_increase_pct: float
    hotspot_level: HotspotLevel
    consecutive_breaches: int
    infrastructure_load_pct: float
    risk_factors: List[str]
    recommendation: str
    is_hotspot: bool


class InfrastructureDemand(BaseModel):
    """Infrastructure demand for a district."""
    rank: int
    district: str
    state: str
    current_load: float
    capacity: float
    utilization_pct: float
    failure_risk: str
    devices_needed: int
    action: str


class HotspotResponse(BaseModel):
    """Response from hotspot detection endpoint."""
    state: str
    simulation_date: str
    summary: Dict[str, Any]
    hotspots: List[HotspotResult]
    model_info: Dict[str, Any]


# ==================== COHORT MODEL SCHEMAS ====================

class YearlyPrediction(BaseModel):
    """Yearly MBU prediction."""
    year: int
    expected_mbu_demand: int
    daily_average: float
    contributing_cohorts: List[Dict[str, Any]]


class EquipmentRecommendation(BaseModel):
    """Equipment recommendation."""
    devices_needed_at_peak: int
    peak_daily_demand: float
    years_to_peak: int
    procurement_timeline: str
    recommended_action: str
    estimated_investment_inr: int


class FiveYearPrediction(BaseModel):
    """5-year MBU prediction for a district."""
    district: str
    state: str
    current_year: int
    yearly_predictions: List[YearlyPrediction]
    total_5_year_demand: int
    peak_year: int
    peak_demand: int
    cumulative_by_year: List[int]
    demand_urgency: DemandUrgency
    equipment_recommendation: EquipmentRecommendation


class PriorityDistrict(BaseModel):
    """Priority district for MBU preparation."""
    district: str
    state: str
    total_5_year_demand: int
    peak_year: int
    peak_demand: int
    urgency: str
    equipment_needed: int
    action_timeline: str


class CohortPredictionResponse(BaseModel):
    """Response from cohort prediction endpoint."""
    state: str
    district: Optional[str] = None
    current_year: int
    aggregate_5_year_demand: Dict[int, int]
    priority_districts: List[PriorityDistrict]
    detailed_predictions: List[FiveYearPrediction]
    model_info: Dict[str, Any]


class EquipmentPlan(BaseModel):
    """Equipment plan for a district."""
    district: str
    state: str
    current_devices: int
    peak_year_demand: int
    devices_at_peak: int
    additional_devices_needed: int
    investment_schedule: List[Dict[str, Any]]
    total_5_year_investment_lakhs: float
    roi_assessment: str


class EquipmentPlanResponse(BaseModel):
    """Response from equipment plan endpoint."""
    state: str
    summary: Dict[str, Any]
    equipment_plans: List[EquipmentPlan]
    model_info: Dict[str, Any]


# ==================== MODEL INFO SCHEMAS ====================

class ModelInfo(BaseModel):
    """Information about an ML model."""
    id: str
    name: str
    description: str
    endpoints: List[str]
    use_cases: List[str]


class ModelsListResponse(BaseModel):
    """Response listing all available models."""
    models: List[ModelInfo]
    total_models: int
    solution_coverage: Dict[str, List[str]]
