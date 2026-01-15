"""
Schemas package initialization.
"""
from app.schemas.common import (
    SimulationContext,
    GeographyFilter,
    PaginationParams,
    PaginatedResponse,
    DateRangeFilter,
)
from app.schemas.enrollment import (
    EnrollmentRecord,
    EnrollmentTrendsRequest,
    EnrollmentTrendsResponse,
    EnrollmentSummaryResponse,
)
from app.schemas.demographic import (
    DemographicRecord,
    DemographicTrendsResponse,
    DemographicSummaryResponse,
)
from app.schemas.biometric import (
    BiometricRecord,
    BiometricTrendsResponse,
    BiometricSummaryResponse,
)
from app.schemas.forecast import (
    ForecastRequest,
    ForecastDataPoint,
    MBUForecastResponse,
)
from app.schemas.anomaly import (
    AnomalyRecord,
    AnomalyDetectionResponse,
)
from app.schemas.analytics import (
    KPIResponse,
    UpdateBurdenResponse,
    DigitalReadinessResponse,
)
from app.schemas.geospatial import (
    HeatmapDataPoint,
    HeatmapResponse,
)

__all__ = [
    # Common
    "SimulationContext",
    "GeographyFilter",
    "PaginationParams",
    "PaginatedResponse",
    "DateRangeFilter",
    # Enrollment
    "EnrollmentRecord",
    "EnrollmentTrendsRequest",
    "EnrollmentTrendsResponse",
    "EnrollmentSummaryResponse",
    # Demographic
    "DemographicRecord",
    "DemographicTrendsResponse",
    "DemographicSummaryResponse",
    # Biometric
    "BiometricRecord",
    "BiometricTrendsResponse",
    "BiometricSummaryResponse",
    # Forecast
    "ForecastRequest",
    "ForecastDataPoint",
    "MBUForecastResponse",
    # Anomaly
    "AnomalyRecord",
    "AnomalyDetectionResponse",
    # Analytics
    "KPIResponse",
    "UpdateBurdenResponse",
    "DigitalReadinessResponse",
    # Geospatial
    "HeatmapDataPoint",
    "HeatmapResponse",
]
