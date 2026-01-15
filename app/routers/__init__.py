"""
Routers package initialization.
"""
from app.routers import enrollment
from app.routers import demographic
from app.routers import biometric
from app.routers import analytics
from app.routers import geospatial
from app.routers import forecasting
from app.routers import anomaly

__all__ = [
    "enrollment",
    "demographic",
    "biometric",
    "analytics",
    "geospatial",
    "forecasting",
    "anomaly",
]
