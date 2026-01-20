"""
ML Models for Aadhaar Pulse Analytics Platform.

This package contains 7 core ML models:
1. Hierarchical Time Series Forecasting - demand prediction
2. Queueing/Capacity Planning Model - operator requirements
3. Underserved Scoring Model - mobile unit placement, equity targeting
4. Forensic Digit Fraud Model - Benford's Law analysis
5. Clustering/Segmentation Model - district zone detection
6. EWMA Hotspot Detection Model - biometric infrastructure load
7. Cohort Transition Model - 5-year MBU prediction
"""

from .time_series_forecasting import HierarchicalForecaster
from .capacity_planning import CapacityPlanningModel
from .underserved_scoring import UnderservedScoringModel
from .fraud_detection import ForensicFraudDetector
from .clustering import DistrictClusteringModel
from .hotspot_detection import EWMAHotspotDetector
from .cohort_model import CohortTransitionModel
from .model_registry import (
    ModelRegistry,
    registry,
    get_model,
    load_all_models,
    get_model_status,
    is_model_trained
)

__all__ = [
    'HierarchicalForecaster',
    'CapacityPlanningModel',
    'UnderservedScoringModel',
    'ForensicFraudDetector',
    'DistrictClusteringModel',
    'EWMAHotspotDetector',
    'CohortTransitionModel',
    'ModelRegistry',
    'registry',
    'get_model',
    'load_all_models',
    'get_model_status',
    'is_model_trained'
]
