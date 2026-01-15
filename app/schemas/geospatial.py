"""
Geospatial Pydantic schemas for heatmap data.
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import date


class HeatmapDataPoint(BaseModel):
    """Single heatmap data point."""
    lat: float
    lng: float
    intensity: float  # 0-100 normalized
    name: str  # District or state name
    value: int  # Raw count
    state: Optional[str] = None


class HeatmapResponse(BaseModel):
    """Response for heatmap endpoints."""
    locations: List[HeatmapDataPoint]
    total_locations: int
    max_intensity: float
    min_intensity: float
    data_type: str  # enrollment, demographic, biometric
    aggregation_level: str  # state, district
    simulation_date: date


class GeoJSONFeature(BaseModel):
    """GeoJSON feature."""
    type: str = "Feature"
    geometry: dict
    properties: dict


class GeoJSONResponse(BaseModel):
    """GeoJSON response."""
    type: str = "FeatureCollection"
    features: List[GeoJSONFeature]


class DistrictCentroid(BaseModel):
    """District centroid with data."""
    district: str
    state: str
    lat: float
    lng: float
    enrollment_count: int
    demographic_count: int
    biometric_count: int


class CentroidsResponse(BaseModel):
    """Response with district centroids."""
    centroids: List[DistrictCentroid]
    total_count: int


class MigrationHeatmapPoint(BaseModel):
    """Migration heatmap data point."""
    lat: float
    lng: float
    intensity: float
    district: str
    state: str
    inflow: int  # People moving in
    outflow: int  # People moving out
    net_migration: int


class MigrationHeatmapResponse(BaseModel):
    """Response for migration heatmap."""
    districts: List[MigrationHeatmapPoint]
    total_districts: int
    simulation_date: date
    view_mode: str
