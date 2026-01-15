"""
Geospatial API endpoints for heatmap data.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.geospatial_processor import GeospatialProcessor
from app.utils.date_utils import validate_simulation_date
from app.utils.constants import STATE_CENTROIDS, DISTRICT_CENTROIDS
from datetime import date
from typing import Literal, Optional

router = APIRouter()


@router.get("/heatmap/enrollment")
def get_enrollment_heatmap(
    simulation_date: date = Query(...),
    view_mode: str = Query("monthly", regex="^(daily|monthly)$"),
    level: Literal["state", "district"] = Query("district"),
    db: Session = Depends(get_db)
):
    """
    Generate enrollment intensity heatmap data.
    
    Returns array of {lat, lng, intensity, name, value} objects 
    for rendering on a map.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    processor = GeospatialProcessor(db)
    heatmap_data = processor.generate_enrollment_heatmap(
        simulation_date=simulation_date,
        view_mode=view_mode,
        level=level
    )
    
    return {
        "locations": heatmap_data,
        "total_locations": len(heatmap_data),
        "data_type": "enrollment",
        "aggregation_level": level,
        "simulation_date": simulation_date.isoformat()
    }


@router.get("/heatmap/demographic")
def get_demographic_heatmap(
    simulation_date: date = Query(...),
    view_mode: str = Query("monthly", regex="^(daily|monthly)$"),
    level: Literal["state", "district"] = Query("district"),
    db: Session = Depends(get_db)
):
    """Generate demographic update intensity heatmap data."""
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    processor = GeospatialProcessor(db)
    heatmap_data = processor.generate_demographic_heatmap(
        simulation_date=simulation_date,
        view_mode=view_mode,
        level=level
    )
    
    return {
        "locations": heatmap_data,
        "total_locations": len(heatmap_data),
        "data_type": "demographic",
        "aggregation_level": level,
        "simulation_date": simulation_date.isoformat()
    }


@router.get("/heatmap/biometric")
def get_biometric_heatmap(
    simulation_date: date = Query(...),
    view_mode: str = Query("monthly", regex="^(daily|monthly)$"),
    level: Literal["state", "district"] = Query("district"),
    db: Session = Depends(get_db)
):
    """Generate biometric update intensity heatmap data."""
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    processor = GeospatialProcessor(db)
    heatmap_data = processor.generate_biometric_heatmap(
        simulation_date=simulation_date,
        view_mode=view_mode,
        level=level
    )
    
    return {
        "locations": heatmap_data,
        "total_locations": len(heatmap_data),
        "data_type": "biometric",
        "aggregation_level": level,
        "simulation_date": simulation_date.isoformat()
    }


@router.get("/heatmap/combined")
def get_combined_heatmap(
    simulation_date: date = Query(...),
    view_mode: str = Query("monthly", regex="^(daily|monthly)$"),
    level: Literal["state", "district"] = Query("district"),
    db: Session = Depends(get_db)
):
    """
    Generate combined activity heatmap (all data types).
    
    Returns locations with enrollment, demographic, and biometric counts.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    processor = GeospatialProcessor(db)
    heatmap_data = processor.generate_combined_heatmap(
        simulation_date=simulation_date,
        view_mode=view_mode,
        level=level
    )
    
    return {
        "locations": heatmap_data,
        "total_locations": len(heatmap_data),
        "data_type": "combined",
        "aggregation_level": level,
        "simulation_date": simulation_date.isoformat()
    }


@router.get("/centroids/states")
def get_state_centroids():
    """
    Get all state centroids.
    
    Returns state names with lat/lng coordinates.
    """
    centroids = [
        {"name": name, **coords}
        for name, coords in STATE_CENTROIDS.items()
    ]
    
    return {
        "centroids": centroids,
        "count": len(centroids)
    }


@router.get("/centroids/districts")
def get_district_centroids(
    state: Optional[str] = Query(None, description="Filter by state")
):
    """
    Get district centroids.
    
    Returns district names with lat/lng coordinates.
    Optionally filter by state.
    """
    centroids = []
    for name, data in DISTRICT_CENTROIDS.items():
        if state is None or data.get("state", "").lower() == state.lower():
            centroids.append({
                "name": name,
                "lat": data["lat"],
                "lng": data["lng"],
                "state": data.get("state", "")
            })
    
    return {
        "centroids": centroids,
        "count": len(centroids)
    }
