"""
Demographic Update API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.demographic_service import DemographicService
from app.utils.date_utils import validate_simulation_date
from datetime import date, timedelta
from typing import Optional

router = APIRouter()


@router.get("/trends")
def get_demographic_trends(
    simulation_date: date = Query(..., description="Simulated current date"),
    view_mode: str = Query("daily", regex="^(daily|monthly|quarterly)$"),
    chart_start_date: Optional[date] = Query(None),
    chart_end_date: Optional[date] = Query(None),
    state: Optional[str] = Query(None),
    district: Optional[str] = Query(None),
    age_group: Optional[str] = Query(None, regex="^(5-17|17\\+|all)?$"),
    db: Session = Depends(get_db)
):
    """
    Get demographic update trends filtered by simulation date and geography.
    
    - **simulation_date**: The simulated "today"
    - **view_mode**: daily, monthly, or quarterly aggregation
    - **chart_start_date**: Start of historical range
    - **chart_end_date**: End of historical range
    - **state**: Filter by state name
    - **district**: Filter by district name
    - **age_group**: Filter by age category (5-17, 17+, or all)
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = DemographicService(db)
    
    # Set defaults
    if chart_end_date is None:
        chart_end_date = simulation_date
    if chart_start_date is None:
        chart_start_date = simulation_date - timedelta(days=90)
    
    if chart_end_date > simulation_date:
        raise HTTPException(
            status_code=400,
            detail="chart_end_date cannot be after simulation_date"
        )
    
    data = service.get_trends(
        start_date=chart_start_date,
        end_date=chart_end_date,
        view_mode=view_mode,
        state=state,
        district=district,
        age_group=age_group
    )
    
    return {
        "data": data,
        "total_records": len(data),
        "aggregation_level": view_mode,
        "requested_date": simulation_date.isoformat(),
        "actual_date_range": {
            "start": chart_start_date.isoformat(),
            "end": chart_end_date.isoformat()
        }
    }


@router.get("/summary")
def get_demographic_summary(
    simulation_date: date = Query(..., description="Simulated current date"),
    db: Session = Depends(get_db)
):
    """
    Get KPI summary for demographic updates.
    
    Returns update counts for last 30 days, 7 days, and today.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = DemographicService(db)
    summary = service.get_summary(simulation_date)
    
    return summary


@router.get("/by-district")
def get_demographic_by_district(
    simulation_date: date = Query(...),
    state: Optional[str] = Query(None),
    days_back: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """Get demographic data aggregated by district."""
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = DemographicService(db)
    districts = service.get_by_district(
        simulation_date=simulation_date,
        state=state,
        days_back=days_back,
        limit=limit
    )
    
    return {
        "districts": districts,
        "total_count": len(districts),
        "simulation_date": simulation_date.isoformat()
    }
