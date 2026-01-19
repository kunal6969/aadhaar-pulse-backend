"""
Enrollment API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.enrollment_service import EnrollmentService
from app.utils.date_utils import get_date_range_for_view_mode, validate_simulation_date
from datetime import date, timedelta
from typing import Optional, List

router = APIRouter()


@router.get("/trends")
def get_enrollment_trends(
    simulation_date: date = Query(..., description="Simulated current date"),
    view_mode: str = Query("daily", regex="^(daily|monthly|quarterly)$"),
    chart_start_date: Optional[date] = Query(None, description="Start of chart range"),
    chart_end_date: Optional[date] = Query(None, description="End of chart range"),
    state: Optional[str] = Query(None, description="Filter by state"),
    district: Optional[str] = Query(None, description="Filter by district"),
    age_group: Optional[str] = Query(None, regex="^(0-5|5-17|18\\+|all)?$"),
    db: Session = Depends(get_db)
):
    """
    Get enrollment trends filtered by simulation date and geography.
    
    - **simulation_date**: The simulated "today" (data shown up to this date)
    - **view_mode**: daily, monthly, or quarterly aggregation
    - **chart_start_date**: Start of historical range (defaults to 90 days before simulation_date)
    - **chart_end_date**: End of historical range (defaults to simulation_date)
    - **state**: Filter by state name
    - **district**: Filter by district name
    - **age_group**: Filter by age category (0-5, 5-17, 18+, or all)
    """
    # Validate simulation date
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = EnrollmentService(db)
    
    # Set default date range
    if chart_end_date is None:
        chart_end_date = simulation_date
    
    if chart_start_date is None:
        chart_start_date = simulation_date - timedelta(days=90)
    
    # Validate: chart_end_date should not exceed simulation_date
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
def get_enrollment_summary(
    simulation_date: date = Query(..., description="Simulated current date"),
    state: Optional[str] = Query(None, description="Filter by state"),
    district: Optional[str] = Query(None, description="Filter by district"),
    db: Session = Depends(get_db)
):
    """
    Get KPI summary for overview page.
    
    Returns enrollment counts for last 30 days, 7 days, and today 
    relative to simulation_date. Can be filtered by state/district.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = EnrollmentService(db)
    summary = service.get_summary(simulation_date, state=state, district=district)
    
    return summary


@router.get("/by-district")
def get_enrollment_by_district(
    simulation_date: date = Query(..., description="Simulated current date"),
    state: Optional[str] = Query(None, description="Filter by state"),
    days_back: int = Query(30, ge=1, le=365, description="Days to look back"),
    limit: int = Query(50, ge=1, le=500, description="Max districts to return"),
    db: Session = Depends(get_db)
):
    """
    Get enrollment data aggregated by district.
    
    Returns top districts by enrollment count with age breakdown.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = EnrollmentService(db)
    districts = service.get_by_district(
        simulation_date=simulation_date,
        state=state,
        days_back=days_back,
        limit=limit
    )
    
    return {
        "districts": districts,
        "total_count": len(districts),
        "simulation_date": simulation_date.isoformat(),
        "days_back": days_back
    }


@router.get("/states")
def get_states_list(db: Session = Depends(get_db)):
    """Get list of all states in the dataset."""
    service = EnrollmentService(db)
    states = service.get_states_list()
    
    return {"states": states, "count": len(states)}


@router.get("/districts")
def get_districts_list(
    state: Optional[str] = Query(None, description="Filter by state"),
    db: Session = Depends(get_db)
):
    """Get list of districts, optionally filtered by state."""
    service = EnrollmentService(db)
    districts = service.get_districts_list(state=state)
    
    return {"districts": districts, "count": len(districts)}
