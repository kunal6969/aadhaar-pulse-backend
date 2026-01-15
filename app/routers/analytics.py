"""
Analytics & KPI API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.kpi_calculator import KPICalculator
from app.utils.date_utils import validate_simulation_date
from datetime import date
from typing import Optional

router = APIRouter()


@router.get("/kpis")
def get_dashboard_kpis(
    simulation_date: date = Query(..., description="Simulated current date"),
    db: Session = Depends(get_db)
):
    """
    Get all KPIs for overview dashboard.
    
    Returns:
    - Total enrollments (30d, 7d, today)
    - Total demographic updates (30d, 7d, today)
    - Total biometric updates (30d, 7d, today)
    - Active districts and states
    - Pending MBU count
    - Top 5 states by activity
    - Growth rates
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    calculator = KPICalculator(db)
    kpis = calculator.calculate_all_kpis(simulation_date)
    
    return kpis


@router.get("/update-burden-index")
def get_update_burden_index(
    simulation_date: date = Query(...),
    state: Optional[str] = Query(None, description="Filter by state"),
    limit: int = Query(100, ge=1, le=1000),
    db: Session = Depends(get_db)
):
    """
    Calculate Update Burden Index for districts.
    
    Higher burden score indicates more update activity, potentially
    indicating strained resources or higher demand.
    
    Returns districts ranked by burden score with breakdown.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    calculator = KPICalculator(db)
    burden_data = calculator.calculate_update_burden_index(
        simulation_date=simulation_date,
        state_filter=state
    )
    
    # Apply limit
    burden_data = burden_data[:limit]
    
    # Calculate summary stats
    high_burden_count = sum(1 for d in burden_data if d["burden_level"] in ["HIGH", "CRITICAL"])
    
    return {
        "districts": burden_data,
        "total_districts": len(burden_data),
        "high_burden_count": high_burden_count,
        "simulation_date": simulation_date.isoformat()
    }


@router.get("/digital-readiness")
def get_digital_readiness_scores(
    simulation_date: date = Query(...),
    limit: int = Query(100, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Calculate digital readiness score based on update stability.
    
    Lower update frequency = Higher stability = Better digital readiness.
    
    Returns districts with readiness scores and service recommendations.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    calculator = KPICalculator(db)
    scores = calculator.calculate_digital_readiness(
        simulation_date=simulation_date,
        limit=limit
    )
    
    high_readiness = sum(1 for d in scores if d["readiness_score"] > 70)
    low_readiness = sum(1 for d in scores if d["readiness_score"] < 30)
    
    return {
        "districts": scores,
        "total_districts": len(scores),
        "high_readiness_count": high_readiness,
        "low_readiness_count": low_readiness,
        "simulation_date": simulation_date.isoformat()
    }


@router.get("/comparison")
def get_period_comparison(
    simulation_date: date = Query(...),
    compare_days: int = Query(30, ge=7, le=90, description="Days to compare"),
    db: Session = Depends(get_db)
):
    """
    Compare current period with previous period.
    
    Shows growth trends for enrollments, demographic updates, and biometric updates.
    """
    from datetime import timedelta
    from sqlalchemy import func
    from app.models.enrollment import Enrollment
    from app.models.demographic_update import DemographicUpdate
    from app.models.biometric_update import BiometricUpdate
    
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    current_start = simulation_date - timedelta(days=compare_days)
    previous_start = current_start - timedelta(days=compare_days)
    previous_end = current_start - timedelta(days=1)
    
    # Current period
    current_enrollment = db.query(func.sum(Enrollment.total)).filter(
        Enrollment.date >= current_start,
        Enrollment.date <= simulation_date
    ).scalar() or 0
    
    current_demo = db.query(func.sum(DemographicUpdate.total)).filter(
        DemographicUpdate.date >= current_start,
        DemographicUpdate.date <= simulation_date
    ).scalar() or 0
    
    current_bio = db.query(func.sum(BiometricUpdate.total)).filter(
        BiometricUpdate.date >= current_start,
        BiometricUpdate.date <= simulation_date
    ).scalar() or 0
    
    # Previous period
    prev_enrollment = db.query(func.sum(Enrollment.total)).filter(
        Enrollment.date >= previous_start,
        Enrollment.date <= previous_end
    ).scalar() or 0
    
    prev_demo = db.query(func.sum(DemographicUpdate.total)).filter(
        DemographicUpdate.date >= previous_start,
        DemographicUpdate.date <= previous_end
    ).scalar() or 0
    
    prev_bio = db.query(func.sum(BiometricUpdate.total)).filter(
        BiometricUpdate.date >= previous_start,
        BiometricUpdate.date <= previous_end
    ).scalar() or 0
    
    def calc_growth(current, previous):
        if previous == 0:
            return 100.0 if current > 0 else 0.0
        return round(((current - previous) / previous) * 100, 2)
    
    return {
        "simulation_date": simulation_date.isoformat(),
        "compare_days": compare_days,
        "current_period": {
            "start": current_start.isoformat(),
            "end": simulation_date.isoformat(),
            "enrollment": int(current_enrollment),
            "demographic": int(current_demo),
            "biometric": int(current_bio)
        },
        "previous_period": {
            "start": previous_start.isoformat(),
            "end": previous_end.isoformat(),
            "enrollment": int(prev_enrollment),
            "demographic": int(prev_demo),
            "biometric": int(prev_bio)
        },
        "growth": {
            "enrollment": calc_growth(int(current_enrollment), int(prev_enrollment)),
            "demographic": calc_growth(int(current_demo), int(prev_demo)),
            "biometric": calc_growth(int(current_bio), int(prev_bio))
        }
    }
