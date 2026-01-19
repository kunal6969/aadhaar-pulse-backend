"""
Biometric Update API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.biometric_service import BiometricService
from app.utils.date_utils import validate_simulation_date
from datetime import date, timedelta
from typing import Optional

router = APIRouter()


@router.get("/trends")
def get_biometric_trends(
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
    Get biometric update trends filtered by simulation date and geography.
    
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
    
    service = BiometricService(db)
    
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
def get_biometric_summary(
    simulation_date: date = Query(..., description="Simulated current date"),
    state: Optional[str] = Query(None, description="Filter by state"),
    district: Optional[str] = Query(None, description="Filter by district"),
    db: Session = Depends(get_db)
):
    """
    Get KPI summary for biometric updates.
    
    Includes MBU (Mandatory Biometric Update) specific metrics.
    Can be filtered by state/district.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = BiometricService(db)
    summary = service.get_summary(simulation_date, state=state, district=district)
    
    return summary


@router.get("/by-district")
def get_biometric_by_district(
    simulation_date: date = Query(...),
    state: Optional[str] = Query(None),
    days_back: int = Query(30, ge=1, le=365),
    limit: int = Query(50, ge=1, le=500),
    db: Session = Depends(get_db)
):
    """
    Get biometric data aggregated by district with MBU risk assessment.
    
    Each district includes a risk level (LOW, MEDIUM, HIGH) based on
    the ratio of biometric updates to expected MBU demand.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = BiometricService(db)
    districts = service.get_by_district_with_risk(
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


@router.get("/mbu-risk")
def get_mbu_risk_overview(
    simulation_date: date = Query(...),
    state: Optional[str] = Query(None),
    db: Session = Depends(get_db)
):
    """
    Get MBU risk overview across all districts.
    
    Returns count of districts in each risk category.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = BiometricService(db)
    districts = service.get_by_district_with_risk(
        simulation_date=simulation_date,
        state=state,
        limit=1000  # Get all districts
    )
    
    # Count by risk level
    risk_counts = {"LOW": 0, "MEDIUM": 0, "HIGH": 0}
    high_risk_districts = []
    
    for d in districts:
        risk_level = d.get("mbu_risk_level", "LOW")
        risk_counts[risk_level] = risk_counts.get(risk_level, 0) + 1
        
        if risk_level == "HIGH":
            high_risk_districts.append({
                "district": d["district"],
                "state": d["state"],
                "bio_age_5_17": d["bio_age_5_17"]
            })
    
    return {
        "simulation_date": simulation_date.isoformat(),
        "total_districts": len(districts),
        "risk_breakdown": risk_counts,
        "high_risk_districts": high_risk_districts[:10],  # Top 10
        "alert_message": f"{risk_counts['HIGH']} districts at high MBU risk" if risk_counts['HIGH'] > 0 else "No high-risk districts"
    }
