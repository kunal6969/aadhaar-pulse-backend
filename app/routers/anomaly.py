"""
Anomaly Detection API endpoints.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.enrollment import Enrollment
from app.models.demographic_update import DemographicUpdate
from app.models.biometric_update import BiometricUpdate
from app.utils.date_utils import validate_simulation_date
from app.config import settings
from datetime import date, timedelta
from typing import Optional, Literal
import uuid

router = APIRouter()


def detect_anomalies_in_data(
    db: Session,
    model,
    start_date: date,
    end_date: date,
    data_type: str,
    state_filter: Optional[str] = None,
    severity_filter: Optional[str] = None
):
    """
    Detect anomalies using statistical methods.
    
    Anomaly detection based on:
    - Z-score: Values > 3 standard deviations from mean
    - Percentage change: > 200% change from previous period
    """
    # Query data grouped by district and date
    query = db.query(
        model.date,
        model.state,
        model.district,
        func.sum(model.total).label('count')
    ).filter(
        model.date >= start_date,
        model.date <= end_date
    )
    
    if state_filter:
        query = query.filter(model.state == state_filter)
    
    query = query.group_by(model.date, model.state, model.district)
    results = query.all()
    
    if not results:
        return []
    
    # Calculate statistics per district
    district_stats = {}
    for r in results:
        key = f"{r.state}|{r.district}"
        if key not in district_stats:
            district_stats[key] = {
                "state": r.state,
                "district": r.district,
                "values": [],
                "dates": []
            }
        district_stats[key]["values"].append(int(r.count))
        district_stats[key]["dates"].append(r.date)
    
    anomalies = []
    
    for key, data in district_stats.items():
        values = data["values"]
        dates = data["dates"]
        
        if len(values) < 7:
            continue
        
        # Calculate rolling statistics
        mean_val = sum(values) / len(values)
        std_val = (sum((v - mean_val) ** 2 for v in values) / len(values)) ** 0.5
        std_val = max(std_val, 1)  # Avoid division by zero
        
        for i, (value, dt) in enumerate(zip(values, dates)):
            # Calculate z-score
            z_score = (value - mean_val) / std_val
            
            # Determine if anomaly (|z| > 2)
            if abs(z_score) < 2:
                continue
            
            # Calculate deviation ratio
            deviation_ratio = value / max(mean_val, 1)
            
            # Determine severity
            if abs(z_score) > 4 or deviation_ratio > 5:
                severity = "CRITICAL"
            elif abs(z_score) > 3 or deviation_ratio > 3:
                severity = "HIGH"
            elif abs(z_score) > 2.5 or deviation_ratio > 2:
                severity = "MEDIUM"
            else:
                severity = "LOW"
            
            if severity_filter and severity != severity_filter:
                continue
            
            # Generate message
            if z_score > 0:
                direction = "spike"
                message = f"Unusual {direction}: {deviation_ratio:.1f}x baseline ({mean_val:.0f})"
            else:
                direction = "drop"
                message = f"Unusual {direction}: {deviation_ratio:.1f}x of baseline ({mean_val:.0f})"
            
            anomalies.append({
                "id": str(uuid.uuid4())[:8],
                "pincode": None,
                "district": data["district"],
                "state": data["state"],
                "date": dt.isoformat(),
                "data_type": data_type,
                "observed_count": value,
                "baseline_mean": round(mean_val, 1),
                "deviation_ratio": round(deviation_ratio, 2),
                "severity": severity,
                "anomaly_score": round(z_score, 3),
                "message": message
            })
    
    # Sort by date (descending) and severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    anomalies.sort(key=lambda x: (x["date"], severity_order.get(x["severity"], 4)), reverse=True)
    
    return anomalies


@router.get("/detect")
def detect_anomalies(
    simulation_date: date = Query(...),
    time_window: Literal["7d", "30d", "90d", "all"] = Query("30d"),
    severity: Optional[Literal["CRITICAL", "HIGH", "MEDIUM", "LOW"]] = Query(None),
    state: Optional[str] = Query(None),
    data_type: Optional[Literal["enrollment", "demographic", "biometric", "all"]] = Query("all"),
    db: Session = Depends(get_db)
):
    """
    Detect anomalies in enrollment/update data.
    
    - **simulation_date**: The simulated "today"
    - **time_window**: Look back period (7d, 30d, 90d, or all from dataset start)
    - **severity**: Filter by severity level
    - **state**: Filter by state
    - **data_type**: Filter by data type (enrollment, demographic, biometric, all)
    
    Returns list of detected anomalies with explanations.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Calculate start_date based on time_window
    if time_window == "7d":
        start_date = simulation_date - timedelta(days=7)
    elif time_window == "30d":
        start_date = simulation_date - timedelta(days=30)
    elif time_window == "90d":
        start_date = simulation_date - timedelta(days=90)
    else:  # "all"
        start_date = date.fromisoformat(settings.SIMULATION_START_DATE)
    
    all_anomalies = []
    
    # Detect anomalies for each data type
    if data_type in ["enrollment", "all"]:
        enrollment_anomalies = detect_anomalies_in_data(
            db, Enrollment, start_date, simulation_date, "enrollment", state, severity
        )
        all_anomalies.extend(enrollment_anomalies)
    
    if data_type in ["demographic", "all"]:
        demo_anomalies = detect_anomalies_in_data(
            db, DemographicUpdate, start_date, simulation_date, "demographic", state, severity
        )
        all_anomalies.extend(demo_anomalies)
    
    if data_type in ["biometric", "all"]:
        bio_anomalies = detect_anomalies_in_data(
            db, BiometricUpdate, start_date, simulation_date, "biometric", state, severity
        )
        all_anomalies.extend(bio_anomalies)
    
    # Sort combined results
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    all_anomalies.sort(key=lambda x: (x["date"], severity_order.get(x["severity"], 4)), reverse=True)
    
    # Calculate severity breakdown
    severity_breakdown = {
        "CRITICAL": sum(1 for a in all_anomalies if a["severity"] == "CRITICAL"),
        "HIGH": sum(1 for a in all_anomalies if a["severity"] == "HIGH"),
        "MEDIUM": sum(1 for a in all_anomalies if a["severity"] == "MEDIUM"),
        "LOW": sum(1 for a in all_anomalies if a["severity"] == "LOW")
    }
    
    return {
        "anomalies": all_anomalies,
        "total_count": len(all_anomalies),
        "time_window": f"{start_date.isoformat()} to {simulation_date.isoformat()}",
        "severity_breakdown": severity_breakdown,
        "simulation_date": simulation_date.isoformat()
    }


@router.get("/recent")
def get_recent_anomalies(
    simulation_date: date = Query(...),
    limit: int = Query(10, ge=1, le=100),
    db: Session = Depends(get_db)
):
    """
    Get most recent anomalies for alert feed.
    
    Returns last N anomalies detected up to simulation_date.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    start_date = simulation_date - timedelta(days=30)
    
    # Get anomalies from all data types
    all_anomalies = []
    
    for model, data_type in [(Enrollment, "enrollment"), (DemographicUpdate, "demographic"), (BiometricUpdate, "biometric")]:
        anomalies = detect_anomalies_in_data(db, model, start_date, simulation_date, data_type)
        all_anomalies.extend(anomalies)
    
    # Sort by date and severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    all_anomalies.sort(key=lambda x: (x["date"], severity_order.get(x["severity"], 4)), reverse=True)
    
    # Take top N
    recent = all_anomalies[:limit]
    
    # Simplify for alert feed
    alerts = [
        {
            "id": a["id"],
            "district": a["district"],
            "state": a["state"],
            "date": a["date"],
            "severity": a["severity"],
            "message": a["message"],
            "data_type": a["data_type"]
        }
        for a in recent
    ]
    
    return {
        "alerts": alerts,
        "count": len(alerts),
        "as_of_date": simulation_date.isoformat()
    }


@router.get("/summary")
def get_anomaly_summary(
    simulation_date: date = Query(...),
    db: Session = Depends(get_db)
):
    """
    Get summary of anomalies for dashboard widget.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get anomalies for different time windows
    windows = {
        "7d": simulation_date - timedelta(days=7),
        "30d": simulation_date - timedelta(days=30)
    }
    
    summary = {}
    
    for window_name, start_date in windows.items():
        all_anomalies = []
        for model, data_type in [(Enrollment, "enrollment"), (DemographicUpdate, "demographic"), (BiometricUpdate, "biometric")]:
            anomalies = detect_anomalies_in_data(db, model, start_date, simulation_date, data_type)
            all_anomalies.extend(anomalies)
        
        summary[window_name] = {
            "total": len(all_anomalies),
            "critical": sum(1 for a in all_anomalies if a["severity"] == "CRITICAL"),
            "high": sum(1 for a in all_anomalies if a["severity"] == "HIGH")
        }
    
    # Most affected district
    all_30d_anomalies = []
    start_30d = simulation_date - timedelta(days=30)
    for model, data_type in [(Enrollment, "enrollment"), (DemographicUpdate, "demographic"), (BiometricUpdate, "biometric")]:
        anomalies = detect_anomalies_in_data(db, model, start_30d, simulation_date, data_type)
        all_30d_anomalies.extend(anomalies)
    
    district_counts = {}
    for a in all_30d_anomalies:
        key = f"{a['district']}, {a['state']}"
        district_counts[key] = district_counts.get(key, 0) + 1
    
    most_affected = sorted(district_counts.items(), key=lambda x: x[1], reverse=True)[:5]
    
    return {
        "simulation_date": simulation_date.isoformat(),
        "summary_7d": summary.get("7d", {}),
        "summary_30d": summary.get("30d", {}),
        "most_affected_districts": [
            {"location": loc, "anomaly_count": count}
            for loc, count in most_affected
        ]
    }
