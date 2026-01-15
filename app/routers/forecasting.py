"""
Forecasting API endpoints for ML predictions.
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from app.database import get_db
from app.services.biometric_service import BiometricService
from app.utils.date_utils import validate_simulation_date
from datetime import date, timedelta
from typing import Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np

router = APIRouter()


class ForecastRequest(BaseModel):
    """Request body for forecast endpoint."""
    district: str
    forecast_from: date
    horizon_days: int = 180


@router.post("/mbu")
def forecast_mbu_demand(
    request: ForecastRequest,
    db: Session = Depends(get_db)
):
    """
    Forecast Mandatory Biometric Update (MBU) demand.
    
    Uses historical biometric update data to predict future demand.
    
    - **district**: District name
    - **forecast_from**: Start forecast from this date (simulation_date)
    - **horizon_days**: How many days ahead to predict (default: 180)
    
    Returns historical data up to forecast_from and predictions for next horizon_days.
    """
    is_valid, error_msg = validate_simulation_date(request.forecast_from)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    service = BiometricService(db)
    
    # Get historical data
    historical_df = service.get_historical_data_for_forecast(
        district=request.district,
        end_date=request.forecast_from,
        days_back=180
    )
    
    if len(historical_df) < 14:
        raise HTTPException(
            status_code=400,
            detail=f"Insufficient historical data for district '{request.district}' (need at least 14 days, got {len(historical_df)})"
        )
    
    # Simple forecasting using trend and seasonality
    # For production, use Prophet or other ML models
    historical = historical_df.to_dict('records')
    
    # Calculate simple trend
    values = historical_df['total'].values
    mean_value = values.mean()
    trend = (values[-7:].mean() - values[:7].mean()) / max(len(values) - 7, 1)
    
    # Generate forecast
    forecast = []
    for i in range(request.horizon_days):
        forecast_date = request.forecast_from + timedelta(days=i+1)
        
        # Simple prediction: mean + trend * days + some seasonality
        predicted = mean_value + (trend * i)
        
        # Add weekly seasonality (weekends have less activity)
        day_of_week = forecast_date.weekday()
        if day_of_week in [5, 6]:  # Weekend
            predicted *= 0.7
        
        # Ensure non-negative
        predicted = max(0, predicted)
        
        # Confidence bounds
        std = values.std() if len(values) > 1 else mean_value * 0.2
        lower = max(0, predicted - 1.96 * std)
        upper = predicted + 1.96 * std
        
        forecast.append({
            "date": forecast_date.isoformat(),
            "predicted": round(predicted, 0),
            "lower_bound": round(lower, 0),
            "upper_bound": round(upper, 0)
        })
    
    # Calculate risk assessment
    # Get enrollment data for this district
    from sqlalchemy import func
    from app.models.enrollment import Enrollment
    
    enrollment_5_17 = db.query(
        func.sum(Enrollment.age_5_17)
    ).filter(
        Enrollment.district == request.district,
        Enrollment.date <= request.forecast_from
    ).scalar() or 0
    
    # Estimate expected MBU (15% of 5-17 will turn 15 in 6 months)
    expected_mbu = int(enrollment_5_17 * 0.15)
    
    # Recent updates
    recent_updates = sum(values[-30:]) if len(values) >= 30 else sum(values)
    
    # Calculate risk
    monthly_expected = expected_mbu / 6 if expected_mbu > 0 else 1
    completion_ratio = recent_updates / monthly_expected if monthly_expected > 0 else 1
    
    if completion_ratio < 0.3:
        risk_level = "HIGH"
        risk_message = "Very low biometric update rate - surge imminent"
    elif completion_ratio < 0.7:
        risk_level = "MEDIUM"
        risk_message = "Moderate preparation needed"
    else:
        risk_level = "LOW"
        risk_message = "Updates tracking expected demand"
    
    return {
        "district": request.district,
        "forecast_from": request.forecast_from.isoformat(),
        "horizon_days": request.horizon_days,
        "historical": [
            {"date": h["date"].isoformat() if hasattr(h["date"], 'isoformat') else h["date"], "actual": h["total"]}
            for h in historical
        ],
        "forecast": forecast,
        "risk_assessment": {
            "risk_level": risk_level,
            "risk_score": round(completion_ratio, 2),
            "message": risk_message,
            "expected_mbu_6months": expected_mbu,
            "actual_updates_30days": int(recent_updates),
            "current_5_17_population": int(enrollment_5_17)
        },
        "model_info": {
            "type": "Simple Trend + Seasonality",
            "training_days": len(historical_df),
            "note": "For production, integrate Prophet or similar ML model"
        }
    }


@router.get("/trends")
def get_forecast_trends(
    simulation_date: date = Query(...),
    district: str = Query(...),
    data_type: str = Query("biometric", regex="^(enrollment|demographic|biometric)$"),
    horizon_days: int = Query(90, ge=7, le=365),
    db: Session = Depends(get_db)
):
    """
    Get trend forecast for a specific district and data type.
    
    Simplified endpoint that returns historical and forecast data.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get historical data based on data_type
    from sqlalchemy import func
    from app.models.enrollment import Enrollment
    from app.models.demographic_update import DemographicUpdate
    from app.models.biometric_update import BiometricUpdate
    
    start_date = simulation_date - timedelta(days=180)
    
    if data_type == "enrollment":
        Model = Enrollment
    elif data_type == "demographic":
        Model = DemographicUpdate
    else:
        Model = BiometricUpdate
    
    results = db.query(
        Model.date,
        func.sum(Model.total).label('total')
    ).filter(
        Model.district == district,
        Model.date >= start_date,
        Model.date <= simulation_date
    ).group_by(Model.date).order_by(Model.date).all()
    
    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No {data_type} data found for district '{district}'"
        )
    
    historical = [{"date": r.date.isoformat(), "value": int(r.total)} for r in results]
    
    # Simple forecast
    values = [r.total for r in results]
    mean_value = sum(values) / len(values)
    
    forecast = []
    for i in range(horizon_days):
        forecast_date = simulation_date + timedelta(days=i+1)
        # Simple moving average forecast
        predicted = mean_value * (0.95 + 0.1 * (1 - i / horizon_days))
        
        forecast.append({
            "date": forecast_date.isoformat(),
            "predicted": round(predicted, 0)
        })
    
    return {
        "district": district,
        "data_type": data_type,
        "simulation_date": simulation_date.isoformat(),
        "historical": historical,
        "forecast": forecast,
        "summary": {
            "historical_mean": round(mean_value, 0),
            "historical_days": len(historical),
            "forecast_days": horizon_days
        }
    }
