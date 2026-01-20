"""
ML Analytics API Router.

Exposes all 7 ML models through REST API endpoints:
1. Hierarchical Time Series Forecasting
2. Queueing/Capacity Planning
3. Underserved Scoring
4. Forensic Fraud Detection
5. District Clustering/Segmentation
6. EWMA Hotspot Detection
7. Cohort Transition (5-Year MBU)
"""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.database import get_db
from app.models.enrollment import Enrollment
from app.models.demographic_update import DemographicUpdate
from app.models.biometric_update import BiometricUpdate
from app.utils.date_utils import validate_simulation_date
from datetime import date, timedelta
from typing import Optional, List
from pydantic import BaseModel, Field
import pandas as pd
import numpy as np

from app.ml_models import (
    HierarchicalForecaster,
    CapacityPlanningModel,
    UnderservedScoringModel,
    ForensicFraudDetector,
    DistrictClusteringModel,
    EWMAHotspotDetector,
    CohortTransitionModel,
    get_model,
    get_model_status,
    is_model_trained
)

router = APIRouter()


# ==================== REQUEST MODELS ====================

class DemandForecastRequest(BaseModel):
    """Request for demand forecasting."""
    state: Optional[str] = None
    district: Optional[str] = None
    simulation_date: date
    horizon_days: int = Field(default=90, ge=7, le=365)
    data_type: str = Field(default="enrollment", description="enrollment, demographic, or biometric")


class CapacityPlanRequest(BaseModel):
    """Request for capacity planning."""
    state: Optional[str] = None
    simulation_date: date
    service_rate_per_operator: float = Field(default=40, description="Requests per operator per day")


class UnderservedRequest(BaseModel):
    """Request for underserved scoring."""
    state: Optional[str] = None
    simulation_date: date
    num_mobile_units: int = Field(default=5, ge=1, le=50)


class FraudAnalysisRequest(BaseModel):
    """Request for fraud analysis."""
    state: Optional[str] = None
    simulation_date: date
    min_risk_level: str = Field(default="HIGH", description="CLEAN, LOW, MEDIUM, HIGH, CRITICAL")


class ClusteringRequest(BaseModel):
    """Request for district clustering."""
    state: Optional[str] = None
    simulation_date: date
    n_clusters: int = Field(default=5, ge=3, le=10)


class HotspotRequest(BaseModel):
    """Request for hotspot detection."""
    state: Optional[str] = None
    simulation_date: date
    top_n: int = Field(default=20, ge=5, le=100)


class CohortPredictionRequest(BaseModel):
    """Request for 5-year MBU prediction."""
    state: Optional[str] = None
    district: Optional[str] = None
    simulation_date: date


# ==================== HELPER FUNCTIONS ====================

def get_district_stats(db: Session, simulation_date: date, state: Optional[str] = None):
    """Get aggregated stats by district for ML models."""
    start_date = simulation_date - timedelta(days=180)
    
    # Enrollment stats
    enroll_query = db.query(
        Enrollment.state,
        Enrollment.district,
        func.sum(Enrollment.total).label('total_enrollments'),
        func.sum(Enrollment.age_0_5).label('age_0_5'),
        func.sum(Enrollment.age_5_17).label('age_5_17'),
        func.avg(Enrollment.total).label('avg_daily')
    ).filter(
        Enrollment.date >= start_date,
        Enrollment.date <= simulation_date
    )
    
    if state:
        enroll_query = enroll_query.filter(Enrollment.state == state)
    
    enroll_query = enroll_query.group_by(Enrollment.state, Enrollment.district)
    
    return enroll_query.all()


def get_daily_values(db: Session, model, district: str, simulation_date: date, days: int = 90):
    """Get daily values for a district."""
    start_date = simulation_date - timedelta(days=days)
    
    results = db.query(
        model.date,
        func.sum(model.total).label('total')
    ).filter(
        model.district == district,
        model.date >= start_date,
        model.date <= simulation_date
    ).group_by(model.date).order_by(model.date).all()
    
    return [int(r.total) for r in results]


# ==================== FORECASTING ENDPOINTS ====================

@router.post("/forecasting/demand")
def forecast_demand(request: DemandForecastRequest, db: Session = Depends(get_db)):
    """
    Generate demand forecasts using Hierarchical Time Series model.
    
    Predicts future enrollment/demographic/biometric demand with seasonality.
    """
    is_valid, error_msg = validate_simulation_date(request.simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Select model based on data type
    if request.data_type == "enrollment":
        model = Enrollment
    elif request.data_type == "demographic":
        model = DemographicUpdate
    else:
        model = BiometricUpdate
    
    # Get historical data
    start_date = request.simulation_date - timedelta(days=180)
    
    query = db.query(
        model.date,
        func.sum(model.total).label('total')
    ).filter(
        model.date >= start_date,
        model.date <= request.simulation_date
    )
    
    if request.state:
        query = query.filter(model.state == request.state)
    if request.district:
        query = query.filter(model.district == request.district)
    
    query = query.group_by(model.date).order_by(model.date)
    results = query.all()
    
    if len(results) < 14:
        raise HTTPException(status_code=400, detail="Insufficient historical data (need at least 14 days)")
    
    # Prepare DataFrame
    df = pd.DataFrame([{"date": r.date, "total": int(r.total)} for r in results])
    
    # Fit forecaster
    forecaster = HierarchicalForecaster()
    forecaster.fit(df)
    
    # Generate predictions
    predictions = forecaster.predict(request.simulation_date, request.horizon_days)
    
    # Get seasonality calendar
    seasonality = forecaster.get_seasonality_calendar()
    
    return {
        "location": {
            "state": request.state or "All India",
            "district": request.district
        },
        "data_type": request.data_type,
        "forecast_from": request.simulation_date.isoformat(),
        "horizon_days": request.horizon_days,
        "historical_summary": {
            "days": len(df),
            "mean": round(df['total'].mean(), 0),
            "std": round(df['total'].std(), 0),
            "trend": "increasing" if forecaster._model_params['trend_slope'] > 0 else "decreasing"
        },
        "forecasts": [p.to_dict() for p in predictions],
        "seasonality_calendar": seasonality,
        "model_info": forecaster.get_model_info()
    }


@router.get("/forecasting/seasonality")
def get_seasonality_calendar(
    simulation_date: date = Query(...),
    state: Optional[str] = None,
    data_type: str = Query(default="enrollment"),
    db: Session = Depends(get_db)
):
    """
    Get seasonality calendar showing expected patterns.
    
    Useful for planning staffing and operations.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    if data_type == "enrollment":
        model = Enrollment
    elif data_type == "demographic":
        model = DemographicUpdate
    else:
        model = BiometricUpdate
    
    start_date = simulation_date - timedelta(days=365)
    
    query = db.query(
        model.date,
        func.sum(model.total).label('total')
    ).filter(
        model.date >= start_date,
        model.date <= simulation_date
    )
    
    if state:
        query = query.filter(model.state == state)
    
    query = query.group_by(model.date).order_by(model.date)
    results = query.all()
    
    if len(results) < 30:
        raise HTTPException(status_code=400, detail="Need at least 30 days for seasonality analysis")
    
    df = pd.DataFrame([{"date": r.date, "total": int(r.total)} for r in results])
    
    forecaster = HierarchicalForecaster()
    forecaster.fit(df)
    
    return {
        "state": state or "All India",
        "data_type": data_type,
        "analysis_period": f"{start_date} to {simulation_date}",
        "seasonality": forecaster.get_seasonality_calendar()
    }


# ==================== CAPACITY PLANNING ENDPOINTS ====================

@router.post("/capacity/operators")
def calculate_operator_requirements(request: CapacityPlanRequest, db: Session = Depends(get_db)):
    """
    Calculate operator requirements using Queueing Theory model.
    
    Uses Little's Law for capacity estimation.
    """
    is_valid, error_msg = validate_simulation_date(request.simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get forecasted demand per district
    start_date = request.simulation_date - timedelta(days=30)
    
    query = db.query(
        Enrollment.state,
        Enrollment.district,
        func.avg(Enrollment.total).label('avg_daily')
    ).filter(
        Enrollment.date >= start_date,
        Enrollment.date <= request.simulation_date
    )
    
    if request.state:
        query = query.filter(Enrollment.state == request.state)
    
    query = query.group_by(Enrollment.state, Enrollment.district)
    results = query.all()
    
    if not results:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Calculate requirements for each district
    model = CapacityPlanningModel(service_rate_per_operator=request.service_rate_per_operator)
    
    requirements = []
    for r in results:
        req = model.calculate_requirements(
            district=r.district,
            state=r.state,
            forecasted_demand=float(r.avg_daily)
        )
        requirements.append(req.to_dict())
    
    # Sort by operators required
    requirements.sort(key=lambda x: x['operators_required'], reverse=True)
    
    # Summary stats
    total_operators = sum(r['operators_required'] for r in requirements)
    high_risk = [r for r in requirements if r['sla_risk_level'] in ['HIGH', 'CRITICAL']]
    
    return {
        "state": request.state or "All India",
        "simulation_date": request.simulation_date.isoformat(),
        "summary": {
            "total_districts": len(requirements),
            "total_operators_needed": total_operators,
            "high_risk_districts": len(high_risk),
            "avg_utilization": round(np.mean([r['utilization_rate'] for r in requirements]), 2)
        },
        "district_requirements": requirements[:50],  # Top 50
        "model_info": model.get_model_info()
    }


@router.get("/capacity/yearly-plan/{district}")
def get_yearly_capacity_plan(
    district: str,
    simulation_date: date = Query(...),
    db: Session = Depends(get_db)
):
    """
    Generate yearly capacity plan for a district.
    
    Shows monthly operator requirements and recommendations.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get monthly averages
    start_date = simulation_date - timedelta(days=365)
    
    results = db.query(
        func.extract('month', Enrollment.date).label('month'),
        Enrollment.state,
        func.avg(Enrollment.total).label('avg_daily')
    ).filter(
        Enrollment.district == district,
        Enrollment.date >= start_date,
        Enrollment.date <= simulation_date
    ).group_by(
        func.extract('month', Enrollment.date),
        Enrollment.state
    ).all()
    
    if not results:
        raise HTTPException(status_code=404, detail=f"No data for district: {district}")
    
    # Build monthly demands
    monthly_demands = [0.0] * 12
    state = results[0].state
    
    for r in results:
        month_idx = int(r.month) - 1
        monthly_demands[month_idx] = float(r.avg_daily)
    
    # Fill missing months with average
    avg_demand = np.mean([d for d in monthly_demands if d > 0]) or 50
    monthly_demands = [d if d > 0 else avg_demand for d in monthly_demands]
    
    model = CapacityPlanningModel()
    plan = model.generate_yearly_plan(district, state, monthly_demands)
    
    return plan.to_dict()


# ==================== UNDERSERVED SCORING ENDPOINTS ====================

@router.post("/underserved/scores")
def calculate_underserved_scores(request: UnderservedRequest, db: Session = Depends(get_db)):
    """
    Calculate underserved scores for districts.
    
    Higher score = more underserved = needs more attention.
    """
    is_valid, error_msg = validate_simulation_date(request.simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    stats = get_district_stats(db, request.simulation_date, request.state)
    
    if not stats:
        raise HTTPException(status_code=404, detail="No data found")
    
    model = UnderservedScoringModel()
    scores = []
    
    for s in stats:
        # Estimate population from enrollments (assuming ~80% coverage)
        estimated_pop = int(s.total_enrollments / 0.8) if s.total_enrollments else 100000
        
        score = model.calculate_score(
            district=s.district,
            state=s.state,
            total_enrollments=int(s.total_enrollments or 0),
            estimated_population=estimated_pop,
            enrollment_growth_rate=5.0,  # Default assumption
            num_centers=max(1, int(s.total_enrollments / 50000)),  # Estimate
            child_population_0_5=int(estimated_pop * 0.1),
            child_enrollments_0_5=int(s.age_0_5 or 0)
        )
        scores.append(score.to_dict())
    
    # Sort by overall score
    scores.sort(key=lambda x: x['overall_score'], reverse=True)
    
    # Get intervention summary
    interventions = {}
    for s in scores:
        intervention = s['recommended_intervention']
        if intervention not in interventions:
            interventions[intervention] = 0
        interventions[intervention] += 1
    
    return {
        "state": request.state or "All India",
        "simulation_date": request.simulation_date.isoformat(),
        "summary": {
            "total_districts": len(scores),
            "action_required": len([s for s in scores if s['action_required']]),
            "interventions_breakdown": interventions
        },
        "district_scores": scores[:50],
        "model_info": model.get_model_info()
    }


@router.post("/underserved/mobile-placement")
def optimize_mobile_unit_placement(request: UnderservedRequest, db: Session = Depends(get_db)):
    """
    Optimize mobile enrollment unit placement.
    
    Returns recommended districts and days per month for each unit.
    """
    is_valid, error_msg = validate_simulation_date(request.simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    stats = get_district_stats(db, request.simulation_date, request.state)
    
    if not stats:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Prepare district data
    districts_data = []
    for s in stats:
        estimated_pop = int(s.total_enrollments / 0.8) if s.total_enrollments else 100000
        districts_data.append({
            'district': s.district,
            'state': s.state,
            'total_enrollments': int(s.total_enrollments or 0),
            'population': estimated_pop,
            'growth_rate': 5.0,
            'num_centers': max(1, int(s.total_enrollments / 50000)),
            'child_population': int(estimated_pop * 0.1),
            'child_enrollments': int(s.age_0_5 or 0)
        })
    
    model = UnderservedScoringModel()
    placements = model.optimize_mobile_unit_placement(
        districts_data, 
        num_units=request.num_mobile_units
    )
    
    return {
        "state": request.state or "All India",
        "num_mobile_units": request.num_mobile_units,
        "placements": [p.to_dict() for p in placements],
        "total_expected_enrollments": sum(p.expected_enrollments for p in placements),
        "model_info": model.get_model_info()
    }


# ==================== FRAUD DETECTION ENDPOINTS ====================

@router.post("/fraud/analysis")
def analyze_fraud_indicators(request: FraudAnalysisRequest, db: Session = Depends(get_db)):
    """
    Analyze districts for fraud indicators using digit analysis.
    
    Uses Benford's Law and digit distribution tests.
    """
    is_valid, error_msg = validate_simulation_date(request.simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    start_date = request.simulation_date - timedelta(days=90)
    
    # Get enrollment values per district
    query = db.query(
        Enrollment.state,
        Enrollment.district,
        Enrollment.total
    ).filter(
        Enrollment.date >= start_date,
        Enrollment.date <= request.simulation_date
    )
    
    if request.state:
        query = query.filter(Enrollment.state == request.state)
    
    results = query.all()
    
    if not results:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Group values by district
    district_values = {}
    for r in results:
        key = f"{r.state}|{r.district}"
        if key not in district_values:
            district_values[key] = {
                'district': r.district,
                'state': r.state,
                'values': []
            }
        district_values[key]['values'].append(int(r.total))
    
    # Analyze each district
    detector = ForensicFraudDetector()
    scores = detector.batch_analyze(list(district_values.values()))
    
    # Get audit list
    from app.ml_models.fraud_detection import FraudRiskLevel
    risk_levels = {"CLEAN": FraudRiskLevel.CLEAN, "LOW": FraudRiskLevel.LOW, 
                   "MEDIUM": FraudRiskLevel.MEDIUM, "HIGH": FraudRiskLevel.HIGH,
                   "CRITICAL": FraudRiskLevel.CRITICAL}
    
    min_level = risk_levels.get(request.min_risk_level, FraudRiskLevel.HIGH)
    audit_list = detector.get_districts_for_audit(scores, min_level)
    
    # Risk summary
    risk_summary = {}
    for level in FraudRiskLevel:
        risk_summary[level.value] = len([s for s in scores if s.risk_level == level])
    
    return {
        "state": request.state or "All India",
        "analysis_period": f"{start_date} to {request.simulation_date}",
        "summary": {
            "total_districts_analyzed": len(scores),
            "risk_breakdown": risk_summary,
            "districts_requiring_audit": len(audit_list)
        },
        "audit_list": audit_list,
        "all_scores": [s.to_dict() for s in scores[:30]],  # Top 30 by fraud score
        "model_info": detector.get_model_info()
    }


# ==================== CLUSTERING ENDPOINTS ====================

@router.post("/clustering/segments")
def segment_districts(request: ClusteringRequest, db: Session = Depends(get_db)):
    """
    Segment districts into behavior-based groups.
    
    Identifies stable users, high churn, hotspots, and suspicious patterns.
    """
    is_valid, error_msg = validate_simulation_date(request.simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    start_date = request.simulation_date - timedelta(days=90)
    
    # Get stats for each data type
    # Enrollment rates
    enroll_stats = db.query(
        Enrollment.state,
        Enrollment.district,
        func.avg(Enrollment.total).label('enrollment_rate'),
        func.stddev(Enrollment.total).label('enrollment_std')
    ).filter(
        Enrollment.date >= start_date,
        Enrollment.date <= request.simulation_date
    )
    if request.state:
        enroll_stats = enroll_stats.filter(Enrollment.state == request.state)
    enroll_stats = enroll_stats.group_by(Enrollment.state, Enrollment.district).all()
    
    # Demographic rates
    demo_stats = db.query(
        DemographicUpdate.district,
        func.avg(DemographicUpdate.total).label('demo_rate')
    ).filter(
        DemographicUpdate.date >= start_date,
        DemographicUpdate.date <= request.simulation_date
    )
    if request.state:
        demo_stats = demo_stats.filter(DemographicUpdate.state == request.state)
    demo_stats = demo_stats.group_by(DemographicUpdate.district).all()
    demo_dict = {d.district: float(d.demo_rate or 0) for d in demo_stats}
    
    # Biometric rates
    bio_stats = db.query(
        BiometricUpdate.district,
        func.avg(BiometricUpdate.total).label('bio_rate')
    ).filter(
        BiometricUpdate.date >= start_date,
        BiometricUpdate.date <= request.simulation_date
    )
    if request.state:
        bio_stats = bio_stats.filter(BiometricUpdate.state == request.state)
    bio_stats = bio_stats.group_by(BiometricUpdate.district).all()
    bio_dict = {b.district: float(b.bio_rate or 0) for b in bio_stats}
    
    if not enroll_stats:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Build feature DataFrame
    max_enroll = max(float(e.enrollment_rate or 1) for e in enroll_stats)
    max_demo = max(demo_dict.values()) if demo_dict else 1
    max_bio = max(bio_dict.values()) if bio_dict else 1
    
    data = []
    for e in enroll_stats:
        enroll_rate = float(e.enrollment_rate or 0) / max_enroll
        demo_rate = demo_dict.get(e.district, 0) / max_demo if max_demo > 0 else 0
        bio_rate = bio_dict.get(e.district, 0) / max_bio if max_bio > 0 else 0
        volatility = float(e.enrollment_std or 0) / float(e.enrollment_rate or 1) if e.enrollment_rate else 0
        
        data.append({
            'district': e.district,
            'state': e.state,
            'enrollment_rate': enroll_rate,
            'demographic_update_rate': demo_rate,
            'biometric_update_rate': bio_rate,
            'growth_trend': 0.5,  # Placeholder
            'volatility': min(1, volatility)
        })
    
    df = pd.DataFrame(data)
    
    # Run clustering
    model = DistrictClusteringModel(n_clusters=request.n_clusters)
    results = model.fit_predict(df)
    
    # Get summaries
    segment_summary = model.get_segment_summary(results)
    infiltration_zones = model.get_infiltration_zones(results)
    interventions = model.get_intervention_recommendations(results)
    
    return {
        "state": request.state or "All India",
        "n_clusters": request.n_clusters,
        "segment_summary": segment_summary,
        "infiltration_zones": infiltration_zones,
        "intervention_recommendations": interventions,
        "district_assignments": [r.to_dict() for r in results[:50]],
        "model_info": model.get_model_info()
    }


# ==================== HOTSPOT DETECTION ENDPOINTS ====================

@router.post("/hotspots/biometric-load")
def detect_biometric_hotspots(request: HotspotRequest, db: Session = Depends(get_db)):
    """
    Detect biometric infrastructure load hotspots using EWMA.
    
    Identifies districts with unusual demand spikes.
    """
    is_valid, error_msg = validate_simulation_date(request.simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get districts
    districts_query = db.query(
        BiometricUpdate.state,
        BiometricUpdate.district
    ).distinct()
    
    if request.state:
        districts_query = districts_query.filter(BiometricUpdate.state == request.state)
    
    districts = districts_query.all()
    
    if not districts:
        raise HTTPException(status_code=404, detail="No data found")
    
    detector = EWMAHotspotDetector()
    hotspots = []
    
    for d in districts[:100]:  # Limit to 100 districts
        daily_values = get_daily_values(
            db, BiometricUpdate, d.district, request.simulation_date, 90
        )
        
        if len(daily_values) >= 7:
            result = detector.detect_hotspot(
                district=d.district,
                state=d.state,
                daily_values=daily_values
            )
            hotspots.append(result.to_dict())
    
    # Sort by deviation
    hotspots.sort(key=lambda x: x['deviation_sigma'], reverse=True)
    
    # Level summary
    level_summary = {}
    for h in hotspots:
        level = h['hotspot_level']
        if level not in level_summary:
            level_summary[level] = 0
        level_summary[level] += 1
    
    return {
        "state": request.state or "All India",
        "simulation_date": request.simulation_date.isoformat(),
        "summary": {
            "total_districts": len(hotspots),
            "level_breakdown": level_summary,
            "critical_hotspots": len([h for h in hotspots if h['is_hotspot']])
        },
        "hotspots": hotspots[:request.top_n],
        "model_info": detector.get_model_info()
    }


@router.post("/hotspots/infrastructure-demand")
def map_infrastructure_demand(request: HotspotRequest, db: Session = Depends(get_db)):
    """
    Map biometric infrastructure demand across districts.
    
    Returns top load districts with equipment recommendations.
    """
    is_valid, error_msg = validate_simulation_date(request.simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get districts with daily values
    districts_query = db.query(
        BiometricUpdate.state,
        BiometricUpdate.district
    ).distinct()
    
    if request.state:
        districts_query = districts_query.filter(BiometricUpdate.state == request.state)
    
    districts = districts_query.all()
    
    districts_data = []
    for d in districts[:100]:
        daily_values = get_daily_values(
            db, BiometricUpdate, d.district, request.simulation_date, 90
        )
        
        if daily_values:
            districts_data.append({
                'district': d.district,
                'state': d.state,
                'daily_values': daily_values,
                'current_devices': 2  # Default assumption
            })
    
    detector = EWMAHotspotDetector()
    demands = detector.map_infrastructure_demand(districts_data)
    top_load = detector.get_top_load_districts(demands, request.top_n)
    risk_zones = detector.get_device_failure_risk_zones(demands)
    
    return {
        "state": request.state or "All India",
        "top_load_districts": top_load,
        "device_failure_risk_zones": {
            zone: len(districts) for zone, districts in risk_zones.items()
        },
        "risk_zone_details": risk_zones,
        "total_additional_devices_needed": sum(d.additional_devices_needed for d in demands),
        "model_info": detector.get_model_info()
    }


# ==================== COHORT MODEL ENDPOINTS ====================

@router.post("/cohort/mbu-prediction")
def predict_5_year_mbu(request: CohortPredictionRequest, db: Session = Depends(get_db)):
    """
    Predict 5-year MBU demand from child enrollments.
    
    Uses cohort transition model to forecast biometric update demand.
    """
    is_valid, error_msg = validate_simulation_date(request.simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get child enrollment data by year
    query = db.query(
        func.extract('year', Enrollment.date).label('year'),
        Enrollment.state,
        Enrollment.district,
        func.sum(Enrollment.age_0_5).label('age_0_5'),
        func.sum(Enrollment.age_5_17).label('age_5_17')
    )
    
    if request.state:
        query = query.filter(Enrollment.state == request.state)
    if request.district:
        query = query.filter(Enrollment.district == request.district)
    
    query = query.group_by(
        func.extract('year', Enrollment.date),
        Enrollment.state,
        Enrollment.district
    )
    
    results = query.all()
    
    if not results:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Build cohort data structure
    districts = {}
    for r in results:
        key = f"{r.state}|{r.district}"
        if key not in districts:
            districts[key] = {
                'state': r.state,
                'district': r.district,
                'enrollments_by_year': {}
            }
        
        year = int(r.year)
        districts[key]['enrollments_by_year'][year] = {
            "0-5": int(r.age_0_5 or 0),
            "5-10": int(r.age_5_17 or 0) // 2,  # Approximate split
            "10-15": int(r.age_5_17 or 0) // 2
        }
    
    model = CohortTransitionModel()
    predictions = []
    
    current_year = request.simulation_date.year
    
    for key, data in list(districts.items())[:50]:  # Limit to 50 districts
        prediction = model.predict_mbu_demand(
            district=data['district'],
            state=data['state'],
            child_enrollments_by_year=data['enrollments_by_year'],
            current_year=current_year
        )
        predictions.append(prediction)
    
    # Get priority districts
    priority_districts = model.identify_priority_districts(predictions, top_n=20)
    
    # Aggregate yearly demand
    yearly_totals = {year: 0 for year in range(current_year, current_year + 6)}
    for pred in predictions:
        for yp in pred.yearly_predictions:
            yearly_totals[yp['year']] += yp['expected_mbu_demand']
    
    return {
        "state": request.state or "All India",
        "district": request.district,
        "current_year": current_year,
        "aggregate_5_year_demand": yearly_totals,
        "priority_districts": priority_districts,
        "detailed_predictions": [p.to_dict() for p in predictions[:20]],
        "model_info": model.get_model_info()
    }


@router.get("/cohort/equipment-plan")
def get_equipment_plan(
    state: Optional[str] = None,
    simulation_date: date = Query(...),
    db: Session = Depends(get_db)
):
    """
    Generate equipment planning based on 5-year MBU predictions.
    
    Returns investment schedule and ROI assessment.
    """
    is_valid, error_msg = validate_simulation_date(simulation_date)
    if not is_valid:
        raise HTTPException(status_code=400, detail=error_msg)
    
    # Get child enrollment data
    query = db.query(
        func.extract('year', Enrollment.date).label('year'),
        Enrollment.state,
        Enrollment.district,
        func.sum(Enrollment.age_0_5).label('age_0_5'),
        func.sum(Enrollment.age_5_17).label('age_5_17')
    )
    
    if state:
        query = query.filter(Enrollment.state == state)
    
    query = query.group_by(
        func.extract('year', Enrollment.date),
        Enrollment.state,
        Enrollment.district
    )
    
    results = query.all()
    
    if not results:
        raise HTTPException(status_code=404, detail="No data found")
    
    # Build predictions
    districts = {}
    for r in results:
        key = f"{r.state}|{r.district}"
        if key not in districts:
            districts[key] = {
                'state': r.state,
                'district': r.district,
                'enrollments_by_year': {}
            }
        
        year = int(r.year)
        districts[key]['enrollments_by_year'][year] = {
            "0-5": int(r.age_0_5 or 0),
            "5-10": int(r.age_5_17 or 0) // 2,
            "10-15": int(r.age_5_17 or 0) // 2
        }
    
    model = CohortTransitionModel()
    predictions = []
    current_devices = {}
    
    current_year = simulation_date.year
    
    for key, data in list(districts.items())[:50]:
        prediction = model.predict_mbu_demand(
            district=data['district'],
            state=data['state'],
            child_enrollments_by_year=data['enrollments_by_year'],
            current_year=current_year
        )
        predictions.append(prediction)
        current_devices[data['district']] = 2  # Default
    
    equipment_plans = model.generate_equipment_plan(predictions, current_devices)
    
    # Summary
    total_investment = sum(p.total_5_year_investment for p in equipment_plans)
    total_devices = sum(p.additional_devices_needed for p in equipment_plans)
    
    return {
        "state": state or "All India",
        "summary": {
            "total_districts": len(equipment_plans),
            "total_additional_devices": total_devices,
            "total_investment_lakhs": round(total_investment / 100000, 2),
            "avg_devices_per_district": round(total_devices / len(equipment_plans), 1) if equipment_plans else 0
        },
        "equipment_plans": [p.to_dict() for p in equipment_plans[:30]],
        "model_info": model.get_model_info()
    }


# ==================== MODEL INFO ENDPOINT ====================

@router.get("/models")
def list_available_models():
    """
    List all available ML models and their capabilities.
    """
    return {
        "models": [
            {
                "id": "hierarchical_forecasting",
                "name": "Hierarchical Time Series Forecasting",
                "description": "Demand prediction for enrollments, demographics, and biometrics",
                "endpoints": ["/ml/forecasting/demand", "/ml/forecasting/seasonality"],
                "use_cases": ["Operator forecasting", "Mobile unit planning", "Seasonality calendar"]
            },
            {
                "id": "capacity_planning",
                "name": "Queueing/Capacity Planning Model",
                "description": "Operator requirements using Little's Law",
                "endpoints": ["/ml/capacity/operators", "/ml/capacity/yearly-plan/{district}"],
                "use_cases": ["Operator staffing", "Wait time estimation", "SLA risk"]
            },
            {
                "id": "underserved_scoring",
                "name": "Underserved Scoring Model",
                "description": "Identify underserved regions for targeted intervention",
                "endpoints": ["/ml/underserved/scores", "/ml/underserved/mobile-placement"],
                "use_cases": ["Mobile unit placement", "Equity expansion", "Child targeting"]
            },
            {
                "id": "fraud_detection",
                "name": "Forensic Fraud Detection",
                "description": "Digit analysis using Benford's Law",
                "endpoints": ["/ml/fraud/analysis"],
                "use_cases": ["Fraud suspicion scoring", "Audit prioritization"]
            },
            {
                "id": "clustering",
                "name": "District Clustering/Segmentation",
                "description": "Behavioral segmentation with anomaly detection",
                "endpoints": ["/ml/clustering/segments"],
                "use_cases": ["User segmentation", "Infiltration zone detection"]
            },
            {
                "id": "hotspot_detection",
                "name": "EWMA Hotspot Detection",
                "description": "Biometric infrastructure load monitoring",
                "endpoints": ["/ml/hotspots/biometric-load", "/ml/hotspots/infrastructure-demand"],
                "use_cases": ["Top load districts", "Device failure risk", "Capacity alerts"]
            },
            {
                "id": "cohort_model",
                "name": "Cohort Transition Model",
                "description": "5-year MBU prediction from child enrollments",
                "endpoints": ["/ml/cohort/mbu-prediction", "/ml/cohort/equipment-plan"],
                "use_cases": ["5-year MBU demand", "Equipment planning", "Investment scheduling"]
            }
        ],
        "total_models": 7,
        "solution_coverage": {
            "Operator requirement forecasting": ["hierarchical_forecasting", "capacity_planning"],
            "Mobile enrollment unit placement": ["hierarchical_forecasting", "underserved_scoring"],
            "Fraud suspicion scoring": ["fraud_detection"],
            "Infiltration zone detection": ["clustering"],
            "Biometric infrastructure demand": ["hierarchical_forecasting", "hotspot_detection"],
            "Seasonality calendar": ["hierarchical_forecasting"],
            "User behavior segmentation": ["clustering"],
            "5-year biometric demand": ["cohort_model"]
        }
    }


@router.get("/models/status")
def get_models_training_status():
    """
    Get training status for all ML models.
    
    Returns whether each model is trained, file sizes, and training timestamps.
    """
    status = get_model_status()
    
    trained_count = sum(1 for s in status.values() if s['trained'])
    
    return {
        "summary": {
            "total_models": len(status),
            "trained": trained_count,
            "untrained": len(status) - trained_count,
            "all_models_ready": trained_count == len(status)
        },
        "models": status
    }


@router.get("/models/{model_name}/data")
def get_trained_model_data(model_name: str):
    """
    Get pre-computed data from a trained model.
    
    Valid model names:
    - forecaster
    - capacity_planning
    - underserved_scoring
    - fraud_detector
    - clustering
    - hotspot_detector
    - cohort_model
    """
    model = get_model(model_name)
    
    if model is None:
        raise HTTPException(
            status_code=404, 
            detail=f"Model '{model_name}' not found or not trained. Run training first."
        )
    
    # Return model-specific data
    if model_name == "forecaster":
        return {
            "model": model_name,
            "trained": True,
            "data_types": list(model.trained_params.keys()) if hasattr(model, 'trained_params') else [],
            "params": model.trained_params if hasattr(model, 'trained_params') else {}
        }
    
    elif model_name == "capacity_planning":
        return {
            "model": model_name,
            "trained": True,
            "districts_count": len(model.district_metrics) if hasattr(model, 'district_metrics') else 0,
            "service_rate": model.service_rate_per_operator if hasattr(model, 'service_rate_per_operator') else 32,
            "top_districts": dict(sorted(
                model.district_metrics.items(),
                key=lambda x: x[1].get('total_daily_arrival', 0),
                reverse=True
            )[:10]) if hasattr(model, 'district_metrics') else {}
        }
    
    elif model_name == "underserved_scoring":
        scores = model.district_scores if hasattr(model, 'district_scores') else {}
        # Sort by underserved score
        sorted_districts = sorted(
            scores.items(),
            key=lambda x: x[1].get('underserved_score', 0),
            reverse=True
        )
        return {
            "model": model_name,
            "trained": True,
            "districts_scored": len(scores),
            "top_underserved": sorted_districts[:20],
            "least_underserved": sorted_districts[-10:] if len(sorted_districts) > 10 else []
        }
    
    elif model_name == "fraud_detector":
        scores = model.district_scores if hasattr(model, 'district_scores') else {}
        sorted_districts = sorted(
            scores.items(),
            key=lambda x: x[1].get('fraud_risk_score', 0),
            reverse=True
        )
        return {
            "model": model_name,
            "trained": True,
            "districts_analyzed": len(scores),
            "high_risk_districts": [
                {"district": d, **v} for d, v in sorted_districts[:20]
            ],
            "low_risk_districts": [
                {"district": d, **v} for d, v in sorted_districts[-10:]
            ] if len(sorted_districts) > 10 else []
        }
    
    elif model_name == "clustering":
        return {
            "model": model_name,
            "trained": True,
            "n_clusters": len(model.segment_profiles) if hasattr(model, 'segment_profiles') else 0,
            "segment_profiles": model.segment_profiles if hasattr(model, 'segment_profiles') else {},
            "sample_assignments": dict(list(model.district_segments.items())[:20]) if hasattr(model, 'district_segments') else {}
        }
    
    elif model_name == "hotspot_detector":
        ewma = model.district_ewma if hasattr(model, 'district_ewma') else {}
        hotspots = {d: v for d, v in ewma.items() if v.get('is_hotspot', False)}
        return {
            "model": model_name,
            "trained": True,
            "districts_monitored": len(ewma),
            "current_hotspots_count": len(hotspots),
            "current_hotspots": hotspots,
            "top_by_ewma": dict(sorted(
                ewma.items(),
                key=lambda x: x[1].get('ewma_current', 0),
                reverse=True
            )[:20])
        }
    
    elif model_name == "cohort_model":
        projections = model.district_projections if hasattr(model, 'district_projections') else {}
        sorted_proj = sorted(
            projections.items(),
            key=lambda x: x[1].get('total_5_year_mbu', 0),
            reverse=True
        )
        total_5year = sum(p.get('total_5_year_mbu', 0) for p in projections.values())
        return {
            "model": model_name,
            "trained": True,
            "districts_projected": len(projections),
            "total_5_year_mbu_demand": total_5year,
            "top_mbu_districts": [
                {"district": d, **v} for d, v in sorted_proj[:20]
            ],
            "yearly_breakdown": {
                f"year_{i}": sum(p.get('projected_mbu', {}).get(f'year_{i}', 0) for p in projections.values())
                for i in range(1, 6)
            }
        }
    
    else:
        return {"model": model_name, "trained": True, "data": "Model data retrieval not implemented"}

