"""
ML Analytics V2 API Router - Enhanced with Hierarchical Analysis.

Provides access to trained ML models with:
- State-level comparisons (districts within a state)
- National-level comparisons (states compared)
- Monthly time intervals (Sep, Oct, Nov, Dec)
- Top 50 national rankings for all parameters
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Optional, List
import pickle
from pathlib import Path

router = APIRouter()

# Model paths
TRAINED_DIR = Path(__file__).parent.parent / "ml_models" / "trained"


def load_model(name: str):
    """Load a trained model from disk."""
    path = TRAINED_DIR / f"{name}.pkl"
    if not path.exists():
        return None
    with open(path, 'rb') as f:
        return pickle.load(f)


# ==================== COMMON ENDPOINTS ====================

@router.get("/status")
def get_v2_models_status():
    """Get status of all V2 trained models."""
    models = [
        'forecaster_v2', 'capacity_planning_v2', 'underserved_scoring_v2',
        'fraud_detector_v2', 'clustering_v2', 'hotspot_detector_v2',
        'cohort_model_v2', 'master_rankings'
    ]
    
    status = {}
    for name in models:
        path = TRAINED_DIR / f"{name}.pkl"
        meta_path = TRAINED_DIR / f"{name}_metadata.json"
        
        if path.exists():
            import json
            metadata = {}
            if meta_path.exists():
                with open(meta_path, 'r') as f:
                    metadata = json.load(f)
            
            status[name] = {
                'trained': True,
                'file_size_kb': round(path.stat().st_size / 1024, 1),
                'trained_at': metadata.get('trained_at', 'unknown'),
                'training_time_sec': round(metadata.get('training_time_sec', 0), 2)
            }
        else:
            status[name] = {'trained': False}
    
    return {
        'summary': {
            'total': len(models),
            'trained': sum(1 for s in status.values() if s.get('trained')),
            'ready': all(s.get('trained') for s in status.values())
        },
        'models': status
    }


@router.get("/rankings/top-50-needy")
def get_top_50_needy(
    parameter: Optional[str] = Query(
        None, 
        description="Filter by: capacity_stress, underserved, fraud_risk, hotspot_intensity, mbu_demand, or overall"
    )
):
    """
    Get Top 50 neediest districts nationally across all parameters.
    
    Returns composite neediness score or parameter-specific ranking.
    """
    data = load_model('master_rankings')
    if not data:
        raise HTTPException(status_code=404, detail="Master rankings not trained. Run train_models_v2.py first.")
    
    if parameter and parameter != 'overall':
        if parameter not in data.get('by_parameter', {}):
            raise HTTPException(status_code=400, detail=f"Invalid parameter. Choose from: capacity_stress, underserved, fraud_risk, hotspot_intensity, mbu_demand")
        return {
            'parameter': parameter,
            'top_50': data['by_parameter'][parameter]
        }
    
    return {
        'parameter': 'overall (composite)',
        'top_50': data.get('top_50_needy_overall', [])
    }


# ==================== CAPACITY PLANNING ====================

@router.get("/capacity")
def get_capacity_data(
    state: Optional[str] = Query(None, description="Filter by state"),
    period: Optional[str] = Query(None, description="Filter by period: september, october, november, december"),
    level: str = Query("national", description="Level: national, state, district, all_states")
):
    """
    Get capacity planning data with hierarchical breakdown.
    
    Uses Erlang-C queueing model for mathematically rigorous operator estimation.
    
    - national: India-level aggregates with recommended FTE operators
    - state: State-level with district rankings (requires state param)
    - district: All districts in a state (requires state param)
    - all_states: Compare all states
    """
    data = load_model('capacity_planning_v2')
    if not data:
        raise HTTPException(status_code=404, detail="Capacity model not trained")
    
    # Include methodology in all responses
    methodology = data.get('methodology', {})
    
    if period and period in data.get('by_period', {}):
        period_data = data['by_period'][period]
        if state:
            return {
                'period': period,
                'state': state,
                'methodology': methodology,
                'data': period_data.get('by_state', {}).get(state, {}),
                'top_districts_in_period': [d for d in period_data.get('top_50_districts', []) if d.get('state') == state][:20]
            }
        return {
            'period': period,
            'methodology': methodology,
            'national_avg_daily': period_data.get('national_avg_daily'),
            'national_p95_daily': period_data.get('national_p95_daily'),
            'national_recommended_operators': period_data.get('national_recommended_operators'),
            'top_50_districts': period_data.get('top_50_districts', [])[:50]
        }
    
    if level == "national":
        national = data.get('national', {})
        return {
            'level': 'national',
            'methodology': methodology,
            'summary': {
                'avg_daily_demand': national.get('avg_daily_demand', 0),
                'p95_daily_demand': national.get('p95_daily_demand', 0),
                'peak_daily_demand': national.get('peak_daily_demand', 0),
                'recommended_operators': national.get('recommended_fte_operators', national.get('operators_for_p95', 0)),
                'peak_surge_operators': national.get('peak_surge_operators', 0),
                'utilization': national.get('utilization_at_p95', 0),
                'avg_wait_minutes': national.get('avg_wait_minutes', 0),
                'service_level': national.get('service_level', 0.8),
                'weekend_operators': national.get('weekend_operators', 0),
                'service_capacity_per_operator': national.get('service_capacity_per_operator_per_day', 32.5)
            },
            'data': national,
            'top_50_needy': data.get('rankings', {}).get('top_50_needy', []),
            'top_50_stress': data.get('rankings', {}).get('top_50_stress', [])
        }
    
    if level == "state" and state:
        state_data = data.get('by_state', {}).get(state, {})
        districts = [
            v for k, v in data.get('by_district', {}).items() 
            if v.get('state') == state
        ]
        # Sort by capacity stress index (new metric)
        sorted_districts = sorted(
            districts, 
            key=lambda x: x.get('capacity_stress_index', x.get('capacity_stress_score', 0)), 
            reverse=True
        )
        
        return {
            'level': 'state',
            'state': state,
            'methodology': methodology,
            'summary': {
                'avg_daily_demand': state_data.get('avg_daily_demand', 0),
                'p95_daily_demand': state_data.get('p95_daily_demand', 0),
                'recommended_operators': state_data.get('recommended_operators', 0),
                'utilization': state_data.get('utilization', 0),
                'avg_wait_minutes': state_data.get('avg_wait_minutes', 0),
                'weekend_operators': state_data.get('weekend_operators', 0),
                'district_count': state_data.get('district_count', len(districts))
            },
            'districts': sorted_districts[:50]
        }
    
    if level == "district" and state:
        districts = [
            v for k, v in data.get('by_district', {}).items()
            if v.get('state') == state
        ]
        return {
            'level': 'district',
            'state': state,
            'methodology': methodology,
            'count': len(districts),
            'districts': sorted(
                districts, 
                key=lambda x: x.get('recommended_operators', x.get('operators_needed_peak', 0)), 
                reverse=True
            )
        }
    
    # Return state-level comparison
    if level == "all_states":
        states_summary = {}
        for state_name, state_data in data.get('by_state', {}).items():
            states_summary[state_name] = {
                'recommended_operators': state_data.get('recommended_operators', state_data.get('operators_needed_avg', 0)),
                'avg_daily_demand': state_data.get('avg_daily_demand', 0),
                'p95_daily_demand': state_data.get('p95_daily_demand', state_data.get('peak_daily_demand', 0)),
                'utilization': state_data.get('utilization', 0),
                'district_count': state_data.get('district_count', 0)
            }
        return {
            'level': 'all_states',
            'methodology': methodology,
            'states': states_summary
        }
    
    return {
        'level': 'all_states',
        'states': data.get('by_state', {})
    }


# ==================== UNDERSERVED SCORING ====================

@router.get("/underserved")
def get_underserved_data(
    state: Optional[str] = Query(None),
    period: Optional[str] = Query(None, description="september, october, november, december"),
    level: str = Query("national")
):
    """Get underserved scoring with hierarchical breakdown."""
    data = load_model('underserved_scoring_v2')
    if not data:
        raise HTTPException(status_code=404, detail="Underserved model not trained")
    
    if period and period in data.get('by_period', {}):
        period_data = data['by_period'][period]
        if state:
            return {
                'period': period,
                'state': state,
                'avg_score': period_data.get('by_state', {}).get(state, {}).get('avg_score', 0),
                'top_underserved': [d for d in period_data.get('top_50_underserved', []) if d.get('state') == state][:20]
            }
        return {
            'period': period,
            'avg_score': period_data.get('avg_score'),
            'top_50_underserved': period_data.get('top_50_underserved', [])
        }
    
    if level == "national":
        national = data.get('national', {})
        rankings = data.get('rankings', {})
        return {
            'level': 'national',
            'national': national,
            'summary': national,
            'rankings': {
                'top_50_underserved': rankings.get('top_50_underserved', []),
                'mobile_unit_priority': rankings.get('mobile_unit_priority', []),
            },
            'mbu_recommendations': {
                'total_recommended': national.get('mbu_recommended_total', 0),
                'methodology': 'Districts with underserved score > 60, low update activity, and moderate+ child population',
                'districts': [d for d in rankings.get('top_50_underserved', []) if d.get('mbu_recommended', False)][:30],
            },
            'scoring_methodology': national.get('scoring_methodology', 'Relative percentile ranking across all districts')
        }
    
    if level == "state" and state:
        state_data = data.get('by_state', {}).get(state, {})
        return {
            'level': 'state',
            'state': state,
            'data': state_data,
            'mbu_recommended_count': state_data.get('mbu_recommended_count', 0),
            'high_priority_districts': state_data.get('high_priority_districts', 0),
            'medium_priority_districts': state_data.get('medium_priority_districts', 0)
        }
    
    return {
        'level': 'all_states',
        'states': data.get('by_state', {})
    }


# ==================== FRAUD DETECTION ====================

@router.get("/fraud")
def get_fraud_data(
    state: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    level: str = Query("national")
):
    """Get fraud detection data with hierarchical breakdown."""
    data = load_model('fraud_detector_v2')
    if not data:
        raise HTTPException(status_code=404, detail="Fraud model not trained")
    
    if period and period in data.get('by_period', {}):
        period_data = data['by_period'][period]
        if state:
            return {
                'period': period,
                'state': state,
                'data': period_data.get('by_state', {}).get(state, {}),
                'high_risk_districts': [d for d in period_data.get('top_50_high_risk', []) if d.get('state') == state][:20]
            }
        return {
            'period': period,
            'avg_fraud_risk': period_data.get('avg_fraud_risk'),
            'top_50_high_risk': period_data.get('top_50_high_risk', [])
        }
    
    if level == "national":
        return {
            'level': 'national',
            'summary': data.get('national', {}),
            'top_50_high_risk': data.get('rankings', {}).get('top_50_high_risk', []),
            'audit_priority': data.get('rankings', {}).get('audit_priority', [])
        }
    
    if level == "state" and state:
        return {
            'level': 'state',
            'state': state,
            'data': data.get('by_state', {}).get(state, {})
        }
    
    return {'level': 'all_states', 'states': data.get('by_state', {})}


# ==================== CLUSTERING ====================

@router.get("/clustering")
def get_clustering_data(
    state: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    segment: Optional[str] = Query(None, description="Filter by segment name")
):
    """Get clustering/segmentation data."""
    data = load_model('clustering_v2')
    if not data:
        raise HTTPException(status_code=404, detail="Clustering model not trained")
    
    if period and period in data.get('by_period', {}):
        return {
            'period': period,
            'data': data['by_period'][period]
        }
    
    if segment and segment in data.get('rankings', {}):
        return {
            'segment': segment,
            'districts': data['rankings'][segment]
        }
    
    if state:
        return {
            'state': state,
            'data': data.get('by_state', {}).get(state, {}),
            'districts': [
                v for k, v in data.get('by_district', {}).items()
                if k.startswith(f"{state}|")
            ]
        }
    
    return {
        'summary': data.get('national', {}),
        'segment_profiles': data.get('segment_profiles', {}),
        'by_state': data.get('by_state', {})
    }


# ==================== HOTSPOT DETECTION ====================

@router.get("/hotspots")
def get_hotspot_data(
    state: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    level: str = Query("national")
):
    """Get EWMA hotspot detection data."""
    data = load_model('hotspot_detector_v2')
    if not data:
        raise HTTPException(status_code=404, detail="Hotspot model not trained")
    
    if period and period in data.get('by_period', {}):
        period_data = data['by_period'][period]
        if state:
            return {
                'period': period,
                'state': state,
                'data': period_data.get('by_state', {}).get(state, {}),
                'hotspots': [d for d in period_data.get('top_50_hotspots', []) if d.get('state') == state][:20]
            }
        return {
            'period': period,
            'hotspot_count': period_data.get('hotspot_count'),
            'top_50_hotspots': period_data.get('top_50_hotspots', [])
        }
    
    if level == "national":
        return {
            'level': 'national',
            'summary': data.get('national', {}),
            'top_50_hotspots': data.get('rankings', {}).get('top_50_hotspots', []),
            'infrastructure_priority': data.get('rankings', {}).get('infrastructure_priority', [])
        }
    
    if level == "state" and state:
        return {
            'level': 'state',
            'state': state,
            'data': data.get('by_state', {}).get(state, {})
        }
    
    return {'level': 'all_states', 'states': data.get('by_state', {})}


# ==================== COHORT / MBU ====================

@router.get("/mbu-projection")
def get_mbu_data(
    state: Optional[str] = Query(None),
    period: Optional[str] = Query(None),
    level: str = Query("national")
):
    """Get 5-year MBU projection data."""
    data = load_model('cohort_model_v2')
    if not data:
        raise HTTPException(status_code=404, detail="Cohort model not trained")
    
    if period and period in data.get('by_period', {}):
        period_data = data['by_period'][period]
        if state:
            return {
                'period': period,
                'state': state,
                'data': period_data.get('by_state', {}).get(state, {}),
                'top_districts': [d for d in period_data.get('top_50_mbu_demand', []) if d.get('state') == state][:20]
            }
        return {
            'period': period,
            'total_mbu_projected': period_data.get('total_mbu_projected'),
            'top_50_mbu_demand': period_data.get('top_50_mbu_demand', [])
        }
    
    if level == "national":
        return {
            'level': 'national',
            'summary': data.get('national', {}),
            'top_50_mbu_demand': data.get('rankings', {}).get('top_50_mbu_demand', []),
            'equipment_priority': data.get('rankings', {}).get('equipment_priority', [])
        }
    
    if level == "state" and state:
        return {
            'level': 'state',
            'state': state,
            'data': data.get('by_state', {}).get(state, {})
        }
    
    return {'level': 'all_states', 'states': data.get('by_state', {})}


# ==================== FORECASTING ====================

@router.get("/forecast")
def get_forecast_data(
    data_type: str = Query("enrollment", description="enrollment, demographic, biometric"),
    state: Optional[str] = Query(None),
    period: Optional[str] = Query(None)
):
    """Get forecasting data with trend and seasonality."""
    data = load_model('forecaster_v2')
    if not data:
        raise HTTPException(status_code=404, detail="Forecaster not trained")
    
    if data_type not in ['enrollment', 'demographic', 'biometric']:
        raise HTTPException(status_code=400, detail="data_type must be: enrollment, demographic, or biometric")
    
    if period and period in data.get('by_period', {}).get(data_type, {}):
        period_data = data['by_period'][data_type][period]
        if state:
            return {
                'data_type': data_type,
                'period': period,
                'state': state,
                'state_total': period_data.get('by_state', {}).get(state, 0),
                'top_districts': [d for d in period_data.get('top_districts', []) if d.get('state') == state][:20]
            }
        return {
            'data_type': data_type,
            'period': period,
            'national_total': period_data.get('national_total'),
            'national_daily_avg': period_data.get('national_daily_avg'),
            'top_50_districts': period_data.get('top_districts', [])[:50],
            'by_state': period_data.get('by_state', {})
        }
    
    if state:
        return {
            'data_type': data_type,
            'state': state,
            'data': data.get('by_state', {}).get(data_type, {}).get(state, {})
        }
    
    return {
        'data_type': data_type,
        'national': data.get('national', {}).get(data_type, {}),
        'by_state': data.get('by_state', {}).get(data_type, {})
    }


# ==================== COMPARISON ENDPOINTS ====================

@router.get("/compare/districts")
def compare_districts_in_state(
    state: str = Query(..., description="State to compare districts within"),
    metric: str = Query("underserved_score", description="Metric to rank by"),
    period: Optional[str] = Query(None),
    top_n: int = Query(20, ge=5, le=100)
):
    """
    Compare all districts within a state by specified metric.
    
    Metrics: underserved_score, capacity_stress, fraud_risk, hotspot_intensity, mbu_demand
    """
    # Load the relevant model based on metric
    model_map = {
        'underserved_score': 'underserved_scoring_v2',
        'capacity_stress': 'capacity_planning_v2',
        'fraud_risk': 'fraud_detector_v2',
        'hotspot_intensity': 'hotspot_detector_v2',
        'mbu_demand': 'cohort_model_v2'
    }
    
    metric_key_map = {
        'underserved_score': 'underserved_score',
        'capacity_stress': 'capacity_stress_score',
        'fraud_risk': 'fraud_risk_score',
        'hotspot_intensity': 'intensity_score',
        'mbu_demand': 'total_5_year_mbu'
    }
    
    if metric not in model_map:
        raise HTTPException(status_code=400, detail=f"Invalid metric. Choose from: {list(model_map.keys())}")
    
    data = load_model(model_map[metric])
    if not data:
        raise HTTPException(status_code=404, detail=f"Model for {metric} not trained")
    
    # Get districts for this state
    districts = [
        {**v, 'key': k} for k, v in data.get('by_district', {}).items()
        if v.get('state') == state or k.startswith(f"{state}|")
    ]
    
    if not districts:
        raise HTTPException(status_code=404, detail=f"No districts found for state: {state}")
    
    # Sort by metric
    sorted_districts = sorted(
        districts,
        key=lambda x: x.get(metric_key_map[metric], 0),
        reverse=True
    )
    
    return {
        'state': state,
        'metric': metric,
        'total_districts': len(districts),
        'top_n': sorted_districts[:top_n],
        'bottom_n': sorted_districts[-min(top_n, len(sorted_districts)):][::-1]
    }


@router.get("/compare/states")
def compare_states_nationally(
    metric: str = Query("underserved_score", description="Metric to rank by"),
    period: Optional[str] = Query(None)
):
    """
    Compare all states nationally by specified metric.
    """
    model_map = {
        'underserved_score': 'underserved_scoring_v2',
        'capacity_stress': 'capacity_planning_v2',
        'fraud_risk': 'fraud_detector_v2',
        'hotspot_intensity': 'hotspot_detector_v2',
        'mbu_demand': 'cohort_model_v2'
    }
    
    metric_key_map = {
        'underserved_score': 'avg_underserved_score',
        'capacity_stress': 'operators_needed_peak',
        'fraud_risk': 'avg_fraud_risk',
        'hotspot_intensity': 'avg_intensity',
        'mbu_demand': 'total_5_year_mbu'
    }
    
    if metric not in model_map:
        raise HTTPException(status_code=400, detail=f"Invalid metric. Choose from: {list(model_map.keys())}")
    
    data = load_model(model_map[metric])
    if not data:
        raise HTTPException(status_code=404, detail=f"Model for {metric} not trained")
    
    # Get states
    states_data = data.get('by_state', {})
    
    state_list = [
        {'state': state, 'data': info, 'value': info.get(metric_key_map[metric], 0)}
        for state, info in states_data.items()
    ]
    
    sorted_states = sorted(state_list, key=lambda x: x['value'], reverse=True)
    
    return {
        'metric': metric,
        'total_states': len(sorted_states),
        'ranking': sorted_states
    }


@router.get("/monthly-comparison")
def get_monthly_comparison(
    metric: str = Query("underserved_score"),
    state: Optional[str] = Query(None)
):
    """
    Compare metrics across months (Sep, Oct, Nov, Dec).
    """
    model_map = {
        'underserved_score': 'underserved_scoring_v2',
        'fraud_risk': 'fraud_detector_v2',
        'hotspot_intensity': 'hotspot_detector_v2',
        'mbu_demand': 'cohort_model_v2',
        'capacity': 'capacity_planning_v2'
    }
    
    if metric not in model_map:
        raise HTTPException(status_code=400, detail=f"Invalid metric")
    
    data = load_model(model_map[metric])
    if not data:
        raise HTTPException(status_code=404, detail="Model not trained")
    
    periods = ['september', 'october', 'november', 'december']
    comparison = {}
    
    for period in periods:
        period_data = data.get('by_period', {}).get(period, {})
        
        if state:
            state_data = period_data.get('by_state', {}).get(state, {})
            comparison[period] = {
                'state': state,
                'data': state_data,
                'top_5_districts': [
                    d for d in period_data.get(f'top_50_{metric.replace("_", "")}', period_data.get('top_50_underserved', period_data.get('top_50_high_risk', period_data.get('top_50_hotspots', period_data.get('top_50_mbu_demand', period_data.get('top_50_districts', []))))))
                    if d.get('state') == state
                ][:5]
            }
        else:
            comparison[period] = {
                'national_summary': {k: v for k, v in period_data.items() if k not in ['by_state', 'top_50_underserved', 'top_50_high_risk', 'top_50_hotspots', 'top_50_mbu_demand', 'top_50_districts']}
            }
    
    return {
        'metric': metric,
        'state': state or 'All India',
        'monthly_comparison': comparison
    }
