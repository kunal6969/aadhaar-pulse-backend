"""
ML Model Training Script V2 - Enhanced with Hierarchical Analysis.

This script trains all 7 ML models with:
1. State-level comparisons (districts within a state)
2. National-level comparisons (states compared)
3. Monthly time intervals (Sep, Oct, Nov, Dec 2025)
4. National Top 50 rankings for all parameters

Usage:
    python scripts/train_models_v2.py
"""

import os
import sys
import json
import pickle
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from collections import defaultdict

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sqlalchemy import func

from app.database import SessionLocal, engine
from app.models.enrollment import Enrollment
from app.models.demographic_update import DemographicUpdate
from app.models.biometric_update import BiometricUpdate
from app.config import settings

# Training output directory
TRAINED_DIR = Path(__file__).parent.parent / "app" / "ml_models" / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)

# Time periods for analysis
TIME_PERIODS = {
    'september': {'start': date(2025, 9, 1), 'end': date(2025, 9, 30)},
    'october': {'start': date(2025, 10, 1), 'end': date(2025, 10, 31)},
    'november': {'start': date(2025, 11, 1), 'end': date(2025, 11, 30)},
    'december': {'start': date(2025, 12, 1), 'end': date(2025, 12, 31)},
    'all': {'start': date(2025, 6, 1), 'end': date(2025, 12, 31)}
}


def get_db_stats():
    """Get database statistics."""
    db = SessionLocal()
    try:
        enrollment_count = db.query(func.count(Enrollment.id)).scalar()
        demographic_count = db.query(func.count(DemographicUpdate.id)).scalar()
        biometric_count = db.query(func.count(BiometricUpdate.id)).scalar()
        
        print(f"📊 Database Statistics:")
        print(f"   Enrollments: {enrollment_count:,} rows")
        print(f"   Demographic Updates: {demographic_count:,} rows")
        print(f"   Biometric Updates: {biometric_count:,} rows")
        
        return {
            "enrollments": enrollment_count,
            "demographics": demographic_count,
            "biometrics": biometric_count
        }
    finally:
        db.close()


def load_all_data():
    """Load all data as DataFrames."""
    print("\n📥 Loading all data...")
    db = SessionLocal()
    try:
        # Enrollment
        enroll_query = db.query(
            Enrollment.date, Enrollment.state, Enrollment.district,
            Enrollment.age_0_5, Enrollment.age_5_17, Enrollment.age_18_plus, Enrollment.total
        ).all()
        enrollment_df = pd.DataFrame(enroll_query, columns=[
            'date', 'state', 'district', 'age_0_5', 'age_5_17', 'age_18_plus', 'total'
        ])
        enrollment_df['date'] = pd.to_datetime(enrollment_df['date'])
        print(f"   ✓ Enrollments: {len(enrollment_df):,}")
        
        # Demographic
        demo_query = db.query(
            DemographicUpdate.date, DemographicUpdate.state, DemographicUpdate.district,
            DemographicUpdate.demo_age_5_17, DemographicUpdate.demo_age_17_plus, DemographicUpdate.total
        ).all()
        demographic_df = pd.DataFrame(demo_query, columns=[
            'date', 'state', 'district', 'demo_age_5_17', 'demo_age_17_plus', 'total'
        ])
        demographic_df['date'] = pd.to_datetime(demographic_df['date'])
        print(f"   ✓ Demographics: {len(demographic_df):,}")
        
        # Biometric
        bio_query = db.query(
            BiometricUpdate.date, BiometricUpdate.state, BiometricUpdate.district,
            BiometricUpdate.bio_age_5_17, BiometricUpdate.bio_age_17_plus, BiometricUpdate.total
        ).all()
        biometric_df = pd.DataFrame(bio_query, columns=[
            'date', 'state', 'district', 'bio_age_5_17', 'bio_age_17_plus', 'total'
        ])
        biometric_df['date'] = pd.to_datetime(biometric_df['date'])
        print(f"   ✓ Biometrics: {len(biometric_df):,}")
        
        return enrollment_df, demographic_df, biometric_df
    finally:
        db.close()


def filter_by_period(df, period_name):
    """Filter DataFrame by time period."""
    period = TIME_PERIODS[period_name]
    mask = (df['date'] >= pd.Timestamp(period['start'])) & (df['date'] <= pd.Timestamp(period['end']))
    return df[mask]


def save_model(data, name: str, metadata: dict = None):
    """Save trained model/data to disk."""
    model_path = TRAINED_DIR / f"{name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(data, f)
    print(f"   💾 Saved: {name}.pkl ({model_path.stat().st_size / 1024:.1f} KB)")
    
    if metadata:
        meta_path = TRAINED_DIR / f"{name}_metadata.json"
        metadata['trained_at'] = datetime.now().isoformat()
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


# ==================== MODEL 1: FORECASTING ====================

def train_forecaster(enrollment_df, demographic_df, biometric_df):
    """Train hierarchical forecaster with state/district/monthly breakdown."""
    print("\n🔮 Training Hierarchical Forecaster...")
    start = time.time()
    
    results = {
        'national': {},      # India-level aggregates
        'by_state': {},      # State-level aggregates
        'by_district': {},   # District-level details
        'by_period': {},     # Monthly breakdowns
        'rankings': {}       # Top performers
    }
    
    for data_type, df in [('enrollment', enrollment_df), ('demographic', demographic_df), ('biometric', biometric_df)]:
        # National level
        daily = df.groupby('date')['total'].sum().reset_index()
        daily = daily.sort_values('date')
        
        if len(daily) > 7:
            values = daily['total'].values
            x = np.arange(len(values))
            slope, intercept = np.polyfit(x, values, 1) if len(x) > 1 else (0, values.mean())
            
            # Weekly seasonality
            daily['dow'] = pd.to_datetime(daily['date']).dt.dayofweek
            weekly = daily.groupby('dow')['total'].mean().to_dict()
            
            results['national'][data_type] = {
                'trend_slope': float(slope),
                'trend_intercept': float(intercept),
                'mean': float(values.mean()),
                'std': float(values.std()) if len(values) > 1 else 0,
                'weekly_seasonality': {int(k): float(v) for k, v in weekly.items()},
                'data_points': len(daily)
            }
        
        # By State
        results['by_state'][data_type] = {}
        for state in df['state'].unique():
            state_df = df[df['state'] == state]
            state_daily = state_df.groupby('date')['total'].sum()
            if len(state_daily) > 0:
                results['by_state'][data_type][state] = {
                    'mean': float(state_daily.mean()),
                    'total': int(state_daily.sum()),
                    'trend': float(np.polyfit(range(len(state_daily)), state_daily.values, 1)[0]) if len(state_daily) > 1 else 0,
                    'days': len(state_daily)
                }
        
        # By Period (monthly)
        results['by_period'][data_type] = {}
        for period_name in ['september', 'october', 'november', 'december']:
            period_df = filter_by_period(df, period_name)
            if len(period_df) > 0:
                # District-level for this period
                district_stats = period_df.groupby(['state', 'district']).agg({
                    'total': ['sum', 'mean', 'count']
                }).reset_index()
                district_stats.columns = ['state', 'district', 'total', 'daily_avg', 'days']
                
                results['by_period'][data_type][period_name] = {
                    'national_total': int(period_df['total'].sum()),
                    'national_daily_avg': float(period_df.groupby('date')['total'].sum().mean()),
                    'top_districts': district_stats.nlargest(50, 'total')[['state', 'district', 'total', 'daily_avg']].to_dict('records'),
                    'by_state': period_df.groupby('state')['total'].sum().to_dict()
                }
    
    elapsed = time.time() - start
    save_model(results, 'forecaster_v2', {
        'data_types': ['enrollment', 'demographic', 'biometric'],
        'periods': list(TIME_PERIODS.keys()),
        'training_time_sec': elapsed
    })
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    return results


# ==================== MODEL 2: CAPACITY PLANNING ====================

def train_capacity_model(enrollment_df, demographic_df, biometric_df):
    """Train capacity planning with state/district/monthly breakdown."""
    print("\n🏢 Training Capacity Planning Model...")
    start = time.time()
    
    SERVICE_RATE = 32  # transactions per operator per day
    
    results = {
        'national': {},
        'by_state': {},
        'by_district': {},
        'by_period': {},
        'rankings': {'top_50_needy': []}
    }
    
    # Combine all transaction types
    all_data = []
    for df, dtype in [(enrollment_df, 'enrollment'), (demographic_df, 'demographic'), (biometric_df, 'biometric')]:
        temp = df[['date', 'state', 'district', 'total']].copy()
        temp['type'] = dtype
        all_data.append(temp)
    
    combined = pd.concat(all_data, ignore_index=True)
    
    # National level
    national_daily = combined.groupby('date')['total'].sum()
    results['national'] = {
        'avg_daily_demand': float(national_daily.mean()),
        'peak_daily_demand': int(national_daily.max()),
        'operators_needed_avg': int(np.ceil(national_daily.mean() / SERVICE_RATE)),
        'operators_needed_peak': int(np.ceil(national_daily.max() / SERVICE_RATE)),
        'service_rate': SERVICE_RATE
    }
    
    # By State
    for state in combined['state'].unique():
        state_df = combined[combined['state'] == state]
        state_daily = state_df.groupby('date')['total'].sum()
        
        results['by_state'][state] = {
            'avg_daily_demand': float(state_daily.mean()),
            'peak_daily_demand': int(state_daily.max()),
            'operators_needed_avg': int(np.ceil(state_daily.mean() / SERVICE_RATE)),
            'operators_needed_peak': int(np.ceil(state_daily.max() / SERVICE_RATE)),
            'district_count': state_df['district'].nunique()
        }
    
    # By District
    district_metrics = []
    for (state, district), group in combined.groupby(['state', 'district']):
        daily = group.groupby('date')['total'].sum()
        
        metrics = {
            'state': state,
            'district': district,
            'avg_daily_demand': float(daily.mean()),
            'peak_daily_demand': int(daily.max()) if len(daily) > 0 else 0,
            'operators_needed_avg': int(np.ceil(daily.mean() / SERVICE_RATE)),
            'operators_needed_peak': int(np.ceil(daily.max() / SERVICE_RATE)) if len(daily) > 0 else 0,
            'total_transactions': int(daily.sum()),
            'days_of_data': len(daily),
            'variance': float(daily.var()) if len(daily) > 1 else 0
        }
        
        # Capacity stress score (higher = more stressed)
        metrics['capacity_stress_score'] = float(
            (metrics['peak_daily_demand'] / max(metrics['avg_daily_demand'], 1)) * 
            (metrics['variance'] / max(metrics['avg_daily_demand'], 1))
        ) if metrics['avg_daily_demand'] > 0 else 0
        
        district_metrics.append(metrics)
        results['by_district'][f"{state}|{district}"] = metrics
    
    # Top 50 needy districts (by capacity stress)
    sorted_districts = sorted(district_metrics, key=lambda x: x['capacity_stress_score'], reverse=True)
    results['rankings']['top_50_needy'] = sorted_districts[:50]
    
    # By Period
    for period_name in ['september', 'october', 'november', 'december']:
        period_df = filter_by_period(combined, period_name)
        if len(period_df) > 0:
            # District rankings for this period
            period_districts = []
            for (state, district), group in period_df.groupby(['state', 'district']):
                daily = group.groupby('date')['total'].sum()
                if len(daily) > 0:
                    period_districts.append({
                        'state': state,
                        'district': district,
                        'avg_daily': float(daily.mean()),
                        'peak_daily': int(daily.max()),
                        'operators_needed': int(np.ceil(daily.max() / SERVICE_RATE)),
                        'total': int(daily.sum())
                    })
            
            sorted_period = sorted(period_districts, key=lambda x: x['operators_needed'], reverse=True)
            
            results['by_period'][period_name] = {
                'national_avg_daily': float(period_df.groupby('date')['total'].sum().mean()),
                'top_50_districts': sorted_period[:50],
                'by_state': {}
            }
            
            # State-level for period
            for state in period_df['state'].unique():
                state_period = period_df[period_df['state'] == state]
                state_daily = state_period.groupby('date')['total'].sum()
                results['by_period'][period_name]['by_state'][state] = {
                    'avg_daily': float(state_daily.mean()),
                    'operators_needed': int(np.ceil(state_daily.mean() / SERVICE_RATE))
                }
    
    elapsed = time.time() - start
    save_model(results, 'capacity_planning_v2', {
        'service_rate': SERVICE_RATE,
        'districts_analyzed': len(district_metrics),
        'training_time_sec': elapsed
    })
    print(f"   ✓ Analyzed {len(district_metrics)} districts")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    return results


# ==================== MODEL 3: UNDERSERVED SCORING ====================

def train_underserved_model(enrollment_df, demographic_df, biometric_df):
    """Train underserved scoring with state/district/monthly breakdown."""
    print("\n🚐 Training Underserved Scoring Model...")
    start = time.time()
    
    results = {
        'national': {},
        'by_state': {},
        'by_district': {},
        'by_period': {},
        'rankings': {'top_50_underserved': [], 'mobile_unit_priority': []}
    }
    
    def calculate_underserved_score(enroll_total, demo_total, bio_total, child_ratio):
        """Calculate underserved score (0-100, higher = more underserved)."""
        if enroll_total == 0:
            return 100
        
        # Low update activity indicates underserved
        update_ratio = (demo_total + bio_total) / enroll_total
        activity_score = 100 * (1 - min(update_ratio / 2, 1))
        
        # High child ratio indicates future demand
        child_score = 100 * min(child_ratio, 1)
        
        # Composite
        return float(0.5 * activity_score + 0.5 * child_score)
    
    # Calculate for all districts
    district_scores = []
    
    for (state, district) in enrollment_df.groupby(['state', 'district']).groups.keys():
        enroll = enrollment_df[(enrollment_df['state'] == state) & (enrollment_df['district'] == district)]
        demo = demographic_df[(demographic_df['state'] == state) & (demographic_df['district'] == district)]
        bio = biometric_df[(biometric_df['state'] == state) & (biometric_df['district'] == district)]
        
        enroll_total = enroll['total'].sum()
        child_enroll = enroll['age_0_5'].sum() + enroll['age_5_17'].sum()
        demo_total = demo['total'].sum()
        bio_total = bio['total'].sum()
        
        child_ratio = child_enroll / enroll_total if enroll_total > 0 else 0
        score = calculate_underserved_score(enroll_total, demo_total, bio_total, child_ratio)
        
        district_data = {
            'state': state,
            'district': district,
            'underserved_score': score,
            'enrollments': int(enroll_total),
            'child_enrollments': int(child_enroll),
            'demographic_updates': int(demo_total),
            'biometric_updates': int(bio_total),
            'child_ratio': float(child_ratio),
            'update_activity_ratio': float((demo_total + bio_total) / max(enroll_total, 1))
        }
        
        district_scores.append(district_data)
        results['by_district'][f"{state}|{district}"] = district_data
    
    # Top 50 underserved nationally
    sorted_districts = sorted(district_scores, key=lambda x: x['underserved_score'], reverse=True)
    results['rankings']['top_50_underserved'] = sorted_districts[:50]
    results['rankings']['mobile_unit_priority'] = sorted_districts[:20]  # Top priority for mobile units
    
    # By State
    for state in enrollment_df['state'].unique():
        state_districts = [d for d in district_scores if d['state'] == state]
        if state_districts:
            avg_score = np.mean([d['underserved_score'] for d in state_districts])
            results['by_state'][state] = {
                'avg_underserved_score': float(avg_score),
                'district_count': len(state_districts),
                'most_underserved': sorted(state_districts, key=lambda x: x['underserved_score'], reverse=True)[:10],
                'total_enrollments': sum(d['enrollments'] for d in state_districts),
                'total_child_enrollments': sum(d['child_enrollments'] for d in state_districts)
            }
    
    # National summary
    results['national'] = {
        'avg_underserved_score': float(np.mean([d['underserved_score'] for d in district_scores])),
        'median_underserved_score': float(np.median([d['underserved_score'] for d in district_scores])),
        'total_districts': len(district_scores),
        'high_underserved_count': len([d for d in district_scores if d['underserved_score'] > 70]),
        'medium_underserved_count': len([d for d in district_scores if 40 <= d['underserved_score'] <= 70]),
        'low_underserved_count': len([d for d in district_scores if d['underserved_score'] < 40])
    }
    
    # By Period
    for period_name in ['september', 'october', 'november', 'december']:
        enroll_period = filter_by_period(enrollment_df, period_name)
        demo_period = filter_by_period(demographic_df, period_name)
        bio_period = filter_by_period(biometric_df, period_name)
        
        period_scores = []
        for (state, district) in enroll_period.groupby(['state', 'district']).groups.keys():
            e = enroll_period[(enroll_period['state'] == state) & (enroll_period['district'] == district)]
            d = demo_period[(demo_period['state'] == state) & (demo_period['district'] == district)]
            b = bio_period[(bio_period['state'] == state) & (bio_period['district'] == district)]
            
            et = e['total'].sum()
            ct = e['age_0_5'].sum() + e['age_5_17'].sum()
            dt = d['total'].sum()
            bt = b['total'].sum()
            
            cr = ct / et if et > 0 else 0
            score = calculate_underserved_score(et, dt, bt, cr)
            
            period_scores.append({
                'state': state,
                'district': district,
                'underserved_score': score,
                'enrollments': int(et),
                'child_enrollments': int(ct)
            })
        
        sorted_period = sorted(period_scores, key=lambda x: x['underserved_score'], reverse=True)
        results['by_period'][period_name] = {
            'avg_score': float(np.mean([p['underserved_score'] for p in period_scores])) if period_scores else 0,
            'top_50_underserved': sorted_period[:50],
            'by_state': {}
        }
        
        # State averages for period
        for state in set(p['state'] for p in period_scores):
            state_scores = [p for p in period_scores if p['state'] == state]
            results['by_period'][period_name]['by_state'][state] = {
                'avg_score': float(np.mean([p['underserved_score'] for p in state_scores])),
                'district_count': len(state_scores)
            }
    
    elapsed = time.time() - start
    save_model(results, 'underserved_scoring_v2', {
        'districts_scored': len(district_scores),
        'periods': list(TIME_PERIODS.keys()),
        'training_time_sec': elapsed
    })
    print(f"   ✓ Scored {len(district_scores)} districts")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    return results


# ==================== MODEL 4: FRAUD DETECTION ====================

def train_fraud_detector(enrollment_df, demographic_df, biometric_df):
    """Train fraud detector with state/district/monthly breakdown."""
    print("\n🔍 Training Forensic Fraud Detector...")
    start = time.time()
    
    # Benford's expected distribution
    BENFORD = {1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097, 5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046}
    
    results = {
        'national': {},
        'by_state': {},
        'by_district': {},
        'by_period': {},
        'rankings': {'top_50_high_risk': [], 'audit_priority': []}
    }
    
    def calculate_fraud_score(values):
        """Calculate fraud risk score based on digit analysis."""
        if len(values) < 30:
            return {'fraud_risk_score': 0, 'insufficient_data': True}
        
        values = values[values > 0]
        if len(values) < 30:
            return {'fraud_risk_score': 0, 'insufficient_data': True}
        
        # First digit (Benford's Law)
        first_digits = [int(str(abs(int(v)))[0]) for v in values if v >= 1]
        first_dist = {d: first_digits.count(d) / len(first_digits) for d in range(1, 10)} if first_digits else {}
        benford_dev = sum((first_dist.get(d, 0) - BENFORD[d]) ** 2 / BENFORD[d] for d in range(1, 10))
        
        # Last digit (should be uniform)
        last_digits = [int(str(abs(int(v)))[-1]) for v in values]
        last_dist = {d: last_digits.count(d) / len(last_digits) for d in range(10)} if last_digits else {}
        uniform_dev = sum((last_dist.get(d, 0) - 0.1) ** 2 for d in range(10)) * 100
        
        # Round numbers
        round_ratio = sum(1 for v in values if int(v) % 5 == 0) / len(values)
        
        # Composite score
        fraud_score = min(
            (benford_dev * 10) + (uniform_dev * 20) + max(0, (round_ratio - 0.2) * 100),
            100
        )
        
        return {
            'fraud_risk_score': float(fraud_score),
            'benford_deviation': float(benford_dev),
            'last_digit_deviation': float(uniform_dev),
            'round_number_ratio': float(round_ratio),
            'transactions_analyzed': len(values),
            'insufficient_data': False
        }
    
    # Combine all values per district
    district_fraud = []
    
    for (state, district) in enrollment_df.groupby(['state', 'district']).groups.keys():
        enroll_vals = enrollment_df[(enrollment_df['state'] == state) & (enrollment_df['district'] == district)]['total'].values
        demo_vals = demographic_df[(demographic_df['state'] == state) & (demographic_df['district'] == district)]['total'].values
        bio_vals = biometric_df[(biometric_df['state'] == state) & (biometric_df['district'] == district)]['total'].values
        
        all_vals = np.concatenate([enroll_vals, demo_vals, bio_vals])
        
        score_data = calculate_fraud_score(all_vals)
        score_data['state'] = state
        score_data['district'] = district
        
        district_fraud.append(score_data)
        results['by_district'][f"{state}|{district}"] = score_data
    
    # Top 50 high risk
    valid_districts = [d for d in district_fraud if not d.get('insufficient_data', True)]
    sorted_districts = sorted(valid_districts, key=lambda x: x['fraud_risk_score'], reverse=True)
    results['rankings']['top_50_high_risk'] = sorted_districts[:50]
    results['rankings']['audit_priority'] = [d for d in sorted_districts[:20] if d['fraud_risk_score'] > 50]
    
    # By State
    for state in enrollment_df['state'].unique():
        state_districts = [d for d in valid_districts if d['state'] == state]
        if state_districts:
            results['by_state'][state] = {
                'avg_fraud_risk': float(np.mean([d['fraud_risk_score'] for d in state_districts])),
                'max_fraud_risk': float(max(d['fraud_risk_score'] for d in state_districts)),
                'high_risk_count': len([d for d in state_districts if d['fraud_risk_score'] > 70]),
                'district_count': len(state_districts),
                'top_5_risky': sorted(state_districts, key=lambda x: x['fraud_risk_score'], reverse=True)[:5]
            }
    
    # National
    results['national'] = {
        'avg_fraud_risk': float(np.mean([d['fraud_risk_score'] for d in valid_districts])) if valid_districts else 0,
        'total_analyzed': len(valid_districts),
        'high_risk_count': len([d for d in valid_districts if d['fraud_risk_score'] > 70]),
        'medium_risk_count': len([d for d in valid_districts if 40 <= d['fraud_risk_score'] <= 70]),
        'low_risk_count': len([d for d in valid_districts if d['fraud_risk_score'] < 40])
    }
    
    # By Period
    for period_name in ['september', 'october', 'november', 'december']:
        enroll_p = filter_by_period(enrollment_df, period_name)
        demo_p = filter_by_period(demographic_df, period_name)
        bio_p = filter_by_period(biometric_df, period_name)
        
        period_scores = []
        for (state, district) in enroll_p.groupby(['state', 'district']).groups.keys():
            e_vals = enroll_p[(enroll_p['state'] == state) & (enroll_p['district'] == district)]['total'].values
            d_vals = demo_p[(demo_p['state'] == state) & (demo_p['district'] == district)]['total'].values
            b_vals = bio_p[(bio_p['state'] == state) & (bio_p['district'] == district)]['total'].values
            
            all_v = np.concatenate([e_vals, d_vals, b_vals])
            score = calculate_fraud_score(all_v)
            score['state'] = state
            score['district'] = district
            period_scores.append(score)
        
        valid_period = [p for p in period_scores if not p.get('insufficient_data', True)]
        sorted_period = sorted(valid_period, key=lambda x: x['fraud_risk_score'], reverse=True)
        
        results['by_period'][period_name] = {
            'avg_fraud_risk': float(np.mean([p['fraud_risk_score'] for p in valid_period])) if valid_period else 0,
            'top_50_high_risk': sorted_period[:50],
            'by_state': {}
        }
        
        for state in set(p['state'] for p in valid_period):
            state_scores = [p for p in valid_period if p['state'] == state]
            results['by_period'][period_name]['by_state'][state] = {
                'avg_fraud_risk': float(np.mean([p['fraud_risk_score'] for p in state_scores])),
                'high_risk_count': len([p for p in state_scores if p['fraud_risk_score'] > 70])
            }
    
    elapsed = time.time() - start
    save_model(results, 'fraud_detector_v2', {
        'districts_analyzed': len(valid_districts),
        'training_time_sec': elapsed
    })
    print(f"   ✓ Analyzed {len(valid_districts)} districts")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    return results


# ==================== MODEL 5: CLUSTERING ====================

def train_clustering_model(enrollment_df, demographic_df, biometric_df):
    """Train clustering with state/district breakdown."""
    print("\n🎯 Training District Clustering Model...")
    start = time.time()
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    results = {
        'national': {},
        'by_state': {},
        'by_district': {},
        'by_period': {},
        'segment_profiles': {},
        'rankings': {}
    }
    
    SEGMENT_NAMES = [
        'Stable Operations',
        'High Demographic Churn',
        'High Biometric Activity',
        'Enrollment Hotspots',
        'Low Activity/Dormant'
    ]
    
    def cluster_districts(enroll_df, demo_df, bio_df, n_clusters=5):
        """Cluster districts based on activity patterns."""
        # Aggregate by district
        e_agg = enroll_df.groupby(['state', 'district']).agg({'total': ['sum', 'mean', 'std']}).reset_index()
        e_agg.columns = ['state', 'district', 'e_sum', 'e_mean', 'e_std']
        
        d_agg = demo_df.groupby(['state', 'district']).agg({'total': ['sum', 'mean', 'std']}).reset_index()
        d_agg.columns = ['state', 'district', 'd_sum', 'd_mean', 'd_std']
        
        b_agg = bio_df.groupby(['state', 'district']).agg({'total': ['sum', 'mean', 'std']}).reset_index()
        b_agg.columns = ['state', 'district', 'b_sum', 'b_mean', 'b_std']
        
        # Merge
        merged = e_agg.merge(d_agg, on=['state', 'district'], how='outer').merge(b_agg, on=['state', 'district'], how='outer')
        merged = merged.fillna(0)
        
        if len(merged) < n_clusters:
            return None, None, merged
        
        # Features
        feature_cols = ['e_sum', 'e_mean', 'e_std', 'd_sum', 'd_mean', 'd_std', 'b_sum', 'b_mean', 'b_std']
        X = merged[feature_cols].values
        
        # Scale
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Cluster
        kmeans = KMeans(n_clusters=min(n_clusters, len(merged)), random_state=42, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        
        merged['cluster'] = labels
        
        return kmeans, scaler, merged
    
    # National clustering
    kmeans, scaler, clustered = cluster_districts(enrollment_df, demographic_df, biometric_df)
    
    if kmeans is not None:
        # Segment profiles
        for i in range(kmeans.n_clusters):
            cluster_data = clustered[clustered['cluster'] == i]
            results['segment_profiles'][i] = {
                'name': SEGMENT_NAMES[i] if i < len(SEGMENT_NAMES) else f'Segment {i}',
                'count': len(cluster_data),
                'avg_enrollment': float(cluster_data['e_sum'].mean()),
                'avg_demographic': float(cluster_data['d_sum'].mean()),
                'avg_biometric': float(cluster_data['b_sum'].mean())
            }
        
        # By district
        for _, row in clustered.iterrows():
            key = f"{row['state']}|{row['district']}"
            results['by_district'][key] = {
                'cluster_id': int(row['cluster']),
                'segment_name': SEGMENT_NAMES[int(row['cluster'])] if int(row['cluster']) < len(SEGMENT_NAMES) else f'Segment {int(row["cluster"])}',
                'enrollments': float(row['e_sum']),
                'demographics': float(row['d_sum']),
                'biometrics': float(row['b_sum'])
            }
        
        # By state
        for state in clustered['state'].unique():
            state_data = clustered[clustered['state'] == state]
            segment_dist = state_data['cluster'].value_counts().to_dict()
            results['by_state'][state] = {
                'district_count': len(state_data),
                'segment_distribution': {SEGMENT_NAMES[k] if k < len(SEGMENT_NAMES) else f'Segment {k}': v for k, v in segment_dist.items()},
                'dominant_segment': SEGMENT_NAMES[state_data['cluster'].mode().iloc[0]] if len(state_data) > 0 else 'Unknown'
            }
        
        # National summary
        results['national'] = {
            'total_districts': len(clustered),
            'n_clusters': kmeans.n_clusters,
            'segment_counts': {results['segment_profiles'][i]['name']: results['segment_profiles'][i]['count'] for i in range(kmeans.n_clusters)}
        }
        
        # Rankings by segment
        for i in range(kmeans.n_clusters):
            segment_districts = clustered[clustered['cluster'] == i].sort_values('e_sum', ascending=False)
            segment_name = SEGMENT_NAMES[i] if i < len(SEGMENT_NAMES) else f'Segment {i}'
            results['rankings'][segment_name] = segment_districts[['state', 'district', 'e_sum', 'd_sum', 'b_sum']].head(20).to_dict('records')
    
    # By Period
    for period_name in ['september', 'october', 'november', 'december']:
        e_p = filter_by_period(enrollment_df, period_name)
        d_p = filter_by_period(demographic_df, period_name)
        b_p = filter_by_period(biometric_df, period_name)
        
        _, _, period_clustered = cluster_districts(e_p, d_p, b_p, n_clusters=5)
        
        if period_clustered is not None and 'cluster' in period_clustered.columns:
            results['by_period'][period_name] = {
                'segment_counts': period_clustered['cluster'].value_counts().to_dict(),
                'districts_analyzed': len(period_clustered)
            }
    
    elapsed = time.time() - start
    save_model(results, 'clustering_v2', {
        'n_clusters': 5,
        'districts_clustered': len(clustered) if clustered is not None else 0,
        'training_time_sec': elapsed
    })
    print(f"   ✓ Clustered {len(clustered) if clustered is not None else 0} districts")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    return results


# ==================== MODEL 6: HOTSPOT DETECTION ====================

def train_hotspot_detector(enrollment_df, demographic_df, biometric_df):
    """Train EWMA hotspot detector with state/district/monthly breakdown."""
    print("\n🔥 Training EWMA Hotspot Detector...")
    start = time.time()
    
    LAMBDA = 0.2  # EWMA smoothing parameter
    
    results = {
        'national': {},
        'by_state': {},
        'by_district': {},
        'by_period': {},
        'rankings': {'top_50_hotspots': [], 'infrastructure_priority': []}
    }
    
    def calculate_ewma_metrics(values):
        """Calculate EWMA and control chart metrics."""
        if len(values) < 7:
            return None
        
        # EWMA calculation
        ewma = [values[0]]
        for i in range(1, len(values)):
            ewma.append(LAMBDA * values[i] + (1 - LAMBDA) * ewma[-1])
        ewma = np.array(ewma)
        
        mean = values.mean()
        std = values.std() if len(values) > 1 else 0
        sigma_ewma = std * np.sqrt(LAMBDA / (2 - LAMBDA))
        
        ucl = mean + 3 * sigma_ewma
        lcl = max(0, mean - 3 * sigma_ewma)
        
        violations = np.sum((ewma > ucl) | (ewma < lcl))
        
        return {
            'mean': float(mean),
            'std': float(std),
            'ewma_current': float(ewma[-1]),
            'ucl': float(ucl),
            'lcl': float(lcl),
            'violation_count': int(violations),
            'violation_rate': float(violations / len(ewma)),
            'is_hotspot': bool(ewma[-1] > ucl),
            'trend': 'increasing' if len(ewma) > 1 and ewma[-1] > ewma[-2] else 'decreasing',
            'intensity_score': float(max(0, (ewma[-1] - ucl) / max(sigma_ewma, 1)) * 100) if ewma[-1] > ucl else 0
        }
    
    # By District (using biometric data primarily)
    district_hotspots = []
    
    for (state, district), group in biometric_df.groupby(['state', 'district']):
        daily = group.groupby('date')['total'].sum().sort_index()
        
        metrics = calculate_ewma_metrics(daily.values)
        if metrics:
            metrics['state'] = state
            metrics['district'] = district
            metrics['total_biometric'] = int(daily.sum())
            
            district_hotspots.append(metrics)
            results['by_district'][f"{state}|{district}"] = metrics
    
    # Top 50 hotspots (by intensity score)
    sorted_hotspots = sorted(district_hotspots, key=lambda x: x['intensity_score'], reverse=True)
    results['rankings']['top_50_hotspots'] = sorted_hotspots[:50]
    results['rankings']['infrastructure_priority'] = [h for h in sorted_hotspots[:30] if h['is_hotspot']]
    
    # By State
    for state in biometric_df['state'].unique():
        state_districts = [d for d in district_hotspots if d['state'] == state]
        if state_districts:
            results['by_state'][state] = {
                'hotspot_count': len([d for d in state_districts if d['is_hotspot']]),
                'district_count': len(state_districts),
                'avg_intensity': float(np.mean([d['intensity_score'] for d in state_districts])),
                'top_5_hotspots': sorted(state_districts, key=lambda x: x['intensity_score'], reverse=True)[:5]
            }
    
    # National
    results['national'] = {
        'total_hotspots': len([d for d in district_hotspots if d['is_hotspot']]),
        'total_districts': len(district_hotspots),
        'hotspot_rate': float(len([d for d in district_hotspots if d['is_hotspot']]) / max(len(district_hotspots), 1)),
        'avg_intensity': float(np.mean([d['intensity_score'] for d in district_hotspots])) if district_hotspots else 0
    }
    
    # By Period
    for period_name in ['september', 'october', 'november', 'december']:
        bio_p = filter_by_period(biometric_df, period_name)
        
        period_hotspots = []
        for (state, district), group in bio_p.groupby(['state', 'district']):
            daily = group.groupby('date')['total'].sum().sort_index()
            metrics = calculate_ewma_metrics(daily.values)
            if metrics:
                metrics['state'] = state
                metrics['district'] = district
                period_hotspots.append(metrics)
        
        sorted_period = sorted(period_hotspots, key=lambda x: x['intensity_score'], reverse=True)
        
        results['by_period'][period_name] = {
            'hotspot_count': len([p for p in period_hotspots if p['is_hotspot']]),
            'districts_analyzed': len(period_hotspots),
            'top_50_hotspots': sorted_period[:50],
            'by_state': {}
        }
        
        for state in set(p['state'] for p in period_hotspots):
            state_p = [p for p in period_hotspots if p['state'] == state]
            results['by_period'][period_name]['by_state'][state] = {
                'hotspot_count': len([p for p in state_p if p['is_hotspot']]),
                'avg_intensity': float(np.mean([p['intensity_score'] for p in state_p]))
            }
    
    elapsed = time.time() - start
    save_model(results, 'hotspot_detector_v2', {
        'districts_monitored': len(district_hotspots),
        'lambda_param': LAMBDA,
        'training_time_sec': elapsed
    })
    print(f"   ✓ Monitored {len(district_hotspots)} districts")
    print(f"   🔥 Current hotspots: {results['national']['total_hotspots']}")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    return results


# ==================== MODEL 7: COHORT MODEL ====================

def train_cohort_model(enrollment_df, biometric_df):
    """Train cohort transition model with state/district/monthly breakdown."""
    print("\n👶 Training Cohort Transition Model...")
    start = time.time()
    
    results = {
        'national': {},
        'by_state': {},
        'by_district': {},
        'by_period': {},
        'rankings': {'top_50_mbu_demand': [], 'equipment_priority': []}
    }
    
    def calculate_mbu_projection(child_0_5, child_5_17, current_bio_rate):
        """Calculate 5-year MBU projection."""
        # Children 0-5 will need MBU at age 5, 10, 15
        # Children 5-17 need MBU at 15 (some already done)
        
        bio_rate = min(current_bio_rate, 1.0)
        
        # Yearly projections
        year_1 = child_0_5 * 0.1 + child_5_17 * bio_rate * 0.2
        year_2 = child_0_5 * 0.2 + child_5_17 * bio_rate * 0.2
        year_3 = child_0_5 * 0.2 + child_5_17 * bio_rate * 0.2
        year_4 = child_0_5 * 0.25 + child_5_17 * bio_rate * 0.2
        year_5 = child_0_5 * 0.25 + child_5_17 * bio_rate * 0.2
        
        return {
            'year_1': int(year_1),
            'year_2': int(year_2),
            'year_3': int(year_3),
            'year_4': int(year_4),
            'year_5': int(year_5),
            'total_5_year': int(year_1 + year_2 + year_3 + year_4 + year_5)
        }
    
    # By District
    district_projections = []
    
    for (state, district) in enrollment_df.groupby(['state', 'district']).groups.keys():
        enroll = enrollment_df[(enrollment_df['state'] == state) & (enrollment_df['district'] == district)]
        bio = biometric_df[(biometric_df['state'] == state) & (biometric_df['district'] == district)]
        
        child_0_5 = enroll['age_0_5'].sum()
        child_5_17 = enroll['age_5_17'].sum()
        bio_total = bio['total'].sum()
        
        current_bio_rate = bio_total / max(child_5_17, 1) if child_5_17 > 0 else 0
        
        projection = calculate_mbu_projection(child_0_5, child_5_17, current_bio_rate)
        
        district_data = {
            'state': state,
            'district': district,
            'current_children_0_5': int(child_0_5),
            'current_children_5_17': int(child_5_17),
            'current_bio_rate': float(min(current_bio_rate, 1.0)),
            'projected_mbu': projection,
            'total_5_year_mbu': projection['total_5_year'],
            'equipment_score': float(projection['total_5_year'] / 10000)  # Normalized score
        }
        
        district_projections.append(district_data)
        results['by_district'][f"{state}|{district}"] = district_data
    
    # Top 50 MBU demand
    sorted_districts = sorted(district_projections, key=lambda x: x['total_5_year_mbu'], reverse=True)
    results['rankings']['top_50_mbu_demand'] = sorted_districts[:50]
    results['rankings']['equipment_priority'] = sorted_districts[:30]
    
    # By State
    for state in enrollment_df['state'].unique():
        state_districts = [d for d in district_projections if d['state'] == state]
        if state_districts:
            total_mbu = sum(d['total_5_year_mbu'] for d in state_districts)
            results['by_state'][state] = {
                'total_5_year_mbu': total_mbu,
                'district_count': len(state_districts),
                'avg_mbu_per_district': float(total_mbu / len(state_districts)),
                'top_5_districts': sorted(state_districts, key=lambda x: x['total_5_year_mbu'], reverse=True)[:5],
                'yearly_breakdown': {
                    f'year_{i}': sum(d['projected_mbu'][f'year_{i}'] for d in state_districts)
                    for i in range(1, 6)
                }
            }
    
    # National
    total_national = sum(d['total_5_year_mbu'] for d in district_projections)
    results['national'] = {
        'total_5_year_mbu': total_national,
        'total_districts': len(district_projections),
        'avg_mbu_per_district': float(total_national / max(len(district_projections), 1)),
        'yearly_breakdown': {
            f'year_{i}': sum(d['projected_mbu'][f'year_{i}'] for d in district_projections)
            for i in range(1, 6)
        }
    }
    
    # By Period (calculate based on enrollments in that period)
    for period_name in ['september', 'october', 'november', 'december']:
        enroll_p = filter_by_period(enrollment_df, period_name)
        bio_p = filter_by_period(biometric_df, period_name)
        
        period_projections = []
        for (state, district) in enroll_p.groupby(['state', 'district']).groups.keys():
            e = enroll_p[(enroll_p['state'] == state) & (enroll_p['district'] == district)]
            b = bio_p[(bio_p['state'] == state) & (bio_p['district'] == district)]
            
            c05 = e['age_0_5'].sum()
            c517 = e['age_5_17'].sum()
            bt = b['total'].sum()
            br = bt / max(c517, 1)
            
            proj = calculate_mbu_projection(c05, c517, br)
            period_projections.append({
                'state': state,
                'district': district,
                'children_0_5': int(c05),
                'children_5_17': int(c517),
                'total_5_year_mbu': proj['total_5_year']
            })
        
        sorted_period = sorted(period_projections, key=lambda x: x['total_5_year_mbu'], reverse=True)
        
        results['by_period'][period_name] = {
            'total_mbu_projected': sum(p['total_5_year_mbu'] for p in period_projections),
            'districts_analyzed': len(period_projections),
            'top_50_mbu_demand': sorted_period[:50],
            'by_state': {}
        }
        
        for state in set(p['state'] for p in period_projections):
            state_p = [p for p in period_projections if p['state'] == state]
            results['by_period'][period_name]['by_state'][state] = {
                'total_mbu': sum(p['total_5_year_mbu'] for p in state_p),
                'district_count': len(state_p)
            }
    
    elapsed = time.time() - start
    save_model(results, 'cohort_model_v2', {
        'districts_projected': len(district_projections),
        'total_5_year_mbu': total_national,
        'training_time_sec': elapsed
    })
    print(f"   ✓ Projected MBU for {len(district_projections)} districts")
    print(f"   📊 Total 5-year MBU: {total_national:,}")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    return results


# ==================== MASTER RANKINGS ====================

def create_master_rankings(all_results):
    """Create unified national rankings across all models."""
    print("\n🏆 Creating Master National Rankings...")
    
    rankings = {
        'top_50_needy_overall': [],
        'by_parameter': {},
        'by_period': {}
    }
    
    # Collect scores for each district
    district_scores = defaultdict(lambda: {
        'capacity_stress': 0,
        'underserved_score': 0,
        'fraud_risk': 0,
        'hotspot_intensity': 0,
        'mbu_demand': 0
    })
    
    # Capacity Planning
    if 'capacity_planning_v2' in all_results:
        for key, data in all_results['capacity_planning_v2'].get('by_district', {}).items():
            district_scores[key]['capacity_stress'] = data.get('capacity_stress_score', 0)
            district_scores[key]['state'] = data.get('state', key.split('|')[0])
            district_scores[key]['district'] = data.get('district', key.split('|')[1] if '|' in key else key)
    
    # Underserved
    if 'underserved_scoring_v2' in all_results:
        for key, data in all_results['underserved_scoring_v2'].get('by_district', {}).items():
            district_scores[key]['underserved_score'] = data.get('underserved_score', 0)
    
    # Fraud
    if 'fraud_detector_v2' in all_results:
        for key, data in all_results['fraud_detector_v2'].get('by_district', {}).items():
            district_scores[key]['fraud_risk'] = data.get('fraud_risk_score', 0)
    
    # Hotspot
    if 'hotspot_detector_v2' in all_results:
        for key, data in all_results['hotspot_detector_v2'].get('by_district', {}).items():
            district_scores[key]['hotspot_intensity'] = data.get('intensity_score', 0)
    
    # MBU
    if 'cohort_model_v2' in all_results:
        for key, data in all_results['cohort_model_v2'].get('by_district', {}).items():
            district_scores[key]['mbu_demand'] = data.get('total_5_year_mbu', 0)
    
    # Calculate composite "neediness" score
    for key, scores in district_scores.items():
        # Normalize MBU demand (scale to 0-100)
        max_mbu = max(d.get('mbu_demand', 0) for d in district_scores.values()) or 1
        mbu_normalized = (scores['mbu_demand'] / max_mbu) * 100
        
        # Composite score (weighted average)
        scores['composite_neediness'] = (
            scores['capacity_stress'] * 0.2 +
            scores['underserved_score'] * 0.25 +
            scores['fraud_risk'] * 0.15 +
            scores['hotspot_intensity'] * 0.15 +
            mbu_normalized * 0.25
        )
    
    # Top 50 overall needy
    sorted_overall = sorted(
        [{'key': k, **v} for k, v in district_scores.items()],
        key=lambda x: x['composite_neediness'],
        reverse=True
    )
    rankings['top_50_needy_overall'] = sorted_overall[:50]
    
    # By parameter
    rankings['by_parameter'] = {
        'capacity_stress': sorted(sorted_overall, key=lambda x: x['capacity_stress'], reverse=True)[:50],
        'underserved': sorted(sorted_overall, key=lambda x: x['underserved_score'], reverse=True)[:50],
        'fraud_risk': sorted(sorted_overall, key=lambda x: x['fraud_risk'], reverse=True)[:50],
        'hotspot_intensity': sorted(sorted_overall, key=lambda x: x['hotspot_intensity'], reverse=True)[:50],
        'mbu_demand': sorted(sorted_overall, key=lambda x: x['mbu_demand'], reverse=True)[:50]
    }
    
    save_model(rankings, 'master_rankings', {
        'total_districts': len(district_scores),
        'parameters': ['capacity_stress', 'underserved', 'fraud_risk', 'hotspot_intensity', 'mbu_demand']
    })
    
    print(f"   ✓ Ranked {len(district_scores)} districts")
    print(f"   🏆 Top needy: {rankings['top_50_needy_overall'][0]['district'] if rankings['top_50_needy_overall'] else 'N/A'}")
    
    return rankings


def main():
    """Main training function."""
    print("=" * 70)
    print("🚀 AADHAAR PULSE ML MODEL TRAINING V2 - HIERARCHICAL ANALYSIS")
    print("=" * 70)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {TRAINED_DIR}")
    print("=" * 70)
    
    total_start = time.time()
    
    # Check database
    stats = get_db_stats()
    if stats['enrollments'] == 0:
        print("\n❌ ERROR: No data in database.")
        return
    
    # Load all data
    enrollment_df, demographic_df, biometric_df = load_all_data()
    
    # Train all models
    print("\n" + "=" * 70)
    print("🎓 TRAINING MODELS WITH HIERARCHICAL BREAKDOWN")
    print("=" * 70)
    
    all_results = {}
    
    # 1. Forecaster
    all_results['forecaster_v2'] = train_forecaster(enrollment_df, demographic_df, biometric_df)
    
    # 2. Capacity Planning
    all_results['capacity_planning_v2'] = train_capacity_model(enrollment_df, demographic_df, biometric_df)
    
    # 3. Underserved Scoring
    all_results['underserved_scoring_v2'] = train_underserved_model(enrollment_df, demographic_df, biometric_df)
    
    # 4. Fraud Detection
    all_results['fraud_detector_v2'] = train_fraud_detector(enrollment_df, demographic_df, biometric_df)
    
    # 5. Clustering
    all_results['clustering_v2'] = train_clustering_model(enrollment_df, demographic_df, biometric_df)
    
    # 6. Hotspot Detection
    all_results['hotspot_detector_v2'] = train_hotspot_detector(enrollment_df, demographic_df, biometric_df)
    
    # 7. Cohort Model
    all_results['cohort_model_v2'] = train_cohort_model(enrollment_df, biometric_df)
    
    # Master Rankings
    all_results['master_rankings'] = create_master_rankings(all_results)
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "=" * 70)
    print("✅ TRAINING COMPLETE")
    print("=" * 70)
    print(f"Total training time: {total_time:.2f}s ({total_time/60:.1f} min)")
    print(f"\nTrained models saved to: {TRAINED_DIR}")
    
    # List saved files
    print("\nSaved files:")
    for f in sorted(TRAINED_DIR.glob("*_v2*.pkl")):
        size = f.stat().st_size
        print(f"   {f.name} ({size / 1024:.1f} KB)")
    
    print("\n" + "=" * 70)
    print("🎉 All 7 ML models + Master Rankings trained with hierarchical data!")
    print("=" * 70)
    print("\nNew capabilities:")
    print("  ✓ State-level comparisons (districts within state)")
    print("  ✓ National-level comparisons (state rankings)")
    print("  ✓ Monthly breakdowns (Sep, Oct, Nov, Dec)")
    print("  ✓ Top 50 national rankings per parameter")
    print("  ✓ Composite neediness score for prioritization")


if __name__ == "__main__":
    main()
