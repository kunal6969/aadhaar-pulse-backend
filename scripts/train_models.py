"""
ML Model Training Script for Aadhaar Pulse Analytics.

This script trains all 7 ML models on the full dataset and saves them
to the trained/ directory for later use by the API.

Usage:
    python scripts/train_models.py

Models trained:
1. HierarchicalForecaster - Time series forecasting
2. CapacityPlanningModel - Operator/queue requirements
3. UnderservedScoringModel - Mobile unit placement
4. ForensicFraudDetector - Benford's Law fraud detection
5. DistrictClusteringModel - Zone segmentation
6. EWMAHotspotDetector - Biometric infrastructure hotspots
7. CohortTransitionModel - 5-year MBU prediction
"""

import os
import sys
import json
import pickle
import time
from datetime import datetime, date, timedelta
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from sqlalchemy import func, text

from app.database import SessionLocal, engine
from app.models.enrollment import Enrollment
from app.models.demographic_update import DemographicUpdate
from app.models.biometric_update import BiometricUpdate
from app.config import settings

# Import ML models
from app.ml_models import (
    HierarchicalForecaster,
    CapacityPlanningModel,
    UnderservedScoringModel,
    ForensicFraudDetector,
    DistrictClusteringModel,
    EWMAHotspotDetector,
    CohortTransitionModel
)

# Training output directory
TRAINED_DIR = Path(__file__).parent.parent / "app" / "ml_models" / "trained"
TRAINED_DIR.mkdir(parents=True, exist_ok=True)


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
        print(f"   Total: {enrollment_count + demographic_count + biometric_count:,} rows")
        
        return {
            "enrollments": enrollment_count,
            "demographics": demographic_count,
            "biometrics": biometric_count
        }
    finally:
        db.close()


def load_enrollment_data():
    """Load enrollment data as DataFrame."""
    print("📥 Loading enrollment data...")
    db = SessionLocal()
    try:
        query = db.query(
            Enrollment.date,
            Enrollment.state,
            Enrollment.district,
            Enrollment.pincode,
            Enrollment.age_0_5,
            Enrollment.age_5_17,
            Enrollment.age_18_plus,
            Enrollment.total
        ).all()
        
        df = pd.DataFrame(query, columns=[
            'date', 'state', 'district', 'pincode',
            'age_0_5', 'age_5_17', 'age_18_plus', 'total'
        ])
        df['date'] = pd.to_datetime(df['date'])
        print(f"   Loaded {len(df):,} enrollment records")
        return df
    finally:
        db.close()


def load_demographic_data():
    """Load demographic update data as DataFrame."""
    print("📥 Loading demographic update data...")
    db = SessionLocal()
    try:
        query = db.query(
            DemographicUpdate.date,
            DemographicUpdate.state,
            DemographicUpdate.district,
            DemographicUpdate.pincode,
            DemographicUpdate.demo_age_5_17,
            DemographicUpdate.demo_age_17_plus,
            DemographicUpdate.total
        ).all()
        
        df = pd.DataFrame(query, columns=[
            'date', 'state', 'district', 'pincode',
            'demo_age_5_17', 'demo_age_17_plus', 'total'
        ])
        df['date'] = pd.to_datetime(df['date'])
        print(f"   Loaded {len(df):,} demographic records")
        return df
    finally:
        db.close()


def load_biometric_data():
    """Load biometric update data as DataFrame."""
    print("📥 Loading biometric update data...")
    db = SessionLocal()
    try:
        query = db.query(
            BiometricUpdate.date,
            BiometricUpdate.state,
            BiometricUpdate.district,
            BiometricUpdate.pincode,
            BiometricUpdate.bio_age_5_17,
            BiometricUpdate.bio_age_17_plus,
            BiometricUpdate.total
        ).all()
        
        df = pd.DataFrame(query, columns=[
            'date', 'state', 'district', 'pincode',
            'bio_age_5_17', 'bio_age_17_plus', 'total'
        ])
        df['date'] = pd.to_datetime(df['date'])
        print(f"   Loaded {len(df):,} biometric records")
        return df
    finally:
        db.close()


def save_model(model, name: str, metadata: dict = None):
    """Save trained model to disk."""
    model_path = TRAINED_DIR / f"{name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"   💾 Saved: {model_path}")
    
    # Save metadata
    if metadata:
        meta_path = TRAINED_DIR / f"{name}_metadata.json"
        metadata['trained_at'] = datetime.now().isoformat()
        metadata['model_file'] = str(model_path)
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)


def train_forecaster(enrollment_df, demographic_df, biometric_df):
    """Train hierarchical time series forecaster."""
    print("\n🔮 Training Hierarchical Time Series Forecaster...")
    start = time.time()
    
    # Aggregate daily totals across all districts
    training_data = {}
    
    # Enrollment time series
    enroll_daily = enrollment_df.groupby('date')['total'].sum().reset_index()
    enroll_daily.columns = ['date', 'value']
    training_data['enrollment'] = enroll_daily
    
    # Demographic time series
    demo_daily = demographic_df.groupby('date')['total'].sum().reset_index()
    demo_daily.columns = ['date', 'value']
    training_data['demographic'] = demo_daily
    
    # Biometric time series
    bio_daily = biometric_df.groupby('date')['total'].sum().reset_index()
    bio_daily.columns = ['date', 'value']
    training_data['biometric'] = bio_daily
    
    # Train forecaster
    forecaster = HierarchicalForecaster()
    
    # Store learned parameters for each data type
    trained_params = {}
    
    for data_type, df in training_data.items():
        if len(df) > 14:  # Need at least 2 weeks of data
            df = df.sort_values('date')
            values = df['value'].values
            dates = df['date'].values
            
            # Calculate trend (linear regression)
            x = np.arange(len(values))
            if len(x) > 1:
                slope, intercept = np.polyfit(x, values, 1)
            else:
                slope, intercept = 0, values.mean() if len(values) > 0 else 0
            
            # Calculate weekly seasonality
            df['day_of_week'] = pd.to_datetime(df['date']).dt.dayofweek
            weekly_pattern = df.groupby('day_of_week')['value'].mean().to_dict()
            
            # Calculate monthly seasonality
            df['month'] = pd.to_datetime(df['date']).dt.month
            monthly_pattern = df.groupby('month')['value'].mean().to_dict()
            
            trained_params[data_type] = {
                'trend_slope': float(slope),
                'trend_intercept': float(intercept),
                'weekly_seasonality': {int(k): float(v) for k, v in weekly_pattern.items()},
                'monthly_seasonality': {int(k): float(v) for k, v in monthly_pattern.items()},
                'mean': float(values.mean()),
                'std': float(values.std()) if len(values) > 1 else 0,
                'last_date': str(df['date'].max()),
                'data_points': len(df)
            }
            print(f"   ✓ {data_type}: {len(df)} data points, trend={slope:.2f}/day")
    
    # Store trained parameters in forecaster
    forecaster.trained_params = trained_params
    forecaster.is_trained = True
    
    elapsed = time.time() - start
    save_model(forecaster, 'forecaster', {
        'data_types': list(trained_params.keys()),
        'training_time_sec': elapsed
    })
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    
    return forecaster


def train_capacity_model(enrollment_df, demographic_df, biometric_df):
    """Train capacity planning / queueing model."""
    print("\n🏢 Training Capacity Planning Model...")
    start = time.time()
    
    # Calculate district-level service metrics
    district_metrics = {}
    
    # Combine all service types
    all_districts = set(enrollment_df['district'].unique()) | \
                   set(demographic_df['district'].unique()) | \
                   set(biometric_df['district'].unique())
    
    for district in all_districts:
        # Get district data
        enroll = enrollment_df[enrollment_df['district'] == district]
        demo = demographic_df[demographic_df['district'] == district]
        bio = biometric_df[biometric_df['district'] == district]
        
        # Calculate daily arrival rates
        enroll_daily = enroll.groupby('date')['total'].sum()
        demo_daily = demo.groupby('date')['total'].sum()
        bio_daily = bio.groupby('date')['total'].sum()
        
        # Arrival rate (lambda) = mean daily arrivals
        lambda_enroll = enroll_daily.mean() if len(enroll_daily) > 0 else 0
        lambda_demo = demo_daily.mean() if len(demo_daily) > 0 else 0
        lambda_bio = bio_daily.mean() if len(bio_daily) > 0 else 0
        
        # Peak factor (ratio of max to mean)
        peak_factor = 1.0
        if len(enroll_daily) > 0 and enroll_daily.mean() > 0:
            peak_factor = max(peak_factor, enroll_daily.max() / enroll_daily.mean())
        
        district_metrics[district] = {
            'arrival_rate_enrollment': float(lambda_enroll),
            'arrival_rate_demographic': float(lambda_demo),
            'arrival_rate_biometric': float(lambda_bio),
            'total_daily_arrival': float(lambda_enroll + lambda_demo + lambda_bio),
            'peak_factor': float(peak_factor),
            'variance_enrollment': float(enroll_daily.var()) if len(enroll_daily) > 1 else 0,
            'variance_demographic': float(demo_daily.var()) if len(demo_daily) > 1 else 0,
            'variance_biometric': float(bio_daily.var()) if len(bio_daily) > 1 else 0
        }
    
    # Create and train model
    model = CapacityPlanningModel()
    model.district_metrics = district_metrics
    model.is_trained = True
    
    # Calculate global service rate estimate (mu)
    # Assume average service time of 15 minutes per transaction
    model.service_rate_per_operator = 32  # transactions per 8-hour day per operator
    
    elapsed = time.time() - start
    save_model(model, 'capacity_planning', {
        'districts_trained': len(district_metrics),
        'service_rate': model.service_rate_per_operator,
        'training_time_sec': elapsed
    })
    print(f"   ✓ Trained on {len(district_metrics)} districts")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    
    return model


def train_underserved_model(enrollment_df, demographic_df, biometric_df):
    """Train underserved area scoring model."""
    print("\n🚐 Training Underserved Scoring Model...")
    start = time.time()
    
    # Calculate district-level underserved metrics
    district_scores = {}
    
    # Aggregate by district
    enroll_by_district = enrollment_df.groupby('district').agg({
        'total': 'sum',
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'pincode': 'nunique'
    }).reset_index()
    
    demo_by_district = demographic_df.groupby('district').agg({
        'total': 'sum'
    }).reset_index()
    
    bio_by_district = biometric_df.groupby('district').agg({
        'total': 'sum'
    }).reset_index()
    
    # Merge all
    merged = enroll_by_district.merge(
        demo_by_district, on='district', how='outer', suffixes=('_enroll', '_demo')
    ).merge(
        bio_by_district, on='district', how='outer'
    )
    merged = merged.fillna(0)
    merged.columns = ['district', 'enrollments', 'child_0_5', 'child_5_17', 
                      'pincode_count', 'demographic_updates', 'biometric_updates']
    
    # Calculate underserved score (0-100)
    # Higher score = more underserved
    for _, row in merged.iterrows():
        district = row['district']
        
        # Factors contributing to "underserved" status:
        # 1. Low enrollment per pincode
        enrollment_per_pincode = row['enrollments'] / max(row['pincode_count'], 1)
        
        # 2. High child population needing future biometric updates
        child_enrollment_ratio = (row['child_0_5'] + row['child_5_17']) / max(row['enrollments'], 1)
        
        # 3. Low demographic update rate (people not updating)
        update_ratio = row['demographic_updates'] / max(row['enrollments'], 1)
        
        # 4. Low biometric update rate
        bio_ratio = row['biometric_updates'] / max(row['enrollments'], 1)
        
        district_scores[district] = {
            'enrollments': int(row['enrollments']),
            'child_enrollments': int(row['child_0_5'] + row['child_5_17']),
            'demographic_updates': int(row['demographic_updates']),
            'biometric_updates': int(row['biometric_updates']),
            'pincode_count': int(row['pincode_count']),
            'enrollment_per_pincode': float(enrollment_per_pincode),
            'child_ratio': float(child_enrollment_ratio),
            'update_activity_ratio': float(update_ratio + bio_ratio)
        }
    
    # Normalize scores
    if district_scores:
        # Get percentiles for scoring
        enrollment_values = [d['enrollment_per_pincode'] for d in district_scores.values()]
        activity_values = [d['update_activity_ratio'] for d in district_scores.values()]
        
        enroll_p25 = np.percentile(enrollment_values, 25) if enrollment_values else 0
        enroll_p75 = np.percentile(enrollment_values, 75) if enrollment_values else 1
        
        for district, metrics in district_scores.items():
            # Score based on how low enrollment activity is (inverted)
            enroll_score = 100 * (1 - min(metrics['enrollment_per_pincode'] / max(enroll_p75, 1), 1))
            
            # Score based on high child ratio (future demand)
            child_score = 100 * metrics['child_ratio']
            
            # Score based on low update activity
            activity_score = 100 * (1 - min(metrics['update_activity_ratio'] / 2, 1))
            
            # Composite score
            composite = 0.4 * enroll_score + 0.3 * child_score + 0.3 * activity_score
            metrics['underserved_score'] = float(min(max(composite, 0), 100))
    
    # Create model
    model = UnderservedScoringModel()
    model.district_scores = district_scores
    model.is_trained = True
    
    elapsed = time.time() - start
    
    # Find top underserved
    top_underserved = sorted(
        district_scores.items(), 
        key=lambda x: x[1].get('underserved_score', 0), 
        reverse=True
    )[:5]
    
    save_model(model, 'underserved_scoring', {
        'districts_scored': len(district_scores),
        'top_underserved': [d[0] for d in top_underserved],
        'training_time_sec': elapsed
    })
    print(f"   ✓ Scored {len(district_scores)} districts")
    print(f"   📍 Top underserved: {[d[0] for d in top_underserved[:3]]}")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    
    return model


def train_fraud_detector(enrollment_df, demographic_df, biometric_df):
    """Train forensic fraud detection model using Benford's Law."""
    print("\n🔍 Training Forensic Fraud Detector...")
    start = time.time()
    
    # Expected Benford's Law distribution for first digit
    benford_expected = {
        1: 0.301, 2: 0.176, 3: 0.125, 4: 0.097,
        5: 0.079, 6: 0.067, 7: 0.058, 8: 0.051, 9: 0.046
    }
    
    district_fraud_scores = {}
    
    # Analyze each district
    all_districts = set(enrollment_df['district'].unique())
    
    for district in all_districts:
        # Get all transaction values for this district
        enroll_vals = enrollment_df[enrollment_df['district'] == district]['total'].values
        demo_vals = demographic_df[demographic_df['district'] == district]['total'].values
        bio_vals = biometric_df[biometric_df['district'] == district]['total'].values
        
        all_values = np.concatenate([enroll_vals, demo_vals, bio_vals])
        all_values = all_values[all_values > 0]  # Remove zeros
        
        if len(all_values) < 50:
            continue  # Need sufficient data
        
        # First digit analysis (Benford's Law)
        first_digits = [int(str(abs(int(v)))[0]) for v in all_values if v >= 1]
        first_digit_dist = {}
        total = len(first_digits)
        for d in range(1, 10):
            first_digit_dist[d] = first_digits.count(d) / total if total > 0 else 0
        
        # Chi-square deviation from Benford
        benford_deviation = sum(
            (first_digit_dist.get(d, 0) - benford_expected[d]) ** 2 / benford_expected[d]
            for d in range(1, 10)
        )
        
        # Last digit analysis (should be uniform)
        last_digits = [int(str(abs(int(v)))[-1]) for v in all_values]
        last_digit_dist = {}
        for d in range(10):
            last_digit_dist[d] = last_digits.count(d) / len(last_digits) if last_digits else 0
        
        # Deviation from uniform (0.1 each)
        uniform_deviation = sum(
            (last_digit_dist.get(d, 0) - 0.1) ** 2
            for d in range(10)
        ) * 100
        
        # Round number analysis (ending in 0 or 5)
        round_numbers = sum(1 for v in all_values if int(v) % 5 == 0)
        round_ratio = round_numbers / len(all_values) if len(all_values) > 0 else 0
        
        # Duplicate analysis
        unique_ratio = len(set(all_values)) / len(all_values) if len(all_values) > 0 else 1
        
        # Composite fraud risk score (0-100)
        benford_score = min(benford_deviation * 10, 40)  # Max 40 points
        uniform_score = min(uniform_deviation * 20, 30)  # Max 30 points
        round_score = max(0, (round_ratio - 0.2) * 100)  # Penalty if > 20% round numbers
        dup_score = max(0, (1 - unique_ratio) * 50)  # Penalty for many duplicates
        
        fraud_risk = min(benford_score + uniform_score + round_score + dup_score, 100)
        
        district_fraud_scores[district] = {
            'benford_deviation': float(benford_deviation),
            'last_digit_deviation': float(uniform_deviation),
            'round_number_ratio': float(round_ratio),
            'unique_value_ratio': float(unique_ratio),
            'fraud_risk_score': float(fraud_risk),
            'first_digit_distribution': {int(k): float(v) for k, v in first_digit_dist.items()},
            'transactions_analyzed': int(len(all_values))
        }
    
    # Create model
    model = ForensicFraudDetector()
    model.district_scores = district_fraud_scores
    model.benford_expected = benford_expected
    model.is_trained = True
    
    elapsed = time.time() - start
    
    # Find highest risk districts
    high_risk = sorted(
        district_fraud_scores.items(),
        key=lambda x: x[1]['fraud_risk_score'],
        reverse=True
    )[:5]
    
    save_model(model, 'fraud_detector', {
        'districts_analyzed': len(district_fraud_scores),
        'high_risk_districts': [d[0] for d in high_risk],
        'training_time_sec': elapsed
    })
    print(f"   ✓ Analyzed {len(district_fraud_scores)} districts")
    high_risk_str = [f"{d[0]} ({d[1]['fraud_risk_score']:.1f})" for d in high_risk[:3]]
    print(f"   ⚠️  High risk: {high_risk_str}")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    
    return model


def train_clustering_model(enrollment_df, demographic_df, biometric_df):
    """Train district clustering/segmentation model."""
    print("\n🎯 Training District Clustering Model...")
    start = time.time()
    
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    
    # Aggregate features by district
    enroll_agg = enrollment_df.groupby('district').agg({
        'total': ['sum', 'mean', 'std'],
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'age_18_plus': 'sum'
    }).reset_index()
    enroll_agg.columns = ['district', 'enroll_total', 'enroll_mean', 'enroll_std',
                          'child_0_5', 'child_5_17', 'adult_18plus']
    
    demo_agg = demographic_df.groupby('district').agg({
        'total': ['sum', 'mean', 'std']
    }).reset_index()
    demo_agg.columns = ['district', 'demo_total', 'demo_mean', 'demo_std']
    
    bio_agg = biometric_df.groupby('district').agg({
        'total': ['sum', 'mean', 'std']
    }).reset_index()
    bio_agg.columns = ['district', 'bio_total', 'bio_mean', 'bio_std']
    
    # Merge all features
    features = enroll_agg.merge(demo_agg, on='district', how='outer') \
                         .merge(bio_agg, on='district', how='outer')
    features = features.fillna(0)
    
    districts = features['district'].values
    feature_cols = [c for c in features.columns if c != 'district']
    X = features[feature_cols].values
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Determine optimal clusters (use 5 for interpretability)
    n_clusters = min(5, len(districts))
    
    # Train KMeans
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Create segment profiles
    segment_profiles = {}
    segment_names = [
        'Stable Districts',
        'High Demographic Churn',
        'High Biometric Activity',
        'Enrollment Hotspots',
        'Low Activity Districts'
    ]
    
    for i in range(n_clusters):
        mask = cluster_labels == i
        cluster_features = X[mask]
        
        segment_profiles[i] = {
            'name': segment_names[i] if i < len(segment_names) else f'Segment {i}',
            'count': int(mask.sum()),
            'avg_enrollment': float(cluster_features[:, 0].mean()) if len(cluster_features) > 0 else 0,
            'avg_demographic': float(cluster_features[:, 6].mean()) if len(cluster_features) > 0 else 0,
            'avg_biometric': float(cluster_features[:, 9].mean()) if len(cluster_features) > 0 else 0,
        }
    
    # Create district assignments
    district_segments = {}
    for i, district in enumerate(districts):
        district_segments[district] = {
            'cluster_id': int(cluster_labels[i]),
            'segment_name': segment_profiles[cluster_labels[i]]['name'],
            'features': {col: float(X[i, j]) for j, col in enumerate(feature_cols)}
        }
    
    # Create model
    model = DistrictClusteringModel()
    model.kmeans = kmeans
    model.scaler = scaler
    model.feature_columns = feature_cols
    model.segment_profiles = segment_profiles
    model.district_segments = district_segments
    model.is_trained = True
    
    elapsed = time.time() - start
    
    save_model(model, 'clustering', {
        'n_clusters': n_clusters,
        'districts_clustered': len(districts),
        'feature_count': len(feature_cols),
        'segment_sizes': {sp['name']: sp['count'] for sp in segment_profiles.values()},
        'training_time_sec': elapsed
    })
    
    for seg_id, profile in segment_profiles.items():
        print(f"   Segment {seg_id}: {profile['name']} ({profile['count']} districts)")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    
    return model


def train_hotspot_detector(enrollment_df, demographic_df, biometric_df):
    """Train EWMA hotspot detection model."""
    print("\n🔥 Training EWMA Hotspot Detector...")
    start = time.time()
    
    # Calculate EWMA control charts for each district
    district_ewma = {}
    
    # Focus on biometric data for infrastructure demand
    for district in biometric_df['district'].unique():
        district_data = biometric_df[biometric_df['district'] == district].copy()
        district_data = district_data.sort_values('date')
        
        if len(district_data) < 7:
            continue
        
        daily = district_data.groupby('date')['total'].sum()
        values = daily.values
        
        # Calculate EWMA with lambda=0.2
        lambda_param = 0.2
        ewma = [values[0]]
        for i in range(1, len(values)):
            ewma.append(lambda_param * values[i] + (1 - lambda_param) * ewma[-1])
        ewma = np.array(ewma)
        
        # Control limits (3 sigma)
        mean = values.mean()
        std = values.std() if len(values) > 1 else 0
        sigma_ewma = std * np.sqrt(lambda_param / (2 - lambda_param))
        
        ucl = mean + 3 * sigma_ewma  # Upper control limit
        lcl = max(0, mean - 3 * sigma_ewma)  # Lower control limit
        
        # Count violations
        violations = np.sum((ewma > ucl) | (ewma < lcl))
        violation_rate = violations / len(ewma) if len(ewma) > 0 else 0
        
        # Current state
        current_ewma = ewma[-1] if len(ewma) > 0 else mean
        is_hotspot = current_ewma > ucl
        
        district_ewma[district] = {
            'mean': float(mean),
            'std': float(std),
            'ewma_current': float(current_ewma),
            'ucl': float(ucl),
            'lcl': float(lcl),
            'violation_rate': float(violation_rate),
            'is_hotspot': bool(is_hotspot),
            'trend': 'increasing' if len(ewma) > 1 and ewma[-1] > ewma[-2] else 'decreasing',
            'data_points': int(len(values))
        }
    
    # Create model
    model = EWMAHotspotDetector()
    model.district_ewma = district_ewma
    model.lambda_param = 0.2
    model.is_trained = True
    
    elapsed = time.time() - start
    
    # Count hotspots
    hotspots = [d for d, v in district_ewma.items() if v['is_hotspot']]
    
    save_model(model, 'hotspot_detector', {
        'districts_monitored': len(district_ewma),
        'current_hotspots': len(hotspots),
        'hotspot_list': hotspots[:10],
        'lambda_param': 0.2,
        'training_time_sec': elapsed
    })
    
    print(f"   ✓ Monitoring {len(district_ewma)} districts")
    print(f"   🔥 Current hotspots: {len(hotspots)}")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    
    return model


def train_cohort_model(enrollment_df, biometric_df):
    """Train cohort transition model for 5-year MBU prediction."""
    print("\n👶 Training Cohort Transition Model...")
    start = time.time()
    
    # Calculate transition matrix from child enrollments to future biometric needs
    # Children enrolled at age 0-5 will need MBU at age 5, 10, 15
    # Children enrolled at age 5-17 will need MBU at age 15
    
    # Aggregate child enrollments by district
    child_enrollments = enrollment_df.groupby('district').agg({
        'age_0_5': 'sum',
        'age_5_17': 'sum',
        'total': 'sum'
    }).reset_index()
    
    # Current biometric update rates by age group
    bio_rates = biometric_df.groupby('district').agg({
        'bio_age_5_17': 'sum',
        'bio_age_17_plus': 'sum',
        'total': 'sum'
    }).reset_index()
    
    # Merge data
    cohort_data = child_enrollments.merge(bio_rates, on='district', how='outer', suffixes=('_enroll', '_bio'))
    cohort_data = cohort_data.fillna(0)
    
    # Calculate transition probabilities
    # P(needs_mbu | enrolled_as_child) based on historical data
    district_projections = {}
    
    for _, row in cohort_data.iterrows():
        district = row['district']
        
        # Current child population
        children_0_5 = row['age_0_5']
        children_5_17 = row['age_5_17']
        
        # Biometric update rate (proxy for MBU compliance)
        current_bio_rate = row['bio_age_5_17'] / max(row['age_5_17'], 1)
        
        # 5-year projection
        # Year 1-2: Children 0-5 become 2-7 (some need first MBU)
        # Year 3-5: Children 0-5 become 5-10 (all need MBU at 5)
        
        mbu_year1 = children_0_5 * 0.1  # 10% reach age 5
        mbu_year2 = children_0_5 * 0.2
        mbu_year3 = children_0_5 * 0.2
        mbu_year4 = children_0_5 * 0.25
        mbu_year5 = children_0_5 * 0.25
        
        # Plus existing children aging into biometric requirements
        existing_mbu = children_5_17 * current_bio_rate * 0.2  # Annual rate
        
        district_projections[district] = {
            'current_children_0_5': int(children_0_5),
            'current_children_5_17': int(children_5_17),
            'current_bio_rate': float(min(current_bio_rate, 1.0)),
            'projected_mbu': {
                'year_1': int(mbu_year1 + existing_mbu),
                'year_2': int(mbu_year2 + existing_mbu),
                'year_3': int(mbu_year3 + existing_mbu),
                'year_4': int(mbu_year4 + existing_mbu),
                'year_5': int(mbu_year5 + existing_mbu)
            },
            'total_5_year_mbu': int(mbu_year1 + mbu_year2 + mbu_year3 + mbu_year4 + mbu_year5 + existing_mbu * 5)
        }
    
    # Create model
    model = CohortTransitionModel()
    model.district_projections = district_projections
    model.is_trained = True
    
    elapsed = time.time() - start
    
    # Top districts by projected MBU
    top_mbu = sorted(
        district_projections.items(),
        key=lambda x: x[1]['total_5_year_mbu'],
        reverse=True
    )[:5]
    
    save_model(model, 'cohort_model', {
        'districts_projected': len(district_projections),
        'top_mbu_districts': [d[0] for d in top_mbu],
        'projection_years': 5,
        'training_time_sec': elapsed
    })
    
    total_mbu = sum(d['total_5_year_mbu'] for d in district_projections.values())
    print(f"   ✓ Projected MBU for {len(district_projections)} districts")
    print(f"   📊 Total 5-year MBU demand: {total_mbu:,}")
    print(f"   ⏱️  Training time: {elapsed:.2f}s")
    
    return model


def main():
    """Main training function."""
    print("=" * 60)
    print("🚀 AADHAAR PULSE ML MODEL TRAINING")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Output: {TRAINED_DIR}")
    print("=" * 60)
    
    total_start = time.time()
    
    # Check database
    stats = get_db_stats()
    
    if stats['enrollments'] == 0:
        print("\n❌ ERROR: No data in database. Run data migration first.")
        print("   python scripts/init_database.py")
        return
    
    # Load data
    print("\n" + "=" * 60)
    print("📂 LOADING DATA")
    print("=" * 60)
    
    enrollment_df = load_enrollment_data()
    demographic_df = load_demographic_data()
    biometric_df = load_biometric_data()
    
    # Train all models
    print("\n" + "=" * 60)
    print("🎓 TRAINING MODELS")
    print("=" * 60)
    
    models = {}
    
    # 1. Time Series Forecaster
    models['forecaster'] = train_forecaster(enrollment_df, demographic_df, biometric_df)
    
    # 2. Capacity Planning
    models['capacity'] = train_capacity_model(enrollment_df, demographic_df, biometric_df)
    
    # 3. Underserved Scoring
    models['underserved'] = train_underserved_model(enrollment_df, demographic_df, biometric_df)
    
    # 4. Fraud Detection
    models['fraud'] = train_fraud_detector(enrollment_df, demographic_df, biometric_df)
    
    # 5. Clustering
    models['clustering'] = train_clustering_model(enrollment_df, demographic_df, biometric_df)
    
    # 6. Hotspot Detection
    models['hotspot'] = train_hotspot_detector(enrollment_df, demographic_df, biometric_df)
    
    # 7. Cohort Model
    models['cohort'] = train_cohort_model(enrollment_df, biometric_df)
    
    # Summary
    total_time = time.time() - total_start
    
    print("\n" + "=" * 60)
    print("✅ TRAINING COMPLETE")
    print("=" * 60)
    print(f"Total training time: {total_time:.2f}s")
    print(f"\nTrained models saved to: {TRAINED_DIR}")
    
    # List saved files
    print("\nSaved files:")
    for f in sorted(TRAINED_DIR.iterdir()):
        size = f.stat().st_size
        print(f"   {f.name} ({size / 1024:.1f} KB)")
    
    print("\n" + "=" * 60)
    print("🎉 All 7 ML models trained successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
