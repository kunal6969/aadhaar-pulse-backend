"""
District Clustering and Segmentation Model.

Covers:
- Infiltration zone detection using clustering + anomaly models
- User behavior segmentation for targeted interventions

Uses:
- KMeans / Gaussian Mixture Model for clustering
- Isolation Forest for anomaly detection within clusters

Segments districts into behavior-based groups for targeted interventions:
- Segment A: Stable Users (no intervention needed)
- Segment B: High Demographic Churn (migration support)
- Segment C: High Biometric Retry (device quality issues)
- Segment D: New Enrollment Hotspots (add operators)
- Segment E: Suspicious Update Pattern (fraud audit)
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
import warnings
warnings.filterwarnings('ignore')


class DistrictSegment(Enum):
    STABLE = "A_STABLE"
    HIGH_DEMOGRAPHIC_CHURN = "B_HIGH_DEMOGRAPHIC_CHURN"
    HIGH_BIOMETRIC_RETRY = "C_HIGH_BIOMETRIC_RETRY"
    ENROLLMENT_HOTSPOT = "D_ENROLLMENT_HOTSPOT"
    SUSPICIOUS_PATTERN = "E_SUSPICIOUS_PATTERN"
    UNKNOWN = "UNKNOWN"


@dataclass
class SegmentProfile:
    """Profile of a district segment."""
    segment: DistrictSegment
    segment_name: str
    description: str
    recommended_action: str
    typical_characteristics: Dict[str, str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "segment_code": self.segment.value,
            "segment_name": self.segment_name,
            "description": self.description,
            "recommended_action": self.recommended_action,
            "characteristics": self.typical_characteristics
        }


# Predefined segment profiles
SEGMENT_PROFILES = {
    DistrictSegment.STABLE: SegmentProfile(
        segment=DistrictSegment.STABLE,
        segment_name="Stable Users",
        description="Low update rate, occasional activity. Well-functioning district.",
        recommended_action="No intervention needed. Maintain standard operations.",
        typical_characteristics={
            "update_frequency": "Low",
            "enrollment_rate": "Moderate",
            "anomaly_rate": "Very Low"
        }
    ),
    DistrictSegment.HIGH_DEMOGRAPHIC_CHURN: SegmentProfile(
        segment=DistrictSegment.HIGH_DEMOGRAPHIC_CHURN,
        segment_name="High Demographic Churn",
        description="Frequent address/mobile updates indicating high migration or data quality issues.",
        recommended_action="Set up migration support camps. Improve update process with mobile units.",
        typical_characteristics={
            "demographic_updates": "Very High",
            "address_changes": "Frequent",
            "migration_indicator": "High"
        }
    ),
    DistrictSegment.HIGH_BIOMETRIC_RETRY: SegmentProfile(
        segment=DistrictSegment.HIGH_BIOMETRIC_RETRY,
        segment_name="High Biometric Retry/Update",
        description="Unusually high biometric updates suggesting device issues or authentication failures.",
        recommended_action="Check device quality. Deploy extra helpdesk support. Investigate operator training.",
        typical_characteristics={
            "biometric_updates": "Very High",
            "retry_rate": "High",
            "device_issues": "Likely"
        }
    ),
    DistrictSegment.ENROLLMENT_HOTSPOT: SegmentProfile(
        segment=DistrictSegment.ENROLLMENT_HOTSPOT,
        segment_name="New Enrollment Hotspot",
        description="Very high enrollment volume. Growing population or previously underserved area.",
        recommended_action="Add operators immediately. Deploy additional enrollment kits. Extend operating hours.",
        typical_characteristics={
            "enrollment_rate": "Very High",
            "growth_trend": "Strong Positive",
            "capacity_utilization": "Over 100%"
        }
    ),
    DistrictSegment.SUSPICIOUS_PATTERN: SegmentProfile(
        segment=DistrictSegment.SUSPICIOUS_PATTERN,
        segment_name="Suspicious Update Pattern",
        description="Too many updates combined with anomalies. Potential fraud or system abuse.",
        recommended_action="Initiate fraud audit. Implement stricter verification. Flag for investigation.",
        typical_characteristics={
            "update_frequency": "Abnormally High",
            "anomaly_score": "High",
            "pattern_irregularity": "Significant"
        }
    )
}


@dataclass
class ClusterResult:
    """Result of district clustering."""
    district: str
    state: str
    cluster_id: int
    segment: DistrictSegment
    anomaly_score: float  # -1 to 1, higher = more anomalous within cluster
    is_infiltration_zone: bool
    features: Dict[str, float]
    profile: SegmentProfile
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "cluster_id": self.cluster_id,
            "segment": self.segment.value,
            "anomaly_score": round(self.anomaly_score, 3),
            "is_infiltration_zone": self.is_infiltration_zone,
            "features": {k: round(v, 3) for k, v in self.features.items()},
            "profile": self.profile.to_dict()
        }


class DistrictClusteringModel:
    """
    Unsupervised District Clustering and Segmentation.
    
    Combines KMeans/GMM clustering with Isolation Forest for anomaly detection.
    """
    
    FEATURE_COLUMNS = [
        'enrollment_rate',
        'demographic_update_rate',
        'biometric_update_rate',
        'growth_trend',
        'volatility'
    ]
    
    def __init__(self,
                 n_clusters: int = 5,
                 use_gmm: bool = False,
                 contamination: float = 0.1):
        """
        Initialize clustering model.
        
        Args:
            n_clusters: Number of clusters (should match number of segments)
            use_gmm: Use Gaussian Mixture Model instead of KMeans
            contamination: Expected proportion of outliers for Isolation Forest
        """
        self.n_clusters = n_clusters
        self.use_gmm = use_gmm
        self.contamination = contamination
        
        self.scaler = StandardScaler()
        
        if use_gmm:
            self.clusterer = GaussianMixture(
                n_components=n_clusters,
                covariance_type='full',
                random_state=42
            )
        else:
            self.clusterer = KMeans(
                n_clusters=n_clusters,
                random_state=42,
                n_init=10
            )
        
        self.anomaly_detector = IsolationForest(
            contamination=contamination,
            random_state=42,
            n_estimators=100
        )
        
        self._fitted = False
        self._cluster_to_segment: Dict[int, DistrictSegment] = {}
    
    def fit(self, data: pd.DataFrame) -> 'DistrictClusteringModel':
        """
        Fit the clustering model on district data.
        
        Args:
            data: DataFrame with feature columns
        """
        # Prepare features
        features = self._prepare_features(data)
        
        # Scale features
        scaled_features = self.scaler.fit_transform(features)
        
        # Fit clustering
        self.clusterer.fit(scaled_features)
        
        # Fit anomaly detection
        self.anomaly_detector.fit(scaled_features)
        
        # Map clusters to segments based on centroid characteristics
        self._map_clusters_to_segments(features, scaled_features)
        
        self._fitted = True
        return self
    
    def _prepare_features(self, data: pd.DataFrame) -> np.ndarray:
        """Prepare feature matrix from data."""
        features = []
        
        for col in self.FEATURE_COLUMNS:
            if col in data.columns:
                features.append(data[col].fillna(0).values)
            else:
                # Use zeros if column missing
                features.append(np.zeros(len(data)))
        
        return np.column_stack(features)
    
    def _map_clusters_to_segments(self, features: np.ndarray, scaled_features: np.ndarray):
        """Map cluster IDs to meaningful segments based on characteristics."""
        # Get cluster assignments
        if self.use_gmm:
            labels = self.clusterer.predict(scaled_features)
        else:
            labels = self.clusterer.labels_
        
        # Calculate cluster centroids in original feature space
        cluster_profiles = {}
        for cluster_id in range(self.n_clusters):
            mask = labels == cluster_id
            if mask.sum() > 0:
                centroid = features[mask].mean(axis=0)
                cluster_profiles[cluster_id] = {
                    'enrollment_rate': centroid[0],
                    'demographic_update_rate': centroid[1],
                    'biometric_update_rate': centroid[2],
                    'growth_trend': centroid[3],
                    'volatility': centroid[4]
                }
        
        # Map to segments based on dominant characteristics
        segment_assignments = {}
        segments_used = set()
        
        for cluster_id, profile in sorted(cluster_profiles.items(), 
                                          key=lambda x: x[1]['volatility'], 
                                          reverse=True):
            segment = self._determine_segment(profile, segments_used)
            segment_assignments[cluster_id] = segment
            segments_used.add(segment)
        
        self._cluster_to_segment = segment_assignments
    
    def _determine_segment(self, profile: Dict[str, float], used: set) -> DistrictSegment:
        """Determine segment based on profile characteristics."""
        enroll = profile['enrollment_rate']
        demo = profile['demographic_update_rate']
        bio = profile['biometric_update_rate']
        growth = profile['growth_trend']
        volatility = profile['volatility']
        
        # Check for suspicious pattern first (high volatility + high updates)
        if volatility > 0.7 and (demo > 0.6 or bio > 0.6):
            if DistrictSegment.SUSPICIOUS_PATTERN not in used:
                return DistrictSegment.SUSPICIOUS_PATTERN
        
        # Enrollment hotspot
        if enroll > 0.7 and growth > 0.5:
            if DistrictSegment.ENROLLMENT_HOTSPOT not in used:
                return DistrictSegment.ENROLLMENT_HOTSPOT
        
        # High biometric retry
        if bio > 0.6:
            if DistrictSegment.HIGH_BIOMETRIC_RETRY not in used:
                return DistrictSegment.HIGH_BIOMETRIC_RETRY
        
        # High demographic churn
        if demo > 0.5:
            if DistrictSegment.HIGH_DEMOGRAPHIC_CHURN not in used:
                return DistrictSegment.HIGH_DEMOGRAPHIC_CHURN
        
        # Default to stable
        return DistrictSegment.STABLE
    
    def predict(self, data: pd.DataFrame) -> List[ClusterResult]:
        """
        Assign districts to clusters and segments.
        
        Args:
            data: DataFrame with district data and features
        
        Returns:
            List of ClusterResult objects
        """
        if not self._fitted:
            raise ValueError("Model must be fitted before prediction")
        
        features = self._prepare_features(data)
        scaled_features = self.scaler.transform(features)
        
        # Get cluster assignments
        if self.use_gmm:
            cluster_labels = self.clusterer.predict(scaled_features)
        else:
            cluster_labels = self.clusterer.predict(scaled_features)
        
        # Get anomaly scores (-1 = anomaly, 1 = normal)
        anomaly_labels = self.anomaly_detector.predict(scaled_features)
        anomaly_scores = self.anomaly_detector.decision_function(scaled_features)
        
        # Normalize anomaly scores to -1 to 1 range
        min_score = anomaly_scores.min()
        max_score = anomaly_scores.max()
        if max_score > min_score:
            normalized_scores = 2 * (anomaly_scores - min_score) / (max_score - min_score) - 1
        else:
            normalized_scores = np.zeros_like(anomaly_scores)
        
        results = []
        for i, row in data.iterrows():
            cluster_id = int(cluster_labels[i])
            segment = self._cluster_to_segment.get(cluster_id, DistrictSegment.UNKNOWN)
            profile = SEGMENT_PROFILES.get(segment, SEGMENT_PROFILES[DistrictSegment.STABLE])
            
            # Is infiltration zone? (Anomaly within suspicious segment)
            is_infiltration = (
                anomaly_labels[i] == -1 and 
                segment in [DistrictSegment.SUSPICIOUS_PATTERN, DistrictSegment.HIGH_DEMOGRAPHIC_CHURN]
            )
            
            feature_dict = {
                col: float(features[i, j]) 
                for j, col in enumerate(self.FEATURE_COLUMNS)
            }
            
            results.append(ClusterResult(
                district=row.get('district', f'District_{i}'),
                state=row.get('state', 'Unknown'),
                cluster_id=cluster_id,
                segment=segment,
                anomaly_score=float(normalized_scores[i]),
                is_infiltration_zone=is_infiltration,
                features=feature_dict,
                profile=profile
            ))
        
        return results
    
    def fit_predict(self, data: pd.DataFrame) -> List[ClusterResult]:
        """Fit model and predict in one step."""
        self.fit(data)
        return self.predict(data)
    
    def get_infiltration_zones(self, results: List[ClusterResult]) -> List[Dict[str, Any]]:
        """
        Extract districts flagged as potential infiltration zones.
        
        Infiltration zones are districts that are anomalies within 
        suspicious or high-churn segments.
        """
        zones = []
        for r in results:
            if r.is_infiltration_zone:
                zones.append({
                    "district": r.district,
                    "state": r.state,
                    "segment": r.segment.value,
                    "anomaly_score": round(r.anomaly_score, 3),
                    "reason": self._get_infiltration_reason(r),
                    "recommended_action": "Immediate investigation required. Potential irregular activity detected."
                })
        
        # Sort by anomaly score (most anomalous first)
        zones.sort(key=lambda x: x['anomaly_score'])
        
        return zones
    
    def _get_infiltration_reason(self, result: ClusterResult) -> str:
        """Generate reason for infiltration zone flag."""
        reasons = []
        
        if result.features.get('volatility', 0) > 0.7:
            reasons.append("High volatility in activity patterns")
        
        if result.features.get('demographic_update_rate', 0) > 0.6:
            reasons.append("Unusually high demographic update rate")
        
        if result.features.get('biometric_update_rate', 0) > 0.6:
            reasons.append("Unusually high biometric update rate")
        
        if result.anomaly_score < -0.5:
            reasons.append("Statistical outlier within peer group")
        
        return "; ".join(reasons) if reasons else "Multiple anomaly indicators detected"
    
    def get_segment_summary(self, results: List[ClusterResult]) -> Dict[str, Any]:
        """Get summary statistics for each segment."""
        summary = {}
        
        for segment in DistrictSegment:
            segment_results = [r for r in results if r.segment == segment]
            
            if segment_results:
                summary[segment.value] = {
                    "count": len(segment_results),
                    "profile": SEGMENT_PROFILES[segment].to_dict() if segment in SEGMENT_PROFILES else None,
                    "districts": [r.district for r in segment_results[:10]],  # Top 10
                    "avg_anomaly_score": round(
                        np.mean([r.anomaly_score for r in segment_results]), 3
                    ),
                    "infiltration_zones": sum(1 for r in segment_results if r.is_infiltration_zone)
                }
        
        return summary
    
    def get_intervention_recommendations(self, results: List[ClusterResult]) -> List[Dict[str, Any]]:
        """
        Generate intervention recommendations for each segment.
        
        Returns prioritized list of interventions.
        """
        recommendations = []
        
        # Group by segment
        segment_groups = {}
        for r in results:
            if r.segment not in segment_groups:
                segment_groups[r.segment] = []
            segment_groups[r.segment].append(r)
        
        # Priority order
        priority_order = [
            DistrictSegment.SUSPICIOUS_PATTERN,
            DistrictSegment.ENROLLMENT_HOTSPOT,
            DistrictSegment.HIGH_BIOMETRIC_RETRY,
            DistrictSegment.HIGH_DEMOGRAPHIC_CHURN,
            DistrictSegment.STABLE
        ]
        
        for priority, segment in enumerate(priority_order, 1):
            if segment in segment_groups:
                districts = segment_groups[segment]
                profile = SEGMENT_PROFILES.get(segment)
                
                if profile:
                    recommendations.append({
                        "priority": priority,
                        "segment": segment.value,
                        "segment_name": profile.segment_name,
                        "district_count": len(districts),
                        "action": profile.recommended_action,
                        "top_districts": [
                            {"district": d.district, "state": d.state}
                            for d in sorted(districts, key=lambda x: x.anomaly_score)[:5]
                        ]
                    })
        
        return recommendations
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration."""
        return {
            "model_type": "Unsupervised District Segmentation",
            "algorithms": {
                "clustering": "Gaussian Mixture Model" if self.use_gmm else "KMeans",
                "anomaly_detection": "Isolation Forest"
            },
            "parameters": {
                "n_clusters": self.n_clusters,
                "contamination": self.contamination
            },
            "features": self.FEATURE_COLUMNS,
            "segments": {
                segment.value: SEGMENT_PROFILES[segment].to_dict()
                for segment in DistrictSegment if segment in SEGMENT_PROFILES
            },
            "outputs": [
                "Cluster assignment",
                "Segment classification",
                "Anomaly score",
                "Infiltration zone flag",
                "Intervention recommendations"
            ]
        }
