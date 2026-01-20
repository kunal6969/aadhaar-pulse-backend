"""
Forensic Digit Fraud Detection Model.

Covers:
- Fraud suspicion scoring using digit distribution anomalies

Uses Forensic Statistics methods:
- Benford's Law (leading digit distribution)
- Last-digit uniformity test
- Too-round-number detection

This is an unsupervised fraud detection approach - no labels required.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from scipy import stats
import math


class FraudRiskLevel(Enum):
    CLEAN = "CLEAN"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class DigitDistribution:
    """Distribution of digits in a dataset."""
    leading_digits: Dict[int, float]  # 1-9 -> frequency
    last_digits: Dict[int, float]  # 0-9 -> frequency
    round_number_pct: float
    sample_size: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "leading_digits": {str(k): round(v, 4) for k, v in self.leading_digits.items()},
            "last_digits": {str(k): round(v, 4) for k, v in self.last_digits.items()},
            "round_number_pct": round(self.round_number_pct, 2),
            "sample_size": self.sample_size
        }


@dataclass
class FraudScore:
    """Fraud suspicion scoring result."""
    district: str
    state: str
    overall_fraud_score: float  # 0-1, higher = more suspicious
    benford_deviation: float
    last_digit_anomaly: float
    round_number_anomaly: float
    risk_level: FraudRiskLevel
    reason_codes: List[str]
    recommendation: str
    digit_distribution: DigitDistribution
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "overall_fraud_score": round(self.overall_fraud_score, 3),
            "component_scores": {
                "benford_deviation": round(self.benford_deviation, 3),
                "last_digit_anomaly": round(self.last_digit_anomaly, 3),
                "round_number_anomaly": round(self.round_number_anomaly, 3)
            },
            "risk_level": self.risk_level.value,
            "reason_codes": self.reason_codes,
            "recommendation": self.recommendation,
            "digit_distribution": self.digit_distribution.to_dict(),
            "requires_audit": self.risk_level in [FraudRiskLevel.HIGH, FraudRiskLevel.CRITICAL]
        }


class ForensicFraudDetector:
    """
    Forensic Statistics-based Fraud Detection.
    
    Uses digit analysis techniques commonly used in financial auditing:
    1. Benford's Law - natural datasets follow specific leading digit distribution
    2. Last Digit Uniformity - last digits should be uniformly distributed
    3. Round Number Detection - too many round numbers indicate fabrication
    """
    
    # Benford's Law expected distribution for leading digits
    BENFORD_EXPECTED = {
        1: 0.301,
        2: 0.176,
        3: 0.125,
        4: 0.097,
        5: 0.079,
        6: 0.067,
        7: 0.058,
        8: 0.051,
        9: 0.046
    }
    
    # Expected uniform distribution for last digits
    UNIFORM_EXPECTED = {i: 0.1 for i in range(10)}
    
    # Threshold for round number percentage (normal is ~10%)
    ROUND_NUMBER_THRESHOLD = 0.25
    
    def __init__(self,
                 benford_weight: float = 0.4,
                 last_digit_weight: float = 0.35,
                 round_number_weight: float = 0.25):
        """
        Initialize fraud detector.
        
        Args:
            benford_weight: Weight for Benford's Law deviation
            last_digit_weight: Weight for last digit uniformity test
            round_number_weight: Weight for round number detection
        """
        self.weights = {
            'benford': benford_weight,
            'last_digit': last_digit_weight,
            'round_number': round_number_weight
        }
        
        # Validate weights
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
    
    def analyze(self,
                district: str,
                state: str,
                values: List[int]) -> FraudScore:
        """
        Analyze a dataset for fraud indicators.
        
        Args:
            district: District name
            state: State name
            values: List of numeric values to analyze (e.g., enrollment counts)
        
        Returns:
            FraudScore with detailed analysis
        """
        # Filter valid values (positive integers)
        valid_values = [v for v in values if v > 0]
        
        if len(valid_values) < 10:
            # Not enough data for meaningful analysis
            return FraudScore(
                district=district,
                state=state,
                overall_fraud_score=0.0,
                benford_deviation=0.0,
                last_digit_anomaly=0.0,
                round_number_anomaly=0.0,
                risk_level=FraudRiskLevel.CLEAN,
                reason_codes=["INSUFFICIENT_DATA"],
                recommendation="Need more data for fraud analysis",
                digit_distribution=DigitDistribution(
                    leading_digits={}, last_digits={}, round_number_pct=0, sample_size=0
                )
            )
        
        # Extract digit distributions
        leading_dist = self._get_leading_digit_distribution(valid_values)
        last_dist = self._get_last_digit_distribution(valid_values)
        round_pct = self._get_round_number_percentage(valid_values)
        
        digit_distribution = DigitDistribution(
            leading_digits=leading_dist,
            last_digits=last_dist,
            round_number_pct=round_pct,
            sample_size=len(valid_values)
        )
        
        # Calculate anomaly scores
        benford_score = self._benford_test(leading_dist)
        last_digit_score = self._last_digit_test(last_dist)
        round_score = self._round_number_test(round_pct)
        
        # Calculate overall fraud score
        overall_score = (
            self.weights['benford'] * benford_score +
            self.weights['last_digit'] * last_digit_score +
            self.weights['round_number'] * round_score
        )
        
        # Determine risk level
        risk_level = self._determine_risk_level(overall_score)
        
        # Generate reason codes
        reason_codes = self._generate_reason_codes(
            benford_score, last_digit_score, round_score, leading_dist, last_dist
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(risk_level, reason_codes)
        
        return FraudScore(
            district=district,
            state=state,
            overall_fraud_score=overall_score,
            benford_deviation=benford_score,
            last_digit_anomaly=last_digit_score,
            round_number_anomaly=round_score,
            risk_level=risk_level,
            reason_codes=reason_codes,
            recommendation=recommendation,
            digit_distribution=digit_distribution
        )
    
    def _get_leading_digit_distribution(self, values: List[int]) -> Dict[int, float]:
        """Extract leading digit distribution."""
        leading_digits = []
        for v in values:
            first_digit = int(str(abs(v))[0])
            if 1 <= first_digit <= 9:
                leading_digits.append(first_digit)
        
        if not leading_digits:
            return {}
        
        counts = {i: 0 for i in range(1, 10)}
        for d in leading_digits:
            counts[d] += 1
        
        total = len(leading_digits)
        return {k: v / total for k, v in counts.items()}
    
    def _get_last_digit_distribution(self, values: List[int]) -> Dict[int, float]:
        """Extract last digit distribution."""
        last_digits = [abs(v) % 10 for v in values]
        
        counts = {i: 0 for i in range(10)}
        for d in last_digits:
            counts[d] += 1
        
        total = len(last_digits)
        return {k: v / total for k, v in counts.items()}
    
    def _get_round_number_percentage(self, values: List[int]) -> float:
        """Calculate percentage of round numbers (ending in 0 or 00)."""
        round_count = sum(1 for v in values if v % 10 == 0)
        return round_count / len(values) if values else 0
    
    def _benford_test(self, observed: Dict[int, float]) -> float:
        """
        Test deviation from Benford's Law using chi-squared.
        
        Returns a score 0-1 where higher = more deviation = more suspicious.
        """
        if not observed:
            return 0.0
        
        # Calculate Mean Absolute Deviation (MAD)
        mad = 0
        for digit in range(1, 10):
            expected = self.BENFORD_EXPECTED.get(digit, 0)
            actual = observed.get(digit, 0)
            mad += abs(actual - expected)
        
        mad /= 9  # Average
        
        # Convert MAD to score (0-1)
        # MAD > 0.015 is considered significant deviation
        # MAD > 0.03 is very suspicious
        score = min(1.0, mad / 0.03)
        
        return score
    
    def _last_digit_test(self, observed: Dict[int, float]) -> float:
        """
        Test deviation from uniform distribution for last digits.
        
        Returns a score 0-1 where higher = more deviation = more suspicious.
        """
        if not observed:
            return 0.0
        
        # Chi-squared test against uniform distribution
        expected = 0.1  # Each digit should appear 10% of the time
        
        chi_sq = 0
        for digit in range(10):
            actual = observed.get(digit, 0)
            chi_sq += ((actual - expected) ** 2) / expected
        
        # Convert chi-squared to score
        # Critical value for df=9 at p=0.05 is ~16.9
        # Higher chi-sq = more deviation
        score = min(1.0, chi_sq / 30)  # Normalize
        
        return score
    
    def _round_number_test(self, round_pct: float) -> float:
        """
        Test for excessive round numbers.
        
        Returns a score 0-1 where higher = more round numbers = more suspicious.
        """
        # Expected ~10% round numbers naturally
        expected = 0.10
        
        if round_pct <= expected:
            return 0.0
        
        # Score based on excess over expected
        excess = round_pct - expected
        score = min(1.0, excess / (self.ROUND_NUMBER_THRESHOLD - expected))
        
        return score
    
    def _determine_risk_level(self, overall_score: float) -> FraudRiskLevel:
        """Determine risk level from overall score."""
        if overall_score >= 0.8:
            return FraudRiskLevel.CRITICAL
        elif overall_score >= 0.6:
            return FraudRiskLevel.HIGH
        elif overall_score >= 0.4:
            return FraudRiskLevel.MEDIUM
        elif overall_score >= 0.2:
            return FraudRiskLevel.LOW
        else:
            return FraudRiskLevel.CLEAN
    
    def _generate_reason_codes(self,
                               benford: float,
                               last_digit: float,
                               round_num: float,
                               leading_dist: Dict[int, float],
                               last_dist: Dict[int, float]) -> List[str]:
        """Generate specific reason codes for detected anomalies."""
        codes = []
        
        if benford >= 0.5:
            # Find most deviant digits
            deviant = []
            for d in range(1, 10):
                expected = self.BENFORD_EXPECTED[d]
                actual = leading_dist.get(d, 0)
                if abs(actual - expected) > 0.05:
                    direction = "HIGH" if actual > expected else "LOW"
                    deviant.append(f"D{d}_{direction}")
            if deviant:
                codes.append(f"BENFORD_VIOLATION:{','.join(deviant[:3])}")
        
        if last_digit >= 0.5:
            # Find overrepresented last digits
            over_rep = [str(d) for d, freq in last_dist.items() if freq > 0.15]
            if over_rep:
                codes.append(f"LAST_DIGIT_BIAS:{','.join(over_rep)}")
        
        if round_num >= 0.5:
            codes.append("EXCESSIVE_ROUND_NUMBERS")
        
        if benford >= 0.7 and last_digit >= 0.5:
            codes.append("COMBINED_DIGIT_MANIPULATION")
        
        if not codes:
            codes.append("NO_ANOMALIES_DETECTED")
        
        return codes
    
    def _generate_recommendation(self, risk_level: FraudRiskLevel, reason_codes: List[str]) -> str:
        """Generate actionable recommendation."""
        if risk_level == FraudRiskLevel.CRITICAL:
            return "IMMEDIATE AUDIT REQUIRED: Multiple strong fraud indicators detected. Suspend operations and conduct forensic audit."
        elif risk_level == FraudRiskLevel.HIGH:
            return "URGENT REVIEW: Significant anomalies detected. Schedule audit within 30 days and investigate specific operators."
        elif risk_level == FraudRiskLevel.MEDIUM:
            return "MONITOR CLOSELY: Some statistical anomalies present. Increase reporting frequency and conduct spot checks."
        elif risk_level == FraudRiskLevel.LOW:
            return "ROUTINE MONITORING: Minor deviations observed. Continue standard monitoring procedures."
        else:
            return "NO ACTION NEEDED: Data patterns appear natural. Continue routine operations."
    
    def batch_analyze(self, 
                      data: List[Dict[str, Any]],
                      value_column: str = 'values') -> List[FraudScore]:
        """
        Analyze multiple districts for fraud.
        
        Args:
            data: List of dictionaries with 'district', 'state', and value data
            value_column: Name of column containing values to analyze
        
        Returns:
            List of FraudScore objects, sorted by fraud score (highest first)
        """
        results = []
        
        for item in data:
            values = item.get(value_column, [])
            if isinstance(values, (int, float)):
                values = [int(values)]
            
            score = self.analyze(
                district=item.get('district', 'Unknown'),
                state=item.get('state', 'Unknown'),
                values=values
            )
            results.append(score)
        
        # Sort by fraud score descending
        results.sort(key=lambda x: x.overall_fraud_score, reverse=True)
        
        return results
    
    def get_districts_for_audit(self, 
                                scores: List[FraudScore],
                                min_risk_level: FraudRiskLevel = FraudRiskLevel.HIGH) -> List[Dict[str, Any]]:
        """
        Get list of districts requiring audit.
        
        Args:
            scores: List of fraud scores
            min_risk_level: Minimum risk level to include
        
        Returns:
            List of districts requiring audit with details
        """
        risk_order = [
            FraudRiskLevel.CRITICAL,
            FraudRiskLevel.HIGH,
            FraudRiskLevel.MEDIUM,
            FraudRiskLevel.LOW,
            FraudRiskLevel.CLEAN
        ]
        
        min_idx = risk_order.index(min_risk_level)
        
        audit_list = []
        for score in scores:
            if risk_order.index(score.risk_level) <= min_idx:
                audit_list.append({
                    "district": score.district,
                    "state": score.state,
                    "risk_level": score.risk_level.value,
                    "fraud_score": round(score.overall_fraud_score, 3),
                    "reason_codes": score.reason_codes,
                    "recommendation": score.recommendation,
                    "priority": "P1" if score.risk_level == FraudRiskLevel.CRITICAL else "P2"
                })
        
        return audit_list
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and methodology."""
        return {
            "model_type": "Forensic Statistics - Digit Analysis",
            "methodology": "Unsupervised fraud detection (no labels required)",
            "techniques": {
                "benfords_law": {
                    "description": "Tests if leading digits follow natural distribution",
                    "expected": self.BENFORD_EXPECTED,
                    "weight": self.weights['benford']
                },
                "last_digit_uniformity": {
                    "description": "Tests if last digits are uniformly distributed",
                    "expected": "10% each digit",
                    "weight": self.weights['last_digit']
                },
                "round_number_detection": {
                    "description": "Detects excessive round numbers indicating fabrication",
                    "threshold": f"{self.ROUND_NUMBER_THRESHOLD * 100}%",
                    "weight": self.weights['round_number']
                }
            },
            "risk_levels": [level.value for level in FraudRiskLevel],
            "output_range": "0-1 (higher = more suspicious)",
            "references": [
                "Benford, F. (1938). The Law of Anomalous Numbers",
                "Nigrini, M. (2012). Benford's Law: Applications for Forensic Accounting"
            ]
        }
