"""
Underserved Region Scoring Model.

Covers:
- Mobile enrollment unit placement optimizer (underserved ROI)
- Newborn & child cohort targeting (equity-driven expansion)

Uses composite scoring similar to Small Area Estimation / Deprivation Index.
This is the standard policy analytics approach for equity targeting.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class InterventionType(Enum):
    MOBILE_UNIT = "mobile_unit"
    CAMP = "awareness_camp"
    EXTRA_CENTER = "additional_center"
    NO_ACTION = "no_action_needed"
    PRIORITY_TARGETING = "priority_targeting"


@dataclass
class UnderservedScore:
    """Underserved region scoring result."""
    district: str
    state: str
    overall_score: float  # 0-100, higher = more underserved
    enrollment_gap_score: float
    growth_trend_score: float
    service_density_score: float
    equity_score: float
    child_cohort_score: float
    recommended_intervention: InterventionType
    intervention_priority: int  # 1 = highest priority
    estimated_roi: float  # Expected return on investment
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "overall_score": round(self.overall_score, 1),
            "component_scores": {
                "enrollment_gap": round(self.enrollment_gap_score, 1),
                "growth_trend": round(self.growth_trend_score, 1),
                "service_density": round(self.service_density_score, 1),
                "equity": round(self.equity_score, 1),
                "child_cohort": round(self.child_cohort_score, 1)
            },
            "recommended_intervention": self.recommended_intervention.value,
            "intervention_priority": self.intervention_priority,
            "estimated_roi": round(self.estimated_roi, 2),
            "action_required": self.overall_score >= 50
        }


@dataclass
class MobileUnitPlacement:
    """Mobile unit placement recommendation."""
    district: str
    state: str
    placement_score: float
    estimated_daily_demand: float
    coverage_gap: float
    population_density_factor: float
    recommended_days_per_month: int
    expected_enrollments: int
    roi_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "placement_score": round(self.placement_score, 1),
            "estimated_daily_demand": round(self.estimated_daily_demand, 0),
            "coverage_gap_pct": round(self.coverage_gap * 100, 1),
            "population_density_factor": round(self.population_density_factor, 2),
            "recommended_days_per_month": self.recommended_days_per_month,
            "expected_monthly_enrollments": self.expected_enrollments,
            "roi_score": round(self.roi_score, 2)
        }


class UnderservedScoringModel:
    """
    Composite Scoring Model for Underserved Region Identification.
    
    Uses weighted multi-criteria scoring similar to deprivation indices.
    Scores range from 0-100 where higher = more underserved = needs more attention.
    """
    
    # Default weights for scoring components
    DEFAULT_WEIGHTS = {
        'enrollment_gap': 0.25,
        'growth_trend': 0.15,
        'service_density': 0.20,
        'equity': 0.25,
        'child_cohort': 0.15
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        """
        Initialize scoring model.
        
        Args:
            weights: Custom weights for scoring components (must sum to 1.0)
        """
        self.weights = weights or self.DEFAULT_WEIGHTS
        
        # Validate weights sum to 1
        if abs(sum(self.weights.values()) - 1.0) > 0.01:
            raise ValueError("Weights must sum to 1.0")
    
    def calculate_score(self,
                       district: str,
                       state: str,
                       total_enrollments: int,
                       estimated_population: int,
                       enrollment_growth_rate: float,
                       num_centers: int,
                       child_population_0_5: int,
                       child_enrollments_0_5: int,
                       neighbor_avg_penetration: Optional[float] = None) -> UnderservedScore:
        """
        Calculate underserved score for a district.
        
        Args:
            district: District name
            state: State name
            total_enrollments: Total Aadhaar enrollments in district
            estimated_population: Estimated population of district
            enrollment_growth_rate: Recent enrollment growth rate (%)
            num_centers: Number of enrollment centers
            child_population_0_5: Estimated child population (0-5 years)
            child_enrollments_0_5: Child enrollments (0-5 years)
            neighbor_avg_penetration: Average penetration of neighboring districts
        """
        # 1. Enrollment Gap Score (0-100)
        # Higher score = lower penetration = more underserved
        penetration = total_enrollments / estimated_population if estimated_population > 0 else 0
        enrollment_gap_score = max(0, min(100, (1 - penetration) * 100))
        
        # 2. Growth Trend Score (0-100)
        # Lower growth = more concerning = higher score
        # Negative growth = very concerning
        if enrollment_growth_rate < -5:
            growth_trend_score = 100
        elif enrollment_growth_rate < 0:
            growth_trend_score = 80
        elif enrollment_growth_rate < 5:
            growth_trend_score = 60
        elif enrollment_growth_rate < 10:
            growth_trend_score = 30
        else:
            growth_trend_score = 10
        
        # 3. Service Density Score (0-100)
        # Fewer centers per capita = higher score
        centers_per_lakh = (num_centers / estimated_population) * 100000 if estimated_population > 0 else 0
        if centers_per_lakh < 1:
            service_density_score = 100
        elif centers_per_lakh < 2:
            service_density_score = 80
        elif centers_per_lakh < 3:
            service_density_score = 50
        elif centers_per_lakh < 5:
            service_density_score = 25
        else:
            service_density_score = 10
        
        # 4. Equity Score (0-100)
        # Compare to neighbors - if below average, higher score
        if neighbor_avg_penetration is not None:
            equity_gap = neighbor_avg_penetration - penetration
            equity_score = max(0, min(100, equity_gap * 200))  # Scale appropriately
        else:
            # If no neighbor data, use enrollment gap as proxy
            equity_score = enrollment_gap_score * 0.8
        
        # 5. Child Cohort Score (0-100)
        # Lower child enrollment rate = higher priority for targeting
        child_penetration = child_enrollments_0_5 / child_population_0_5 if child_population_0_5 > 0 else 0
        child_cohort_score = max(0, min(100, (1 - child_penetration) * 120))  # Slightly weighted higher
        
        # Calculate overall weighted score
        overall_score = (
            self.weights['enrollment_gap'] * enrollment_gap_score +
            self.weights['growth_trend'] * growth_trend_score +
            self.weights['service_density'] * service_density_score +
            self.weights['equity'] * equity_score +
            self.weights['child_cohort'] * child_cohort_score
        )
        
        # Determine intervention type
        intervention = self._recommend_intervention(
            overall_score, enrollment_gap_score, service_density_score, child_cohort_score
        )
        
        # Calculate priority (1-5)
        priority = self._calculate_priority(overall_score)
        
        # Estimate ROI
        roi = self._estimate_roi(overall_score, estimated_population, penetration)
        
        return UnderservedScore(
            district=district,
            state=state,
            overall_score=overall_score,
            enrollment_gap_score=enrollment_gap_score,
            growth_trend_score=growth_trend_score,
            service_density_score=service_density_score,
            equity_score=equity_score,
            child_cohort_score=child_cohort_score,
            recommended_intervention=intervention,
            intervention_priority=priority,
            estimated_roi=roi
        )
    
    def _recommend_intervention(self,
                               overall: float,
                               enrollment_gap: float,
                               service_density: float,
                               child_cohort: float) -> InterventionType:
        """Determine recommended intervention based on scores."""
        if overall < 30:
            return InterventionType.NO_ACTION
        
        if service_density >= 80:
            # Very few centers - need permanent infrastructure
            return InterventionType.EXTRA_CENTER
        
        if enrollment_gap >= 70 and service_density >= 50:
            # Gap + density issue = mobile unit
            return InterventionType.MOBILE_UNIT
        
        if child_cohort >= 70:
            # High child gap = targeted intervention
            return InterventionType.PRIORITY_TARGETING
        
        if overall >= 50:
            # Moderate underserved = awareness camps
            return InterventionType.CAMP
        
        return InterventionType.NO_ACTION
    
    def _calculate_priority(self, overall_score: float) -> int:
        """Calculate intervention priority (1 = highest)."""
        if overall_score >= 80:
            return 1
        elif overall_score >= 65:
            return 2
        elif overall_score >= 50:
            return 3
        elif overall_score >= 35:
            return 4
        else:
            return 5
    
    def _estimate_roi(self, score: float, population: int, current_penetration: float) -> float:
        """
        Estimate ROI of intervention.
        
        Higher underserved score + larger population = higher potential ROI.
        """
        # Potential new enrollments
        potential_gap = 1 - current_penetration
        potential_enrollments = population * potential_gap
        
        # Scale by score (higher score = more likely to convert)
        conversion_factor = score / 100
        
        # ROI proxy (enrollments per unit investment)
        roi = (potential_enrollments * conversion_factor) / 10000
        
        return roi
    
    def optimize_mobile_unit_placement(self,
                                       districts_data: List[Dict[str, Any]],
                                       num_units: int = 5,
                                       days_per_month: int = 20) -> List[MobileUnitPlacement]:
        """
        Optimize mobile enrollment unit placement across districts.
        
        Args:
            districts_data: List of district data dictionaries
            num_units: Number of mobile units available
            days_per_month: Operating days per month
        
        Returns:
            List of placement recommendations
        """
        placements = []
        
        for data in districts_data:
            # Calculate placement score
            score = self.calculate_score(
                district=data['district'],
                state=data['state'],
                total_enrollments=data.get('total_enrollments', 0),
                estimated_population=data.get('population', 100000),
                enrollment_growth_rate=data.get('growth_rate', 0),
                num_centers=data.get('num_centers', 1),
                child_population_0_5=data.get('child_population', 10000),
                child_enrollments_0_5=data.get('child_enrollments', 5000)
            )
            
            # Estimate daily demand
            coverage_gap = 1 - (data.get('total_enrollments', 0) / data.get('population', 100000))
            population_factor = data.get('population', 100000) / 100000
            estimated_demand = coverage_gap * population_factor * 50  # Base: 50 per day at full gap
            
            # Calculate recommended days
            if score.overall_score >= 70:
                recommended_days = min(15, days_per_month)
            elif score.overall_score >= 50:
                recommended_days = 10
            elif score.overall_score >= 30:
                recommended_days = 5
            else:
                recommended_days = 0
            
            expected_enrollments = int(estimated_demand * recommended_days * 0.7)  # 70% efficiency
            
            # ROI score (enrollments per day of deployment)
            roi_score = expected_enrollments / max(recommended_days, 1)
            
            placements.append(MobileUnitPlacement(
                district=data['district'],
                state=data['state'],
                placement_score=score.overall_score,
                estimated_daily_demand=estimated_demand,
                coverage_gap=coverage_gap,
                population_density_factor=population_factor,
                recommended_days_per_month=recommended_days,
                expected_enrollments=expected_enrollments,
                roi_score=roi_score
            ))
        
        # Sort by ROI score (highest first)
        placements.sort(key=lambda x: x.roi_score, reverse=True)
        
        # Allocate units to top districts
        total_days = num_units * days_per_month
        allocated_days = 0
        
        for p in placements:
            if allocated_days + p.recommended_days_per_month <= total_days:
                allocated_days += p.recommended_days_per_month
            else:
                # Reduce days for this district
                remaining = total_days - allocated_days
                p.recommended_days_per_month = remaining
                p.expected_enrollments = int(p.estimated_daily_demand * remaining * 0.7)
                break
        
        return placements[:num_units * 2]  # Return top candidates
    
    def identify_child_cohort_targets(self,
                                     districts_data: List[Dict[str, Any]],
                                     top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Identify districts for newborn/child cohort targeting (equity expansion).
        
        Args:
            districts_data: List of district data dictionaries
            top_n: Number of top priority districts to return
        """
        targets = []
        
        for data in districts_data:
            child_pop = data.get('child_population_0_5', 0)
            child_enroll = data.get('child_enrollments_0_5', 0)
            
            if child_pop == 0:
                continue
            
            coverage_rate = child_enroll / child_pop
            gap = child_pop - child_enroll
            
            # Priority score
            priority_score = (1 - coverage_rate) * 50 + (gap / 1000) * 30
            
            targets.append({
                "district": data['district'],
                "state": data['state'],
                "child_population_0_5": child_pop,
                "child_enrollments_0_5": child_enroll,
                "coverage_rate": round(coverage_rate * 100, 1),
                "enrollment_gap": gap,
                "priority_score": round(priority_score, 1),
                "recommended_actions": self._get_child_targeting_actions(coverage_rate, gap)
            })
        
        # Sort by priority score
        targets.sort(key=lambda x: x['priority_score'], reverse=True)
        
        return targets[:top_n]
    
    def _get_child_targeting_actions(self, coverage_rate: float, gap: int) -> List[str]:
        """Generate recommended actions for child cohort targeting."""
        actions = []
        
        if coverage_rate < 0.3:
            actions.append("URGENT: Launch dedicated child enrollment drive")
            actions.append("Partner with hospitals for birth registration integration")
        
        if coverage_rate < 0.5:
            actions.append("Deploy school enrollment camps")
            actions.append("Coordinate with Anganwadi centers")
        
        if gap > 10000:
            actions.append("Allocate additional operators for child enrollments")
            actions.append("Set up child-friendly enrollment booths")
        
        if coverage_rate < 0.7:
            actions.append("Awareness campaign for parents about child Aadhaar benefits")
        
        if not actions:
            actions.append("Maintain current coverage with periodic monitoring")
        
        return actions
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and methodology."""
        return {
            "model_type": "Composite Deprivation Index / Small Area Estimation",
            "methodology": "Multi-criteria weighted scoring (0-100)",
            "weights": self.weights,
            "components": {
                "enrollment_gap": "Measures penetration vs population",
                "growth_trend": "Recent enrollment growth trajectory",
                "service_density": "Enrollment centers per capita",
                "equity": "Comparison with neighboring districts",
                "child_cohort": "Child (0-5) enrollment coverage"
            },
            "interventions": [
                "mobile_unit: Deploy mobile enrollment van",
                "awareness_camp: Organize enrollment awareness camps",
                "additional_center: Establish new permanent center",
                "priority_targeting: Focused outreach for specific cohorts",
                "no_action_needed: District adequately served"
            ],
            "output_range": "0-100 (higher = more underserved)"
        }
