"""
Cohort Transition Model for 5-Year MBU Prediction.

Covers:
- 5-year biometric demand prediction from child enrollments today
- Newborn cohort targeting
- Equipment planning recommendations

This is NOT a time series model - it's a cohort/survival-style forecasting approach.
Children enrolled in year t will need biometric update in year t+5 (at age 15).
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DemandUrgency(Enum):
    LOW = "LOW"
    MODERATE = "MODERATE"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CohortData:
    """Data for a single age cohort."""
    enrollment_year: int
    age_at_enrollment: str  # "0-5", "5-10", "10-15"
    enrollment_count: int
    expected_mbu_year: int
    years_until_mbu: int
    compliance_rate: float
    expected_mbu_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "enrollment_year": self.enrollment_year,
            "age_at_enrollment": self.age_at_enrollment,
            "enrollment_count": self.enrollment_count,
            "expected_mbu_year": self.expected_mbu_year,
            "years_until_mbu": self.years_until_mbu,
            "compliance_rate": round(self.compliance_rate, 2),
            "expected_mbu_count": self.expected_mbu_count
        }


@dataclass
class FiveYearMBUPrediction:
    """5-year MBU demand prediction for a district."""
    district: str
    state: str
    current_year: int
    yearly_predictions: List[Dict[str, Any]]
    total_5_year_demand: int
    peak_year: int
    peak_demand: int
    cumulative_by_year: List[int]
    demand_urgency: DemandUrgency
    equipment_recommendation: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "current_year": self.current_year,
            "yearly_predictions": self.yearly_predictions,
            "total_5_year_demand": self.total_5_year_demand,
            "peak_year": self.peak_year,
            "peak_demand": self.peak_demand,
            "cumulative_by_year": self.cumulative_by_year,
            "demand_urgency": self.demand_urgency.value,
            "equipment_recommendation": self.equipment_recommendation
        }


@dataclass
class EquipmentPlan:
    """Equipment planning based on MBU predictions."""
    district: str
    state: str
    current_devices: int
    peak_year_demand: int
    devices_at_peak: int
    additional_devices_needed: int
    investment_schedule: List[Dict[str, Any]]
    total_5_year_investment: float
    roi_assessment: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "current_devices": self.current_devices,
            "peak_year_demand": self.peak_year_demand,
            "devices_at_peak": self.devices_at_peak,
            "additional_devices_needed": self.additional_devices_needed,
            "investment_schedule": self.investment_schedule,
            "total_5_year_investment_lakhs": round(self.total_5_year_investment / 100000, 2),
            "roi_assessment": self.roi_assessment
        }


class CohortTransitionModel:
    """
    Cohort-based 5-Year MBU (Mandatory Biometric Update) Prediction Model.
    
    Key insight: Children enrolled at age 5-10 must do MBU at age 15.
    So enrollments today predict MBU demand 5-10 years from now.
    
    MBU_future = child_enrollment(t) × compliance_rate × transition_probability
    """
    
    # Default parameters
    DEFAULT_COMPLIANCE_RATE = 0.85  # 85% of eligible actually update
    DEFAULT_TRANSITION_PROB = 0.95  # 95% survive/stay in system
    DEVICE_CAPACITY_PER_DAY = 50  # Biometric updates per device per day
    OPERATING_DAYS_PER_YEAR = 300  # Working days
    DEVICE_COST_INR = 150000  # Cost per biometric device
    
    # Age to MBU year mapping
    AGE_TO_MBU_YEARS = {
        "0-5": 15,  # Need MBU at age 15, so 10-15 years from enrollment
        "5-10": 10,  # 5-10 years from enrollment
        "10-15": 5,  # 0-5 years from enrollment
    }
    
    def __init__(self,
                 compliance_rate: float = DEFAULT_COMPLIANCE_RATE,
                 transition_probability: float = DEFAULT_TRANSITION_PROB):
        """
        Initialize cohort model.
        
        Args:
            compliance_rate: Fraction of eligible who actually do MBU
            transition_probability: Probability of staying in system (survival)
        """
        self.compliance_rate = compliance_rate
        self.transition_prob = transition_probability
    
    def predict_mbu_demand(self,
                           district: str,
                           state: str,
                           child_enrollments_by_year: Dict[int, Dict[str, int]],
                           current_year: int = 2026) -> FiveYearMBUPrediction:
        """
        Predict 5-year MBU demand from historical child enrollments.
        
        Args:
            district: District name
            state: State name
            child_enrollments_by_year: Dict of year -> {"0-5": count, "5-10": count, "10-15": count}
            current_year: Current year for prediction start
        
        Example:
            child_enrollments_by_year = {
                2021: {"0-5": 1000, "5-10": 800, "10-15": 600},
                2022: {"0-5": 1100, "5-10": 850, "10-15": 620},
                ...
            }
        """
        yearly_demand = {year: 0 for year in range(current_year, current_year + 6)}
        cohort_details = []
        
        for enroll_year, age_groups in child_enrollments_by_year.items():
            for age_group, count in age_groups.items():
                if age_group not in self.AGE_TO_MBU_YEARS:
                    continue
                
                # Calculate when this cohort needs MBU
                years_to_mbu = self.AGE_TO_MBU_YEARS[age_group]
                mbu_year = enroll_year + years_to_mbu
                
                # Skip if MBU year is outside prediction window
                if mbu_year < current_year or mbu_year > current_year + 5:
                    continue
                
                # Calculate expected MBU count
                expected_count = int(
                    count * 
                    self.compliance_rate * 
                    self.transition_prob ** (mbu_year - enroll_year)
                )
                
                yearly_demand[mbu_year] += expected_count
                
                cohort_details.append(CohortData(
                    enrollment_year=enroll_year,
                    age_at_enrollment=age_group,
                    enrollment_count=count,
                    expected_mbu_year=mbu_year,
                    years_until_mbu=mbu_year - current_year,
                    compliance_rate=self.compliance_rate,
                    expected_mbu_count=expected_count
                ))
        
        # Build yearly predictions
        yearly_predictions = []
        cumulative = []
        running_total = 0
        
        for year in range(current_year, current_year + 6):
            demand = yearly_demand.get(year, 0)
            running_total += demand
            cumulative.append(running_total)
            
            yearly_predictions.append({
                "year": year,
                "expected_mbu_demand": demand,
                "daily_average": round(demand / self.OPERATING_DAYS_PER_YEAR, 1),
                "contributing_cohorts": [
                    c.to_dict() for c in cohort_details 
                    if c.expected_mbu_year == year
                ]
            })
        
        # Find peak
        demands = [p["expected_mbu_demand"] for p in yearly_predictions]
        peak_idx = np.argmax(demands) if demands else 0
        peak_year = current_year + peak_idx
        peak_demand = demands[peak_idx] if demands else 0
        total_demand = sum(demands)
        
        # Determine urgency
        daily_peak = peak_demand / self.OPERATING_DAYS_PER_YEAR
        urgency = self._determine_urgency(daily_peak)
        
        # Equipment recommendation
        equipment_rec = self._generate_equipment_recommendation(
            peak_demand, current_year, peak_year
        )
        
        return FiveYearMBUPrediction(
            district=district,
            state=state,
            current_year=current_year,
            yearly_predictions=yearly_predictions,
            total_5_year_demand=total_demand,
            peak_year=peak_year,
            peak_demand=peak_demand,
            cumulative_by_year=cumulative,
            demand_urgency=urgency,
            equipment_recommendation=equipment_rec
        )
    
    def _determine_urgency(self, daily_demand: float) -> DemandUrgency:
        """Determine demand urgency based on daily average."""
        if daily_demand > 200:
            return DemandUrgency.CRITICAL
        elif daily_demand > 100:
            return DemandUrgency.HIGH
        elif daily_demand > 50:
            return DemandUrgency.MODERATE
        else:
            return DemandUrgency.LOW
    
    def _generate_equipment_recommendation(self,
                                          peak_demand: int,
                                          current_year: int,
                                          peak_year: int) -> Dict[str, Any]:
        """Generate equipment planning recommendation."""
        daily_peak = peak_demand / self.OPERATING_DAYS_PER_YEAR
        devices_needed = int(np.ceil(daily_peak / self.DEVICE_CAPACITY_PER_DAY))
        
        years_to_peak = peak_year - current_year
        
        if years_to_peak <= 1:
            timeline = "IMMEDIATE"
            action = "Deploy devices within 6 months"
        elif years_to_peak <= 2:
            timeline = "SHORT_TERM"
            action = "Budget and procure within 1 year"
        elif years_to_peak <= 3:
            timeline = "MEDIUM_TERM"
            action = "Include in next annual planning cycle"
        else:
            timeline = "LONG_TERM"
            action = "Monitor and plan in 2+ year horizon"
        
        return {
            "devices_needed_at_peak": devices_needed,
            "peak_daily_demand": round(daily_peak, 0),
            "years_to_peak": years_to_peak,
            "procurement_timeline": timeline,
            "recommended_action": action,
            "estimated_investment_inr": devices_needed * self.DEVICE_COST_INR
        }
    
    def generate_equipment_plan(self,
                               predictions: List[FiveYearMBUPrediction],
                               current_devices: Dict[str, int]) -> List[EquipmentPlan]:
        """
        Generate detailed equipment plans for multiple districts.
        
        Args:
            predictions: List of 5-year predictions
            current_devices: Dict of district -> current device count
        """
        plans = []
        
        for pred in predictions:
            curr_devices = current_devices.get(pred.district, 1)
            
            # Calculate device needs per year
            investment_schedule = []
            total_investment = 0
            
            for yp in pred.yearly_predictions:
                daily_demand = yp["daily_average"]
                devices_needed = int(np.ceil(daily_demand / self.DEVICE_CAPACITY_PER_DAY))
                additional = max(0, devices_needed - curr_devices)
                
                if additional > 0:
                    investment = additional * self.DEVICE_COST_INR
                    total_investment += investment
                    investment_schedule.append({
                        "year": yp["year"],
                        "devices_to_add": additional,
                        "investment_inr": investment,
                        "cumulative_devices": curr_devices + additional
                    })
                    curr_devices += additional  # Update for next year
            
            # Peak year analysis
            peak_daily = pred.peak_demand / self.OPERATING_DAYS_PER_YEAR
            devices_at_peak = int(np.ceil(peak_daily / self.DEVICE_CAPACITY_PER_DAY))
            
            # ROI assessment
            if total_investment == 0:
                roi = "NO_INVESTMENT_NEEDED"
            elif pred.total_5_year_demand / (total_investment / self.DEVICE_COST_INR) > 3000:
                roi = "HIGH_ROI"
            elif pred.total_5_year_demand / (total_investment / self.DEVICE_COST_INR) > 1500:
                roi = "MODERATE_ROI"
            else:
                roi = "LOW_ROI"
            
            plans.append(EquipmentPlan(
                district=pred.district,
                state=pred.state,
                current_devices=current_devices.get(pred.district, 1),
                peak_year_demand=pred.peak_demand,
                devices_at_peak=devices_at_peak,
                additional_devices_needed=max(0, devices_at_peak - current_devices.get(pred.district, 1)),
                investment_schedule=investment_schedule,
                total_5_year_investment=total_investment,
                roi_assessment=roi
            ))
        
        # Sort by investment needed
        plans.sort(key=lambda x: x.additional_devices_needed, reverse=True)
        
        return plans
    
    def simulate_from_current_enrollments(self,
                                          district: str,
                                          state: str,
                                          current_enrollments: Dict[str, int],
                                          growth_rate: float = 0.05,
                                          current_year: int = 2026) -> FiveYearMBUPrediction:
        """
        Simulate 5-year prediction from current enrollment rates.
        
        Useful when only current year data is available.
        Extrapolates backwards assuming growth rate.
        
        Args:
            district: District name
            state: State name
            current_enrollments: Current year enrollments by age {"0-5": x, "5-10": y, "10-15": z}
            growth_rate: Annual enrollment growth rate
            current_year: Current year
        """
        # Generate historical enrollment estimates by projecting backwards
        historical = {}
        
        for years_back in range(10):
            year = current_year - years_back
            factor = (1 + growth_rate) ** (-years_back)
            
            historical[year] = {
                age: int(count * factor)
                for age, count in current_enrollments.items()
            }
        
        return self.predict_mbu_demand(
            district=district,
            state=state,
            child_enrollments_by_year=historical,
            current_year=current_year
        )
    
    def get_cohort_summary(self, prediction: FiveYearMBUPrediction) -> Dict[str, Any]:
        """Get summary of contributing cohorts."""
        all_cohorts = []
        for yp in prediction.yearly_predictions:
            all_cohorts.extend(yp["contributing_cohorts"])
        
        by_age_group = {}
        for cohort in all_cohorts:
            age = cohort["age_at_enrollment"]
            if age not in by_age_group:
                by_age_group[age] = {
                    "total_enrollments": 0,
                    "expected_mbu": 0,
                    "cohort_count": 0
                }
            by_age_group[age]["total_enrollments"] += cohort["enrollment_count"]
            by_age_group[age]["expected_mbu"] += cohort["expected_mbu_count"]
            by_age_group[age]["cohort_count"] += 1
        
        return {
            "district": prediction.district,
            "by_age_group": by_age_group,
            "total_contributing_cohorts": len(all_cohorts),
            "conversion_rate": round(
                prediction.total_5_year_demand / 
                sum(by_age_group[ag]["total_enrollments"] for ag in by_age_group)
                if by_age_group else 0, 2
            )
        }
    
    def identify_priority_districts(self,
                                    predictions: List[FiveYearMBUPrediction],
                                    top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Identify priority districts for MBU preparation.
        
        Sorted by total 5-year demand and urgency.
        """
        priority_list = []
        
        for pred in predictions:
            priority_list.append({
                "district": pred.district,
                "state": pred.state,
                "total_5_year_demand": pred.total_5_year_demand,
                "peak_year": pred.peak_year,
                "peak_demand": pred.peak_demand,
                "urgency": pred.demand_urgency.value,
                "equipment_needed": pred.equipment_recommendation["devices_needed_at_peak"],
                "action_timeline": pred.equipment_recommendation["procurement_timeline"]
            })
        
        # Sort by urgency then by total demand
        urgency_order = {"CRITICAL": 0, "HIGH": 1, "MODERATE": 2, "LOW": 3}
        priority_list.sort(key=lambda x: (urgency_order[x["urgency"]], -x["total_5_year_demand"]))
        
        return priority_list[:top_n]
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and methodology."""
        return {
            "model_type": "Cohort Transition / Survival-style Forecasting",
            "methodology": "MBU_future = enrollments(t) × compliance × survival_probability",
            "parameters": {
                "compliance_rate": self.compliance_rate,
                "transition_probability": self.transition_prob,
                "device_capacity_per_day": self.DEVICE_CAPACITY_PER_DAY,
                "operating_days_per_year": self.OPERATING_DAYS_PER_YEAR,
                "device_cost_inr": self.DEVICE_COST_INR
            },
            "age_cohort_mapping": {
                "0-5_enrolled": "MBU at age 15 (10-15 years later)",
                "5-10_enrolled": "MBU at age 15 (5-10 years later)",
                "10-15_enrolled": "MBU at age 15 (0-5 years later)"
            },
            "key_insight": "Children enrolled today = MBU demand in 5-15 years",
            "outputs": [
                "5-year MBU demand by year",
                "Peak year identification",
                "Equipment planning schedule",
                "Investment recommendations"
            ]
        }
