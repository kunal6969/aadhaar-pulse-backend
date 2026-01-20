"""
Queueing Theory Capacity Planning Model.

Covers:
- District-wise operator requirement (next year planning)
- Weekend vs weekday service gap correction
- Wait time and SLA risk estimation

Uses Little's Law (L = λW) for capacity estimation.
This is the standard operations research approach used in government planning.
"""
import numpy as np
from datetime import date, timedelta
from typing import Dict, Any, List, Optional
from dataclasses import dataclass
from enum import Enum


class RiskLevel(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


@dataclass
class CapacityRequirement:
    """Capacity requirement calculation result."""
    district: str
    state: str
    forecasted_demand_daily: float  # λ (requests/day)
    service_rate_per_operator: float  # μ (requests/operator/day)
    operators_required: int
    current_operators: Optional[int]
    utilization_rate: float  # ρ = λ / (n * μ)
    expected_wait_time_hours: float
    expected_queue_length: float
    sla_risk_level: RiskLevel
    sla_risk_score: float
    weekday_operators: int
    weekend_operators: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "forecasted_demand_daily": round(self.forecasted_demand_daily, 0),
            "service_rate_per_operator": round(self.service_rate_per_operator, 1),
            "operators_required": self.operators_required,
            "current_operators": self.current_operators,
            "operator_gap": (self.operators_required - self.current_operators) if self.current_operators else None,
            "utilization_rate": round(self.utilization_rate, 2),
            "expected_wait_time_hours": round(self.expected_wait_time_hours, 2),
            "expected_queue_length": round(self.expected_queue_length, 1),
            "sla_risk_level": self.sla_risk_level.value,
            "sla_risk_score": round(self.sla_risk_score, 2),
            "weekday_operators": self.weekday_operators,
            "weekend_operators": self.weekend_operators
        }


@dataclass
class YearlyCapacityPlan:
    """Yearly capacity planning result."""
    district: str
    state: str
    monthly_requirements: List[Dict[str, Any]]
    annual_peak_operators: int
    annual_avg_operators: int
    recommended_permanent_staff: int
    recommended_seasonal_staff: int
    total_annual_demand: int
    capacity_utilization_yearly: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "monthly_requirements": self.monthly_requirements,
            "annual_peak_operators": self.annual_peak_operators,
            "annual_avg_operators": self.annual_avg_operators,
            "recommended_permanent_staff": self.recommended_permanent_staff,
            "recommended_seasonal_staff": self.recommended_seasonal_staff,
            "total_annual_demand": self.total_annual_demand,
            "capacity_utilization_yearly": round(self.capacity_utilization_yearly, 2)
        }


class CapacityPlanningModel:
    """
    Queueing Theory based Capacity Planning Model.
    
    Uses M/M/c queueing model for capacity estimation.
    - M: Poisson arrival (random citizen arrivals)
    - M: Exponential service time
    - c: Multiple servers (operators)
    """
    
    # Default operational parameters
    DEFAULT_SERVICE_RATE = 40  # requests per operator per day (8 hours, 12 min each)
    DEFAULT_OPERATING_HOURS = 8
    DEFAULT_TARGET_WAIT_TIME = 0.5  # Target: 30 minutes max wait
    DEFAULT_TARGET_UTILIZATION = 0.85  # 85% utilization target
    
    def __init__(self,
                 service_rate_per_operator: float = DEFAULT_SERVICE_RATE,
                 operating_hours: float = DEFAULT_OPERATING_HOURS,
                 target_wait_hours: float = DEFAULT_TARGET_WAIT_TIME,
                 target_utilization: float = DEFAULT_TARGET_UTILIZATION):
        """
        Initialize capacity planning model.
        
        Args:
            service_rate_per_operator: Number of requests one operator can handle per day
            operating_hours: Daily operating hours
            target_wait_hours: Target maximum wait time in hours
            target_utilization: Target utilization rate (0-1)
        """
        self.service_rate = service_rate_per_operator
        self.operating_hours = operating_hours
        self.target_wait = target_wait_hours
        self.target_utilization = target_utilization
    
    def calculate_requirements(self,
                              district: str,
                              state: str,
                              forecasted_demand: float,
                              weekend_factor: float = 0.7,
                              current_operators: Optional[int] = None) -> CapacityRequirement:
        """
        Calculate operator requirements using queueing theory.
        
        Uses Little's Law: L = λW
        And M/M/c queue formulas for wait time estimation.
        
        Args:
            district: District name
            state: State name
            forecasted_demand: Forecasted daily demand (λ)
            weekend_factor: Relative demand on weekends vs weekdays
            current_operators: Current number of operators (if known)
        """
        λ = forecasted_demand  # Arrival rate
        μ = self.service_rate  # Service rate per operator
        
        # Calculate minimum operators needed for stability (ρ < 1)
        min_operators = max(1, int(np.ceil(λ / μ)))
        
        # Calculate operators for target utilization
        operators_for_utilization = max(1, int(np.ceil(λ / (μ * self.target_utilization))))
        
        # Use the higher of the two
        c = max(min_operators, operators_for_utilization)
        
        # Calculate utilization ρ = λ / (c * μ)
        utilization = λ / (c * μ) if c > 0 else 1.0
        
        # M/M/c queue wait time estimation
        # Simplified Erlang-C approximation
        wait_time_hours = self._estimate_wait_time(λ, μ, c)
        
        # Adjust operators if wait time exceeds target
        while wait_time_hours > self.target_wait and c < 100:
            c += 1
            wait_time_hours = self._estimate_wait_time(λ, μ, c)
            utilization = λ / (c * μ)
        
        # Expected queue length (Little's Law: L = λW)
        queue_length = λ * (wait_time_hours / self.operating_hours) if self.operating_hours > 0 else 0
        
        # SLA risk calculation
        sla_risk_score, sla_risk_level = self._calculate_sla_risk(
            utilization, wait_time_hours, current_operators, c
        )
        
        # Weekend vs weekday staffing
        weekend_demand = forecasted_demand * weekend_factor
        weekend_operators = max(1, int(np.ceil(weekend_demand / (μ * self.target_utilization))))
        
        return CapacityRequirement(
            district=district,
            state=state,
            forecasted_demand_daily=forecasted_demand,
            service_rate_per_operator=self.service_rate,
            operators_required=c,
            current_operators=current_operators,
            utilization_rate=utilization,
            expected_wait_time_hours=wait_time_hours,
            expected_queue_length=queue_length,
            sla_risk_level=sla_risk_level,
            sla_risk_score=sla_risk_score,
            weekday_operators=c,
            weekend_operators=weekend_operators
        )
    
    def _estimate_wait_time(self, λ: float, μ: float, c: int) -> float:
        """
        Estimate wait time using M/M/c queue approximation.
        
        Uses simplified Erlang-C formula.
        """
        if c == 0 or μ == 0:
            return float('inf')
        
        ρ = λ / (c * μ)
        
        if ρ >= 1:
            return float('inf')  # Unstable queue
        
        # Simplified wait time: W_q ≈ ρ^(sqrt(2*(c+1))-1) / (c*μ*(1-ρ))
        # This is an approximation; exact formula requires Erlang-C calculation
        try:
            # Average time in queue (simplified)
            W_q = (ρ ** np.sqrt(2 * c)) / (c * μ * (1 - ρ))
            # Convert to hours
            wait_hours = W_q * self.operating_hours
            return min(wait_hours, 8.0)  # Cap at 8 hours
        except:
            return 8.0
    
    def _calculate_sla_risk(self,
                           utilization: float,
                           wait_hours: float,
                           current_operators: Optional[int],
                           required_operators: int) -> tuple:
        """Calculate SLA risk score and level."""
        risk_score = 0.0
        
        # Utilization risk (weight: 0.3)
        if utilization > 0.95:
            risk_score += 0.3
        elif utilization > 0.85:
            risk_score += 0.15
        elif utilization > 0.75:
            risk_score += 0.05
        
        # Wait time risk (weight: 0.4)
        if wait_hours > 2.0:
            risk_score += 0.4
        elif wait_hours > 1.0:
            risk_score += 0.2
        elif wait_hours > 0.5:
            risk_score += 0.1
        
        # Operator gap risk (weight: 0.3)
        if current_operators is not None:
            gap_ratio = (required_operators - current_operators) / max(required_operators, 1)
            if gap_ratio > 0.5:
                risk_score += 0.3
            elif gap_ratio > 0.25:
                risk_score += 0.15
            elif gap_ratio > 0:
                risk_score += 0.05
        
        # Determine risk level
        if risk_score >= 0.7:
            risk_level = RiskLevel.CRITICAL
        elif risk_score >= 0.5:
            risk_level = RiskLevel.HIGH
        elif risk_score >= 0.25:
            risk_level = RiskLevel.MEDIUM
        else:
            risk_level = RiskLevel.LOW
        
        return risk_score, risk_level
    
    def generate_yearly_plan(self,
                            district: str,
                            state: str,
                            monthly_demands: List[float]) -> YearlyCapacityPlan:
        """
        Generate yearly capacity plan from monthly demand forecasts.
        
        Args:
            district: District name
            state: State name
            monthly_demands: List of 12 monthly average daily demands
        """
        if len(monthly_demands) != 12:
            raise ValueError("monthly_demands must have exactly 12 values")
        
        monthly_requirements = []
        operators_per_month = []
        month_names = ['January', 'February', 'March', 'April', 'May', 'June',
                      'July', 'August', 'September', 'October', 'November', 'December']
        
        for i, demand in enumerate(monthly_demands):
            req = self.calculate_requirements(district, state, demand)
            operators_per_month.append(req.operators_required)
            monthly_requirements.append({
                "month": month_names[i],
                "average_daily_demand": round(demand, 0),
                "operators_required": req.operators_required,
                "utilization": round(req.utilization_rate, 2),
                "wait_time_hours": round(req.expected_wait_time_hours, 2)
            })
        
        peak_operators = max(operators_per_month)
        avg_operators = int(np.mean(operators_per_month))
        
        # Recommendation: permanent staff at 75th percentile, seasonal for peaks
        permanent_staff = int(np.percentile(operators_per_month, 75))
        seasonal_staff = max(0, peak_operators - permanent_staff)
        
        # Total annual demand
        total_demand = int(sum(monthly_demands) * 30)  # Approximate monthly days
        
        # Yearly utilization
        yearly_capacity = permanent_staff * self.service_rate * 365
        yearly_utilization = total_demand / yearly_capacity if yearly_capacity > 0 else 1.0
        
        return YearlyCapacityPlan(
            district=district,
            state=state,
            monthly_requirements=monthly_requirements,
            annual_peak_operators=peak_operators,
            annual_avg_operators=avg_operators,
            recommended_permanent_staff=permanent_staff,
            recommended_seasonal_staff=seasonal_staff,
            total_annual_demand=total_demand,
            capacity_utilization_yearly=yearly_utilization
        )
    
    def optimize_weekend_staffing(self,
                                 weekday_demand: float,
                                 weekend_demand: float) -> Dict[str, Any]:
        """
        Optimize staffing for weekday vs weekend operations.
        
        Returns recommendations for flexible staffing.
        """
        weekday_req = self.calculate_requirements("", "", weekday_demand)
        weekend_req = self.calculate_requirements("", "", weekend_demand)
        
        staff_reduction = weekday_req.operators_required - weekend_req.operators_required
        cost_savings_pct = (staff_reduction / weekday_req.operators_required * 100) if weekday_req.operators_required > 0 else 0
        
        return {
            "weekday_staff": weekday_req.operators_required,
            "weekend_staff": weekend_req.operators_required,
            "staff_reduction_weekend": staff_reduction,
            "potential_cost_savings_pct": round(cost_savings_pct, 1),
            "recommendation": self._get_weekend_recommendation(staff_reduction, weekend_demand)
        }
    
    def _get_weekend_recommendation(self, reduction: int, weekend_demand: float) -> str:
        if reduction >= 3:
            return f"Significant weekend reduction possible. Consider rotating {reduction} operators to maintenance/training on weekends."
        elif reduction >= 1:
            return f"Moderate weekend reduction. {reduction} fewer operators needed."
        elif weekend_demand < 10:
            return "Very low weekend demand. Consider weekend closure with mobile services for emergencies."
        else:
            return "Weekend demand similar to weekdays. Maintain consistent staffing."
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and methodology info."""
        return {
            "model_type": "M/M/c Queueing Model (Little's Law)",
            "methodology": "Operations Research - Erlang-C Approximation",
            "parameters": {
                "service_rate_per_operator_per_day": self.service_rate,
                "operating_hours": self.operating_hours,
                "target_wait_time_hours": self.target_wait,
                "target_utilization": self.target_utilization
            },
            "assumptions": [
                "Poisson arrival process (random citizen arrivals)",
                "Exponential service time distribution",
                "First-come-first-served queue discipline",
                "No customer abandonment"
            ],
            "outputs": [
                "Operators required per district",
                "Expected wait time",
                "SLA risk assessment",
                "Weekend vs weekday staffing"
            ]
        }
