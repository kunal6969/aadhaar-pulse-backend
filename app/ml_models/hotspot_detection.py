"""
EWMA Hotspot Detection Model.

Covers:
- Biometric infrastructure demand mapping (top load districts)
- Spatio-temporal hotspot detection

Uses:
- Exponentially Weighted Moving Average (EWMA) control charts
- Rolling percentage increase analysis
- Anomaly severity classification

This is a standard operations/quality control approach used in manufacturing
and service operations for detecting process anomalies.
"""
import numpy as np
import pandas as pd
from datetime import date, timedelta
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class HotspotLevel(Enum):
    NORMAL = "NORMAL"
    ELEVATED = "ELEVATED"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"
    EMERGENCY = "EMERGENCY"


@dataclass
class EWMAControlLimits:
    """EWMA control chart limits."""
    center_line: float
    upper_warning: float
    upper_control: float
    lower_warning: float
    lower_control: float
    
    def to_dict(self) -> Dict[str, float]:
        return {
            "center_line": round(self.center_line, 2),
            "upper_warning": round(self.upper_warning, 2),
            "upper_control": round(self.upper_control, 2),
            "lower_warning": round(self.lower_warning, 2),
            "lower_control": round(self.lower_control, 2)
        }


@dataclass
class HotspotResult:
    """Hotspot detection result for a district."""
    district: str
    state: str
    current_value: float
    ewma_value: float
    control_limits: EWMAControlLimits
    deviation_sigma: float
    rolling_increase_pct: float
    hotspot_level: HotspotLevel
    consecutive_breaches: int
    infrastructure_load_pct: float
    risk_factors: List[str]
    recommendation: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "current_value": round(self.current_value, 0),
            "ewma_value": round(self.ewma_value, 2),
            "control_limits": self.control_limits.to_dict(),
            "deviation_sigma": round(self.deviation_sigma, 2),
            "rolling_increase_pct": round(self.rolling_increase_pct, 1),
            "hotspot_level": self.hotspot_level.value,
            "consecutive_breaches": self.consecutive_breaches,
            "infrastructure_load_pct": round(self.infrastructure_load_pct, 1),
            "risk_factors": self.risk_factors,
            "recommendation": self.recommendation,
            "is_hotspot": self.hotspot_level in [HotspotLevel.HIGH, HotspotLevel.CRITICAL, HotspotLevel.EMERGENCY]
        }


@dataclass
class InfrastructureDemand:
    """Infrastructure demand mapping result."""
    district: str
    state: str
    current_daily_load: float
    projected_load_7d: float
    projected_load_30d: float
    current_capacity: float
    utilization_pct: float
    device_failure_risk: str
    additional_devices_needed: int
    priority_rank: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "district": self.district,
            "state": self.state,
            "current_daily_load": round(self.current_daily_load, 0),
            "projected_load_7d": round(self.projected_load_7d, 0),
            "projected_load_30d": round(self.projected_load_30d, 0),
            "current_capacity": round(self.current_capacity, 0),
            "utilization_pct": round(self.utilization_pct, 1),
            "device_failure_risk": self.device_failure_risk,
            "additional_devices_needed": self.additional_devices_needed,
            "priority_rank": self.priority_rank
        }


class EWMAHotspotDetector:
    """
    EWMA-based Hotspot Detection for Biometric Infrastructure.
    
    Uses Exponentially Weighted Moving Average control charts to detect
    abnormal spikes in biometric update demand that could overwhelm infrastructure.
    """
    
    def __init__(self,
                 lambda_param: float = 0.2,
                 sigma_warning: float = 2.0,
                 sigma_control: float = 3.0,
                 lookback_days: int = 30):
        """
        Initialize EWMA detector.
        
        Args:
            lambda_param: EWMA smoothing parameter (0-1, lower = more smoothing)
            sigma_warning: Warning limit in standard deviations
            sigma_control: Control limit in standard deviations
            lookback_days: Days to look back for baseline calculation
        """
        self.lambda_param = lambda_param
        self.sigma_warning = sigma_warning
        self.sigma_control = sigma_control
        self.lookback_days = lookback_days
    
    def calculate_ewma(self, values: List[float]) -> Tuple[List[float], EWMAControlLimits]:
        """
        Calculate EWMA and control limits.
        
        Args:
            values: Time series of values
        
        Returns:
            Tuple of (EWMA values, Control limits)
        """
        if not values:
            return [], EWMAControlLimits(0, 0, 0, 0, 0)
        
        values = np.array(values)
        n = len(values)
        
        # Calculate baseline statistics
        mean = np.mean(values)
        std = np.std(values) if n > 1 else mean * 0.2
        
        # Calculate EWMA
        ewma = np.zeros(n)
        ewma[0] = values[0]
        
        for i in range(1, n):
            ewma[i] = self.lambda_param * values[i] + (1 - self.lambda_param) * ewma[i-1]
        
        # Calculate control limits
        # EWMA variance: σ² * (λ/(2-λ)) * (1 - (1-λ)^(2i))
        # For large i, this converges to: σ² * (λ/(2-λ))
        ewma_std = std * np.sqrt(self.lambda_param / (2 - self.lambda_param))
        
        limits = EWMAControlLimits(
            center_line=mean,
            upper_warning=mean + self.sigma_warning * ewma_std,
            upper_control=mean + self.sigma_control * ewma_std,
            lower_warning=mean - self.sigma_warning * ewma_std,
            lower_control=max(0, mean - self.sigma_control * ewma_std)
        )
        
        return ewma.tolist(), limits
    
    def detect_hotspot(self,
                       district: str,
                       state: str,
                       daily_values: List[float],
                       capacity: Optional[float] = None) -> HotspotResult:
        """
        Detect if a district is a hotspot based on EWMA analysis.
        
        Args:
            district: District name
            state: State name
            daily_values: List of daily biometric update counts
            capacity: Optional daily processing capacity
        """
        if len(daily_values) < 7:
            # Insufficient data
            return HotspotResult(
                district=district,
                state=state,
                current_value=daily_values[-1] if daily_values else 0,
                ewma_value=0,
                control_limits=EWMAControlLimits(0, 0, 0, 0, 0),
                deviation_sigma=0,
                rolling_increase_pct=0,
                hotspot_level=HotspotLevel.NORMAL,
                consecutive_breaches=0,
                infrastructure_load_pct=0,
                risk_factors=["INSUFFICIENT_DATA"],
                recommendation="Need more data for analysis"
            )
        
        # Calculate EWMA
        ewma_values, limits = self.calculate_ewma(daily_values)
        
        current_value = daily_values[-1]
        current_ewma = ewma_values[-1]
        
        # Calculate deviation in sigma units
        ewma_std = (limits.upper_control - limits.center_line) / self.sigma_control
        deviation_sigma = (current_ewma - limits.center_line) / ewma_std if ewma_std > 0 else 0
        
        # Calculate rolling increase
        if len(daily_values) >= 14:
            recent_avg = np.mean(daily_values[-7:])
            previous_avg = np.mean(daily_values[-14:-7])
            rolling_increase = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg > 0 else 0
        else:
            rolling_increase = 0
        
        # Count consecutive breaches
        consecutive_breaches = 0
        for ewma_val in reversed(ewma_values):
            if ewma_val > limits.upper_warning:
                consecutive_breaches += 1
            else:
                break
        
        # Determine hotspot level
        hotspot_level = self._determine_hotspot_level(
            deviation_sigma, consecutive_breaches, rolling_increase
        )
        
        # Calculate infrastructure load
        if capacity and capacity > 0:
            load_pct = (current_value / capacity) * 100
        else:
            # Estimate capacity as 1.5x historical average
            estimated_capacity = limits.center_line * 1.5
            load_pct = (current_value / estimated_capacity) * 100 if estimated_capacity > 0 else 100
        
        # Generate risk factors
        risk_factors = self._identify_risk_factors(
            deviation_sigma, consecutive_breaches, rolling_increase, load_pct
        )
        
        # Generate recommendation
        recommendation = self._generate_recommendation(hotspot_level, risk_factors)
        
        return HotspotResult(
            district=district,
            state=state,
            current_value=current_value,
            ewma_value=current_ewma,
            control_limits=limits,
            deviation_sigma=deviation_sigma,
            rolling_increase_pct=rolling_increase,
            hotspot_level=hotspot_level,
            consecutive_breaches=consecutive_breaches,
            infrastructure_load_pct=load_pct,
            risk_factors=risk_factors,
            recommendation=recommendation
        )
    
    def _determine_hotspot_level(self,
                                 deviation: float,
                                 consecutive: int,
                                 increase_pct: float) -> HotspotLevel:
        """Determine hotspot severity level."""
        # Emergency: Extreme deviation or sustained breach
        if deviation > 4 or (consecutive >= 5 and deviation > 3):
            return HotspotLevel.EMERGENCY
        
        # Critical: Beyond control limit
        if deviation > 3 or (consecutive >= 3 and deviation > 2):
            return HotspotLevel.CRITICAL
        
        # High: Approaching control limit
        if deviation > 2 or (consecutive >= 2 and deviation > 1.5):
            return HotspotLevel.HIGH
        
        # Elevated: Beyond warning limit
        if deviation > 1.5 or increase_pct > 50:
            return HotspotLevel.ELEVATED
        
        return HotspotLevel.NORMAL
    
    def _identify_risk_factors(self,
                               deviation: float,
                               consecutive: int,
                               increase_pct: float,
                               load_pct: float) -> List[str]:
        """Identify specific risk factors."""
        factors = []
        
        if deviation > 3:
            factors.append("EXTREME_DEVIATION")
        elif deviation > 2:
            factors.append("HIGH_DEVIATION")
        
        if consecutive >= 5:
            factors.append("SUSTAINED_BREACH_CRITICAL")
        elif consecutive >= 3:
            factors.append("SUSTAINED_BREACH")
        
        if increase_pct > 100:
            factors.append("DEMAND_SURGE_EXTREME")
        elif increase_pct > 50:
            factors.append("DEMAND_SURGE_HIGH")
        elif increase_pct > 25:
            factors.append("DEMAND_SURGE_MODERATE")
        
        if load_pct > 120:
            factors.append("CAPACITY_EXCEEDED")
        elif load_pct > 90:
            factors.append("CAPACITY_CRITICAL")
        elif load_pct > 75:
            factors.append("CAPACITY_HIGH")
        
        if not factors:
            factors.append("NORMAL_OPERATIONS")
        
        return factors
    
    def _generate_recommendation(self,
                                level: HotspotLevel,
                                risk_factors: List[str]) -> str:
        """Generate actionable recommendation."""
        if level == HotspotLevel.EMERGENCY:
            return "EMERGENCY: Deploy backup resources immediately. Consider temporary closure for capacity management. Escalate to senior management."
        
        if level == HotspotLevel.CRITICAL:
            return "CRITICAL: Add biometric devices and operators within 24 hours. Extend operating hours. Monitor queue lengths."
        
        if level == HotspotLevel.HIGH:
            return "HIGH LOAD: Plan additional capacity within 1 week. Review device maintenance schedule. Consider mobile units."
        
        if level == HotspotLevel.ELEVATED:
            return "ELEVATED: Monitor closely. Prepare contingency capacity. No immediate action required."
        
        return "NORMAL: Continue routine operations and monitoring."
    
    def map_infrastructure_demand(self,
                                  districts_data: List[Dict[str, Any]],
                                  devices_per_capacity: float = 50) -> List[InfrastructureDemand]:
        """
        Map biometric infrastructure demand across districts.
        
        Args:
            districts_data: List of dicts with 'district', 'state', 'daily_values', 'current_devices'
            devices_per_capacity: Daily processing capacity per device
        
        Returns:
            Sorted list of infrastructure demand assessments
        """
        demands = []
        
        for data in districts_data:
            daily_values = data.get('daily_values', [])
            current_devices = data.get('current_devices', 1)
            
            if not daily_values:
                continue
            
            # Current load
            current_load = daily_values[-1] if daily_values else 0
            
            # Project future load using EWMA
            if len(daily_values) >= 7:
                ewma_values, _ = self.calculate_ewma(daily_values)
                current_ewma = ewma_values[-1]
                
                # Trend calculation
                if len(daily_values) >= 14:
                    trend = (np.mean(daily_values[-7:]) - np.mean(daily_values[-14:-7])) / 7
                else:
                    trend = 0
                
                projected_7d = current_ewma + trend * 7
                projected_30d = current_ewma + trend * 30
            else:
                projected_7d = current_load
                projected_30d = current_load
            
            # Calculate capacity and utilization
            current_capacity = current_devices * devices_per_capacity
            utilization = (current_load / current_capacity * 100) if current_capacity > 0 else 100
            
            # Device failure risk based on utilization
            if utilization > 120:
                failure_risk = "CRITICAL"
            elif utilization > 100:
                failure_risk = "HIGH"
            elif utilization > 80:
                failure_risk = "MODERATE"
            else:
                failure_risk = "LOW"
            
            # Additional devices needed
            peak_projected = max(projected_7d, projected_30d)
            required_capacity = peak_projected * 1.2  # 20% buffer
            required_devices = max(0, int(np.ceil(required_capacity / devices_per_capacity)) - current_devices)
            
            demands.append(InfrastructureDemand(
                district=data['district'],
                state=data['state'],
                current_daily_load=current_load,
                projected_load_7d=max(0, projected_7d),
                projected_load_30d=max(0, projected_30d),
                current_capacity=current_capacity,
                utilization_pct=utilization,
                device_failure_risk=failure_risk,
                additional_devices_needed=required_devices,
                priority_rank=0  # Will be set after sorting
            ))
        
        # Sort by utilization (highest first) and assign priority ranks
        demands.sort(key=lambda x: x.utilization_pct, reverse=True)
        for i, d in enumerate(demands):
            d.priority_rank = i + 1
        
        return demands
    
    def get_top_load_districts(self,
                               demands: List[InfrastructureDemand],
                               top_n: int = 20) -> List[Dict[str, Any]]:
        """
        Get top load districts requiring attention.
        
        Args:
            demands: List of infrastructure demand assessments
            top_n: Number of top districts to return
        """
        top_districts = []
        
        for d in demands[:top_n]:
            top_districts.append({
                "rank": d.priority_rank,
                "district": d.district,
                "state": d.state,
                "current_load": round(d.current_daily_load, 0),
                "capacity": round(d.current_capacity, 0),
                "utilization_pct": round(d.utilization_pct, 1),
                "failure_risk": d.device_failure_risk,
                "devices_needed": d.additional_devices_needed,
                "action": self._get_load_action(d)
            })
        
        return top_districts
    
    def _get_load_action(self, demand: InfrastructureDemand) -> str:
        """Get recommended action for a district."""
        if demand.utilization_pct > 120:
            return "IMMEDIATE: Deploy backup devices. Critical capacity breach."
        elif demand.utilization_pct > 100:
            return "URGENT: Add devices within 48 hours."
        elif demand.utilization_pct > 80:
            return "PLAN: Order additional devices for deployment within 2 weeks."
        elif demand.utilization_pct > 60:
            return "MONITOR: Track utilization trend. No immediate action."
        else:
            return "NORMAL: Adequate capacity available."
    
    def get_device_failure_risk_zones(self,
                                      demands: List[InfrastructureDemand]) -> Dict[str, List[Dict[str, Any]]]:
        """
        Group districts by device failure risk.
        
        High utilization correlates with device wear and failure risk.
        """
        risk_zones = {
            "CRITICAL": [],
            "HIGH": [],
            "MODERATE": [],
            "LOW": []
        }
        
        for d in demands:
            zone_data = {
                "district": d.district,
                "state": d.state,
                "utilization_pct": round(d.utilization_pct, 1),
                "devices_needed": d.additional_devices_needed
            }
            risk_zones[d.device_failure_risk].append(zone_data)
        
        return risk_zones
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information and configuration."""
        return {
            "model_type": "EWMA Control Chart",
            "methodology": "Statistical Process Control (SPC)",
            "parameters": {
                "lambda": self.lambda_param,
                "sigma_warning": self.sigma_warning,
                "sigma_control": self.sigma_control,
                "lookback_days": self.lookback_days
            },
            "hotspot_levels": [level.value for level in HotspotLevel],
            "control_chart": {
                "center_line": "Historical mean",
                "upper_warning": f"+{self.sigma_warning}σ",
                "upper_control": f"+{self.sigma_control}σ",
                "interpretation": "Points beyond control limits indicate unusual demand"
            },
            "outputs": [
                "Hotspot detection with severity levels",
                "Infrastructure load mapping",
                "Device failure risk assessment",
                "Capacity planning recommendations"
            ]
        }
