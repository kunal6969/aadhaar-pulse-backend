"""
Hierarchical Time Series Forecasting Model.

Covers:
- Operator requirement forecasting
- Mobile unit placement (demand gaps)
- Seasonality calendar
- Biometric infrastructure demand mapping

Uses trend + seasonality decomposition with hierarchical reconciliation.
Can be upgraded to Prophet/SARIMA/XGBoost for production.
"""
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class DataType(Enum):
    ENROLLMENT = "enrollment"
    DEMOGRAPHIC = "demographic"
    BIOMETRIC = "biometric"


@dataclass
class ForecastResult:
    """Result of a forecast prediction."""
    date: date
    predicted: float
    lower_bound: float
    upper_bound: float
    trend_component: float
    seasonal_component: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "predicted": round(self.predicted, 0),
            "lower_bound": round(self.lower_bound, 0),
            "upper_bound": round(self.upper_bound, 0),
            "trend": round(self.trend_component, 2),
            "seasonal": round(self.seasonal_component, 2)
        }


@dataclass
class SeasonalityPattern:
    """Detected seasonality patterns."""
    day_of_week_effects: Dict[int, float]  # 0=Monday, 6=Sunday
    monthly_effects: Dict[int, float]  # 1-12
    quarterly_effects: Dict[int, float]  # 1-4
    weekend_factor: float
    
    def to_dict(self) -> Dict[str, Any]:
        day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
        return {
            "day_of_week": {day_names[k]: round(v, 3) for k, v in self.day_of_week_effects.items()},
            "monthly": {f"Month_{k}": round(v, 3) for k, v in self.monthly_effects.items()},
            "quarterly": {f"Q{k}": round(v, 3) for k, v in self.quarterly_effects.items()},
            "weekend_factor": round(self.weekend_factor, 3)
        }


class HierarchicalForecaster:
    """
    Hierarchical Time Series Forecasting for Aadhaar data.
    
    Hierarchy: India → State → District
    Ensures forecasts are consistent across levels.
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """
        Initialize forecaster.
        
        Args:
            confidence_level: Confidence level for prediction intervals (default 95%)
        """
        self.confidence_level = confidence_level
        self.z_score = 1.96 if confidence_level == 0.95 else 2.576  # 95% or 99%
        self._trained = False
        self._model_params: Dict[str, Any] = {}
        
    def fit(self, 
            historical_df: pd.DataFrame, 
            date_col: str = 'date',
            value_col: str = 'total',
            group_cols: Optional[List[str]] = None) -> 'HierarchicalForecaster':
        """
        Fit the forecasting model on historical data.
        
        Args:
            historical_df: DataFrame with historical data
            date_col: Name of date column
            value_col: Name of value column to forecast
            group_cols: Columns for hierarchical grouping (e.g., ['state', 'district'])
        """
        df = historical_df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
        
        # Calculate global statistics
        self._model_params['global_mean'] = df[value_col].mean()
        self._model_params['global_std'] = df[value_col].std()
        
        # Calculate trend using linear regression
        df['days'] = (df[date_col] - df[date_col].min()).dt.days
        if len(df) > 1:
            trend = np.polyfit(df['days'].values, df[value_col].values, 1)
            self._model_params['trend_slope'] = trend[0]
            self._model_params['trend_intercept'] = trend[1]
        else:
            self._model_params['trend_slope'] = 0
            self._model_params['trend_intercept'] = self._model_params['global_mean']
        
        # Calculate seasonality patterns
        self._model_params['seasonality'] = self._extract_seasonality(df, date_col, value_col)
        
        # Store group-level parameters if hierarchical
        if group_cols:
            self._model_params['group_params'] = {}
            for name, group in df.groupby(group_cols):
                key = '|'.join(name) if isinstance(name, tuple) else str(name)
                group_mean = group[value_col].mean()
                group_std = group[value_col].std() if len(group) > 1 else group_mean * 0.2
                self._model_params['group_params'][key] = {
                    'mean': group_mean,
                    'std': group_std,
                    'count': len(group)
                }
        
        self._trained = True
        return self
    
    def _extract_seasonality(self, df: pd.DataFrame, date_col: str, value_col: str) -> SeasonalityPattern:
        """Extract seasonality patterns from data."""
        df = df.copy()
        df['day_of_week'] = pd.to_datetime(df[date_col]).dt.dayofweek
        df['month'] = pd.to_datetime(df[date_col]).dt.month
        df['quarter'] = pd.to_datetime(df[date_col]).dt.quarter
        
        mean_val = df[value_col].mean()
        
        # Day of week effects (relative to mean)
        dow_effects = {}
        for dow in range(7):
            dow_mean = df[df['day_of_week'] == dow][value_col].mean()
            dow_effects[dow] = dow_mean / mean_val if mean_val > 0 else 1.0
        
        # Monthly effects
        monthly_effects = {}
        for month in range(1, 13):
            month_data = df[df['month'] == month][value_col]
            if len(month_data) > 0:
                monthly_effects[month] = month_data.mean() / mean_val if mean_val > 0 else 1.0
            else:
                monthly_effects[month] = 1.0
        
        # Quarterly effects
        quarterly_effects = {}
        for q in range(1, 5):
            q_data = df[df['quarter'] == q][value_col]
            if len(q_data) > 0:
                quarterly_effects[q] = q_data.mean() / mean_val if mean_val > 0 else 1.0
            else:
                quarterly_effects[q] = 1.0
        
        # Weekend factor
        weekend_data = df[df['day_of_week'].isin([5, 6])][value_col]
        weekday_data = df[~df['day_of_week'].isin([5, 6])][value_col]
        weekend_factor = weekend_data.mean() / weekday_data.mean() if len(weekday_data) > 0 and weekday_data.mean() > 0 else 0.7
        
        return SeasonalityPattern(
            day_of_week_effects=dow_effects,
            monthly_effects=monthly_effects,
            quarterly_effects=quarterly_effects,
            weekend_factor=weekend_factor
        )
    
    def predict(self, 
                forecast_from: date, 
                horizon_days: int = 90,
                group_key: Optional[str] = None) -> List[ForecastResult]:
        """
        Generate forecasts for future dates.
        
        Args:
            forecast_from: Start date for forecast
            horizon_days: Number of days to forecast
            group_key: Optional group key for hierarchical forecast
        """
        if not self._trained:
            raise ValueError("Model must be fitted before prediction")
        
        # Get relevant parameters
        if group_key and 'group_params' in self._model_params and group_key in self._model_params['group_params']:
            params = self._model_params['group_params'][group_key]
            base_mean = params['mean']
            std = params['std']
        else:
            base_mean = self._model_params['global_mean']
            std = self._model_params['global_std']
        
        trend_slope = self._model_params['trend_slope']
        seasonality = self._model_params['seasonality']
        
        forecasts = []
        for i in range(horizon_days):
            forecast_date = forecast_from + timedelta(days=i + 1)
            
            # Trend component
            trend_component = trend_slope * i
            
            # Seasonal component
            dow = forecast_date.weekday()
            month = forecast_date.month
            quarter = (month - 1) // 3 + 1
            
            dow_effect = seasonality.day_of_week_effects.get(dow, 1.0)
            monthly_effect = seasonality.monthly_effects.get(month, 1.0)
            
            seasonal_multiplier = (dow_effect + monthly_effect) / 2
            seasonal_component = base_mean * (seasonal_multiplier - 1)
            
            # Final prediction
            predicted = base_mean + trend_component + seasonal_component
            predicted = max(0, predicted)  # Ensure non-negative
            
            # Confidence bounds
            lower = max(0, predicted - self.z_score * std)
            upper = predicted + self.z_score * std
            
            forecasts.append(ForecastResult(
                date=forecast_date,
                predicted=predicted,
                lower_bound=lower,
                upper_bound=upper,
                trend_component=trend_component,
                seasonal_component=seasonal_component
            ))
        
        return forecasts
    
    def get_seasonality_calendar(self) -> Dict[str, Any]:
        """
        Generate a seasonality calendar showing expected patterns.
        
        Returns a calendar with relative activity levels for planning.
        """
        if not self._trained:
            raise ValueError("Model must be fitted first")
        
        seasonality = self._model_params['seasonality']
        
        return {
            "day_of_week_pattern": seasonality.to_dict()['day_of_week'],
            "monthly_pattern": seasonality.to_dict()['monthly'],
            "quarterly_pattern": seasonality.to_dict()['quarterly'],
            "weekend_reduction": f"{(1 - seasonality.weekend_factor) * 100:.1f}%",
            "peak_days": [k for k, v in sorted(seasonality.day_of_week_effects.items(), 
                                               key=lambda x: x[1], reverse=True)[:3]],
            "low_days": [k for k, v in sorted(seasonality.day_of_week_effects.items(), 
                                              key=lambda x: x[1])[:2]],
            "recommendations": self._generate_seasonality_recommendations(seasonality)
        }
    
    def _generate_seasonality_recommendations(self, seasonality: SeasonalityPattern) -> List[str]:
        """Generate actionable recommendations from seasonality patterns."""
        recommendations = []
        
        # Weekend recommendation
        if seasonality.weekend_factor < 0.7:
            recommendations.append(
                f"Weekend activity is {(1-seasonality.weekend_factor)*100:.0f}% lower. "
                "Consider reduced staffing on weekends or weekend awareness campaigns."
            )
        
        # Monthly patterns
        high_months = [k for k, v in seasonality.monthly_effects.items() if v > 1.1]
        low_months = [k for k, v in seasonality.monthly_effects.items() if v < 0.9]
        
        if high_months:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            recommendations.append(
                f"Peak months: {', '.join([month_names[m-1] for m in high_months])}. "
                "Plan additional capacity during these periods."
            )
        
        if low_months:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            recommendations.append(
                f"Low activity months: {', '.join([month_names[m-1] for m in low_months])}. "
                "Consider maintenance and training during these periods."
            )
        
        return recommendations
    
    def hierarchical_reconcile(self, 
                               district_forecasts: Dict[str, List[ForecastResult]],
                               state_forecast: List[ForecastResult]) -> Dict[str, List[ForecastResult]]:
        """
        Reconcile district forecasts to match state-level forecast.
        
        Uses proportional scaling to ensure sum(district) = state.
        """
        reconciled = {}
        
        for i in range(len(state_forecast)):
            state_pred = state_forecast[i].predicted
            district_sum = sum(df[i].predicted for df in district_forecasts.values())
            
            if district_sum > 0:
                scale_factor = state_pred / district_sum
            else:
                scale_factor = 1.0
            
            for district, forecasts in district_forecasts.items():
                if district not in reconciled:
                    reconciled[district] = []
                
                orig = forecasts[i]
                reconciled[district].append(ForecastResult(
                    date=orig.date,
                    predicted=orig.predicted * scale_factor,
                    lower_bound=orig.lower_bound * scale_factor,
                    upper_bound=orig.upper_bound * scale_factor,
                    trend_component=orig.trend_component * scale_factor,
                    seasonal_component=orig.seasonal_component * scale_factor
                ))
        
        return reconciled
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the fitted model."""
        if not self._trained:
            return {"status": "not_trained"}
        
        return {
            "status": "trained",
            "model_type": "Hierarchical Time Series (Trend + Seasonality)",
            "parameters": {
                "global_mean": round(self._model_params['global_mean'], 2),
                "global_std": round(self._model_params['global_std'], 2),
                "trend_slope": round(self._model_params['trend_slope'], 4),
                "confidence_level": self.confidence_level
            },
            "seasonality": self._model_params['seasonality'].to_dict(),
            "num_groups": len(self._model_params.get('group_params', {})),
            "upgrade_path": "For production: integrate Prophet/SARIMA/XGBoost with lag features"
        }
