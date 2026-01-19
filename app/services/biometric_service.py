"""
Biometric Update service - business logic for biometric update data queries.
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.biometric_update import BiometricUpdate
from app.models.enrollment import Enrollment
from app.utils.aggregators import aggregate_by_view_mode
from typing import List, Optional, Dict, Any
from datetime import date, timedelta
import pandas as pd


class BiometricService:
    """Business logic for biometric update data queries."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def get_trends(
        self,
        start_date: date,
        end_date: date,
        view_mode: str = "daily",
        state: Optional[str] = None,
        district: Optional[str] = None,
        age_group: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get biometric update trends with filtering and aggregation.
        
        Args:
            start_date: Start of date range
            end_date: End of date range (simulation_date)
            view_mode: daily, monthly, or quarterly
            state: Filter by state
            district: Filter by district
            age_group: Filter by age group (5-17, 17+, all)
            
        Returns:
            List of trend data points
        """
        # Build query
        query = self.db.query(
            BiometricUpdate.date,
            func.sum(BiometricUpdate.bio_age_5_17).label('bio_age_5_17'),
            func.sum(BiometricUpdate.bio_age_17_plus).label('bio_age_17_plus'),
            func.sum(BiometricUpdate.total).label('total')
        ).filter(
            BiometricUpdate.date >= start_date,
            BiometricUpdate.date <= end_date
        )
        
        # Apply geography filters
        if state:
            query = query.filter(BiometricUpdate.state == state)
        if district:
            query = query.filter(BiometricUpdate.district == district)
        
        # Group by date
        query = query.group_by(BiometricUpdate.date).order_by(BiometricUpdate.date)
        
        # Execute query
        results = query.all()
        
        if not results:
            return []
        
        # Convert to DataFrame for aggregation
        df = pd.DataFrame([{
            'date': r.date,
            'bio_age_5_17': int(r.bio_age_5_17 or 0),
            'bio_age_17_plus': int(r.bio_age_17_plus or 0),
            'total': int(r.total or 0)
        } for r in results])
        
        # Aggregate by view_mode
        value_columns = ['bio_age_5_17', 'bio_age_17_plus', 'total']
        agg_df = aggregate_by_view_mode(df, view_mode, 'date', value_columns)
        
        return agg_df.to_dict('records')
    
    def get_summary(
        self,
        simulation_date: date,
        state: Optional[str] = None,
        district: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate KPI summary for biometric updates.
        Includes MBU (Mandatory Biometric Update) specific metrics.
        
        Args:
            simulation_date: The simulated current date
            state: Optional state filter
            district: Optional district filter
            
        Returns:
            Dictionary with biometric update summary KPIs
        """
        # Last 30 days
        date_30d_ago = simulation_date - timedelta(days=30)
        
        # Build base query with optional filters
        base_filter = [
            BiometricUpdate.date >= date_30d_ago,
            BiometricUpdate.date <= simulation_date
        ]
        if state:
            base_filter.append(BiometricUpdate.state == state)
        if district:
            base_filter.append(BiometricUpdate.district == district)
        
        updates_30d = self.db.query(
            func.sum(BiometricUpdate.total),
            func.sum(BiometricUpdate.bio_age_5_17),
            func.sum(BiometricUpdate.bio_age_17_plus)
        ).filter(*base_filter).first()
        
        # Last 7 days
        date_7d_ago = simulation_date - timedelta(days=7)
        filter_7d = [
            BiometricUpdate.date >= date_7d_ago,
            BiometricUpdate.date <= simulation_date
        ]
        if state:
            filter_7d.append(BiometricUpdate.state == state)
        if district:
            filter_7d.append(BiometricUpdate.district == district)
        
        updates_7d = self.db.query(
            func.sum(BiometricUpdate.total)
        ).filter(*filter_7d).scalar() or 0
        
        # Today
        filter_today = [BiometricUpdate.date == simulation_date]
        if state:
            filter_today.append(BiometricUpdate.state == state)
        if district:
            filter_today.append(BiometricUpdate.district == district)
        
        updates_today = self.db.query(
            func.sum(BiometricUpdate.total)
        ).filter(*filter_today).scalar() or 0
        
        # Active districts and states (within filter)
        active_districts = self.db.query(
            func.count(func.distinct(BiometricUpdate.district))
        ).filter(*filter_today).scalar() or 0
        
        active_states = self.db.query(
            func.count(func.distinct(BiometricUpdate.state))
        ).filter(*filter_today).scalar() or 0
        
        # Top 5 states (only if no state filter)
        top_states_list = []
        total_30d = int(updates_30d[0] or 0)
        if not state:
            top_states = self.db.query(
                BiometricUpdate.state,
                func.sum(BiometricUpdate.total).label('total')
            ).filter(*base_filter).group_by(BiometricUpdate.state).order_by(
                func.sum(BiometricUpdate.total).desc()
            ).limit(5).all()
            
            for s in top_states:
                percentage = (s.total / total_30d * 100) if total_30d > 0 else 0
                top_states_list.append({
                    "state": s.state,
                    "count": int(s.total),
                    "percentage": round(percentage, 2)
                })
        
        # Calculate MBU metrics
        # Get total 5-17 age group enrollments (with filters if applicable)
        enrollment_filter = [Enrollment.date <= simulation_date]
        if state:
            enrollment_filter.append(Enrollment.state == state)
        if district:
            enrollment_filter.append(Enrollment.district == district)
            
        total_5_17_enrolled = self.db.query(
            func.sum(Enrollment.age_5_17)
        ).filter(*enrollment_filter).scalar() or 0
        
        # Estimate: ~15% of 5-17 will need MBU in next 6 months (approaching 15)
        pending_mbu_estimate = int(total_5_17_enrolled * 0.15)
        
        # MBU completion rate based on biometric updates
        bio_5_17_updates = int(updates_30d[1] or 0)
        # Estimate monthly expected MBU = pending / 6
        monthly_expected = pending_mbu_estimate / 6 if pending_mbu_estimate > 0 else 1
        mbu_completion_rate = min((bio_5_17_updates / monthly_expected) * 100, 100) if monthly_expected > 0 else 0
        
        return {
            "total_updates_30d": total_30d,
            "total_updates_7d": int(updates_7d),
            "total_updates_today": int(updates_today),
            "active_districts": active_districts,
            "active_states": active_states,
            "top_states": top_states_list,
            "simulation_date": simulation_date.isoformat(),
            "age_5_17_total": int(updates_30d[1] or 0),
            "age_17_plus_total": int(updates_30d[2] or 0),
            "pending_mbu_estimate": pending_mbu_estimate,
            "mbu_completion_rate": round(mbu_completion_rate, 2),
            "state_filter": state,
            "district_filter": district
        }
    
    def get_by_district_with_risk(
        self,
        simulation_date: date,
        state: Optional[str] = None,
        days_back: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get biometric data aggregated by district with MBU risk assessment."""
        start_date = simulation_date - timedelta(days=days_back)
        
        query = self.db.query(
            BiometricUpdate.district,
            BiometricUpdate.state,
            func.sum(BiometricUpdate.bio_age_5_17).label('bio_age_5_17'),
            func.sum(BiometricUpdate.bio_age_17_plus).label('bio_age_17_plus'),
            func.sum(BiometricUpdate.total).label('total'),
            func.count(func.distinct(BiometricUpdate.date)).label('days_with_data')
        ).filter(
            BiometricUpdate.date >= start_date,
            BiometricUpdate.date <= simulation_date
        )
        
        if state:
            query = query.filter(BiometricUpdate.state == state)
        
        query = query.group_by(
            BiometricUpdate.district, BiometricUpdate.state
        ).order_by(
            func.sum(BiometricUpdate.total).desc()
        ).limit(limit)
        
        results = query.all()
        
        districts = []
        for r in results:
            total = int(r.total or 0)
            days = int(r.days_with_data or 1)
            bio_5_17 = int(r.bio_age_5_17 or 0)
            
            # Calculate risk level based on update rate
            daily_avg = total / days
            
            # Get enrollment for this district to estimate expected MBU
            enrollment_5_17 = self.db.query(
                func.sum(Enrollment.age_5_17)
            ).filter(
                Enrollment.district == r.district,
                Enrollment.date <= simulation_date
            ).scalar() or 0
            
            expected_mbu = enrollment_5_17 * 0.15 / 6  # Monthly expected
            
            if expected_mbu > 0:
                completion_ratio = bio_5_17 / expected_mbu
                if completion_ratio < 0.3:
                    risk_level = "HIGH"
                elif completion_ratio < 0.7:
                    risk_level = "MEDIUM"
                else:
                    risk_level = "LOW"
            else:
                risk_level = "LOW"
            
            districts.append({
                "district": r.district,
                "state": r.state,
                "total": total,
                "bio_age_5_17": bio_5_17,
                "bio_age_17_plus": int(r.bio_age_17_plus or 0),
                "daily_average": round(daily_avg, 2),
                "mbu_risk_level": risk_level
            })
        
        return districts
    
    def get_historical_data_for_forecast(
        self,
        district: str,
        end_date: date,
        days_back: int = 180
    ) -> pd.DataFrame:
        """Get historical biometric data for ML forecasting."""
        start_date = end_date - timedelta(days=days_back)
        
        results = self.db.query(
            BiometricUpdate.date,
            func.sum(BiometricUpdate.total).label('total')
        ).filter(
            BiometricUpdate.district == district,
            BiometricUpdate.date >= start_date,
            BiometricUpdate.date <= end_date
        ).group_by(BiometricUpdate.date).order_by(BiometricUpdate.date).all()
        
        df = pd.DataFrame([{
            'date': r.date,
            'total': int(r.total or 0)
        } for r in results])
        
        return df
