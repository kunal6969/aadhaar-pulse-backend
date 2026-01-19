"""
Enrollment service - business logic for enrollment data queries.
"""
from sqlalchemy.orm import Session
from sqlalchemy import func, and_
from app.models.enrollment import Enrollment
from app.utils.aggregators import aggregate_by_view_mode, calculate_growth_rate
from typing import List, Optional, Dict, Any
from datetime import date, timedelta
import pandas as pd


class EnrollmentService:
    """Business logic for enrollment data queries."""
    
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
        Get enrollment trends with filtering and aggregation.
        
        Args:
            start_date: Start of date range
            end_date: End of date range (simulation_date)
            view_mode: daily, monthly, or quarterly
            state: Filter by state
            district: Filter by district
            age_group: Filter by age group (0-5, 5-17, 18+, all)
            
        Returns:
            List of trend data points
        """
        # Build query
        query = self.db.query(
            Enrollment.date,
            func.sum(Enrollment.age_0_5).label('age_0_5'),
            func.sum(Enrollment.age_5_17).label('age_5_17'),
            func.sum(Enrollment.age_18_plus).label('age_18_plus'),
            func.sum(Enrollment.total).label('total')
        ).filter(
            Enrollment.date >= start_date,
            Enrollment.date <= end_date
        )
        
        # Apply geography filters
        if state:
            query = query.filter(Enrollment.state == state)
        if district:
            query = query.filter(Enrollment.district == district)
        
        # Group by date
        query = query.group_by(Enrollment.date).order_by(Enrollment.date)
        
        # Execute query
        results = query.all()
        
        if not results:
            return []
        
        # Convert to DataFrame for aggregation
        df = pd.DataFrame([{
            'date': r.date,
            'age_0_5': int(r.age_0_5 or 0),
            'age_5_17': int(r.age_5_17 or 0),
            'age_18_plus': int(r.age_18_plus or 0),
            'total': int(r.total or 0)
        } for r in results])
        
        # Aggregate by view_mode
        value_columns = ['age_0_5', 'age_5_17', 'age_18_plus', 'total']
        agg_df = aggregate_by_view_mode(df, view_mode, 'date', value_columns)
        
        # Convert to list of dicts
        return agg_df.to_dict('records')
    
    def get_summary(
        self,
        simulation_date: date,
        state: Optional[str] = None,
        district: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate KPI summary for overview page.
        
        Args:
            simulation_date: The simulated current date
            state: Optional state filter
            district: Optional district filter
            
        Returns:
            Dictionary with enrollment summary KPIs
        """
        # Last 30 days
        date_30d_ago = simulation_date - timedelta(days=30)
        
        # Build base query with optional filters
        base_filter = [
            Enrollment.date >= date_30d_ago,
            Enrollment.date <= simulation_date
        ]
        if state:
            base_filter.append(Enrollment.state == state)
        if district:
            base_filter.append(Enrollment.district == district)
        
        enrollments_30d = self.db.query(
            func.sum(Enrollment.total),
            func.sum(Enrollment.age_0_5),
            func.sum(Enrollment.age_5_17),
            func.sum(Enrollment.age_18_plus)
        ).filter(*base_filter).first()
        
        # Last 7 days
        date_7d_ago = simulation_date - timedelta(days=7)
        filter_7d = [
            Enrollment.date >= date_7d_ago,
            Enrollment.date <= simulation_date
        ]
        if state:
            filter_7d.append(Enrollment.state == state)
        if district:
            filter_7d.append(Enrollment.district == district)
        
        enrollments_7d = self.db.query(
            func.sum(Enrollment.total)
        ).filter(*filter_7d).scalar() or 0
        
        # Today
        filter_today = [Enrollment.date == simulation_date]
        if state:
            filter_today.append(Enrollment.state == state)
        if district:
            filter_today.append(Enrollment.district == district)
        
        enrollments_today = self.db.query(
            func.sum(Enrollment.total)
        ).filter(*filter_today).scalar() or 0
        
        # Active districts and states (within filter)
        active_districts = self.db.query(
            func.count(func.distinct(Enrollment.district))
        ).filter(*filter_today).scalar() or 0
        
        active_states = self.db.query(
            func.count(func.distinct(Enrollment.state))
        ).filter(*filter_today).scalar() or 0
        
        # Top 5 states by enrollment in last 30 days (only if no state filter)
        top_states_list = []
        if not state:
            top_states = self.db.query(
                Enrollment.state,
                func.sum(Enrollment.total).label('total')
            ).filter(*base_filter).group_by(Enrollment.state).order_by(
                func.sum(Enrollment.total).desc()
            ).limit(5).all()
            
            total_30d = int(enrollments_30d[0] or 0)
            for s in top_states:
                percentage = (s.total / total_30d * 100) if total_30d > 0 else 0
                top_states_list.append({
                    "state": s.state,
                    "count": int(s.total),
                    "percentage": round(percentage, 2)
                })
        
        return {
            "total_enrollments_30d": int(enrollments_30d[0] or 0),
            "total_enrollments_7d": int(enrollments_7d),
            "total_enrollments_today": int(enrollments_today),
            "active_districts": active_districts,
            "active_states": active_states,
            "top_states": top_states_list,
            "simulation_date": simulation_date.isoformat(),
            "age_0_5_total": int(enrollments_30d[1] or 0),
            "age_5_17_total": int(enrollments_30d[2] or 0),
            "age_18_plus_total": int(enrollments_30d[3] or 0),
            "state_filter": state,
            "district_filter": district
        }
    
    def get_by_district(
        self,
        simulation_date: date,
        state: Optional[str] = None,
        days_back: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """
        Get enrollment data aggregated by district.
        
        Args:
            simulation_date: The simulated current date
            state: Optional state filter
            days_back: Number of days to look back
            limit: Maximum number of districts to return
            
        Returns:
            List of districts with enrollment data
        """
        start_date = simulation_date - timedelta(days=days_back)
        
        query = self.db.query(
            Enrollment.district,
            Enrollment.state,
            func.sum(Enrollment.age_0_5).label('age_0_5'),
            func.sum(Enrollment.age_5_17).label('age_5_17'),
            func.sum(Enrollment.age_18_plus).label('age_18_plus'),
            func.sum(Enrollment.total).label('total'),
            func.count(func.distinct(Enrollment.date)).label('days_with_data')
        ).filter(
            Enrollment.date >= start_date,
            Enrollment.date <= simulation_date
        )
        
        if state:
            query = query.filter(Enrollment.state == state)
        
        query = query.group_by(
            Enrollment.district, Enrollment.state
        ).order_by(
            func.sum(Enrollment.total).desc()
        ).limit(limit)
        
        results = query.all()
        
        districts = []
        for r in results:
            total = int(r.total or 0)
            days = int(r.days_with_data or 1)
            districts.append({
                "district": r.district,
                "state": r.state,
                "total": total,
                "age_0_5": int(r.age_0_5 or 0),
                "age_5_17": int(r.age_5_17 or 0),
                "age_18_plus": int(r.age_18_plus or 0),
                "daily_average": round(total / days, 2)
            })
        
        return districts
    
    def get_states_list(self) -> List[str]:
        """Get list of all states in the data."""
        states = self.db.query(
            func.distinct(Enrollment.state)
        ).order_by(Enrollment.state).all()
        return [s[0] for s in states]
    
    def get_districts_list(self, state: Optional[str] = None) -> List[str]:
        """Get list of districts, optionally filtered by state."""
        query = self.db.query(func.distinct(Enrollment.district))
        if state:
            query = query.filter(Enrollment.state == state)
        districts = query.order_by(Enrollment.district).all()
        return [d[0] for d in districts]
