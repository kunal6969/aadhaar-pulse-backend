"""
Demographic Update service - business logic for demographic update data queries.
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.demographic_update import DemographicUpdate
from app.utils.aggregators import aggregate_by_view_mode
from typing import List, Optional, Dict, Any
from datetime import date, timedelta
import pandas as pd


class DemographicService:
    """Business logic for demographic update data queries."""
    
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
        Get demographic update trends with filtering and aggregation.
        
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
            DemographicUpdate.date,
            func.sum(DemographicUpdate.demo_age_5_17).label('demo_age_5_17'),
            func.sum(DemographicUpdate.demo_age_17_plus).label('demo_age_17_plus'),
            func.sum(DemographicUpdate.total).label('total')
        ).filter(
            DemographicUpdate.date >= start_date,
            DemographicUpdate.date <= end_date
        )
        
        # Apply geography filters
        if state:
            query = query.filter(DemographicUpdate.state == state)
        if district:
            query = query.filter(DemographicUpdate.district == district)
        
        # Group by date
        query = query.group_by(DemographicUpdate.date).order_by(DemographicUpdate.date)
        
        # Execute query
        results = query.all()
        
        if not results:
            return []
        
        # Convert to DataFrame for aggregation
        df = pd.DataFrame([{
            'date': r.date,
            'demo_age_5_17': int(r.demo_age_5_17 or 0),
            'demo_age_17_plus': int(r.demo_age_17_plus or 0),
            'total': int(r.total or 0)
        } for r in results])
        
        # Aggregate by view_mode
        value_columns = ['demo_age_5_17', 'demo_age_17_plus', 'total']
        agg_df = aggregate_by_view_mode(df, view_mode, 'date', value_columns)
        
        return agg_df.to_dict('records')
    
    def get_summary(self, simulation_date: date) -> Dict[str, Any]:
        """
        Calculate KPI summary for demographic updates.
        
        Args:
            simulation_date: The simulated current date
            
        Returns:
            Dictionary with demographic update summary KPIs
        """
        # Last 30 days
        date_30d_ago = simulation_date - timedelta(days=30)
        updates_30d = self.db.query(
            func.sum(DemographicUpdate.total),
            func.sum(DemographicUpdate.demo_age_5_17),
            func.sum(DemographicUpdate.demo_age_17_plus)
        ).filter(
            DemographicUpdate.date >= date_30d_ago,
            DemographicUpdate.date <= simulation_date
        ).first()
        
        # Last 7 days
        date_7d_ago = simulation_date - timedelta(days=7)
        updates_7d = self.db.query(
            func.sum(DemographicUpdate.total)
        ).filter(
            DemographicUpdate.date >= date_7d_ago,
            DemographicUpdate.date <= simulation_date
        ).scalar() or 0
        
        # Today
        updates_today = self.db.query(
            func.sum(DemographicUpdate.total)
        ).filter(
            DemographicUpdate.date == simulation_date
        ).scalar() or 0
        
        # Active districts and states
        active_districts = self.db.query(
            func.count(func.distinct(DemographicUpdate.district))
        ).filter(
            DemographicUpdate.date == simulation_date
        ).scalar() or 0
        
        active_states = self.db.query(
            func.count(func.distinct(DemographicUpdate.state))
        ).filter(
            DemographicUpdate.date == simulation_date
        ).scalar() or 0
        
        # Top 5 states
        top_states = self.db.query(
            DemographicUpdate.state,
            func.sum(DemographicUpdate.total).label('total')
        ).filter(
            DemographicUpdate.date >= date_30d_ago,
            DemographicUpdate.date <= simulation_date
        ).group_by(DemographicUpdate.state).order_by(
            func.sum(DemographicUpdate.total).desc()
        ).limit(5).all()
        
        total_30d = int(updates_30d[0] or 0)
        
        top_states_list = []
        for s in top_states:
            percentage = (s.total / total_30d * 100) if total_30d > 0 else 0
            top_states_list.append({
                "state": s.state,
                "count": int(s.total),
                "percentage": round(percentage, 2)
            })
        
        return {
            "total_updates_30d": total_30d,
            "total_updates_7d": int(updates_7d),
            "total_updates_today": int(updates_today),
            "active_districts": active_districts,
            "active_states": active_states,
            "top_states": top_states_list,
            "simulation_date": simulation_date.isoformat(),
            "age_5_17_total": int(updates_30d[1] or 0),
            "age_17_plus_total": int(updates_30d[2] or 0)
        }
    
    def get_by_district(
        self,
        simulation_date: date,
        state: Optional[str] = None,
        days_back: int = 30,
        limit: int = 50
    ) -> List[Dict[str, Any]]:
        """Get demographic data aggregated by district."""
        start_date = simulation_date - timedelta(days=days_back)
        
        query = self.db.query(
            DemographicUpdate.district,
            DemographicUpdate.state,
            func.sum(DemographicUpdate.demo_age_5_17).label('demo_age_5_17'),
            func.sum(DemographicUpdate.demo_age_17_plus).label('demo_age_17_plus'),
            func.sum(DemographicUpdate.total).label('total'),
            func.count(func.distinct(DemographicUpdate.date)).label('days_with_data')
        ).filter(
            DemographicUpdate.date >= start_date,
            DemographicUpdate.date <= simulation_date
        )
        
        if state:
            query = query.filter(DemographicUpdate.state == state)
        
        query = query.group_by(
            DemographicUpdate.district, DemographicUpdate.state
        ).order_by(
            func.sum(DemographicUpdate.total).desc()
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
                "demo_age_5_17": int(r.demo_age_5_17 or 0),
                "demo_age_17_plus": int(r.demo_age_17_plus or 0),
                "daily_average": round(total / days, 2)
            })
        
        return districts
