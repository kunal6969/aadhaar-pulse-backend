"""
KPI Calculator service - compute various analytics KPIs.
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.enrollment import Enrollment
from app.models.demographic_update import DemographicUpdate
from app.models.biometric_update import BiometricUpdate
from app.utils.aggregators import calculate_growth_rate
from typing import Dict, List, Any, Optional
from datetime import date, timedelta


class KPICalculator:
    """Calculate various KPIs for the dashboard."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def calculate_all_kpis(self, simulation_date: date) -> Dict[str, Any]:
        """
        Calculate all KPIs for the overview dashboard.
        
        Args:
            simulation_date: The simulated current date
            
        Returns:
            Dictionary with all KPIs
        """
        date_30d_ago = simulation_date - timedelta(days=30)
        date_7d_ago = simulation_date - timedelta(days=7)
        date_60d_ago = simulation_date - timedelta(days=60)
        
        # Enrollment KPIs
        enrollments_30d = self.db.query(
            func.sum(Enrollment.total)
        ).filter(
            Enrollment.date >= date_30d_ago,
            Enrollment.date <= simulation_date
        ).scalar() or 0
        
        enrollments_7d = self.db.query(
            func.sum(Enrollment.total)
        ).filter(
            Enrollment.date >= date_7d_ago,
            Enrollment.date <= simulation_date
        ).scalar() or 0
        
        enrollments_today = self.db.query(
            func.sum(Enrollment.total)
        ).filter(
            Enrollment.date == simulation_date
        ).scalar() or 0
        
        # Previous 30 days for growth calculation
        enrollments_prev_30d = self.db.query(
            func.sum(Enrollment.total)
        ).filter(
            Enrollment.date >= date_60d_ago,
            Enrollment.date < date_30d_ago
        ).scalar() or 0
        
        # Demographic KPIs
        demo_30d = self.db.query(
            func.sum(DemographicUpdate.total)
        ).filter(
            DemographicUpdate.date >= date_30d_ago,
            DemographicUpdate.date <= simulation_date
        ).scalar() or 0
        
        demo_7d = self.db.query(
            func.sum(DemographicUpdate.total)
        ).filter(
            DemographicUpdate.date >= date_7d_ago,
            DemographicUpdate.date <= simulation_date
        ).scalar() or 0
        
        demo_today = self.db.query(
            func.sum(DemographicUpdate.total)
        ).filter(
            DemographicUpdate.date == simulation_date
        ).scalar() or 0
        
        demo_prev_30d = self.db.query(
            func.sum(DemographicUpdate.total)
        ).filter(
            DemographicUpdate.date >= date_60d_ago,
            DemographicUpdate.date < date_30d_ago
        ).scalar() or 0
        
        # Biometric KPIs
        bio_30d = self.db.query(
            func.sum(BiometricUpdate.total)
        ).filter(
            BiometricUpdate.date >= date_30d_ago,
            BiometricUpdate.date <= simulation_date
        ).scalar() or 0
        
        bio_7d = self.db.query(
            func.sum(BiometricUpdate.total)
        ).filter(
            BiometricUpdate.date >= date_7d_ago,
            BiometricUpdate.date <= simulation_date
        ).scalar() or 0
        
        bio_today = self.db.query(
            func.sum(BiometricUpdate.total)
        ).filter(
            BiometricUpdate.date == simulation_date
        ).scalar() or 0
        
        bio_prev_30d = self.db.query(
            func.sum(BiometricUpdate.total)
        ).filter(
            BiometricUpdate.date >= date_60d_ago,
            BiometricUpdate.date < date_30d_ago
        ).scalar() or 0
        
        # Geographic KPIs
        active_districts = self.db.query(
            func.count(func.distinct(Enrollment.district))
        ).filter(
            Enrollment.date == simulation_date
        ).scalar() or 0
        
        active_states = self.db.query(
            func.count(func.distinct(Enrollment.state))
        ).filter(
            Enrollment.date == simulation_date
        ).scalar() or 0
        
        # MBU metrics
        total_5_17 = self.db.query(
            func.sum(Enrollment.age_5_17)
        ).filter(
            Enrollment.date >= date_30d_ago,
            Enrollment.date <= simulation_date
        ).scalar() or 0
        
        pending_mbu = int(total_5_17 * 0.15)
        
        bio_5_17 = self.db.query(
            func.sum(BiometricUpdate.bio_age_5_17)
        ).filter(
            BiometricUpdate.date >= date_30d_ago,
            BiometricUpdate.date <= simulation_date
        ).scalar() or 0
        
        mbu_completion = (bio_5_17 / (pending_mbu / 6) * 100) if pending_mbu > 0 else 100
        
        # Top 5 states
        top_states = self._get_top_states(simulation_date, 5)
        
        return {
            "simulation_date": simulation_date.isoformat(),
            
            "total_enrollments_30d": int(enrollments_30d),
            "total_enrollments_7d": int(enrollments_7d),
            "total_enrollments_today": int(enrollments_today),
            "enrollment_growth_rate": calculate_growth_rate(int(enrollments_30d), int(enrollments_prev_30d)),
            
            "total_demo_updates_30d": int(demo_30d),
            "total_demo_updates_7d": int(demo_7d),
            "total_demo_updates_today": int(demo_today),
            "demo_growth_rate": calculate_growth_rate(int(demo_30d), int(demo_prev_30d)),
            
            "total_bio_updates_30d": int(bio_30d),
            "total_bio_updates_7d": int(bio_7d),
            "total_bio_updates_today": int(bio_today),
            "bio_growth_rate": calculate_growth_rate(int(bio_30d), int(bio_prev_30d)),
            
            "active_districts": active_districts,
            "active_states": active_states,
            
            "pending_mbu_count": pending_mbu,
            "mbu_completion_percentage": round(min(mbu_completion, 100), 2),
            
            "top_5_states": top_states
        }
    
    def _get_top_states(self, simulation_date: date, limit: int = 5) -> List[Dict]:
        """Get top states by total activity."""
        date_30d_ago = simulation_date - timedelta(days=30)
        
        # Get enrollment by state
        enrollment_by_state = dict(self.db.query(
            Enrollment.state,
            func.sum(Enrollment.total)
        ).filter(
            Enrollment.date >= date_30d_ago,
            Enrollment.date <= simulation_date
        ).group_by(Enrollment.state).all())
        
        # Get demographic by state
        demo_by_state = dict(self.db.query(
            DemographicUpdate.state,
            func.sum(DemographicUpdate.total)
        ).filter(
            DemographicUpdate.date >= date_30d_ago,
            DemographicUpdate.date <= simulation_date
        ).group_by(DemographicUpdate.state).all())
        
        # Get biometric by state
        bio_by_state = dict(self.db.query(
            BiometricUpdate.state,
            func.sum(BiometricUpdate.total)
        ).filter(
            BiometricUpdate.date >= date_30d_ago,
            BiometricUpdate.date <= simulation_date
        ).group_by(BiometricUpdate.state).all())
        
        # Combine all states
        all_states = set(enrollment_by_state.keys()) | set(demo_by_state.keys()) | set(bio_by_state.keys())
        
        state_totals = []
        for state in all_states:
            enroll = int(enrollment_by_state.get(state, 0) or 0)
            demo = int(demo_by_state.get(state, 0) or 0)
            bio = int(bio_by_state.get(state, 0) or 0)
            total = enroll + demo + bio
            
            state_totals.append({
                "state": state,
                "enrollment_count": enroll,
                "demographic_count": demo,
                "biometric_count": bio,
                "total_activity": total
            })
        
        # Sort by total activity and take top N
        state_totals.sort(key=lambda x: x["total_activity"], reverse=True)
        
        return state_totals[:limit]
    
    def calculate_update_burden_index(
        self,
        simulation_date: date,
        state_filter: Optional[str] = None
    ) -> List[Dict]:
        """
        Calculate Update Burden Index for districts.
        
        Higher burden = more updates needed = potentially strained resources.
        """
        date_30d_ago = simulation_date - timedelta(days=30)
        
        # Get demographic updates by district
        demo_query = self.db.query(
            DemographicUpdate.district,
            DemographicUpdate.state,
            func.sum(DemographicUpdate.total).label('demo_total')
        ).filter(
            DemographicUpdate.date >= date_30d_ago,
            DemographicUpdate.date <= simulation_date
        )
        
        if state_filter:
            demo_query = demo_query.filter(DemographicUpdate.state == state_filter)
        
        demo_results = demo_query.group_by(
            DemographicUpdate.district, DemographicUpdate.state
        ).all()
        
        demo_by_district = {
            (r.district, r.state): int(r.demo_total or 0) 
            for r in demo_results
        }
        
        # Get biometric updates by district
        bio_query = self.db.query(
            BiometricUpdate.district,
            BiometricUpdate.state,
            func.sum(BiometricUpdate.total).label('bio_total')
        ).filter(
            BiometricUpdate.date >= date_30d_ago,
            BiometricUpdate.date <= simulation_date
        )
        
        if state_filter:
            bio_query = bio_query.filter(BiometricUpdate.state == state_filter)
        
        bio_results = bio_query.group_by(
            BiometricUpdate.district, BiometricUpdate.state
        ).all()
        
        bio_by_district = {
            (r.district, r.state): int(r.bio_total or 0)
            for r in bio_results
        }
        
        # Combine districts
        all_districts = set(demo_by_district.keys()) | set(bio_by_district.keys())
        
        burden_data = []
        for district, state in all_districts:
            demo = demo_by_district.get((district, state), 0)
            bio = bio_by_district.get((district, state), 0)
            
            # Simple burden score = total updates (can be weighted)
            burden_score = demo * 0.4 + bio * 0.6
            
            # Determine burden level
            if burden_score > 50000:
                level = "CRITICAL"
            elif burden_score > 20000:
                level = "HIGH"
            elif burden_score > 5000:
                level = "MEDIUM"
            else:
                level = "LOW"
            
            burden_data.append({
                "district": district,
                "state": state,
                "burden_score": round(burden_score, 2),
                "demographic_updates": demo,
                "biometric_updates": bio,
                "population_estimate": 0,  # Would need external data
                "burden_level": level
            })
        
        # Sort by burden score
        burden_data.sort(key=lambda x: x["burden_score"], reverse=True)
        
        return burden_data
    
    def calculate_digital_readiness(
        self,
        simulation_date: date,
        limit: int = 100
    ) -> List[Dict]:
        """
        Calculate digital readiness score based on demographic update patterns.
        
        Low demographic update frequency = More stable data = Higher readiness.
        """
        date_30d_ago = simulation_date - timedelta(days=30)
        
        # Get demographic updates by district
        results = self.db.query(
            DemographicUpdate.district,
            DemographicUpdate.state,
            func.sum(DemographicUpdate.demo_age_5_17).label('age_5_17'),
            func.sum(DemographicUpdate.demo_age_17_plus).label('age_17_plus'),
            func.sum(DemographicUpdate.total).label('total'),
            func.count(func.distinct(DemographicUpdate.date)).label('days')
        ).filter(
            DemographicUpdate.date >= date_30d_ago,
            DemographicUpdate.date <= simulation_date
        ).group_by(
            DemographicUpdate.district, DemographicUpdate.state
        ).order_by(
            func.sum(DemographicUpdate.total).asc()  # Lower updates = higher readiness
        ).limit(limit).all()
        
        if not results:
            return []
        
        # Find max for normalization
        max_updates = max(int(r.total or 1) for r in results)
        
        readiness_data = []
        for r in results:
            total = int(r.total or 0)
            days = int(r.days or 1)
            
            # Inverse relationship: fewer updates = higher readiness
            stability_score = max(0, 100 - (total / max_updates * 100))
            
            # Readiness combines multiple factors
            readiness_score = stability_score
            
            # Recommendation based on score
            if readiness_score > 70:
                recommendation = "Excellent for digital services"
            elif readiness_score > 40:
                recommendation = "Good for basic digital services"
            else:
                recommendation = "May need additional verification steps"
            
            readiness_data.append({
                "district": r.district,
                "state": r.state,
                "readiness_score": round(readiness_score, 2),
                "mobile_update_frequency": round(total / days, 2),
                "stability_score": round(stability_score, 2),
                "recommendation": recommendation
            })
        
        return readiness_data
