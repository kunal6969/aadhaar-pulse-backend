"""
Geospatial Processor service - generate heatmap data for frontend maps.
"""
from sqlalchemy.orm import Session
from sqlalchemy import func
from app.models.enrollment import Enrollment
from app.models.demographic_update import DemographicUpdate
from app.models.biometric_update import BiometricUpdate
from app.utils.constants import DISTRICT_CENTROIDS, STATE_CENTROIDS
from app.utils.aggregators import normalize_intensity
from typing import List, Dict, Any, Optional, Literal
from datetime import date, timedelta
import calendar


class GeospatialProcessor:
    """Generate heatmap data for frontend maps."""
    
    def __init__(self, db: Session):
        self.db = db
    
    def generate_enrollment_heatmap(
        self,
        simulation_date: date,
        view_mode: str = "monthly",
        level: Literal["state", "district"] = "district"
    ) -> List[Dict[str, Any]]:
        """
        Generate enrollment intensity heatmap data.
        
        Args:
            simulation_date: The simulated current date
            view_mode: monthly or daily
            level: Geographic level (state or district)
            
        Returns:
            List of {lat, lng, intensity, name, value} objects
        """
        # Calculate date range based on view_mode
        if view_mode == "monthly":
            start_date = simulation_date.replace(day=1)
            last_day = calendar.monthrange(simulation_date.year, simulation_date.month)[1]
            end_date = simulation_date.replace(day=last_day)
            end_date = min(end_date, simulation_date)
        else:  # daily
            start_date = simulation_date
            end_date = simulation_date
        
        # Query based on level
        if level == "state":
            query = self.db.query(
                Enrollment.state.label('name'),
                func.sum(Enrollment.total).label('total')
            ).filter(
                Enrollment.date >= start_date,
                Enrollment.date <= end_date
            ).group_by(Enrollment.state)
            
            centroids = STATE_CENTROIDS
        else:  # district
            query = self.db.query(
                Enrollment.district.label('name'),
                Enrollment.state,
                func.sum(Enrollment.total).label('total')
            ).filter(
                Enrollment.date >= start_date,
                Enrollment.date <= end_date
            ).group_by(Enrollment.district, Enrollment.state)
            
            centroids = DISTRICT_CENTROIDS
        
        results = query.all()
        
        if not results:
            return []
        
        # Get max for normalization
        values = [int(r.total or 0) for r in results]
        max_val = max(values) if values else 1
        
        heatmap_data = []
        for r in results:
            name = r.name
            total = int(r.total or 0)
            
            # Get coordinates
            if level == "state":
                if name in centroids:
                    lat = centroids[name]["lat"]
                    lng = centroids[name]["lng"]
                else:
                    continue  # Skip if no coordinates
            else:
                if name in centroids:
                    lat = centroids[name]["lat"]
                    lng = centroids[name]["lng"]
                    state = centroids[name].get("state", r.state if hasattr(r, 'state') else "")
                else:
                    continue  # Skip if no coordinates
            
            # Calculate intensity (0-100)
            intensity = (total / max_val * 100) if max_val > 0 else 0
            
            data_point = {
                "lat": lat,
                "lng": lng,
                "intensity": round(intensity, 2),
                "name": name,
                "value": total
            }
            
            if level == "district" and name in centroids:
                data_point["state"] = centroids[name].get("state", "")
            
            heatmap_data.append(data_point)
        
        return heatmap_data
    
    def generate_demographic_heatmap(
        self,
        simulation_date: date,
        view_mode: str = "monthly",
        level: Literal["state", "district"] = "district"
    ) -> List[Dict[str, Any]]:
        """Generate demographic update intensity heatmap data."""
        # Similar logic to enrollment
        if view_mode == "monthly":
            start_date = simulation_date.replace(day=1)
            last_day = calendar.monthrange(simulation_date.year, simulation_date.month)[1]
            end_date = min(simulation_date.replace(day=last_day), simulation_date)
        else:
            start_date = simulation_date
            end_date = simulation_date
        
        if level == "state":
            query = self.db.query(
                DemographicUpdate.state.label('name'),
                func.sum(DemographicUpdate.total).label('total')
            ).filter(
                DemographicUpdate.date >= start_date,
                DemographicUpdate.date <= end_date
            ).group_by(DemographicUpdate.state)
            
            centroids = STATE_CENTROIDS
        else:
            query = self.db.query(
                DemographicUpdate.district.label('name'),
                DemographicUpdate.state,
                func.sum(DemographicUpdate.total).label('total')
            ).filter(
                DemographicUpdate.date >= start_date,
                DemographicUpdate.date <= end_date
            ).group_by(DemographicUpdate.district, DemographicUpdate.state)
            
            centroids = DISTRICT_CENTROIDS
        
        results = query.all()
        
        if not results:
            return []
        
        values = [int(r.total or 0) for r in results]
        max_val = max(values) if values else 1
        
        heatmap_data = []
        for r in results:
            name = r.name
            total = int(r.total or 0)
            
            if name not in centroids:
                continue
            
            lat = centroids[name]["lat"]
            lng = centroids[name]["lng"]
            intensity = (total / max_val * 100) if max_val > 0 else 0
            
            data_point = {
                "lat": lat,
                "lng": lng,
                "intensity": round(intensity, 2),
                "name": name,
                "value": total
            }
            
            if level == "district":
                data_point["state"] = centroids[name].get("state", "")
            
            heatmap_data.append(data_point)
        
        return heatmap_data
    
    def generate_biometric_heatmap(
        self,
        simulation_date: date,
        view_mode: str = "monthly",
        level: Literal["state", "district"] = "district"
    ) -> List[Dict[str, Any]]:
        """Generate biometric update intensity heatmap data."""
        if view_mode == "monthly":
            start_date = simulation_date.replace(day=1)
            last_day = calendar.monthrange(simulation_date.year, simulation_date.month)[1]
            end_date = min(simulation_date.replace(day=last_day), simulation_date)
        else:
            start_date = simulation_date
            end_date = simulation_date
        
        if level == "state":
            query = self.db.query(
                BiometricUpdate.state.label('name'),
                func.sum(BiometricUpdate.total).label('total')
            ).filter(
                BiometricUpdate.date >= start_date,
                BiometricUpdate.date <= end_date
            ).group_by(BiometricUpdate.state)
            
            centroids = STATE_CENTROIDS
        else:
            query = self.db.query(
                BiometricUpdate.district.label('name'),
                BiometricUpdate.state,
                func.sum(BiometricUpdate.total).label('total')
            ).filter(
                BiometricUpdate.date >= start_date,
                BiometricUpdate.date <= end_date
            ).group_by(BiometricUpdate.district, BiometricUpdate.state)
            
            centroids = DISTRICT_CENTROIDS
        
        results = query.all()
        
        if not results:
            return []
        
        values = [int(r.total or 0) for r in results]
        max_val = max(values) if values else 1
        
        heatmap_data = []
        for r in results:
            name = r.name
            total = int(r.total or 0)
            
            if name not in centroids:
                continue
            
            lat = centroids[name]["lat"]
            lng = centroids[name]["lng"]
            intensity = (total / max_val * 100) if max_val > 0 else 0
            
            data_point = {
                "lat": lat,
                "lng": lng,
                "intensity": round(intensity, 2),
                "name": name,
                "value": total
            }
            
            if level == "district":
                data_point["state"] = centroids[name].get("state", "")
            
            heatmap_data.append(data_point)
        
        return heatmap_data
    
    def generate_combined_heatmap(
        self,
        simulation_date: date,
        view_mode: str = "monthly",
        level: Literal["state", "district"] = "district"
    ) -> List[Dict[str, Any]]:
        """Generate combined activity heatmap (all data types)."""
        enrollment_data = self.generate_enrollment_heatmap(simulation_date, view_mode, level)
        demo_data = self.generate_demographic_heatmap(simulation_date, view_mode, level)
        bio_data = self.generate_biometric_heatmap(simulation_date, view_mode, level)
        
        # Combine by name
        combined = {}
        
        for item in enrollment_data:
            key = item["name"]
            if key not in combined:
                combined[key] = {
                    "lat": item["lat"],
                    "lng": item["lng"],
                    "name": key,
                    "state": item.get("state", ""),
                    "enrollment": item["value"],
                    "demographic": 0,
                    "biometric": 0,
                    "total": 0
                }
            combined[key]["enrollment"] = item["value"]
        
        for item in demo_data:
            key = item["name"]
            if key not in combined:
                combined[key] = {
                    "lat": item["lat"],
                    "lng": item["lng"],
                    "name": key,
                    "state": item.get("state", ""),
                    "enrollment": 0,
                    "demographic": item["value"],
                    "biometric": 0,
                    "total": 0
                }
            else:
                combined[key]["demographic"] = item["value"]
        
        for item in bio_data:
            key = item["name"]
            if key not in combined:
                combined[key] = {
                    "lat": item["lat"],
                    "lng": item["lng"],
                    "name": key,
                    "state": item.get("state", ""),
                    "enrollment": 0,
                    "demographic": 0,
                    "biometric": item["value"],
                    "total": 0
                }
            else:
                combined[key]["biometric"] = item["value"]
        
        # Calculate totals and intensity
        result = []
        for key, data in combined.items():
            total = data["enrollment"] + data["demographic"] + data["biometric"]
            data["total"] = total
            result.append(data)
        
        # Calculate intensity based on total
        if result:
            max_total = max(d["total"] for d in result)
            for item in result:
                item["intensity"] = round((item["total"] / max_total * 100) if max_total > 0 else 0, 2)
                item["value"] = item["total"]
        
        return result
