"""
Common Pydantic schemas shared across endpoints.
"""
from pydantic import BaseModel, Field, field_validator
from typing import Optional, Literal, List, Any
from datetime import date, timedelta


class SimulationContext(BaseModel):
    """Common simulation parameters for all requests."""
    simulation_date: date = Field(..., description="The simulated 'current date' in YYYY-MM-DD format")
    view_mode: Literal["daily", "monthly", "quarterly"] = Field(
        "daily", description="Aggregation level"
    )
    chart_start_date: Optional[date] = Field(
        None, description="Start of historical chart range"
    )
    chart_end_date: Optional[date] = Field(
        None, description="End of historical chart range (defaults to simulation_date)"
    )
    
    @field_validator('chart_end_date', mode='before')
    @classmethod
    def set_chart_end_date(cls, v, info):
        if v is None and 'simulation_date' in info.data:
            return info.data['simulation_date']
        return v


class DateRangeFilter(BaseModel):
    """Date range filter parameters."""
    start_date: Optional[date] = Field(None, description="Start date for filtering")
    end_date: Optional[date] = Field(None, description="End date for filtering")
    
    def get_default_range(self, simulation_date: date, days_back: int = 90) -> tuple:
        """Get date range with defaults based on simulation date."""
        end = self.end_date or simulation_date
        start = self.start_date or (simulation_date - timedelta(days=days_back))
        return start, min(end, simulation_date)


class GeographyFilter(BaseModel):
    """Geography-based filters."""
    state: Optional[str] = Field(None, description="Filter by state name")
    district: Optional[str] = Field(None, description="Filter by district name")
    pincode: Optional[str] = Field(None, description="Filter by pincode")


class PaginationParams(BaseModel):
    """Pagination parameters."""
    page: int = Field(1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(50, ge=1, le=1000, description="Items per page")
    
    @property
    def offset(self) -> int:
        """Calculate offset for database query."""
        return (self.page - 1) * self.page_size


class PaginatedResponse(BaseModel):
    """Standard paginated response wrapper."""
    data: List[Any]
    total: int
    page: int
    page_size: int
    total_pages: int
    
    @classmethod
    def create(cls, data: List[Any], total: int, page: int, page_size: int):
        """Factory method to create paginated response."""
        total_pages = (total + page_size - 1) // page_size
        return cls(
            data=data,
            total=total,
            page=page,
            page_size=page_size,
            total_pages=total_pages
        )


class StateDistrictInfo(BaseModel):
    """State and district information."""
    name: str
    count: int


class TopItemsResponse(BaseModel):
    """Response for top items queries."""
    items: List[StateDistrictInfo]
    total_count: int


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    database: str
    
    
class AvailableDatesResponse(BaseModel):
    """Response for available dates endpoint."""
    dates: List[str]
    count: int
    start: Optional[str]
    end: Optional[str]
