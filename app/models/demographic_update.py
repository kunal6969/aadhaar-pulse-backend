"""
Demographic Update SQLAlchemy model.
Stores demographic update records by date, geography, and age group.
"""
from sqlalchemy import Column, Integer, String, Date, BigInteger, Index
from app.database import Base


class DemographicUpdate(Base):
    """
    Demographic Update table model.
    
    Stores daily demographic update counts by state, district, pincode, and age group.
    Based on CSV columns: date, state, district, pincode, demo_age_5_17, demo_age_17_
    Expected row count: ~2,000,000+ rows
    """
    __tablename__ = "demographic_updates"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Temporal
    date = Column(Date, nullable=False, index=True)
    
    # Geography
    state = Column(String(100), nullable=False, index=True)
    district = Column(String(100), nullable=False, index=True)
    pincode = Column(String(10), index=True)
    
    # Age-wise demographic update counts (matching CSV columns)
    demo_age_5_17 = Column(BigInteger, default=0, nullable=False)
    demo_age_17_plus = Column(BigInteger, default=0, nullable=False)  # demo_age_17_ in CSV
    
    # Total (computed)
    total = Column(BigInteger, nullable=False)
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_demo_date_state', 'date', 'state'),
        Index('idx_demo_date_district', 'date', 'district'),
        Index('idx_demo_state_district', 'state', 'district'),
        Index('idx_demo_date_state_district', 'date', 'state', 'district'),
    )
    
    def __repr__(self):
        return f"<DemographicUpdate(id={self.id}, date={self.date}, district={self.district}, total={self.total})>"
    
    @property
    def age_breakdown(self) -> dict:
        """Return age-wise breakdown as dictionary."""
        return {
            "5-17": self.demo_age_5_17,
            "17+": self.demo_age_17_plus
        }
