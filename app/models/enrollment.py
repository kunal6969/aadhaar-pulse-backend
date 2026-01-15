"""
Enrollment SQLAlchemy model.
Stores Aadhaar enrollment records by date, geography, and age group.
"""
from sqlalchemy import Column, Integer, String, Date, BigInteger, Index, func
from app.database import Base


class Enrollment(Base):
    """
    Enrollment table model.
    
    Stores daily enrollment counts by state, district, pincode, and age group.
    Expected row count: ~1,000,000+ rows
    """
    __tablename__ = "enrollments"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Temporal
    date = Column(Date, nullable=False, index=True)
    
    # Geography
    state = Column(String(100), nullable=False, index=True)
    district = Column(String(100), nullable=False, index=True)
    pincode = Column(String(10), index=True)
    
    # Age-wise enrollment counts (matching CSV columns)
    age_0_5 = Column(BigInteger, default=0, nullable=False)
    age_5_17 = Column(BigInteger, default=0, nullable=False)
    age_18_plus = Column(BigInteger, default=0, nullable=False)  # age_18_greater in CSV
    
    # Total (computed)
    total = Column(BigInteger, nullable=False)
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_enroll_date_state', 'date', 'state'),
        Index('idx_enroll_date_district', 'date', 'district'),
        Index('idx_enroll_state_district', 'state', 'district'),
        Index('idx_enroll_date_state_district', 'date', 'state', 'district'),
    )
    
    def __repr__(self):
        return f"<Enrollment(id={self.id}, date={self.date}, district={self.district}, total={self.total})>"
    
    @property
    def age_breakdown(self) -> dict:
        """Return age-wise breakdown as dictionary."""
        return {
            "0-5": self.age_0_5,
            "5-17": self.age_5_17,
            "18+": self.age_18_plus
        }
