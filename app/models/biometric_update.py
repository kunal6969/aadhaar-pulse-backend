"""
Biometric Update SQLAlchemy model.
Stores biometric update records by date, geography, and age group.
"""
from sqlalchemy import Column, Integer, String, Date, BigInteger, Index
from app.database import Base


class BiometricUpdate(Base):
    """
    Biometric Update table model.
    
    Stores daily biometric update counts by state, district, pincode, and age group.
    Based on CSV columns: date, state, district, pincode, bio_age_5_17, bio_age_17_
    Expected row count: ~1,800,000+ rows
    """
    __tablename__ = "biometric_updates"
    
    # Primary key
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    
    # Temporal
    date = Column(Date, nullable=False, index=True)
    
    # Geography
    state = Column(String(100), nullable=False, index=True)
    district = Column(String(100), nullable=False, index=True)
    pincode = Column(String(10), index=True)
    
    # Age-wise biometric update counts (matching CSV columns)
    bio_age_5_17 = Column(BigInteger, default=0, nullable=False)
    bio_age_17_plus = Column(BigInteger, default=0, nullable=False)  # bio_age_17_ in CSV
    
    # Total (computed)
    total = Column(BigInteger, nullable=False)
    
    # Composite indexes for common queries
    __table_args__ = (
        Index('idx_bio_date_state', 'date', 'state'),
        Index('idx_bio_date_district', 'date', 'district'),
        Index('idx_bio_state_district', 'state', 'district'),
        Index('idx_bio_date_state_district', 'date', 'state', 'district'),
    )
    
    def __repr__(self):
        return f"<BiometricUpdate(id={self.id}, date={self.date}, district={self.district}, total={self.total})>"
    
    @property
    def age_breakdown(self) -> dict:
        """Return age-wise breakdown as dictionary."""
        return {
            "5-17": self.bio_age_5_17,
            "17+": self.bio_age_17_plus
        }
