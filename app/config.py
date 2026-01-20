"""
Application configuration settings.
"""
from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List
import os


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""
    
    # Database - PostgreSQL for production, SQLite for local
    DATABASE_URL: str = ""  # PostgreSQL connection string (production)
    DATABASE_PATH: str = "data/aadhaar_pulse.db"  # SQLite path (local fallback)
    USE_POSTGRES: bool = False  # Set to True to use PostgreSQL
    
    # Data paths
    RAW_DATA_DIR: str = "data/raw"
    PROCESSED_DATA_DIR: str = "data/processed"
    GEOJSON_DIR: str = "data/geojson"
    
    # CSV Source directories (relative to project root)
    ENROLLMENT_CSV_DIR: str = "../api_data_aadhar_enrolment/api_data_aadhar_enrolment"
    DEMOGRAPHIC_CSV_DIR: str = "../api_data_aadhar_demographic/api_data_aadhar_demographic"
    BIOMETRIC_CSV_DIR: str = "../api_data_aadhar_biometric/api_data_aadhar_biometric"
    
    # ML models
    ML_MODELS_DIR: str = "app/ml_models/trained"
    
    # API settings
    API_V1_PREFIX: str = "/api"
    PROJECT_NAME: str = "Aadhaar Pulse Simulator"
    VERSION: str = "1.0.0"
    
    # CORS
    ALLOWED_ORIGINS: List[str] = [
        "http://localhost:5173",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:3000",
        "https://aadhaar-pulse-react.vercel.app",
        "https://aadhaar-pulse-frontend.vercel.app",
        "*"
    ]
    
    # Simulation date range
    SIMULATION_START_DATE: str = "2025-06-01"
    SIMULATION_END_DATE: str = "2025-12-31"
    
    # Pagination
    DEFAULT_PAGE_SIZE: int = 50
    MAX_PAGE_SIZE: int = 1000
    
    @property
    def database_url(self) -> str:
        """Get database URL - PostgreSQL if configured, else SQLite."""
        if self.USE_POSTGRES and self.DATABASE_URL:
            return self.DATABASE_URL
        return f"sqlite:///{self.DATABASE_PATH}"
    
    @property
    def is_postgres(self) -> bool:
        """Check if using PostgreSQL."""
        return self.USE_POSTGRES and bool(self.DATABASE_URL)
    
    @property
    def base_dir(self) -> Path:
        """Get base directory of the project."""
        return Path(__file__).parent.parent
    
    class Config:
        env_file = ".env"
        extra = "allow"


# Global settings instance
settings = Settings()


# Ensure directories exist
def ensure_directories():
    """Create necessary directories if they don't exist."""
    dirs = [
        settings.RAW_DATA_DIR,
        settings.PROCESSED_DATA_DIR,
        settings.GEOJSON_DIR,
        settings.ML_MODELS_DIR,
    ]
    for dir_path in dirs:
        os.makedirs(dir_path, exist_ok=True)
