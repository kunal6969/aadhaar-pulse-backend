"""
FastAPI application entry point.

Aadhaar Pulse Simulator - Time-traveling analytics platform for 
Aadhaar enrollment and update data (Mar-Dec 2025).
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from app.config import settings, ensure_directories
from app.database import init_db
from app.routers import (
    enrollment,
    demographic,
    biometric,
    forecasting,
    anomaly,
    analytics,
    geospatial,
    ml_analytics,
    ml_analytics_v2
)
import os
import json

# Create FastAPI application
app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="""
    🕐 **Aadhaar Pulse Simulator API**
    
    A time-traveling analytics platform that serves data about India's 
    Aadhaar enrollment and updates from March 1, 2025 to December 31, 2025.
    
    ## Key Features
    
    * **Time Simulation**: Query data "as of" any date in the simulation range
    * **Enrollment Analytics**: Track Aadhaar enrollment by age group and geography
    * **Demographic Updates**: Monitor demographic data updates
    * **Biometric Updates**: Track MBU (Mandatory Biometric Update) data
    * **ML Forecasting**: Predict future demand using time-series models
    * **Anomaly Detection**: Identify unusual patterns in the data
    * **Geospatial Heatmaps**: Generate map visualization data
    
    ## Important Concept
    
    This is NOT a real-time API. All data is historical (Mar-Dec 2025).
    The frontend asks "What did the data look like on June 15, 2025?" 
    and receives data filtered to that simulation date.
    """,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize application on startup."""
    # Ensure directories exist
    ensure_directories()
    
    # Initialize database (create tables if not exist)
    try:
        init_db()
        print("✅ Database initialized successfully")
    except Exception as e:
        print(f"⚠️ Database initialization warning: {e}")


# Include routers with prefixes
app.include_router(
    enrollment.router,
    prefix=f"{settings.API_V1_PREFIX}/enrollment",
    tags=["Enrollment"]
)
app.include_router(
    demographic.router,
    prefix=f"{settings.API_V1_PREFIX}/demographic",
    tags=["Demographic Updates"]
)
app.include_router(
    biometric.router,
    prefix=f"{settings.API_V1_PREFIX}/biometric",
    tags=["Biometric Updates"]
)
app.include_router(
    forecasting.router,
    prefix=f"{settings.API_V1_PREFIX}/forecast",
    tags=["Forecasting"]
)
app.include_router(
    anomaly.router,
    prefix=f"{settings.API_V1_PREFIX}/anomaly",
    tags=["Anomaly Detection"]
)
app.include_router(
    analytics.router,
    prefix=f"{settings.API_V1_PREFIX}/analytics",
    tags=["Analytics & KPIs"]
)
app.include_router(
    geospatial.router,
    prefix=f"{settings.API_V1_PREFIX}/geospatial",
    tags=["Geospatial Data"]
)
app.include_router(
    ml_analytics.router,
    prefix=f"{settings.API_V1_PREFIX}/ml",
    tags=["ML Analytics"]
)
app.include_router(
    ml_analytics_v2.router,
    prefix=f"{settings.API_V1_PREFIX}/ml-v2",
    tags=["ML Analytics V2 (Hierarchical)"]
)


# Root endpoint
@app.get("/", tags=["Root"])
def root():
    """API root endpoint with basic information."""
    return {
        "message": "🕐 Aadhaar Pulse Simulator API",
        "version": settings.VERSION,
        "description": "Time-traveling analytics for Aadhaar enrollment data",
        "docs": "/docs",
        "redoc": "/redoc",
        "simulation_period": {
            "start": settings.SIMULATION_START_DATE,
            "end": settings.SIMULATION_END_DATE
        },
        "endpoints": {
            "enrollment": f"{settings.API_V1_PREFIX}/enrollment",
            "demographic": f"{settings.API_V1_PREFIX}/demographic",
            "biometric": f"{settings.API_V1_PREFIX}/biometric",
            "forecast": f"{settings.API_V1_PREFIX}/forecast",
            "anomaly": f"{settings.API_V1_PREFIX}/anomaly",
            "analytics": f"{settings.API_V1_PREFIX}/analytics",
            "geospatial": f"{settings.API_V1_PREFIX}/geospatial",
            "ml": f"{settings.API_V1_PREFIX}/ml",
            "ml_v2_hierarchical": f"{settings.API_V1_PREFIX}/ml-v2"
        }
    }


# Health check
@app.get(f"{settings.API_V1_PREFIX}/health", tags=["Health"])
def health_check():
    """Health check endpoint."""
    from app.database import engine
    from sqlalchemy import text
    
    # Check database connection
    try:
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        db_status = "healthy"
    except Exception as e:
        db_status = f"unhealthy: {str(e)}"
    
    return {
        "status": "healthy" if db_status == "healthy" else "degraded",
        "version": settings.VERSION,
        "database": db_status
    }


# Available dates endpoint
@app.get(f"{settings.API_V1_PREFIX}/available-dates", tags=["Metadata"])
def get_available_dates():
    """
    Return all dates that have data in the dataset.
    
    Use this to populate date pickers in the frontend.
    """
    dates_file = os.path.join(settings.PROCESSED_DATA_DIR, "available_dates.json")
    
    if os.path.exists(dates_file):
        with open(dates_file, 'r') as f:
            return json.load(f)
    else:
        # Generate from database if file doesn't exist
        from app.database import SessionLocal
        from app.models.enrollment import Enrollment
        from sqlalchemy import func
        
        db = SessionLocal()
        try:
            dates = db.query(Enrollment.date).distinct().order_by(Enrollment.date).all()
            available_dates = [d[0].strftime('%Y-%m-%d') for d in dates]
            
            result = {
                "dates": available_dates,
                "count": len(available_dates),
                "start": available_dates[0] if available_dates else None,
                "end": available_dates[-1] if available_dates else None
            }
            
            # Save for future use
            os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
            with open(dates_file, 'w') as f:
                json.dump(result, f, indent=2)
            
            return result
        except Exception as e:
            return {
                "error": f"Could not retrieve dates: {str(e)}",
                "hint": "Run the database initialization script first"
            }
        finally:
            db.close()


# Simulation info endpoint
@app.get(f"{settings.API_V1_PREFIX}/simulation-info", tags=["Metadata"])
def get_simulation_info():
    """
    Get information about the simulation period and data.
    """
    from app.database import SessionLocal
    from app.models.enrollment import Enrollment
    from app.models.demographic_update import DemographicUpdate
    from app.models.biometric_update import BiometricUpdate
    from sqlalchemy import func
    
    db = SessionLocal()
    try:
        # Get counts
        enrollment_count = db.query(func.count(Enrollment.id)).scalar() or 0
        demographic_count = db.query(func.count(DemographicUpdate.id)).scalar() or 0
        biometric_count = db.query(func.count(BiometricUpdate.id)).scalar() or 0
        
        # Get unique states and districts
        unique_states = db.query(func.count(func.distinct(Enrollment.state))).scalar() or 0
        unique_districts = db.query(func.count(func.distinct(Enrollment.district))).scalar() or 0
        
        return {
            "simulation_period": {
                "start": settings.SIMULATION_START_DATE,
                "end": settings.SIMULATION_END_DATE
            },
            "data_counts": {
                "enrollment_records": enrollment_count,
                "demographic_records": demographic_count,
                "biometric_records": biometric_count,
                "total_records": enrollment_count + demographic_count + biometric_count
            },
            "geography": {
                "unique_states": unique_states,
                "unique_districts": unique_districts
            },
            "api_version": settings.VERSION
        }
    except Exception as e:
        return {
            "error": f"Could not retrieve info: {str(e)}",
            "simulation_period": {
                "start": settings.SIMULATION_START_DATE,
                "end": settings.SIMULATION_END_DATE
            }
        }
    finally:
        db.close()


# Exception handlers
@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code
        }
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc),
            "status_code": 500
        }
    )
