"""
Run script to start the FastAPI server (no reload).
"""
import uvicorn
import os
import sys

# Add the project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    """Start the Uvicorn server."""
    print("🚀 Starting Aadhaar Pulse Simulator API...")
    print("📖 API Documentation: http://localhost:8000/docs")
    print("📊 ReDoc: http://localhost:8000/redoc")
    print("-" * 50)
    
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,  # No reload for stability
        log_level="info"
    )


if __name__ == "__main__":
    main()
