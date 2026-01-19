"""
Test ML Forecasting Models Locally

This script tests the forecasting logic using local SQLite database.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from datetime import date, timedelta
import pandas as pd
import numpy as np

# Check if we have data locally
def test_forecasting_logic():
    """Test the forecasting algorithm with sample data."""
    
    print("=" * 60)
    print("AADHAAR PULSE - ML FORECASTING TEST")
    print("=" * 60)
    
    # Generate sample historical data (simulating biometric updates)
    np.random.seed(42)
    
    # Create 180 days of historical data with trend and seasonality
    dates = pd.date_range(end=date(2025, 9, 1), periods=180)
    
    # Base level + trend + weekly seasonality + noise
    base = 500
    trend = np.linspace(0, 100, 180)  # Increasing trend
    weekly = np.array([100 if d.weekday() < 5 else -150 for d in dates])  # Weekend dip
    noise = np.random.normal(0, 50, 180)
    
    values = base + trend + weekly + noise
    values = np.maximum(values, 0)  # No negatives
    
    historical_df = pd.DataFrame({
        'date': dates,
        'total': values.astype(int)
    })
    
    print(f"\n📊 Historical Data Summary (last 180 days):")
    print(f"   - Records: {len(historical_df)}")
    print(f"   - Date Range: {historical_df['date'].min().date()} to {historical_df['date'].max().date()}")
    print(f"   - Mean: {historical_df['total'].mean():.0f}")
    print(f"   - Std: {historical_df['total'].std():.0f}")
    print(f"   - Min: {historical_df['total'].min()}")
    print(f"   - Max: {historical_df['total'].max()}")
    
    # Forecasting parameters
    forecast_from = date(2025, 9, 1)
    horizon_days = 90
    
    print(f"\n🔮 Forecasting Parameters:")
    print(f"   - Forecast From: {forecast_from}")
    print(f"   - Horizon: {horizon_days} days")
    print(f"   - Forecast Until: {forecast_from + timedelta(days=horizon_days)}")
    
    # Apply the same forecasting logic as the API
    values_arr = historical_df['total'].values
    mean_value = values_arr.mean()
    
    # Calculate trend (difference between last 7 days avg and first 7 days avg)
    trend_calc = (values_arr[-7:].mean() - values_arr[:7].mean()) / max(len(values_arr) - 7, 1)
    
    print(f"\n📈 Model Parameters:")
    print(f"   - Base Mean: {mean_value:.0f}")
    print(f"   - Trend: {trend_calc:.2f} per day")
    print(f"   - Std Dev: {values_arr.std():.0f}")
    
    # Generate forecast
    forecast = []
    for i in range(horizon_days):
        forecast_date = forecast_from + timedelta(days=i+1)
        
        # Simple prediction: mean + trend * days
        predicted = mean_value + (trend_calc * i)
        
        # Add weekly seasonality (weekends have less activity)
        day_of_week = forecast_date.weekday()
        if day_of_week in [5, 6]:  # Weekend
            predicted *= 0.7
        
        # Ensure non-negative
        predicted = max(0, predicted)
        
        # Confidence bounds (95%)
        std = values_arr.std()
        lower = max(0, predicted - 1.96 * std)
        upper = predicted + 1.96 * std
        
        forecast.append({
            "date": forecast_date,
            "predicted": round(predicted, 0),
            "lower_bound": round(lower, 0),
            "upper_bound": round(upper, 0)
        })
    
    forecast_df = pd.DataFrame(forecast)
    
    print(f"\n🎯 Forecast Summary:")
    print(f"   - First 7 days prediction avg: {forecast_df.head(7)['predicted'].mean():.0f}")
    print(f"   - Last 7 days prediction avg: {forecast_df.tail(7)['predicted'].mean():.0f}")
    print(f"   - Overall prediction avg: {forecast_df['predicted'].mean():.0f}")
    
    # Show sample predictions
    print(f"\n📅 Sample Predictions:")
    print("-" * 60)
    print(f"{'Date':<15} {'Predicted':<12} {'Lower':<12} {'Upper':<12}")
    print("-" * 60)
    
    # First 5 days
    for _, row in forecast_df.head(5).iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d'):<15} {row['predicted']:<12.0f} {row['lower_bound']:<12.0f} {row['upper_bound']:<12.0f}")
    
    print("...")
    
    # Last 5 days
    for _, row in forecast_df.tail(5).iterrows():
        print(f"{row['date'].strftime('%Y-%m-%d'):<15} {row['predicted']:<12.0f} {row['lower_bound']:<12.0f} {row['upper_bound']:<12.0f}")
    
    print("-" * 60)
    
    # Risk Assessment
    print(f"\n⚠️  Risk Assessment:")
    recent_updates = values_arr[-30:].sum()
    expected_mbu = 10000  # Simulated
    monthly_expected = expected_mbu / 6
    completion_ratio = recent_updates / monthly_expected if monthly_expected > 0 else 1
    
    if completion_ratio < 0.3:
        risk_level = "HIGH 🔴"
    elif completion_ratio < 0.7:
        risk_level = "MEDIUM 🟡"
    else:
        risk_level = "LOW 🟢"
    
    print(f"   - Risk Level: {risk_level}")
    print(f"   - Completion Ratio: {completion_ratio:.2%}")
    print(f"   - Recent 30-day Updates: {recent_updates:.0f}")
    print(f"   - Expected Monthly: {monthly_expected:.0f}")
    
    # Model info
    print(f"\n🤖 Model Information:")
    print(f"   - Type: Simple Trend + Seasonality")
    print(f"   - Training Data: {len(historical_df)} days")
    print(f"   - Features: Weekly seasonality, Linear trend")
    print(f"   - Note: For production, integrate Prophet or ARIMA")
    
    print("\n" + "=" * 60)
    print("✅ Forecasting test completed successfully!")
    print("=" * 60)
    
    return forecast_df


def test_with_real_database():
    """Test with actual database if available."""
    try:
        from app.database import SessionLocal, engine
        from app.models.biometric_update import BiometricUpdate
        from sqlalchemy import func
        
        print("\n" + "=" * 60)
        print("TESTING WITH ACTUAL DATABASE")
        print("=" * 60)
        
        db = SessionLocal()
        
        # Get a sample district
        sample = db.query(BiometricUpdate.district).first()
        if sample:
            district = sample.district
            print(f"\nUsing district: {district}")
            
            # Get data count
            count = db.query(func.count(BiometricUpdate.id)).filter(
                BiometricUpdate.district == district
            ).scalar()
            print(f"Records for this district: {count}")
            
            # Get date range
            min_date = db.query(func.min(BiometricUpdate.date)).filter(
                BiometricUpdate.district == district
            ).scalar()
            max_date = db.query(func.max(BiometricUpdate.date)).filter(
                BiometricUpdate.district == district
            ).scalar()
            print(f"Date range: {min_date} to {max_date}")
        else:
            print("No data found in database")
        
        db.close()
        
    except Exception as e:
        print(f"\nCould not connect to database: {e}")
        print("Running with simulated data only.")


if __name__ == "__main__":
    # Test with simulated data
    forecast_df = test_forecasting_logic()
    
    # Try to test with real database
    test_with_real_database()
