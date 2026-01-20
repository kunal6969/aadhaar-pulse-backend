"""
Test script for V2 ML Analytics API endpoints.
Run this while the server is running in a separate terminal.
"""
import requests
import json

BASE_URL = "http://localhost:8002/api/v1/ml-v2"

def test_endpoint(name, url, params=None):
    """Test an endpoint and print results."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"URL: {url}")
    if params:
        print(f"Params: {params}")
    print("-" * 60)
    
    try:
        r = requests.get(url, params=params, timeout=10)
        if r.status_code == 200:
            data = r.json()
            # Print summary
            if isinstance(data, dict):
                print(f"✅ Success! Keys: {list(data.keys())}")
                # Show sample data
                for key in list(data.keys())[:3]:
                    val = data[key]
                    if isinstance(val, list) and len(val) > 0:
                        print(f"  - {key}: {len(val)} items, first: {val[0] if len(str(val[0])) < 100 else '...'}")
                    elif isinstance(val, dict):
                        print(f"  - {key}: {list(val.keys())[:5]}...")
                    else:
                        print(f"  - {key}: {val}")
            else:
                print(f"✅ Success! Response: {str(data)[:200]}")
        else:
            print(f"❌ Error {r.status_code}: {r.text[:200]}")
    except Exception as e:
        print(f"❌ Exception: {e}")

def main():
    print("🧪 Testing ML Analytics V2 API Endpoints")
    print("=" * 60)
    
    # 1. Status
    test_endpoint("Status", f"{BASE_URL}/status")
    
    # 2. Top 50 Needy Districts
    test_endpoint("Top 50 Needy Districts", f"{BASE_URL}/rankings/top-50-needy")
    
    # 3. State Comparison
    test_endpoint("State Comparison (All States)", f"{BASE_URL}/compare/states", 
                  {"metric": "composite_score"})
    
    # 4. District Comparison within State
    test_endpoint("District Comparison (Maharashtra)", f"{BASE_URL}/compare/districts",
                  {"state": "Maharashtra", "metric": "underserved_score"})
    
    # 5. Monthly Comparison
    test_endpoint("Monthly Comparison", f"{BASE_URL}/monthly-comparison",
                  {"metric": "total_transactions"})
    
    # 6. Capacity Planning
    test_endpoint("Capacity Planning (National)", f"{BASE_URL}/capacity")
    
    # 7. Underserved Scoring
    test_endpoint("Underserved Scoring", f"{BASE_URL}/underserved", {"limit": 10})
    
    # 8. Fraud Detection
    test_endpoint("Fraud Detection", f"{BASE_URL}/fraud", {"limit": 10})
    
    # 9. Clustering
    test_endpoint("District Clustering", f"{BASE_URL}/clustering")
    
    # 10. Hotspots
    test_endpoint("Demand Hotspots", f"{BASE_URL}/hotspots")
    
    # 11. MBU Projection
    test_endpoint("5-Year MBU Projection", f"{BASE_URL}/mbu-projection")
    
    # 12. Forecast
    test_endpoint("Demand Forecast", f"{BASE_URL}/forecast")
    
    print("\n" + "=" * 60)
    print("🏁 Testing Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
