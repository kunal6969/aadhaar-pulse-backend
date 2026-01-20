"""
Train ONLY the Underserved Scoring Model.

This script trains just the underserved areas ML model with:
- Relative percentile ranking across all districts in India
- Proper MBU (Mobile Biometric Unit) recommendations
- State-level aggregations

Usage:
    python scripts/train_underserved_only.py
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.train_models_v2 import (
    get_db_stats,
    load_all_data,
    train_underserved_model
)

def main():
    print("=" * 70)
    print("🚐 TRAINING UNDERSERVED SCORING MODEL ONLY")
    print("=" * 70)
    
    # Get DB stats
    get_db_stats()
    
    # Load data
    enrollment_df, demographic_df, biometric_df = load_all_data()
    
    # Train only the underserved model
    result = train_underserved_model(enrollment_df, demographic_df, biometric_df)
    
    # Print summary
    print("\n" + "=" * 70)
    print("✅ UNDERSERVED MODEL TRAINING COMPLETE")
    print("=" * 70)
    
    if result and 'national' in result:
        national = result['national']
        print(f"\n📊 National Summary:")
        print(f"   Total Districts Scored: {national.get('total_districts', 0)}")
        print(f"   Avg Underserved Score: {national.get('avg_underserved_score', 0):.1f}")
        print(f"   Median Underserved Score: {national.get('median_underserved_score', 0):.1f}")
        print(f"   High Priority (>70): {national.get('high_underserved_count', 0)} districts")
        print(f"   Medium Priority (40-70): {national.get('medium_underserved_count', 0)} districts")
        print(f"   Low Priority (<40): {national.get('low_underserved_count', 0)} districts")
        print(f"   MBU Recommended: {national.get('mbu_recommended_total', 0)} districts")
    
    if result and 'rankings' in result:
        top_5 = result['rankings'].get('top_50_underserved', [])[:5]
        if top_5:
            print(f"\n🔝 Top 5 Most Underserved Districts:")
            for i, d in enumerate(top_5, 1):
                print(f"   {i}. {d['district']}, {d['state']} - Score: {d['underserved_score']:.1f}")
    
    print("\n" + "=" * 70)
    print("✅ Model saved to: app/ml_models/trained/underserved_scoring_v2.pkl")
    print("=" * 70)


if __name__ == "__main__":
    main()
