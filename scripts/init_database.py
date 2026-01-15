"""
Database initialization script.

This script loads data from the CSV files into the SQLite database.
Run this once before starting the API server.
"""
import os
import sys
import pandas as pd
from datetime import datetime
from tqdm import tqdm
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.config import settings
from app.database import engine, SessionLocal, init_db, Base
from app.models.enrollment import Enrollment
from app.models.demographic_update import DemographicUpdate
from app.models.biometric_update import BiometricUpdate

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_date(date_str: str) -> datetime:
    """Parse date string in DD-MM-YYYY format to datetime."""
    try:
        return datetime.strptime(date_str, '%d-%m-%Y')
    except ValueError:
        try:
            # Try alternative format
            return datetime.strptime(date_str, '%Y-%m-%d')
        except ValueError:
            logger.warning(f"Could not parse date: {date_str}")
            return None


def clean_string(value) -> str:
    """Clean string values."""
    if pd.isna(value):
        return None
    return str(value).strip()


def clean_int(value) -> int:
    """Clean integer values."""
    if pd.isna(value):
        return 0
    try:
        return int(float(value))
    except (ValueError, TypeError):
        return 0


def load_enrollment_data(db, csv_dir: str, batch_size: int = 5000):
    """Load enrollment data from CSV files."""
    logger.info("Loading enrollment data...")
    
    # Find all enrollment CSV files
    csv_files = []
    for f in os.listdir(csv_dir):
        if f.endswith('.csv'):
            csv_files.append(os.path.join(csv_dir, f))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}")
        return 0
    
    total_records = 0
    
    for csv_file in sorted(csv_files):
        logger.info(f"  Processing: {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            # Process in batches
            records = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="    Rows"):
                date = parse_date(str(row.get('date', '')))
                if date is None:
                    continue
                
                record = Enrollment(
                    date=date,
                    state=clean_string(row.get('state')),
                    district=clean_string(row.get('district')),
                    pincode=clean_string(row.get('pincode')),
                    age_0_5=clean_int(row.get('age_0_5', 0)),
                    age_5_17=clean_int(row.get('age_5_17', 0)),
                    age_18_plus=clean_int(row.get('age_18_greater', 0)),  # CSV uses age_18_greater
                    total=(
                        clean_int(row.get('age_0_5', 0)) +
                        clean_int(row.get('age_5_17', 0)) +
                        clean_int(row.get('age_18_greater', 0))
                    )
                )
                records.append(record)
                
                if len(records) >= batch_size:
                    db.bulk_save_objects(records)
                    db.commit()
                    total_records += len(records)
                    records = []
            
            # Save remaining records
            if records:
                db.bulk_save_objects(records)
                db.commit()
                total_records += len(records)
                
        except Exception as e:
            logger.error(f"  Error processing {csv_file}: {e}")
            db.rollback()
    
    logger.info(f"✅ Loaded {total_records} enrollment records")
    return total_records


def load_demographic_data(db, csv_dir: str, batch_size: int = 5000):
    """Load demographic update data from CSV files."""
    logger.info("Loading demographic update data...")
    
    # Find all demographic CSV files
    csv_files = []
    for f in os.listdir(csv_dir):
        if f.endswith('.csv'):
            csv_files.append(os.path.join(csv_dir, f))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}")
        return 0
    
    total_records = 0
    
    for csv_file in sorted(csv_files):
        logger.info(f"  Processing: {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            records = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="    Rows"):
                date = parse_date(str(row.get('date', '')))
                if date is None:
                    continue
                
                record = DemographicUpdate(
                    date=date,
                    state=clean_string(row.get('state')),
                    district=clean_string(row.get('district')),
                    pincode=clean_string(row.get('pincode')),
                    demo_age_5_17=clean_int(row.get('demo_age_5_17', 0)),
                    demo_age_17_plus=clean_int(row.get('demo_age_17_', 0)),  # CSV uses demo_age_17_
                    total=(
                        clean_int(row.get('demo_age_5_17', 0)) +
                        clean_int(row.get('demo_age_17_', 0))
                    )
                )
                records.append(record)
                
                if len(records) >= batch_size:
                    db.bulk_save_objects(records)
                    db.commit()
                    total_records += len(records)
                    records = []
            
            # Save remaining records
            if records:
                db.bulk_save_objects(records)
                db.commit()
                total_records += len(records)
                
        except Exception as e:
            logger.error(f"  Error processing {csv_file}: {e}")
            db.rollback()
    
    logger.info(f"✅ Loaded {total_records} demographic records")
    return total_records


def load_biometric_data(db, csv_dir: str, batch_size: int = 5000):
    """Load biometric update data from CSV files."""
    logger.info("Loading biometric update data...")
    
    # Find all biometric CSV files
    csv_files = []
    for f in os.listdir(csv_dir):
        if f.endswith('.csv'):
            csv_files.append(os.path.join(csv_dir, f))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_dir}")
        return 0
    
    total_records = 0
    
    for csv_file in sorted(csv_files):
        logger.info(f"  Processing: {os.path.basename(csv_file)}")
        
        try:
            df = pd.read_csv(csv_file, low_memory=False)
            
            records = []
            for idx, row in tqdm(df.iterrows(), total=len(df), desc="    Rows"):
                date = parse_date(str(row.get('date', '')))
                if date is None:
                    continue
                
                record = BiometricUpdate(
                    date=date,
                    state=clean_string(row.get('state')),
                    district=clean_string(row.get('district')),
                    pincode=clean_string(row.get('pincode')),
                    bio_age_5_17=clean_int(row.get('bio_age_5_17', 0)),
                    bio_age_17_plus=clean_int(row.get('bio_age_17_', 0)),  # CSV uses bio_age_17_
                    total=(
                        clean_int(row.get('bio_age_5_17', 0)) +
                        clean_int(row.get('bio_age_17_', 0))
                    )
                )
                records.append(record)
                
                if len(records) >= batch_size:
                    db.bulk_save_objects(records)
                    db.commit()
                    total_records += len(records)
                    records = []
            
            # Save remaining records
            if records:
                db.bulk_save_objects(records)
                db.commit()
                total_records += len(records)
                
        except Exception as e:
            logger.error(f"  Error processing {csv_file}: {e}")
            db.rollback()
    
    logger.info(f"✅ Loaded {total_records} biometric records")
    return total_records


def generate_available_dates(db):
    """Generate the available_dates.json file."""
    import json
    
    logger.info("Generating available dates file...")
    
    try:
        dates = db.query(Enrollment.date).distinct().order_by(Enrollment.date).all()
        available_dates = [d[0].strftime('%Y-%m-%d') for d in dates]
        
        result = {
            "dates": available_dates,
            "count": len(available_dates),
            "start": available_dates[0] if available_dates else None,
            "end": available_dates[-1] if available_dates else None
        }
        
        # Save to file
        os.makedirs(settings.PROCESSED_DATA_DIR, exist_ok=True)
        dates_file = os.path.join(settings.PROCESSED_DATA_DIR, "available_dates.json")
        
        with open(dates_file, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"✅ Generated available_dates.json with {len(available_dates)} dates")
        
    except Exception as e:
        logger.error(f"Error generating available dates: {e}")


def main():
    """Main initialization function."""
    print("=" * 60)
    print("🚀 Aadhaar Pulse Simulator - Database Initialization")
    print("=" * 60)
    
    # Get CSV directories from settings
    enrollment_csv_dir = settings.ENROLLMENT_CSV_DIR
    demographic_csv_dir = settings.DEMOGRAPHIC_CSV_DIR
    biometric_csv_dir = settings.BIOMETRIC_CSV_DIR
    
    # Verify directories exist
    print("\n📂 Checking CSV directories...")
    dirs_ok = True
    for name, path in [
        ("Enrollment", enrollment_csv_dir),
        ("Demographic", demographic_csv_dir),
        ("Biometric", biometric_csv_dir)
    ]:
        if os.path.exists(path):
            csv_count = len([f for f in os.listdir(path) if f.endswith('.csv')])
            print(f"  ✅ {name}: {path} ({csv_count} CSV files)")
        else:
            print(f"  ❌ {name}: {path} (NOT FOUND)")
            dirs_ok = False
    
    if not dirs_ok:
        print("\n❌ Some directories are missing. Please check the .env file.")
        print("   Expected paths:")
        print(f"   ENROLLMENT_CSV_DIR={enrollment_csv_dir}")
        print(f"   DEMOGRAPHIC_CSV_DIR={demographic_csv_dir}")
        print(f"   BIOMETRIC_CSV_DIR={biometric_csv_dir}")
        sys.exit(1)
    
    # Initialize database tables
    print("\n📊 Initializing database...")
    init_db()
    print(f"  ✅ Database created at: {settings.database_url}")
    
    # Create session
    db = SessionLocal()
    
    try:
        # Check if data already exists
        existing_enrollment = db.query(Enrollment).count()
        if existing_enrollment > 0:
            print(f"\n⚠️  Database already contains {existing_enrollment} enrollment records.")
            response = input("   Do you want to clear and reload? (y/N): ").strip().lower()
            if response == 'y':
                print("   Clearing existing data...")
                db.query(BiometricUpdate).delete()
                db.query(DemographicUpdate).delete()
                db.query(Enrollment).delete()
                db.commit()
                print("   ✅ Cleared all existing data")
            else:
                print("   Keeping existing data. Exiting.")
                return
        
        print("\n📥 Loading data from CSV files...")
        print("-" * 40)
        
        # Load each data type
        enrollment_count = load_enrollment_data(db, enrollment_csv_dir)
        demographic_count = load_demographic_data(db, demographic_csv_dir)
        biometric_count = load_biometric_data(db, biometric_csv_dir)
        
        # Generate available dates file
        generate_available_dates(db)
        
        print("\n" + "=" * 60)
        print("✅ Database initialization complete!")
        print("-" * 40)
        print(f"  📊 Enrollment records: {enrollment_count:,}")
        print(f"  📊 Demographic records: {demographic_count:,}")
        print(f"  📊 Biometric records: {biometric_count:,}")
        print(f"  📊 Total records: {enrollment_count + demographic_count + biometric_count:,}")
        print("=" * 60)
        print("\n🚀 You can now start the API server with: python run.py")
        
    except Exception as e:
        logger.error(f"Error during initialization: {e}")
        db.rollback()
        raise
    finally:
        db.close()


if __name__ == "__main__":
    main()
