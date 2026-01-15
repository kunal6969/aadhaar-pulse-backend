"""
Migration script: SQLite to Supabase using REST API

This script migrates data using Supabase's REST API (port 443)
which works even when PostgreSQL ports are blocked.
"""
import os
import sys
import sqlite3
import requests
import json
from datetime import datetime
import logging
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Supabase Configuration - UPDATE THESE
SUPABASE_URL = os.environ.get("SUPABASE_URL", "https://your-project.supabase.co")
SUPABASE_ANON_KEY = os.environ.get("SUPABASE_ANON_KEY", "your-anon-key")
SUPABASE_SERVICE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "your-service-key")

SQLITE_PATH = "data/aadhaar_pulse.db"
BATCH_SIZE = 500  # Smaller batches for REST API


def get_sqlite_connection():
    """Create SQLite connection."""
    return sqlite3.connect(SQLITE_PATH)


def supabase_request(endpoint: str, data: list, method: str = "POST"):
    """Make a request to Supabase REST API."""
    url = f"{SUPABASE_URL}/rest/v1/{endpoint}"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"  # Don't return data to speed up inserts
    }
    
    response = requests.request(method, url, headers=headers, json=data)
    
    if response.status_code not in [200, 201, 204]:
        logger.error(f"API Error: {response.status_code} - {response.text[:200]}")
        return False
    return True


def create_tables_via_sql():
    """
    Create tables using Supabase SQL Editor.
    
    Run this SQL in Supabase Dashboard → SQL Editor:
    """
    sql = """
-- Run this in Supabase SQL Editor (Dashboard → SQL Editor → New Query)

-- Drop existing tables
DROP TABLE IF EXISTS biometric_updates CASCADE;
DROP TABLE IF EXISTS demographic_updates CASCADE;
DROP TABLE IF EXISTS enrollments CASCADE;

-- Create enrollments table
CREATE TABLE enrollments (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    state VARCHAR(100),
    district VARCHAR(100),
    pincode VARCHAR(10),
    age_0_5 INTEGER DEFAULT 0,
    age_5_17 INTEGER DEFAULT 0,
    age_18_plus INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0
);

-- Create demographic_updates table
CREATE TABLE demographic_updates (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    state VARCHAR(100),
    district VARCHAR(100),
    pincode VARCHAR(10),
    demo_age_5_17 INTEGER DEFAULT 0,
    demo_age_17_plus INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0
);

-- Create biometric_updates table
CREATE TABLE biometric_updates (
    id SERIAL PRIMARY KEY,
    date DATE NOT NULL,
    state VARCHAR(100),
    district VARCHAR(100),
    pincode VARCHAR(10),
    bio_age_5_17 INTEGER DEFAULT 0,
    bio_age_17_plus INTEGER DEFAULT 0,
    total INTEGER DEFAULT 0
);

-- Create indexes for performance
CREATE INDEX idx_enrollments_date ON enrollments(date);
CREATE INDEX idx_enrollments_state ON enrollments(state);
CREATE INDEX idx_enrollments_date_state ON enrollments(date, state);

CREATE INDEX idx_demographic_date ON demographic_updates(date);
CREATE INDEX idx_demographic_state ON demographic_updates(state);
CREATE INDEX idx_demographic_date_state ON demographic_updates(date, state);

CREATE INDEX idx_biometric_date ON biometric_updates(date);
CREATE INDEX idx_biometric_state ON biometric_updates(state);
CREATE INDEX idx_biometric_date_state ON biometric_updates(date, state);

-- Enable Row Level Security (optional but recommended)
ALTER TABLE enrollments ENABLE ROW LEVEL SECURITY;
ALTER TABLE demographic_updates ENABLE ROW LEVEL SECURITY;
ALTER TABLE biometric_updates ENABLE ROW LEVEL SECURITY;

-- Create policies for public read access
CREATE POLICY "Allow public read access on enrollments" ON enrollments FOR SELECT USING (true);
CREATE POLICY "Allow public read access on demographic_updates" ON demographic_updates FOR SELECT USING (true);
CREATE POLICY "Allow public read access on biometric_updates" ON biometric_updates FOR SELECT USING (true);

-- Create policies for service role write access
CREATE POLICY "Allow service role insert on enrollments" ON enrollments FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow service role insert on demographic_updates" ON demographic_updates FOR INSERT WITH CHECK (true);
CREATE POLICY "Allow service role insert on biometric_updates" ON biometric_updates FOR INSERT WITH CHECK (true);
    """
    print("\n" + "=" * 60)
    print("⚠️  STEP 1: CREATE TABLES IN SUPABASE")
    print("=" * 60)
    print("\n1. Go to Supabase Dashboard")
    print("2. Click 'SQL Editor' in the sidebar")
    print("3. Click 'New Query'")
    print("4. Paste and run this SQL:\n")
    print("-" * 60)
    print(sql)
    print("-" * 60)
    print("\n5. Click 'Run' to execute")
    print("6. Come back here and press Enter to continue...\n")
    

def migrate_enrollments(sqlite_conn):
    """Migrate enrollment data."""
    logger.info("Migrating enrollments...")
    
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM enrollments")
    total = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT date, state, district, pincode, 
               age_0_5, age_5_17, age_18_plus, total
        FROM enrollments
    """)
    
    batch = []
    success_count = 0
    
    for row in tqdm(cursor.fetchall(), total=total, desc="  Enrollments"):
        record = {
            "date": row[0],
            "state": row[1],
            "district": row[2],
            "pincode": row[3],
            "age_0_5": row[4] or 0,
            "age_5_17": row[5] or 0,
            "age_18_plus": row[6] or 0,
            "total": row[7] or 0
        }
        batch.append(record)
        
        if len(batch) >= BATCH_SIZE:
            if supabase_request("enrollments", batch):
                success_count += len(batch)
            batch = []
    
    # Insert remaining
    if batch:
        if supabase_request("enrollments", batch):
            success_count += len(batch)
    
    logger.info(f"  ✅ Migrated {success_count:,} enrollment records")
    return success_count


def migrate_demographic(sqlite_conn):
    """Migrate demographic data."""
    logger.info("Migrating demographic updates...")
    
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM demographic_updates")
    total = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT date, state, district, pincode,
               demo_age_5_17, demo_age_17_plus, total
        FROM demographic_updates
    """)
    
    batch = []
    success_count = 0
    
    for row in tqdm(cursor.fetchall(), total=total, desc="  Demographic"):
        record = {
            "date": row[0],
            "state": row[1],
            "district": row[2],
            "pincode": row[3],
            "demo_age_5_17": row[4] or 0,
            "demo_age_17_plus": row[5] or 0,
            "total": row[6] or 0
        }
        batch.append(record)
        
        if len(batch) >= BATCH_SIZE:
            if supabase_request("demographic_updates", batch):
                success_count += len(batch)
            batch = []
    
    if batch:
        if supabase_request("demographic_updates", batch):
            success_count += len(batch)
    
    logger.info(f"  ✅ Migrated {success_count:,} demographic records")
    return success_count


def migrate_biometric(sqlite_conn):
    """Migrate biometric data."""
    logger.info("Migrating biometric updates...")
    
    cursor = sqlite_conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM biometric_updates")
    total = cursor.fetchone()[0]
    
    cursor.execute("""
        SELECT date, state, district, pincode,
               bio_age_5_17, bio_age_17_plus, total
        FROM biometric_updates
    """)
    
    batch = []
    success_count = 0
    
    for row in tqdm(cursor.fetchall(), total=total, desc="  Biometric"):
        record = {
            "date": row[0],
            "state": row[1],
            "district": row[2],
            "pincode": row[3],
            "bio_age_5_17": row[4] or 0,
            "bio_age_17_plus": row[5] or 0,
            "total": row[6] or 0
        }
        batch.append(record)
        
        if len(batch) >= BATCH_SIZE:
            if supabase_request("biometric_updates", batch):
                success_count += len(batch)
            batch = []
    
    if batch:
        if supabase_request("biometric_updates", batch):
            success_count += len(batch)
    
    logger.info(f"  ✅ Migrated {success_count:,} biometric records")
    return success_count


def test_connection():
    """Test Supabase connection."""
    url = f"{SUPABASE_URL}/rest/v1/"
    headers = {
        "apikey": SUPABASE_SERVICE_KEY,
        "Authorization": f"Bearer {SUPABASE_SERVICE_KEY}"
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 200:
            return True
        else:
            logger.error(f"Connection test failed: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"Connection error: {e}")
        return False


def main():
    print("=" * 60)
    print("🚀 SQLite to Supabase Migration (REST API)")
    print("=" * 60)
    
    # Check if keys are configured
    if "YOUR_" in SUPABASE_SERVICE_KEY or "YOUR_" in SUPABASE_ANON_KEY:
        print("\n⚠️  FIRST: Get your API keys from Supabase!")
        print("\n1. Go to: Supabase Dashboard → Settings → API")
        print("2. Copy 'anon public' key → paste as SUPABASE_ANON_KEY")
        print("3. Copy 'service_role' key → paste as SUPABASE_SERVICE_KEY")
        print("4. Update this script with the keys")
        print("5. Run this script again\n")
        return
    
    # Test connection
    print("\n🔌 Testing Supabase connection...")
    if not test_connection():
        print("❌ Could not connect to Supabase. Check your keys.")
        return
    print("✅ Connected to Supabase!")
    
    # Show table creation instructions
    create_tables_via_sql()
    input("Press Enter after creating tables in Supabase...")
    
    # Connect to SQLite
    print("\n📂 Connecting to SQLite...")
    sqlite_conn = get_sqlite_connection()
    print("✅ Connected to SQLite")
    
    # Migrate data
    print("\n📤 Migrating data to Supabase...")
    print("-" * 40)
    
    enrollment_count = migrate_enrollments(sqlite_conn)
    demographic_count = migrate_demographic(sqlite_conn)
    biometric_count = migrate_biometric(sqlite_conn)
    
    sqlite_conn.close()
    
    # Summary
    print("\n" + "=" * 60)
    print("✅ Migration Complete!")
    print("-" * 40)
    print(f"  📊 Enrollments: {enrollment_count:,}")
    print(f"  📊 Demographic: {demographic_count:,}")
    print(f"  📊 Biometric: {biometric_count:,}")
    print(f"  📊 Total: {enrollment_count + demographic_count + biometric_count:,}")
    print("=" * 60)
    print("\n🎉 Your data is now in Supabase!")
    print("   Next: Deploy your app to Render with USE_POSTGRES=true")


if __name__ == "__main__":
    main()
