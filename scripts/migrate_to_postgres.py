"""
Migration script: SQLite to PostgreSQL (Supabase)

This script migrates data from your local SQLite database to Supabase PostgreSQL.
Run this once after setting up Supabase.
"""
import os
import sys
import sqlite3
from datetime import datetime
import logging

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# PostgreSQL connection string - UPDATE THIS
# Using Session Pooler URL for better connectivity
POSTGRES_URL = os.environ.get("DATABASE_URL", "postgresql://user:password@host:5432/database")
SQLITE_PATH = "data/aadhaar_pulse.db"
BATCH_SIZE = 5000


def get_postgres_connection():
    """Create PostgreSQL connection using psycopg2."""
    import psycopg2
    return psycopg2.connect(POSTGRES_URL)


def get_sqlite_connection():
    """Create SQLite connection."""
    return sqlite3.connect(SQLITE_PATH)


def create_tables_postgres(pg_conn):
    """Create tables in PostgreSQL."""
    cursor = pg_conn.cursor()
    
    # Drop existing tables if they exist
    cursor.execute("DROP TABLE IF EXISTS biometric_updates CASCADE")
    cursor.execute("DROP TABLE IF EXISTS demographic_updates CASCADE")
    cursor.execute("DROP TABLE IF EXISTS enrollments CASCADE")
    
    # Create enrollments table
    cursor.execute("""
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
        )
    """)
    
    # Create demographic_updates table
    cursor.execute("""
        CREATE TABLE demographic_updates (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            state VARCHAR(100),
            district VARCHAR(100),
            pincode VARCHAR(10),
            demo_age_5_17 INTEGER DEFAULT 0,
            demo_age_17_plus INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0
        )
    """)
    
    # Create biometric_updates table
    cursor.execute("""
        CREATE TABLE biometric_updates (
            id SERIAL PRIMARY KEY,
            date DATE NOT NULL,
            state VARCHAR(100),
            district VARCHAR(100),
            pincode VARCHAR(10),
            bio_age_5_17 INTEGER DEFAULT 0,
            bio_age_17_plus INTEGER DEFAULT 0,
            total INTEGER DEFAULT 0
        )
    """)
    
    # Create indexes for faster queries
    cursor.execute("CREATE INDEX idx_enrollments_date ON enrollments(date)")
    cursor.execute("CREATE INDEX idx_enrollments_state ON enrollments(state)")
    cursor.execute("CREATE INDEX idx_enrollments_district ON enrollments(district)")
    cursor.execute("CREATE INDEX idx_enrollments_date_state ON enrollments(date, state)")
    
    cursor.execute("CREATE INDEX idx_demographic_date ON demographic_updates(date)")
    cursor.execute("CREATE INDEX idx_demographic_state ON demographic_updates(state)")
    cursor.execute("CREATE INDEX idx_demographic_date_state ON demographic_updates(date, state)")
    
    cursor.execute("CREATE INDEX idx_biometric_date ON biometric_updates(date)")
    cursor.execute("CREATE INDEX idx_biometric_state ON biometric_updates(state)")
    cursor.execute("CREATE INDEX idx_biometric_date_state ON biometric_updates(date, state)")
    
    pg_conn.commit()
    logger.info("✅ Created tables and indexes in PostgreSQL")


def migrate_enrollments(sqlite_conn, pg_conn):
    """Migrate enrollments table."""
    logger.info("Migrating enrollments...")
    
    sqlite_cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    # Get total count
    sqlite_cursor.execute("SELECT COUNT(*) FROM enrollments")
    total = sqlite_cursor.fetchone()[0]
    logger.info(f"  Total records to migrate: {total:,}")
    
    # Fetch and insert in batches
    offset = 0
    migrated = 0
    
    while offset < total:
        sqlite_cursor.execute(f"""
            SELECT date, state, district, pincode, age_0_5, age_5_17, age_18_plus, total
            FROM enrollments
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """)
        
        rows = sqlite_cursor.fetchall()
        if not rows:
            break
        
        # Insert into PostgreSQL
        for row in rows:
            pg_cursor.execute("""
                INSERT INTO enrollments (date, state, district, pincode, age_0_5, age_5_17, age_18_plus, total)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, row)
        
        pg_conn.commit()
        migrated += len(rows)
        offset += BATCH_SIZE
        
        progress = (migrated / total) * 100
        logger.info(f"  Progress: {migrated:,}/{total:,} ({progress:.1f}%)")
    
    logger.info(f"✅ Migrated {migrated:,} enrollment records")


def migrate_demographic(sqlite_conn, pg_conn):
    """Migrate demographic_updates table."""
    logger.info("Migrating demographic updates...")
    
    sqlite_cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    # Get total count
    sqlite_cursor.execute("SELECT COUNT(*) FROM demographic_updates")
    total = sqlite_cursor.fetchone()[0]
    logger.info(f"  Total records to migrate: {total:,}")
    
    # Fetch and insert in batches
    offset = 0
    migrated = 0
    
    while offset < total:
        sqlite_cursor.execute(f"""
            SELECT date, state, district, pincode, demo_age_5_17, demo_age_17_plus, total
            FROM demographic_updates
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """)
        
        rows = sqlite_cursor.fetchall()
        if not rows:
            break
        
        # Insert into PostgreSQL
        for row in rows:
            pg_cursor.execute("""
                INSERT INTO demographic_updates (date, state, district, pincode, demo_age_5_17, demo_age_17_plus, total)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, row)
        
        pg_conn.commit()
        migrated += len(rows)
        offset += BATCH_SIZE
        
        progress = (migrated / total) * 100
        logger.info(f"  Progress: {migrated:,}/{total:,} ({progress:.1f}%)")
    
    logger.info(f"✅ Migrated {migrated:,} demographic records")


def migrate_biometric(sqlite_conn, pg_conn):
    """Migrate biometric_updates table."""
    logger.info("Migrating biometric updates...")
    
    sqlite_cursor = sqlite_conn.cursor()
    pg_cursor = pg_conn.cursor()
    
    # Get total count
    sqlite_cursor.execute("SELECT COUNT(*) FROM biometric_updates")
    total = sqlite_cursor.fetchone()[0]
    logger.info(f"  Total records to migrate: {total:,}")
    
    # Fetch and insert in batches
    offset = 0
    migrated = 0
    
    while offset < total:
        sqlite_cursor.execute(f"""
            SELECT date, state, district, pincode, bio_age_5_17, bio_age_17_plus, total
            FROM biometric_updates
            LIMIT {BATCH_SIZE} OFFSET {offset}
        """)
        
        rows = sqlite_cursor.fetchall()
        if not rows:
            break
        
        # Insert into PostgreSQL
        for row in rows:
            pg_cursor.execute("""
                INSERT INTO biometric_updates (date, state, district, pincode, bio_age_5_17, bio_age_17_plus, total)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
            """, row)
        
        pg_conn.commit()
        migrated += len(rows)
        offset += BATCH_SIZE
        
        progress = (migrated / total) * 100
        logger.info(f"  Progress: {migrated:,}/{total:,} ({progress:.1f}%)")
    
    logger.info(f"✅ Migrated {migrated:,} biometric records")


def verify_migration(pg_conn):
    """Verify migration was successful."""
    cursor = pg_conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM enrollments")
    enrollment_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM demographic_updates")
    demographic_count = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM biometric_updates")
    biometric_count = cursor.fetchone()[0]
    
    return enrollment_count, demographic_count, biometric_count


def main():
    print("=" * 60)
    print("🚀 SQLite to PostgreSQL Migration (Supabase)")
    print("=" * 60)
    
    # Check SQLite exists
    if not os.path.exists(SQLITE_PATH):
        logger.error(f"❌ SQLite database not found: {SQLITE_PATH}")
        logger.error("   Run scripts/init_database.py first to create the SQLite database.")
        sys.exit(1)
    
    print(f"\n📂 Source: {SQLITE_PATH}")
    print(f"📂 Target: Supabase PostgreSQL")
    
    # Connect to databases
    print("\n🔌 Connecting to databases...")
    try:
        sqlite_conn = get_sqlite_connection()
        logger.info("  ✅ Connected to SQLite")
    except Exception as e:
        logger.error(f"  ❌ SQLite connection failed: {e}")
        sys.exit(1)
    
    try:
        pg_conn = get_postgres_connection()
        logger.info("  ✅ Connected to PostgreSQL (Supabase)")
    except Exception as e:
        logger.error(f"  ❌ PostgreSQL connection failed: {e}")
        logger.error("     Check your connection string in the script.")
        sys.exit(1)
    
    try:
        # Create tables
        print("\n📊 Creating tables in PostgreSQL...")
        create_tables_postgres(pg_conn)
        
        # Migrate data
        print("\n📥 Migrating data...")
        print("-" * 40)
        
        migrate_enrollments(sqlite_conn, pg_conn)
        migrate_demographic(sqlite_conn, pg_conn)
        migrate_biometric(sqlite_conn, pg_conn)
        
        # Verify
        print("\n✅ Verifying migration...")
        e_count, d_count, b_count = verify_migration(pg_conn)
        
        print("\n" + "=" * 60)
        print("✅ Migration Complete!")
        print("-" * 40)
        print(f"  📊 Enrollment records:   {e_count:,}")
        print(f"  📊 Demographic records:  {d_count:,}")
        print(f"  📊 Biometric records:    {b_count:,}")
        print(f"  📊 Total:                {e_count + d_count + b_count:,}")
        print("=" * 60)
        
        print("\n🎉 Next steps:")
        print("   1. Update .env: Set USE_POSTGRES=true")
        print("   2. Deploy to Render")
        print("   3. Set DATABASE_URL in Render environment variables")
        
    except Exception as e:
        logger.error(f"❌ Migration failed: {e}")
        pg_conn.rollback()
        raise
    finally:
        sqlite_conn.close()
        pg_conn.close()


if __name__ == "__main__":
    main()
