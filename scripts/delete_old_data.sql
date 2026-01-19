-- ============================================================
-- SQL Script to Reduce Supabase Database Size
-- Deletes data from March 1 to May 31, 2025
-- Keeps data from June 1 to December 31, 2025
-- ============================================================

-- Run this in Supabase SQL Editor: https://supabase.com/dashboard
-- Go to: Your Project → SQL Editor → New Query → Paste & Run

-- ============================================================
-- STEP 1: Check current data counts (run this first)
-- ============================================================

SELECT 'enrollments' as table_name, 
       COUNT(*) as total_rows,
       COUNT(*) FILTER (WHERE date < '2025-06-01') as rows_to_delete,
       COUNT(*) FILTER (WHERE date >= '2025-06-01') as rows_to_keep
FROM enrollments

UNION ALL

SELECT 'demographic_updates' as table_name,
       COUNT(*) as total_rows,
       COUNT(*) FILTER (WHERE date < '2025-06-01') as rows_to_delete,
       COUNT(*) FILTER (WHERE date >= '2025-06-01') as rows_to_keep
FROM demographic_updates

UNION ALL

SELECT 'biometric_updates' as table_name,
       COUNT(*) as total_rows,
       COUNT(*) FILTER (WHERE date < '2025-06-01') as rows_to_delete,
       COUNT(*) FILTER (WHERE date >= '2025-06-01') as rows_to_keep
FROM biometric_updates;

-- ============================================================
-- STEP 2: Delete old data (run each DELETE separately)
-- ============================================================

-- Delete enrollments from March-May 2025
DELETE FROM enrollments 
WHERE date < '2025-06-01';

-- Delete demographic updates from March-May 2025
DELETE FROM demographic_updates 
WHERE date < '2025-06-01';

-- Delete biometric updates from March-May 2025
DELETE FROM biometric_updates 
WHERE date < '2025-06-01';

-- ============================================================
-- STEP 3: Reclaim disk space with VACUUM
-- ============================================================

-- This reclaims the space from deleted rows
VACUUM FULL enrollments;
VACUUM FULL demographic_updates;
VACUUM FULL biometric_updates;

-- ============================================================
-- STEP 4: Verify deletion (run after deletes)
-- ============================================================

SELECT 'enrollments' as table_name, COUNT(*) as remaining_rows, MIN(date) as earliest_date, MAX(date) as latest_date FROM enrollments
UNION ALL
SELECT 'demographic_updates', COUNT(*), MIN(date), MAX(date) FROM demographic_updates
UNION ALL
SELECT 'biometric_updates', COUNT(*), MIN(date), MAX(date) FROM biometric_updates;

-- ============================================================
-- STEP 5: Check table sizes
-- ============================================================

SELECT 
    relname as table_name,
    pg_size_pretty(pg_total_relation_size(relid)) as total_size
FROM pg_catalog.pg_statio_user_tables
WHERE relname IN ('enrollments', 'demographic_updates', 'biometric_updates')
ORDER BY pg_total_relation_size(relid) DESC;
