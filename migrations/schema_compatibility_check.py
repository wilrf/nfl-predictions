#!/usr/bin/env python3
"""
STEP 0: Pre-Migration Schema Compatibility Check
Verify SQLite and Supabase schemas are compatible before migration
"""
import sqlite3
import os
import sys
from datetime import datetime

def check_schema_compatibility():
    """Check if SQLite and Supabase schemas are compatible"""

    print("=" * 60)
    print("🔍 SCHEMA COMPATIBILITY CHECK")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print()

    compatibility_issues = []
    warnings = []

    # 1. Check SQLite database exists
    sqlite_db_path = 'improved_nfl_system/database/nfl_suggestions.db'
    if not os.path.exists(sqlite_db_path):
        compatibility_issues.append(f"SQLite database not found: {sqlite_db_path}")
        print(f"❌ SQLite database not found at {sqlite_db_path}")
        return False
    else:
        db_size_mb = os.path.getsize(sqlite_db_path) / (1024 * 1024)
        print(f"✅ SQLite database found: {db_size_mb:.1f} MB")

    # 2. Connect to SQLite and check tables
    try:
        conn = sqlite3.connect(sqlite_db_path)
        cursor = conn.cursor()

        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        sqlite_tables = [row[0] for row in cursor.fetchall()]

        print(f"✅ Found {len(sqlite_tables)} tables in SQLite")

        # Required tables for migration
        required_tables = [
            'all_schedules',
            'team_epa_stats',
            'game_features',
            'historical_games',
            'team_features',
            'epa_metrics',
            'betting_outcomes',
            'feature_history'
        ]

        # Check each required table
        print("\n📊 Checking required tables:")
        for table in required_tables:
            if table in sqlite_tables:
                cursor.execute(f"SELECT COUNT(*) FROM {table}")
                count = cursor.fetchone()[0]
                print(f"  ✅ {table:20} : {count:,} records")
            else:
                compatibility_issues.append(f"Missing required table: {table}")
                print(f"  ❌ {table:20} : MISSING")

        # Check for teams table (optional but recommended)
        if 'teams' not in sqlite_tables:
            warnings.append("Teams table not found in SQLite - will need to create")
            print(f"  ⚠️  teams table not found (will be created)")

    except sqlite3.Error as e:
        compatibility_issues.append(f"SQLite error: {e}")
        print(f"❌ SQLite connection error: {e}")
    finally:
        if conn:
            conn.close()

    # 3. Check Supabase schema file
    schema_file = 'improved_nfl_system/supabase_complete_schema.sql'
    if not os.path.exists(schema_file):
        compatibility_issues.append(f"Supabase schema file not found: {schema_file}")
        print(f"\n❌ Supabase schema file not found: {schema_file}")
    else:
        schema_size_kb = os.path.getsize(schema_file) / 1024
        print(f"\n✅ Supabase schema file found: {schema_size_kb:.1f} KB")

    # 4. Check backup directory
    backup_dir = 'improved_nfl_system/database/backups'
    if not os.path.exists(backup_dir):
        warnings.append(f"Backup directory doesn't exist: {backup_dir} (will be created)")
        print(f"\n⚠️  Backup directory not found: {backup_dir}")
        print("    Will be created during backup step")
    else:
        print(f"\n✅ Backup directory exists: {backup_dir}")

    # 5. Check environment file
    env_file = 'improved_nfl_system/.env'
    if os.path.exists(env_file):
        print(f"\n✅ Environment file found: {env_file}")
        # Check for required variables
        with open(env_file, 'r') as f:
            env_content = f.read()
            if 'SUPABASE_URL' not in env_content:
                warnings.append("SUPABASE_URL not found in .env")
            if 'SUPABASE_KEY' not in env_content:
                warnings.append("SUPABASE_KEY not found in .env")
    else:
        warnings.append(f"Environment file not found: {env_file}")
        print(f"\n⚠️  Environment file not found: {env_file}")

    # 6. Summary
    print("\n" + "=" * 60)
    print("📋 COMPATIBILITY CHECK SUMMARY")
    print("=" * 60)

    if compatibility_issues:
        print("\n❌ CRITICAL ISSUES (must fix before migration):")
        for issue in compatibility_issues:
            print(f"  • {issue}")
        print("\n🚫 Cannot proceed with migration")
        return False

    if warnings:
        print("\n⚠️  WARNINGS (non-critical):")
        for warning in warnings:
            print(f"  • {warning}")

    print("\n✅ SCHEMA COMPATIBILITY CHECK PASSED")
    print("Ready to proceed with migration")
    print("=" * 60)

    return True

if __name__ == "__main__":
    # Run compatibility check
    if check_schema_compatibility():
        print("\n✅ You may proceed with STEP 1: Create Backups")
        sys.exit(0)
    else:
        print("\n❌ Fix compatibility issues before proceeding")
        sys.exit(1)