#!/usr/bin/env python3
"""
Consolidate databases - merge validation_data.db into production nfl_suggestions.db
"""
import sqlite3
import shutil
from datetime import datetime
import os

def consolidate_databases():
    """Merge validation database into production database"""

    # Paths
    validation_db = "improved_nfl_system/database/validation_data.db"
    production_db = "improved_nfl_system/database/nfl_suggestions.db"
    backup_dir = "improved_nfl_system/database/backups"

    # Create backup directory
    os.makedirs(backup_dir, exist_ok=True)

    # Backup production database
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = f"{backup_dir}/nfl_suggestions_backup_{timestamp}.db"
    shutil.copy2(production_db, backup_path)
    print(f"✅ Backed up production database to: {backup_path}")

    # Connect to both databases
    prod_conn = sqlite3.connect(production_db)
    val_conn = sqlite3.connect(validation_db)

    try:
        # Get all tables from validation database
        val_cursor = val_conn.cursor()
        val_cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = val_cursor.fetchall()

        print(f"\n📊 Found {len(tables)} tables to consolidate")

        for (table_name,) in tables:
            # Skip sqlite internal tables
            if table_name.startswith('sqlite_'):
                continue

            print(f"\n📋 Processing table: {table_name}")

            # Get data from validation database
            val_cursor.execute(f"SELECT * FROM {table_name}")
            data = val_cursor.fetchall()

            if not data:
                print(f"  ⚠️  No data in {table_name}")
                continue

            # Get column names
            val_cursor.execute(f"PRAGMA table_info({table_name})")
            columns = [col[1] for col in val_cursor.fetchall()]

            # Create table in production if it doesn't exist
            val_cursor.execute(f"SELECT sql FROM sqlite_master WHERE name='{table_name}'")
            create_sql = val_cursor.fetchone()
            if create_sql:
                prod_conn.execute(create_sql[0])

            # Clear existing data in production table
            prod_conn.execute(f"DELETE FROM {table_name}")

            # Insert data into production
            placeholders = ','.join(['?' for _ in columns])
            insert_sql = f"INSERT INTO {table_name} ({','.join(columns)}) VALUES ({placeholders})"

            prod_conn.executemany(insert_sql, data)
            print(f"  ✅ Copied {len(data)} records to production")

        # Commit changes
        prod_conn.commit()
        print("\n✅ Database consolidation complete!")

        # Verify
        prod_cursor = prod_conn.cursor()
        prod_cursor.execute("SELECT COUNT(*) FROM all_schedules")
        count = prod_cursor.fetchone()[0]
        print(f"📊 Production database now has {count} games")

    finally:
        prod_conn.close()
        val_conn.close()

if __name__ == "__main__":
    consolidate_databases()