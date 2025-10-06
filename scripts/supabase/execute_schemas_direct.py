#!/usr/bin/env python3
"""
Direct PostgreSQL execution of NFL schemas
Uses psycopg2 to bypass Supabase client limitations
"""

import psycopg2
from psycopg2 import sql
import os
from pathlib import Path
import logging
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def execute_sql_file(conn, file_path: str) -> bool:
    """Execute SQL file using direct PostgreSQL connection"""
    try:
        with open(file_path, 'r') as f:
            sql_content = f.read()

        with conn.cursor() as cur:
            # Split by semicolon and execute each statement
            statements = [s.strip() for s in sql_content.split(';') if s.strip()]

            for i, statement in enumerate(statements, 1):
                # Skip comments-only statements
                if statement.startswith('--') or not statement:
                    continue

                # Skip SELECT statements that are just for verification
                if statement.strip().startswith('SELECT') and 'INSERT' not in statement:
                    logger.info(f"Skipping verification query {i}")
                    continue

                try:
                    cur.execute(statement + ';')
                    logger.info(f"Statement {i}: Executed successfully")
                except psycopg2.Error as e:
                    if 'already exists' in str(e) or 'does not exist' in str(e):
                        logger.warning(f"Statement {i}: {e}")
                    else:
                        logger.error(f"Statement {i} failed: {e}")
                        logger.error(f"Statement was: {statement[:100]}...")
                        raise

            conn.commit()
            logger.info(f"✅ Completed executing {file_path}")
            return True

    except Exception as e:
        logger.error(f"❌ Failed to execute {file_path}: {e}")
        conn.rollback()
        return False


def main():
    """Execute all schema files in order using direct PostgreSQL connection"""

    # Supabase connection details
    # Using the actual database password (not JWT)
    # Get database credentials from environment
    db_password = os.getenv('SUPABASE_DB_PASSWORD')
    project_ref = os.getenv('SUPABASE_PROJECT_REF', 'cqslvbxsqsgjagjkpiro')

    if not db_password:
        logger.error("❌ Missing SUPABASE_DB_PASSWORD environment variable")
        return False

    # Try both pooler and direct connection
    connections_to_try = [
        {
            'host': 'aws-0-us-east-1.pooler.supabase.com',
            'port': 5432,
            'database': 'postgres',
            'user': f'postgres.{project_ref}',
            'password': db_password
        },
        {
            'host': f'db.{project_ref}.supabase.co',
            'port': 5432,
            'database': 'postgres',
            'user': 'postgres',
            'password': db_password
        }
    ]

    # Schema files in execution order
    schema_dir = Path(__file__).parent
    schema_files = [
        'nfl_enhanced_schema.sql',
        'stadium_history_schema.sql',
        'stadium_history_data.sql'
    ]

    conn = None
    for conn_params in connections_to_try:
        try:
            logger.info(f"Trying connection to {conn_params['host']}...")
            conn = psycopg2.connect(**conn_params)
            logger.info("✅ Connected successfully")
            break
        except psycopg2.Error as e:
            logger.warning(f"Failed to connect to {conn_params['host']}: {e}")
            continue

    if not conn:
        logger.error("Could not establish database connection with any method")
        logger.info("\nPlease verify your connection string in Supabase Dashboard:")
        logger.info("Settings > Database > Connection string")
        return

    try:

        # Execute each schema file
        for file_name in schema_files:
            file_path = schema_dir / file_name
            if not file_path.exists():
                logger.error(f"File not found: {file_path}")
                continue

            logger.info(f"\n{'='*60}")
            logger.info(f"Executing {file_name}...")
            logger.info(f"{'='*60}")

            if not execute_sql_file(conn, str(file_path)):
                logger.error(f"Failed to execute {file_name}, stopping...")
                break

        # Verify tables were created
        logger.info(f"\n{'='*60}")
        logger.info("Verifying created tables...")
        logger.info(f"{'='*60}")

        tables_to_check = [
            'teams', 'games', 'plays', 'odds_movements',
            'officials', 'rosters', 'snap_counts',
            'player_weekly_stats', 'win_totals', 'stadium_history'
        ]

        with conn.cursor() as cur:
            for table in tables_to_check:
                cur.execute(
                    "SELECT COUNT(*) FROM information_schema.tables "
                    "WHERE table_name = %s",
                    (table,)
                )
                exists = cur.fetchone()[0] > 0
                if exists:
                    logger.info(f"✅ Table '{table}' exists")
                else:
                    logger.warning(f"❌ Table '{table}' not found")

        logger.info("\n✅ Schema execution completed!")
        logger.info("Next steps:")
        logger.info("1. Run nfl_data_loader.py to populate with NFL data")
        logger.info("2. Test the enhanced schema with your betting system")

    except psycopg2.Error as e:
        logger.error(f"Database connection error: {e}")
        logger.info("\nTry using the Supabase Dashboard instead:")
        logger.info("1. Go to https://supabase.com/dashboard")
        logger.info("2. Select your project")
        logger.info("3. Click 'SQL Editor'")
        logger.info("4. Run each .sql file in order")
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
    finally:
        if 'conn' in locals():
            conn.close()
            logger.info("Database connection closed")


if __name__ == "__main__":
    main()