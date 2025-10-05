#!/usr/bin/env python3
"""
Execute SQL via Supabase Management API
Requires a Personal Access Token from Supabase Dashboard
"""

import requests
import json
from pathlib import Path
import logging
import os

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def execute_sql_via_management_api(access_token: str, project_ref: str, sql: str):
    """
    Execute SQL using Supabase Management API

    Args:
        access_token: Personal Access Token from https://supabase.com/dashboard/account/tokens
        project_ref: Your project reference (e.g., 'cqslvbxsqsgjagjkpiro')
        sql: SQL statement to execute
    """

    url = f"https://api.supabase.com/v1/projects/{project_ref}/database/query"

    headers = {
        'Authorization': f'Bearer {access_token}',
        'Content-Type': 'application/json',
    }

    payload = {
        'query': sql
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        logger.info("✅ SQL executed successfully")
        return True
    else:
        logger.error(f"Failed: {response.status_code} - {response.text}")
        return False


def get_database_password(access_token: str, project_ref: str):
    """
    Retrieve database password using Management API
    """
    url = f"https://api.supabase.com/v1/projects/{project_ref}"

    headers = {
        'Authorization': f'Bearer {access_token}',
    }

    response = requests.get(url, headers=headers)

    if response.status_code == 200:
        data = response.json()
        # Database password might be in the response
        return data.get('database', {}).get('password')
    else:
        logger.error("Could not retrieve database password")
        return None


def main():
    """
    Execute schemas using Supabase Management API
    """

    logger.info("="*60)
    logger.info("SUPABASE MANAGEMENT API SETUP")
    logger.info("="*60)
    logger.info("\nTo use this script, you need:")
    logger.info("1. Go to https://supabase.com/dashboard/account/tokens")
    logger.info("2. Create a new Personal Access Token")
    logger.info("3. Set it as environment variable: SUPABASE_ACCESS_TOKEN")
    logger.info("\nOR")
    logger.info("\n1. Go to Supabase Dashboard > Settings > Database")
    logger.info("2. Copy your database password (NOT the JWT token)")
    logger.info("3. Use execute_schemas_direct.py with the actual password")
    logger.info("="*60)

    # Check for access token
    access_token = os.getenv('SUPABASE_ACCESS_TOKEN')
    if not access_token:
        logger.error("\n❌ SUPABASE_ACCESS_TOKEN environment variable not set")
        logger.info("\nAlternative: Get your database password from:")
        logger.info("Supabase Dashboard > Settings > Database > Connection string")
        logger.info("Look for: postgresql://postgres:[THIS-IS-YOUR-PASSWORD]@db...")
        return

    project_ref = "cqslvbxsqsgjagjkpiro"

    # Try to get database password
    db_password = get_database_password(access_token, project_ref)
    if db_password:
        logger.info(f"✅ Retrieved database password: {db_password[:8]}...")
        logger.info("You can now use this with execute_schemas_direct.py")

    # Schema files
    schema_dir = Path(__file__).parent
    schema_files = [
        'nfl_enhanced_schema.sql',
        'stadium_history_schema.sql',
        'stadium_history_data.sql'
    ]

    for file_name in schema_files:
        file_path = schema_dir / file_name
        if not file_path.exists():
            continue

        with open(file_path, 'r') as f:
            sql_content = f.read()

        logger.info(f"\nExecuting {file_name}...")
        execute_sql_via_management_api(access_token, project_ref, sql_content)


if __name__ == "__main__":
    main()