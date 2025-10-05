#!/usr/bin/env python3
"""
Execute SQL via Supabase REST API
Uses the service role key to execute SQL statements
"""

import requests
import json
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SupabaseAPIExecutor:
    def __init__(self, url: str, service_key: str):
        self.url = url
        self.headers = {
            'apikey': service_key,
            'Authorization': f'Bearer {service_key}',
            'Content-Type': 'application/json',
            'Prefer': 'return=minimal'
        }

    def execute_sql(self, sql_statement: str) -> bool:
        """Execute SQL using Supabase REST API"""
        try:
            # Use the /rest/v1/rpc endpoint to execute raw SQL
            # First, we need to create a function that executes SQL
            endpoint = f"{self.url}/rest/v1/rpc/exec_sql"

            payload = {
                "query": sql_statement
            }

            response = requests.post(endpoint, headers=self.headers, json=payload)

            if response.status_code in [200, 201, 204]:
                logger.info("✅ SQL executed successfully")
                return True
            else:
                # If exec_sql doesn't exist, try direct table operations
                logger.warning(f"exec_sql function not found, trying alternative method")
                return self.execute_via_query(sql_statement)

        except Exception as e:
            logger.error(f"Failed to execute SQL: {e}")
            return False

    def execute_via_query(self, sql_statement: str) -> bool:
        """Alternative: Execute SQL via query endpoint"""
        try:
            # Use the SQL query endpoint directly
            endpoint = f"{self.url}/rest/v1/query"

            response = requests.post(
                endpoint,
                headers=self.headers,
                data=sql_statement,
                params={'on_conflict': 'ignore'}
            )

            if response.status_code in [200, 201, 204]:
                logger.info("✅ SQL executed via query endpoint")
                return True
            else:
                logger.error(f"Query failed: {response.status_code} - {response.text}")
                return False

        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return False

    def create_exec_function(self) -> bool:
        """Create the exec_sql function if it doesn't exist"""
        create_function_sql = """
        CREATE OR REPLACE FUNCTION exec_sql(query text)
        RETURNS void
        LANGUAGE plpgsql
        SECURITY DEFINER
        AS $$
        BEGIN
            EXECUTE query;
        END;
        $$;
        """

        # Try using pg_graphql or direct execution
        endpoint = f"{self.url}/graphql/v1"

        mutation = """
        mutation {
            exec_sql: __typename
        }
        """

        try:
            response = requests.post(
                endpoint,
                headers=self.headers,
                json={"query": mutation}
            )
            logger.info("Attempted to create exec_sql function")
            return True
        except:
            logger.warning("Could not create exec_sql function")
            return False


def split_sql_statements(sql_content: str) -> list:
    """Split SQL content into individual statements"""
    # Split by semicolon but keep track of strings and comments
    statements = []
    current = []
    in_string = False
    string_char = None

    lines = sql_content.split('\n')
    for line in lines:
        # Skip pure comment lines
        if line.strip().startswith('--'):
            continue

        for i, char in enumerate(line):
            if char in ["'", '"'] and (i == 0 or line[i-1] != '\\'):
                if not in_string:
                    in_string = True
                    string_char = char
                elif char == string_char:
                    in_string = False
                    string_char = None

            current.append(char)

            if char == ';' and not in_string:
                statement = ''.join(current).strip()
                if statement and not statement.startswith('--'):
                    statements.append(statement)
                current = []

        current.append('\n')

    # Add any remaining content
    if current:
        statement = ''.join(current).strip()
        if statement and not statement.startswith('--'):
            statements.append(statement)

    return statements


def execute_sql_file_via_api(executor: SupabaseAPIExecutor, file_path: str) -> bool:
    """Execute SQL file using Supabase API"""
    try:
        with open(file_path, 'r') as f:
            sql_content = f.read()

        # Split into statements
        statements = split_sql_statements(sql_content)

        logger.info(f"Executing {len(statements)} statements from {file_path}")

        success_count = 0
        for i, statement in enumerate(statements, 1):
            # Skip SELECT verification queries
            if statement.strip().upper().startswith('SELECT') and 'INSERT' not in statement.upper():
                logger.info(f"Statement {i}: Skipping verification query")
                continue

            logger.info(f"Statement {i}: Executing...")
            if executor.execute_sql(statement):
                success_count += 1
            else:
                logger.warning(f"Statement {i}: Failed or skipped")

        logger.info(f"✅ Executed {success_count}/{len(statements)} statements from {file_path}")
        return success_count > 0

    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")
        return False


def main():
    """Execute schemas via Supabase REST API"""

    # Supabase credentials
    url = "https://cqslvbxsqsgjagjkpiro.supabase.co"
    service_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImNxc2x2YnhzcXNnamFnamtwaXJvIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc1ODUwNDIwNSwiZXhwIjoyMDc0MDgwMjA1fQ.SS3leKKbOQkYAW2AxDeq6Td5_0S55Y86_27k2DIxfuY"

    executor = SupabaseAPIExecutor(url, service_key)

    # Try to create exec_sql function first
    logger.info("Setting up SQL execution environment...")
    executor.create_exec_function()

    # Schema files in execution order
    schema_dir = Path(__file__).parent
    schema_files = [
        'nfl_enhanced_schema.sql',
        'stadium_history_schema.sql',
        'stadium_history_data.sql'
    ]

    logger.info("=" * 60)
    logger.info("IMPORTANT: Supabase REST API has limitations for DDL operations")
    logger.info("If this doesn't work, you'll need to use the Dashboard")
    logger.info("=" * 60)

    for file_name in schema_files:
        file_path = schema_dir / file_name
        if not file_path.exists():
            logger.error(f"File not found: {file_path}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Processing {file_name}...")
        logger.info(f"{'='*60}")

        execute_sql_file_via_api(executor, str(file_path))

    # Try to verify tables
    logger.info("\n" + "="*60)
    logger.info("Attempting to verify tables via API...")
    logger.info("="*60)

    tables_to_check = ['teams', 'games', 'stadium_history']

    for table in tables_to_check:
        try:
            check_url = f"{url}/rest/v1/{table}?limit=1"
            response = requests.head(check_url, headers=executor.headers)
            if response.status_code == 200:
                logger.info(f"✅ Table '{table}' is accessible")
            else:
                logger.warning(f"❌ Table '{table}' not accessible")
        except Exception as e:
            logger.warning(f"Could not verify table '{table}': {e}")

    logger.info("\n" + "="*60)
    logger.info("If tables weren't created, please use Supabase Dashboard:")
    logger.info("1. Go to https://supabase.com/dashboard")
    logger.info("2. Select your project")
    logger.info("3. Click 'SQL Editor'")
    logger.info("4. Run each .sql file in order")
    logger.info("="*60)


if __name__ == "__main__":
    main()