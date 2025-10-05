#!/usr/bin/env python3
"""
Test PostgreSQL connection to Supabase
"""

import psycopg2
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# The connection string you provided
# postgresql://postgres:[P@ssword9804746196$]@db.cqslvbxsqsgjagjkpiro.supabase.co:5432/postgres

connections_to_try = [
    {
        'name': 'Pooler with project prefix',
        'host': 'aws-0-us-east-1.pooler.supabase.com',
        'port': 6543,  # Try pooler port
        'database': 'postgres',
        'user': 'postgres.cqslvbxsqsgjagjkpiro',
        'password': 'P@ssword9804746196$',
        'connect_timeout': 5
    },
    {
        'name': 'Direct DB connection',
        'host': 'db.cqslvbxsqsgjagjkpiro.supabase.co',
        'port': 5432,
        'database': 'postgres',
        'user': 'postgres',
        'password': 'P@ssword9804746196$',
        'connect_timeout': 5
    },
    {
        'name': 'Pooler standard port',
        'host': 'aws-0-us-east-1.pooler.supabase.com',
        'port': 5432,
        'database': 'postgres',
        'user': 'postgres.cqslvbxsqsgjagjkpiro',
        'password': 'P@ssword9804746196$',
        'connect_timeout': 5
    }
]

for config in connections_to_try:
    name = config.pop('name')
    logger.info(f"\nTrying: {name}")
    logger.info(f"Host: {config['host']}:{config['port']}")

    try:
        conn = psycopg2.connect(**config)
        logger.info(f"✅ SUCCESS! Connected via {name}")

        # Test query
        cur = conn.cursor()
        cur.execute("SELECT version()")
        version = cur.fetchone()
        logger.info(f"PostgreSQL version: {version[0][:50]}...")

        cur.close()
        conn.close()

        logger.info(f"\nUse this configuration:")
        config['name'] = name
        print(f"conn_params = {config}")
        break

    except psycopg2.OperationalError as e:
        logger.error(f"❌ Failed: {e}")
    except Exception as e:
        logger.error(f"❌ Error: {e}")
else:
    logger.error("\nAll connection attempts failed")
    logger.info("\nPlease check in Supabase Dashboard:")
    logger.info("1. Settings > Database")
    logger.info("2. Look for 'Connection string' section")
    logger.info("3. Verify the exact format and host")