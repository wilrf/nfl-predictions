import os
from dotenv import load_dotenv

load_dotenv()

#!/usr/bin/env python3
"""
Parse the connection string to extract correct components
"""

from urllib.parse import urlparse, unquote

# Your provided connection string
connection_string = "postgresql://postgres:{os.getenv("SUPABASE_DB_PASSWORD")}@db.cqslvbxsqsgjagjkpiro.supabase.co:5432/postgres"

# Alternative parsing - maybe the password has brackets
alt_string1 = "postgresql://postgres:[{os.getenv("SUPABASE_DB_PASSWORD")}]@db.cqslvbxsqsgjagjkpiro.supabase.co:5432/postgres"

# Parse both
for cs in [connection_string, alt_string1]:
    print(f"\nParsing: {cs[:50]}...")

    try:
        # Manual parsing for complex passwords
        if ":[" in cs and "]@" in cs:
            # Password is in brackets
            start = cs.index(":[") + 2
            end = cs.index("]@")
            password = cs[start:end]
            # Reconstruct without brackets
            parts = cs.split(":[")
            user_part = parts[0]
            after_pass = cs[end+2:]  # Skip ]@

            print(f"User: postgres")
            print(f"Password: {password}")
            print(f"Rest: {after_pass}")

            # Extract host and port
            host_port = after_pass.split("/")[0]
            if ":" in host_port:
                host, port = host_port.split(":")
            else:
                host = host_port
                port = "5432"

            print(f"Host: {host}")
            print(f"Port: {port}")
            print(f"Database: postgres")

        else:
            # Standard parsing
            parsed = urlparse(cs)
            print(f"Scheme: {parsed.scheme}")
            print(f"Username: {parsed.username}")
            print(f"Password: {parsed.password}")
            print(f"Host: {parsed.hostname}")
            print(f"Port: {parsed.port}")
            print(f"Database: {parsed.path.lstrip('/')}")

    except Exception as e:
        print(f"Error parsing: {e}")

print("\n" + "="*60)
print("Based on the connection string format:")
print("It appears the password might be: {os.getenv("SUPABASE_DB_PASSWORD")}")
print("And the host should be: db.cqslvbxsqsgjagjkpiro.supabase.co")
print("="*60)