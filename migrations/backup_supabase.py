#!/usr/bin/env python3
"""
STEP 1: Backup current Supabase data before migration
"""
import json
from datetime import datetime
import os

print("📦 Backing up current Supabase data...")
print("=" * 60)

# Tables to backup from Supabase
tables_to_backup = [
    'games',
    'historical_games',
    'teams',
    'team_epa_stats',
    'game_features',
    'team_features',
    'epa_metrics',
    'betting_outcomes'
]

# Create backup data structure
supabase_backup = {
    'backup_timestamp': datetime.now().isoformat(),
    'database': 'Supabase Production',
    'tables': {}
}

# Count total records
total_records = 0

print("Exporting tables from Supabase:")
print("-" * 40)