#!/usr/bin/env python3
"""
Final database validation after completion
"""
import sqlite3
import os
from datetime import datetime

def validate_database():
    """Final validation of the production database"""

    production_db = "improved_nfl_system/database/nfl_suggestions.db"

    # Get database size
    db_size_mb = os.path.getsize(production_db) / (1024 * 1024)

    conn = sqlite3.connect(production_db)
    cursor = conn.cursor()

    print("=" * 50)
    print("🏈 NFL DATABASE FINAL VALIDATION REPORT")
    print("=" * 50)
    print(f"📅 Validation Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"📊 Database: {production_db}")
    print(f"💾 Size: {db_size_mb:.1f} MB")
    print()

    # Check tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [t[0] for t in cursor.fetchall()]
    print(f"📋 Tables Found: {len(tables)}")

    # Detailed table analysis
    print("\n📊 TABLE ANALYSIS:")
    print("-" * 40)

    total_records = 0
    empty_tables = []
    populated_tables = []

    for table in tables:
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        count = cursor.fetchone()[0]
        total_records += count

        if count == 0:
            empty_tables.append(table)
            print(f"  ⚠️  {table:25} : EMPTY")
        else:
            populated_tables.append(table)
            print(f"  ✅ {table:25} : {count:,} records")

    # Season analysis
    print("\n📅 SEASON COVERAGE:")
    print("-" * 40)

    cursor.execute("SELECT DISTINCT season FROM all_schedules ORDER BY season")
    seasons = [s[0] for s in cursor.fetchall()]
    print(f"  Seasons in database: {seasons}")

    cursor.execute("SELECT season, COUNT(*) as games FROM all_schedules GROUP BY season ORDER BY season")
    season_counts = cursor.fetchall()
    for season, count in season_counts:
        print(f"  Season {season}: {count} games")

    # Game totals
    cursor.execute("SELECT COUNT(*) FROM all_schedules")
    total_games = cursor.fetchone()[0]

    # EPA stats check
    cursor.execute("SELECT COUNT(DISTINCT team) FROM team_epa_stats")
    teams_with_epa = cursor.fetchone()[0]

    cursor.execute("SELECT COUNT(DISTINCT season) FROM team_epa_stats")
    seasons_with_epa = cursor.fetchone()[0]

    # Calculate score
    score = 0
    max_score = 100

    # Scoring criteria
    if len(tables) >= 10:
        score += 20
    else:
        score += (len(tables) / 10) * 20

    if total_games >= 2000:
        score += 30
    else:
        score += (total_games / 2000) * 30

    if len(seasons) >= 8:
        score += 20
    else:
        score += (len(seasons) / 8) * 20

    if len(empty_tables) == 0:
        score += 20
    elif len(empty_tables) <= 2:
        score += 10
    elif len(empty_tables) <= 4:
        score += 5

    if teams_with_epa >= 32:
        score += 10
    else:
        score += (teams_with_epa / 32) * 10

    # Summary
    print("\n" + "=" * 50)
    print("📈 FINAL SUMMARY:")
    print("-" * 40)
    print(f"  Total Records: {total_records:,}")
    print(f"  Total Games: {total_games:,}")
    print(f"  Seasons: {len(seasons)} ({min(seasons) if seasons else 'N/A'} - {max(seasons) if seasons else 'N/A'})")
    print(f"  Populated Tables: {len(populated_tables)}/{len(tables)}")
    print(f"  Empty Tables: {len(empty_tables)}")
    if empty_tables:
        print(f"    Empty: {', '.join(empty_tables)}")
    print(f"  Teams with EPA: {teams_with_epa}")
    print(f"  Seasons with EPA: {seasons_with_epa}")

    print("\n🏆 FINAL SCORE:")
    print("-" * 40)
    print(f"  Score: {score:.0f}/{max_score}")

    if score >= 85:
        status = "✅ EXCELLENT - Database is production-ready!"
        emoji = "🎉"
    elif score >= 70:
        status = "✅ GOOD - Database is functional with minor gaps"
        emoji = "👍"
    elif score >= 50:
        status = "⚠️  FAIR - Database needs some work"
        emoji = "🔧"
    else:
        status = "❌ NEEDS WORK - Significant data missing"
        emoji = "🚧"

    print(f"  Status: {status} {emoji}")
    print("=" * 50)

    # Improvements made
    print("\n📊 IMPROVEMENTS FROM BASELINE:")
    print("-" * 40)
    print("  Before: Score 40/100 (CRITICAL)")
    print("  - 1,343 games (2020-2024 only)")
    print("  - Missing 4 seasons")
    print("  - 2 empty critical tables")
    print()
    print(f"  After: Score {score:.0f}/100")
    print(f"  - {total_games:,} games ({min(seasons) if seasons else 'N/A'}-{max(seasons) if seasons else 'N/A'})")
    print(f"  - Added {len([s for s in seasons if s in [2016,2017,2018,2019]])} missing seasons")
    print(f"  - {len(empty_tables)} empty tables remaining")
    print()
    improvement = score - 40
    print(f"  🎯 Improvement: +{improvement:.0f} points ({improvement/40*100:.0f}% better)")

    conn.close()

if __name__ == "__main__":
    validate_database()