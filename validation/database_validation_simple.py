#!/usr/bin/env python3
"""
Simple NFL Database Validation Script
=====================================

This script provides database validation without external dependencies.
Perfect for quick validation checks.

Author: NFL Betting System
Date: 2025-01-27
"""

import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Any

class SimpleDatabaseValidator:
    """Simple database validation without external dependencies"""
    
    def __init__(self, db_path: str = 'improved_nfl_system/database/validation_data.db'):
        """Initialize validator"""
        self.db_path = db_path
        self.conn = None
        self.results = {}
        
    def connect(self) -> bool:
        """Connect to database"""
        try:
            if not os.path.exists(self.db_path):
                print(f"❌ Database not found: {self.db_path}")
                return False
                
            self.conn = sqlite3.connect(self.db_path)
            print(f"✅ Connected to database: {self.db_path}")
            print(f"📊 Database size: {os.path.getsize(self.db_path) / 1024 / 1024:.1f} MB")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def get_table_info(self) -> Dict[str, Any]:
        """Get table information"""
        cursor = self.conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = [row[0] for row in cursor.fetchall()]
        
        table_info = {}
        for table in tables:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            count = cursor.fetchone()[0]
            table_info[table] = count
            
        return table_info
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity"""
        cursor = self.conn.cursor()
        issues = {}
        
        # Check for NULL values in critical fields
        null_checks = {
            'all_schedules_game_id': "SELECT COUNT(*) FROM all_schedules WHERE game_id IS NULL",
            'all_schedules_season': "SELECT COUNT(*) FROM all_schedules WHERE season IS NULL",
            'all_schedules_week': "SELECT COUNT(*) FROM all_schedules WHERE week IS NULL",
            'all_schedules_home_team': "SELECT COUNT(*) FROM all_schedules WHERE home_team IS NULL",
            'all_schedules_away_team': "SELECT COUNT(*) FROM all_schedules WHERE away_team IS NULL"
        }
        
        for check_name, query in null_checks.items():
            try:
                cursor.execute(query)
                count = cursor.fetchone()[0]
                issues[check_name] = count
            except Exception as e:
                issues[check_name] = f"Error: {e}"
        
        # Check for duplicates
        duplicate_checks = {
            'all_schedules_duplicates': "SELECT COUNT(*) - COUNT(DISTINCT game_id) FROM all_schedules",
            'historical_games_duplicates': "SELECT COUNT(*) - COUNT(DISTINCT game_id) FROM historical_games"
        }
        
        for check_name, query in duplicate_checks.items():
            try:
                cursor.execute(query)
                count = cursor.fetchone()[0]
                issues[check_name] = count
            except Exception as e:
                issues[check_name] = f"Error: {e}"
        
        # Check data ranges
        range_checks = {
            'invalid_scores': "SELECT COUNT(*) FROM all_schedules WHERE home_score < 0 OR away_score < 0 OR home_score > 100 OR away_score > 100",
            'invalid_weeks': "SELECT COUNT(*) FROM all_schedules WHERE week < 1 OR week > 22",
            'invalid_seasons': "SELECT COUNT(*) FROM all_schedules WHERE season < 2016 OR season > 2025"
        }
        
        for check_name, query in range_checks.items():
            try:
                cursor.execute(query)
                count = cursor.fetchone()[0]
                issues[check_name] = count
            except Exception as e:
                issues[check_name] = f"Error: {e}"
        
        return issues
    
    def get_season_coverage(self) -> Dict[str, Any]:
        """Get season coverage information"""
        cursor = self.conn.cursor()
        
        # Get season range
        cursor.execute("SELECT MIN(season), MAX(season), COUNT(DISTINCT season) FROM all_schedules")
        min_season, max_season, season_count = cursor.fetchone()
        
        # Get week coverage per season
        cursor.execute("""
            SELECT season, COUNT(DISTINCT week) as week_count, 
                   MIN(week) as min_week, MAX(week) as max_week
            FROM all_schedules 
            GROUP BY season 
            ORDER BY season
        """)
        week_coverage = {}
        for row in cursor.fetchall():
            season, week_count, min_week, max_week = row
            week_coverage[season] = {
                'week_count': week_count,
                'min_week': min_week,
                'max_week': max_week
            }
        
        return {
            'min_season': min_season,
            'max_season': max_season,
            'season_count': season_count,
            'week_coverage': week_coverage,
            'missing_seasons': [s for s in range(2016, 2025) if s not in week_coverage]
        }
    
    def generate_report(self) -> str:
        """Generate validation report"""
        report = f"""
# NFL Database Validation Report
Generated: {datetime.now().isoformat()}
Database: {self.db_path}

## Database Overview
- Size: {os.path.getsize(self.db_path) / 1024 / 1024:.1f} MB
- Tables: {len(self.get_table_info())}

## Table Information
"""
        
        table_info = self.get_table_info()
        for table, count in table_info.items():
            report += f"- {table}: {count:,} records\n"
        
        # Season coverage
        season_info = self.get_season_coverage()
        report += f"""
## Season Coverage
- Range: {season_info['min_season']}-{season_info['max_season']}
- Seasons: {season_info['season_count']}
- Missing seasons: {season_info['missing_seasons']}

## Week Coverage by Season
"""
        
        for season, info in season_info['week_coverage'].items():
            report += f"- {season}: {info['week_count']} weeks ({info['min_week']}-{info['max_week']})\n"
        
        # Data integrity
        integrity = self.validate_data_integrity()
        report += f"""
## Data Integrity Issues
"""
        
        total_issues = 0
        for check_name, count in integrity.items():
            if isinstance(count, int):
                total_issues += count
                if count > 0:
                    report += f"- {check_name}: {count} issues\n"
            else:
                report += f"- {check_name}: {count}\n"
        
        report += f"""
## Summary
- Total integrity issues: {total_issues}
- Database status: {'GOOD' if total_issues == 0 else 'NEEDS ATTENTION'}
- Missing seasons: {len(season_info['missing_seasons'])}
- Empty tables: {len([t for t, c in table_info.items() if c == 0])}

## Recommendations
"""
        
        if season_info['missing_seasons']:
            report += f"1. Import missing seasons: {season_info['missing_seasons']}\n"
        
        empty_tables = [t for t, c in table_info.items() if c == 0]
        if empty_tables:
            report += f"2. Populate empty tables: {', '.join(empty_tables)}\n"
        
        if total_issues > 0:
            report += f"3. Fix {total_issues} data integrity issues\n"
        
        if total_issues == 0 and not season_info['missing_seasons'] and not empty_tables:
            report += "1. Database is in excellent condition!\n"
            report += "2. Consider expanding to comprehensive schema\n"
            report += "3. Add ML features and advanced analytics\n"
        
        return report
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete validation"""
        if not self.connect():
            return {'status': 'error', 'message': 'Database connection failed'}
        
        try:
            results = {
                'timestamp': datetime.now().isoformat(),
                'database_path': self.db_path,
                'table_info': self.get_table_info(),
                'season_coverage': self.get_season_coverage(),
                'data_integrity': self.validate_data_integrity(),
                'status': 'success'
            }
            
            # Calculate overall score
            total_issues = sum([v for v in results['data_integrity'].values() if isinstance(v, int)])
            missing_seasons = len(results['season_coverage']['missing_seasons'])
            empty_tables = len([t for t, c in results['table_info'].items() if c == 0])
            
            if total_issues == 0 and missing_seasons == 0 and empty_tables == 0:
                results['overall_score'] = 100
                results['overall_status'] = 'excellent'
            elif total_issues < 10 and missing_seasons < 2 and empty_tables < 3:
                results['overall_score'] = 80
                results['overall_status'] = 'good'
            elif total_issues < 50 and missing_seasons < 4:
                results['overall_score'] = 60
                results['overall_status'] = 'needs_improvement'
            else:
                results['overall_score'] = 40
                results['overall_status'] = 'critical'
            
            return results
            
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
        finally:
            if self.conn:
                self.conn.close()
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()

def main():
    """Main execution function"""
    print("NFL Database Simple Validation")
    print("=" * 35)
    
    # Initialize validator
    validator = SimpleDatabaseValidator()
    
    try:
        # Run validation
        results = validator.run_validation()
        
        if results['status'] == 'success':
            print(f"✅ Validation completed successfully!")
            print(f"📊 Overall Score: {results['overall_score']}/100")
            print(f"📈 Status: {results['overall_status'].upper()}")
            
            # Print key findings
            print(f"\n📋 Key Findings:")
            print(f"- Tables: {len(results['table_info'])}")
            print(f"- Total records: {sum(results['table_info'].values()):,}")
            print(f"- Season range: {results['season_coverage']['min_season']}-{results['season_coverage']['max_season']}")
            print(f"- Missing seasons: {len(results['season_coverage']['missing_seasons'])}")
            print(f"- Empty tables: {len([t for t, c in results['table_info'].items() if c == 0])}")
            
            # Generate and save report
            report = validator.generate_report()
            report_file = f"simple_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w') as f:
                f.write(report)
            
            print(f"\n📄 Report saved to: {report_file}")
            
        else:
            print(f"❌ Validation failed: {results['message']}")
    
    except Exception as e:
        print(f"❌ Error: {e}")
    
    finally:
        validator.close()

if __name__ == "__main__":
    main()
