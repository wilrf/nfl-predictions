#!/usr/bin/env python3
"""
COMPLETE SUPABASE MIGRATION EXECUTION
=====================================
Execute all 132 SQL batch files to complete the Supabase migration.

This script will:
1. Connect to Supabase
2. Execute all SQL batch files in dependency order
3. Validate data integrity
4. Generate completion report

Author: NFL Betting System
Date: 2025-01-27
"""

import os
import glob
import json
import time
from datetime import datetime
from pathlib import Path
import psycopg2
from dotenv import load_dotenv

# Load environment variables
load_dotenv('improved_nfl_system/.env')

# Supabase connection configuration
SUPABASE_CONFIG = {
    'host': 'db.cqslvbxsqsgjagjkpiro.supabase.co',
    'port': 5432,
    'database': 'postgres',
    'user': 'postgres',
    'password': 'P@ssword9804746196$',
    'connect_timeout': 30
}

class SupabaseMigrationExecutor:
    def __init__(self):
        self.conn = None
        self.migration_log = {
            'start_time': datetime.now().isoformat(),
            'files_executed': [],
            'errors': [],
            'record_counts': {},
            'validation_results': {}
        }
        
    def connect(self):
        """Connect to Supabase database"""
        try:
            print("🔌 Connecting to Supabase...")
            self.conn = psycopg2.connect(**SUPABASE_CONFIG)
            self.conn.autocommit = True
            print("✅ Connected successfully")
            return True
        except Exception as e:
            print(f"❌ Connection failed: {e}")
            return False
    
    def execute_sql_file(self, file_path):
        """Execute a single SQL file"""
        try:
            with open(file_path, 'r') as f:
                sql = f.read()
            
            cursor = self.conn.cursor()
            cursor.execute(sql)
            cursor.close()
            
            return True
        except Exception as e:
            print(f"❌ Error executing {file_path}: {e}")
            return False
    
    def get_table_count(self, table_name):
        """Get record count for a table"""
        try:
            cursor = self.conn.cursor()
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            cursor.close()
            return count
        except Exception as e:
            print(f"❌ Error counting {table_name}: {e}")
            return 0
    
    def execute_migration(self):
        """Execute all SQL batch files in correct order"""
        print("🚀 Starting Supabase migration execution...")
        
        # Define execution order (respecting dependencies)
        execution_order = [
            'games_batch_*.sql',
            'historical_games_batch_*.sql', 
            'team_epa_stats_batch_*.sql',
            'game_features_batch_*.sql',
            'team_features_batch_*.sql',
            'epa_metrics_batch_*.sql',
            'betting_outcomes_batch_*.sql',
            'feature_history_batch_*.sql'
        ]
        
        total_files = 0
        executed_files = 0
        
        for pattern in execution_order:
            files = glob.glob(f'/tmp/{pattern}')
            files.sort()  # Execute in order
            
            if not files:
                print(f"⚠️ No files found for pattern: {pattern}")
                continue
                
            print(f"\n📊 Processing {len(files)} files for {pattern}")
            
            for file_path in files:
                total_files += 1
                file_name = os.path.basename(file_path)
                
                print(f"  Executing {file_name}...", end=' ')
                
                if self.execute_sql_file(file_path):
                    executed_files += 1
                    self.migration_log['files_executed'].append(file_name)
                    print("✅")
                else:
                    self.migration_log['errors'].append(file_name)
                    print("❌")
                
                # Small delay to avoid overwhelming the database
                time.sleep(0.1)
        
        print(f"\n📈 Execution Summary:")
        print(f"  Total files: {total_files}")
        print(f"  Executed: {executed_files}")
        print(f"  Errors: {len(self.migration_log['errors'])}")
        
        return executed_files == total_files
    
    def validate_migration(self):
        """Validate the migration results"""
        print("\n🔍 Validating migration results...")
        
        # Expected record counts
        expected_counts = {
            'games': 2678,
            'historical_games': 1087,
            'team_epa_stats': 2816,
            'game_features': 1343,
            'team_features': 2174,
            'epa_metrics': 1087,
            'betting_outcomes': 1087,
            'feature_history': 648
        }
        
        validation_passed = True
        
        for table, expected in expected_counts.items():
            actual = self.get_table_count(table)
            self.migration_log['record_counts'][table] = {
                'expected': expected,
                'actual': actual,
                'match': actual == expected
            }
            
            status = "✅" if actual == expected else "❌"
            print(f"  {table}: {actual}/{expected} {status}")
            
            if actual != expected:
                validation_passed = False
        
        # Check season coverage
        try:
            cursor = self.conn.cursor()
            cursor.execute("""
                SELECT MIN(season), MAX(season), COUNT(DISTINCT season)
                FROM games
            """)
            min_season, max_season, total_seasons = cursor.fetchone()
            cursor.close()
            
            self.migration_log['validation_results']['season_coverage'] = {
                'min_season': min_season,
                'max_season': max_season,
                'total_seasons': total_seasons,
                'expected_range': '2016-2024',
                'valid': min_season == 2016 and max_season == 2024
            }
            
            print(f"  Season coverage: {min_season}-{max_season} ({total_seasons} seasons)")
            
        except Exception as e:
            print(f"❌ Error checking season coverage: {e}")
            validation_passed = False
        
        return validation_passed
    
    def generate_report(self):
        """Generate final migration report"""
        self.migration_log['end_time'] = datetime.now().isoformat()
        self.migration_log['total_files'] = len(self.migration_log['files_executed'])
        self.migration_log['total_errors'] = len(self.migration_log['errors'])
        
        # Calculate success rate
        total_expected = sum(count['expected'] for count in self.migration_log['record_counts'].values())
        total_actual = sum(count['actual'] for count in self.migration_log['record_counts'].values())
        
        self.migration_log['success_rate'] = (total_actual / total_expected * 100) if total_expected > 0 else 0
        
        # Save report
        report_file = f"migration_completion_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.migration_log, f, indent=2)
        
        print(f"\n📝 Migration report saved to: {report_file}")
        
        # Print summary
        print(f"\n🎯 MIGRATION COMPLETION SUMMARY")
        print(f"  Files executed: {self.migration_log['total_files']}")
        print(f"  Errors: {self.migration_log['total_errors']}")
        print(f"  Success rate: {self.migration_log['success_rate']:.1f}%")
        
        if self.migration_log['success_rate'] >= 99:
            print("  Status: ✅ MIGRATION SUCCESSFUL")
        elif self.migration_log['success_rate'] >= 95:
            print("  Status: ⚠️ MIGRATION MOSTLY SUCCESSFUL")
        else:
            print("  Status: ❌ MIGRATION NEEDS ATTENTION")
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            print("🔌 Database connection closed")

def main():
    """Main execution function"""
    print("🚀 SUPABASE MIGRATION COMPLETION")
    print("=" * 50)
    
    executor = SupabaseMigrationExecutor()
    
    try:
        # Connect to database
        if not executor.connect():
            return
        
        # Execute migration
        if not executor.execute_migration():
            print("⚠️ Migration completed with errors")
        
        # Validate results
        validation_passed = executor.validate_migration()
        
        # Generate report
        executor.generate_report()
        
        if validation_passed:
            print("\n🎉 MIGRATION COMPLETED SUCCESSFULLY!")
        else:
            print("\n⚠️ MIGRATION COMPLETED WITH VALIDATION ISSUES")
            
    except Exception as e:
        print(f"❌ Migration failed: {e}")
    finally:
        executor.close()

if __name__ == "__main__":
    main()
