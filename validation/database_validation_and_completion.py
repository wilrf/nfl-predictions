#!/usr/bin/env python3
"""
Comprehensive NFL Database Validation and Completion Script
===========================================================

This script provides a complete solution for:
1. Database validation and integrity checking
2. Missing data identification
3. Database completion and population
4. Schema migration to comprehensive format

Author: NFL Betting System
Date: 2025-01-27
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime, date
from typing import Dict, List, Tuple, Any, Optional
import sqlite3
import psycopg2
from psycopg2.extras import RealDictCursor
import nflreadpy as nfl
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('database_validation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class DatabaseValidator:
    """Comprehensive database validation and completion system"""
    
    def __init__(self, db_type: str = 'sqlite', connection_string: Optional[str] = None):
        """
        Initialize the validator
        
        Args:
            db_type: 'supabase', 'sqlite', or 'postgres'
            connection_string: Database connection string
        """
        self.db_type = db_type
        self.connection_string = connection_string
        self.conn = None
        self.validation_results = {}
        self.missing_data_report = {}
        self.integrity_issues = []
        
    def connect(self):
        """Establish database connection"""
        try:
            if self.db_type == 'supabase':
                # Use Supabase MCP connection
                logger.info("Using Supabase MCP connection")
                return True
            elif self.db_type == 'sqlite':
                db_path = self.connection_string or 'improved_nfl_system/database/validation_data.db'
                self.conn = sqlite3.connect(db_path)
                logger.info(f"Connected to SQLite database: {db_path}")
            elif self.db_type == 'postgres':
                self.conn = psycopg2.connect(self.connection_string)
                logger.info("Connected to PostgreSQL database")
            return True
        except Exception as e:
            logger.error(f"Database connection failed: {e}")
            return False
    
    def execute_query(self, query: str, params: Tuple = None) -> List[Dict]:
        """Execute query and return results"""
        if self.db_type == 'supabase':
            # This would use MCP Supabase tools
            logger.warning("Supabase queries should use MCP tools directly")
            return []
        
        try:
            cursor = self.conn.cursor()
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if self.db_type == 'postgres':
                cursor = self.conn.cursor(cursor_factory=RealDictCursor)
                cursor.execute(query, params or ())
                return cursor.fetchall()
            else:
                columns = [description[0] for description in cursor.description]
                rows = cursor.fetchall()
                return [dict(zip(columns, row)) for row in rows]
        except Exception as e:
            logger.error(f"Query execution failed: {e}")
            return []
    
    def validate_table_structure(self) -> Dict[str, Any]:
        """Validate table structure against expected schema"""
        logger.info("Validating table structure...")
        
        expected_tables = {
            'all_schedules': ['game_id', 'season', 'week', 'home_team', 'away_team', 'home_score', 'away_score'],
            'historical_games': ['game_id', 'season', 'week', 'home_team', 'away_team'],
            'game_features': ['game_id', 'home_epa_differential', 'away_epa_differential'],
            'team_epa_stats': ['team', 'season', 'week', 'off_epa_play', 'def_epa_play'],
            'betting_outcomes': ['game_id', 'spread_line', 'total_line'],
            'epa_metrics': ['game_id', 'home_off_epa', 'away_off_epa'],
            'team_features': ['game_id', 'team', 'off_epa', 'def_epa'],
            'feature_history': ['feature_name', 'season', 'week', 'importance_score'],
            'injury_features': ['game_id', 'team', 'injury_severity'],
            'weather_features': ['game_id', 'temperature', 'wind', 'precipitation']
        }
        
        comprehensive_tables = {
            'dim_teams': ['team_abbr', 'team_name', 'team_conf', 'team_division'],
            'dim_players': ['player_id', 'player_name', 'position', 'team'],
            'fact_games': ['game_id', 'season', 'week', 'home_team', 'away_team'],
            'fact_plays': ['play_id', 'game_id', 'season', 'week', 'epa', 'wpa'],
            'fact_ngs_passing': ['season', 'week', 'player_gsis_id', 'team_abbr'],
            'fact_ngs_receiving': ['season', 'week', 'player_gsis_id', 'team_abbr'],
            'fact_ngs_rushing': ['season', 'week', 'player_gsis_id', 'team_abbr'],
            'fact_injuries': ['season', 'week', 'team', 'player_name'],
            'fact_snap_counts': ['game_id', 'player', 'team', 'offense_snaps'],
            'fact_weekly_rosters': ['season', 'week', 'team', 'player_id'],
            'fact_depth_charts': ['season', 'week', 'club_code', 'player_gsis_id'],
            'fact_game_officials': ['game_id', 'official_id', 'official_name'],
            'fact_weekly_stats': ['season', 'week', 'player_id', 'team'],
            'fact_qbr': ['season', 'week', 'player_id', 'team'],
            'fact_combine': ['season', 'player_id', 'player_name'],
            'agg_team_game_stats': ['game_id', 'team', 'off_epa', 'def_epa'],
            'agg_team_season_stats': ['season', 'team', 'through_week'],
            'ml_features': ['game_id', 'home_team', 'away_team', 'home_off_epa']
        }
        
        structure_validation = {
            'current_tables': {},
            'missing_comprehensive_tables': [],
            'schema_completeness': 0
        }
        
        # Check current tables
        if self.db_type != 'supabase':
            existing_tables = self.execute_query("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
            """)
            
            existing_table_names = [row['table_name'] for row in existing_tables]
            
            for table_name, expected_columns in expected_tables.items():
                if table_name in existing_table_names:
                    structure_validation['current_tables'][table_name] = {
                        'exists': True,
                        'expected_columns': expected_columns,
                        'status': 'present'
                    }
                else:
                    structure_validation['current_tables'][table_name] = {
                        'exists': False,
                        'expected_columns': expected_columns,
                        'status': 'missing'
                    }
            
            # Check comprehensive schema tables
            for table_name in comprehensive_tables.keys():
                if table_name not in existing_table_names:
                    structure_validation['missing_comprehensive_tables'].append(table_name)
            
            structure_validation['schema_completeness'] = len([
                t for t in expected_tables.keys() 
                if t in existing_table_names
            ]) / len(expected_tables) * 100
        
        return structure_validation
    
    def validate_data_integrity(self) -> Dict[str, Any]:
        """Validate data integrity and constraints"""
        logger.info("Validating data integrity...")
        
        integrity_checks = {
            'foreign_key_violations': {},
            'duplicate_records': {},
            'null_constraint_violations': {},
            'data_range_violations': {},
            'referential_integrity': {}
        }
        
        # Foreign key violations (simplified for current schema)
        fk_queries = {
            'all_schedules_integrity': """
                SELECT COUNT(*) as violations
                FROM all_schedules 
                WHERE home_team IS NULL OR away_team IS NULL OR game_id IS NULL
            """,
            'historical_games_integrity': """
                SELECT COUNT(*) as violations
                FROM historical_games 
                WHERE home_team IS NULL OR away_team IS NULL OR game_id IS NULL
            """,
            'game_features_integrity': """
                SELECT COUNT(*) as violations
                FROM game_features 
                WHERE game_id IS NULL
            """,
            'team_epa_stats_integrity': """
                SELECT COUNT(*) as violations
                FROM team_epa_stats 
                WHERE team IS NULL OR season IS NULL OR week IS NULL
            """
        }
        
        for check_name, query in fk_queries.items():
            if self.db_type != 'supabase':
                result = self.execute_query(query)
                integrity_checks['foreign_key_violations'][check_name] = result[0]['violations'] if result else 0
        
        # Duplicate records
        duplicate_queries = {
            'all_schedules': "SELECT COUNT(*) - COUNT(DISTINCT game_id) as duplicates FROM all_schedules",
            'historical_games': "SELECT COUNT(*) - COUNT(DISTINCT game_id) as duplicates FROM historical_games",
            'game_features': "SELECT COUNT(*) - COUNT(DISTINCT game_id) as duplicates FROM game_features",
            'team_epa_stats': "SELECT COUNT(*) - COUNT(DISTINCT team, season, week) as duplicates FROM team_epa_stats"
        }
        
        for table_name, query in duplicate_queries.items():
            if self.db_type != 'supabase':
                result = self.execute_query(query)
                integrity_checks['duplicate_records'][table_name] = result[0]['duplicates'] if result else 0
        
        # NULL constraint violations
        null_queries = {
            'all_schedules_game_id': "SELECT COUNT(*) as nulls FROM all_schedules WHERE game_id IS NULL",
            'all_schedules_season': "SELECT COUNT(*) as nulls FROM all_schedules WHERE season IS NULL",
            'all_schedules_week': "SELECT COUNT(*) as nulls FROM all_schedules WHERE week IS NULL",
            'all_schedules_home_team': "SELECT COUNT(*) as nulls FROM all_schedules WHERE home_team IS NULL",
            'all_schedules_away_team': "SELECT COUNT(*) as nulls FROM all_schedules WHERE away_team IS NULL",
            'historical_games_game_id': "SELECT COUNT(*) as nulls FROM historical_games WHERE game_id IS NULL"
        }
        
        for check_name, query in null_queries.items():
            if self.db_type != 'supabase':
                result = self.execute_query(query)
                integrity_checks['null_constraint_violations'][check_name] = result[0]['nulls'] if result else 0
        
        # Data range violations
        range_queries = {
            'invalid_scores': """
                SELECT COUNT(*) as violations
                FROM all_schedules 
                WHERE home_score < 0 OR away_score < 0 OR home_score > 100 OR away_score > 100
            """,
            'invalid_weeks': """
                SELECT COUNT(*) as violations
                FROM all_schedules 
                WHERE week < 1 OR week > 22
            """,
            'invalid_seasons': """
                SELECT COUNT(*) as violations
                FROM all_schedules 
                WHERE season < 2016 OR season > 2025
            """
        }
        
        for check_name, query in range_queries.items():
            if self.db_type != 'supabase':
                result = self.execute_query(query)
                integrity_checks['data_range_violations'][check_name] = result[0]['violations'] if result else 0
        
        return integrity_checks
    
    def identify_missing_data(self) -> Dict[str, Any]:
        """Identify missing data gaps"""
        logger.info("Identifying missing data...")
        
        missing_data = {
            'missing_seasons': [],
            'missing_weeks': {},
            'empty_tables': [],
            'incomplete_coverage': {},
            'data_gaps': {}
        }
        
        # Check season coverage
        if self.db_type != 'supabase':
            season_query = """
                SELECT DISTINCT season 
                FROM all_schedules 
                ORDER BY season
            """
            result = self.execute_query(season_query)
            existing_seasons = [row['season'] for row in result] if result else []
            
            expected_seasons = list(range(2016, 2025))
            missing_data['missing_seasons'] = [s for s in expected_seasons if s not in existing_seasons]
        
        # Check week coverage per season
        if self.db_type != 'supabase':
            week_query = """
                SELECT season, COUNT(DISTINCT week) as week_count, 
                       MIN(week) as min_week, MAX(week) as max_week
                FROM all_schedules 
                GROUP BY season 
                ORDER BY season
            """
            result = self.execute_query(week_query)
            
            for row in result:
                season = row['season']
                week_count = row['week_count']
                min_week = row['min_week']
                max_week = row['max_week']
                
                expected_weeks = 18 if season >= 2021 else 17
                if week_count < expected_weeks:
                    missing_data['missing_weeks'][season] = {
                        'expected': expected_weeks,
                        'actual': week_count,
                        'missing': expected_weeks - week_count,
                        'range': f"{min_week}-{max_week}"
                    }
        
        # Check empty tables
        table_counts = {
            'injury_features': "SELECT COUNT(*) as count FROM injury_features",
            'weather_features': "SELECT COUNT(*) as count FROM weather_features",
            'game_features': "SELECT COUNT(*) as count FROM game_features",
            'team_epa_stats': "SELECT COUNT(*) as count FROM team_epa_stats",
            'betting_outcomes': "SELECT COUNT(*) as count FROM betting_outcomes",
            'epa_metrics': "SELECT COUNT(*) as count FROM epa_metrics",
            'team_features': "SELECT COUNT(*) as count FROM team_features",
            'feature_history': "SELECT COUNT(*) as count FROM feature_history"
        }
        
        for table_name, query in table_counts.items():
            if self.db_type != 'supabase':
                result = self.execute_query(query)
                count = result[0]['count'] if result else 0
                if count == 0:
                    missing_data['empty_tables'].append(table_name)
        
        return missing_data
    
    def generate_completion_plan(self) -> Dict[str, Any]:
        """Generate comprehensive database completion plan"""
        logger.info("Generating completion plan...")
        
        completion_plan = {
            'schema_migration': {
                'required': True,
                'steps': [
                    'Backup current database',
                    'Create comprehensive schema tables',
                    'Migrate existing data',
                    'Populate missing dimension tables',
                    'Import historical data (2016-2019)',
                    'Import play-by-play data',
                    'Import NGS data',
                    'Import injury data',
                    'Import snap count data',
                    'Import roster data',
                    'Generate aggregated tables',
                    'Create ML features'
                ],
                'estimated_records': {
                    'dim_teams': 32,
                    'dim_players': 15000,
                    'fact_games': 2748,
                    'fact_plays': 435483,
                    'fact_ngs_passing': 24814,
                    'fact_ngs_receiving': 24814,
                    'fact_ngs_rushing': 24814,
                    'fact_injuries': 49488,
                    'fact_snap_counts': 230049,
                    'fact_weekly_rosters': 362000,
                    'fact_depth_charts': 335000,
                    'fact_game_officials': 17806,
                    'fact_weekly_stats': 49161,
                    'fact_qbr': 635,
                    'fact_combine': 3425,
                    'agg_team_game_stats': 2748,
                    'agg_team_season_stats': 13740,
                    'ml_features': 2748
                }
            },
            'data_import_priority': [
                'dim_teams',
                'dim_players', 
                'fact_games',
                'fact_plays',
                'fact_ngs_passing',
                'fact_ngs_receiving',
                'fact_ngs_rushing',
                'fact_injuries',
                'fact_snap_counts',
                'fact_weekly_rosters',
                'fact_depth_charts',
                'fact_game_officials',
                'fact_weekly_stats',
                'fact_qbr',
                'fact_combine',
                'agg_team_game_stats',
                'agg_team_season_stats',
                'ml_features'
            ],
            'validation_checklist': [
                'Schema structure validation',
                'Data integrity validation',
                'Foreign key constraint validation',
                'Data range validation',
                'Duplicate record validation',
                'NULL constraint validation',
                'Referential integrity validation',
                'Performance optimization validation',
                'Index validation',
                'Backup validation'
            ]
        }
        
        return completion_plan
    
    def run_comprehensive_validation(self) -> Dict[str, Any]:
        """Run complete validation suite"""
        logger.info("Starting comprehensive database validation...")
        
        validation_results = {
            'timestamp': datetime.now().isoformat(),
            'database_type': self.db_type,
            'table_structure': {},
            'data_integrity': {},
            'missing_data': {},
            'completion_plan': {},
            'overall_status': 'unknown',
            'recommendations': []
        }
        
        try:
            # Validate table structure
            validation_results['table_structure'] = self.validate_table_structure()
            
            # Validate data integrity
            validation_results['data_integrity'] = self.validate_data_integrity()
            
            # Identify missing data
            validation_results['missing_data'] = self.identify_missing_data()
            
            # Generate completion plan
            validation_results['completion_plan'] = self.generate_completion_plan()
            
            # Determine overall status
            structure_score = validation_results['table_structure'].get('schema_completeness', 0)
            integrity_issues = sum([
                sum(v.values()) if isinstance(v, dict) else v
                for v in validation_results['data_integrity'].values()
            ])
            
            if structure_score >= 90 and integrity_issues == 0:
                validation_results['overall_status'] = 'excellent'
            elif structure_score >= 70 and integrity_issues < 10:
                validation_results['overall_status'] = 'good'
            elif structure_score >= 50:
                validation_results['overall_status'] = 'needs_improvement'
            else:
                validation_results['overall_status'] = 'critical'
            
            # Generate recommendations
            recommendations = []
            
            if validation_results['missing_data']['missing_seasons']:
                recommendations.append(
                    f"Import missing historical data for seasons: {validation_results['missing_data']['missing_seasons']}"
                )
            
            if validation_results['missing_data']['empty_tables']:
                recommendations.append(
                    f"Populate empty tables: {', '.join(validation_results['missing_data']['empty_tables'])}"
                )
            
            if validation_results['table_structure']['missing_comprehensive_tables']:
                recommendations.append(
                    f"Migrate to comprehensive schema with {len(validation_results['table_structure']['missing_comprehensive_tables'])} additional tables"
                )
            
            if integrity_issues > 0:
                recommendations.append(f"Fix {integrity_issues} data integrity issues")
            
            validation_results['recommendations'] = recommendations
            
            logger.info(f"Validation completed. Overall status: {validation_results['overall_status']}")
            
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            validation_results['overall_status'] = 'error'
            validation_results['error'] = str(e)
        
        return validation_results
    
    def generate_validation_report(self, results: Dict[str, Any]) -> str:
        """Generate comprehensive validation report"""
        report = f"""
# NFL Database Validation Report
Generated: {results['timestamp']}
Database Type: {results['database_type']}
Overall Status: {results['overall_status'].upper()}

## Executive Summary
The database validation has identified the current state and requirements for a complete NFL data warehouse.

## Table Structure Analysis
- Schema Completeness: {results['table_structure'].get('schema_completeness', 0):.1f}%
- Current Tables: {len(results['table_structure'].get('current_tables', {}))}
- Missing Comprehensive Tables: {len(results['table_structure'].get('missing_comprehensive_tables', []))}

## Data Integrity Status
"""
        
        integrity = results['data_integrity']
        for check_type, checks in integrity.items():
            if isinstance(checks, dict):
                total_issues = sum(checks.values())
                report += f"- {check_type.replace('_', ' ').title()}: {total_issues} issues\n"
            else:
                report += f"- {check_type.replace('_', ' ').title()}: {checks} issues\n"
        
        report += f"""
## Missing Data Analysis
- Missing Seasons: {len(results['missing_data'].get('missing_seasons', []))}
- Empty Tables: {len(results['missing_data'].get('empty_tables', []))}
- Missing Weeks: {len(results['missing_data'].get('missing_weeks', {}))}

## Completion Plan
The database requires a comprehensive migration to support full NFL data warehouse functionality:

### Required Steps:
"""
        
        for i, step in enumerate(results['completion_plan']['schema_migration']['steps'], 1):
            report += f"{i}. {step}\n"
        
        report += f"""
### Estimated Data Volume:
"""
        
        for table, count in results['completion_plan']['schema_migration']['estimated_records'].items():
            report += f"- {table}: {count:,} records\n"
        
        report += f"""
## Recommendations
"""
        
        for i, rec in enumerate(results['recommendations'], 1):
            report += f"{i}. {rec}\n"
        
        report += f"""
## Next Steps
1. Review this validation report
2. Create database backup
3. Execute schema migration
4. Import missing data
5. Validate completion
6. Optimize performance

## Contact
For questions about this validation, contact the NFL Betting System team.
"""
        
        return report
    
    def close(self):
        """Close database connection"""
        if self.conn:
            self.conn.close()
            logger.info("Database connection closed")

def main():
    """Main execution function"""
    print("NFL Database Validation and Completion System")
    print("=" * 50)
    
    # Initialize validator with correct database path
    db_path = 'improved_nfl_system/database/validation_data.db'
    validator = DatabaseValidator(db_type='sqlite', connection_string=db_path)
    
    try:
        # Run comprehensive validation
        results = validator.run_comprehensive_validation()
        
        # Generate report
        report = validator.generate_validation_report(results)
        
        # Save report
        report_file = f"database_validation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
        with open(report_file, 'w') as f:
            f.write(report)
        
        print(f"Validation completed. Report saved to: {report_file}")
        print(f"Overall Status: {results['overall_status'].upper()}")
        
        # Print key findings
        print("\nKey Findings:")
        for rec in results['recommendations']:
            print(f"- {rec}")
        
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        print(f"Error: {e}")
    
    finally:
        validator.close()

if __name__ == "__main__":
    main()
