#!/usr/bin/env python3
"""
Comprehensive Data Importer - Phase 1 Implementation
Fail-fast validation gates to ensure data quality before proceeding
"""

import pandas as pd
import numpy as np
from supabase import create_client
import os
import logging
from datetime import datetime
from pathlib import Path
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComprehensiveDataImporter:
    """Import all available NFL data sources with validation gates"""

    def __init__(self):
        # Get Supabase credentials from environment variables
        self.supabase_url = os.getenv('SUPABASE_URL')
        self.supabase_key = os.getenv('SUPABASE_SERVICE_KEY')

        if not self.supabase_url or not self.supabase_key:
            raise ValueError(
                "Missing required environment variables: SUPABASE_URL and/or SUPABASE_SERVICE_KEY. "
                "Please set them in your .env file."
            )

        self.supabase = create_client(self.supabase_url, self.supabase_key)
        self.output_dir = Path('data_integration/output')
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validation thresholds
        self.min_games_threshold = 2000
        self.min_ngs_records_threshold = 500
        self.min_injury_records_threshold = 1000
        
        logger.info("ComprehensiveDataImporter initialized")
    
    def import_all_data_with_validation(self):
        """Import all data with fail-fast validation gates"""
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE DATA IMPORT WITH VALIDATION GATES")
        logger.info("=" * 60)
        
        results = {}
        
        # Gate 1: Core Games Data
        logger.info("\n🔍 VALIDATION GATE 1: Core Games Data")
        games_data = self._validate_and_import_games()
        if games_data is None:
            logger.error("❌ FAILED: Core games data validation failed")
            return None
        results['games'] = games_data
        logger.info("✅ PASSED: Core games data validation")
        
        # Gate 2: NGS Data
        logger.info("\n🔍 VALIDATION GATE 2: Next Gen Stats Data")
        ngs_data = self._validate_and_import_ngs()
        if ngs_data is None:
            logger.error("❌ FAILED: NGS data validation failed")
            return None
        results['ngs'] = ngs_data
        logger.info("✅ PASSED: NGS data validation")
        
        # Gate 3: Injury Data
        logger.info("\n🔍 VALIDATION GATE 3: Injury Data")
        injury_data = self._validate_and_import_injuries()
        if injury_data is None:
            logger.error("❌ FAILED: Injury data validation failed")
            return None
        results['injuries'] = injury_data
        logger.info("✅ PASSED: Injury data validation")
        
        # Gate 4: Data Quality Check
        logger.info("\n🔍 VALIDATION GATE 4: Overall Data Quality")
        quality_score = self._validate_data_quality(results)
        if quality_score < 0.8:
            logger.error(f"❌ FAILED: Data quality score {quality_score:.2f} below threshold 0.8")
            return None
        logger.info(f"✅ PASSED: Data quality score {quality_score:.2f}")
        
        # Save results
        self._save_import_results(results)
        
        logger.info("\n🎉 ALL VALIDATION GATES PASSED - PROCEEDING TO PHASE 2")
        return results
    
    def _validate_and_import_games(self):
        """Validate and import core games data"""
        try:
            # Import games with pagination to get all records
            # Supabase has a default limit of 1000, so we need to use range queries
            all_games = []
            offset = 0
            limit = 1000
            
            while True:
                result = self.supabase.table('games').select('*').range(offset, offset + limit - 1).execute()
                if not result.data:
                    break
                all_games.extend(result.data)
                if len(result.data) < limit:
                    break
                offset += limit
            
            games_df = pd.DataFrame(all_games)
            
            logger.info(f"Imported {len(games_df)} games")
            
            # Validation checks
            if len(games_df) < self.min_games_threshold:
                logger.error(f"❌ Insufficient games: {len(games_df)} < {self.min_games_threshold}")
                return None
            
            # Check for required columns
            required_columns = ['game_id', 'season', 'week', 'home_team', 'away_team']
            missing_columns = [col for col in required_columns if col not in games_df.columns]
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return None
            
            # Check for data completeness
            null_counts = games_df[required_columns].isnull().sum()
            if null_counts.sum() > 0:
                logger.error(f"❌ Null values in required columns: {null_counts.to_dict()}")
                return None
            
            # Check season range
            seasons = games_df['season'].unique()
            logger.info(f"Seasons available: {sorted(seasons)}")
            
            if len(seasons) < 5:
                logger.error(f"❌ Insufficient season coverage: {len(seasons)} seasons")
                return None
            
            logger.info(f"✅ Games validation passed: {len(games_df)} games, {len(seasons)} seasons")
            return games_df
            
        except Exception as e:
            logger.error(f"❌ Games import failed: {e}")
            return None
    
    def _validate_and_import_ngs(self):
        """Validate and import Next Gen Stats data"""
        try:
            ngs_data = {}
            
            # Import NGS passing
            passing_result = self.supabase.table('fact_ngs_passing').select('*').execute()
            ngs_data['passing'] = pd.DataFrame(passing_result.data)
            
            # Import NGS rushing
            rushing_result = self.supabase.table('fact_ngs_rushing').select('*').execute()
            ngs_data['rushing'] = pd.DataFrame(rushing_result.data)
            
            # Import NGS receiving
            receiving_result = self.supabase.table('fact_ngs_receiving').select('*').execute()
            ngs_data['receiving'] = pd.DataFrame(receiving_result.data)
            
            total_records = sum(len(df) for df in ngs_data.values())
            logger.info(f"Imported {total_records} NGS records")
            
            # Validation checks
            if total_records < self.min_ngs_records_threshold:
                logger.error(f"❌ Insufficient NGS records: {total_records} < {self.min_ngs_records_threshold}")
                return None
            
            # Check for key NGS features
            key_features = {
                'passing': ['completion_percentage_above_expectation', 'avg_time_to_throw', 'aggressiveness'],
                'rushing': ['efficiency', 'rush_yards_over_expected'],
                'receiving': ['avg_separation', 'avg_cushion']
            }
            
            for category, features in key_features.items():
                if category in ngs_data:
                    missing_features = [f for f in features if f not in ngs_data[category].columns]
                    if missing_features:
                        logger.warning(f"⚠️ Missing NGS features in {category}: {missing_features}")
            
            logger.info(f"✅ NGS validation passed: {total_records} records across 3 categories")
            return ngs_data
            
        except Exception as e:
            logger.error(f"❌ NGS import failed: {e}")
            return None
    
    def _validate_and_import_injuries(self):
        """Validate and import injury data"""
        try:
            # Import injuries
            result = self.supabase.table('fact_injuries').select('*').execute()
            injuries_df = pd.DataFrame(result.data)
            
            logger.info(f"Imported {len(injuries_df)} injury records")
            
            # Validation checks
            if len(injuries_df) < self.min_injury_records_threshold:
                logger.error(f"❌ Insufficient injury records: {len(injuries_df)} < {self.min_injury_records_threshold}")
                return None
            
            # Check for required columns
            required_columns = ['season', 'week', 'team', 'player_name', 'position']
            missing_columns = [col for col in required_columns if col not in injuries_df.columns]
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return None
            
            # Check for QB injury data (critical for model)
            qb_injuries = injuries_df[injuries_df['position'] == 'QB']
            if len(qb_injuries) < 100:
                logger.warning(f"⚠️ Limited QB injury data: {len(qb_injuries)} records")
            
            logger.info(f"✅ Injury validation passed: {len(injuries_df)} records, {len(qb_injuries)} QB records")
            return injuries_df
            
        except Exception as e:
            logger.error(f"❌ Injury import failed: {e}")
            return None
    
    def _validate_data_quality(self, results):
        """Validate overall data quality"""
        quality_score = 0.0
        max_score = 4.0
        
        # Games data quality (1.0 points)
        if 'games' in results and len(results['games']) > 0:
            games_df = results['games']
            completeness = 1 - (games_df.isnull().sum().sum() / (len(games_df) * len(games_df.columns)))
            quality_score += min(completeness, 1.0)
        
        # NGS data quality (1.0 points)
        if 'ngs' in results:
            ngs_total = sum(len(df) for df in results['ngs'].values())
            if ngs_total > 1000:
                quality_score += 1.0
            elif ngs_total > 500:
                quality_score += 0.5
        
        # Injury data quality (1.0 points)
        if 'injuries' in results and len(results['injuries']) > 0:
            injuries_df = results['injuries']
            qb_coverage = len(injuries_df[injuries_df['position'] == 'QB']) / len(injuries_df)
            quality_score += min(qb_coverage * 10, 1.0)  # QB coverage is critical
        
        # Data consistency (1.0 points)
        if 'games' in results and 'ngs' in results:
            games_df = results['games']
            ngs_df = results['ngs']['passing']  # Use passing as reference
            
            # Check for overlapping game_ids
            games_ids = set(games_df['game_id'])
            ngs_ids = set(ngs_df['game_id'])
            overlap = len(games_ids.intersection(ngs_ids))
            
            if overlap > 100:
                quality_score += 1.0
            elif overlap > 50:
                quality_score += 0.5
        
        final_score = quality_score / max_score
        logger.info(f"Data quality score: {final_score:.2f} ({quality_score:.1f}/{max_score})")
        
        return final_score
    
    def _save_import_results(self, results):
        """Save import results for next phase"""
        # Save games data
        if 'games' in results:
            results['games'].to_csv(self.output_dir / 'games_complete.csv', index=False)
        
        # Save NGS data
        if 'ngs' in results:
            for category, df in results['ngs'].items():
                df.to_csv(self.output_dir / f'ngs_{category}.csv', index=False)
        
        # Save injury data
        if 'injuries' in results:
            results['injuries'].to_csv(self.output_dir / 'injuries_complete.csv', index=False)
        
        # Save metadata
        metadata = {
            'import_timestamp': datetime.now().isoformat(),
            'games_count': len(results['games']) if 'games' in results else 0,
            'ngs_records': sum(len(df) for df in results['ngs'].values()) if 'ngs' in results else 0,
            'injury_records': len(results['injuries']) if 'injuries' in results else 0,
            'validation_passed': True
        }
        
        with open(self.output_dir / 'import_metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Import results saved to {self.output_dir}")

if __name__ == "__main__":
    importer = ComprehensiveDataImporter()
    results = importer.import_all_data_with_validation()
    
    if results:
        logger.info("🎉 Phase 1 Data Foundation: SUCCESS")
        logger.info("Ready to proceed to Phase 2: Feature Engineering")
    else:
        logger.error("❌ Phase 1 Data Foundation: FAILED")
        logger.error("Fix data issues before proceeding")
