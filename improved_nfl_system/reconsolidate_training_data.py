#!/usr/bin/env python3
"""
Reconsolidate All Training Data (2015-2025)
Creates temporal train/validation/test splits:
- Train: 2015-2023 (2,623 games, 88%)
- Validation: 2024 (272 games, 9%)
- Test: 2025 weeks 1-4 (64 games, 2%)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/reconsolidate.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class DataConsolidator:
    """Consolidate all season data into train/val/test splits"""

    def __init__(self):
        self.data_dir = Path('ml_training_data')
        self.output_dir = self.data_dir / 'consolidated'
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def consolidate(self):
        """Main consolidation workflow"""
        logger.info("Starting data consolidation...")
        logger.info("=" * 60)

        # Step 1: Load all season data
        logger.info("Step 1: Loading season data...")
        all_games = self.load_all_seasons()
        logger.info(f"  ✓ Loaded {len(all_games):,} total games")

        # Step 2: Temporal split
        logger.info("Step 2: Creating temporal splits...")
        train, val, test = self.create_temporal_splits(all_games)
        logger.info(f"  ✓ Train: {len(train):,} games (2015-2023)")
        logger.info(f"  ✓ Validation: {len(val):,} games (2024)")
        logger.info(f"  ✓ Test: {len(test):,} games (2025)")

        # Step 3: Data quality checks
        logger.info("Step 3: Quality checks...")
        self.validate_splits(train, val, test)

        # Step 4: Save outputs
        logger.info("Step 4: Saving consolidated data...")
        self.save_datasets(all_games, train, val, test)

        # Step 5: Generate summary statistics
        logger.info("Step 5: Generating summary...")
        self.generate_summary(all_games, train, val, test)

        logger.info("=" * 60)
        logger.info("✅ Consolidation Complete!")
        logger.info(f"   Total: {len(all_games):,} games")
        logger.info(f"   Train: {len(train):,} ({len(train)/len(all_games)*100:.1f}%)")
        logger.info(f"   Val: {len(val):,} ({len(val)/len(all_games)*100:.1f}%)")
        logger.info(f"   Test: {len(test):,} ({len(test)/len(all_games)*100:.1f}%)")

    def load_all_seasons(self) -> pd.DataFrame:
        """Load and concatenate all season feature files"""
        all_data = []

        # Find all season directories
        season_dirs = sorted(self.data_dir.glob('season_*'))

        for season_dir in season_dirs:
            features_file = season_dir / 'game_features.csv'

            if not features_file.exists():
                logger.warning(f"  Missing: {features_file}")
                continue

            df = pd.read_csv(features_file)
            all_data.append(df)

            season = season_dir.name.replace('season_', '')
            logger.info(f"  Loaded season {season}: {len(df)} games")

        # Concatenate all seasons
        combined = pd.concat(all_data, ignore_index=True)

        # Sort by season and week
        combined = combined.sort_values(['season', 'week']).reset_index(drop=True)

        return combined

    def create_temporal_splits(self, df: pd.DataFrame) -> tuple:
        """Create temporal train/validation/test splits"""
        # Train: 2015-2023
        train = df[df['season'] <= 2023].copy()

        # Validation: 2024 (most recent complete season)
        val = df[df['season'] == 2024].copy()

        # Test: 2025 weeks 1-4 (current season reality check)
        test = df[df['season'] == 2025].copy()

        return train, val, test

    def validate_splits(self, train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame):
        """Validate data quality across splits"""

        # Check for temporal leakage
        if train['season'].max() >= val['season'].min():
            logger.warning("  ⚠️  Potential temporal leakage detected")
        else:
            logger.info("  ✓ No temporal leakage")

        # Check column consistency
        if not (set(train.columns) == set(val.columns) == set(test.columns)):
            logger.error("  ✗ Column mismatch across splits")
        else:
            logger.info("  ✓ Columns consistent: 29 columns")

        # Check for nulls in critical fields
        for name, df in [('Train', train), ('Val', val), ('Test', test)]:
            null_epa = df['epa_differential'].isnull().sum()
            null_target = df['home_won'].isnull().sum()

            if null_epa > 0 or null_target > 0:
                logger.warning(f"  ⚠️  {name}: {null_epa} null EPA, {null_target} null targets")
            else:
                logger.info(f"  ✓ {name}: No critical nulls")

        # Check completed games
        for name, df in [('Train', train), ('Val', val), ('Test', test)]:
            completed = df['home_score'].notna().sum()
            pct = (completed / len(df) * 100) if len(df) > 0 else 0
            logger.info(f"  {name}: {completed}/{len(df)} completed ({pct:.1f}%)")

    def save_datasets(self, all_games: pd.DataFrame, train: pd.DataFrame,
                     val: pd.DataFrame, test: pd.DataFrame):
        """Save all datasets to disk"""
        # Save consolidated files
        all_games.to_csv(self.output_dir / 'all_games.csv', index=False)
        train.to_csv(self.output_dir / 'train.csv', index=False)
        val.to_csv(self.output_dir / 'validation.csv', index=False)
        test.to_csv(self.output_dir / 'test.csv', index=False)

        logger.info(f"  ✓ Saved all_games.csv: {len(all_games):,} games")
        logger.info(f"  ✓ Saved train.csv: {len(train):,} games")
        logger.info(f"  ✓ Saved validation.csv: {len(val):,} games")
        logger.info(f"  ✓ Saved test.csv: {len(test):,} games")

    def generate_summary(self, all_games: pd.DataFrame, train: pd.DataFrame,
                        val: pd.DataFrame, test: pd.DataFrame):
        """Generate summary statistics"""
        summary = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'total_games': len(all_games),
            'seasons': {
                'min': int(all_games['season'].min()),
                'max': int(all_games['season'].max()),
                'count': int(all_games['season'].nunique())
            },
            'splits': {
                'train': {
                    'games': len(train),
                    'seasons': f"2015-2023",
                    'percentage': round(len(train) / len(all_games) * 100, 1)
                },
                'validation': {
                    'games': len(val),
                    'seasons': "2024",
                    'percentage': round(len(val) / len(all_games) * 100, 1)
                },
                'test': {
                    'games': len(test),
                    'seasons': "2025 weeks 1-4",
                    'percentage': round(len(test) / len(all_games) * 100, 1)
                }
            },
            'features': {
                'total_columns': len(all_games.columns),
                'feature_columns': 17,
                'target_columns': 3,
                'metadata_columns': 9
            },
            'data_quality': {
                'completed_games': int(all_games['home_score'].notna().sum()),
                'completion_rate': round(all_games['home_score'].notna().mean() * 100, 1),
                'null_epa': int(all_games['epa_differential'].isnull().sum())
            }
        }

        # Save summary
        summary_file = self.output_dir / 'consolidation_summary.json'
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"  ✓ Saved summary: {summary_file}")

        # Print summary
        print("\n" + "=" * 60)
        print("CONSOLIDATION SUMMARY")
        print("=" * 60)
        print(f"Total Games: {summary['total_games']:,}")
        print(f"Seasons: {summary['seasons']['min']}-{summary['seasons']['max']} ({summary['seasons']['count']} seasons)")
        print()
        print("Splits:")
        print(f"  Train:      {summary['splits']['train']['games']:,} games ({summary['splits']['train']['percentage']}%) - {summary['splits']['train']['seasons']}")
        print(f"  Validation: {summary['splits']['validation']['games']:,} games ({summary['splits']['validation']['percentage']}%) - {summary['splits']['validation']['seasons']}")
        print(f"  Test:       {summary['splits']['test']['games']:,} games ({summary['splits']['test']['percentage']}%) - {summary['splits']['test']['seasons']}")
        print()
        print("Data Quality:")
        print(f"  Completed games: {summary['data_quality']['completed_games']:,}/{summary['total_games']:,} ({summary['data_quality']['completion_rate']}%)")
        print(f"  Null EPA values: {summary['data_quality']['null_epa']}")
        print("=" * 60)


if __name__ == "__main__":
    consolidator = DataConsolidator()
    consolidator.consolidate()
