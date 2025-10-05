#!/usr/bin/env python3
"""
Comprehensive data quality audit for ML training data
Checks completeness, correctness, and identifies issues
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def audit_completeness(df: pd.DataFrame):
    """Check for missing values and completeness"""
    logger.info("\n" + "="*60)
    logger.info("COMPLETENESS AUDIT")
    logger.info("="*60)

    # Check missing values
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)

    if missing.any():
        logger.warning("⚠️  Missing values found:")
        for col, count in missing[missing > 0].items():
            logger.warning(f"  {col}: {count} ({missing_pct[col]}%)")
    else:
        logger.info("✅ No missing values - all columns complete")

    # Check for zero EPA data (Week 1 is expected)
    epa_zero = (df['epa_differential'] == 0).sum()
    epa_zero_pct = (epa_zero / len(df) * 100)
    logger.info(f"\nEPA data:")
    logger.info(f"  Games with EPA = 0: {epa_zero} ({epa_zero_pct:.1f}%)")
    logger.info(f"  Expected Week 1 games: {(df['week'] == 1).sum()}")

    if epa_zero > (df['week'] == 1).sum() * 1.1:  # 10% tolerance
        logger.warning(f"⚠️  More zero EPA than expected Week 1 games")
    else:
        logger.info(f"✅ EPA coverage looks good")

    return missing


def audit_data_types(df: pd.DataFrame):
    """Verify data types are correct"""
    logger.info("\n" + "="*60)
    logger.info("DATA TYPE AUDIT")
    logger.info("="*60)

    expected_types = {
        'season': 'int',
        'week': 'int',
        'home_score': 'int',
        'away_score': 'int',
        'home_won': 'int',
        'is_home': 'int',
        'is_divisional': 'int',
        'is_outdoor': 'int',
        'epa_differential': 'float',
        'home_off_epa': 'float',
        'total_points': 'int',
        'point_differential': 'int'
    }

    issues = []
    for col, expected in expected_types.items():
        if col not in df.columns:
            issues.append(f"Missing column: {col}")
            continue

        actual = str(df[col].dtype)
        if expected == 'int' and not actual.startswith('int'):
            issues.append(f"{col}: expected {expected}, got {actual}")
        elif expected == 'float' and not actual.startswith('float'):
            issues.append(f"{col}: expected {expected}, got {actual}")

    if issues:
        logger.warning("⚠️  Data type issues:")
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("✅ All data types correct")


def audit_value_ranges(df: pd.DataFrame):
    """Check if values are in expected ranges"""
    logger.info("\n" + "="*60)
    logger.info("VALUE RANGE AUDIT")
    logger.info("="*60)

    checks = {
        'season': (2015, 2024),
        'week': (1, 18),
        'home_score': (0, 100),
        'away_score': (0, 100),
        'total_points': (0, 150),
        'point_differential': (-50, 50),
        'home_won': (0, 1),
        'is_home': (1, 1),  # Always 1
        'is_divisional': (0, 1),
        'is_outdoor': (0, 1),
        'epa_differential': (-1.5, 1.5),
        'home_off_epa': (-1.0, 1.0),
        'home_def_epa': (-1.0, 1.0)
    }

    issues = []
    for col, (min_val, max_val) in checks.items():
        if col not in df.columns:
            continue

        actual_min = df[col].min()
        actual_max = df[col].max()

        if actual_min < min_val or actual_max > max_val:
            issues.append(f"{col}: range [{actual_min:.2f}, {actual_max:.2f}] outside expected [{min_val}, {max_val}]")

        logger.info(f"{col:20s}: [{actual_min:>8.2f}, {actual_max:>8.2f}]")

    if issues:
        logger.warning("\n⚠️  Value range issues:")
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("\n✅ All values in expected ranges")


def audit_temporal_integrity(df: pd.DataFrame):
    """Verify no data leakage - features only use past data"""
    logger.info("\n" + "="*60)
    logger.info("TEMPORAL INTEGRITY AUDIT")
    logger.info("="*60)

    # Sort by season and week
    df_sorted = df.sort_values(['season', 'week']).reset_index(drop=True)

    # Check Week 1 games have zero EPA
    week1 = df_sorted[df_sorted['week'] == 1]
    week1_non_zero_epa = (week1['epa_differential'] != 0).sum()

    if week1_non_zero_epa > 0:
        logger.warning(f"⚠️  {week1_non_zero_epa} Week 1 games have non-zero EPA (data leakage?)")
    else:
        logger.info(f"✅ All Week 1 games have zero EPA (no prior data)")

    # Check that EPA values increase as season progresses
    for season in df_sorted['season'].unique():
        season_data = df_sorted[df_sorted['season'] == season]

        # Count non-zero EPA by week
        for week in range(1, 6):
            week_data = season_data[season_data['week'] == week]
            if len(week_data) == 0:
                continue

            non_zero = (week_data['epa_differential'] != 0).sum()
            pct = (non_zero / len(week_data) * 100)

            if week == 1 and pct > 5:
                logger.warning(f"  Season {season} Week {week}: {pct:.0f}% have EPA (should be ~0%)")
            elif week > 1 and pct < 50:
                logger.warning(f"  Season {season} Week {week}: Only {pct:.0f}% have EPA (expected >50%)")

    logger.info("✅ Temporal integrity check complete")


def audit_season_coverage(df: pd.DataFrame):
    """Check that all seasons are complete"""
    logger.info("\n" + "="*60)
    logger.info("SEASON COVERAGE AUDIT")
    logger.info("="*60)

    expected_games = {
        2015: 256, 2016: 256, 2017: 256, 2018: 256,
        2019: 256, 2020: 256, 2021: 272, 2022: 271,
        2023: 272, 2024: 272
    }

    issues = []
    for season, expected in expected_games.items():
        actual = len(df[df['season'] == season])

        if actual < expected:
            issues.append(f"Season {season}: {actual} games (expected {expected}) - MISSING {expected - actual}")
        elif actual > expected:
            issues.append(f"Season {season}: {actual} games (expected {expected}) - EXTRA {actual - expected}")

        logger.info(f"  {season}: {actual:3d} games (expected {expected})")

    if issues:
        logger.warning("\n⚠️  Season coverage issues:")
        for issue in issues:
            logger.warning(f"  {issue}")
    else:
        logger.info("\n✅ All seasons complete")

    return issues


def audit_target_distributions(df: pd.DataFrame):
    """Check target variable distributions"""
    logger.info("\n" + "="*60)
    logger.info("TARGET DISTRIBUTION AUDIT")
    logger.info("="*60)

    # Home win rate
    home_wins = df['home_won'].sum()
    home_win_pct = (home_wins / len(df) * 100)
    logger.info(f"Home wins: {home_wins} ({home_win_pct:.1f}%)")

    if home_win_pct < 52 or home_win_pct > 58:
        logger.warning(f"⚠️  Home win rate {home_win_pct:.1f}% outside expected 52-58%")
    else:
        logger.info(f"✅ Home win rate in expected range")

    # Point differential
    avg_diff = df['point_differential'].mean()
    logger.info(f"Avg point differential: {avg_diff:.2f}")

    if abs(avg_diff) > 3:
        logger.warning(f"⚠️  Avg point differential {avg_diff:.2f} seems high")
    else:
        logger.info(f"✅ Point differential looks good")

    # Total points
    avg_total = df['total_points'].mean()
    logger.info(f"Avg total points: {avg_total:.2f}")

    if avg_total < 40 or avg_total > 50:
        logger.warning(f"⚠️  Avg total {avg_total:.2f} outside expected 40-50")
    else:
        logger.info(f"✅ Total points in expected range")


def audit_duplicates(df: pd.DataFrame):
    """Check for duplicate games"""
    logger.info("\n" + "="*60)
    logger.info("DUPLICATE AUDIT")
    logger.info("="*60)

    # Check game_id duplicates
    duplicates = df[df.duplicated(subset=['game_id'], keep=False)]

    if len(duplicates) > 0:
        logger.warning(f"⚠️  Found {len(duplicates)} duplicate game_ids:")
        logger.warning(duplicates[['game_id', 'season', 'week', 'home_team', 'away_team']].to_string())
    else:
        logger.info("✅ No duplicate games found")


def audit_feature_correlations(df: pd.DataFrame):
    """Check feature correlations with target"""
    logger.info("\n" + "="*60)
    logger.info("FEATURE CORRELATION AUDIT")
    logger.info("="*60)

    features = [
        'epa_differential', 'is_home', 'home_off_epa', 'home_def_epa',
        'away_off_epa', 'away_def_epa', 'home_off_success_rate',
        'away_off_success_rate', 'is_divisional', 'week_number'
    ]

    # Remove rows with zero EPA for correlation analysis
    df_with_epa = df[df['epa_differential'] != 0].copy()

    logger.info(f"Correlations with 'home_won' (n={len(df_with_epa)} games with EPA):")

    correlations = []
    for feature in features:
        if feature not in df.columns:
            continue

        corr = df_with_epa[feature].corr(df_with_epa['home_won'])
        correlations.append((feature, corr))
        logger.info(f"  {feature:25s}: {corr:>7.3f}")

    # Check if any correlations are suspiciously high
    for feature, corr in correlations:
        if abs(corr) > 0.5:
            logger.warning(f"⚠️  {feature} has suspiciously high correlation: {corr:.3f}")


def main():
    """Run complete data quality audit"""
    logger.info("="*60)
    logger.info("NFL ML DATA QUALITY AUDIT")
    logger.info("="*60)

    # Load consolidated data
    data_file = Path('ml_training_data/consolidated/all_games.csv')

    if not data_file.exists():
        logger.error(f"❌ Data file not found: {data_file}")
        return

    df = pd.read_csv(data_file)
    logger.info(f"\nLoaded {len(df):,} games from {data_file}")

    # Run all audits
    audit_completeness(df)
    audit_data_types(df)
    audit_value_ranges(df)
    audit_temporal_integrity(df)
    coverage_issues = audit_season_coverage(df)
    audit_target_distributions(df)
    audit_duplicates(df)
    audit_feature_correlations(df)

    # Final summary
    logger.info("\n" + "="*60)
    logger.info("AUDIT SUMMARY")
    logger.info("="*60)

    if len(coverage_issues) > 0:
        logger.warning(f"⚠️  Found {len(coverage_issues)} season coverage issues")
        logger.warning("Recommendation: Re-run bulk import for incomplete seasons")
    else:
        logger.info("✅ Data quality audit passed!")
        logger.info("✅ Ready for ML training")

    logger.info(f"\nDataset stats:")
    logger.info(f"  Total games: {len(df):,}")
    logger.info(f"  Games with EPA: {(df['epa_differential'] != 0).sum():,}")
    logger.info(f"  Seasons: {df['season'].min()}-{df['season'].max()}")
    logger.info(f"  Home win rate: {df['home_won'].mean()*100:.1f}%")


if __name__ == '__main__':
    main()
