#!/usr/bin/env python3
"""
Pre-Training Readiness Checklist
Verifies all requirements before ML training
"""

import sys
import subprocess
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def check_python_version():
    """Verify Python version"""
    logger.info("\n" + "="*60)
    logger.info("1. PYTHON VERSION CHECK")
    logger.info("="*60)

    version = sys.version_info
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")

    if version.major >= 3 and version.minor >= 8:
        logger.info("✅ Python version OK (3.8+)")
        return True
    else:
        logger.error("❌ Python 3.8+ required")
        return False


def check_required_packages():
    """Check if required ML packages are installed"""
    logger.info("\n" + "="*60)
    logger.info("2. REQUIRED PACKAGES CHECK")
    logger.info("="*60)

    required = {
        'numpy': 'numpy',
        'pandas': 'pandas',
        'scikit-learn': 'sklearn',
        'xgboost': 'xgboost',
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn'
    }

    missing = []
    installed = []

    for package, import_name in required.items():
        try:
            __import__(import_name)

            # Get version
            if import_name == 'sklearn':
                import sklearn
                version = sklearn.__version__
            else:
                module = __import__(import_name)
                version = getattr(module, '__version__', 'unknown')

            logger.info(f"✅ {package:15s} - version {version}")
            installed.append(package)
        except ImportError:
            logger.warning(f"❌ {package:15s} - NOT INSTALLED")
            missing.append(package)

    if missing:
        logger.warning(f"\n⚠️  Missing packages: {', '.join(missing)}")
        logger.warning("Install with: pip install " + " ".join(missing))
        return False
    else:
        logger.info(f"\n✅ All {len(installed)} required packages installed")
        return True


def check_data_files():
    """Verify training data files exist"""
    logger.info("\n" + "="*60)
    logger.info("3. TRAINING DATA CHECK")
    logger.info("="*60)

    required_files = [
        'ml_training_data/consolidated/train.csv',
        'ml_training_data/consolidated/validation.csv',
        'ml_training_data/consolidated/test.csv',
        'ml_training_data/consolidated/feature_reference.json'
    ]

    all_exist = True
    for filepath in required_files:
        path = Path(filepath)
        if path.exists():
            size_kb = path.stat().st_size / 1024
            logger.info(f"✅ {filepath:50s} ({size_kb:>8.1f} KB)")
        else:
            logger.error(f"❌ {filepath:50s} - NOT FOUND")
            all_exist = False

    if all_exist:
        logger.info("\n✅ All training data files present")

        # Verify data quality
        import pandas as pd
        train = pd.read_csv('ml_training_data/consolidated/train.csv')
        logger.info(f"\nTraining set stats:")
        logger.info(f"  - Rows: {len(train):,}")
        logger.info(f"  - Columns: {len(train.columns)}")
        logger.info(f"  - Missing values: {train.isnull().sum().sum()}")
        logger.info(f"  - EPA coverage: {(train['epa_differential'] != 0).sum():,} / {len(train):,}")

        return True
    else:
        logger.error("\n❌ Some data files missing")
        return False


def check_model_directory():
    """Check model save directory exists"""
    logger.info("\n" + "="*60)
    logger.info("4. MODEL DIRECTORY CHECK")
    logger.info("="*60)

    model_dir = Path('models/saved_models')

    if not model_dir.exists():
        logger.warning(f"⚠️  Model directory doesn't exist: {model_dir}")
        logger.info("Creating directory...")
        model_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created: {model_dir}")
    else:
        logger.info(f"✅ Model directory exists: {model_dir}")

    # Check for existing models
    existing_models = list(model_dir.glob('*.pkl'))
    if existing_models:
        logger.info(f"\nExisting model files found:")
        for model_file in existing_models:
            size_kb = model_file.stat().st_size / 1024
            logger.info(f"  - {model_file.name} ({size_kb:.1f} KB)")
        logger.warning("⚠️  These will be overwritten during training")
    else:
        logger.info("\nNo existing models (will create new)")

    return True


def check_feature_reference():
    """Verify feature reference file"""
    logger.info("\n" + "="*60)
    logger.info("5. FEATURE REFERENCE CHECK")
    logger.info("="*60)

    import json

    ref_file = Path('ml_training_data/consolidated/feature_reference.json')

    if not ref_file.exists():
        logger.error("❌ feature_reference.json not found")
        return False

    with open(ref_file, 'r') as f:
        feature_ref = json.load(f)

    logger.info(f"✅ Feature reference loaded")
    logger.info(f"  - Total features: {feature_ref['total_features']}")
    logger.info(f"  - Feature columns: {len(feature_ref['feature_columns'])}")
    logger.info(f"  - Target columns: {len(feature_ref['target_columns'])}")

    logger.info(f"\nFeature list:")
    for i, feature in enumerate(feature_ref['feature_columns'], 1):
        desc = feature_ref['feature_descriptions'].get(feature, 'No description')
        logger.info(f"  {i:2d}. {feature:25s} - {desc}")

    return True


def check_logs_directory():
    """Verify logs directory exists"""
    logger.info("\n" + "="*60)
    logger.info("6. LOGS DIRECTORY CHECK")
    logger.info("="*60)

    logs_dir = Path('logs')

    if not logs_dir.exists():
        logger.warning(f"⚠️  Logs directory doesn't exist")
        logger.info("Creating directory...")
        logs_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ Created: {logs_dir}")
    else:
        logger.info(f"✅ Logs directory exists: {logs_dir}")

    return True


def check_disk_space():
    """Check available disk space"""
    logger.info("\n" + "="*60)
    logger.info("7. DISK SPACE CHECK")
    logger.info("="*60)

    import shutil

    total, used, free = shutil.disk_usage('.')

    free_gb = free / (1024**3)
    total_gb = total / (1024**3)
    used_pct = (used / total) * 100

    logger.info(f"Disk space:")
    logger.info(f"  - Total: {total_gb:.1f} GB")
    logger.info(f"  - Used: {used_pct:.1f}%")
    logger.info(f"  - Free: {free_gb:.1f} GB")

    if free_gb < 1:
        logger.warning("⚠️  Less than 1 GB free - consider cleaning up")
        return False
    else:
        logger.info("✅ Sufficient disk space")
        return True


def generate_install_command():
    """Generate pip install command for missing packages"""
    logger.info("\n" + "="*60)
    logger.info("INSTALLATION HELPER")
    logger.info("="*60)

    logger.info("\nIf packages are missing, run:")
    logger.info("\npip install numpy pandas scikit-learn xgboost matplotlib seaborn")
    logger.info("\nOr use requirements file:")
    logger.info("pip install -r requirements.txt")


def main():
    """Run all pre-training checks"""
    logger.info("="*60)
    logger.info("NFL ML TRAINING - PRE-FLIGHT CHECK")
    logger.info("="*60)

    checks = [
        ("Python Version", check_python_version),
        ("Required Packages", check_required_packages),
        ("Training Data", check_data_files),
        ("Model Directory", check_model_directory),
        ("Feature Reference", check_feature_reference),
        ("Logs Directory", check_logs_directory),
        ("Disk Space", check_disk_space),
    ]

    results = []
    for name, check_func in checks:
        try:
            passed = check_func()
            results.append((name, passed))
        except Exception as e:
            logger.error(f"❌ {name} check failed with error: {e}")
            results.append((name, False))

    # Summary
    logger.info("\n" + "="*60)
    logger.info("PRE-FLIGHT CHECK SUMMARY")
    logger.info("="*60)

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"{status:10s} - {name}")

    logger.info(f"\nResult: {passed}/{total} checks passed")

    if passed == total:
        logger.info("\n" + "="*60)
        logger.info("🎉 ALL CHECKS PASSED - READY FOR ML TRAINING!")
        logger.info("="*60)
        logger.info("\nNext steps:")
        logger.info("1. Run: python train_spread_model.py")
        logger.info("2. Run: python train_total_model.py")
        logger.info("3. Evaluate models on validation set")
        logger.info("4. Test final models on test set")
        return True
    else:
        logger.warning("\n" + "="*60)
        logger.warning("⚠️  SOME CHECKS FAILED")
        logger.warning("="*60)
        logger.warning("\nPlease fix the issues above before training")
        generate_install_command()
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
