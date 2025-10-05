"""
Centralized configuration for web interface
FAIL FAST: Missing config stops system
"""
import os
from pathlib import Path
from typing import Dict, List


class WebConfig:
    """Web interface configuration"""

    # Confidence tier thresholds
    CONFIDENCE_TIERS = {
        'premium': float(os.getenv('PREMIUM_THRESHOLD', 80)),
        'standard': float(os.getenv('STANDARD_THRESHOLD', 65)),
        'reference': float(os.getenv('REFERENCE_THRESHOLD', 50))
    }

    # Enabled bet types (configurable)
    ENABLED_BET_TYPES = os.getenv('ENABLED_BET_TYPES', 'spread,total,moneyline').split(',')

    # Update intervals (in seconds)
    UPDATE_INTERVALS = {
        'suggestions': int(os.getenv('SUGGESTIONS_UPDATE_INTERVAL', 3600)),  # 1 hour
        'performance': int(os.getenv('PERFORMANCE_UPDATE_INTERVAL', 7200))   # 2 hours
    }

    # Web server settings
    WEB_HOST = os.getenv('WEB_HOST', '0.0.0.0')
    WEB_PORT = int(os.getenv('WEB_PORT', 8000))

    # Static file paths
    BASE_DIR = Path(__file__).parent.parent
    STATIC_DIR = BASE_DIR / 'static'
    TEMPLATES_DIR = BASE_DIR / 'templates'

    # Theme settings
    THEME = {
        'primary_color': '#10B981',  # Green
        'secondary_color': '#F59E0B',  # Yellow
        'danger_color': '#EF4444',   # Red
        'dark_bg': '#111827',        # Dark gray
        'card_bg': '#1F2937'         # Medium gray
    }

    @classmethod
    def validate(cls) -> bool:
        """Validate all required configuration exists"""
        required_vars = ['ODDS_API_KEY']
        missing = [var for var in required_vars if not os.getenv(var)]

        if missing:
            raise ConfigError(f"Missing required environment variables: {missing}")

        # Validate directories exist
        cls.STATIC_DIR.mkdir(parents=True, exist_ok=True)
        cls.TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

        return True

    @classmethod
    def get_tier_config(cls, tier: str) -> Dict:
        """Get tier-specific configuration"""
        tier_configs = {
            'premium': {
                'title': '🟢 PREMIUM PICKS',
                'border_color': 'border-green-500',
                'bg_color': 'bg-green-900/20',
                'text_color': 'text-green-400',
                'threshold': cls.CONFIDENCE_TIERS['premium']
            },
            'standard': {
                'title': '🟡 STANDARD PICKS',
                'border_color': 'border-yellow-500',
                'bg_color': 'bg-yellow-900/20',
                'text_color': 'text-yellow-400',
                'threshold': cls.CONFIDENCE_TIERS['standard']
            },
            'reference': {
                'title': '⚪ REFERENCE PICKS',
                'border_color': 'border-gray-500',
                'bg_color': 'bg-gray-900/20',
                'text_color': 'text-gray-400',
                'threshold': cls.CONFIDENCE_TIERS['reference']
            }
        }
        return tier_configs.get(tier, tier_configs['reference'])


class ConfigError(Exception):
    """Configuration validation error"""
    pass