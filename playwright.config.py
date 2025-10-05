"""
Playwright Configuration for NFL Betting System
Black & White Theme Visual Testing
"""
import os
from playwright.sync_api import sync_playwright

# Playwright test configuration
PLAYWRIGHT_CONFIG = {
    "base_url": "http://localhost:8000",
    "headless": True,
    "slow_mo": 0,
    "timeout": 30000,
    "screenshot_on_failure": True,
    "video_on_failure": True,
    "trace_on_failure": True,
}

# Visual regression testing paths
SCREENSHOT_DIR = "improved_nfl_system/web/tests/screenshots"
BASELINE_DIR = os.path.join(SCREENSHOT_DIR, "baseline")
CURRENT_DIR = os.path.join(SCREENSHOT_DIR, "current")
DIFF_DIR = os.path.join(SCREENSHOT_DIR, "diff")

# Create directories if they don't exist
for directory in [SCREENSHOT_DIR, BASELINE_DIR, CURRENT_DIR, DIFF_DIR]:
    os.makedirs(directory, exist_ok=True)

# Browser configurations
BROWSERS = ["chromium"]  # Add "firefox", "webkit" for cross-browser testing

# Viewport sizes for responsive testing
VIEWPORTS = {
    "mobile": {"width": 375, "height": 667},
    "tablet": {"width": 768, "height": 1024},
    "desktop": {"width": 1920, "height": 1080},
}

# Black & White Theme Color Palette (for validation)
THEME_COLORS = {
    "primary_black": "#000000",
    "charcoal": "#1a1a1a",
    "dark_gray": "#2d2d2d",
    "medium_gray": "#6b6b6b",
    "light_gray": "#e5e5e5",
    "pure_white": "#ffffff",
    "accent": "#f0f0f0",
}
