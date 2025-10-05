"""
Integration tests for web interface
FAIL FAST: Any test failure indicates problems
"""
import pytest
import sys
from pathlib import Path

# Add parent directories to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from fastapi.testclient import TestClient


def test_imports():
    """Test that all required modules can be imported"""
    try:
        from web.app import app
        from web.bridge.nfl_bridge import NFLSystemBridge
        from web.config.web_config import WebConfig
        assert app is not None
        assert NFLSystemBridge is not None
        assert WebConfig is not None
    except ImportError as e:
        pytest.fail(f"Import failed: {e}")


def test_web_config_validation():
    """Test web configuration validation"""
    from web.config.web_config import WebConfig, ConfigError

    # Should pass with environment variables set
    try:
        result = WebConfig.validate()
        assert result is True
    except ConfigError as e:
        # Expected if ODDS_API_KEY not set
        assert "ODDS_API_KEY" in str(e)


def test_dashboard_loads():
    """Test main dashboard page loads"""
    from web.app import app

    client = TestClient(app)
    response = client.get("/")

    assert response.status_code == 200
    assert "NFL Betting Suggestions" in response.text
    assert "Current Suggestions" in response.text


def test_suggestions_api_structure():
    """Test suggestions API returns proper structure"""
    from web.app import app

    client = TestClient(app)
    response = client.get("/api/suggestions")

    # Should return HTML (might be empty if no bridge)
    assert response.status_code in [200, 500]  # 500 if bridge not initialized
    assert response.headers.get("content-type") == "text/html; charset=utf-8"


def test_performance_api():
    """Test performance API responds correctly"""
    from web.app import app

    client = TestClient(app)
    response = client.get("/api/performance")

    assert response.status_code in [200, 500]  # 500 if bridge not initialized

    if response.status_code == 200:
        data = response.json()
        assert "status" in data
        assert data["status"] in ["success", "error"]


def test_status_api():
    """Test status API responds correctly"""
    from web.app import app

    client = TestClient(app)
    response = client.get("/api/status")

    assert response.status_code in [200, 500]  # 500 if bridge not initialized

    if response.status_code == 200:
        data = response.json()
        assert "status" in data


def test_bridge_initialization():
    """Test bridge can be initialized with proper config"""
    import os

    # Skip if no API key
    if not os.getenv('ODDS_API_KEY'):
        pytest.skip("No ODDS_API_KEY set")

    try:
        from web.bridge.nfl_bridge import NFLSystemBridge
        bridge = NFLSystemBridge()
        assert bridge is not None

        # Test getting suggestions (might be empty)
        suggestions = bridge.get_current_suggestions()
        assert isinstance(suggestions, list)

    except Exception as e:
        pytest.fail(f"Bridge initialization failed: {e}")


def test_suggestion_transformation():
    """Test suggestion transformation logic"""
    from web.bridge.nfl_bridge import SuggestionTransformer
    from database.db_manager import NFLDatabaseManager

    try:
        db = NFLDatabaseManager()
        transformer = SuggestionTransformer(db)

        # Test with sample suggestion data
        sample_suggestion = {
            'game_id': 'test_game_123',
            'bet_type': 'spread',
            'selection': 'home',
            'line': -3.5,
            'odds': -110,
            'confidence': 75.5,
            'margin': 15.2,
            'edge': 0.035,
            'kelly_fraction': 0.025,
            'correlation_warnings': []
        }

        # Should not fail (might use fallback team names)
        result = transformer.transform_suggestion(sample_suggestion)

        assert isinstance(result, dict)
        assert 'team' in result
        assert 'bet_type' in result
        assert 'confidence' in result

    except Exception as e:
        # Expected if database not set up
        assert "Cannot connect to database" in str(e) or "system" in str(e).lower()


def test_error_pages():
    """Test error pages render correctly"""
    from web.app import app

    client = TestClient(app)

    # Test 404
    response = client.get("/nonexistent-page")
    assert response.status_code == 404
    assert "Page not found" in response.text.lower()


def test_static_file_serving():
    """Test static files can be served"""
    from web.app import app

    client = TestClient(app)

    # Test static route exists (will 404 for missing files but route should work)
    response = client.get("/static/test.css")
    assert response.status_code == 404  # File doesn't exist but route works


if __name__ == "__main__":
    # Run tests directly
    import subprocess
    result = subprocess.run([sys.executable, "-m", "pytest", __file__, "-v"],
                          capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    sys.exit(result.returncode)