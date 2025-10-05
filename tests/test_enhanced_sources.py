"""
Test Enhanced Data Sources
Quick tests to verify ESPN, Weather, and NFL.com integrations work
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_sources.espn_client import ESPNClient
from data_sources.weather_client import WeatherClient
from data_sources.nfl_official_client import NFLOfficialClient
from data_pipeline import NFLDataPipeline
import logging
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_espn_client():
    """Test ESPN API client functionality"""
    print("\n" + "="*50)
    print("TESTING ESPN CLIENT")
    print("="*50)

    client = ESPNClient()

    try:
        # Test live scores
        print("\n1. Testing ESPN Live Scores...")
        scores = client.get_live_scores(2024, 1)
        print(f"   ✅ Retrieved {len(scores)} games")
        if not scores.empty:
            print(f"   Sample game: {scores.iloc[0]['away_team']} @ {scores.iloc[0]['home_team']}")

        # Test standings
        print("\n2. Testing ESPN Standings...")
        standings = client.get_standings(2024)
        print(f"   ✅ Retrieved {len(standings)} teams")
        if not standings.empty:
            print(f"   Sample team: {standings.iloc[0]['team_name']} ({standings.iloc[0]['wins']}-{standings.iloc[0]['losses']})")

        # Test team stats
        print("\n3. Testing ESPN Team Stats...")
        team_stats = client.get_team_stats(2024)
        print(f"   ✅ Retrieved {len(team_stats)} teams")
        if not team_stats.empty:
            print(f"   Sample team: {team_stats.iloc[0]['team_name']}")

        return True

    except Exception as e:
        print(f"   ❌ ESPN test failed: {e}")
        return False


def test_weather_client():
    """Test Weather API client functionality"""
    print("\n" + "="*50)
    print("TESTING WEATHER CLIENT")
    print("="*50)

    # Test without API key first
    client = WeatherClient()

    try:
        # Test Green Bay weather (outdoor stadium)
        print("\n1. Testing Weather for Green Bay...")
        weather = client.get_game_weather('GB', datetime.now())
        print(f"   ✅ Retrieved weather data: {len(weather)} fields")
        if weather:
            print(f"   Temperature: {weather.get('temperature', 'N/A')}")
            print(f"   Conditions: {weather.get('conditions', 'N/A')}")

        # Test indoor stadium
        print("\n2. Testing Weather for Indoor Stadium (Dallas)...")
        indoor_weather = client.get_game_weather('DAL', datetime.now())
        print(f"   ✅ Indoor conditions: {indoor_weather.get('conditions', 'N/A')}")

        # Test National Weather Service
        print("\n3. Testing NWS Data...")
        nws_data = client._get_nws_data(44.5013, -88.0622, datetime.now())  # Lambeau Field
        print(f"   ✅ NWS data: {len(nws_data)} fields")
        if nws_data:
            print(f"   NWS conditions: {nws_data.get('nws_conditions', 'N/A')}")

        return True

    except Exception as e:
        print(f"   ❌ Weather test failed: {e}")
        return False


def test_nfl_official_client():
    """Test NFL.com official client functionality"""
    print("\n" + "="*50)
    print("TESTING NFL OFFICIAL CLIENT")
    print("="*50)

    client = NFLOfficialClient()

    try:
        # Test live scores
        print("\n1. Testing NFL.com Live Scores...")
        scores = client.get_live_scores(2024, 1)
        print(f"   ✅ Retrieved {len(scores)} games")
        if not scores.empty:
            print(f"   Sample game: {scores.iloc[0]['away_team']} @ {scores.iloc[0]['home_team']}")

        # Test endpoints info
        print("\n2. Available NFL.com Endpoints...")
        from data_sources.nfl_official_client import NFLAPIAlternatives
        endpoints = NFLAPIAlternatives.get_nfl_feed_endpoints()
        print(f"   ✅ {len(endpoints)} endpoints available:")
        for name, url in endpoints.items():
            print(f"     - {name}: {url}")

        return True

    except Exception as e:
        print(f"   ❌ NFL Official test failed: {e}")
        return False


def test_enhanced_pipeline():
    """Test the enhanced pipeline with all sources"""
    print("\n" + "="*50)
    print("TESTING ENHANCED PIPELINE")
    print("="*50)

    # Mock config
    config = {
        'openweather_api_key': None,  # Test without API key
        'odds_sources': {
            'sharp_books': [],
            'soft_books': [],
            'api_keys': {}
        }
    }

    try:
        # Initialize enhanced pipeline
        print("\n1. Initializing Enhanced Pipeline...")
        pipeline = NFLDataPipeline(config)
        print(f"   ✅ Pipeline initialized")
        print(f"   Enhanced sources: {pipeline.enhanced_sources_available}")

        # Test data source status
        print("\n2. Testing Data Source Status...")
        status = pipeline.get_data_source_status()
        print("   ✅ Status retrieved:")
        for key, value in status.items():
            if isinstance(value, dict):
                print(f"     {key}: {value}")
            else:
                print(f"     {key}: {value}")

        # Test weekly data collection
        print("\n3. Testing Weekly Data Collection...")
        weekly_data = pipeline.get_weekly_data(1, 2024)
        print(f"   ✅ Weekly data collected:")
        for data_type, df in weekly_data.items():
            print(f"     {data_type}: {len(df)} records")

        return True

    except Exception as e:
        print(f"   ❌ Enhanced pipeline test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🏈 TESTING ENHANCED NFL DATA SOURCES")
    print("====================================")

    results = {}

    # Test individual clients
    results['espn'] = test_espn_client()
    results['weather'] = test_weather_client()
    results['nfl_official'] = test_nfl_official_client()
    results['enhanced_pipeline'] = test_enhanced_pipeline()

    # Summary
    print("\n" + "="*50)
    print("TEST RESULTS SUMMARY")
    print("="*50)

    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name.upper()}: {status}")

    total_passed = sum(results.values())
    print(f"\nOverall: {total_passed}/{len(results)} tests passed")

    if total_passed == len(results):
        print("\n🎉 All enhanced data sources are working!")
        print("\nNext steps:")
        print("- Add OpenWeatherMap API key for enhanced weather data")
        print("- Configure odds API keys for betting lines")
        print("- Set up automated data collection schedules")
    else:
        print(f"\n⚠️  {len(results) - total_passed} tests failed")
        print("Check the logs above for specific error details")


if __name__ == "__main__":
    main()