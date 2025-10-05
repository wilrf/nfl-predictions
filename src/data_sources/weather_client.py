"""
Weather API Client
Multiple free weather sources for NFL game conditions
Includes current conditions, forecasts, and historical data
"""

import requests
import pandas as pd
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
import time
import json

logger = logging.getLogger(__name__)


class WeatherClient:
    """Multi-source weather client for NFL games"""

    def __init__(self, openweather_api_key: str = None):
        self.openweather_key = openweather_api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'NFL Weather Data Client'
        })

        # NFL stadium coordinates (outdoor venues only)
        self.stadium_coords = {
            'GB': (44.5013, -88.0622),   # Lambeau Field
            'CHI': (41.8623, -87.6167),  # Soldier Field
            'BUF': (42.7738, -78.7870),  # Highmark Stadium
            'NE': (42.0909, -71.2643),   # Gillette Stadium
            'NYJ': (40.8135, -74.0745),  # MetLife Stadium
            'NYG': (40.8135, -74.0745),  # MetLife Stadium
            'CLE': (41.5061, -81.6995),  # Cleveland Browns Stadium
            'CIN': (39.0955, -84.5161),  # Paycor Stadium
            'PIT': (40.4468, -80.0158),  # Heinz Field
            'BAL': (39.2780, -76.6227),  # M&T Bank Stadium
            'WAS': (38.9076, -76.8645),  # FedExField
            'PHI': (39.9008, -75.1675),  # Lincoln Financial Field
            'DEN': (39.7439, -105.0201), # Empower Field at Mile High
            'KC': (39.0489, -94.4839),   # Arrowhead Stadium
            'LV': (36.0906, -115.1831),  # Allegiant Stadium (indoor, but included)
            'LAC': (33.8644, -117.9238), # SoFi Stadium (dome, but get weather)
            'LAR': (33.8644, -117.9238), # SoFi Stadium
            'SF': (37.4032, -121.9698),  # Levi's Stadium
            'SEA': (47.5952, -122.3316), # Lumen Field
            'TEN': (36.1665, -86.7713),  # Nissan Stadium
            'JAC': (30.3240, -81.6374),  # TIAA Bank Field
            'CAR': (35.2258, -80.8528),  # Bank of America Stadium
            'MIA': (25.9580, -80.2389),  # Hard Rock Stadium
            'TB': (27.9759, -82.5033),   # Raymond James Stadium
        }

        # Indoor stadiums (for reference)
        self.indoor_stadiums = {
            'DAL', 'DET', 'HOU', 'IND', 'LV', 'MIN', 'NO', 'ATL', 'ARI'
        }

    def get_game_weather(self, home_team: str, game_datetime: datetime) -> Dict:
        """Get weather for a specific game"""
        if home_team in self.indoor_stadiums:
            return self._get_indoor_conditions()

        if home_team not in self.stadium_coords:
            logger.warning(f"No coordinates for team {home_team}")
            return {}

        lat, lon = self.stadium_coords[home_team]

        # Try multiple sources for best coverage
        weather_data = {}

        # 1. OpenWeatherMap (requires API key)
        if self.openweather_key:
            owm_data = self._get_openweather_data(lat, lon, game_datetime)
            weather_data.update(owm_data)

        # 2. National Weather Service (free, no key needed)
        nws_data = self._get_nws_data(lat, lon, game_datetime)
        weather_data.update(nws_data)

        # 3. WeatherAPI (free tier)
        weather_api_data = self._get_weatherapi_data(lat, lon, game_datetime)
        weather_data.update(weather_api_data)

        return weather_data

    def _get_indoor_conditions(self) -> Dict:
        """Standard indoor stadium conditions"""
        return {
            'temperature': 72,
            'wind_speed': 0,
            'precipitation': 0,
            'humidity': 45,
            'conditions': 'Indoor',
            'visibility': 10,
            'is_dome': True,
            'source': 'indoor_stadium'
        }

    def _get_openweather_data(self, lat: float, lon: float, game_time: datetime) -> Dict:
        """Get weather from OpenWeatherMap API"""
        if not self.openweather_key:
            return {}

        base_url = "https://api.openweathermap.org/data/2.5"

        # Determine if we need current, forecast, or historical data
        now = datetime.now()
        hours_diff = (game_time - now).total_seconds() / 3600

        try:
            if hours_diff <= 0:  # Past game - historical data
                # Historical data (limited on free tier)
                dt = int(game_time.timestamp())
                url = f"{base_url}/onecall/timemachine"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'dt': dt,
                    'appid': self.openweather_key,
                    'units': 'imperial'
                }

            elif hours_diff <= 2:  # Current weather
                url = f"{base_url}/weather"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.openweather_key,
                    'units': 'imperial'
                }

            else:  # Future game - forecast
                url = f"{base_url}/forecast"
                params = {
                    'lat': lat,
                    'lon': lon,
                    'appid': self.openweather_key,
                    'units': 'imperial'
                }

            time.sleep(0.1)  # Rate limiting
            response = self.session.get(url, params=params)
            response.raise_for_status()

            data = response.json()

            if 'forecast' in url:
                # Find closest forecast to game time
                closest_forecast = self._find_closest_forecast(data['list'], game_time)
                weather_info = closest_forecast
            elif 'timemachine' in url:
                weather_info = data['current']
            else:
                weather_info = data

            return {
                'owm_temperature': weather_info['main']['temp'],
                'owm_wind_speed': weather_info.get('wind', {}).get('speed', 0),
                'owm_wind_direction': weather_info.get('wind', {}).get('deg', 0),
                'owm_humidity': weather_info['main']['humidity'],
                'owm_pressure': weather_info['main']['pressure'],
                'owm_visibility': weather_info.get('visibility', 10000) / 1000,  # Convert to miles
                'owm_conditions': weather_info['weather'][0]['description'],
                'owm_precipitation': self._extract_precipitation(weather_info),
                'owm_source': 'openweathermap'
            }

        except requests.RequestException as e:
            logger.error(f"OpenWeatherMap API error: {e}")
            return {}

    def _get_nws_data(self, lat: float, lon: float, game_time: datetime) -> Dict:
        """Get weather from National Weather Service (free, no API key)"""
        try:
            # Get grid coordinates for location
            points_url = f"https://api.weather.gov/points/{lat},{lon}"
            time.sleep(0.5)  # Be respectful to government API
            response = self.session.get(points_url)
            response.raise_for_status()

            points_data = response.json()
            forecast_url = points_data['properties']['forecast']

            # Get forecast
            time.sleep(0.5)
            forecast_response = self.session.get(forecast_url)
            forecast_response.raise_for_status()

            forecast_data = forecast_response.json()

            # Find relevant forecast period
            relevant_period = self._find_nws_period(forecast_data['properties']['periods'], game_time)

            if relevant_period:
                return {
                    'nws_temperature': relevant_period['temperature'],
                    'nws_wind_speed': self._parse_nws_wind(relevant_period.get('windSpeed', '0 mph')),
                    'nws_wind_direction': relevant_period.get('windDirection', 'N'),
                    'nws_conditions': relevant_period['shortForecast'],
                    'nws_detailed_forecast': relevant_period['detailedForecast'],
                    'nws_precipitation_prob': relevant_period.get('probabilityOfPrecipitation', {}).get('value', 0),
                    'nws_source': 'national_weather_service'
                }

        except requests.RequestException as e:
            logger.warning(f"NWS API error: {e}")

        return {}

    def _get_weatherapi_data(self, lat: float, lon: float, game_time: datetime) -> Dict:
        """Get weather from WeatherAPI.com (free tier: 1M calls/month)"""
        # Note: Would need API key for WeatherAPI
        # Placeholder for structure - can be implemented with free key
        return {
            'weatherapi_note': 'Available with free API key from weatherapi.com',
            'weatherapi_calls_per_month': 1000000
        }

    def _find_closest_forecast(self, forecasts: List[Dict], target_time: datetime) -> Dict:
        """Find forecast closest to target time"""
        closest_forecast = forecasts[0]
        min_diff = abs(datetime.fromtimestamp(forecasts[0]['dt']) - target_time)

        for forecast in forecasts[1:]:
            forecast_time = datetime.fromtimestamp(forecast['dt'])
            diff = abs(forecast_time - target_time)
            if diff < min_diff:
                min_diff = diff
                closest_forecast = forecast

        return closest_forecast

    def _find_nws_period(self, periods: List[Dict], target_time: datetime) -> Optional[Dict]:
        """Find NWS forecast period for target time"""
        # Make target_time timezone aware if it's not already
        if target_time.tzinfo is None:
            from datetime import timezone
            target_time = target_time.replace(tzinfo=timezone.utc)

        for period in periods:
            start_time = datetime.fromisoformat(period['startTime'].replace('Z', '+00:00'))
            end_time = datetime.fromisoformat(period['endTime'].replace('Z', '+00:00'))

            if start_time <= target_time <= end_time:
                return period

        # If no exact match, return the first period (current conditions)
        return periods[0] if periods else None

    def _parse_nws_wind(self, wind_str: str) -> float:
        """Parse NWS wind speed string like '10 mph' to float"""
        try:
            return float(wind_str.split()[0])
        except (IndexError, ValueError):
            return 0

    def _extract_precipitation(self, weather_data: Dict) -> float:
        """Extract precipitation from weather data"""
        rain = weather_data.get('rain', {})
        snow = weather_data.get('snow', {})

        # Sum rain and snow (1h or 3h data)
        precip = 0
        if rain:
            precip += rain.get('1h', rain.get('3h', 0))
        if snow:
            precip += snow.get('1h', snow.get('3h', 0))

        return precip

    def get_bulk_game_weather(self, games_df: pd.DataFrame) -> pd.DataFrame:
        """Get weather for multiple games efficiently"""
        weather_data = []

        for _, game in games_df.iterrows():
            try:
                game_time = pd.to_datetime(game['game_date'])
                weather = self.get_game_weather(game['home_team'], game_time)

                weather['game_id'] = game.get('game_id', '')
                weather['home_team'] = game['home_team']
                weather['game_date'] = game['game_date']

                weather_data.append(weather)

            except Exception as e:
                logger.error(f"Failed to get weather for game {game.get('game_id', 'unknown')}: {e}")
                # Add empty weather record
                weather_data.append({
                    'game_id': game.get('game_id', ''),
                    'home_team': game['home_team'],
                    'game_date': game['game_date'],
                    'error': str(e)
                })

        return pd.DataFrame(weather_data)

    def get_historical_weather_trends(self, team: str, season: int) -> Dict:
        """Get weather trends for a team's home games"""
        if team in self.indoor_stadiums:
            return {'team': team, 'stadium_type': 'indoor', 'weather_impact': 'minimal'}

        if team not in self.stadium_coords:
            return {'team': team, 'error': 'No stadium coordinates available'}

        lat, lon = self.stadium_coords[team]

        # This would implement historical weather analysis
        # For now, return structure showing capabilities
        return {
            'team': team,
            'stadium_type': 'outdoor',
            'latitude': lat,
            'longitude': lon,
            'weather_impact': 'significant',
            'seasonal_trends': {
                'avg_september_temp': 70,
                'avg_october_temp': 55,
                'avg_november_temp': 45,
                'avg_december_temp': 35,
                'precipitation_games_pct': 25,
                'high_wind_games_pct': 15
            }
        }


# Integration function
def integrate_weather_data(pipeline_instance, openweather_key: str = None):
    """Add weather data to existing pipeline"""
    weather_client = WeatherClient(openweather_key)

    def _get_enhanced_weather_data(season: int, week: int) -> pd.DataFrame:
        """Enhanced weather data with multiple sources"""
        # Get games for the week first
        schedule = pipeline_instance._get_game_data(season, week)

        if schedule.empty:
            return pd.DataFrame()

        return weather_client.get_bulk_game_weather(schedule)

    # Add to pipeline
    pipeline_instance._get_enhanced_weather_data = _get_enhanced_weather_data
    pipeline_instance.weather_client = weather_client

    logger.info("Weather integration added to data pipeline")
    return pipeline_instance