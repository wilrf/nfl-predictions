"""
Data Pipeline Module
Handles all data collection, caching, and validation with proper error handling
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import nfl_data_py as nfl
import logging
import hashlib
import json
from dataclasses import dataclass
import redis
import pickle

# Import new data source clients
try:
    from data_sources.espn_client import ESPNClient, integrate_espn_data
    from data_sources.weather_client import WeatherClient, integrate_weather_data
    from data_sources.nfl_official_client import NFLOfficialClient, integrate_nfl_official_data
except ImportError as e:
    logging.warning(f"Some data source clients not available: {e}")
    ESPNClient = None
    WeatherClient = None
    NFLOfficialClient = None

logger = logging.getLogger(__name__)


@dataclass
class DataQuality:
    """Data quality metrics for validation"""
    completeness: float
    consistency: float
    timeliness: float
    accuracy: float
    issues: List[str]
    
    @property
    def overall_quality(self) -> float:
        return np.mean([self.completeness, self.consistency, self.timeliness, self.accuracy])
    
    def is_acceptable(self, threshold: float = 0.8) -> bool:
        return self.overall_quality >= threshold


class DataValidator:
    """Validates data quality and consistency"""
    
    def __init__(self):
        self.validation_rules = {
            'game_data': self._validate_game_data,
            'odds_data': self._validate_odds_data,
            'weather_data': self._validate_weather_data
        }
        
    def validate(self, data: pd.DataFrame, data_type: str) -> DataQuality:
        """Run validation for specific data type"""
        # Handle new data source types
        base_type = data_type.replace('_data', '')

        # Map new sources to existing validation rules
        validation_mapping = {
            'espn_scores': 'game_data',
            'espn_standings': 'game_data',
            'nfl_scores': 'game_data',
            'nfl_injury_report': 'game_data',
            'enhanced_weather': 'weather_data',
            'enhanced_injuries': 'game_data'
        }

        # Use mapped validation or default to game_data for unknown types
        validation_type = validation_mapping.get(base_type, base_type)
        if validation_type not in self.validation_rules:
            validation_type = 'game_data'  # Safe fallback

        return self.validation_rules[validation_type](data)
    
    def _validate_game_data(self, df: pd.DataFrame) -> DataQuality:
        """Validate game statistics data"""
        issues = []
        
        # Check completeness
        required_cols = ['home_team', 'away_team', 'home_score', 'away_score', 'game_date']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            issues.append(f"Missing columns: {missing_cols}")
        
        completeness = 1 - (df.isnull().sum().sum() / (len(df) * len(df.columns)))
        
        # Check consistency
        invalid_scores = df[(df['home_score'] < 0) | (df['away_score'] < 0)]
        if not invalid_scores.empty:
            issues.append(f"Invalid scores in {len(invalid_scores)} games")
        
        consistency = 1 - (len(invalid_scores) / len(df)) if len(df) > 0 else 1
        
        # Check timeliness
        if 'game_date' in df.columns:
            latest_date = pd.to_datetime(df['game_date']).max()
            days_old = (datetime.now() - latest_date).days
            timeliness = max(0, 1 - (days_old / 7))  # Decay over a week
        else:
            timeliness = 0
            issues.append("No date information")
        
        # Check accuracy (simplified - would compare to known sources)
        accuracy = 0.95  # Placeholder - implement actual checks
        
        return DataQuality(completeness, consistency, timeliness, accuracy, issues)
    
    def _validate_odds_data(self, df: pd.DataFrame) -> DataQuality:
        """Validate odds data"""
        issues = []
        
        # Check for required fields
        required = ['game_id', 'book', 'spread', 'total', 'moneyline']
        completeness = sum(col in df.columns for col in required) / len(required)
        
        # Check odds are in reasonable range
        if 'spread' in df.columns:
            invalid_spreads = df[(df['spread'] < -50) | (df['spread'] > 50)]
            if not invalid_spreads.empty:
                issues.append(f"Unrealistic spreads: {len(invalid_spreads)} entries")
        
        # Check for stale data
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            stale_data = df[df['timestamp'] < datetime.now() - timedelta(hours=24)]
            if not stale_data.empty:
                issues.append(f"Stale odds: {len(stale_data)} entries over 24 hours old")
            
            timeliness = 1 - (len(stale_data) / len(df)) if len(df) > 0 else 1
        else:
            timeliness = 0.5
        
        consistency = 0.9 if len(issues) < 2 else 0.7
        accuracy = 0.95  # Placeholder
        
        return DataQuality(completeness, consistency, timeliness, accuracy, issues)
    
    def _validate_weather_data(self, df: pd.DataFrame) -> DataQuality:
        """Validate weather data"""
        issues = []
        
        required = ['game_id', 'temperature', 'wind_speed', 'precipitation']
        completeness = sum(col in df.columns for col in required) / len(required)
        
        # Check reasonable ranges
        if 'temperature' in df.columns:
            invalid_temp = df[(df['temperature'] < -30) | (df['temperature'] > 120)]
            if not invalid_temp.empty:
                issues.append(f"Invalid temperatures: {len(invalid_temp)} entries")
        
        if 'wind_speed' in df.columns:
            invalid_wind = df[df['wind_speed'] < 0]
            if not invalid_wind.empty:
                issues.append(f"Negative wind speeds: {len(invalid_wind)} entries")
        
        consistency = 1 - (len(issues) / 10)  # Scale issues to consistency score
        timeliness = 0.9  # Weather forecasts are usually current
        accuracy = 0.85  # Weather is inherently uncertain
        
        return DataQuality(completeness, consistency, timeliness, accuracy, issues)


class CachedDataLoader:
    """Handles data loading with intelligent caching"""
    
    def __init__(self, redis_host='localhost', redis_port=6379):
        self.redis_client = redis.StrictRedis(host=redis_host, port=redis_port)
        self.cache_ttl = {
            'historical': 86400 * 7,  # 1 week for historical data
            'current': 3600,  # 1 hour for current data
            'odds': 300,  # 5 minutes for odds
            'weather': 1800  # 30 minutes for weather
        }
    
    def get_or_fetch(self, key: str, fetch_func, ttl_type: str = 'current', *args, **kwargs):
        """Get from cache or fetch and cache"""
        # Try cache first
        cached_data = self.redis_client.get(key)
        if cached_data:
            logger.info(f"Cache hit: {key}")
            return pickle.loads(cached_data)
        
        logger.info(f"Cache miss: {key}, fetching...")
        
        # Fetch data
        data = fetch_func(*args, **kwargs)
        
        # Cache for future use
        ttl = self.cache_ttl.get(ttl_type, 3600)
        self.redis_client.setex(key, ttl, pickle.dumps(data))
        
        return data
    
    def invalidate_pattern(self, pattern: str):
        """Invalidate cache entries matching pattern"""
        for key in self.redis_client.scan_iter(match=pattern):
            self.redis_client.delete(key)
            logger.info(f"Invalidated cache: {key}")


class NFLDataPipeline:
    """Main data pipeline orchestrator"""

    def __init__(self, config: Dict):
        self.config = config
        self.cache = CachedDataLoader()
        self.validator = DataValidator()
        self.executor = ThreadPoolExecutor(max_workers=10)

        # Initialize enhanced data clients
        self.espn_client = ESPNClient() if ESPNClient else None
        self.weather_client = WeatherClient(config.get('openweather_api_key')) if WeatherClient else None
        self.nfl_client = NFLOfficialClient() if NFLOfficialClient else None

        # Integration flags
        self.enhanced_sources_available = {
            'espn': self.espn_client is not None,
            'weather': self.weather_client is not None,
            'nfl_official': self.nfl_client is not None
        }

        logger.info(f"Enhanced data sources: {self.enhanced_sources_available}")
        
    def get_weekly_data(self, week: int, season: int = None) -> Dict[str, pd.DataFrame]:
        """Get all data for a specific week"""
        if season is None:
            season = datetime.now().year
        
        logger.info(f"Fetching data for Season {season} Week {week}")
        
        # Parallel fetch all data types (enhanced with new sources)
        futures = {
            'games': self.executor.submit(self._get_game_data, season, week),
            'odds': self.executor.submit(self._get_odds_data, season, week),
            'weather': self.executor.submit(self._get_enhanced_weather_data, season, week),
            'injuries': self.executor.submit(self._get_enhanced_injury_data, season, week),
            'team_stats': self.executor.submit(self._get_team_stats, season, week)
        }

        # Add enhanced data sources if available
        if self.enhanced_sources_available['espn']:
            futures['espn_scores'] = self.executor.submit(self._get_espn_scores, season, week)
            futures['espn_standings'] = self.executor.submit(self._get_espn_standings, season)

        if self.enhanced_sources_available['nfl_official']:
            futures['nfl_scores'] = self.executor.submit(self._get_nfl_live_scores, season, week)
            futures['nfl_injury_report'] = self.executor.submit(self._get_nfl_injury_report, season, week)
        
        # Collect results
        data = {}
        for key, future in futures.items():
            try:
                result = future.result(timeout=30)
                
                # Validate data
                try:
                    quality = self.validator.validate(result, key)
                    if not quality.is_acceptable():
                        logger.warning(f"Data quality issues for {key}: {quality.issues}")
                except Exception as validation_error:
                    logger.warning(f"Validation failed for {key}: {validation_error}")
                    # Continue without validation rather than failing completely
                
                data[key] = result
            except Exception as e:
                logger.error(f"Failed to fetch {key}: {e}")
                data[key] = pd.DataFrame()  # Empty fallback
        
        return data
    
    def _get_game_data(self, season: int, week: int) -> pd.DataFrame:
        """Fetch game and play-by-play data"""
        cache_key = f"game_data_{season}_{week}"
        
        def fetch():
            # Get schedule
            schedule = nfl.import_schedules([season])
            week_games = schedule[schedule['week'] == week]
            
            # Get play-by-play if games are complete
            if not week_games.empty and week_games.iloc[0]['game_type'] == 'REG':
                # Fix nfl_data_py API call - no weeks parameter in import_pbp_data
                pbp = nfl.import_pbp_data([season])
                
                # Aggregate to game level
                game_stats = self._aggregate_pbp_to_games(pbp)
                
                # Merge with schedule
                result = week_games.merge(game_stats, on='game_id', how='left')
            else:
                result = week_games
            
            return result
        
        return self.cache.get_or_fetch(cache_key, fetch, 'historical')
    
    def _aggregate_pbp_to_games(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Aggregate play-by-play data to game level stats"""
        if pbp.empty:
            return pd.DataFrame()
        
        game_stats = pbp.groupby(['game_id', 'posteam']).agg({
            'yards_gained': ['mean', 'sum'],
            'epa': ['mean', 'sum'],
            'success': 'mean',
            'play_type': lambda x: (x == 'pass').mean(),
            'first_down': 'sum',
            'third_down_converted': lambda x: x.sum() if 'third_down_converted' in pbp else 0,
            'third_down_failed': lambda x: x.sum() if 'third_down_failed' in pbp else 0
        }).reset_index()
        
        # Flatten column names
        game_stats.columns = ['_'.join(col).strip() for col in game_stats.columns.values]
        
        return game_stats
    
    def _get_odds_data(self, season: int, week: int) -> pd.DataFrame:
        """Fetch odds from multiple books"""
        cache_key = f"odds_{season}_{week}"
        
        def fetch():
            odds_list = []
            
            # Fetch from each configured book
            for book in self.config['odds_sources']['sharp_books'] + self.config['odds_sources']['soft_books']:
                try:
                    book_odds = self._fetch_book_odds(book, season, week)
                    odds_list.append(book_odds)
                except Exception as e:
                    logger.error(f"Failed to fetch odds from {book}: {e}")
            
            if odds_list:
                return pd.concat(odds_list, ignore_index=True)
            return pd.DataFrame()
        
        return self.cache.get_or_fetch(cache_key, fetch, 'odds')
    
    def _fetch_book_odds(self, book: str, season: int, week: int) -> pd.DataFrame:
        """Fetch odds from specific sportsbook"""
        # This would implement actual API calls
        # For now, return mock data structure
        return pd.DataFrame({
            'game_id': [],
            'book': book,
            'spread': [],
            'total': [],
            'moneyline_home': [],
            'moneyline_away': [],
            'timestamp': datetime.now()
        })
    
    def _get_weather_data(self, season: int, week: int) -> pd.DataFrame:
        """Fetch weather data for outdoor games"""
        cache_key = f"weather_{season}_{week}"
        
        def fetch():
            # Get schedule to identify outdoor games
            schedule = nfl.import_schedules([season])
            week_games = schedule[schedule['week'] == week]
            
            # Filter to outdoor stadiums (simplified - would use actual stadium database)
            outdoor_teams = ['GB', 'CHI', 'BUF', 'NE', 'NYJ', 'CLE', 'CIN', 'PIT', 'BAL', 'WAS', 'PHI']
            outdoor_games = week_games[week_games['home_team'].isin(outdoor_teams)]
            
            weather_data = []
            for _, game in outdoor_games.iterrows():
                # Fetch weather for game location/time
                # This would use actual weather API
                weather = {
                    'game_id': game['game_id'],
                    'temperature': np.random.normal(60, 15),
                    'wind_speed': np.random.exponential(5),
                    'precipitation': np.random.choice([0, 0, 0, 0.1, 0.3], p=[0.7, 0.1, 0.1, 0.05, 0.05]),
                    'humidity': np.random.uniform(30, 80)
                }
                weather_data.append(weather)
            
            return pd.DataFrame(weather_data)
        
        return self.cache.get_or_fetch(cache_key, fetch, 'weather')

    def _get_enhanced_weather_data(self, season: int, week: int) -> pd.DataFrame:
        """Enhanced weather data using multiple sources"""
        if not self.enhanced_sources_available['weather']:
            # Fallback to original weather method
            return self._get_weather_data(season, week)

        cache_key = f"enhanced_weather_{season}_{week}"

        def fetch():
            # Get games for the week
            schedule = self._get_game_data(season, week)
            if schedule.empty:
                return pd.DataFrame()

            # Use enhanced weather client
            return self.weather_client.get_bulk_game_weather(schedule)

        return self.cache.get_or_fetch(cache_key, fetch, 'weather')

    def _get_enhanced_injury_data(self, season: int, week: int) -> pd.DataFrame:
        """Enhanced injury data combining multiple sources"""
        cache_key = f"enhanced_injuries_{season}_{week}"

        def fetch():
            # Start with original injury data
            base_injuries = self._get_injury_data(season, week)

            # Add NFL official injury report if available
            if self.enhanced_sources_available['nfl_official']:
                try:
                    official_injuries = self.nfl_client.get_injury_report(week, season)
                    if not official_injuries.empty:
                        # Merge with base injuries, preferring official data
                        combined = self._merge_injury_data(base_injuries, official_injuries)
                        return combined
                except Exception as e:
                    logger.warning(f"Failed to get official injury data: {e}")

            return base_injuries

        return self.cache.get_or_fetch(cache_key, fetch, 'current')

    def _merge_injury_data(self, base_df: pd.DataFrame, official_df: pd.DataFrame) -> pd.DataFrame:
        """Merge injury data from multiple sources, preferring official"""
        if base_df.empty:
            return official_df
        if official_df.empty:
            return base_df

        # Simple merge - in practice would need more sophisticated logic
        # to handle conflicts and player matching
        return pd.concat([base_df, official_df], ignore_index=True).drop_duplicates(
            subset=['player', 'team'], keep='last'
        )

    # Enhanced data source methods
    def _get_espn_scores(self, season: int, week: int) -> pd.DataFrame:
        """Get ESPN live scores"""
        if not self.enhanced_sources_available['espn']:
            return pd.DataFrame()

        cache_key = f"espn_scores_{season}_{week}"

        def fetch():
            return self.espn_client.get_live_scores(season, week)

        return self.cache.get_or_fetch(cache_key, fetch, 'current')

    def _get_espn_standings(self, season: int) -> pd.DataFrame:
        """Get ESPN standings"""
        if not self.enhanced_sources_available['espn']:
            return pd.DataFrame()

        cache_key = f"espn_standings_{season}"

        def fetch():
            return self.espn_client.get_standings(season)

        return self.cache.get_or_fetch(cache_key, fetch, 'historical')

    def _get_nfl_live_scores(self, season: int, week: int) -> pd.DataFrame:
        """Get NFL.com official live scores"""
        if not self.enhanced_sources_available['nfl_official']:
            return pd.DataFrame()

        cache_key = f"nfl_scores_{season}_{week}"

        def fetch():
            return self.nfl_client.get_live_scores(season, week)

        return self.cache.get_or_fetch(cache_key, fetch, 'current')

    def _get_nfl_injury_report(self, season: int, week: int) -> pd.DataFrame:
        """Get NFL.com official injury report"""
        if not self.enhanced_sources_available['nfl_official']:
            return pd.DataFrame()

        cache_key = f"nfl_injuries_{season}_{week}"

        def fetch():
            return self.nfl_client.get_injury_report(week, season)

        return self.cache.get_or_fetch(cache_key, fetch, 'current')

    def get_data_source_status(self) -> Dict[str, Any]:
        """Get status of all data sources"""
        status = {
            'nfl_data_py': 'available',
            'enhanced_sources': self.enhanced_sources_available,
            'cache_status': 'connected' if self.cache.redis_client else 'unavailable',
            'last_updated': datetime.now().isoformat()
        }

        # Test each enhanced source
        for source, available in self.enhanced_sources_available.items():
            if available:
                try:
                    if source == 'espn' and self.espn_client:
                        # Quick test call
                        test_data = self.espn_client.get_standings()
                        status[f'{source}_test'] = 'success' if not test_data.empty else 'no_data'
                    elif source == 'weather' and self.weather_client:
                        # Test weather for known outdoor stadium
                        test_weather = self.weather_client.get_game_weather('GB', datetime.now())
                        status[f'{source}_test'] = 'success' if test_weather else 'no_data'
                    elif source == 'nfl_official' and self.nfl_client:
                        # Test NFL scores
                        test_scores = self.nfl_client.get_live_scores()
                        status[f'{source}_test'] = 'success' if not test_scores.empty else 'no_data'
                except Exception as e:
                    status[f'{source}_test'] = f'error: {str(e)}'

        return status
    
    def _get_injury_data(self, season: int, week: int) -> pd.DataFrame:
        """Fetch injury reports with event-time tracking"""
        cache_key = f"injuries_{season}_{week}_{datetime.now().strftime('%Y%m%d_%H')}"
        
        def fetch():
            injuries = []
            
            # Track multiple injury sources with timestamps
            sources = ['official_report', 'beat_writers', 'practice_reports']
            
            for source in sources:
                source_data = self._fetch_injury_source(source, season, week)
                injuries.append(source_data)
            
            if injuries:
                df = pd.concat(injuries, ignore_index=True)
                
                # Add event-time tracking
                df['event_timestamp'] = pd.to_datetime(df['report_time'])
                df['hours_before_game'] = (df['game_time'] - df['event_timestamp']).dt.total_seconds() / 3600
                
                # Calculate player impact scores
                df['player_impact'] = self._calculate_player_impact(df)
                
                # Get latest status per player (most recent update)
                df = df.sort_values('event_timestamp').groupby('player').last().reset_index()
                
                return df
            
            return pd.DataFrame({
                'player': [],
                'team': [],
                'position': [],
                'status': [],  # 'out', 'doubtful', 'questionable', 'probable'
                'injury': [],
                'player_impact': [],  # Player-specific impact
                'snap_pct_last3': [],  # Recent usage
                'target_share_last3': [],  # Recent volume
                'event_timestamp': [],
                'hours_before_game': []
            })
        
        return self.cache.get_or_fetch(cache_key, fetch, 'current')
    
    def _fetch_injury_source(self, source: str, season: int, week: int) -> pd.DataFrame:
        """Fetch injuries from specific source"""
        # Implementation would vary by source
        # This is structured to show the concept
        return pd.DataFrame()
    
    def _calculate_player_impact(self, df: pd.DataFrame) -> pd.Series:
        """Calculate player-specific impact scores"""
        impact_scores = []
        
        for _, player in df.iterrows():
            # Base impact on position value and role
            position_weights = {
                'QB': 10, 'RB': 7, 'WR': 6, 'TE': 5,
                'OL': 4, 'DL': 4, 'LB': 5, 'DB': 4
            }
            
            status_multipliers = {
                'out': 1.0,
                'doubtful': 0.75,
                'questionable': 0.5,
                'probable': 0.25
            }
            
            base_impact = position_weights.get(player['position'], 3)
            status_mult = status_multipliers.get(player['status'], 0.5)
            
            # Adjust for player's recent usage
            usage_mult = player.get('snap_pct_last3', 50) / 100
            
            impact = base_impact * status_mult * usage_mult
            impact_scores.append(impact)
        
        return pd.Series(impact_scores)
    
    def _get_team_stats(self, season: int, week: int) -> pd.DataFrame:
        """Get aggregated team statistics up to current week"""
        cache_key = f"team_stats_{season}_{week}"
        
        def fetch():
            # Get all games up to this week
            try:
                # Get all season data and filter by week
                all_pbp = nfl.import_pbp_data([season])
                pbp_data = [all_pbp[all_pbp['week'] < week]] if not all_pbp.empty else []
            except Exception as e:
                logger.error(f"Failed to load PBP data for season {season}: {e}")
                pbp_data = []
            
            if not pbp_data:
                return pd.DataFrame()
            
            all_pbp = pd.concat(pbp_data, ignore_index=True)
            
            # Calculate team-level statistics
            team_stats = self._calculate_team_metrics(all_pbp)
            
            return team_stats
        
        return self.cache.get_or_fetch(cache_key, fetch, 'historical')
    
    def _calculate_team_metrics(self, pbp: pd.DataFrame) -> pd.DataFrame:
        """Calculate advanced team metrics from play-by-play"""
        if pbp.empty:
            return pd.DataFrame()
        
        metrics = []
        
        for team in pbp['posteam'].dropna().unique():
            team_plays = pbp[pbp['posteam'] == team]
            opp_plays = pbp[pbp['defteam'] == team]
            
            team_metrics = {
                'team': team,
                # Offensive metrics
                'off_epa_play': team_plays['epa'].mean(),
                'off_success_rate': team_plays['success'].mean() if 'success' in team_plays else 0,
                'off_explosive_rate': (team_plays['yards_gained'] >= 20).mean(),
                'off_yards_play': team_plays['yards_gained'].mean(),
                # Defensive metrics  
                'def_epa_play': opp_plays['epa'].mean(),
                'def_success_rate': opp_plays['success'].mean() if 'success' in opp_plays else 0,
                'def_explosive_allowed': (opp_plays['yards_gained'] >= 20).mean(),
                'def_yards_play': opp_plays['yards_gained'].mean(),
                # Special situations
                'redzone_td_pct': self._calculate_redzone_efficiency(team_plays),
                'third_down_pct': self._calculate_third_down_pct(team_plays)
            }
            
            metrics.append(team_metrics)
        
        return pd.DataFrame(metrics)
    
    def _calculate_redzone_efficiency(self, plays: pd.DataFrame) -> float:
        """Calculate red zone touchdown percentage"""
        if 'yardline_100' not in plays.columns:
            return 0.5
        
        rz_plays = plays[plays['yardline_100'] <= 20]
        if rz_plays.empty:
            return 0.5
        
        # Find scoring plays
        td_plays = rz_plays[rz_plays['touchdown'] == 1] if 'touchdown' in rz_plays else pd.DataFrame()
        
        # Calculate drives (simplified)
        return len(td_plays) / max(len(rz_plays) / 10, 1)  # Rough approximation
    
    def _calculate_third_down_pct(self, plays: pd.DataFrame) -> float:
        """Calculate third down conversion percentage"""
        if 'down' not in plays.columns:
            return 0.33
        
        third_downs = plays[plays['down'] == 3]
        if third_downs.empty:
            return 0.33
        
        conversions = third_downs[third_downs['first_down'] == 1] if 'first_down' in third_downs else pd.DataFrame()
        
        return len(conversions) / len(third_downs)
    
    async def fetch_realtime_odds(self, game_ids: List[str]) -> pd.DataFrame:
        """Asynchronously fetch real-time odds for multiple games"""
        async with aiohttp.ClientSession() as session:
            tasks = []
            for game_id in game_ids:
                for book in self.config['odds_sources']['sharp_books']:
                    task = self._async_fetch_book_odds(session, book, game_id)
                    tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter out exceptions and combine results
            valid_results = [r for r in results if not isinstance(r, Exception)]
            
            if valid_results:
                return pd.concat(valid_results, ignore_index=True)
            
            return pd.DataFrame()
    
    async def _async_fetch_book_odds(self, session: aiohttp.ClientSession, 
                                     book: str, game_id: str) -> pd.DataFrame:
        """Async fetch odds from specific book"""
        # This would implement actual async API call
        # Placeholder for structure
        url = f"https://api.{book}.com/odds/{game_id}"
        headers = {'API-Key': self.config['odds_sources']['api_keys'].get(book, '')}
        
        try:
            async with session.get(url, headers=headers) as response:
                if response.status == 200:
                    data = await response.json()
                    return pd.DataFrame(data)
        except Exception as e:
            logger.error(f"Failed to fetch {book} odds for {game_id}: {e}")
        
        return pd.DataFrame()
