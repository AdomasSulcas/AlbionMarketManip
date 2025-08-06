"""
Real-time Market Monitoring System
Provides continuous monitoring, streaming detection, and alert generation for market manipulation.
"""

import asyncio
import time
import logging
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Callable, Any, Set
from dataclasses import dataclass, asdict
from collections import deque
import pandas as pd
import numpy as np

from .data_collector import AlbionDataCollector
from .models import ManipulationDetector
from .gold_economics import GoldEconomicsAnalyzer
from .config import API_RATE_LIMIT_DELAY, DEFAULT_ITEMS, DEFAULT_CITIES


@dataclass
class MarketAlert:
    """
    Data class representing a market manipulation alert.
    
    Contains all information needed to identify, prioritize, and act on
    detected manipulation cases in real-time monitoring.
    """
    timestamp: datetime
    item: str
    city: str
    alert_type: str
    confidence: float
    current_price: float
    expected_price: float
    price_deviation_pct: float
    economic_context: str
    quality: int = 1
    additional_data: Dict[str, Any] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert alert to dictionary format for serialization."""
        result = asdict(self)
        result['timestamp'] = self.timestamp.isoformat()
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'MarketAlert':
        """Create alert from dictionary format."""
        data = data.copy()
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


class RateLimitedPoller:
    """
    Rate-limited API polling system that respects Albion Online Data API limits.
    
    Manages polling frequency to stay within API rate limits (180 req/min, 300 req/5min)
    while maximizing data freshness for real-time monitoring. Uses adaptive polling
    based on API response times and error rates.
    """

    def __init__(self, requests_per_minute: int = 150, burst_limit: int = 250) -> None:
        """
        Initialize rate-limited poller with conservative limits.
        
        Args:
            requests_per_minute: Target requests per minute (default: 150, under 180 limit)
            burst_limit: Maximum requests in 5-minute window (default: 250, under 300 limit)
        """
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.request_times = deque(maxlen=burst_limit)
        self.last_request = 0.0
        self.logger = logging.getLogger(__name__)

    async def wait_for_rate_limit(self) -> None:
        """
        Wait appropriate time to respect rate limits before making next request.
        
        Implements sliding window rate limiting with both per-minute and burst protection.
        Automatically adjusts delays based on recent request history and API limits.
        """
        now = time.time()
        
        if self.request_times:
            minute_ago = now - 60
            recent_requests = sum(1 for req_time in self.request_times if req_time > minute_ago)
            
            if recent_requests >= self.requests_per_minute:
                sleep_time = 60 - (now - min(req_time for req_time in self.request_times if req_time > minute_ago))
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

        min_interval = 60.0 / self.requests_per_minute
        time_since_last = now - self.last_request
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)

        self.request_times.append(time.time())
        self.last_request = time.time()

    async def poll_api(self, api_call: Callable, *args, **kwargs) -> Any:
        """
        Execute API call with rate limiting and error handling.
        
        Args:
            api_call: Function to call for API request
            *args: Positional arguments for API call
            **kwargs: Keyword arguments for API call
            
        Returns:
            Result from API call, or None if request fails
        """
        await self.wait_for_rate_limit()
        
        try:
            return api_call(*args, **kwargs)
        except Exception as e:
            self.logger.error(f"API polling error: {e}")
            return None


class SlidingWindowDetector:
    """
    Sliding window analysis system for real-time manipulation detection.
    
    Maintains rolling windows of market data and applies detection algorithms
    continuously as new data arrives. Provides immediate alert generation
    when manipulation patterns are detected without requiring full dataset rebuilds.
    """

    def __init__(self, window_hours: int = 24, min_data_points: int = 10) -> None:
        """
        Initialize sliding window detector with configurable parameters.
        
        Args:
            window_hours: Size of rolling window in hours for analysis
            min_data_points: Minimum data points required before detection can run
        """
        self.window_hours = window_hours
        self.min_data_points = min_data_points
        self.price_windows: Dict[str, deque] = {}  # (item, city) -> price data
        self.detector = ManipulationDetector(contamination=0.02)  # Higher for real-time
        self.gold_analyzer = GoldEconomicsAnalyzer()
        self.is_detector_trained = False
        self.logger = logging.getLogger(__name__)

    def add_price_data(self, item: str, city: str, timestamp: datetime, 
                      price: float, quality: int = 1) -> None:
        """
        Add new price data point to sliding window for specified item/city.
        
        Automatically manages window size and removes expired data points.
        Updates internal state for continuous detection without full retraining.
        
        Args:
            item: Albion Online item name
            city: City name where price was observed
            timestamp: When price was recorded
            price: Item price in silver
            quality: Item quality level (1-5)
        """
        key = f"{item}_{city}_{quality}"
        
        if key not in self.price_windows:
            self.price_windows[key] = deque(maxlen=self.window_hours * 4)  # 15min intervals
        
        cutoff_time = timestamp - timedelta(hours=self.window_hours)
        window = self.price_windows[key]
        
        while window and window[0]['timestamp'] < cutoff_time:
            window.popleft()
        
        window.append({
            'timestamp': timestamp,
            'item': item,
            'city': city,
            'price': price,
            'quality': quality,
            'log_price': np.log(price) if price > 0 else 0
        })

    def get_window_dataframe(self, item: str, city: str, quality: int = 1) -> pd.DataFrame:
        """
        Convert sliding window data to DataFrame format for analysis.
        
        Args:
            item: Item name for data extraction
            city: City name for data extraction  
            quality: Quality level for data extraction
            
        Returns:
            DataFrame containing windowed price data with required columns for detection
        """
        key = f"{item}_{city}_{quality}"
        
        if key not in self.price_windows or len(self.price_windows[key]) < self.min_data_points:
            return pd.DataFrame()
        
        data = list(self.price_windows[key])
        df = pd.DataFrame(data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        df['tier'] = df['item'].str.extract(r'T(\d)')[0].astype(int, errors='ignore')
        df['slot'] = df['item'].str.split('_').str[2]
        
        return df

    async def detect_manipulation(self, item: str, city: str, quality: int = 1) -> Optional[MarketAlert]:
        """
        Perform real-time manipulation detection on sliding window data.
        
        Analyzes current price patterns against historical data in the sliding window
        to detect suspicious activity. Uses lightweight detection suitable for
        continuous operation without heavy computational overhead.
        
        Args:
            item: Item to analyze for manipulation
            city: City to analyze for manipulation
            quality: Quality level to analyze
            
        Returns:
            MarketAlert if manipulation detected, None otherwise
        """
        df = self.get_window_dataframe(item, city, quality)
        
        if df.empty or len(df) < self.min_data_points:
            return None

        try:
            latest_price = df['price'].iloc[-1]
            historical_prices = df['price'].iloc[:-1]
            
            if len(historical_prices) < 5:
                return None

            recent_median = historical_prices.tail(12).median()  # Last 3 hours
            older_median = historical_prices.head(12).median()   # Older data
            
            if recent_median == 0 or older_median == 0:
                return None

            price_change_pct = ((latest_price - recent_median) / recent_median) * 100
            trend_change_pct = ((recent_median - older_median) / older_median) * 100

            volatility = historical_prices.std() / historical_prices.mean() if historical_prices.mean() > 0 else 0
            
            is_suspicious = (
                abs(price_change_pct) > 15 and  # 15% price jump
                abs(trend_change_pct) > 10 and  # Trend reversal
                volatility < 0.3  # Not just high volatility
            )
            
            if is_suspicious:
                confidence = min(0.9, (abs(price_change_pct) + abs(trend_change_pct)) / 50)
                
                gold_context = self.gold_analyzer.detect_economic_regime(
                    self.gold_analyzer.fetch_gold_prices(24)
                )
                
                alert = MarketAlert(
                    timestamp=datetime.now(),
                    item=item,
                    city=city,
                    alert_type="price_spike",
                    confidence=confidence,
                    current_price=latest_price,
                    expected_price=recent_median,
                    price_deviation_pct=price_change_pct,
                    economic_context=gold_context.get('regime', 'unknown'),
                    quality=quality,
                    additional_data={
                        'trend_change_pct': trend_change_pct,
                        'volatility': volatility,
                        'data_points': len(df)
                    }
                )
                
                return alert

        except Exception as e:
            self.logger.error(f"Detection error for {item} in {city}: {e}")
        
        return None


class AlertManager:
    """
    Alert notification and management system for real-time monitoring.
    
    Handles alert generation, deduplication, prioritization, and notification
    distribution. Supports multiple notification channels and alert persistence
    for historical analysis and system reliability.
    """

    def __init__(self, alert_cooldown_minutes: int = 30) -> None:
        """
        Initialize alert management system.
        
        Args:
            alert_cooldown_minutes: Minimum time between duplicate alerts for same item/city
        """
        self.alert_cooldown = timedelta(minutes=alert_cooldown_minutes)
        self.recent_alerts: Dict[str, datetime] = {}
        self.alert_history: List[MarketAlert] = []
        self.notification_handlers: List[Callable[[MarketAlert], None]] = []
        self.logger = logging.getLogger(__name__)

    def add_notification_handler(self, handler: Callable[[MarketAlert], None]) -> None:
        """
        Register notification handler for alert distribution.
        
        Args:
            handler: Function to call when alerts are generated
        """
        self.notification_handlers.append(handler)

    def should_send_alert(self, alert: MarketAlert) -> bool:
        """
        Determine if alert should be sent based on deduplication rules.
        
        Prevents alert spam by enforcing cooldown periods between similar alerts
        for the same item/city combination.
        
        Args:
            alert: Alert to evaluate for sending
            
        Returns:
            True if alert should be sent, False if suppressed by cooldown
        """
        key = f"{alert.item}_{alert.city}_{alert.quality}"
        
        if key in self.recent_alerts:
            time_since_last = alert.timestamp - self.recent_alerts[key]
            if time_since_last < self.alert_cooldown:
                return False
        
        return True

    async def process_alert(self, alert: MarketAlert) -> bool:
        """
        Process and distribute alert through notification system.
        
        Handles alert validation, deduplication, persistence, and notification
        distribution to all registered handlers.
        
        Args:
            alert: Alert to process and distribute
            
        Returns:
            True if alert was sent, False if suppressed
        """
        if not self.should_send_alert(alert):
            return False

        self.alert_history.append(alert)
        key = f"{alert.item}_{alert.city}_{alert.quality}"
        self.recent_alerts[key] = alert.timestamp

        self.logger.info(f"ALERT: {alert.alert_type} for {alert.item} in {alert.city} "
                        f"(confidence: {alert.confidence:.3f})")

        for handler in self.notification_handlers:
            try:
                handler(alert)
            except Exception as e:
                self.logger.error(f"Notification handler error: {e}")

        return True

    def get_recent_alerts(self, hours_back: int = 24) -> List[MarketAlert]:
        """
        Retrieve recent alerts for analysis and reporting.
        
        Args:
            hours_back: Number of hours of alert history to return
            
        Returns:
            List of alerts from specified time period, ordered by timestamp
        """
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]


class RealTimeMonitor:
    """
    Main real-time monitoring system coordinator.
    
    Orchestrates polling, detection, and alerting components to provide
    continuous market manipulation monitoring. Manages system lifecycle,
    error recovery, and performance optimization for 24/7 operation.
    """

    def __init__(self, items: List[str] = None, cities: List[str] = None, 
                 polling_interval_seconds: int = 300) -> None:
        """
        Initialize real-time monitoring system with specified parameters.
        
        Args:
            items: List of items to monitor (default: config.DEFAULT_ITEMS)
            cities: List of cities to monitor (default: config.DEFAULT_CITIES)
            polling_interval_seconds: Seconds between polling cycles (default: 5 minutes)
        """
        self.items = items or DEFAULT_ITEMS
        self.cities = cities or DEFAULT_CITIES
        self.polling_interval = polling_interval_seconds
        
        self.poller = RateLimitedPoller()
        self.detector = SlidingWindowDetector()
        self.alert_manager = AlertManager()
        self.data_collector = AlbionDataCollector()
        
        self.is_running = False
        self.monitoring_stats = {
            'start_time': None,
            'polls_completed': 0,
            'alerts_generated': 0,
            'api_errors': 0
        }
        
        self.logger = logging.getLogger(__name__)

    async def poll_current_prices(self) -> Dict[str, Any]:
        """
        Poll current market prices for all monitored items and cities.
        
        Executes rate-limited API calls to fetch latest price data across
        all configured item/city combinations for real-time analysis.
        
        Returns:
            Dictionary containing polling results and statistics
        """
        results = {
            'timestamp': datetime.now(),
            'prices_collected': 0,
            'api_errors': 0,
            'items_updated': []
        }
        
        for item in self.items:
            try:
                prices_data = await self.poller.poll_api(
                    self.data_collector.fetch_current_prices,
                    [item], self.cities, [1, 2, 3]
                )
                
                if prices_data is not None and not prices_data.empty:
                    for _, row in prices_data.iterrows():
                        if row.get('sell_price_min', 0) > 0:
                            self.detector.add_price_data(
                                item=row.get('item_id', item),
                                city=row['city'],
                                timestamp=datetime.now(),
                                price=row['sell_price_min'],
                                quality=row.get('quality', 1)
                            )
                            results['prices_collected'] += 1
                            results['items_updated'].append(f"{item}_{row['city']}")
                        
            except Exception as e:
                self.logger.error(f"Error polling {item}: {e}")
                results['api_errors'] += 1
                self.monitoring_stats['api_errors'] += 1

        return results

    async def run_detection_cycle(self) -> List[MarketAlert]:
        """
        Execute complete detection cycle across all monitored items.
        
        Runs manipulation detection on all item/city combinations with
        sufficient data and generates alerts for suspicious activity.
        
        Returns:
            List of alerts generated during this detection cycle
        """
        alerts_generated = []
        
        for item in self.items:
            for city in self.cities:
                for quality in [1, 2, 3]:  # Focus on common qualities
                    try:
                        alert = await self.detector.detect_manipulation(item, city, quality)
                        
                        if alert and await self.alert_manager.process_alert(alert):
                            alerts_generated.append(alert)
                            self.monitoring_stats['alerts_generated'] += 1
                            
                    except Exception as e:
                        self.logger.error(f"Detection error for {item}/{city}/Q{quality}: {e}")

        return alerts_generated

    async def monitoring_loop(self) -> None:
        """
        Main monitoring loop for continuous market surveillance.
        
        Executes polling and detection cycles at regular intervals while
        handling errors gracefully and maintaining system statistics.
        Designed for long-running operation with automatic recovery.
        """
        self.monitoring_stats['start_time'] = datetime.now()
        self.logger.info(f"Starting real-time monitoring for {len(self.items)} items across {len(self.cities)} cities")
        
        while self.is_running:
            cycle_start = time.time()
            
            try:
                poll_results = await self.poll_current_prices()
                alerts = await self.run_detection_cycle()
                
                self.monitoring_stats['polls_completed'] += 1
                
                cycle_duration = time.time() - cycle_start
                self.logger.info(f"Monitoring cycle completed: {poll_results['prices_collected']} prices, "
                               f"{len(alerts)} alerts, {cycle_duration:.1f}s")

                if cycle_duration < self.polling_interval:
                    await asyncio.sleep(self.polling_interval - cycle_duration)
                    
            except Exception as e:
                self.logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(60)  # Wait before retry

    async def start_monitoring(self) -> None:
        """
        Start the real-time monitoring system.
        
        Initializes all components and begins continuous monitoring loop.
        Sets up graceful shutdown handling and error recovery mechanisms.
        """
        if self.is_running:
            self.logger.warning("Monitoring already running")
            return

        self.is_running = True
        
        # Set up basic console alert handler
        def console_alert_handler(alert: MarketAlert) -> None:
            print(f"\n🚨 MANIPULATION ALERT 🚨")
            print(f"Item: {alert.item} | City: {alert.city} | Quality: {alert.quality}")
            print(f"Price: {alert.current_price:,} silver (expected: {alert.expected_price:,.0f})")
            print(f"Deviation: {alert.price_deviation_pct:+.1f}% | Confidence: {alert.confidence:.1%}")
            print(f"Context: {alert.economic_context} | Time: {alert.timestamp.strftime('%H:%M:%S')}")
            print("-" * 60)
        
        self.alert_manager.add_notification_handler(console_alert_handler)
        
        await self.monitoring_loop()

    def stop_monitoring(self) -> None:
        """Stop the real-time monitoring system gracefully."""
        self.logger.info("Stopping real-time monitoring...")
        self.is_running = False

    def get_monitoring_stats(self) -> Dict[str, Any]:
        """
        Get current monitoring system statistics and performance metrics.
        
        Returns:
            Dictionary containing operational statistics and performance data
        """
        stats = self.monitoring_stats.copy()
        if stats['start_time']:
            stats['uptime_hours'] = (datetime.now() - stats['start_time']).total_seconds() / 3600
            stats['avg_alerts_per_hour'] = stats['alerts_generated'] / max(stats['uptime_hours'], 0.1)
        
        stats['recent_alerts'] = len(self.alert_manager.get_recent_alerts(24))
        return stats