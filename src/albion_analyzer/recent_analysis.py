"""
Recent Market Anomaly Detection
Focuses on detecting manipulation in the last few hours with comprehensive item coverage.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import defaultdict

from .data_collector import AlbionDataCollector
from .models import ManipulationDetector, SimpleRuleBasedDetector
from .gold_economics import GoldEconomicsAnalyzer
from .config import DEFAULT_CITIES


class RecentAnomalyDetector:
    """
    Specialized detector for recent market manipulation focusing on the last few hours.
    
    Optimized for detecting fresh manipulation attempts by analyzing recent price movements
    against short-term historical baselines. Uses lightweight detection suitable for
    frequent execution against comprehensive item lists.
    """

    def __init__(self, recent_hours: int = 6, baseline_days: int = 30) -> None:
        """
        Initialize recent anomaly detector with configurable time windows.
        
        Args:
            recent_hours: Hours to analyze for recent anomalies (default: 6 hours)
            baseline_days: Days of historical data for baseline comparison (default: 30 days)
        """
        self.recent_hours = recent_hours
        self.baseline_days = baseline_days
        self.data_collector = AlbionDataCollector()
        self.gold_analyzer = GoldEconomicsAnalyzer()
        self.logger = logging.getLogger(__name__)
        
        # Lightweight detector for frequent execution
        self.detector = SimpleRuleBasedDetector(z_threshold=1.8, peer_dev_threshold=1.2)

    def get_all_tradeable_items(self) -> List[str]:
        """
        Get comprehensive list of all tradeable items in Albion Online.
        
        Returns extensive list covering all major item categories for thorough
        market surveillance instead of just a small subset.
        
        Returns:
            List of item IDs covering all major categories
        """
        # Comprehensive item list covering all major categories
        items = []
        
        # Tiers 3-8 for better coverage
        tiers = ['T3', 'T4', 'T5', 'T6', 'T7', 'T8']
        
        # Bags and Capes (high-volume items)
        for tier in tiers:
            items.extend([
                f"{tier}_BAG", f"{tier}_CAPE"
            ])
        
        # Weapons - All major weapon types
        weapon_types = [
            "2H_AXE", "2H_BOW", "2H_CROSSBOW", "2H_SWORD", "2H_MACE", "2H_HAMMER",
            "MAIN_AXE", "MAIN_BOW", "MAIN_CROSSBOW", "MAIN_SWORD", "MAIN_MACE", "MAIN_HAMMER",
            "2H_ARCANESTAFF", "2H_CURSEDSTAFF", "2H_FIRESTAFF", "2H_FROSTSTAFF", "2H_HOLYSTAFF", "2H_NATURESTAFF",
            "MAIN_ARCANESTAFF", "MAIN_CURSEDSTAFF", "MAIN_FIRESTAFF", "MAIN_FROSTSTAFF", "MAIN_HOLYSTAFF", "MAIN_NATURESTAFF",
            "2H_KNIFE_MORGANA", "MAIN_KNIFE", "2H_TWINSCYTHE_HELL", "2H_RAM_KEEPER"
        ]
        
        for tier in tiers:
            for weapon in weapon_types:
                items.append(f"{tier}_{weapon}")
        
        # Armor - All sets
        armor_types = [
            "HEAD_PLATE_SET1", "ARMOR_PLATE_SET1", "SHOES_PLATE_SET1",  # Plate
            "HEAD_LEATHER_SET1", "ARMOR_LEATHER_SET1", "SHOES_LEATHER_SET1",  # Leather  
            "HEAD_CLOTH_SET1", "ARMOR_CLOTH_SET1", "SHOES_CLOTH_SET1",  # Cloth
        ]
        
        for tier in tiers:
            for armor in armor_types:
                items.append(f"{tier}_{armor}")
        
        # Tools
        tool_types = [
            "TOOL_PICKAXE", "TOOL_AXE", "TOOL_HAMMER", "TOOL_HOE", "TOOL_KNIFE", "TOOL_SICKLE", "TOOL_SKINNING_KNIFE", "TOOL_NEEDLE"
        ]
        
        for tier in tiers:
            for tool in tool_types:
                items.append(f"{tier}_{tool}")
        
        # Resources (high manipulation targets)
        resource_types = [
            "WOOD", "STONE", "ORE", "HIDE", "FIBER", "CLOTH", "METALBAR", "PLANKS", "STONEBLOCK", "LEATHER"
        ]
        
        for tier in tiers:
            for resource in resource_types:
                items.append(f"{tier}_{resource}")
        
        # Food items (often manipulated)
        food_types = [
            "MEAL_PIE", "MEAL_SOUP", "MEAL_BREAD", "MEAL_SANDWICH", "MEAL_SALAD", "MEAL_FISH", "MEAL_OMELETTE"
        ]
        
        for tier in ['T3', 'T4', 'T5', 'T6', 'T7', 'T8']:
            for food in food_types:
                items.append(f"{tier}_{food}")
        
        # Potions (high-value targets)
        potion_types = [
            "POTION_HEAL", "POTION_ENERGY", "POTION_STONESKIN", "POTION_RESISTANCE", "POTION_STICKY", "POTION_SLOW"
        ]
        
        for tier in ['T3', 'T4', 'T5', 'T6', 'T7', 'T8']:
            for potion in potion_types:
                items.append(f"{tier}_{potion}")
        
        # Mounts (expensive items, manipulation targets)
        mount_types = [
            "MOUNT_HORSE", "MOUNT_HORSE_UNDEAD", "MOUNT_OX", "MOUNT_STAG", "MOUNT_SWIFTCLAW", "MOUNT_BEAR", "MOUNT_DIREWOLF"
        ]
        
        for tier in ['T3', 'T4', 'T5', 'T6', 'T7', 'T8']:
            for mount in mount_types:
                items.append(f"{tier}_{mount}")
        
        # Remove duplicates and sort
        items = sorted(list(set(items)))
        
        self.logger.info(f"Generated comprehensive item list with {len(items)} items across all categories")
        return items

    def analyze_recent_anomalies(self, items: List[str] = None, cities: List[str] = None, 
                               quality: int = 1) -> Dict[str, Any]:
        """
        Analyze recent market anomalies focusing on the last few hours.
        
        Compares recent price movements (last few hours) against short-term baseline
        to detect fresh manipulation attempts. Optimized for frequent execution
        with comprehensive item coverage.
        
        Args:
            items: Items to analyze (default: comprehensive list of all tradeable items)
            cities: Cities to analyze (default: all major cities)
            quality: Item quality to analyze (default: 1)
            
        Returns:
            Dictionary containing recent anomaly analysis results
        """
        if items is None:
            items = self.get_all_tradeable_items()
        if cities is None:
            cities = DEFAULT_CITIES
        
        analysis_start = datetime.now()
        self.logger.info(f"Analyzing recent anomalies: {len(items)} items, {len(cities)} cities")
        self.logger.info(f"Comparing current prices against {self.baseline_days}-day historical baseline")
        
        current_data = []
        baseline_data = []
        processing_stats = {'items_processed': 0, 'api_errors': 0, 'anomalies_detected': 0}
        
        # Process items in smaller batches to manage API calls with delays
        batch_size = 5  # Smaller batches to respect rate limits
        import time
        
        for i in range(0, len(items), batch_size):
            batch_items = items[i:i + batch_size]
            
            try:
                # Add delay between batches to respect rate limits
                if i > 0:
                    time.sleep(2)  # 2 second delay between batches
                
                # Get CURRENT prices (what items are selling for right now)
                current_prices = self.data_collector.fetch_current_prices(batch_items, cities, [quality])
                
                if not current_prices.empty:
                    # Add timestamp for analysis
                    current_prices['timestamp'] = datetime.now()
                    current_data.append(current_prices)
                
                # Add delay before historical data request
                time.sleep(1)
                
                # Get HISTORICAL baseline data (30+ days to establish normal patterns)
                historical_data = self.data_collector.fetch_historical_data(batch_items, cities, self.baseline_days, quality)
                if not historical_data.empty:
                    baseline_data.append(historical_data)
                
                processing_stats['items_processed'] += len(batch_items)
                
            except Exception as e:
                self.logger.error(f"Error processing batch {i//batch_size + 1}: {e}")
                processing_stats['api_errors'] += 1
                continue
            
            # Log progress less frequently
            if (i // batch_size + 1) % 5 == 0:
                self.logger.info(f"Processed {i + len(batch_items)}/{len(items)} items...")
        
        # Combine data
        if current_data:
            current_df = pd.concat(current_data, ignore_index=True)
        else:
            current_df = pd.DataFrame()
            
        if baseline_data:
            baseline_df = pd.concat(baseline_data, ignore_index=True)
        else:
            baseline_df = pd.DataFrame()
        
        if current_df.empty or baseline_df.empty:
            return {
                'error': 'Insufficient data for anomaly analysis - need both current prices and historical baseline',
                'processing_stats': processing_stats
            }
        
        # Detect anomalies by comparing CURRENT prices vs HISTORICAL baseline
        anomalies = self._detect_current_vs_historical_anomalies(current_df, baseline_df)
        processing_stats['anomalies_detected'] = len(anomalies)
        
        # Get economic context using historical data
        gold_context = self.gold_analyzer.detect_economic_regime(
            self.gold_analyzer.fetch_gold_prices(hours_back=self.baseline_days * 24)
        )
        
        analysis_duration = (datetime.now() - analysis_start).total_seconds()
        
        return {
            'analysis_timestamp': datetime.now().isoformat(),
            'analysis_approach': 'current_vs_historical_baseline',
            'time_window': {
                'current_prices': 'real-time',
                'historical_baseline_days': self.baseline_days
            },
            'processing_stats': processing_stats,
            'anomalies': anomalies,
            'gold_economic_context': gold_context,
            'analysis_duration_seconds': analysis_duration,
            'items_analyzed': len(items),
            'cities_analyzed': len(cities)
        }

    def _detect_current_vs_historical_anomalies(self, current_df: pd.DataFrame, baseline_df: pd.DataFrame) -> List[Dict[str, Any]]:
        """
        Detect anomalies by comparing current prices against historical baseline patterns.
        
        This is the correct approach: compare what items are selling for RIGHT NOW against
        what they've sold for over the past 30+ days to detect manipulation.
        
        Args:
            current_df: Current price data (real-time market prices)
            baseline_df: Historical baseline data (30+ days of historical patterns)
            
        Returns:
            List of anomaly dictionaries with details
        """
        anomalies = []
        
        # Calculate baseline statistics for each item/city combination
        baseline_stats = {}
        
        # Handle different column naming conventions
        item_col = 'item_id' if 'item_id' in baseline_df.columns else 'item'
        city_col = 'city' if 'city' in baseline_df.columns else 'location'
        
        for (item, city), group in baseline_df.groupby([item_col, city_col]):
            if len(group) >= 3:  # Need minimum data points
                baseline_stats[(item, city)] = {
                    'median_price': group['price'].median(),
                    'mean_price': group['price'].mean(),
                    'std_price': group['price'].std(),
                    'price_75th': group['price'].quantile(0.75),
                    'price_95th': group['price'].quantile(0.95),
                    'data_points': len(group)
                }
        
        # Compare CURRENT prices against HISTORICAL baseline
        for _, current_row in current_df.iterrows():
            # Handle different column names from API
            item = current_row.get('item_id', current_row.get('item', 'unknown'))
            city = current_row.get('city', 'unknown')
            current_price = current_row.get('sell_price_min', current_row.get('price', 0))
            
            if current_price <= 0:
                continue
            
            baseline_key = (item, city)
            if baseline_key not in baseline_stats:
                continue
                
            baseline = baseline_stats[baseline_key]
            
            # Calculate anomaly indicators
            price_vs_median = (current_price - baseline['median_price']) / baseline['median_price']
            price_vs_mean = (current_price - baseline['mean_price']) / baseline['mean_price']
            
            # Z-score calculation
            if baseline['std_price'] > 0:
                z_score = (current_price - baseline['mean_price']) / baseline['std_price']
            else:
                z_score = 0
            
            # Multiple anomaly detection criteria
            is_anomaly = False
            anomaly_reasons = []
            confidence = 0.0
            
            # Criteria 1: Significant deviation from median (>30% increase)
            if price_vs_median > 0.3:
                is_anomaly = True
                anomaly_reasons.append(f"Price {price_vs_median:.1%} above baseline median")
                confidence += min(price_vs_median * 2, 0.5)
            
            # Criteria 2: High Z-score (>2.0)
            if abs(z_score) > 2.0:
                is_anomaly = True
                anomaly_reasons.append(f"High Z-score: {z_score:.2f}")
                confidence += min(abs(z_score) * 0.1, 0.3)
            
            # Criteria 3: Price above 95th percentile
            if current_price > baseline['price_95th']:
                is_anomaly = True
                anomaly_reasons.append("Price above 95th percentile of baseline")
                confidence += 0.2
            
            # Criteria 4: Extreme deviation (>100% increase)
            if price_vs_median > 1.0:
                is_anomaly = True
                anomaly_reasons.append(f"Extreme price increase: {price_vs_median:.1%}")
                confidence += min(price_vs_median, 1.0)
            
            if is_anomaly:
                anomaly = {
                    'timestamp': current_row['timestamp'].isoformat() if pd.notna(current_row['timestamp']) else datetime.now().isoformat(),
                    'item': item,
                    'city': city,
                    'current_price': float(current_price),
                    'baseline_median': float(baseline['median_price']),
                    'baseline_mean': float(baseline['mean_price']),
                    'price_deviation_pct': float(price_vs_median * 100),
                    'z_score': float(z_score),
                    'confidence': min(confidence, 1.0),
                    'anomaly_reasons': anomaly_reasons,
                    'baseline_data_points': baseline['data_points'],
                    'quality': current_row.get('quality', 1)
                }
                anomalies.append(anomaly)
        
        # Sort by confidence and return top anomalies
        anomalies.sort(key=lambda x: x['confidence'], reverse=True)
        return anomalies[:50]  # Limit to top 50 most significant anomalies

    def get_high_priority_items(self) -> List[str]:
        """
        Get list of high-priority items that are frequently manipulated.
        
        Returns:
            List of item IDs that are common manipulation targets
        """
        # High-value, frequently traded items that are common manipulation targets
        priority_items = []
        
        # High-tier valuable items
        high_tiers = ['T6', 'T7', 'T8']
        
        # Expensive weapons and armor
        priority_weapons = [
            "2H_AXE", "2H_BOW", "2H_SWORD", "2H_ARCANESTAFF", "2H_CURSEDSTAFF"
        ]
        
        priority_armor = [
            "HEAD_PLATE_SET1", "ARMOR_PLATE_SET1", "SHOES_PLATE_SET1"
        ]
        
        for tier in high_tiers:
            for item in priority_weapons + priority_armor:
                priority_items.append(f"{tier}_{item}")
        
        # All bags and capes (universally useful)
        for tier in ['T4', 'T5', 'T6', 'T7', 'T8']:
            priority_items.extend([f"{tier}_BAG", f"{tier}_CAPE"])
        
        # High-value mounts
        priority_mounts = ["T5_MOUNT_HORSE", "T6_MOUNT_HORSE", "T7_MOUNT_HORSE", "T8_MOUNT_HORSE"]
        priority_items.extend(priority_mounts)
        
        # Rare materials
        rare_materials = []
        for tier in ['T6', 'T7', 'T8']:
            rare_materials.extend([
                f"{tier}_ORE", f"{tier}_HIDE", f"{tier}_FIBER", f"{tier}_WOOD", f"{tier}_STONE"
            ])
        priority_items.extend(rare_materials)
        
        return priority_items

    def quick_recent_scan(self) -> Dict[str, Any]:
        """
        Quick scan of high-priority items comparing current prices vs historical baseline.
        
        Optimized for frequent execution (every 15-30 minutes) focusing on
        items most likely to be manipulated. Compares current market prices
        against 30-day historical baselines for immediate anomaly detection.
        
        Returns:
            Dictionary with quick scan results
        """
        self.logger.info(f"Running quick scan: current prices vs {self.baseline_days}-day baseline")
        
        priority_items = self.get_high_priority_items()
        
        # Use priority items with current vs historical baseline approach
        results = self.analyze_recent_anomalies(
            items=priority_items,
            cities=DEFAULT_CITIES,
            quality=1
        )
        
        # Add quick scan specific metadata
        results['scan_type'] = 'quick_priority_scan'
        results['priority_items_count'] = len(priority_items)
        
        return results