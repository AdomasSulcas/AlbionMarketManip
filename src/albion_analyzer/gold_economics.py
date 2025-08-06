"""
Gold Economics Module
Handles gold price data and silver purchasing power analysis for market manipulation detection.
"""

import requests
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Union
from datetime import datetime, timedelta
import logging

class GoldEconomicsAnalyzer:
    """
    Analyzes gold/silver exchange rates to provide economic context for market manipulation detection.
    
    This class distinguishes between item-specific price manipulation and server-wide economic
    changes by analyzing gold/silver exchange rate patterns. Gold is Albion Online's premium
    currency that can be exchanged for silver (the main trading currency). Fluctuations in
    gold prices indicate server-wide economic conditions that affect all items proportionally.
    
    Attributes:
        base_url (str): API base URL for Albion Online data
        logger (logging.Logger): Logger instance for this analyzer
    """
    
    def __init__(self, base_url: str = "https://europe.albion-online-data.com") -> None:
        """
        Initialize the Gold Economics Analyzer.
        
        Args:
            base_url: Base URL for the Albion Online Data API
        """
        self.base_url = base_url
        self.logger = logging.getLogger(__name__)
        
    def fetch_gold_prices(self, hours_back: int = 168) -> pd.DataFrame:
        """
        Fetch historical gold price data from the Albion Online Data API.
        
        Retrieves hourly gold/silver exchange rate data for the specified time period.
        Gold prices represent the server's economic state - higher gold prices indicate
        silver is relatively cheaper (inflation period), while lower gold prices indicate
        silver is more expensive (deflation period).
        
        Args:
            hours_back: Number of hours of historical data to fetch. Default is 168 (1 week).
                       Maximum depends on API limits, typically several weeks of data available.
        
        Returns:
            DataFrame containing gold price history with columns:
            - timestamp (datetime): When the price was recorded
            - price (float): Gold price in silver
            - gold_price_change (float): Percentage change from previous hour
            - gold_volatility (float): 24-hour rolling standard deviation of prices
            
            Returns empty DataFrame if API request fails or no data available.
        
        Raises:
            No exceptions raised directly, but logs errors for failed API requests.
        """
        url = f"{self.base_url}/api/v2/stats/gold"
        params = {"count": hours_back}
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            if not data:
                return pd.DataFrame()
            
            df = pd.DataFrame(data)
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df = df.sort_values('timestamp')
            
            df['gold_price_change'] = df['price'].pct_change()
            df['gold_volatility'] = df['price'].rolling(24).std()
            
            return df
            
        except requests.RequestException as e:
            self.logger.error(f"Failed to fetch gold prices: {e}")
            return pd.DataFrame()
    
    def calculate_purchasing_power_adjustment(self, item_prices: pd.DataFrame, 
                                           gold_prices: pd.DataFrame) -> pd.DataFrame:
        """
        Adjust item prices by gold/silver purchasing power to normalize for server-wide economic changes.
        
        This method accounts for server-wide economic inflation/deflation by normalizing item prices
        against gold/silver exchange rates. When gold is expensive (high gold prices), silver is
        relatively cheap, so item prices should be adjusted upward to maintain purchasing power
        equivalency across time periods.
        
        Args:
            item_prices: DataFrame containing item price data with columns:
                        - timestamp (datetime): When prices were recorded
                        - price (float): Item price in silver
                        - log_price (float): Natural log of item price
                        Additional columns are preserved in output.
                        
            gold_prices: DataFrame containing gold price data with columns:
                        - timestamp (datetime): When gold prices were recorded  
                        - price (float): Gold price in silver
        
        Returns:
            DataFrame containing original item price data plus new columns:
            - date (datetime): Daily timestamp for merging
            - price_gold (float): Daily average gold price
            - purchasing_power_multiplier (float): Adjustment factor based on gold prices
            - adjusted_price (float): Gold-normalized item price
            - adjusted_log_price (float): Natural log of adjusted price
            
            If gold_prices is empty, returns original data with multiplier = 1.0.
        
        Note:
            Uses median gold price as baseline for calculating purchasing power multipliers.
            Missing gold price data is forward-filled to handle gaps in API data.
        """
        if gold_prices.empty or item_prices.empty:
            return item_prices
        
        gold_daily = gold_prices.copy()
        gold_daily['date'] = gold_daily['timestamp'].dt.date
        gold_daily = gold_daily.groupby('date')['price'].mean().reset_index()
        gold_daily['date'] = pd.to_datetime(gold_daily['date'])
        
        item_prices = item_prices.copy()
        item_prices['date'] = item_prices['timestamp'].dt.floor('D')
        
        merged = item_prices.merge(
            gold_daily, 
            on='date', 
            how='left',
            suffixes=('', '_gold')
        )
        
        merged['price_gold'] = merged['price_gold'].fillna(method='ffill')
        
        if 'price_gold' in merged.columns and not merged['price_gold'].isna().all():
            baseline_gold = merged['price_gold'].median()
            merged['purchasing_power_multiplier'] = merged['price_gold'] / baseline_gold
            merged['adjusted_price'] = merged['price'] * merged['purchasing_power_multiplier']
            merged['adjusted_log_price'] = np.log(merged['adjusted_price'])
        else:
            merged['purchasing_power_multiplier'] = 1.0
            merged['adjusted_price'] = merged['price']
            merged['adjusted_log_price'] = merged['log_price']
        
        return merged
    
    def detect_economic_regime(self, gold_prices: pd.DataFrame) -> Dict[str, Union[str, float]]:
        """
        Detect the current server-wide economic regime based on gold price trends.
        
        Analyzes recent gold price movements to classify the economic state as either
        silver inflation, silver deflation, or stable. This classification helps distinguish
        between item-specific price manipulation and server-wide economic trends that
        affect all items proportionally.
        
        Economic regimes:
        - silver_inflation: Gold prices increased >5%, indicating silver is cheaper
        - silver_deflation: Gold prices decreased >5%, indicating silver is more expensive  
        - stable: Gold price changes <5%, normal economic conditions
        
        Args:
            gold_prices: DataFrame containing gold price history with columns:
                        - timestamp (datetime): Price timestamps
                        - price (float): Gold prices in silver
                        
        Returns:
            Dictionary containing regime analysis:
            - regime (str): Economic regime classification
            - confidence (float): Confidence in classification (0.0 to 1.0)
            - price_change (float): Percentage change in gold prices
            - volatility (float): Coefficient of variation for recent prices
            - recent_avg (float): Average gold price in recent period
            - older_avg (float): Average gold price in comparison period
            
            Returns {"regime": "unknown", "confidence": 0.0} if insufficient data.
        
        Note:
            Requires at least 24 hours of data for analysis. Uses last 24 hours vs
            previous 24 hours for trend comparison. High volatility reduces confidence.
        """
        if gold_prices.empty or len(gold_prices) < 24:
            return {"regime": "unknown", "confidence": 0.0}
        
        recent = gold_prices.tail(24)
        older = gold_prices.iloc[-48:-24] if len(gold_prices) >= 48 else gold_prices.iloc[:-24]
        
        if older.empty:
            return {"regime": "stable", "confidence": 0.5}
        
        recent_avg = recent['price'].mean()
        older_avg = older['price'].mean()
        
        price_change = (recent_avg - older_avg) / older_avg
        volatility = recent['price'].std() / recent['price'].mean()
        
        if price_change > 0.05:
            regime = "silver_inflation"
            confidence = min(0.9, abs(price_change) * 10)
        elif price_change < -0.05:
            regime = "silver_deflation"
            confidence = min(0.9, abs(price_change) * 10)
        else:
            regime = "stable"
            confidence = 0.7 - volatility
        
        return {
            "regime": regime,
            "confidence": max(0.1, confidence),
            "price_change": price_change,
            "volatility": volatility,
            "recent_avg": recent_avg,
            "older_avg": older_avg
        }
    
    def filter_manipulation_vs_inflation(self, suspicious_cases: pd.DataFrame,
                                       gold_context: Dict[str, Union[str, float]]) -> pd.DataFrame:
        """
        Filter manipulation alerts by distinguishing true manipulation from economic trends.
        
        Adjusts manipulation detection confidence based on current economic regime.
        During server-wide economic changes (inflation/deflation), many items will show
        price increases that are not due to manipulation but rather to economic conditions.
        This method reduces false positives by adjusting confidence scores and filtering
        out cases likely caused by economic trends rather than targeted manipulation.
        
        Filtering logic:
        - silver_inflation: Price increases are less suspicious (confidence * 0.7)
        - silver_deflation: Price increases are more suspicious (confidence * 1.3)  
        - stable: Normal confidence scoring
        
        Args:
            suspicious_cases: DataFrame containing detected manipulation cases with columns:
                            - confidence_score (float): Original manipulation confidence
                            Additional columns are preserved and returned.
                            
            gold_context: Dictionary from detect_economic_regime containing:
                         - regime (str): Current economic regime
                         - confidence (float): Confidence in regime classification
                         
        Returns:
            DataFrame containing filtered manipulation cases with new columns:
            - confidence_score (float): Adjusted confidence score
            - economic_context (str): Economic regime label for case
            
            Cases below confidence threshold are removed from results.
            Returns empty DataFrame if no cases pass filtering.
        
        Note:
            Uses higher confidence threshold during unstable economic periods
            to reduce false positives from economic noise.
        """
        if suspicious_cases.empty:
            return suspicious_cases
        
        filtered_cases = suspicious_cases.copy()
        
        if gold_context["regime"] == "silver_inflation":
            filtered_cases['confidence_score'] *= 0.7
            filtered_cases['economic_context'] = "silver_inflation_period"
        elif gold_context["regime"] == "silver_deflation":
            filtered_cases['confidence_score'] *= 1.3
            filtered_cases['economic_context'] = "silver_deflation_period"
        else:
            filtered_cases['economic_context'] = "stable_period"
        
        confidence_threshold = 0.3 if gold_context["regime"] == "stable" else 0.4
        filtered_cases = filtered_cases[
            filtered_cases['confidence_score'] >= confidence_threshold
        ]
        
        return filtered_cases
    
    def add_gold_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Enhance dataset with gold-based economic features for machine learning models.
        
        Adds purchasing power adjustments and gold market indicators to item price data.
        These features help ML models distinguish between item-specific price manipulation
        and server-wide economic trends by providing economic context for each price point.
        
        Added features include:
        - Purchasing power adjusted prices normalized by gold/silver exchange rates
        - Gold price indicators showing current economic state
        - Economic regime context for each data point
        
        Args:
            df: DataFrame containing item price data with columns:
                - timestamp (datetime): Price timestamps
                - price (float): Item prices in silver  
                - log_price (float): Natural log of prices
                Additional columns are preserved.
                
        Returns:
            DataFrame containing original data plus new gold-based features:
            - purchasing_power_multiplier (float): Gold-based price adjustment factor
            - adjusted_price (float): Gold-normalized item price
            - adjusted_log_price (float): Natural log of adjusted price
            - current_gold_price (float): Latest gold price for context
            - gold_volatility (float): Recent gold price volatility
            
            If gold data unavailable, adds features with default values (multiplier=1.0).
        
        Note:
            Fetches gold price history spanning the dataset timeframe plus one week buffer
            to ensure adequate data for analysis. Uses latest gold prices for market context.
        """
        gold_prices = self.fetch_gold_prices(hours_back=len(df) * 24 + 168)
        
        if gold_prices.empty:
            df['gold_price'] = np.nan
            df['gold_volatility'] = np.nan
            df['purchasing_power_multiplier'] = 1.0
            return df
        
        enhanced_df = self.calculate_purchasing_power_adjustment(df, gold_prices)
        
        latest_gold = gold_prices.iloc[-1] if not gold_prices.empty else None
        if latest_gold is not None:
            enhanced_df['current_gold_price'] = latest_gold['price']
            enhanced_df['gold_volatility'] = latest_gold.get('gold_volatility', 0)
        
        return enhanced_df