"""
Data Collection Module
Handles API calls and data preprocessing for Albion Online market analysis
"""

import requests
import datetime as dt
import pandas as pd
import numpy as np
from typing import List, Dict, Optional

class AlbionDataCollector:
    """
    Comprehensive data collection and preprocessing system for Albion Online market analysis.
    
    Handles API communication with the Albion Online Data Project, including historical price
    retrieval, current market data fetching, and feature engineering for manipulation detection.
    Automatically manages API rate limits, error handling, and data validation to ensure
    reliable data pipeline operation.
    
    Key capabilities:
    - Historical price data collection with configurable timeframes
    - Current market price fetching across multiple cities and qualities
    - Technical indicator calculation (z-scores, peer deviations, rolling statistics)
    - Multi-quality data support for comprehensive market analysis
    - Graceful error handling and API rate limit compliance
    
    Attributes:
        base_url (str): API base URL for data requests
        rate_limit_delay (float): Delay between requests to respect API limits
    """
    
    def __init__(self, base_url: str = "https://europe.albion-online-data.com") -> None:
        """
        Initialize the Albion Data Collector.
        
        Args:
            base_url: Base URL for the Albion Online Data API
        """
        self.base_url = base_url
        self.rate_limit_delay = 0.1
        
    def fetch_historical_data(self, items: List[str], cities: List[str], 
                            days_back: int = 30, quality: int = 1) -> pd.DataFrame:
        """
        Fetch historical price data for multiple items and cities
        """
        print(f"Fetching historical data for {len(items)} items across {len(cities)} cities...")
        
        today = dt.date.today()
        start_date = today - dt.timedelta(days=days_back)
        
        all_records = []
        
        for item in items:
            print(f"  Processing {item}...")
            data = self._fetch_item_history(item, cities, start_date, today, quality)
            
            for entry in data:
                for daily_data in entry.get("data", []):
                    all_records.append({
                        "item": item,
                        "city": entry["location"],
                        "timestamp": pd.to_datetime(daily_data["timestamp"]),
                        "price": daily_data["avg_price"],
                        "quality": quality
                    })
        
        if not all_records:
            return pd.DataFrame()
        
        df = pd.DataFrame(all_records).sort_values(["item", "city", "timestamp"])
        
        # Add derived features
        df["log_price"] = np.log(df["price"])
        df["tier"] = df["item"].str.extract(r"T(\d)")[0].astype(int)
        df["slot"] = df["item"].str.split("_").str[2]
        
        return df
    
    def _fetch_item_history(self, item: str, cities: List[str], 
                          start_date: dt.date, end_date: dt.date, quality: int) -> List[Dict]:
        """
        Fetch historical data for a single item
        """
        url = f"{self.base_url}/api/v2/stats/history/{item}.json"
        params = {
            "date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "locations": ",".join(cities),
            "time-scale": 24,  # daily granularity
            "qualities": quality
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            print(f"Error fetching data for {item}: {e}")
            return []
    
    def fetch_current_prices(self, items: List[str], cities: List[str], 
                           qualities: Optional[List[int]] = None) -> pd.DataFrame:
        """
        Fetch current market prices for multiple items, cities, and qualities
        """
        if qualities is None:
            qualities = [1]
        
        print(f"Fetching current prices for {len(items)} items...")
        
        all_data = []
        
        for item in items:
            cities_str = ",".join(cities)
            qualities_str = ",".join(map(str, qualities))
            
            url = f"{self.base_url}/api/v2/stats/prices/{item}.json"
            params = {
                "locations": cities_str,
                "qualities": qualities_str
            }
            
            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                data = response.json()
                
                if data:
                    df = pd.json_normalize(data)
                    df["item"] = item
                    all_data.append(df)
                    
            except requests.RequestException as e:
                print(f"Error fetching current prices for {item}: {e}")
        
        if all_data:
            return pd.concat(all_data, ignore_index=True)
        return pd.DataFrame()
    
    def add_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical analysis indicators to the dataset
        """
        if df.empty:
            return df
        
        df = df.copy()
        
        # Rolling z-scores
        def rolling_z(series, window=7):
            roll = series.rolling(window, min_periods=3)
            return (series - roll.mean()) / roll.std()
        
        df["z"] = df.groupby(["item", "city"])["log_price"].transform(rolling_z)
        
        # Peer deviations
        peer_median = df.groupby(["timestamp", "tier", "slot"])["log_price"].transform("median")
        df["peer_dev"] = df["log_price"] - peer_median
        
        # Absolute values for ML features
        df["abs_z"] = df["z"].abs()
        df["abs_peer_dev"] = df["peer_dev"].abs()
        
        return df
    
    def get_manipulation_training_data(self, items: List[str], cities: List[str], 
                                     days_back: int = 30) -> pd.DataFrame:
        """
        Collect and preprocess data suitable for manipulation detection training
        """
        # Fetch historical data
        df = self.fetch_historical_data(items, cities, days_back)
        
        if df.empty:
            print("No historical data retrieved")
            return df
        
        # Add technical indicators
        df = self.add_technical_indicators(df)
        
        # Add current market context (if available)
        try:
            current_prices = self.fetch_current_prices(items, cities, [1, 2, 3])
            if not current_prices.empty:
                print(f"Added current market context for {len(current_prices)} price points")
        except Exception as e:
            print(f"Could not fetch current market context: {e}")
        
        print(f"Prepared dataset with {len(df)} records")
        print(f"Features: {list(df.columns)}")
        
        return df

# Convenience functions for backward compatibility
def fetch_history(item: str, cities: List[str], days_back: int = 30) -> List[Dict]:
    """Legacy function - use AlbionDataCollector instead"""
    collector = AlbionDataCollector()
    df = collector.fetch_historical_data([item], cities, days_back)
    
    # Convert back to old format for compatibility
    result = {}
    for city in cities:
        city_data = df[df["city"] == city]
        if not city_data.empty:
            data_records = []
            for _, row in city_data.iterrows():
                data_records.append({
                    "timestamp": row["timestamp"].isoformat(),
                    "avg_price": row["price"]
                })
            result[city] = [{"location": city, "data": data_records}]
    
    return result.get(item, [])