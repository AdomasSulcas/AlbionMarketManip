"""
Utility functions for Albion Online Market Analysis
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import json

def setup_logging(level: str = "INFO") -> logging.Logger:
    """
    Setup logging configuration
    """
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger("albion_analysis")

def validate_items(items: List[str]) -> List[str]:
    """
    Validate item names follow Albion Online format
    """
    valid_items = []
    for item in items:
        if item.startswith(('T1_', 'T2_', 'T3_', 'T4_', 'T5_', 'T6_', 'T7_', 'T8_')):
            valid_items.append(item)
        else:
            logging.warning(f"Invalid item format: {item}")
    
    return valid_items

def validate_cities(cities: List[str]) -> List[str]:
    """
    Validate city names
    """
    valid_cities = [
        "Caerleon", "Bridgewatch", "Lymhurst", 
        "Fort Sterling", "Martlock", "Thetford"
    ]
    
    validated = []
    for city in cities:
        if city in valid_cities:
            validated.append(city)
        else:
            logging.warning(f"Unknown city: {city}")
    
    return validated

def format_price(price: float) -> str:
    """
    Format price for display
    """
    if price >= 1000000:
        return f"{price/1000000:.1f}M"
    elif price >= 1000:
        return f"{price/1000:.1f}K"
    else:
        return f"{price:.0f}"

def calculate_percentage_change(old_value: float, new_value: float) -> float:
    """
    Calculate percentage change between two values
    """
    if old_value == 0:
        return 0.0
    return ((new_value - old_value) / old_value) * 100

def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safe division with default value for zero denominator
    """
    return numerator / denominator if denominator != 0 else default

def extract_tier_from_item(item: str) -> int:
    """
    Extract tier number from item name (T4_AXE -> 4)
    """
    try:
        return int(item[1])
    except (IndexError, ValueError):
        return 1

def extract_slot_from_item(item: str) -> str:
    """
    Extract equipment slot from item name
    T4_2H_AXE -> AXE
    T5_HEAD_PLATE_SET1 -> HEAD
    """
    parts = item.split("_")
    if len(parts) < 3:
        return parts[-1]
    
    # Handle special cases
    if "2H" in parts:
        return parts[2]  # AXE, BOW, etc.
    elif "HEAD" in parts:
        return "HEAD"
    elif "CHEST" in parts:
        return "CHEST"
    elif "SHOES" in parts:
        return "SHOES"
    else:
        return parts[2]

def detect_outliers_iqr(series: pd.Series, k: float = 1.5) -> pd.Series:
    """
    Detect outliers using Interquartile Range method
    """
    Q1 = series.quantile(0.25)
    Q3 = series.quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - k * IQR
    upper_bound = Q3 + k * IQR
    
    return (series < lower_bound) | (series > upper_bound)

def rolling_statistics(df: pd.DataFrame, column: str, windows: List[int]) -> pd.DataFrame:
    """
    Add rolling statistics for specified windows
    """
    result_df = df.copy()
    
    for window in windows:
        result_df[f'{column}_rolling_mean_{window}'] = (
            df.groupby(['item', 'city'])[column]
            .transform(lambda x: x.rolling(window, min_periods=2).mean())
        )
        result_df[f'{column}_rolling_std_{window}'] = (
            df.groupby(['item', 'city'])[column]
            .transform(lambda x: x.rolling(window, min_periods=2).std())
        )
    
    return result_df

def save_results_to_json(results: Dict[str, Any], filepath: str) -> None:
    """
    Save analysis results to JSON file
    """
    # Convert pandas objects to JSON-serializable format
    serializable_results = {}
    
    for key, value in results.items():
        if isinstance(value, pd.DataFrame):
            serializable_results[key] = value.to_dict('records')
        elif isinstance(value, pd.Series):
            serializable_results[key] = value.to_list()
        elif isinstance(value, (pd.Timestamp, datetime)):
            serializable_results[key] = value.isoformat()
        elif isinstance(value, np.ndarray):
            serializable_results[key] = value.tolist()
        elif isinstance(value, (np.int64, np.float64)):
            serializable_results[key] = value.item()
        else:
            serializable_results[key] = value
    
    with open(filepath, 'w') as f:
        json.dump(serializable_results, f, indent=2, default=str)
    
    logging.info(f"Results saved to {filepath}")

def load_results_from_json(filepath: str) -> Dict[str, Any]:
    """
    Load analysis results from JSON file
    """
    with open(filepath, 'r') as f:
        results = json.load(f)
    
    logging.info(f"Results loaded from {filepath}")
    return results

def create_summary_table(results: Dict[str, Any]) -> pd.DataFrame:
    """
    Create a summary table from analysis results
    """
    if 'item_summary' not in results:
        return pd.DataFrame()
    
    summary_df = pd.DataFrame(results['item_summary'])
    
    # Format columns for better display
    if not summary_df.empty:
        summary_df['manipulation_rate'] = summary_df['manipulation_rate'].apply(lambda x: f"{x:.1%}")
        summary_df['avg_confidence'] = summary_df['avg_confidence'].apply(lambda x: f"{x:.3f}")
    
    return summary_df

def filter_suspicious_cases(results: Dict[str, Any], 
                          min_confidence: float = 0.5) -> List[Dict[str, Any]]:
    """
    Filter suspicious cases by minimum confidence threshold
    """
    if 'suspicious_cases' not in results:
        return []
    
    return [
        case for case in results['suspicious_cases'] 
        if case.get('confidence_score', 0) >= min_confidence
    ]

def get_top_manipulated_items(results: Dict[str, Any], top_n: int = 5) -> List[Dict[str, Any]]:
    """
    Get top N most manipulated items by manipulation rate
    """
    if 'item_summary' not in results:
        return []
    
    item_summary = results['item_summary']
    sorted_items = sorted(item_summary, key=lambda x: x['manipulation_rate'], reverse=True)
    
    return sorted_items[:top_n]

def calculate_market_health_score(results: Dict[str, Any]) -> float:
    """
    Calculate overall market health score (0-100, higher is healthier)
    """
    if 'summary' not in results:
        return 0.0
    
    summary = results['summary']
    
    # Base score starts at 100
    health_score = 100.0
    
    # Deduct points for manipulation rate
    manipulation_rate = summary.get('manipulation_rate', 0)
    health_score -= manipulation_rate * 1000  # 10% manipulation = -100 points
    
    # Deduct points for quality anomalies
    quality_anomalies = summary.get('quality_anomalies', 0)
    total_records = summary.get('total_records', 1)
    quality_anomaly_rate = quality_anomalies / total_records
    health_score -= quality_anomaly_rate * 500  # Moderate penalty for quality issues
    
    # Ensure score is between 0 and 100
    return max(0.0, min(100.0, health_score))

class PerformanceTimer:
    """
    Context manager for timing operations
    """
    
    def __init__(self, operation_name: str):
        self.operation_name = operation_name
        self.start_time = None
        
    def __enter__(self):
        self.start_time = datetime.now()
        logging.info(f"Starting {self.operation_name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = end_time - self.start_time
        logging.info(f"{self.operation_name} completed in {duration.total_seconds():.2f} seconds")
        
        if exc_type is not None:
            logging.error(f"{self.operation_name} failed: {exc_val}")

# Data validation functions
def validate_dataframe_structure(df: pd.DataFrame, required_columns: List[str]) -> bool:
    """
    Validate that DataFrame has required columns
    """
    missing_columns = set(required_columns) - set(df.columns)
    if missing_columns:
        logging.error(f"Missing required columns: {missing_columns}")
        return False
    return True

def clean_price_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean price data by removing invalid entries
    """
    cleaned_df = df.copy()
    
    # Remove rows with invalid prices
    cleaned_df = cleaned_df[cleaned_df['price'] > 0]
    cleaned_df = cleaned_df[cleaned_df['price'] < 1e9]  # Remove unrealistic prices
    
    # Remove duplicate entries
    cleaned_df = cleaned_df.drop_duplicates(subset=['item', 'city', 'timestamp'])
    
    logging.info(f"Data cleaning: {len(df)} -> {len(cleaned_df)} records")
    
    return cleaned_df