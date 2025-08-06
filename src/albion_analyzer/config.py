"""
Configuration settings for Albion Online Market Analysis
"""

# API Configuration
API_BASE_URL = "https://europe.albion-online-data.com"
API_RATE_LIMIT_DELAY = 0.1  # seconds between requests
API_TIMEOUT = 30  # request timeout in seconds

# Default Analysis Settings
DEFAULT_ITEMS = [
    "T4_BAG", "T5_BAG", 
    "T4_CAPE", "T5_CAPE", 
    "T4_HEAD_PLATE_SET1", "T5_HEAD_PLATE_SET1",
    "T4_2H_AXE", "T5_2H_AXE", 
    "T4_2H_BOW", "T5_2H_BOW"
]

DEFAULT_CITIES = [
    "Caerleon", "Bridgewatch", "Lymhurst", 
    "Fort Sterling", "Martlock"
]

DEFAULT_QUALITIES = [1, 2, 3, 4, 5]

# Analysis Parameters
DEFAULT_LOOKBACK_DAYS = 30
DEFAULT_CONTAMINATION = 0.01  # 1% expected manipulation rate
DEFAULT_Z_THRESHOLD = 2.0
DEFAULT_PEER_DEV_THRESHOLD = 1.5
DEFAULT_SPREAD_THRESHOLD = 50.0  # 50% bid-ask spread threshold

# Feature Engineering
ROLLING_WINDOWS = [3, 7, 14]  # days for rolling statistics
MIN_HISTORY_POINTS = 8  # minimum points needed for forecasting

# Quality Analysis
EXPECTED_QUALITY_MULTIPLIERS = {
    "1-2": 1.2,   # Quality 2 typically 20% more expensive than Quality 1
    "2-3": 1.15,  # Quality 3 typically 15% more expensive than Quality 2  
    "3-4": 1.3,   # Quality 4 typically 30% more expensive than Quality 3
    "4-5": 1.5    # Quality 5 typically 50% more expensive than Quality 4
}

QUALITY_ANOMALY_THRESHOLD = 0.3  # 30% deviation from expected ratios

# Output Settings
MAX_SUSPICIOUS_CASES_DISPLAY = 20
CONFIDENCE_DECIMAL_PLACES = 3
PRICE_DECIMAL_PLACES = 0

# Logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"