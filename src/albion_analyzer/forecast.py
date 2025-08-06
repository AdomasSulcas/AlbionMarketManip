from statsmodels.tsa.arima.model import ARIMA
import numpy as np
from typing import Dict, Tuple, Union
import pandas as pd

FORECAST_CACHE: Dict[Tuple[str, str, int], ARIMA] = {}

def expected_price(item: str, city: str, date: pd.Timestamp, df_all: pd.DataFrame, 
                  quality: int = 1, alpha: float = 0.7) -> float:
    """
    Generate expected price forecast using hybrid ARIMA and peer median approach.
    
    Combines time series forecasting (ARIMA) with peer group analysis to predict
    expected item prices. Uses weighted average of ARIMA forecast and peer median
    to balance historical trends with current market conditions across similar items.
    
    Args:
        item: Albion Online item name (e.g., "T4_2H_AXE")
        city: City name for price forecast  
        date: Date for forecast generation
        df_all: Historical price dataset with required columns
        quality: Item quality level (1-5, default: 1)
        alpha: Weight for ARIMA vs peer median (0.7 = 70% ARIMA, 30% peer)
        
    Returns:
        Expected price in silver as float value
    """
    key = (item, city, quality)
    sub = df_all[(df_all["item"] == item) &
                 (df_all["city"] == city) &
                 (df_all.get("quality", 1) == quality)].sort_values("timestamp")

    if len(sub) < 8:                       # not enough history
        peer = median_peer(date, item, df_all, quality)
        return np.exp(peer)

    # Fit or retrieve cached ARIMA
    if key not in FORECAST_CACHE:
        y = sub.set_index("timestamp")["log_price"].asfreq("D")
        model = ARIMA(y, order=(1, 1, 1)).fit(method="statespace")
        FORECAST_CACHE[key] = model
    else:
        model = FORECAST_CACHE[key]

    # 1-step-ahead forecast
    arima_mu = model.forecast().iloc[0]
    peer_mu  = median_peer(date, item, df_all, quality)

    blended = alpha * arima_mu + (1 - alpha) * peer_mu
    return np.exp(blended)

def median_peer(date, item, df_all, quality=1):
    tier = int(item[1])               # "T4_..." -> 4
    slot = slot_from_item(item)        # "AXE", "BAG", ...
    peers = df_all[(df_all["tier"] == tier) &
                   (df_all["slot"] == slot) &
                   (df_all.get("quality", 1) == quality) &
                   (df_all["timestamp"].dt.floor("D") == date.date())]
    return peers["log_price"].median()

def slot_from_item(item: str) -> str:
    """
    T4_2H_AXE      -> AXE
    T5_HEAD_PLATE_SET1 -> HEAD
    """
    parts = item.split("_")
    if len(parts) < 3:
        return parts[-1]        # fallback
    # 2H, HEAD, CAPE, etc.
    return parts[2]
