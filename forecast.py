from statsmodels.tsa.arima.model import ARIMA
import numpy as np

FORECAST_CACHE = {}   # (item, city) -> last fitted model

def expected_price(item, city, date, df_all, alpha=0.7):
    """
    alpha = weight on ARIMA vs peer-median.
    0.7 means 70 % ARIMA, 30 % peer.
    """
    key = (item, city)
    sub = df_all[(df_all["item"] == item) &
                 (df_all["city"] == city)].sort_values("timestamp")

    if len(sub) < 8:                       # not enough history
        peer = median_peer(date, item, df_all)
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
    peer_mu  = median_peer(date, item, df_all)

    blended = alpha * arima_mu + (1 - alpha) * peer_mu
    return np.exp(blended)

def median_peer(date, item, df_all):
    tier = int(item[1])               # "T4_..." -> 4
    slot = slot_from_item(item)        # "AXE", "BAG", ...
    peers = df_all[(df_all["tier"] == tier) &
                   (df_all["slot"] == slot) &
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
