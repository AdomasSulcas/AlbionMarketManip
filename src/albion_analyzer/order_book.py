import requests
import pandas as pd

def order_book_features(df_sell):
    """
    df_sell: DataFrame with columns price, qty
    returns: dict with wall_thickness and wall_concentration
    """
    if df_sell.empty:
        return {"wall_thickness": 0, "wall_concentration": 0}

    cheapest = df_sell["price"].min()
    band     = cheapest * 1.05
    wall     = df_sell[df_sell["price"] <= band]

    wall_thickness = wall["qty"].sum()
    top_price_qty  = wall.groupby("price")["qty"].sum().max()
    wall_concentration = top_price_qty / wall_thickness if wall_thickness else 0

    return {"wall_thickness": wall_thickness,
            "wall_concentration": wall_concentration}

def fetch_sell_orders(item, city, quality=1, url="https://europe.albion-online-data.com/api/v2/stats/orders/{}.json"):
    params = {"locations": city, "qualities": quality}
    r = requests.get(url.format(item), params=params)
    if r.status_code != 200:
        return pd.DataFrame()                        # graceful fallback
    orders = pd.json_normalize(r.json())
    if orders.empty:
        return pd.DataFrame()
    sells  = orders[orders["buy_price_max"] == 0]  # sell orders only
    if sells.empty:
        return pd.DataFrame()
    sells["price"] = sells["sell_price_min"]
    sells["qty"]   = sells["sell_price_min_date"]
    sells["quality"] = quality
    return sells[["price", "qty", "quality"]].astype({"price": int, "qty": int, "quality": int})

def fetch_current_prices(item, cities, qualities=None, url="https://europe.albion-online-data.com/api/v2/stats/prices/{}.json"):
    """
    Fetch current buy/sell prices for multiple cities and qualities
    """
    if qualities is None:
        qualities = [1]
    
    cities_str = ",".join(cities)
    qualities_str = ",".join(map(str, qualities))
    params = {"locations": cities_str, "qualities": qualities_str}
    
    r = requests.get(url.format(item), params=params)
    if r.status_code != 200:
        return pd.DataFrame()
    
    data = r.json()
    if not data:
        return pd.DataFrame()
    
    df = pd.json_normalize(data)
    df["city"] = df["city"]
    df["quality"] = df["quality"]
    return df

def calculate_bid_ask_features(prices_df):
    """
    Calculate bid-ask spread features from current prices DataFrame
    """
    features = []
    
    for _, row in prices_df.iterrows():
        sell_min = row.get("sell_price_min", 0)
        buy_max = row.get("buy_price_max", 0)
        
        if sell_min > 0 and buy_max > 0:
            spread = sell_min - buy_max
            spread_pct = spread / ((sell_min + buy_max) / 2) * 100
            mid_price = (sell_min + buy_max) / 2
        else:
            spread = 0
            spread_pct = 0
            mid_price = sell_min if sell_min > 0 else buy_max
        
        features.append({
            "item": row.get("item_id", ""),
            "city": row.get("city", ""),
            "quality": row.get("quality", 1),
            "sell_price_min": sell_min,
            "buy_price_max": buy_max,
            "bid_ask_spread": spread,
            "bid_ask_spread_pct": spread_pct,
            "mid_price": mid_price,
            "sell_timestamp": row.get("sell_price_min_date", ""),
            "buy_timestamp": row.get("buy_price_max_date", "")
        })
    
    return pd.DataFrame(features)

def fetch_multi_quality_orders(item, city, qualities=None):
    """
    Fetch order book data for multiple qualities
    """
    if qualities is None:
        qualities = [1, 2, 3, 4, 5]
    
    all_orders = []
    for quality in qualities:
        orders = fetch_sell_orders(item, city, quality)
        if not orders.empty:
            all_orders.append(orders)
    
    if all_orders:
        return pd.concat(all_orders, ignore_index=True)
    return pd.DataFrame()
