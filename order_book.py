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

def fetch_sell_orders(item, city, url="https://europe.albion-online-data.com/api/v2/stats/orders/{}.json"):
    params = {"locations": city, "qualities": 1}   # normal quality only
    r = requests.get(url.format(item), params=params)
    if r.status_code != 200:
        return pd.DataFrame()                        # graceful fallback
    orders = pd.json_normalize(r.json())
    sells  = orders[orders["buy_price_max"] == 0]  # sell orders only
    sells["price"] = sells["sell_price_min"]
    sells["qty"]   = sells["sell_price_min_date"]
    return sells[["price", "qty"]].astype(int)
