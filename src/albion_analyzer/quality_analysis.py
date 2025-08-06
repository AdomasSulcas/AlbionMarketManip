import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from .order_book import fetch_current_prices, calculate_bid_ask_features

def calculate_quality_correlation(item, cities, qualities=None):
    """
    Calculate price correlations across different quality levels
    """
    if qualities is None:
        qualities = [1, 2, 3, 4, 5]
    
    prices_df = fetch_current_prices(item, cities, qualities)
    if prices_df.empty:
        return pd.DataFrame()
    
    correlations = []
    
    for city in cities:
        city_data = prices_df[prices_df["city"] == city]
        if len(city_data) < 2:
            continue
        
        # Calculate correlations between adjacent quality levels
        for i in range(len(qualities) - 1):
            q1, q2 = qualities[i], qualities[i + 1]
            
            q1_data = city_data[city_data["quality"] == q1]
            q2_data = city_data[city_data["quality"] == q2]
            
            if not q1_data.empty and not q2_data.empty:
                q1_price = q1_data["sell_price_min"].iloc[0] if q1_data["sell_price_min"].iloc[0] > 0 else q1_data["buy_price_max"].iloc[0]
                q2_price = q2_data["sell_price_min"].iloc[0] if q2_data["sell_price_min"].iloc[0] > 0 else q2_data["buy_price_max"].iloc[0]
                
                if q1_price > 0 and q2_price > 0:
                    price_ratio = q2_price / q1_price
                    correlations.append({
                        "item": item,
                        "city": city,
                        "quality_pair": f"{q1}-{q2}",
                        "q1_price": q1_price,
                        "q2_price": q2_price,
                        "price_ratio": price_ratio
                    })
    
    return pd.DataFrame(correlations)

def detect_quality_anomalies(item, cities, qualities=None, historical_ratios=None):
    """
    Detect anomalies in quality price relationships
    """
    current_ratios = calculate_quality_correlation(item, cities, qualities)
    if current_ratios.empty:
        return pd.DataFrame()
    
    anomalies = []
    
    # Expected quality multipliers (rough estimates)
    expected_multipliers = {
        "1-2": 1.2,  # Quality 2 typically 20% more expensive than Quality 1
        "2-3": 1.15,
        "3-4": 1.3,
        "4-5": 1.5
    }
    
    for _, row in current_ratios.iterrows():
        quality_pair = row["quality_pair"]
        actual_ratio = row["price_ratio"]
        expected_ratio = expected_multipliers.get(quality_pair, 1.2)
        
        # Calculate deviation from expected ratio
        ratio_deviation = abs(actual_ratio - expected_ratio) / expected_ratio
        
        anomalies.append({
            "item": row["item"],
            "city": row["city"],
            "quality_pair": quality_pair,
            "actual_ratio": actual_ratio,
            "expected_ratio": expected_ratio,
            "ratio_deviation": ratio_deviation,
            "is_anomaly": ratio_deviation > 0.3  # 30% deviation threshold
        })
    
    return pd.DataFrame(anomalies)

def cross_quality_manipulation_score(df_prices):
    """
    Calculate manipulation score based on cross-quality price relationships
    """
    if df_prices.empty:
        return 0
    
    scores = []
    
    # Group by item and city
    for (item_id, city), group in df_prices.groupby(["item_id", "city"]):
        if len(group) < 2:
            continue
        
        group = group.sort_values("quality")
        
        # Check for price inversions (higher quality cheaper than lower quality)
        inversions = 0
        total_pairs = 0
        
        for i in range(len(group) - 1):
            price1 = group.iloc[i]["sell_price_min"] or group.iloc[i]["buy_price_max"]
            price2 = group.iloc[i + 1]["sell_price_min"] or group.iloc[i + 1]["buy_price_max"]
            
            if price1 > 0 and price2 > 0:
                total_pairs += 1
                if price2 < price1:  # Higher quality cheaper than lower
                    inversions += 1
        
        if total_pairs > 0:
            inversion_rate = inversions / total_pairs
            scores.append({
                "item": item_id,
                "city": city,
                "inversion_rate": inversion_rate,
                "manipulation_score": inversion_rate * 100
            })
    
    return pd.DataFrame(scores)

def analyze_quality_spread_patterns(item, cities, qualities=None):
    """
    Analyze bid-ask spread patterns across quality levels
    """
    if qualities is None:
        qualities = [1, 2, 3, 4, 5]
    
    prices_df = fetch_current_prices(item, cities, qualities)
    if prices_df.empty:
        return pd.DataFrame()
    
    bid_ask_features = calculate_bid_ask_features(prices_df)
    
    spread_analysis = []
    
    for city in cities:
        city_data = bid_ask_features[bid_ask_features["city"] == city]
        if city_data.empty:
            continue
        
        # Analyze spread patterns
        spreads = city_data["bid_ask_spread_pct"].values
        qualities_present = city_data["quality"].values
        
        if len(spreads) > 1:
            spread_variance = np.var(spreads)
            max_spread = np.max(spreads)
            min_spread = np.min(spreads)
            spread_range = max_spread - min_spread
            
            spread_analysis.append({
                "item": item,
                "city": city,
                "spread_variance": spread_variance,
                "spread_range": spread_range,
                "max_spread": max_spread,
                "min_spread": min_spread,
                "qualities_analyzed": len(qualities_present)
            })
    
    return pd.DataFrame(spread_analysis)