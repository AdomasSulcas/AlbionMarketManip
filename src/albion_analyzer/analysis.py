"""
Market Analysis Module
Core analysis logic extracted from Jupyter notebook
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from sklearn.ensemble import IsolationForest
from .data_collector import AlbionDataCollector
from .models import ManipulationDetector, SimpleRuleBasedDetector
from .order_book import calculate_bid_ask_features
from .quality_analysis import detect_quality_anomalies, analyze_quality_spread_patterns
from .forecast import expected_price
from .gold_economics import GoldEconomicsAnalyzer

class MarketAnalyzer:
    """
    Main market analysis class - replaces Jupyter notebook functionality
    """
    
    def __init__(self, contamination: float = 0.01):
        self.contamination = contamination
        self.data_collector = AlbionDataCollector()
        self.ml_detector = ManipulationDetector(contamination)
        self.rule_detector = SimpleRuleBasedDetector()
        self.gold_analyzer = GoldEconomicsAnalyzer()
        self.analysis_results = {}
        
    def analyze_market(self, items: List[str], cities: List[str], 
                      days_back: int = 30, use_ml: bool = True) -> Dict[str, any]:
        """
        Execute complete market manipulation analysis pipeline with gold economic context.
        
        Performs comprehensive analysis combining historical data collection, feature engineering,
        machine learning detection, quality analysis, price forecasting, and economic filtering
        to identify genuine market manipulation cases while minimizing false positives from
        server-wide economic trends.
        
        Analysis pipeline:
        1. Collect historical price data and current market state
        2. Engineer features with gold-based purchasing power adjustments  
        3. Detect manipulation using ML ensemble or rule-based methods
        4. Analyze cross-quality pricing relationships for anomalies
        5. Generate price forecasts using ARIMA models
        6. Filter results using economic regime context
        
        Args:
            items: List of Albion Online item names to analyze (e.g., ["T4_2H_AXE", "T5_BAG"])
            cities: List of city names for analysis (e.g., ["Caerleon", "Lymhurst"])  
            days_back: Number of days of historical data to analyze (default: 30)
            use_ml: Whether to use ML ensemble (True) or rule-based detection (False)
            
        Returns:
            Dictionary containing comprehensive analysis results:
            - summary (dict): High-level statistics and regime info
            - suspicious_cases (list): All detected manipulation cases
            - filtered_suspicious_cases (list): Cases surviving economic filtering
            - item_summary (list): Per-item manipulation statistics
            - quality_anomalies (dict): Cross-quality pricing anomalies
            - forecasts (dict): Price predictions for item/city combinations
            - gold_context (dict): Economic regime analysis
            - full_dataset (DataFrame): Complete processed dataset
            
            Returns {"error": str} if data collection fails.
        
        Raises:
            No exceptions raised directly, but logs errors for API failures and analysis issues.
        """
        print("=== Albion Online Market Analysis ===")
        print(f"Analyzing {len(items)} items across {len(cities)} cities")
        print(f"Historical data: {days_back} days")
        print()
        
        # 1. Data Collection
        print("1. Collecting market data...")
        df = self.data_collector.get_manipulation_training_data(items, cities, days_back)
        
        if df.empty:
            return {"error": "No data available for analysis"}
        
        # 2. Feature Engineering with Gold Context
        print("2. Engineering features with gold economics...")
        features_df = self._prepare_features(df)
        features_df = self.gold_analyzer.add_gold_features(features_df)
        gold_context = self.gold_analyzer.detect_economic_regime(
            self.gold_analyzer.fetch_gold_prices()
        )
        
        # 3. Manipulation Detection
        print("3. Running manipulation detection...")
        if use_ml:
            print("   Training ML models...")
            self.ml_detector.fit(features_df)
            ml_results = self.ml_detector.predict(features_df)
            manipulation_detected = ml_results['manipulation_detected']
            confidence_scores = ml_results['confidence_score']
        else:
            print("   Using rule-based detection...")
            rule_results = self.rule_detector.predict(features_df)
            manipulation_detected = rule_results['manipulation_detected']
            confidence_scores = rule_results['confidence_score']
        
        # 4. Quality Analysis (if multi-quality data available)
        print("4. Analyzing quality relationships...")
        quality_anomalies = self._analyze_quality_patterns(items, cities)
        
        # 5. Price Forecasting
        print("5. Generating price forecasts...")
        forecasts = self._generate_forecasts(df)
        
        # 6. Filter by Economic Context and Compile Results
        suspicious_df = features_df[manipulation_detected].copy()
        suspicious_df['confidence_score'] = confidence_scores[manipulation_detected]
        
        filtered_suspicious = self.gold_analyzer.filter_manipulation_vs_inflation(
            suspicious_df, gold_context
        )
        
        results = self._compile_results(
            features_df, manipulation_detected, confidence_scores, 
            quality_anomalies, forecasts, gold_context, filtered_suspicious
        )
        
        self.analysis_results = results
        
        print(f"Analysis complete! Found {results['summary']['total_manipulation_cases']} manipulation cases")
        return results
    
    def _prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare features for manipulation detection
        """
        features_df = df.copy()
        
        # Ensure required features exist
        if 'z' not in features_df.columns:
            # Add rolling z-scores if not present
            def rolling_z(series, window=7):
                roll = series.rolling(window, min_periods=3)
                return (series - roll.mean()) / roll.std()
            
            features_df["z"] = features_df.groupby(["item", "city"])["log_price"].transform(rolling_z)
        
        if 'peer_dev' not in features_df.columns:
            # Add peer deviations if not present
            peer_median = features_df.groupby(["timestamp", "tier", "slot"])["log_price"].transform("median")
            features_df["peer_dev"] = features_df["log_price"] - peer_median
        
        # Add absolute values
        features_df["abs_z"] = features_df["z"].abs()
        features_df["abs_peer_dev"] = features_df["peer_dev"].abs()
        
        return features_df
    
    def _analyze_quality_patterns(self, items: List[str], cities: List[str]) -> Dict[str, Union[int, List[str], List[Dict]]]:
        """
        Analyze cross-quality pricing relationships to detect quality-based manipulation.
        
        Examines pricing patterns across different item quality levels (1-5) to identify
        anomalies that suggest quality-specific manipulation. This includes price inversions
        where higher quality items are cheaper than lower quality items, and unusual
        quality price ratios that deviate from expected patterns.
        
        Args:
            items: List of item names to analyze for quality anomalies
            cities: List of cities to check for cross-quality patterns
            
        Returns:
            Dictionary containing quality analysis results:
            - anomalies_detected (int): Total number of quality anomalies found
            - items_with_anomalies (list): Item names showing quality anomalies
            - details (list): Detailed anomaly information for each affected item
        """
        quality_results = {
            'anomalies_detected': 0,
            'items_with_anomalies': [],
            'details': []
        }
        
        try:
            for item in items:
                anomalies = detect_quality_anomalies(item, cities, [1, 2, 3])
                if not anomalies.empty:
                    anomaly_count = anomalies['is_anomaly'].sum()
                    if anomaly_count > 0:
                        quality_results['anomalies_detected'] += anomaly_count
                        quality_results['items_with_anomalies'].append(item)
                        quality_results['details'].append({
                            'item': item,
                            'anomalies': anomalies[anomalies['is_anomaly']].to_dict('records')
                        })
        except Exception as e:
            print(f"Quality analysis error: {e}")
        
        return quality_results
    
    def _generate_forecasts(self, df: pd.DataFrame) -> Dict:
        """
        Generate price forecasts for detected manipulation cases
        """
        forecasts = {}
        
        try:
            # Get unique item/city combinations
            combinations = df[['item', 'city']].drop_duplicates()
            
            for _, row in combinations.iterrows():
                item, city = row['item'], row['city']
                
                # Get most recent date
                item_data = df[(df['item'] == item) & (df['city'] == city)]
                if not item_data.empty:
                    latest_date = item_data['timestamp'].max()
                    
                    try:
                        forecast = expected_price(item, city, latest_date, df)
                        forecasts[f"{item}_{city}"] = {
                            'item': item,
                            'city': city,
                            'expected_price': forecast,
                            'forecast_date': latest_date.isoformat()
                        }
                    except Exception as e:
                        print(f"Forecast error for {item} in {city}: {e}")
        
        except Exception as e:
            print(f"Forecasting error: {e}")
        
        return forecasts
    
    def _compile_results(self, df: pd.DataFrame, manipulation_detected: pd.Series,
                        confidence_scores: pd.Series, quality_anomalies: Dict,
                        forecasts: Dict, gold_context: Dict = None, 
                        filtered_suspicious: pd.DataFrame = None) -> Dict:
        """
        Compile all analysis results into structured output
        """
        # Add predictions to dataframe
        df = df.copy()
        df['manipulation_detected'] = manipulation_detected
        df['confidence_score'] = confidence_scores
        
        # Extract suspicious cases
        suspicious_cases = df[df['manipulation_detected']].copy()
        
        # Calculate summary statistics
        total_records = len(df)
        manipulation_cases = len(suspicious_cases)
        manipulation_rate = manipulation_cases / total_records if total_records > 0 else 0
        
        # Group by item for item-level summary
        item_summary = []
        for item in df['item'].unique():
            item_data = df[df['item'] == item]
            item_manipulation = item_data['manipulation_detected'].sum()
            avg_confidence = item_data[item_data['manipulation_detected']]['confidence_score'].mean()
            
            item_summary.append({
                'item': item,
                'total_records': len(item_data),
                'manipulation_cases': item_manipulation,
                'manipulation_rate': item_manipulation / len(item_data),
                'avg_confidence': avg_confidence if not pd.isna(avg_confidence) else 0
            })
        
        return {
            'summary': {
                'total_records': total_records,
                'total_manipulation_cases': manipulation_cases,
                'manipulation_rate': manipulation_rate,
                'quality_anomalies': quality_anomalies['anomalies_detected'],
                'items_analyzed': len(df['item'].unique()),
                'cities_analyzed': len(df['city'].unique()),
                'economic_regime': gold_context['regime'] if gold_context else 'unknown',
                'gold_confidence': gold_context['confidence'] if gold_context else 0.0,
                'filtered_cases': len(filtered_suspicious) if filtered_suspicious is not None else manipulation_cases
            },
            'suspicious_cases': suspicious_cases.to_dict('records'),
            'filtered_suspicious_cases': filtered_suspicious.to_dict('records') if filtered_suspicious is not None else [],
            'item_summary': item_summary,
            'quality_anomalies': quality_anomalies,
            'forecasts': forecasts,
            'gold_context': gold_context or {},
            'full_dataset': df
        }
    
    def print_analysis_report(self, results: Optional[Dict[str, any]] = None) -> None:
        """
        Print comprehensive formatted analysis report to console.
        
        Displays detailed analysis results including manipulation statistics, economic context,
        suspicious cases, quality anomalies, and item-level breakdowns in human-readable format.
        Automatically uses most recent analysis results if no specific results provided.
        
        Report sections:
        - Summary statistics with economic regime information
        - Item-level manipulation rates and confidence scores  
        - Detailed suspicious cases with confidence rankings
        - Quality pricing anomalies across item tiers
        - Economic context and filtering effects
        
        Args:
            results: Analysis results dictionary from analyze_market(). If None,
                    uses self.analysis_results from most recent analysis.
                    
        Returns:
            None. Prints formatted report to console.
        """
        if results is None:
            results = self.analysis_results
        
        if not results:
            print("No analysis results available")
            return
        
        print("\n=== MARKET ANALYSIS REPORT ===")
        
        # Summary
        summary = results['summary']
        print(f"Total Records Analyzed: {summary['total_records']:,}")
        print(f"Manipulation Cases Detected: {summary['total_manipulation_cases']}")
        print(f"Overall Manipulation Rate: {summary['manipulation_rate']:.1%}")
        print(f"Quality Anomalies: {summary['quality_anomalies']}")
        print()
        
        # Item-level breakdown
        print("=== ITEM ANALYSIS ===")
        for item_data in results['item_summary']:
            if item_data['manipulation_cases'] > 0:
                print(f"{item_data['item']}:")
                print(f"  Manipulation cases: {item_data['manipulation_cases']} / {item_data['total_records']}")
                print(f"  Rate: {item_data['manipulation_rate']:.1%}")
                print(f"  Avg confidence: {item_data['avg_confidence']:.3f}")
        
        # Suspicious cases detail
        if results['suspicious_cases']:
            print("\n=== SUSPICIOUS CASES ===")
            suspicious_df = pd.DataFrame(results['suspicious_cases'])
            
            # Show top 10 by confidence
            top_cases = suspicious_df.nlargest(10, 'confidence_score')
            for _, case in top_cases.iterrows():
                print(f"{case['item']} in {case['city']} on {case['timestamp']}")
                print(f"  Price: {case['price']:,} (confidence: {case['confidence_score']:.3f})")
                if 'expected_price' in case:
                    print(f"  Expected: {case.get('expected_price', 'N/A'):,}")
        
        # Quality anomalies
        if results['quality_anomalies']['items_with_anomalies']:
            print(f"\n=== QUALITY ANOMALIES ===")
            for item in results['quality_anomalies']['items_with_anomalies']:
                print(f"Quality pricing anomalies detected in: {item}")
        
        print("\n=== ANALYSIS COMPLETE ===")

def run_quick_analysis(items: List[str], cities: List[str], days_back: int = 30) -> Dict[str, any]:
    """
    Execute quick market manipulation analysis with automatic reporting.
    
    Convenience function that performs complete market analysis and automatically
    prints results to console. Ideal for command-line usage and quick investigations
    of specific items or market conditions.
    
    Args:
        items: List of Albion Online item names to analyze
        cities: List of city names for market analysis
        days_back: Number of days of historical data to analyze
        
    Returns:
        Complete analysis results dictionary from MarketAnalyzer.analyze_market()
        containing all detection results, statistics, and economic context.
    """
    analyzer = MarketAnalyzer()
    results = analyzer.analyze_market(items, cities, days_back)
    analyzer.print_analysis_report(results)
    return results