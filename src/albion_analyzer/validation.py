"""
Validation and Performance Optimization Framework
Provides backtesting, false positive analysis, and performance benchmarking for market manipulation detection.
"""

import time
import json
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import logging
from dataclasses import dataclass, asdict
from collections import defaultdict

from .data_collector import AlbionDataCollector
from .models import ManipulationDetector, SimpleRuleBasedDetector
from .analysis import MarketAnalyzer
from .realtime_monitor import MarketAlert, SlidingWindowDetector
from .config import DEFAULT_ITEMS, DEFAULT_CITIES


@dataclass
class BacktestResult:
    """
    Results from backtesting manipulation detection on historical data.
    
    Contains comprehensive metrics for evaluating detection accuracy, performance,
    and system reliability across different market conditions and time periods.
    """
    test_period_start: datetime
    test_period_end: datetime
    total_records_tested: int
    true_positives: int
    false_positives: int
    true_negatives: int
    false_negatives: int
    precision: float
    recall: float
    f1_score: float
    accuracy: float
    avg_confidence: float
    detection_rate: float
    processing_time_seconds: float
    alerts_generated: int
    economic_contexts: Dict[str, int]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert backtest result to dictionary for serialization."""
        result = asdict(self)
        result['test_period_start'] = self.test_period_start.isoformat()
        result['test_period_end'] = self.test_period_end.isoformat()
        return result


@dataclass
class PerformanceMetrics:
    """
    System performance metrics for optimization analysis.
    
    Tracks computational efficiency, memory usage, API performance,
    and system scalability across different operational scenarios.
    """
    api_calls_per_minute: float
    avg_response_time_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    detection_latency_ms: float
    throughput_records_per_second: float
    error_rate: float
    cache_hit_rate: float
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert performance metrics to dictionary for analysis."""
        return asdict(self)


class ValidationFramework:
    """
    Comprehensive validation and testing framework for market manipulation detection.
    
    Provides backtesting against historical data, performance benchmarking, false positive
    analysis, and optimization tools to validate system accuracy and efficiency across
    different market conditions and operational scenarios.
    """

    def __init__(self, data_collector: AlbionDataCollector = None) -> None:
        """
        Initialize validation framework with data access and logging.
        
        Args:
            data_collector: AlbionDataCollector instance for data access (optional)
        """
        self.data_collector = data_collector or AlbionDataCollector()
        self.logger = logging.getLogger(__name__)
        
        # Track validation sessions for analysis
        self.validation_sessions = []
        self.performance_history = []

    def create_synthetic_manipulation_data(self, items: List[str], cities: List[str], 
                                         days_back: int = 30, manipulation_rate: float = 0.05) -> pd.DataFrame:
        """
        Generate synthetic manipulation cases within real historical data for testing.
        
        Creates controlled manipulation scenarios by artificially inflating prices
        in historical datasets, allowing for ground truth validation of detection
        algorithms against known manipulation cases.
        
        Args:
            items: Items to include in synthetic dataset
            cities: Cities to include in synthetic dataset
            days_back: Days of historical data to use as base
            manipulation_rate: Fraction of records to artificially manipulate
            
        Returns:
            DataFrame with real historical data plus synthetic manipulation cases
        """
        self.logger.info(f"Creating synthetic manipulation dataset with {manipulation_rate:.1%} manipulation rate")
        
        # Get real historical data as base
        df = self.data_collector.get_manipulation_training_data(items, cities, days_back)
        
        if df.empty:
            self.logger.warning("No historical data available for synthetic generation")
            return df
        
        # Add ground truth column
        df['is_manipulation'] = False
        df['manipulation_type'] = 'none'
        df['original_price'] = df['price'].copy()
        
        # Select random records for manipulation
        n_manipulate = int(len(df) * manipulation_rate)
        manipulation_indices = np.random.choice(df.index, size=n_manipulate, replace=False)
        
        for idx in manipulation_indices:
            row = df.loc[idx]
            manipulation_type = np.random.choice(['price_spike', 'sustained_inflation', 'buyout_cascade'])
            
            if manipulation_type == 'price_spike':
                # 50-200% price increase
                multiplier = np.random.uniform(1.5, 3.0)
                df.loc[idx, 'price'] = row['price'] * multiplier
                df.loc[idx, 'log_price'] = np.log(df.loc[idx, 'price'])
                
            elif manipulation_type == 'sustained_inflation':
                # 20-80% increase sustained over time
                multiplier = np.random.uniform(1.2, 1.8)
                # Apply to this and nearby time periods for same item/city
                same_item_mask = (
                    (df['item'] == row['item']) & 
                    (df['city'] == row['city']) &
                    (abs((df['timestamp'] - row['timestamp']).dt.total_seconds()) < 86400 * 3)  # 3 days
                )
                affected_indices = df[same_item_mask].index
                df.loc[affected_indices, 'price'] = df.loc[affected_indices, 'price'] * multiplier
                df.loc[affected_indices, 'log_price'] = np.log(df.loc[affected_indices, 'price'])
                df.loc[affected_indices, 'is_manipulation'] = True
                df.loc[affected_indices, 'manipulation_type'] = manipulation_type
                continue
                
            elif manipulation_type == 'buyout_cascade':
                # Gradual 30-120% increase over time
                multiplier = np.random.uniform(1.3, 2.2)
                df.loc[idx, 'price'] = row['price'] * multiplier
                df.loc[idx, 'log_price'] = np.log(df.loc[idx, 'price'])
            
            df.loc[idx, 'is_manipulation'] = True
            df.loc[idx, 'manipulation_type'] = manipulation_type
        
        self.logger.info(f"Generated {n_manipulate} synthetic manipulation cases across {len(items)} items")
        return df

    def run_backtest(self, items: List[str], cities: List[str], days_back: int = 30,
                    use_synthetic: bool = True, detection_method: str = 'ml') -> BacktestResult:
        """
        Execute comprehensive backtest of manipulation detection system.
        
        Tests detection accuracy against historical data with known manipulation cases,
        measuring precision, recall, F1 score, and other performance metrics to
        validate system effectiveness across different market conditions.
        
        Args:
            items: Items to test detection on
            cities: Cities to include in backtest
            days_back: Days of historical data to test against
            use_synthetic: Whether to use synthetic manipulation cases for ground truth
            detection_method: Detection method to test ('ml', 'rules', or 'ensemble')
            
        Returns:
            BacktestResult containing comprehensive validation metrics
        """
        test_start = datetime.now()
        self.logger.info(f"Starting backtest: {detection_method} method, {days_back} days, {len(items)} items")
        
        # Get test dataset
        if use_synthetic:
            df = self.create_synthetic_manipulation_data(items, cities, days_back, 0.05)
            has_ground_truth = True
        else:
            df = self.data_collector.get_manipulation_training_data(items, cities, days_back)
            has_ground_truth = False
            # For real data, we'll use high-confidence detections as proxy ground truth
            df['is_manipulation'] = False
        
        if df.empty:
            raise ValueError("No test data available for backtest")
        
        # Initialize detection system
        if detection_method == 'ml':
            detector = ManipulationDetector(contamination=0.05)
        elif detection_method == 'rules':
            detector = SimpleRuleBasedDetector()
        else:  # ensemble
            detector = ManipulationDetector(contamination=0.05)
        
        # Prepare features
        analyzer = MarketAnalyzer()
        features_df = analyzer._prepare_features(df)
        
        # Run detection
        detection_start = time.time()
        
        if detection_method == 'ml' or detection_method == 'ensemble':
            detector.fit(features_df)
            results = detector.predict(features_df)
            predictions = results['manipulation_detected']
            confidence_scores = results['confidence_score']
        else:
            results = detector.predict(features_df)
            predictions = results['manipulation_detected']
            confidence_scores = results['confidence_score']
        
        detection_time = time.time() - detection_start
        
        # Calculate metrics
        if has_ground_truth:
            ground_truth = df['is_manipulation'].values
            
            tp = np.sum((predictions == True) & (ground_truth == True))
            fp = np.sum((predictions == True) & (ground_truth == False))
            tn = np.sum((predictions == False) & (ground_truth == False))
            fn = np.sum((predictions == False) & (ground_truth == True))
            
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
            accuracy = (tp + tn) / len(predictions)
            
        else:
            # Without ground truth, use detection-based metrics
            tp, fp, tn, fn = 0, 0, 0, 0
            precision = recall = f1 = accuracy = 0.0
        
        # Compile economic contexts
        economic_contexts = defaultdict(int)
        for idx in df[predictions].index:
            context = 'unknown'  # Would need gold economics analysis
            economic_contexts[context] += 1
        
        result = BacktestResult(
            test_period_start=df['timestamp'].min(),
            test_period_end=df['timestamp'].max(),
            total_records_tested=len(df),
            true_positives=tp,
            false_positives=fp,
            true_negatives=tn,
            false_negatives=fn,
            precision=precision,
            recall=recall,
            f1_score=f1,
            accuracy=accuracy,
            avg_confidence=confidence_scores[predictions].mean() if np.any(predictions) else 0,
            detection_rate=np.sum(predictions) / len(predictions),
            processing_time_seconds=detection_time,
            alerts_generated=np.sum(predictions),
            economic_contexts=dict(economic_contexts)
        )
        
        test_duration = (datetime.now() - test_start).total_seconds()
        self.logger.info(f"Backtest completed in {test_duration:.1f}s: {result.alerts_generated} alerts, "
                        f"F1={result.f1_score:.3f}, Precision={result.precision:.3f}")
        
        self.validation_sessions.append(result)
        return result

    def analyze_false_positives(self, items: List[str], cities: List[str], 
                              days_back: int = 7) -> Dict[str, Any]:
        """
        Analyze false positive patterns to improve detection accuracy.
        
        Examines cases flagged as manipulation to identify common false positive
        patterns, economic conditions that cause false alarms, and detection
        parameters that could be tuned to reduce false positive rates.
        
        Args:
            items: Items to analyze for false positive patterns
            cities: Cities to include in false positive analysis
            days_back: Recent days to analyze for false positive trends
            
        Returns:
            Dictionary containing false positive analysis results and recommendations
        """
        self.logger.info(f"Analyzing false positive patterns for {len(items)} items over {days_back} days")
        
        # Get recent data and run detection
        df = self.data_collector.get_manipulation_training_data(items, cities, days_back)
        if df.empty:
            return {"error": "No data available for false positive analysis"}
        
        analyzer = MarketAnalyzer()
        features_df = analyzer._prepare_features(df)
        
        # Run both ML and rules-based detection for comparison
        ml_detector = ManipulationDetector(contamination=0.02)
        ml_detector.fit(features_df)
        ml_results = ml_detector.predict(features_df)
        
        rule_detector = SimpleRuleBasedDetector()
        rule_results = rule_detector.predict(features_df)
        
        # Find disagreements between methods
        ml_flags = ml_results['manipulation_detected']
        rule_flags = rule_results['manipulation_detected']
        
        ml_only = ml_flags & ~rule_flags
        rules_only = rule_flags & ~ml_flags
        both_methods = ml_flags & rule_flags
        
        # Analyze patterns in ML-only flags (likely false positives)
        fp_analysis = {
            'total_ml_detections': np.sum(ml_flags),
            'total_rule_detections': np.sum(rule_flags),
            'ml_only_detections': np.sum(ml_only),
            'rules_only_detections': np.sum(rules_only),
            'both_methods_detections': np.sum(both_methods),
            'agreement_rate': np.sum(both_methods) / max(np.sum(ml_flags | rule_flags), 1),
            'patterns': {}
        }
        
        # Analyze ML-only detections for patterns
        ml_only_cases = features_df[ml_only].copy()
        if len(ml_only_cases) > 0:
            fp_analysis['patterns'] = {
                'high_volatility_items': list(ml_only_cases.groupby('item')['z'].std().nlargest(3).index),
                'affected_cities': ml_only_cases['city'].value_counts().to_dict(),
                'avg_z_score': ml_only_cases['z'].abs().mean(),
                'avg_peer_deviation': ml_only_cases['peer_dev'].abs().mean(),
                'time_distribution': ml_only_cases.groupby(ml_only_cases['timestamp'].dt.hour).size().to_dict()
            }
        
        # Generate recommendations
        recommendations = []
        if fp_analysis['agreement_rate'] < 0.5:
            recommendations.append("Low agreement between ML and rules suggests parameter tuning needed")
        
        if fp_analysis['ml_only_detections'] > fp_analysis['rules_only_detections'] * 2:
            recommendations.append("ML detector may be too sensitive - consider higher contamination parameter")
        
        if len(fp_analysis['patterns'].get('high_volatility_items', [])) > 0:
            recommendations.append(f"Items with high volatility causing false positives: "
                                 f"{fp_analysis['patterns']['high_volatility_items']}")
        
        fp_analysis['recommendations'] = recommendations
        fp_analysis['analysis_date'] = datetime.now().isoformat()
        
        return fp_analysis

    def benchmark_performance(self, items: List[str], cities: List[str],
                            test_sizes: List[int] = None) -> Dict[str, PerformanceMetrics]:
        """
        Benchmark system performance across different data volumes and configurations.
        
        Tests computational efficiency, memory usage, and scalability by running
        detection algorithms on datasets of varying sizes and measuring performance
        metrics for optimization guidance.
        
        Args:
            items: Items to include in performance benchmark
            cities: Cities to include in performance benchmark
            test_sizes: List of record counts to test (default: [1000, 5000, 10000, 50000])
            
        Returns:
            Dictionary mapping test size to PerformanceMetrics for each benchmark
        """
        if test_sizes is None:
            test_sizes = [1000, 5000, 10000, 50000]
        
        self.logger.info(f"Running performance benchmarks for sizes: {test_sizes}")
        
        benchmark_results = {}
        
        # Get large dataset for sampling
        full_df = self.data_collector.get_manipulation_training_data(items, cities, 90)  # 90 days
        if full_df.empty:
            raise ValueError("No data available for performance benchmarking")
        
        for test_size in test_sizes:
            if len(full_df) < test_size:
                self.logger.warning(f"Dataset too small for test size {test_size}, skipping")
                continue
            
            # Sample data for this test size
            test_df = full_df.sample(n=min(test_size, len(full_df)), random_state=42)
            
            # Measure performance
            start_time = time.time()
            start_memory = self._get_memory_usage()
            
            # Run detection
            analyzer = MarketAnalyzer()
            features_df = analyzer._prepare_features(test_df)
            
            detector = ManipulationDetector(contamination=0.02)
            detector.fit(features_df)
            results = detector.predict(features_df)
            
            end_time = time.time()
            end_memory = self._get_memory_usage()
            
            # Calculate metrics
            processing_time = end_time - start_time
            throughput = test_size / processing_time if processing_time > 0 else 0
            memory_delta = end_memory - start_memory
            
            metrics = PerformanceMetrics(
                api_calls_per_minute=0,  # Would need API call tracking
                avg_response_time_ms=processing_time * 1000 / test_size,
                memory_usage_mb=memory_delta,
                cpu_usage_percent=0,  # Would need CPU monitoring
                detection_latency_ms=processing_time * 1000,
                throughput_records_per_second=throughput,
                error_rate=0,  # Would track from actual operations
                cache_hit_rate=0   # Would track from caching system
            )
            
            benchmark_results[f"{test_size}_records"] = metrics
            
            self.logger.info(f"Benchmark {test_size} records: {throughput:.1f} records/sec, "
                           f"{memory_delta:.1f}MB memory")
        
        self.performance_history.append({
            'timestamp': datetime.now().isoformat(),
            'test_type': 'benchmark',
            'results': {k: v.to_dict() for k, v in benchmark_results.items()}
        })
        
        return benchmark_results

    def optimize_thresholds(self, items: List[str], cities: List[str],
                          param_ranges: Dict[str, List[float]] = None) -> Dict[str, Any]:
        """
        Optimize detection thresholds using grid search and validation metrics.
        
        Tests different parameter combinations to find optimal thresholds that
        maximize detection accuracy while minimizing false positives across
        the specified items and cities.
        
        Args:
            items: Items to use for threshold optimization
            cities: Cities to use for threshold optimization
            param_ranges: Parameter ranges to test (default: contamination and z-score thresholds)
            
        Returns:
            Dictionary containing optimal parameters and performance comparison
        """
        if param_ranges is None:
            param_ranges = {
                'contamination': [0.01, 0.02, 0.03, 0.05, 0.1],
                'z_threshold': [1.5, 2.0, 2.5, 3.0],
                'peer_dev_threshold': [1.0, 1.5, 2.0, 2.5]
            }
        
        self.logger.info("Optimizing detection thresholds using grid search")
        
        # Get synthetic test data with ground truth
        test_df = self.create_synthetic_manipulation_data(items, cities, 30, 0.05)
        if test_df.empty:
            raise ValueError("No data available for threshold optimization")
        
        ground_truth = test_df['is_manipulation'].values
        best_score = 0
        best_params = {}
        results = []
        
        # Grid search over parameter combinations
        total_combinations = np.prod([len(values) for values in param_ranges.values()])
        combination_count = 0
        
        for contamination in param_ranges['contamination']:
            for z_threshold in param_ranges['z_threshold']:
                for peer_threshold in param_ranges['peer_dev_threshold']:
                    combination_count += 1
                    
                    try:
                        # Test this parameter combination
                        analyzer = MarketAnalyzer(contamination=contamination)
                        features_df = analyzer._prepare_features(test_df)
                        
                        # Apply thresholds for rule-based component
                        rule_detector = SimpleRuleBasedDetector(
                            z_threshold=z_threshold,
                            peer_dev_threshold=peer_threshold
                        )
                        rule_results = rule_detector.predict(features_df)
                        predictions = rule_results['manipulation_detected']
                        
                        # Calculate metrics
                        tp = np.sum((predictions == True) & (ground_truth == True))
                        fp = np.sum((predictions == True) & (ground_truth == False))
                        tn = np.sum((predictions == False) & (ground_truth == False))
                        fn = np.sum((predictions == False) & (ground_truth == True))
                        
                        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
                        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
                        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                        
                        result = {
                            'contamination': contamination,
                            'z_threshold': z_threshold,
                            'peer_dev_threshold': peer_threshold,
                            'f1_score': f1,
                            'precision': precision,
                            'recall': recall,
                            'true_positives': tp,
                            'false_positives': fp
                        }
                        
                        results.append(result)
                        
                        if f1 > best_score:
                            best_score = f1
                            best_params = {
                                'contamination': contamination,
                                'z_threshold': z_threshold,
                                'peer_dev_threshold': peer_threshold
                            }
                        
                        if combination_count % 10 == 0:
                            self.logger.info(f"Tested {combination_count}/{total_combinations} combinations, "
                                           f"best F1: {best_score:.3f}")
                    
                    except Exception as e:
                        self.logger.error(f"Error testing parameters {contamination}, {z_threshold}, {peer_threshold}: {e}")
        
        optimization_results = {
            'best_parameters': best_params,
            'best_f1_score': best_score,
            'total_combinations_tested': len(results),
            'parameter_performance': results,
            'optimization_date': datetime.now().isoformat()
        }
        
        self.logger.info(f"Threshold optimization complete. Best F1: {best_score:.3f} with params: {best_params}")
        return optimization_results

    def run_comprehensive_validation(self, test_items: List[str] = None,
                                   test_cities: List[str] = None) -> Dict[str, Any]:
        """
        Execute complete validation suite including backtesting, performance analysis, and optimization.
        
        Runs comprehensive testing across multiple validation approaches to provide
        complete system assessment and optimization recommendations.
        
        Args:
            test_items: Items for validation testing (default: subset of DEFAULT_ITEMS)
            test_cities: Cities for validation testing (default: subset of DEFAULT_CITIES)
            
        Returns:
            Dictionary containing all validation results and recommendations
        """
        if test_items is None:
            test_items = DEFAULT_ITEMS[:3]  # Limit scope for testing
        if test_cities is None:
            test_cities = DEFAULT_CITIES[:2]
        
        validation_start = datetime.now()
        self.logger.info(f"Starting comprehensive validation suite")
        
        comprehensive_results = {
            'validation_start': validation_start.isoformat(),
            'test_configuration': {
                'items': test_items,
                'cities': test_cities
            }
        }
        
        try:
            # 1. Backtesting
            self.logger.info("Running backtests...")
            ml_backtest = self.run_backtest(test_items, test_cities, days_back=30, detection_method='ml')
            rules_backtest = self.run_backtest(test_items, test_cities, days_back=30, detection_method='rules')
            
            comprehensive_results['backtests'] = {
                'ml_method': ml_backtest.to_dict(),
                'rules_method': rules_backtest.to_dict()
            }
            
            # 2. False positive analysis
            self.logger.info("Analyzing false positives...")
            fp_analysis = self.analyze_false_positives(test_items, test_cities, days_back=7)
            comprehensive_results['false_positive_analysis'] = fp_analysis
            
            # 3. Performance benchmarking
            self.logger.info("Running performance benchmarks...")
            performance_results = self.benchmark_performance(test_items, test_cities, [1000, 5000])
            comprehensive_results['performance_benchmarks'] = {
                k: v.to_dict() for k, v in performance_results.items()
            }
            
            # 4. Threshold optimization
            self.logger.info("Optimizing thresholds...")
            optimization_results = self.optimize_thresholds(test_items, test_cities)
            comprehensive_results['threshold_optimization'] = optimization_results
            
        except Exception as e:
            self.logger.error(f"Comprehensive validation error: {e}")
            comprehensive_results['error'] = str(e)
        
        validation_duration = (datetime.now() - validation_start).total_seconds()
        comprehensive_results['validation_duration_seconds'] = validation_duration
        comprehensive_results['validation_complete'] = datetime.now().isoformat()
        
        # Generate summary recommendations
        recommendations = []
        
        if 'backtests' in comprehensive_results:
            ml_f1 = comprehensive_results['backtests']['ml_method']['f1_score']
            rules_f1 = comprehensive_results['backtests']['rules_method']['f1_score']
            
            if ml_f1 > rules_f1:
                recommendations.append(f"ML method outperforms rules-based (F1: {ml_f1:.3f} vs {rules_f1:.3f})")
            else:
                recommendations.append(f"Rules-based method competitive with ML (F1: {rules_f1:.3f} vs {ml_f1:.3f})")
        
        if 'false_positive_analysis' in comprehensive_results:
            agreement_rate = comprehensive_results['false_positive_analysis'].get('agreement_rate', 0)
            if agreement_rate < 0.6:
                recommendations.append(f"Low method agreement ({agreement_rate:.1%}) suggests parameter tuning needed")
        
        if 'threshold_optimization' in comprehensive_results:
            best_f1 = comprehensive_results['threshold_optimization']['best_f1_score']
            recommendations.append(f"Optimized thresholds achieve F1 score of {best_f1:.3f}")
        
        comprehensive_results['recommendations'] = recommendations
        
        self.logger.info(f"Comprehensive validation completed in {validation_duration:.1f} seconds")
        return comprehensive_results

    def export_validation_report(self, results: Dict[str, Any], 
                               filename: str = None) -> str:
        """
        Export validation results to JSON report file.
        
        Args:
            results: Validation results dictionary to export
            filename: Output filename (default: timestamped filename)
            
        Returns:
            Path to exported report file
        """
        if filename is None:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"validation_report_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.logger.info(f"Validation report exported to {filename}")
        return filename

    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB (simplified implementation)."""
        try:
            import psutil
            process = psutil.Process()
            return process.memory_info().rss / 1024 / 1024
        except ImportError:
            return 0.0  # Return 0 if psutil not available