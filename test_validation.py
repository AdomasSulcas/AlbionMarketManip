#!/usr/bin/env python3
"""
Validation Testing Script
Comprehensive testing script for Phase 4 validation and performance optimization.
"""

import sys
import logging
import asyncio
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from albion_analyzer.validation import ValidationFramework
from albion_analyzer.optimization import OptimizedDataCollector, MemoryOptimizer, performance_monitor
from albion_analyzer.config import DEFAULT_ITEMS, DEFAULT_CITIES


def setup_test_logging():
    """Setup logging for test execution."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('validation_test.log')
        ]
    )
    return logging.getLogger(__name__)


@performance_monitor
def test_synthetic_data_generation():
    """Test synthetic manipulation data generation."""
    logger = logging.getLogger(__name__)
    logger.info("Testing synthetic data generation...")
    
    framework = ValidationFramework()
    test_items = DEFAULT_ITEMS[:2]  # Limit for testing
    test_cities = DEFAULT_CITIES[:2]
    
    synthetic_df = framework.create_synthetic_manipulation_data(
        test_items, test_cities, days_back=7, manipulation_rate=0.1
    )
    
    if not synthetic_df.empty:
        manipulation_count = synthetic_df['is_manipulation'].sum()
        total_records = len(synthetic_df)
        actual_rate = manipulation_count / total_records
        
        logger.info(f"Generated {total_records} records with {manipulation_count} manipulations ({actual_rate:.1%})")
        logger.info("Synthetic data generation: PASSED")
        return True
    else:
        logger.error("Synthetic data generation failed - no data returned")
        return False


@performance_monitor
def test_backtesting():
    """Test backtesting framework."""
    logger = logging.getLogger(__name__)
    logger.info("Testing backtesting framework...")
    
    framework = ValidationFramework()
    test_items = DEFAULT_ITEMS[:2]
    test_cities = DEFAULT_CITIES[:1]
    
    try:
        # Test ML method backtest
        ml_result = framework.run_backtest(
            test_items, test_cities, days_back=7, 
            use_synthetic=True, detection_method='ml'
        )
        
        logger.info(f"ML Backtest Results:")
        logger.info(f"  F1 Score: {ml_result.f1_score:.3f}")
        logger.info(f"  Precision: {ml_result.precision:.3f}")
        logger.info(f"  Recall: {ml_result.recall:.3f}")
        logger.info(f"  Processing time: {ml_result.processing_time_seconds:.2f}s")
        
        # Test rules method backtest
        rules_result = framework.run_backtest(
            test_items, test_cities, days_back=7,
            use_synthetic=True, detection_method='rules'
        )
        
        logger.info(f"Rules Backtest Results:")
        logger.info(f"  F1 Score: {rules_result.f1_score:.3f}")
        logger.info(f"  Precision: {rules_result.precision:.3f}")
        logger.info(f"  Recall: {rules_result.recall:.3f}")
        
        logger.info("Backtesting framework: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Backtesting failed: {e}")
        return False


@performance_monitor
def test_false_positive_analysis():
    """Test false positive analysis."""
    logger = logging.getLogger(__name__)
    logger.info("Testing false positive analysis...")
    
    framework = ValidationFramework()
    test_items = DEFAULT_ITEMS[:2]
    test_cities = DEFAULT_CITIES[:1]
    
    try:
        fp_results = framework.analyze_false_positives(test_items, test_cities, days_back=7)
        
        if 'error' not in fp_results:
            logger.info(f"False Positive Analysis Results:")
            logger.info(f"  Total ML detections: {fp_results.get('total_ml_detections', 0)}")
            logger.info(f"  Total rule detections: {fp_results.get('total_rule_detections', 0)}")
            logger.info(f"  Agreement rate: {fp_results.get('agreement_rate', 0):.1%}")
            logger.info(f"  Recommendations: {len(fp_results.get('recommendations', []))}")
            
            logger.info("False positive analysis: PASSED")
            return True
        else:
            logger.warning(f"False positive analysis returned error: {fp_results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"False positive analysis failed: {e}")
        return False


@performance_monitor
def test_performance_optimization():
    """Test performance optimization features."""
    logger = logging.getLogger(__name__)
    logger.info("Testing performance optimization...")
    
    try:
        # Test optimized data collector
        collector = OptimizedDataCollector(cache_ttl=60, batch_size=3)
        test_items = DEFAULT_ITEMS[:2]
        test_cities = DEFAULT_CITIES[:1]
        
        # Test caching behavior
        logger.info("Testing cache performance...")
        
        # First call - should be cache miss
        data1 = collector.fetch_current_prices_optimized(test_items, test_cities, [1])
        stats1 = collector.get_performance_stats()
        
        # Second call - should be cache hit
        data2 = collector.fetch_current_prices_optimized(test_items, test_cities, [1])
        stats2 = collector.get_performance_stats()
        
        cache_hit_improvement = stats2['cache_hit_rate'] > stats1['cache_hit_rate']
        logger.info(f"Cache hit rate improvement: {cache_hit_improvement}")
        logger.info(f"Final cache hit rate: {stats2['cache_hit_rate']:.1%}")
        
        # Test memory optimizer
        logger.info("Testing memory optimization...")
        memory_optimizer = MemoryOptimizer()
        
        if not data1.empty:
            optimized_df = memory_optimizer.optimize_dataframe(data1)
            logger.info(f"DataFrame optimization: {len(optimized_df)} records processed")
            
            # Monitor memory usage
            memory_stats = memory_optimizer.monitor_memory_usage()
            logger.info(f"Memory usage: {memory_stats.get('rss_mb', 0):.1f}MB RSS")
        
        logger.info("Performance optimization: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Performance optimization test failed: {e}")
        return False


@performance_monitor 
def test_threshold_optimization():
    """Test threshold optimization."""
    logger = logging.getLogger(__name__)
    logger.info("Testing threshold optimization...")
    
    framework = ValidationFramework()
    test_items = DEFAULT_ITEMS[:1]  # Single item for faster testing
    test_cities = DEFAULT_CITIES[:1]
    
    try:
        # Use smaller parameter ranges for faster testing
        param_ranges = {
            'contamination': [0.02, 0.05],
            'z_threshold': [2.0, 2.5],
            'peer_dev_threshold': [1.5, 2.0]
        }
        
        optimization_results = framework.optimize_thresholds(
            test_items, test_cities, param_ranges=param_ranges
        )
        
        best_params = optimization_results.get('best_parameters', {})
        best_f1 = optimization_results.get('best_f1_score', 0)
        
        logger.info(f"Threshold Optimization Results:")
        logger.info(f"  Best F1 Score: {best_f1:.3f}")
        logger.info(f"  Best Parameters: {best_params}")
        logger.info(f"  Combinations tested: {optimization_results.get('total_combinations_tested', 0)}")
        
        logger.info("Threshold optimization: PASSED")
        return True
        
    except Exception as e:
        logger.error(f"Threshold optimization failed: {e}")
        return False


@performance_monitor
def test_comprehensive_validation():
    """Test complete validation suite."""
    logger = logging.getLogger(__name__)
    logger.info("Testing comprehensive validation suite...")
    
    framework = ValidationFramework()
    test_items = DEFAULT_ITEMS[:1]  # Minimal set for comprehensive test
    test_cities = DEFAULT_CITIES[:1]
    
    try:
        comprehensive_results = framework.run_comprehensive_validation(test_items, test_cities)
        
        if 'error' not in comprehensive_results:
            logger.info("Comprehensive Validation Results:")
            logger.info(f"  Validation duration: {comprehensive_results.get('validation_duration_seconds', 0):.1f}s")
            
            # Check individual components
            if 'backtests' in comprehensive_results:
                ml_f1 = comprehensive_results['backtests']['ml_method']['f1_score']
                rules_f1 = comprehensive_results['backtests']['rules_method']['f1_score']
                logger.info(f"  ML F1: {ml_f1:.3f}, Rules F1: {rules_f1:.3f}")
            
            if 'threshold_optimization' in comprehensive_results:
                opt_f1 = comprehensive_results['threshold_optimization']['best_f1_score']
                logger.info(f"  Optimized F1: {opt_f1:.3f}")
            
            recommendations = comprehensive_results.get('recommendations', [])
            logger.info(f"  Recommendations: {len(recommendations)}")
            
            # Export report for review
            report_path = framework.export_validation_report(comprehensive_results)
            logger.info(f"  Report exported to: {report_path}")
            
            logger.info("Comprehensive validation: PASSED")
            return True
        else:
            logger.error(f"Comprehensive validation error: {comprehensive_results['error']}")
            return False
            
    except Exception as e:
        logger.error(f"Comprehensive validation failed: {e}")
        return False


def main():
    """Run all validation tests."""
    logger = setup_test_logging()
    logger.info("=== Starting Phase 4 Validation Tests ===")
    
    test_functions = [
        test_synthetic_data_generation,
        test_backtesting,
        test_false_positive_analysis,
        test_performance_optimization,
        test_threshold_optimization,
        test_comprehensive_validation
    ]
    
    results = {}
    
    for test_func in test_functions:
        test_name = test_func.__name__
        logger.info(f"\n--- Running {test_name} ---")
        
        try:
            success = test_func()
            results[test_name] = "PASSED" if success else "FAILED"
        except Exception as e:
            logger.error(f"Test {test_name} crashed: {e}")
            results[test_name] = "CRASHED"
    
    # Summary report
    logger.info("\n=== Test Results Summary ===")
    passed = 0
    total = len(results)
    
    for test_name, result in results.items():
        status_emoji = "[PASS]" if result == "PASSED" else "[FAIL]"
        logger.info(f"{status_emoji} {test_name}: {result}")
        if result == "PASSED":
            passed += 1
    
    logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")
    
    # Performance statistics
    logger.info("\n=== Performance Statistics ===")
    for test_func in test_functions:
        if hasattr(test_func, 'get_performance_stats'):
            stats = test_func.get_performance_stats()
            test_name = test_func.__name__
            
            if stats.get(test_name):
                avg_time = sum(call['execution_time'] for call in stats[test_name]) / len(stats[test_name])
                logger.info(f"{test_name}: {avg_time:.2f}s average execution time")
    
    if passed == total:
        logger.info("\n[SUCCESS] All validation tests passed! Phase 4 is ready for production.")
        return 0
    else:
        logger.warning(f"\n[WARNING] {total - passed} test(s) failed. Review logs and fix issues before deployment.")
        return 1


if __name__ == "__main__":
    sys.exit(main())