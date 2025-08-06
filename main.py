#!/usr/bin/env python3
"""
Albion Online Market Manipulation Detection
Main analysis script - replaces Jupyter notebook workflow

Usage:
    python main.py --help
    python main.py --items T4_2H_AXE T5_2H_AXE --cities Caerleon Lymhurst
    python main.py --preset default --days 14 --save-results results.json
    python main.py --model-comparison --items T4_BAG T5_BAG
"""

import argparse
import sys
import logging
from typing import List, Optional
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Import our modules
from albion_analyzer.config import *
from albion_analyzer.utils import setup_logging, save_results_to_json, PerformanceTimer
from albion_analyzer.analysis import MarketAnalyzer, run_quick_analysis
from albion_analyzer.models import compare_models
from albion_analyzer.data_collector import AlbionDataCollector
from albion_analyzer.monitoring_dashboard import PersistentRealTimeMonitor
from albion_analyzer.validation import ValidationFramework
from albion_analyzer.optimization import OptimizedDataCollector
from albion_analyzer.recent_analysis import RecentAnomalyDetector
from albion_analyzer.user_interface import format_fraud_detection_report, format_market_safety_alert, export_to_csv

def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Albion Online Market Manipulation and Fraud Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Professional fraud detection report
  python main.py --quick-recent --output-format professional
  
  # User-friendly market safety alert
  python main.py --quick-recent --output-format user-friendly
  
  # Export to structured CSV files (separate files by fraud level)
  python main.py --quick-recent --output-format csv
  python main.py --quick-recent --output-format csv --csv-output my_reports
  
  # Technical analysis (detailed data)
  python main.py --quick-recent --output-format technical
        """
    )
    
    # Analysis configuration
    parser.add_argument(
        "--items", 
        nargs="+", 
        help="Items to analyze (e.g. T4_2H_AXE T5_BAG)"
    )
    parser.add_argument(
        "--cities",
        nargs="+",
        help="Cities to analyze (e.g. Caerleon Lymhurst)"
    )
    parser.add_argument(
        "--preset",
        choices=["default", "weapons", "armor", "bags"],
        help="Use predefined item/city combinations"
    )
    parser.add_argument(
        "--days",
        type=int,
        default=DEFAULT_LOOKBACK_DAYS,
        help=f"Days of historical data to analyze (default: {DEFAULT_LOOKBACK_DAYS})"
    )
    
    # Analysis options
    parser.add_argument(
        "--contamination",
        type=float,
        default=DEFAULT_CONTAMINATION,
        help=f"Expected manipulation rate (default: {DEFAULT_CONTAMINATION})"
    )
    parser.add_argument(
        "--use-ml",
        action="store_true",
        default=True,
        help="Use machine learning models (default: True)"
    )
    parser.add_argument(
        "--use-rules-only",
        action="store_true",
        help="Use rule-based detection only"
    )
    parser.add_argument(
        "--model-comparison",
        action="store_true",
        help="Compare ML vs rule-based approaches"
    )
    
    # Output options
    parser.add_argument(
        "--save-results",
        type=str,
        help="Save results to JSON file"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level"
    )
    
    # Special modes
    parser.add_argument(
        "--test-api",
        action="store_true",
        help="Test API connectivity and data availability"
    )
    parser.add_argument(
        "--quick-scan",
        action="store_true",
        help="Run quick scan with default settings"
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Start real-time monitoring system"
    )
    parser.add_argument(
        "--monitor-duration",
        type=int,
        default=0,
        help="Minutes to run real-time monitoring (0 = indefinite)"
    )
    parser.add_argument(
        "--polling-interval",
        type=int,
        default=300,
        help="Seconds between polling cycles for real-time monitoring"
    )
    parser.add_argument(
        "--validate",
        action="store_true", 
        help="Run Phase 4 validation and performance testing"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Run performance benchmarking only"
    )
    parser.add_argument(
        "--recent",
        type=int,
        default=0,
        help="Compare current prices vs 30-day historical baseline to detect manipulation (e.g., --recent 6)"
    )
    parser.add_argument(
        "--quick-recent",
        action="store_true",
        help="Quick scan: current prices vs historical baseline for high-priority items"
    )
    parser.add_argument(
        "--all-items",
        action="store_true",
        help="Use comprehensive item list covering all categories instead of defaults"
    )
    parser.add_argument(
        "--output-format",
        choices=["professional", "user-friendly", "technical", "csv"],
        default="technical",
        help="Output format: 'professional' for fraud analysis, 'user-friendly' for safety alerts, 'technical' for detailed analysis, 'csv' for structured data export"
    )
    parser.add_argument(
        "--csv-output",
        type=str,
        default="fraud_reports",
        help="Directory to save CSV files when using csv output format (default: fraud_reports)"
    )
    
    return parser.parse_args()

def get_preset_configuration(preset: str) -> tuple:
    """Get predefined item and city configurations"""
    if preset == "default":
        return DEFAULT_ITEMS, DEFAULT_CITIES
    elif preset == "weapons":
        return [
            "T4_2H_AXE", "T5_2H_AXE", "T6_2H_AXE",
            "T4_2H_BOW", "T5_2H_BOW", "T6_2H_BOW",
            "T4_2H_SWORD", "T5_2H_SWORD"
        ], DEFAULT_CITIES
    elif preset == "armor":
        return [
            "T4_HEAD_PLATE_SET1", "T5_HEAD_PLATE_SET1",
            "T4_ARMOR_PLATE_SET1", "T5_ARMOR_PLATE_SET1",
            "T4_SHOES_PLATE_SET1", "T5_SHOES_PLATE_SET1"
        ], DEFAULT_CITIES
    elif preset == "bags":
        return [
            "T4_BAG", "T5_BAG", "T6_BAG",
            "T4_CAPE", "T5_CAPE", "T6_CAPE"
        ], DEFAULT_CITIES
    else:
        return DEFAULT_ITEMS, DEFAULT_CITIES

def test_api_connectivity():
    """Test API connectivity and data availability"""
    logger = logging.getLogger(__name__)
    logger.info("Testing API connectivity...")
    
    collector = AlbionDataCollector()
    
    # Test with a single item
    test_items = ["T4_2H_AXE"]
    test_cities = ["Caerleon"]
    
    try:
        # Test current prices
        current_prices = collector.fetch_current_prices(test_items, test_cities)
        if not current_prices.empty:
            logger.info(f"[OK] Current prices API working - got {len(current_prices)} records")
        else:
            logger.warning("[WARN] Current prices API returned no data")
        
        # Test historical data
        historical_data = collector.fetch_historical_data(test_items, test_cities, 7)
        if not historical_data.empty:
            logger.info(f"[OK] Historical data API working - got {len(historical_data)} records")
        else:
            logger.warning("[WARN] Historical data API returned no data")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] API test failed: {e}")
        return False

def main():
    """Main analysis function"""
    args = parse_arguments()
    
    # Setup logging
    logger = setup_logging(args.log_level)
    
    if not args.quiet:
        logger.info("=== Albion Online Market Analysis ===")
    
    # Test API if requested
    if args.test_api:
        success = test_api_connectivity()
        sys.exit(0 if success else 1)
    
    # Determine items and cities to analyze
    if args.all_items:
        # Use comprehensive item list
        recent_detector = RecentAnomalyDetector()
        items = recent_detector.get_all_tradeable_items()
        cities = args.cities or DEFAULT_CITIES
        if not args.quiet:
            logger.info(f"Using comprehensive item list: {len(items)} items, {len(cities)} cities")
    elif args.preset:
        items, cities = get_preset_configuration(args.preset)
        if not args.quiet:
            logger.info(f"Using preset '{args.preset}': {len(items)} items, {len(cities)} cities")
    else:
        items = args.items or DEFAULT_ITEMS
        cities = args.cities or DEFAULT_CITIES
    
    if not items or not cities:
        logger.error("No items or cities specified. Use --items and --cities, or --preset")
        sys.exit(1)
    
    # Quick scan mode
    if args.quick_scan:
        if not args.quiet:
            logger.info("Running quick scan...")
        
        with PerformanceTimer("Quick scan"):
            results = run_quick_analysis(items[:3], cities[:2], 7)  # Limit scope for speed
        
        if args.save_results:
            save_results_to_json(results, args.save_results)
        
        sys.exit(0)
    
    # Real-time monitoring mode
    if args.monitor:
        logger.info("Starting real-time monitoring system...")
        
        try:
            import asyncio
            
            monitor = PersistentRealTimeMonitor(
                items=items,
                cities=cities,
                polling_interval_seconds=args.polling_interval
            )
            
            logger.info(f"Monitoring {len(items)} items across {len(cities)} cities")
            logger.info(f"Polling interval: {args.polling_interval} seconds")
            
            if args.monitor_duration > 0:
                logger.info(f"Monitoring duration: {args.monitor_duration} minutes")
            else:
                logger.info("Monitoring indefinitely (Ctrl+C to stop)")
            
            async def run_monitoring():
                monitoring_task = asyncio.create_task(monitor.start_monitoring())
                
                if args.monitor_duration > 0:
                    # Run for specified duration
                    await asyncio.sleep(args.monitor_duration * 60)
                    monitor.stop_monitoring()
                    await monitoring_task
                else:
                    # Run indefinitely
                    await monitoring_task
            
            try:
                asyncio.run(run_monitoring())
            except KeyboardInterrupt:
                logger.info("Monitoring stopped by user")
                monitor.stop_monitoring()
            
            # Show final statistics
            stats = monitor.get_monitoring_stats()
            logger.info(f"Monitoring session completed:")
            logger.info(f"  Polls completed: {stats.get('polls_completed', 0)}")
            logger.info(f"  Alerts generated: {stats.get('alerts_generated', 0)}")
            logger.info(f"  API errors: {stats.get('api_errors', 0)}")
            
            if stats.get('uptime_hours', 0) > 0:
                logger.info(f"  Uptime: {stats['uptime_hours']:.1f} hours")
                logger.info(f"  Avg alerts/hour: {stats.get('avg_alerts_per_hour', 0):.1f}")
            
        except ImportError:
            logger.error("Real-time monitoring requires asyncio (Python 3.7+)")
            sys.exit(1)
        except Exception as e:
            logger.error(f"Monitoring failed: {e}")
            sys.exit(1)
        
        sys.exit(0)
    
    # Quick recent scan mode
    if args.quick_recent:
        logger.info("Running quick scan: current prices vs 30-day historical baseline...")
        
        try:
            recent_detector = RecentAnomalyDetector()
            
            with PerformanceTimer("Quick recent scan"):
                results = recent_detector.quick_recent_scan()
            
            if 'error' not in results:
                anomalies = results.get('anomalies', [])
                
                # Format output based on user preference
                if args.output_format == "professional":
                    print(format_fraud_detection_report(results))
                elif args.output_format == "user-friendly":
                    print(format_market_safety_alert(results))
                elif args.output_format == "csv":
                    # Export structured CSV data
                    exported_files = export_to_csv(results, args.csv_output)
                    if exported_files:
                        print(f"\nFRAUD ANALYSIS EXPORTED TO CSV FILES:")
                        for category, filepath in exported_files.items():
                            print(f"  {category.replace('_', ' ').title()}: {filepath}")
                        print(f"\nTotal CSV files created: {len(exported_files)}")
                        print(f"Output directory: {args.csv_output}")
                    else:
                        print("No fraud data to export - market appears clean")
                else:
                    # Technical output (original format)
                    logger.info(f"Quick scan results: {len(anomalies)} anomalies detected")
                    
                    if anomalies:
                        logger.info("Top current price anomalies vs historical baseline:")
                        for i, anomaly in enumerate(anomalies[:10]):  # Show top 10
                            logger.info(f"  {i+1}. {anomaly['item']} in {anomaly['city']}: "
                                       f"{anomaly['current_price']:,} silver "
                                       f"({anomaly['price_deviation_pct']:+.1f}% vs historical baseline, "
                                       f"confidence: {anomaly['confidence']:.2f})")
                    else:
                        logger.info("No significant price anomalies detected vs historical baseline")
                
                if args.save_results:
                    save_results_to_json(results, args.save_results)
                    logger.info(f"Results saved to {args.save_results}")
            else:
                logger.error(f"Quick recent scan failed: {results['error']}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Quick recent scan failed with exception: {e}")
            if args.log_level == "DEBUG":
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        sys.exit(0)
    
    # Recent anomaly analysis mode
    if args.recent > 0:
        logger.info(f"Comparing current prices vs 30-day historical baseline...")
        
        try:
            recent_detector = RecentAnomalyDetector(recent_hours=args.recent, baseline_days=30)
            
            with PerformanceTimer("Current vs Historical analysis"):
                results = recent_detector.analyze_recent_anomalies(items, cities)
            
            if 'error' not in results:
                anomalies = results.get('anomalies', [])
                stats = results.get('processing_stats', {})
                
                # Format output based on user preference
                if args.output_format == "professional":
                    print(format_fraud_detection_report(results))
                elif args.output_format == "user-friendly":
                    print(format_market_safety_alert(results))
                elif args.output_format == "csv":
                    # Export structured CSV data
                    exported_files = export_to_csv(results, args.csv_output)
                    if exported_files:
                        print(f"\nFRAUD ANALYSIS EXPORTED TO CSV FILES:")
                        for category, filepath in exported_files.items():
                            print(f"  {category.replace('_', ' ').title()}: {filepath}")
                        print(f"\nTotal CSV files created: {len(exported_files)}")
                        print(f"Output directory: {args.csv_output}")
                    else:
                        print("No fraud data to export - market appears clean")
                else:
                    # Technical output (original format)
                    logger.info(f"Current vs Historical analysis results:")
                    logger.info(f"  Items processed: {stats.get('items_processed', 0)}")
                    logger.info(f"  Anomalies detected: {len(anomalies)}")
                    logger.info(f"  API errors: {stats.get('api_errors', 0)}")
                    logger.info(f"  Analysis duration: {results.get('analysis_duration_seconds', 0):.1f}s")
                    
                    if anomalies:
                        logger.info(f"\nTop {min(15, len(anomalies))} current price anomalies vs historical baseline:")
                        for i, anomaly in enumerate(anomalies[:15]):
                            reasons = ", ".join(anomaly['anomaly_reasons'][:2])  # Show first 2 reasons
                            logger.info(f"  {i+1}. {anomaly['item']} in {anomaly['city']}")
                            logger.info(f"      Current: {anomaly['current_price']:,} (Historical median: {anomaly['baseline_median']:,.0f})")
                            logger.info(f"      Deviation: {anomaly['price_deviation_pct']:+.1f}% | Confidence: {anomaly['confidence']:.2f}")
                            logger.info(f"      Reasons: {reasons}")
                    else:
                        logger.info("No significant price anomalies detected vs historical baseline")
                
                if args.save_results:
                    save_results_to_json(results, args.save_results)
                    logger.info(f"Detailed results saved to {args.save_results}")
            else:
                logger.error(f"Recent analysis failed: {results['error']}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Recent analysis failed with exception: {e}")
            if args.log_level == "DEBUG":
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        sys.exit(0)
    
    # Validation mode
    if args.validate:
        logger.info("Running Phase 4 validation and performance testing...")
        
        try:
            framework = ValidationFramework()
            
            # Use subset of items/cities for comprehensive testing
            test_items = items[:2] if len(items) > 2 else items
            test_cities = cities[:2] if len(cities) > 2 else cities
            
            logger.info(f"Validating with {len(test_items)} items, {len(test_cities)} cities")
            
            with PerformanceTimer("Comprehensive validation"):
                validation_results = framework.run_comprehensive_validation(test_items, test_cities)
            
            if 'error' not in validation_results:
                # Display key results
                if 'backtests' in validation_results:
                    ml_f1 = validation_results['backtests']['ml_method']['f1_score']
                    rules_f1 = validation_results['backtests']['rules_method']['f1_score']
                    logger.info(f"Backtest Results - ML F1: {ml_f1:.3f}, Rules F1: {rules_f1:.3f}")
                
                if 'threshold_optimization' in validation_results:
                    best_f1 = validation_results['threshold_optimization']['best_f1_score']
                    best_params = validation_results['threshold_optimization']['best_parameters']
                    logger.info(f"Optimal F1: {best_f1:.3f} with params: {best_params}")
                
                recommendations = validation_results.get('recommendations', [])
                if recommendations:
                    logger.info("Recommendations:")
                    for rec in recommendations:
                        logger.info(f"  - {rec}")
                
                # Export detailed report
                report_file = framework.export_validation_report(validation_results)
                logger.info(f"Detailed validation report saved to: {report_file}")
                
            else:
                logger.error(f"Validation failed: {validation_results['error']}")
                sys.exit(1)
                
        except Exception as e:
            logger.error(f"Validation failed with exception: {e}")
            if args.log_level == "DEBUG":
                import traceback
                traceback.print_exc()
            sys.exit(1)
        
        sys.exit(0)
    
    # Benchmark mode
    if args.benchmark:
        logger.info("Running performance benchmarking...")
        
        try:
            framework = ValidationFramework()
            test_items = items[:2] if len(items) > 2 else items
            test_cities = cities[:1]
            
            with PerformanceTimer("Performance benchmarking"):
                benchmark_results = framework.benchmark_performance(test_items, test_cities, [500, 1000, 2000])
            
            logger.info("Benchmark Results:")
            for size, metrics in benchmark_results.items():
                logger.info(f"  {size}: {metrics.throughput_records_per_second:.1f} records/sec, "
                           f"{metrics.memory_usage_mb:.1f}MB, {metrics.detection_latency_ms:.1f}ms latency")
            
        except Exception as e:
            logger.error(f"Benchmarking failed: {e}")
            sys.exit(1)
        
        sys.exit(0)
    
    # Model comparison mode
    if args.model_comparison:
        logger.info("Running model comparison...")
        
        collector = AlbionDataCollector()
        df = collector.get_manipulation_training_data(items, cities, args.days)
        
        if df.empty:
            logger.error("No data available for model comparison")
            sys.exit(1)
        
        with PerformanceTimer("Model comparison"):
            comparison_results = compare_models(df)
        
        if args.save_results:
            save_results_to_json(comparison_results, args.save_results)
        
        sys.exit(0)
    
    # Main analysis
    logger.info(f"Analyzing {len(items)} items across {len(cities)} cities")
    logger.info(f"Historical lookback: {args.days} days")
    logger.info(f"Expected manipulation rate: {args.contamination:.1%}")
    
    # Initialize analyzer
    analyzer = MarketAnalyzer(contamination=args.contamination)
    
    # Determine detection method
    use_ml = args.use_ml and not args.use_rules_only
    
    # Run analysis
    with PerformanceTimer("Market analysis"):
        try:
            results = analyzer.analyze_market(
                items=items,
                cities=cities,
                days_back=args.days,
                use_ml=use_ml
            )
            
            if "error" in results:
                logger.error(f"Analysis failed: {results['error']}")
                sys.exit(1)
            
        except Exception as e:
            logger.error(f"Analysis failed with exception: {e}")
            if args.log_level == "DEBUG":
                import traceback
                traceback.print_exc()
            sys.exit(1)
    
    # Display results
    if not args.quiet:
        analyzer.print_analysis_report(results)
    
    # Save results if requested
    if args.save_results:
        save_results_to_json(results, args.save_results)
        logger.info(f"Results saved to {args.save_results}")
    
    # Exit with appropriate code
    manipulation_rate = results['summary']['manipulation_rate']
    if manipulation_rate > 0.05:  # More than 5% manipulation
        logger.warning(f"High manipulation rate detected: {manipulation_rate:.1%}")
        sys.exit(2)
    elif manipulation_rate > 0.01:  # More than 1% manipulation
        logger.info(f"Moderate manipulation detected: {manipulation_rate:.1%}")
        sys.exit(1)
    else:
        logger.info("Market appears healthy")
        sys.exit(0)

if __name__ == "__main__":
    main()