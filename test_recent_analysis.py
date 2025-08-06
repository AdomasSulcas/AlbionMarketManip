#!/usr/bin/env python3
"""
Test script for recent analysis functionality
"""

import sys
from pathlib import Path

# Add src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from albion_analyzer.recent_analysis import RecentAnomalyDetector


def test_recent_analysis():
    """Test current vs historical baseline analysis with a small set of items to avoid rate limits."""
    print("Testing current prices vs historical baseline analysis...")
    
    detector = RecentAnomalyDetector(recent_hours=6, baseline_days=30)
    
    # Use a very small set of common items to avoid rate limits
    test_items = ["T4_BAG", "T5_BAG", "T4_CAPE"]
    test_cities = ["Caerleon", "Lymhurst"]
    
    print(f"Testing with {len(test_items)} items and {len(test_cities)} cities")
    print("Items:", test_items)
    print("Cities:", test_cities)
    print("Approach: Current prices vs 30-day historical baseline")
    print()
    
    try:
        results = detector.analyze_recent_anomalies(test_items, test_cities)
        
        if 'error' in results:
            print(f"[ERROR] Analysis failed: {results['error']}")
            return False
        
        print("[SUCCESS] Analysis completed successfully!")
        print()
        
        # Display results
        stats = results.get('processing_stats', {})
        anomalies = results.get('anomalies', [])
        
        print(f"[STATS] Processing Stats:")
        print(f"  Items processed: {stats.get('items_processed', 0)}")
        print(f"  API errors: {stats.get('api_errors', 0)}")
        print(f"  Analysis duration: {results.get('analysis_duration_seconds', 0):.1f}s")
        print(f"  Anomalies detected: {len(anomalies)}")
        print()
        
        if anomalies:
            print(f"[ALERT] Top {min(5, len(anomalies))} anomalies:")
            for i, anomaly in enumerate(anomalies[:5]):
                print(f"  {i+1}. {anomaly['item']} in {anomaly['city']}")
                print(f"     Current: {anomaly['current_price']:,} silver")
                print(f"     Baseline: {anomaly['baseline_median']:,.0f} silver")
                print(f"     Deviation: {anomaly['price_deviation_pct']:+.1f}%")
                print(f"     Confidence: {anomaly['confidence']:.2f}")
                print(f"     Reasons: {', '.join(anomaly['anomaly_reasons'][:2])}")
                print()
        else:
            print("[OK] No significant anomalies detected: current prices align with historical baseline")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Test failed with exception: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_comprehensive_items():
    """Test the comprehensive item list generation."""
    print("Testing comprehensive item list generation...")
    
    detector = RecentAnomalyDetector()
    all_items = detector.get_all_tradeable_items()
    
    print(f"[SUCCESS] Generated {len(all_items)} comprehensive items")
    
    # Show some examples from different categories
    categories = {
        'Bags': [item for item in all_items if 'BAG' in item][:5],
        'Weapons': [item for item in all_items if '2H_AXE' in item or '2H_BOW' in item][:5],
        'Armor': [item for item in all_items if 'HEAD_PLATE' in item or 'ARMOR_PLATE' in item][:5],
        'Resources': [item for item in all_items if '_ORE' in item or '_WOOD' in item][:5],
        'Food': [item for item in all_items if '_MEAL_' in item][:5]
    }
    
    for category, items in categories.items():
        if items:
            print(f"  {category}: {', '.join(items[:3])}...")
    
    return True


def test_priority_items():
    """Test priority items functionality."""
    print("Testing priority items...")
    
    detector = RecentAnomalyDetector()
    priority_items = detector.get_high_priority_items()
    
    print(f"[SUCCESS] Generated {len(priority_items)} priority items")
    print(f"Examples: {', '.join(priority_items[:10])}...")
    
    return True


def main():
    """Run all tests."""
    print("=== Recent Analysis Testing ===")
    print()
    
    tests = [
        ("Comprehensive Items", test_comprehensive_items),
        ("Priority Items", test_priority_items),
        ("Current vs Historical Analysis", test_recent_analysis),
    ]
    
    passed = 0
    for test_name, test_func in tests:
        print(f"--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"[PASS] {test_name}: PASSED")
            else:
                print(f"[FAIL] {test_name}: FAILED")
        except Exception as e:
            print(f"[CRASH] {test_name}: CRASHED - {e}")
        print()
    
    print(f"=== Results: {passed}/{len(tests)} tests passed ===")
    return 0 if passed == len(tests) else 1


if __name__ == "__main__":
    sys.exit(main())