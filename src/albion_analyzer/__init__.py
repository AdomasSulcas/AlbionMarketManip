"""
Albion Online Market Manipulation Detection Package
"""

__version__ = "1.0.0"
__author__ = "Market Analysis Team"
__description__ = "Machine learning system for detecting market manipulation in Albion Online"

# Import main classes for convenience
from .analysis import MarketAnalyzer, run_quick_analysis
from .data_collector import AlbionDataCollector
from .models import ManipulationDetector, SimpleRuleBasedDetector
from .realtime_monitor import RealTimeMonitor, MarketAlert
from .monitoring_dashboard import PersistentRealTimeMonitor
from .validation import ValidationFramework, BacktestResult, PerformanceMetrics
from .optimization import OptimizedDataCollector, MemoryOptimizer, performance_monitor
from .recent_analysis import RecentAnomalyDetector

__all__ = [
    "MarketAnalyzer",
    "run_quick_analysis",
    "AlbionDataCollector", 
    "ManipulationDetector",
    "SimpleRuleBasedDetector",
    "RealTimeMonitor",
    "MarketAlert",
    "PersistentRealTimeMonitor",
    "ValidationFramework",
    "BacktestResult", 
    "PerformanceMetrics",
    "OptimizedDataCollector",
    "MemoryOptimizer",
    "performance_monitor",
    "RecentAnomalyDetector"
]