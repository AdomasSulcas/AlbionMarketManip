# Albion Online Market Manipulation Detection

A machine learning system for detecting market manipulation in Albion Online's city-based auction house system.

## Overview

This system analyzes market data to detect artificial price inflation caused by rich players or guilds buying out items en masse. It uses ensemble machine learning models combined with rule-based detection to identify suspicious market activity.

## Architecture

### Project Structure

```
├── main.py                    # CLI entry point
├── setup.py                   # Package installation
├── requirements.txt           # Dependencies
├── README.md                 # Documentation
└── src/albion_analyzer/      # Main package
    ├── __init__.py           # Package initialization
    ├── analysis.py           # Main analysis orchestration  
    ├── models.py             # ML models (Isolation Forest, LOF, One-Class SVM)
    ├── data_collector.py     # API integration and preprocessing
    ├── forecast.py           # ARIMA-based price forecasting
    ├── order_book.py         # Order book and bid-ask analysis
    ├── quality_analysis.py   # Cross-quality relationship analysis
    ├── config.py            # Configuration constants
    └── utils.py             # Utility functions
```

### Core Modules

- **`main.py`** - Command-line interface and main analysis script
- **`src/albion_analyzer/analysis.py`** - Main market analysis logic and orchestration  
- **`src/albion_analyzer/models.py`** - Machine learning models (Isolation Forest, LOF, One-Class SVM)
- **`src/albion_analyzer/data_collector.py`** - API integration and data preprocessing
- **`src/albion_analyzer/forecast.py`** - ARIMA-based price forecasting with quality support
- **`src/albion_analyzer/order_book.py`** - Order book analysis and bid-ask spread calculations
- **`src/albion_analyzer/quality_analysis.py`** - Cross-quality relationship analysis
- **`src/albion_analyzer/config.py`** - Configuration constants and settings
- **`src/albion_analyzer/utils.py`** - Utility functions and data validation

### Key Features

- **Multi-Quality Analysis**: Analyzes price relationships across item quality levels (1-5)
- **Ensemble ML Detection**: Combines multiple algorithms for robust anomaly detection
- **Gold Economic Context**: Distinguishes manipulation from server-wide economic trends
- **Real-Time Monitoring**: Continuous surveillance with sliding window detection
- **Recent Anomaly Detection**: Focuses on detecting manipulation in the last few hours
- **Comprehensive Item Coverage**: Analyzes 462+ items across all categories instead of small subsets
- **Cross-City Analysis**: Detects manipulation patterns across different cities
- **Bid-Ask Spread Analysis**: Identifies suspicious market depth patterns
- **Price Forecasting**: ARIMA-based expected price calculation
- **Alert System**: Real-time notifications with confidence scoring and deduplication
- **Persistent Storage**: SQLite database for historical analysis and system reliability

## Usage

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd albion-online-market

# Install dependencies
pip install -r requirements.txt

# Or install as package (optional)
pip install -e .
```

### Quick Start

```bash
# Test API connectivity
python main.py --test-api

# Quick scan with default settings
python main.py --quick-scan

# Analyze specific items and cities
python main.py --items T4_2H_AXE T5_BAG --cities Caerleon Lymhurst --days 14
```

### Command Line Options

```bash
# Use predefined item sets
python main.py --preset weapons --days 30

# Compare ML vs rule-based detection
python main.py --model-comparison --items T4_BAG T5_BAG

# Save results to file
python main.py --preset default --save-results results.json

# Use rule-based detection only
python main.py --use-rules-only --items T4_CAPE

# Adjust sensitivity
python main.py --contamination 0.02 --items T4_2H_AXE

# Real-time monitoring (Ctrl+C to stop)
python main.py --monitor --preset default

# Monitor specific items with custom settings
python main.py --monitor --items T4_2H_AXE T5_BAG --cities Caerleon Lymhurst --monitor-duration 60 --polling-interval 180

# Run Phase 4 validation and performance testing
python main.py --validate --preset default

# Run performance benchmarking only
python main.py --benchmark --preset weapons

# Compare current prices vs 30-day historical baseline
python main.py --recent 6 --preset default

# Quick scan: current prices vs historical baseline for priority items
python main.py --quick-recent

# Comprehensive scan: all 462 items, current vs historical analysis
python main.py --all-items --recent 12 --cities Caerleon Lymhurst
```

### Real-Time Monitoring

The system includes a comprehensive real-time monitoring mode that continuously surveys the market for manipulation:

```bash
# Start monitoring with default settings
python main.py --monitor

# Monitor for 2 hours with 3-minute polling
python main.py --monitor --monitor-duration 120 --polling-interval 180

# Monitor specific items and cities
python main.py --monitor --items T4_2H_AXE T5_2H_AXE --cities Caerleon Lymhurst
```

**Monitoring Features:**
- **Sliding Window Detection**: Analyzes price patterns in real-time using rolling data windows
- **Rate-Limited Polling**: Respects API limits (180 req/min, 300 req/5min) with adaptive delays
- **Alert Deduplication**: Prevents spam with configurable cooldown periods between similar alerts
- **Persistent Storage**: Saves all alerts and monitoring data to SQLite database
- **Economic Context**: Integrates gold price analysis to filter false positives
- **Console Notifications**: Real-time alert display with confidence scores and price deviations

### Current vs Historical Analysis

The system includes specialized analysis comparing current market prices against historical baselines:

```bash
# Compare current prices vs 30-day historical baseline
python main.py --recent 6 --preset default

# Quick scan: current prices vs historical baseline for priority items
python main.py --quick-recent

# Comprehensive scan: all 462 items, current vs historical analysis
python main.py --all-items --recent 12
```

**Recent Analysis Features:**
- **Current vs Historical Baseline**: Compares current market prices against 30-day historical baselines
- **Comprehensive Coverage**: Analyzes 462+ items across all categories (weapons, armor, resources, food, tools, mounts)
- **Priority Scanning**: Quick scans of high-value manipulation targets
- **Historical Context**: Uses 30 days of historical data to establish normal price ranges
- **Multiple Detection Criteria**: Price deviations >30%, Z-scores >2.0, 95th percentile violations, extreme changes >100%
- **Rate Limit Management**: Intelligent batching and delays to respect API limits
- **Real-time Detection**: Immediate analysis of current market state vs historical norms

### Available Presets

- **`default`** - Mix of weapons, armor, and accessories
- **`weapons`** - Various weapon types (axes, bows, swords)
- **`armor`** - Armor pieces (head, chest, shoes)
- **`bags`** - Bags and capes

## Detection Methods

### Machine Learning Ensemble
- **Isolation Forest** - General anomaly detection
- **Local Outlier Factor** - Local density-based anomalies  
- **One-Class SVM** - Robust to training outliers

### Features Used
- Rolling z-scores (3, 7, 14-day windows)
- Peer price deviations within item tiers
- Cross-quality price relationships
- Bid-ask spread percentages
- Order book depth metrics

### Rule-Based Detection
- Z-score threshold: >2.0 standard deviations
- Peer deviation: >1.5 log-price difference
- Spread threshold: >50% bid-ask spread

## Output

The system provides:
- **Manipulation Rate**: Percentage of suspicious transactions
- **Confidence Scores**: 0-1 confidence in each detection
- **Quality Anomalies**: Cross-quality pricing inconsistencies
- **Item-Level Summary**: Per-item manipulation statistics
- **Forecasts**: Expected vs actual price comparisons

### Exit Codes
- `0` - Healthy market (<1% manipulation)
- `1` - Moderate manipulation (1-5%) 
- `2` - High manipulation (>5%)

## Configuration

Key settings in `config.py`:
- API endpoints and rate limits
- Default item/city lists  
- Detection thresholds
- Quality tier multipliers

## API Integration

Uses the Albion Online Data Project API:
- **Current Prices**: Real-time buy/sell data
- **Historical Data**: Daily price history
- **Order Book**: Market depth information

Rate limits: 180 requests/minute, 300 requests/5 minutes

## Examples

### Basic Analysis
```bash
python main.py --items T4_2H_AXE --cities Caerleon Lymhurst
```

### Comprehensive Market Scan  
```bash
python main.py --preset default --days 30 --save-results market_report.json
```

### Compare Detection Methods
```bash
python main.py --model-comparison --preset weapons --days 14
```

## Phase 4: Validation & Performance Optimization

The system includes comprehensive validation and performance optimization tools:

### Validation Framework
- **Backtesting**: Test detection accuracy against historical data with synthetic manipulation cases
- **False Positive Analysis**: Identify and analyze patterns in false positive detections  
- **Threshold Optimization**: Grid search optimization to find optimal detection parameters
- **Performance Benchmarking**: Measure computational efficiency across different data volumes

### Optimization Features
- **Intelligent Caching**: TTL cache with automatic cleanup and performance monitoring
- **API Efficiency**: Rate-limited polling with batch processing and request optimization
- **Memory Optimization**: Efficient DataFrame operations and memory usage monitoring
- **Real-time Performance**: Sliding window detection with minimal computational overhead

### Validation Commands
```bash
# Run comprehensive validation suite
python main.py --validate --preset default

# Performance benchmarking only  
python main.py --benchmark --items T4_BAG T5_BAG

# Direct validation testing
python test_validation.py
```

### Validation Metrics
- **Precision/Recall/F1**: Standard classification metrics on synthetic manipulation data
- **Processing Speed**: Records processed per second and detection latency
- **Memory Efficiency**: Memory usage patterns and optimization effectiveness
- **API Performance**: Request rates, cache hit rates, and error handling

## Development

The modular architecture makes it easy to:
- Add new detection algorithms in `models.py`
- Extend feature engineering in `data_collector.py`
- Implement new analysis types in `quality_analysis.py`
- Modify configuration in `config.py`
- Add validation tests in `validation.py`
- Implement performance optimizations in `optimization.py`

## Legacy Files

- `market.ipynb` - Original Jupyter notebook (deprecated)
- `test_multi_quality.py` - Testing scripts for quality features
- `multi_quality_example.py` - Demo scripts

These remain for reference but the main workflow now uses `main.py`.