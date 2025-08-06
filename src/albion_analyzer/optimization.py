"""
Performance Optimization and Efficiency Improvements
Provides memory optimization, API efficiency, and computational performance enhancements.
"""

import time
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from typing import Dict, List, Optional, Any, Callable
from functools import wraps, lru_cache
from collections import defaultdict, deque
from dataclasses import dataclass
import logging
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from .data_collector import AlbionDataCollector
from .config import API_RATE_LIMIT_DELAY


@dataclass
class CacheStats:
    """
    Cache performance statistics for monitoring optimization effectiveness.
    
    Tracks hit rates, memory usage, and cache efficiency metrics to guide
    caching strategy optimization and memory management decisions.
    """
    hits: int = 0
    misses: int = 0
    evictions: int = 0
    memory_usage_mb: float = 0.0
    
    @property
    def hit_rate(self) -> float:
        """Calculate cache hit rate as percentage."""
        total = self.hits + self.misses
        return (self.hits / total) if total > 0 else 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert cache stats to dictionary for reporting."""
        return {
            'hits': self.hits,
            'misses': self.misses,
            'evictions': self.evictions,
            'memory_usage_mb': self.memory_usage_mb,
            'hit_rate': self.hit_rate
        }


class TTLCache:
    """
    Time-To-Live cache with automatic expiration and memory management.
    
    Implements efficient caching with automatic cleanup of expired entries
    and memory usage monitoring. Designed for API response caching and
    computational result storage with configurable TTL policies.
    """

    def __init__(self, max_size: int = 1000, default_ttl: int = 300) -> None:
        """
        Initialize TTL cache with size and time limits.
        
        Args:
            max_size: Maximum number of items to cache
            default_ttl: Default time-to-live in seconds for cached items
        """
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache = {}
        self.access_times = {}
        self.expiry_times = {}
        self.stats = CacheStats()
        self._lock = threading.RLock()

    def get(self, key: str) -> Any:
        """
        Retrieve item from cache if valid and not expired.
        
        Args:
            key: Cache key to retrieve
            
        Returns:
            Cached value if found and valid, None otherwise
        """
        with self._lock:
            current_time = time.time()
            
            if key in self.cache:
                if current_time <= self.expiry_times[key]:
                    self.access_times[key] = current_time
                    self.stats.hits += 1
                    return self.cache[key]
                else:
                    # Expired, remove
                    self._remove_key(key)
            
            self.stats.misses += 1
            return None

    def put(self, key: str, value: Any, ttl: int = None) -> None:
        """
        Store item in cache with specified TTL.
        
        Args:
            key: Cache key to store under
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
        """
        with self._lock:
            current_time = time.time()
            ttl = ttl or self.default_ttl
            
            # Remove oldest items if at capacity
            if len(self.cache) >= self.max_size and key not in self.cache:
                self._evict_oldest()
            
            self.cache[key] = value
            self.access_times[key] = current_time
            self.expiry_times[key] = current_time + ttl

    def _remove_key(self, key: str) -> None:
        """Remove key from all tracking structures."""
        self.cache.pop(key, None)
        self.access_times.pop(key, None)
        self.expiry_times.pop(key, None)

    def _evict_oldest(self) -> None:
        """Evict least recently used item to make space."""
        if not self.access_times:
            return
        
        oldest_key = min(self.access_times.items(), key=lambda x: x[1])[0]
        self._remove_key(oldest_key)
        self.stats.evictions += 1

    def cleanup_expired(self) -> int:
        """
        Remove all expired entries from cache.
        
        Returns:
            Number of entries removed
        """
        with self._lock:
            current_time = time.time()
            expired_keys = [
                key for key, expiry in self.expiry_times.items()
                if current_time > expiry
            ]
            
            for key in expired_keys:
                self._remove_key(key)
            
            return len(expired_keys)

    def get_stats(self) -> CacheStats:
        """Get current cache statistics."""
        with self._lock:
            self.stats.memory_usage_mb = len(str(self.cache)) / (1024 * 1024)  # Rough estimate
            return self.stats


class BatchProcessor:
    """
    Batch processing system for efficient API calls and data processing.
    
    Groups multiple requests into batches to minimize API calls and improve
    throughput while respecting rate limits and optimizing resource usage.
    """

    def __init__(self, batch_size: int = 10, batch_timeout: float = 5.0) -> None:
        """
        Initialize batch processor with size and timing parameters.
        
        Args:
            batch_size: Maximum items per batch
            batch_timeout: Maximum time to wait before processing partial batch
        """
        self.batch_size = batch_size
        self.batch_timeout = batch_timeout
        self.pending_requests = []
        self.batch_lock = threading.Lock()
        self.logger = logging.getLogger(__name__)

    async def add_request(self, request_data: Any, processor: Callable) -> Any:
        """
        Add request to batch for processing.
        
        Args:
            request_data: Data for the request
            processor: Function to process the batch when ready
            
        Returns:
            Processing result when batch completes
        """
        future = asyncio.Future()
        
        with self.batch_lock:
            self.pending_requests.append({
                'data': request_data,
                'future': future,
                'timestamp': time.time()
            })
            
            # Process if batch is full
            if len(self.pending_requests) >= self.batch_size:
                await self._process_batch(processor)
        
        # Set timeout for partial batches
        asyncio.create_task(self._timeout_batch(processor))
        
        return await future

    async def _process_batch(self, processor: Callable) -> None:
        """Process current batch of requests."""
        with self.batch_lock:
            if not self.pending_requests:
                return
            
            current_batch = self.pending_requests.copy()
            self.pending_requests.clear()
        
        try:
            batch_data = [req['data'] for req in current_batch]
            results = await processor(batch_data)
            
            # Distribute results to futures
            for i, req in enumerate(current_batch):
                if i < len(results):
                    req['future'].set_result(results[i])
                else:
                    req['future'].set_result(None)
                    
        except Exception as e:
            self.logger.error(f"Batch processing error: {e}")
            for req in current_batch:
                req['future'].set_exception(e)

    async def _timeout_batch(self, processor: Callable) -> None:
        """Process batch after timeout even if not full."""
        await asyncio.sleep(self.batch_timeout)
        
        with self.batch_lock:
            if self.pending_requests and \
               time.time() - self.pending_requests[0]['timestamp'] >= self.batch_timeout:
                await self._process_batch(processor)


class OptimizedDataCollector:
    """
    Optimized data collection with caching, batching, and efficient API usage.
    
    Extends base data collector with performance optimizations including
    intelligent caching, request batching, and memory-efficient data processing
    to minimize API calls and improve system responsiveness.
    """

    def __init__(self, cache_ttl: int = 300, batch_size: int = 5) -> None:
        """
        Initialize optimized data collector with caching and batching.
        
        Args:
            cache_ttl: Cache time-to-live in seconds
            batch_size: Number of items to batch per API request
        """
        self.base_collector = AlbionDataCollector()
        self.cache = TTLCache(max_size=2000, default_ttl=cache_ttl)
        self.batch_processor = BatchProcessor(batch_size=batch_size)
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.api_call_count = 0
        self.cache_hit_count = 0
        self.total_requests = 0

    def get_cached_or_fetch(self, cache_key: str, fetch_func: Callable, *args, **kwargs) -> Any:
        """
        Get data from cache or fetch if not available.
        
        Args:
            cache_key: Unique key for caching this request
            fetch_func: Function to call if cache miss
            *args: Arguments for fetch function
            **kwargs: Keyword arguments for fetch function
            
        Returns:
            Cached or freshly fetched data
        """
        self.total_requests += 1
        
        # Try cache first
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.cache_hit_count += 1
            return cached_result
        
        # Cache miss - fetch data
        try:
            result = fetch_func(*args, **kwargs)
            if result is not None:
                self.cache.put(cache_key, result)
                self.api_call_count += 1
            return result
        except Exception as e:
            self.logger.error(f"Failed to fetch data for key {cache_key}: {e}")
            return None

    def fetch_current_prices_optimized(self, items: List[str], cities: List[str], 
                                     qualities: List[int] = None) -> pd.DataFrame:
        """
        Fetch current prices with caching and optimization.
        
        Args:
            items: Items to fetch prices for
            cities: Cities to fetch prices for
            qualities: Quality levels to include (default: [1, 2, 3])
            
        Returns:
            DataFrame with current price data, potentially from cache
        """
        if qualities is None:
            qualities = [1, 2, 3]
        
        # Create cache key
        cache_key = f"current_prices_{'-'.join(sorted(items))}_{'-'.join(sorted(cities))}_{'-'.join(map(str, sorted(qualities)))}"
        
        return self.get_cached_or_fetch(
            cache_key,
            self.base_collector.fetch_current_prices,
            items, cities, qualities
        )

    def fetch_historical_data_optimized(self, items: List[str], cities: List[str], 
                                      days_back: int, quality: int = 1) -> pd.DataFrame:
        """
        Fetch historical data with intelligent caching based on data age.
        
        Uses different cache TTL for different time periods - recent data
        cached briefly, older data cached longer since it won't change.
        
        Args:
            items: Items to fetch historical data for
            cities: Cities to fetch historical data for
            days_back: Days of historical data to fetch
            quality: Item quality level
            
        Returns:
            DataFrame with historical price data
        """
        # Adjust cache TTL based on how far back we're looking
        if days_back <= 1:
            cache_ttl = 60  # 1 minute for recent data
        elif days_back <= 7:
            cache_ttl = 300  # 5 minutes for weekly data
        else:
            cache_ttl = 1800  # 30 minutes for older data
        
        cache_key = f"historical_{'-'.join(sorted(items))}_{'-'.join(sorted(cities))}_{days_back}_{quality}"
        
        # Fetch with custom TTL
        cached_result = self.cache.get(cache_key)
        if cached_result is not None:
            self.cache_hit_count += 1
            self.total_requests += 1
            return cached_result
        
        # Cache miss
        try:
            result = self.base_collector.fetch_historical_data(items, cities, days_back, quality)
            if result is not None and not result.empty:
                self.cache.put(cache_key, result, ttl=cache_ttl)
                self.api_call_count += 1
            
            self.total_requests += 1
            return result
        except Exception as e:
            self.logger.error(f"Failed to fetch historical data: {e}")
            return pd.DataFrame()

    async def batch_fetch_multiple_qualities(self, item: str, cities: List[str], 
                                           qualities: List[int]) -> Dict[int, pd.DataFrame]:
        """
        Batch fetch data for multiple qualities efficiently.
        
        Args:
            item: Single item to fetch for all qualities
            cities: Cities to fetch data for
            qualities: Quality levels to fetch
            
        Returns:
            Dictionary mapping quality level to DataFrame
        """
        async def quality_processor(quality_requests: List[Dict]) -> List[pd.DataFrame]:
            """Process batch of quality requests."""
            results = []
            for req in quality_requests:
                try:
                    data = self.fetch_current_prices_optimized(
                        [req['item']], req['cities'], [req['quality']]
                    )
                    results.append(data)
                    # Add delay to respect rate limits
                    await asyncio.sleep(API_RATE_LIMIT_DELAY)
                except Exception as e:
                    self.logger.error(f"Error fetching quality {req['quality']}: {e}")
                    results.append(pd.DataFrame())
            return results
        
        quality_data = {}
        for quality in qualities:
            request_data = {
                'item': item,
                'cities': cities,
                'quality': quality
            }
            result = await self.batch_processor.add_request(request_data, quality_processor)
            quality_data[quality] = result
        
        return quality_data

    def get_performance_stats(self) -> Dict[str, Any]:
        """
        Get performance statistics for optimization analysis.
        
        Returns:
            Dictionary containing cache hit rates, API usage, and efficiency metrics
        """
        cache_stats = self.cache.get_stats()
        
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hit_count,
            'api_calls': self.api_call_count,
            'cache_hit_rate': self.cache_hit_count / max(self.total_requests, 1),
            'api_efficiency': 1 - (self.api_call_count / max(self.total_requests, 1)),
            'cache_stats': cache_stats.to_dict()
        }

    def cleanup_cache(self) -> int:
        """
        Clean up expired cache entries.
        
        Returns:
            Number of expired entries removed
        """
        return self.cache.cleanup_expired()


class MemoryOptimizer:
    """
    Memory usage optimization and monitoring system.
    
    Provides memory-efficient data structures, garbage collection optimization,
    and memory usage monitoring to ensure system scalability and prevent
    memory-related performance degradation.
    """

    def __init__(self) -> None:
        """Initialize memory optimizer with monitoring."""
        self.logger = logging.getLogger(__name__)
        self.memory_snapshots = deque(maxlen=100)  # Keep last 100 snapshots
        self.gc_stats = {'collections': 0, 'freed_objects': 0}

    def optimize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Optimize DataFrame memory usage through efficient data types.
        
        Converts columns to more memory-efficient types where possible
        without losing information, significantly reducing memory footprint
        for large datasets.
        
        Args:
            df: DataFrame to optimize
            
        Returns:
            Memory-optimized DataFrame
        """
        if df.empty:
            return df
        
        optimized_df = df.copy()
        original_memory = optimized_df.memory_usage(deep=True).sum()
        
        # Optimize numeric columns
        for col in optimized_df.select_dtypes(include=['int64']).columns:
            col_min = optimized_df[col].min()
            col_max = optimized_df[col].max()
            
            if col_min >= np.iinfo(np.int8).min and col_max <= np.iinfo(np.int8).max:
                optimized_df[col] = optimized_df[col].astype(np.int8)
            elif col_min >= np.iinfo(np.int16).min and col_max <= np.iinfo(np.int16).max:
                optimized_df[col] = optimized_df[col].astype(np.int16)
            elif col_min >= np.iinfo(np.int32).min and col_max <= np.iinfo(np.int32).max:
                optimized_df[col] = optimized_df[col].astype(np.int32)
        
        # Optimize float columns
        for col in optimized_df.select_dtypes(include=['float64']).columns:
            optimized_df[col] = pd.to_numeric(optimized_df[col], downcast='float')
        
        # Optimize object columns that might be categorical
        for col in optimized_df.select_dtypes(include=['object']).columns:
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:  # Less than 50% unique
                optimized_df[col] = optimized_df[col].astype('category')
        
        optimized_memory = optimized_df.memory_usage(deep=True).sum()
        memory_saved = original_memory - optimized_memory
        
        self.logger.debug(f"DataFrame memory optimized: {memory_saved / 1024 / 1024:.1f}MB saved "
                         f"({memory_saved / original_memory:.1%} reduction)")
        
        return optimized_df

    def create_memory_efficient_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features using memory-efficient operations.
        
        Uses vectorized operations and in-place modifications where possible
        to minimize memory allocation during feature engineering.
        
        Args:
            df: Base DataFrame for feature creation
            
        Returns:
            DataFrame with additional features, memory-optimized
        """
        # Use copy with memory optimization
        result_df = self.optimize_dataframe(df.copy())
        
        # Create features using memory-efficient operations
        if 'price' in result_df.columns:
            # Use in-place operations where possible
            result_df['log_price'] = np.log(result_df['price'].clip(lower=1))
            
            # Use transform to avoid creating intermediate objects
            if 'item' in result_df.columns and 'city' in result_df.columns:
                result_df['price_rank'] = result_df.groupby(['timestamp'])['price'].rank(method='min')
                
                # Rolling operations with limited window to control memory
                result_df['price_ma_7d'] = (
                    result_df.groupby(['item', 'city'])['price']
                    .rolling(window=7, min_periods=3)
                    .mean()
                    .reset_index(level=[0, 1], drop=True)
                )
        
        return self.optimize_dataframe(result_df)

    def monitor_memory_usage(self) -> Dict[str, float]:
        """
        Monitor current memory usage and trends.
        
        Returns:
            Dictionary with memory usage statistics in MB
        """
        try:
            import psutil
            process = psutil.Process()
            memory_info = process.memory_info()
            
            memory_stats = {
                'rss_mb': memory_info.rss / 1024 / 1024,
                'vms_mb': memory_info.vms / 1024 / 1024,
                'percent': process.memory_percent(),
                'available_mb': psutil.virtual_memory().available / 1024 / 1024
            }
            
            self.memory_snapshots.append({
                'timestamp': datetime.now(),
                **memory_stats
            })
            
            return memory_stats
            
        except ImportError:
            self.logger.warning("psutil not available for memory monitoring")
            return {'rss_mb': 0, 'vms_mb': 0, 'percent': 0, 'available_mb': 0}

    def force_garbage_collection(self) -> Dict[str, int]:
        """
        Force garbage collection and return statistics.
        
        Returns:
            Dictionary with garbage collection results
        """
        import gc
        
        before_objects = len(gc.get_objects())
        collected = gc.collect()
        after_objects = len(gc.get_objects())
        
        freed_objects = before_objects - after_objects
        self.gc_stats['collections'] += 1
        self.gc_stats['freed_objects'] += freed_objects
        
        self.logger.debug(f"Garbage collection: {freed_objects} objects freed")
        
        return {
            'objects_before': before_objects,
            'objects_after': after_objects,
            'objects_freed': freed_objects,
            'collections_performed': collected
        }


def performance_monitor(func: Callable) -> Callable:
    """
    Decorator to monitor function performance and memory usage.
    
    Automatically tracks execution time, memory usage, and call frequency
    for decorated functions to identify performance bottlenecks.
    """
    call_stats = defaultdict(list)
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        start_memory = 0
        
        try:
            import psutil
            start_memory = psutil.Process().memory_info().rss
        except ImportError:
            pass
        
        try:
            result = func(*args, **kwargs)
            
            end_time = time.time()
            execution_time = end_time - start_time
            
            end_memory = 0
            try:
                end_memory = psutil.Process().memory_info().rss
            except ImportError:
                pass
            
            call_stats[func.__name__].append({
                'execution_time': execution_time,
                'memory_delta': end_memory - start_memory,
                'timestamp': datetime.now(),
                'success': True
            })
            
            return result
            
        except Exception as e:
            call_stats[func.__name__].append({
                'execution_time': time.time() - start_time,
                'memory_delta': 0,
                'timestamp': datetime.now(),
                'success': False,
                'error': str(e)
            })
            raise
    
    wrapper.get_performance_stats = lambda: dict(call_stats)
    wrapper.reset_stats = lambda: call_stats.clear()
    
    return wrapper


# Memory-efficient LRU cache for expensive computations
def memory_efficient_cache(maxsize: int = 128, ttl: int = 3600):
    """
    Memory-efficient cache with TTL and automatic cleanup.
    
    Args:
        maxsize: Maximum number of items to cache
        ttl: Time-to-live in seconds
        
    Returns:
        Decorator function
    """
    def decorator(func: Callable) -> Callable:
        cache = TTLCache(max_size=maxsize, default_ttl=ttl)
        
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Create cache key from arguments
            cache_key = f"{func.__name__}_{hash(str(args) + str(sorted(kwargs.items())))}"
            
            # Try cache first
            result = cache.get(cache_key)
            if result is not None:
                return result
            
            # Cache miss - compute and store
            result = func(*args, **kwargs)
            cache.put(cache_key, result)
            
            return result
        
        wrapper.cache_stats = cache.get_stats
        wrapper.cache_cleanup = cache.cleanup_expired
        
        return wrapper
    
    return decorator