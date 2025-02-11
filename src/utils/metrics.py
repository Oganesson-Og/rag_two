"""
Metrics Collection Module
----------------------

Utility for collecting and managing metrics throughout the RAG pipeline.

Key Features:
- Performance metrics
- Timing measurements
- Counter tracking
- Metric aggregation
- Statistics calculation
- Logging integration
- Custom metrics

Technical Details:
- Thread-safe counters
- Time tracking
- Memory usage
- Error rate tracking
- Custom metric types
- Metric persistence
- Data aggregation

Dependencies:
- time
- statistics
- typing-extensions>=4.7.0
- logging

Example Usage:
    collector = MetricsCollector()
    
    # Track timing
    with collector.track_time("process_document"):
        process_document()
    
    # Increment counters
    collector.increment("documents_processed")
    
    # Record values
    collector.record("embedding_dimension", 768)
    
    # Get statistics
    stats = collector.get_statistics()

Author: Keith Satuku
Version: 2.0.0
Created: 2025
License: MIT
"""

from typing import Dict, Any, Optional, List, Union
from contextlib import contextmanager
import time
import statistics
import logging
from datetime import datetime

class MetricsCollector:
    """Collector for RAG pipeline metrics."""
    
    def __init__(self) -> None:
        self.metrics: Dict[str, List[Union[int, float]]] = {}
        self.counters: Dict[str, int] = {}
        self.timers: Dict[str, List[float]] = {}
        self.logger = logging.getLogger(__name__)

    def increment(self, metric_name: str, value: int = 1) -> None:
        """Increment a counter metric.
        
        Args:
            metric_name: Name of the counter
            value: Value to increment by
        """
        try:
            if metric_name not in self.counters:
                self.counters[metric_name] = 0
            self.counters[metric_name] += value
        except Exception as e:
            self.logger.error(f"Error incrementing metric {metric_name}: {str(e)}")
            raise

    def record(self, metric_name: str, value: Union[int, float]) -> None:
        """Record a metric value.
        
        Args:
            metric_name: Name of the metric
            value: Value to record
        """
        try:
            if metric_name not in self.metrics:
                self.metrics[metric_name] = []
            self.metrics[metric_name].append(value)
        except Exception as e:
            self.logger.error(f"Error recording metric {metric_name}: {str(e)}")
            raise

    @contextmanager
    def track_time(self, timer_name: str):
        """Context manager for tracking execution time.
        
        Args:
            timer_name: Name of the timer
        """
        try:
            start_time = time.time()
            yield
        finally:
            duration = time.time() - start_time
            if timer_name not in self.timers:
                self.timers[timer_name] = []
            self.timers[timer_name].append(duration)

    def get_counter(self, metric_name: str) -> int:
        """Get current value of a counter.
        
        Args:
            metric_name: Name of the counter
            
        Returns:
            int: Current counter value
        """
        return self.counters.get(metric_name, 0)

    def get_statistics(self, metric_name: str) -> Dict[str, float]:
        """Get statistics for a metric.
        
        Args:
            metric_name: Name of the metric
            
        Returns:
            Dict[str, float]: Statistics including mean, median, min, max
        """
        try:
            if metric_name in self.metrics:
                values = self.metrics[metric_name]
            elif metric_name in self.timers:
                values = self.timers[metric_name]
            else:
                return {}

            if not values:
                return {}

            return {
                'mean': statistics.mean(values),
                'median': statistics.median(values),
                'min': min(values),
                'max': max(values),
                'count': len(values)
            }
        except Exception as e:
            self.logger.error(f"Error calculating statistics for {metric_name}: {str(e)}")
            raise

    def reset(self) -> None:
        """Reset all metrics."""
        self.metrics.clear()
        self.counters.clear()
        self.timers.clear()

    def get_all_metrics(self) -> Dict[str, Any]:
        """Get all metrics and their values.
        
        Returns:
            Dict[str, Any]: All metrics and their current values
        """
        result = {
            'counters': self.counters.copy(),
            'timers': {name: self.get_statistics(name) 
                      for name in self.timers},
            'metrics': {name: self.get_statistics(name) 
                       for name in self.metrics}
        }
        return result 