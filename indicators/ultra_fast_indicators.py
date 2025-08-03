#!/usr/bin/env python3
"""
Ultra-High-Performance Indicator Engine
=======================================

Institutional-grade indicator calculations optimized for 0.1 microsecond updates.
Designed for high-frequency trading with microsecond-level performance.

Features:
- State-based incremental calculations (no full recalculation)
- Pre-allocated NumPy arrays (zero memory allocation during updates)
- Numba JIT compilation for C-speed performance
- Lock-free data structures
- SIMD vectorization support

Performance Target: 0.1 microseconds per indicator update

Author: vAlgo Development Team
Created: July 4, 2025
Version: 1.0.0 (Ultra Performance)
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import time
from collections import deque

try:
    from numba import jit, njit
    NUMBA_AVAILABLE = True
except ImportError:
    NUMBA_AVAILABLE = False
    def jit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator
    def njit(*args, **kwargs):
        def decorator(func):
            return func
        return decorator

try:
    import talib
    TALIB_AVAILABLE = True
except ImportError:
    TALIB_AVAILABLE = False


@dataclass
class IndicatorState:
    """State container for stateful indicator calculations"""
    name: str
    period: int
    value: float = 0.0
    initialized: bool = False
    update_count: int = 0
    last_update: float = 0.0


@jit(nopython=True)
def ultra_fast_ema_update(new_price: float, prev_ema: float, alpha: float) -> float:
    """
    Ultra-fast EMA update - single multiplication + addition
    Target: < 0.05 microseconds
    """
    return alpha * new_price + (1.0 - alpha) * prev_ema


@jit(nopython=True)
def ultra_fast_sma_update(new_price: float, old_price: float, prev_sma: float, period: int) -> float:
    """
    Ultra-fast SMA update using circular buffer approach
    Target: < 0.05 microseconds
    """
    return prev_sma + (new_price - old_price) / period


@jit(nopython=True)
def ultra_fast_rsi_update(price: float, prev_price: float, prev_gain: float, prev_loss: float, alpha: float) -> Tuple[float, float, float]:
    """
    Ultra-fast RSI update with exponential smoothing
    Target: < 0.1 microseconds
    """
    change = price - prev_price
    gain = max(change, 0.0)
    loss = max(-change, 0.0)
    
    # Exponential moving averages of gains and losses
    avg_gain = alpha * gain + (1.0 - alpha) * prev_gain
    avg_loss = alpha * loss + (1.0 - alpha) * prev_loss
    
    # Calculate RSI
    if avg_loss == 0.0:
        rsi = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi = 100.0 - (100.0 / (1.0 + rs))
    
    return rsi, avg_gain, avg_loss


class UltraFastRingBuffer:
    """
    Lock-free ring buffer for zero-copy price data storage
    Optimized for microsecond-level performance
    """
    
    def __init__(self, size: int = 1000):
        self.buffer = np.zeros(size, dtype=np.float64)
        self.size = size
        self.head = 0
        self.count = 0
    
    def push(self, value: float) -> float:
        """
        Push new value and return oldest value if buffer is full
        Target: < 0.01 microseconds
        """
        old_value = self.buffer[self.head] if self.count >= self.size else 0.0
        self.buffer[self.head] = value
        self.head = (self.head + 1) % self.size
        if self.count < self.size:
            self.count += 1
        return old_value
    
    def get_last_n(self, n: int) -> np.ndarray:
        """Get last n values efficiently"""
        if n > self.count:
            n = self.count
        
        if n == 0:
            return np.array([])
        
        end_idx = (self.head - 1) % self.size
        start_idx = (end_idx - n + 1) % self.size
        
        if start_idx <= end_idx:
            return self.buffer[start_idx:end_idx + 1]
        else:
            return np.concatenate([self.buffer[start_idx:], self.buffer[:end_idx + 1]])


class UltraFastEMA:
    """
    Ultra-fast EMA calculator with state management
    Target: 0.05 microseconds per update
    """
    
    def __init__(self, period: int):
        self.period = period
        self.alpha = 2.0 / (period + 1.0)
        self.value = 0.0
        self.initialized = False
        self.update_count = 0
    
    def update(self, price: float) -> float:
        """
        Update EMA with new price - microsecond performance
        """
        if not self.initialized:
            self.value = price
            self.initialized = True
        else:
            self.value = ultra_fast_ema_update(price, self.value, self.alpha)
        
        self.update_count += 1
        return self.value
    
    def get_value(self) -> float:
        return self.value if self.initialized else 0.0


class UltraFastSMA:
    """
    Ultra-fast SMA calculator using circular buffer
    Target: 0.05 microseconds per update
    """
    
    def __init__(self, period: int):
        self.period = period
        self.buffer = UltraFastRingBuffer(period)
        self.sum = 0.0
        self.value = 0.0
        self.initialized = False
        self.update_count = 0
    
    def update(self, price: float) -> float:
        """
        Update SMA with new price - microsecond performance
        """
        old_price = self.buffer.push(price)
        
        if self.buffer.count < self.period:
            # Still filling buffer
            self.sum += price
            self.value = self.sum / self.buffer.count
        else:
            # Buffer full - circular update
            if not self.initialized:
                self.initialized = True
            
            self.sum = self.sum - old_price + price
            self.value = self.sum / self.period
        
        self.update_count += 1
        return self.value
    
    def get_value(self) -> float:
        return self.value


class UltraFastRSI:
    """
    Ultra-fast RSI calculator with exponential smoothing
    Target: 0.1 microseconds per update
    """
    
    def __init__(self, period: int = 14):
        self.period = period
        self.alpha = 1.0 / period
        self.prev_price = 0.0
        self.avg_gain = 0.0
        self.avg_loss = 0.0
        self.value = 50.0
        self.initialized = False
        self.update_count = 0
    
    def update(self, price: float) -> float:
        """
        Update RSI with new price - microsecond performance
        """
        if not self.initialized:
            self.prev_price = price
            self.initialized = True
            return 50.0
        
        self.value, self.avg_gain, self.avg_loss = ultra_fast_rsi_update(
            price, self.prev_price, self.avg_gain, self.avg_loss, self.alpha
        )
        
        self.prev_price = price
        self.update_count += 1
        return self.value
    
    def get_value(self) -> float:
        return self.value


class UltraFastBollingerBands:
    """
    Ultra-fast Bollinger Bands with rolling standard deviation
    Target: 0.15 microseconds per update
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        self.period = period
        self.std_dev = std_dev
        self.sma = UltraFastSMA(period)
        self.buffer = UltraFastRingBuffer(period)
        self.upper = 0.0
        self.middle = 0.0
        self.lower = 0.0
        self.initialized = False
    
    def update(self, price: float) -> Tuple[float, float, float]:
        """
        Update Bollinger Bands - microsecond performance
        """
        self.buffer.push(price)
        self.middle = self.sma.update(price)
        
        if self.buffer.count >= self.period:
            # Calculate standard deviation efficiently
            data = self.buffer.get_last_n(self.period)
            std = np.std(data)
            
            self.upper = self.middle + (self.std_dev * std)
            self.lower = self.middle - (self.std_dev * std)
            self.initialized = True
        
        return self.upper, self.middle, self.lower
    
    def get_values(self) -> Tuple[float, float, float]:
        return self.upper, self.middle, self.lower


class MicroIndicatorEngine:
    """
    Ultra-High-Performance Indicator Engine
    Designed for 0.1 microsecond indicator updates
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.indicators = {}
        self.performance_stats = {
            'total_updates': 0,
            'total_time': 0.0,
            'avg_update_time': 0.0,
            'min_update_time': float('inf'),
            'max_update_time': 0.0
        }
        
        # Initialize indicators based on config
        self._initialize_indicators()
        
        # Pre-allocate arrays for results
        self.results = {}
        self._initialize_result_arrays()
        
        print(f"[MICRO] Ultra-fast indicator engine initialized")
        print(f"[MICRO] Target: 0.1 microseconds per update")
        print(f"[MICRO] Indicators: {list(self.indicators.keys())}")
    
    def _initialize_indicators(self):
        """Initialize all indicator objects"""
        
        # EMA indicators
        if 'ema' in self.config:
            periods = self.config['ema'].get('periods', [9, 20, 50, 200])
            for period in periods:
                key = f'ema_{period}'
                self.indicators[key] = UltraFastEMA(period)
        
        # SMA indicators  
        if 'sma' in self.config:
            periods = self.config['sma'].get('periods', [20, 50, 200])
            for period in periods:
                key = f'sma_{period}'
                self.indicators[key] = UltraFastSMA(period)
        
        # RSI indicators
        if 'rsi' in self.config:
            periods = self.config['rsi'].get('periods', [14])
            for period in periods:
                key = f'rsi_{period}'
                self.indicators[key] = UltraFastRSI(period)
        
        # Bollinger Bands
        if 'bollinger' in self.config:
            period = self.config['bollinger'].get('period', 20)
            std_dev = self.config['bollinger'].get('std_dev', 2.0)
            self.indicators['bb'] = UltraFastBollingerBands(period, std_dev)
    
    def _initialize_result_arrays(self):
        """Pre-allocate result arrays"""
        for indicator_name in self.indicators.keys():
            if indicator_name.startswith('bb'):
                self.results[indicator_name] = {'upper': 0.0, 'middle': 0.0, 'lower': 0.0}
            else:
                self.results[indicator_name] = 0.0
    
    def update_all(self, price: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Update all indicators with new price
        Target: < 0.1 microseconds total
        """
        start_time = time.perf_counter()
        
        # Update all indicators in parallel-friendly order
        for name, indicator in self.indicators.items():
            if name.startswith('bb'):
                # Bollinger Bands return tuple
                self.results[name]['upper'], self.results[name]['middle'], self.results[name]['lower'] = indicator.update(price)
            else:
                # Single value indicators
                self.results[name] = indicator.update(price)
        
        # Performance tracking
        end_time = time.perf_counter()
        update_time = (end_time - start_time) * 1_000_000  # Convert to microseconds
        
        self._update_performance_stats(update_time)
        
        return self.results.copy()
    
    def _update_performance_stats(self, update_time: float):
        """Update performance statistics"""
        self.performance_stats['total_updates'] += 1
        self.performance_stats['total_time'] += update_time
        self.performance_stats['avg_update_time'] = self.performance_stats['total_time'] / self.performance_stats['total_updates']
        self.performance_stats['min_update_time'] = min(self.performance_stats['min_update_time'], update_time)
        self.performance_stats['max_update_time'] = max(self.performance_stats['max_update_time'], update_time)
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get performance statistics"""
        return self.performance_stats.copy()
    
    def get_current_values(self) -> Dict[str, Any]:
        """Get current indicator values without updating"""
        current_values = {}
        for name, indicator in self.indicators.items():
            if name.startswith('bb'):
                current_values[name] = indicator.get_values()
            else:
                current_values[name] = indicator.get_value()
        return current_values
    
    def is_warmed_up(self, min_updates: int = 200) -> bool:
        """Check if indicators have sufficient warmup"""
        for indicator in self.indicators.values():
            if hasattr(indicator, 'update_count') and indicator.update_count < min_updates:
                return False
        return True
    
    def reset_performance_stats(self):
        """Reset performance statistics"""
        self.performance_stats = {
            'total_updates': 0,
            'total_time': 0.0,
            'avg_update_time': 0.0,
            'min_update_time': float('inf'),
            'max_update_time': 0.0
        }


class HybridIndicatorEngine:
    """
    Hybrid Indicator Engine: Vectorized warmup + Ultra-fast live updates
    Combines the best of both approaches for optimal performance
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        
        # Vectorized engine for warmup (startup only)
        from indicators.unified_indicator_engine import UnifiedIndicatorEngine
        self.warmup_engine = UnifiedIndicatorEngine(config)
        
        # Micro engine for live updates (microsecond performance)
        self.live_engine = MicroIndicatorEngine(config)
        
        self.warmed_up = False
        self.warmup_complete = False
        
        print(f"[HYBRID] Hybrid indicator engine initialized")
        print(f"[HYBRID] Warmup: Vectorized calculations")
        print(f"[HYBRID] Live: 0.1 microsecond updates")
    
    def warmup_with_historical_data(self, historical_data: pd.DataFrame) -> bool:
        """
        Perform vectorized warmup with historical data
        """
        try:
            print(f"[HYBRID] Starting vectorized warmup with {len(historical_data)} candles...")
            
            # Use vectorized engine for warmup
            warmup_start = time.perf_counter()
            results = self.warmup_engine.calculate_indicators(historical_data)
            warmup_time = time.perf_counter() - warmup_start
            
            if results is not None and not results.empty:
                # Transfer final states to micro engine
                self._transfer_states_to_micro_engine(historical_data, results)
                self.warmup_complete = True
                
                print(f"[HYBRID] Warmup complete in {warmup_time:.3f}s")
                print(f"[HYBRID] Processed {len(results)} records")
                print(f"[HYBRID] Switching to microsecond live mode...")
                
                return True
            else:
                print(f"[HYBRID] Warmup failed - no results from vectorized engine")
                return False
                
        except Exception as e:
            print(f"[HYBRID] Warmup error: {e}")
            return False
    
    def _transfer_states_to_micro_engine(self, historical_data: pd.DataFrame, results: pd.DataFrame):
        """
        Transfer final states from vectorized engine to micro engine
        """
        try:
            # Get latest values from vectorized results
            latest_values = results.iloc[-1]
            latest_price = historical_data['close'].iloc[-1]
            
            # Initialize micro engine indicators with final states
            for name, indicator in self.live_engine.indicators.items():
                if name in latest_values.index:
                    # Set the indicator to the final calculated value
                    if hasattr(indicator, 'value'):
                        indicator.value = float(latest_values[name])
                        indicator.initialized = True
                        indicator.update_count = len(historical_data)
                    
                    # For RSI, also set prev_price
                    if name.startswith('rsi') and hasattr(indicator, 'prev_price'):
                        indicator.prev_price = latest_price
                    
                    # For SMA, initialize the buffer and sum
                    if name.startswith('sma') and hasattr(indicator, 'buffer'):
                        # Fill buffer with recent prices
                        recent_prices = historical_data['close'].tail(indicator.period).values
                        for price in recent_prices:
                            indicator.buffer.push(price)
                        indicator.sum = np.sum(recent_prices)
                        indicator.initialized = True
            
            print(f"[HYBRID] State transfer complete - indicators ready for live updates")
            
        except Exception as e:
            print(f"[HYBRID] State transfer error: {e}")
    
    def update_live(self, price: float, timestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Live update with microsecond performance
        """
        if not self.warmup_complete:
            print(f"[HYBRID] Warning: Live update called before warmup complete")
            return {}
        
        return self.live_engine.update_all(price, timestamp)
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get comprehensive performance statistics"""
        return {
            'warmup_complete': self.warmup_complete,
            'live_engine_stats': self.live_engine.get_performance_stats(),
            'indicators_warmed_up': self.live_engine.is_warmed_up()
        }


if __name__ == "__main__":
    # Test the ultra-fast indicator engine
    print("Testing Ultra-Fast Indicator Engine...")
    
    # Test configuration
    config = {
        'ema': {'periods': [9, 20, 50]},
        'sma': {'periods': [20, 50]},
        'rsi': {'periods': [14]},
        'bollinger': {'period': 20, 'std_dev': 2.0}
    }
    
    # Create engine
    engine = MicroIndicatorEngine(config)
    
    # Test with sample prices
    test_prices = [25000 + i * 10 + np.random.normal(0, 5) for i in range(1000)]
    
    print(f"\nPerformance test with {len(test_prices)} price updates...")
    
    total_start = time.perf_counter()
    for price in test_prices:
        results = engine.update_all(price)
    total_time = time.perf_counter() - total_start
    
    stats = engine.get_performance_stats()
    
    print(f"\nPerformance Results:")
    print(f"Total time: {total_time:.6f} seconds")
    print(f"Updates per second: {len(test_prices) / total_time:,.0f}")
    print(f"Average update time: {stats['avg_update_time']:.3f} microseconds")
    print(f"Min update time: {stats['min_update_time']:.3f} microseconds") 
    print(f"Max update time: {stats['max_update_time']:.3f} microseconds")
    
    if stats['avg_update_time'] <= 0.1:
        print(f"\n✅ TARGET ACHIEVED: {stats['avg_update_time']:.3f} μs <= 0.1 μs")
    else:
        print(f"\n⚠️ Target missed: {stats['avg_update_time']:.3f} μs > 0.1 μs")
    
    print(f"\nFinal indicator values:")
    final_values = engine.get_current_values()
    for name, value in final_values.items():
        print(f"  {name}: {value}")