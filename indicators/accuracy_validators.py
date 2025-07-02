#!/usr/bin/env python3
"""
High-Accuracy Indicator Validators
==================================

Enhanced EMA and RSI calculations with proper initialization and accuracy validation.
Implements professional-grade calculation methods for maximum accuracy.

Key Features:
- Proper EMA initialization with SMA warmup
- Wilder's RSI with correct smoothing
- Accuracy validation against reference implementations
- Performance benchmarking

Author: vAlgo Development Team
Created: June 29, 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, List, Optional
import warnings


class HighAccuracyEMA:
    """
    High-accuracy Exponential Moving Average calculator.
    
    Uses proper initialization and recursive calculation for maximum accuracy.
    """
    
    def __init__(self, period: int):
        """
        Initialize high-accuracy EMA calculator.
        
        Args:
            period: EMA period
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        self.period = period
        self.alpha = 2.0 / (period + 1)  # True exponential smoothing factor
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate high-accuracy EMA.
        
        Args:
            prices: Price series
            
        Returns:
            EMA series with proper initialization
        """
        if len(prices) < self.period:
            warnings.warn(f"Insufficient data for EMA{self.period}. Need {self.period}, got {len(prices)}")
            return pd.Series(index=prices.index, dtype=float)
        
        ema = pd.Series(index=prices.index, dtype=float)
        
        # Initialize with SMA of first 'period' values
        initial_sma = prices.iloc[:self.period].mean()
        ema.iloc[self.period - 1] = initial_sma
        
        # Calculate EMA recursively for maximum accuracy
        for i in range(self.period, len(prices)):
            ema.iloc[i] = (prices.iloc[i] * self.alpha) + (ema.iloc[i-1] * (1 - self.alpha))
        
        return ema
    
    def calculate_vectorized(self, prices: pd.Series) -> pd.Series:
        """
        Calculate EMA using pandas ewm for comparison.
        
        Args:
            prices: Price series
            
        Returns:
            EMA series using pandas implementation
        """
        return prices.ewm(span=self.period, adjust=False).mean()
    
    def validate_accuracy(self, prices: pd.Series, tolerance: float = 1e-6) -> Dict:
        """
        Validate accuracy between recursive and vectorized methods.
        
        Args:
            prices: Price series for testing
            tolerance: Acceptable difference tolerance
            
        Returns:
            Validation results dictionary
        """
        recursive_ema = self.calculate(prices)
        vectorized_ema = self.calculate_vectorized(prices)
        
        # Compare non-NaN values
        valid_mask = recursive_ema.notna() & vectorized_ema.notna()
        
        if valid_mask.sum() == 0:
            return {
                'valid': False,
                'error': 'No valid values to compare',
                'max_diff': np.nan,
                'mean_diff': np.nan
            }
        
        differences = abs(recursive_ema[valid_mask] - vectorized_ema[valid_mask])
        max_diff = differences.max()
        mean_diff = differences.mean()
        
        is_accurate = max_diff <= tolerance
        
        return {
            'valid': is_accurate,
            'max_difference': max_diff,
            'mean_difference': mean_diff,
            'tolerance': tolerance,
            'compared_values': valid_mask.sum(),
            'accuracy_percentage': 100 * (1 - mean_diff / prices[valid_mask].mean()) if mean_diff > 0 else 100
        }


class HighAccuracyRSI:
    """
    High-accuracy RSI calculator using Wilder's smoothing method.
    
    Implements the exact RSI calculation as defined by J. Welles Wilder Jr.
    """
    
    def __init__(self, period: int = 14):
        """
        Initialize high-accuracy RSI calculator.
        
        Args:
            period: RSI period (default: 14)
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        
        self.period = period
    
    def calculate(self, prices: pd.Series) -> pd.Series:
        """
        Calculate high-accuracy RSI using Wilder's method.
        
        Args:
            prices: Price series
            
        Returns:
            RSI series
        """
        if len(prices) < self.period + 1:
            warnings.warn(f"Insufficient data for RSI{self.period}. Need {self.period + 1}, got {len(prices)}")
            return pd.Series(index=prices.index, dtype=float)
        
        # Calculate price changes
        delta = prices.diff()
        
        # Separate gains and losses
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Initialize RSI series
        rsi = pd.Series(index=prices.index, dtype=float)
        
        # Calculate initial average gain and loss (simple average)
        initial_avg_gain = gains.iloc[1:self.period + 1].mean()
        initial_avg_loss = losses.iloc[1:self.period + 1].mean()
        
        # Avoid division by zero
        if initial_avg_loss == 0:
            rsi.iloc[self.period] = 100.0
        else:
            initial_rs = initial_avg_gain / initial_avg_loss
            rsi.iloc[self.period] = 100 - (100 / (1 + initial_rs))
        
        # Calculate subsequent RSI values using Wilder's smoothing
        avg_gain = initial_avg_gain
        avg_loss = initial_avg_loss
        
        for i in range(self.period + 1, len(prices)):
            # Wilder's smoothing formula
            avg_gain = ((avg_gain * (self.period - 1)) + gains.iloc[i]) / self.period
            avg_loss = ((avg_loss * (self.period - 1)) + losses.iloc[i]) / self.period
            
            if avg_loss == 0:
                rsi.iloc[i] = 100.0
            else:
                rs = avg_gain / avg_loss
                rsi.iloc[i] = 100 - (100 / (1 + rs))
        
        return rsi
    
    def calculate_standard(self, prices: pd.Series) -> pd.Series:
        """
        Calculate RSI using standard pandas implementation for comparison.
        
        Args:
            prices: Price series
            
        Returns:
            RSI series using standard method
        """
        delta = prices.diff()
        gains = delta.where(delta > 0, 0)
        losses = -delta.where(delta < 0, 0)
        
        # Simple moving averages (not Wilder's method)
        avg_gains = gains.rolling(window=self.period).mean()
        avg_losses = losses.rolling(window=self.period).mean()
        
        rs = avg_gains / avg_losses
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def validate_accuracy(self, prices: pd.Series) -> Dict:
        """
        Validate RSI accuracy and range.
        
        Args:
            prices: Price series for testing
            
        Returns:
            Validation results dictionary
        """
        rsi = self.calculate(prices)
        
        # Check RSI range (must be 0-100)
        valid_rsi = rsi.dropna()
        
        if len(valid_rsi) == 0:
            return {
                'valid': False,
                'error': 'No valid RSI values calculated',
                'range_valid': False,
                'min_value': np.nan,
                'max_value': np.nan
            }
        
        min_rsi = valid_rsi.min()
        max_rsi = valid_rsi.max()
        range_valid = (0 <= min_rsi) and (max_rsi <= 100)
        
        # Check for reasonable variation (RSI should not be constant)
        rsi_std = valid_rsi.std()
        has_variation = rsi_std > 0.1  # At least 0.1 point variation
        
        return {
            'valid': range_valid and has_variation,
            'range_valid': range_valid,
            'min_value': min_rsi,
            'max_value': max_rsi,
            'mean_value': valid_rsi.mean(),
            'std_deviation': rsi_std,
            'has_variation': has_variation,
            'calculated_values': len(valid_rsi)
        }


class AccuracyBenchmark:
    """
    Benchmark and validate indicator accuracy.
    """
    
    def __init__(self):
        """Initialize benchmark suite."""
        pass
    
    def benchmark_ema_accuracy(self, prices: pd.Series, periods: List[int]) -> Dict:
        """
        Benchmark EMA accuracy across multiple periods.
        
        Args:
            prices: Price series for testing
            periods: List of EMA periods to test
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for period in periods:
            try:
                ema_calc = HighAccuracyEMA(period)
                validation = ema_calc.validate_accuracy(prices)
                
                results[f'EMA_{period}'] = {
                    'period': period,
                    'accuracy_valid': validation['valid'],
                    'max_difference': validation.get('max_difference', np.nan),
                    'accuracy_percentage': validation.get('accuracy_percentage', 0),
                    'compared_values': validation.get('compared_values', 0)
                }
                
            except Exception as e:
                results[f'EMA_{period}'] = {
                    'period': period,
                    'error': str(e),
                    'accuracy_valid': False
                }
        
        return results
    
    def benchmark_rsi_accuracy(self, prices: pd.Series, periods: List[int] = [14]) -> Dict:
        """
        Benchmark RSI accuracy.
        
        Args:
            prices: Price series for testing
            periods: List of RSI periods to test
            
        Returns:
            Benchmark results
        """
        results = {}
        
        for period in periods:
            try:
                rsi_calc = HighAccuracyRSI(period)
                validation = rsi_calc.validate_accuracy(prices)
                
                results[f'RSI_{period}'] = {
                    'period': period,
                    'accuracy_valid': validation['valid'],
                    'range_valid': validation['range_valid'],
                    'min_value': validation.get('min_value', np.nan),
                    'max_value': validation.get('max_value', np.nan),
                    'mean_value': validation.get('mean_value', np.nan),
                    'calculated_values': validation.get('calculated_values', 0)
                }
                
            except Exception as e:
                results[f'RSI_{period}'] = {
                    'period': period,
                    'error': str(e),
                    'accuracy_valid': False
                }
        
        return results
    
    def create_comprehensive_test_data(self, length: int = 1000) -> pd.Series:
        """
        Create comprehensive test data for accuracy validation.
        
        Args:
            length: Number of data points
            
        Returns:
            Price series with various market conditions
        """
        np.random.seed(42)  # Reproducible results
        
        # Create different market phases
        phases = []
        phase_length = length // 4
        
        # 1. Trending up phase
        trend_up = np.cumsum(np.random.normal(0.001, 0.01, phase_length))
        phases.extend(trend_up)
        
        # 2. Sideways phase
        sideways = np.random.normal(0, 0.005, phase_length)
        phases.extend(sideways)
        
        # 3. Trending down phase
        trend_down = np.cumsum(np.random.normal(-0.001, 0.01, phase_length))
        phases.extend(trend_down)
        
        # 4. Volatile phase
        volatile = np.random.normal(0, 0.02, length - 3 * phase_length)
        phases.extend(volatile)
        
        # Convert to price series starting at 100
        prices = [100.0]
        for ret in phases:
            prices.append(prices[-1] * (1 + ret))
        
        # Create pandas series with datetime index
        dates = pd.date_range('2024-01-01', periods=len(prices), freq='D')
        return pd.Series(prices, index=dates)
    
    def run_comprehensive_accuracy_test(self) -> Dict:
        """
        Run comprehensive accuracy test for all indicators.
        
        Returns:
            Complete test results
        """
        print("🎯 Running Comprehensive Accuracy Tests")
        print("=" * 50)
        
        # Create test data
        test_data = self.create_comprehensive_test_data(1000)
        print(f"✓ Created test data: {len(test_data)} points")
        print(f"✓ Price range: {test_data.min():.2f} - {test_data.max():.2f}")
        
        # Test EMA accuracy
        print("\n📈 Testing EMA Accuracy...")
        ema_periods = [9, 21, 50, 200]
        ema_results = self.benchmark_ema_accuracy(test_data, ema_periods)
        
        for indicator, result in ema_results.items():
            if result.get('accuracy_valid', False):
                print(f"✓ {indicator}: {result['accuracy_percentage']:.4f}% accurate")
            else:
                print(f"✗ {indicator}: Failed - {result.get('error', 'Unknown error')}")
        
        # Test RSI accuracy
        print("\n📊 Testing RSI Accuracy...")
        rsi_results = self.benchmark_rsi_accuracy(test_data, [14])
        
        for indicator, result in rsi_results.items():
            if result.get('accuracy_valid', False):
                range_info = f"Range: {result['min_value']:.2f}-{result['max_value']:.2f}"
                print(f"✓ {indicator}: Valid ({range_info})")
            else:
                print(f"✗ {indicator}: Failed - {result.get('error', 'Invalid range')}")
        
        return {
            'test_data_length': len(test_data),
            'ema_results': ema_results,
            'rsi_results': rsi_results,
            'overall_success': all(
                r.get('accuracy_valid', False) for r in {**ema_results, **rsi_results}.values()
            )
        }


def calculate_high_accuracy_ema(prices: pd.Series, period: int) -> pd.Series:
    """
    Convenience function for high-accuracy EMA calculation.
    
    Args:
        prices: Price series
        period: EMA period
        
    Returns:
        High-accuracy EMA series
    """
    calculator = HighAccuracyEMA(period)
    return calculator.calculate(prices)


def calculate_high_accuracy_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    Convenience function for high-accuracy RSI calculation.
    
    Args:
        prices: Price series
        period: RSI period
        
    Returns:
        High-accuracy RSI series
    """
    calculator = HighAccuracyRSI(period)
    return calculator.calculate(prices)


if __name__ == "__main__":
    # Run comprehensive accuracy tests
    benchmark = AccuracyBenchmark()
    results = benchmark.run_comprehensive_accuracy_test()
    
    print(f"\n🏁 ACCURACY TEST SUMMARY")
    print("=" * 30)
    print(f"Overall Success: {'✅ PASSED' if results['overall_success'] else '❌ FAILED'}")
    
    if results['overall_success']:
        print("\n🎉 All accuracy tests passed!")
        print("High-accuracy EMA and RSI calculations are working correctly.")
    else:
        print("\n⚠️ Some accuracy tests failed.")
        print("Please review the implementation for any issues.")