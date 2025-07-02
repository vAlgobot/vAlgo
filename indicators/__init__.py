#!/usr/bin/env python3
"""
Technical Indicators Module
===========================

Comprehensive technical analysis indicators for vAlgo trading system.

Available Indicators:
- CPR (Central Pivot Range) - All levels with multiple timeframes
- Supertrend - ATR-based trend following
- EMA (Exponential Moving Average) - Multiple periods with crossovers
- RSI (Relative Strength Index) - Momentum oscillator
- VWAP (Volume Weighted Average Price) - Session-based VWAP
- SMA (Simple Moving Average) - Multiple periods with trend analysis
- Bollinger Bands - Volatility bands with squeeze detection
- Candle Values - Current/Previous/Day candle analysis

Usage:
    from indicators import CPR, Supertrend, EMA, RSI, VWAP, SMA, BollingerBands, CandleValues
    
    # Individual indicators
    cpr = CPR(timeframe='daily')
    supertrend = Supertrend(period=10, multiplier=3.0)
    ema = EMA(periods=[9, 21, 50, 200])
    
    # Calculate indicators
    data_with_cpr = cpr.calculate(ohlc_data)
    data_with_supertrend = supertrend.calculate(ohlc_data)
    data_with_ema = ema.calculate(ohlc_data)

Author: vAlgo Development Team
Created: June 28, 2025
"""

from .cpr import CPR, calculate_cpr
from .supertrend import Supertrend, calculate_supertrend
from .ema import EMA, calculate_ema
from .rsi import RSI, calculate_rsi
from .vwap import VWAP, calculate_vwap
from .sma import SMA, calculate_sma
from .bollinger_bands import BollingerBands, calculate_bollinger_bands
from .candle_values import CandleValues, calculate_candle_values

# Unified calculator system
from .unified_calculator import (
    UnifiedIndicatorCalculator,
    calculate_indicators_backtest,
    calculate_indicators_live,
    get_latest_indicator_values,
    export_indicators_to_excel
)

# High-accuracy validators
from .accuracy_validators import (
    HighAccuracyEMA,
    HighAccuracyRSI,
    calculate_high_accuracy_ema,
    calculate_high_accuracy_rsi
)

# Warmup calculator
from .warmup_calculator import WarmupCalculator, calculate_indicator_warmup

# All indicator classes
__all__ = [
    # Indicator classes
    'CPR',
    'Supertrend', 
    'EMA',
    'RSI',
    'VWAP',
    'SMA',
    'BollingerBands',
    'CandleValues',
    
    # Convenience functions
    'calculate_cpr',
    'calculate_supertrend',
    'calculate_ema',
    'calculate_rsi',
    'calculate_vwap',
    'calculate_sma',
    'calculate_bollinger_bands',
    'calculate_candle_values',
    
    # Unified calculator system
    'UnifiedIndicatorCalculator',
    'calculate_indicators_backtest',
    'calculate_indicators_live',
    'get_latest_indicator_values',
    'export_indicators_to_excel',
    
    # High-accuracy validators
    'HighAccuracyEMA',
    'HighAccuracyRSI',
    'calculate_high_accuracy_ema',
    'calculate_high_accuracy_rsi',
    
    # Warmup calculator
    'WarmupCalculator',
    'calculate_indicator_warmup'
]

# Indicator metadata
INDICATOR_INFO = {
    'CPR': {
        'name': 'Central Pivot Range',
        'type': 'Support/Resistance',
        'timeframes': ['daily', 'weekly', 'monthly'],
        'levels': ['pivot', 'r1', 'r2', 'r3', 's1', 's2', 's3', 'tc', 'bc'],
        'description': 'Calculates pivot point and support/resistance levels'
    },
    'Supertrend': {
        'name': 'Supertrend',
        'type': 'Trend Following',
        'parameters': ['period', 'multiplier'],
        'signals': ['bullish', 'bearish'],
        'description': 'ATR-based trend following indicator with dynamic S/R'
    },
    'EMA': {
        'name': 'Exponential Moving Average',
        'type': 'Trend/Moving Average',
        'periods': [9, 21, 50, 200],
        'features': ['crossovers', 'trend_strength', 'support_resistance'],
        'description': 'Exponentially weighted moving average with trend analysis'
    },
    'RSI': {
        'name': 'Relative Strength Index',
        'type': 'Momentum Oscillator',
        'range': [0, 100],
        'levels': ['overbought_70', 'oversold_30'],
        'description': 'Momentum oscillator for overbought/oversold conditions'
    },
    'VWAP': {
        'name': 'Volume Weighted Average Price',
        'type': 'Volume/Price',
        'sessions': ['daily', 'weekly', 'monthly', 'continuous'],
        'features': ['bands', 'volume_profile'],
        'description': 'Volume-weighted average price with session analysis'
    },
    'SMA': {
        'name': 'Simple Moving Average',
        'type': 'Trend/Moving Average',
        'periods': [9, 21, 50, 200],
        'features': ['crossovers', 'trend_strength', 'slopes'],
        'description': 'Simple arithmetic moving average with trend analysis'
    },
    'BollingerBands': {
        'name': 'Bollinger Bands',
        'type': 'Volatility',
        'components': ['upper', 'middle', 'lower'],
        'features': ['squeeze', 'breakouts', 'percent_b', 'bandwidth'],
        'description': 'Volatility bands with squeeze and breakout detection'
    },
    'CandleValues': {
        'name': 'Candle Values Analysis',
        'type': 'Candle/Time-based',
        'modes': ['CurrentCandle', 'PreviousCandle', 'CurrentDayCandle', 'PreviousDayCandle'],
        'features': ['breakouts', 'gaps', 'tests', 'signals'],
        'description': 'Dynamic candle analysis for multiple timeframes'
    }
}


def get_all_indicators():
    """
    Get dictionary of all available indicator classes.
    
    Returns:
        Dictionary mapping indicator names to classes
    """
    return {
        'CPR': CPR,
        'Supertrend': Supertrend,
        'EMA': EMA,
        'RSI': RSI,
        'VWAP': VWAP,
        'SMA': SMA,
        'BollingerBands': BollingerBands,
        'CandleValues': CandleValues
    }


def get_indicator_info(indicator_name: str = None):
    """
    Get information about indicators.
    
    Args:
        indicator_name: Specific indicator name (optional)
        
    Returns:
        Dictionary with indicator information
    """
    if indicator_name:
        return INDICATOR_INFO.get(indicator_name, {})
    return INDICATOR_INFO


def create_indicator_factory(config_data: dict = None):
    """
    Create indicator instances from configuration.
    
    Args:
        config_data: Dictionary with indicator configurations
        
    Returns:
        Dictionary of initialized indicator instances
    """
    indicators = {}
    
    if not config_data:
        # Default configuration
        config_data = {
            'CPR': {'timeframe': 'daily'},
            'Supertrend': {'period': 10, 'multiplier': 3.0},
            'EMA': {'periods': [9, 21, 50, 200]},
            'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
            'VWAP': {'session_type': 'daily', 'std_dev_multiplier': 1.0},
            'SMA': {'periods': [9, 21, 50, 200]},
            'BollingerBands': {'period': 20, 'std_dev': 2.0},
            'CandleValues': {'mode': 'PreviousDayCandle', 'aggregate_day': True}
        }
    
    indicator_classes = get_all_indicators()
    
    for indicator_name, params in config_data.items():
        if indicator_name in indicator_classes:
            try:
                indicators[indicator_name] = indicator_classes[indicator_name](**params)
            except Exception as e:
                print(f"Warning: Failed to create {indicator_name} indicator: {e}")
    
    return indicators


def calculate_all_indicators(data, config_data: dict = None):
    """
    Calculate all indicators for given data.
    
    Args:
        data: OHLCV DataFrame
        config_data: Indicator configurations
        
    Returns:
        DataFrame with all indicators calculated
    """
    indicators = create_indicator_factory(config_data)
    result_data = data.copy()
    
    for name, indicator in indicators.items():
        try:
            result_data = indicator.calculate(result_data)
        except Exception as e:
            print(f"Warning: Failed to calculate {name}: {e}")
    
    return result_data