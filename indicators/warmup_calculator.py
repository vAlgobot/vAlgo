#!/usr/bin/env python3
"""
Indicator Warmup Calculator
===========================

Calculates dynamic warmup periods for technical indicators to ensure high accuracy.
Based on professional backtesting standards with configurable accuracy levels.

Key Principles:
- EMA needs 4x period for 95% accuracy (Good level)
- RSI needs 4x period for 95% accuracy  
- Longer warmup = higher accuracy but more data requirements
- Professional systems use 50-100 candle minimum warmup

Author: vAlgo Development Team
Created: June 29, 2025
"""

from typing import Dict, List, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd


class WarmupCalculator:
    """
    Dynamic warmup period calculator for technical indicators.
    
    Calculates optimal warmup periods based on indicator type, period,
    and desired accuracy level for professional backtesting.
    """
    
    # Accuracy level multipliers for warmup calculation
    WARMUP_MULTIPLIERS = {
        'basic': 3,        # 90% accuracy - fast but less precise
        'good': 4,         # 95% accuracy - balanced approach (recommended)
        'high': 5,         # 99% accuracy - very accurate
        'conservative': 10  # >99% accuracy - maximum accuracy
    }
    
    # Minimum warmup periods for different indicator types
    MIN_WARMUP_PERIODS = {
        'EMA': 20,      # Minimum 20 candles for any EMA
        'SMA': 10,      # Minimum 10 candles for any SMA
        'RSI': 30,      # Minimum 30 candles for RSI (includes initial calculation)
        'VWAP': 50,     # Minimum 50 candles for session-based VWAP
        'BB': 25,       # Minimum 25 candles for Bollinger Bands
        'SUPERTREND': 35,  # Minimum 35 candles for Supertrend (ATR + trend)
        'CPR': 2,       # Minimum 2 days for daily pivot calculation
        'PDL': 2        # Minimum 2 days for previous day levels
    }
    
    def __init__(self, accuracy_level: str = 'good'):
        """
        Initialize warmup calculator.
        
        Args:
            accuracy_level: Desired accuracy level ('basic', 'good', 'high', 'conservative')
        """
        if accuracy_level not in self.WARMUP_MULTIPLIERS:
            raise ValueError(f"Invalid accuracy level. Must be one of: {list(self.WARMUP_MULTIPLIERS.keys())}")
        
        self.accuracy_level = accuracy_level
        self.multiplier = self.WARMUP_MULTIPLIERS[accuracy_level]
    
    def calculate_warmup_period(self, indicator_type: str, period: int) -> int:
        """
        Calculate warmup period for a specific indicator.
        
        Args:
            indicator_type: Type of indicator ('EMA', 'RSI', 'SMA', etc.)
            period: Indicator period (e.g., 21 for EMA21)
            
        Returns:
            Required warmup period in candles
        """
        indicator_type = indicator_type.upper()
        
        if indicator_type not in self.MIN_WARMUP_PERIODS:
            # Default calculation for unknown indicators
            calculated_warmup = period * self.multiplier
        else:
            # Calculate based on indicator type
            if indicator_type in ['EMA', 'SMA']:
                calculated_warmup = period * self.multiplier
            elif indicator_type == 'RSI':
                # RSI needs period + smoothing warmup
                calculated_warmup = period * self.multiplier
            elif indicator_type in ['VWAP', 'BB', 'SUPERTREND']:
                # Complex indicators need more warmup
                calculated_warmup = max(period * self.multiplier, 50)
            elif indicator_type in ['CPR', 'PDL']:
                # Daily-based indicators need fewer candles
                calculated_warmup = max(period * 2, 5)
            else:
                calculated_warmup = period * self.multiplier
        
        # Apply minimum warmup period
        min_warmup = self.MIN_WARMUP_PERIODS.get(indicator_type, 20)
        
        return max(calculated_warmup, min_warmup)
    
    def calculate_max_warmup_needed(self, indicator_configs: Dict) -> int:
        """
        Calculate maximum warmup period needed for all configured indicators.
        
        Args:
            indicator_configs: Dictionary of indicator configurations
            
        Returns:
            Maximum warmup period required
        """
        max_warmup = 0
        
        for indicator_name, config in indicator_configs.items():
            warmup_periods = []
            
            if indicator_name == 'EMA' and 'periods' in config:
                # Multiple EMA periods
                for period in config['periods']:
                    warmup_periods.append(self.calculate_warmup_period('EMA', period))
            
            elif indicator_name == 'SMA' and 'periods' in config:
                # Multiple SMA periods
                for period in config['periods']:
                    warmup_periods.append(self.calculate_warmup_period('SMA', period))
            
            elif indicator_name == 'RSI' and 'period' in config:
                warmup_periods.append(self.calculate_warmup_period('RSI', config['period']))
            
            elif indicator_name == 'BollingerBands' and 'period' in config:
                warmup_periods.append(self.calculate_warmup_period('BB', config['period']))
            
            elif indicator_name == 'Supertrend' and 'period' in config:
                warmup_periods.append(self.calculate_warmup_period('SUPERTREND', config['period']))
            
            elif indicator_name in ['VWAP', 'CPR', 'PreviousDayLevels']:
                # These don't have explicit periods but need warmup
                warmup_periods.append(self.calculate_warmup_period(indicator_name.upper(), 1))
            
            if warmup_periods:
                max_warmup = max(max_warmup, max(warmup_periods))
        
        # Professional minimum: 50-100 candles for stable backtesting
        professional_minimum = 100 if self.accuracy_level in ['high', 'conservative'] else 50
        
        return max(max_warmup, professional_minimum)
    
    def calculate_extended_start_date(self, start_date: Union[str, datetime], 
                                    max_warmup_candles: int, 
                                    timeframe: str = '1min') -> datetime:
        """
        Calculate extended start date to accommodate warmup period.
        
        Args:
            start_date: Desired backtest start date
            max_warmup_candles: Maximum warmup candles needed
            timeframe: Data timeframe ('1min', '5min', '1day', etc.)
            
        Returns:
            Extended start date to fetch data from
        """
        if isinstance(start_date, str):
            start_date = pd.to_datetime(start_date)
        
        # Estimate trading days needed based on timeframe
        if timeframe == '1min':
            # ~375 candles per trading day (6.25 hours)
            trading_days_needed = max_warmup_candles / 375
        elif timeframe == '5min':
            # ~75 candles per trading day
            trading_days_needed = max_warmup_candles / 75
        elif timeframe == '15min':
            # ~25 candles per trading day
            trading_days_needed = max_warmup_candles / 25
        elif timeframe == '1hour':
            # ~6 candles per trading day
            trading_days_needed = max_warmup_candles / 6
        elif timeframe == '1day':
            # 1 candle per trading day
            trading_days_needed = max_warmup_candles
        else:
            # Default assumption: ~100 candles per day
            trading_days_needed = max_warmup_candles / 100
        
        # Add buffer for weekends and holidays (multiply by 1.5)
        calendar_days_needed = int(trading_days_needed * 1.5) + 5
        
        extended_start = start_date - timedelta(days=calendar_days_needed)
        
        return extended_start
    
    def get_accuracy_info(self) -> Dict:
        """
        Get accuracy information for current level.
        
        Returns:
            Dictionary with accuracy details
        """
        accuracy_info = {
            'basic': {
                'accuracy': '90%',
                'description': 'Fast but less precise',
                'use_case': 'Quick analysis'
            },
            'good': {
                'accuracy': '95%',
                'description': 'Balanced approach (recommended)',
                'use_case': 'Production backtesting'
            },
            'high': {
                'accuracy': '99%',
                'description': 'Very accurate',
                'use_case': 'Critical analysis'
            },
            'conservative': {
                'accuracy': '>99%',
                'description': 'Maximum accuracy',
                'use_case': 'Research and validation'
            }
        }
        
        return accuracy_info.get(self.accuracy_level, {})
    
    def validate_data_sufficiency(self, data_length: int, required_warmup: int) -> Tuple[bool, str]:
        """
        Validate if available data is sufficient for accurate calculations.
        
        Args:
            data_length: Available data length in candles
            required_warmup: Required warmup period
            
        Returns:
            (is_sufficient, warning_message)
        """
        if data_length >= required_warmup:
            confidence = min(100, (data_length / required_warmup) * 95)
            return True, f"Sufficient data available. Confidence: {confidence:.1f}%"
        
        else:
            available_accuracy = (data_length / required_warmup) * 95
            shortage = required_warmup - data_length
            
            warning = (f"Insufficient data for {self.accuracy_level} accuracy. "
                      f"Need {shortage} more candles. "
                      f"Current accuracy estimate: {available_accuracy:.1f}%")
            
            return False, warning


def calculate_indicator_warmup(indicator_type: str, period: int, accuracy_level: str = 'good') -> int:
    """
    Convenience function to calculate warmup period for a single indicator.
    
    Args:
        indicator_type: Type of indicator ('EMA', 'RSI', etc.)
        period: Indicator period
        accuracy_level: Desired accuracy level
        
    Returns:
        Required warmup period in candles
    """
    calculator = WarmupCalculator(accuracy_level)
    return calculator.calculate_warmup_period(indicator_type, period)


def get_warmup_examples() -> Dict:
    """
    Get examples of warmup periods for common indicators.
    
    Returns:
        Dictionary with warmup examples
    """
    calculator = WarmupCalculator('good')  # 95% accuracy level
    
    examples = {
        'EMA': {
            'EMA9': calculator.calculate_warmup_period('EMA', 9),    # 36 candles
            'EMA21': calculator.calculate_warmup_period('EMA', 21),  # 84 candles
            'EMA50': calculator.calculate_warmup_period('EMA', 50),  # 200 candles
            'EMA200': calculator.calculate_warmup_period('EMA', 200) # 800 candles
        },
        'RSI': {
            'RSI14': calculator.calculate_warmup_period('RSI', 14),  # 56 candles
        },
        'SMA': {
            'SMA20': calculator.calculate_warmup_period('SMA', 20),  # 80 candles
            'SMA50': calculator.calculate_warmup_period('SMA', 50),  # 200 candles
        },
        'Other': {
            'BollingerBands20': calculator.calculate_warmup_period('BB', 20),     # 80 candles
            'Supertrend10': calculator.calculate_warmup_period('SUPERTREND', 10), # 50 candles
            'VWAP': calculator.calculate_warmup_period('VWAP', 1),                # 50 candles
        }
    }
    
    return examples


if __name__ == "__main__":
    # Example usage and testing
    print("🔥 Indicator Warmup Calculator")
    print("=" * 50)
    
    # Test different accuracy levels
    for level in ['basic', 'good', 'high', 'conservative']:
        calc = WarmupCalculator(level)
        ema21_warmup = calc.calculate_warmup_period('EMA', 21)
        rsi14_warmup = calc.calculate_warmup_period('RSI', 14)
        
        print(f"\n{level.upper()} accuracy ({calc.get_accuracy_info()['accuracy']}):")
        print(f"  EMA21 warmup: {ema21_warmup} candles")
        print(f"  RSI14 warmup: {rsi14_warmup} candles")
    
    # Show examples
    print(f"\n📊 WARMUP EXAMPLES (Good/95% accuracy):")
    examples = get_warmup_examples()
    for category, indicators in examples.items():
        print(f"\n{category}:")
        for name, warmup in indicators.items():
            print(f"  {name}: {warmup} candles")