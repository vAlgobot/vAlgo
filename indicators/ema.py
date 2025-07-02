#!/usr/bin/env python3
"""
Exponential Moving Average (EMA) Indicator
==========================================

Exponential Moving Average gives more weight to recent prices, making it more
responsive to new information compared to Simple Moving Average.

Features:
- Multiple period support (9, 21, 50, 200)
- Excel configuration integration
- Optimized pandas calculation
- Real-time calculation capability

Author: vAlgo Development Team
Created: June 28, 2025
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional


class EMA:
    """
    Exponential Moving Average Calculator
    
    Calculates EMA for single or multiple periods with optimized pandas operations.
    """
    
    def __init__(self, periods: Union[int, List[int]] = [9, 21, 50, 200]):
        """
        Initialize EMA calculator.
        
        Args:
            periods: Single period or list of periods for EMA calculation
        """
        if isinstance(periods, int):
            periods = [periods]
        
        if not periods:
            raise ValueError("At least one period must be specified")
        
        for period in periods:
            if not isinstance(period, int) or period <= 0:
                raise ValueError(f"All periods must be positive integers. Got: {period}")
        
        self.periods = sorted(periods)
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Calculate EMA for specified periods.
        
        Args:
            data: DataFrame with price data
            column: Column name to calculate EMA on (default: 'close')
            
        Returns:
            DataFrame with EMA columns added
            
        Raises:
            ValueError: If specified column is missing
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if data.empty:
            return data
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Calculate EMA for each period
        for period in self.periods:
            if len(df) < period:
                # Not enough data for this period
                df[f'ema_{period}'] = np.nan
            else:
                # Calculate EMA using pandas ewm method
                df[f'ema_{period}'] = df[column].ewm(span=period, adjust=False).mean()
        
        return df
    
    def calculate_single(self, data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Calculate EMA for a single period.
        
        Args:
            data: DataFrame with price data
            period: EMA period
            column: Column name to calculate EMA on
            
        Returns:
            Series with EMA values
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if period <= 0:
            raise ValueError("Period must be positive")
        
        if data.empty or len(data) < period:
            return pd.Series(np.nan, index=data.index)
        
        return data[column].ewm(span=period, adjust=False).mean()
    
    def get_crossovers(self, data: pd.DataFrame, fast_period: int, slow_period: int) -> pd.DataFrame:
        """
        Detect EMA crossovers between two periods.
        
        Args:
            data: DataFrame with EMA values calculated
            fast_period: Faster EMA period
            slow_period: Slower EMA period
            
        Returns:
            DataFrame with crossover signals
        """
        fast_col = f'ema_{fast_period}'
        slow_col = f'ema_{slow_period}'
        
        if fast_col not in data.columns or slow_col not in data.columns:
            raise ValueError(f"EMA columns not found. Ensure EMAs are calculated for periods {fast_period} and {slow_period}")
        
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        df = data.copy()
        
        # Calculate crossover signals
        df['ema_diff'] = df[fast_col] - df[slow_col]
        df['ema_diff_prev'] = df['ema_diff'].shift(1)
        
        # Golden cross: fast EMA crosses above slow EMA
        df['golden_cross'] = ((df['ema_diff'] > 0) & (df['ema_diff_prev'] <= 0)).astype(int)
        
        # Death cross: fast EMA crosses below slow EMA
        df['death_cross'] = ((df['ema_diff'] < 0) & (df['ema_diff_prev'] >= 0)).astype(int)
        
        # Overall signal: 1 for bullish, -1 for bearish, 0 for neutral
        df['ema_signal'] = np.where(df['ema_diff'] > 0, 1, np.where(df['ema_diff'] < 0, -1, 0))
        
        return df
    
    def get_trend_strength(self, data: pd.DataFrame, periods: List[int] = None) -> pd.Series:
        """
        Calculate trend strength based on EMA alignment.
        
        Args:
            data: DataFrame with EMA values calculated
            periods: List of periods to check alignment (default: all calculated periods)
            
        Returns:
            Series with trend strength (0 to 1)
        """
        if periods is None:
            periods = self.periods
        
        # Check if all required EMA columns exist
        ema_cols = [f'ema_{period}' for period in periods]
        missing_cols = [col for col in ema_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing EMA columns: {missing_cols}")
        
        if len(periods) < 2:
            raise ValueError("At least 2 periods required for trend strength calculation")
        
        # Sort periods for proper comparison
        sorted_periods = sorted(periods)
        
        # Count how many EMAs are in proper order
        trend_strength = pd.Series(0.0, index=data.index)
        
        for i in range(len(data)):
            bullish_count = 0
            bearish_count = 0
            total_comparisons = len(sorted_periods) - 1
            
            for j in range(len(sorted_periods) - 1):
                fast_ema = data[f'ema_{sorted_periods[j]}'].iloc[i]
                slow_ema = data[f'ema_{sorted_periods[j+1]}'].iloc[i]
                
                if pd.notna(fast_ema) and pd.notna(slow_ema):
                    if fast_ema > slow_ema:
                        bullish_count += 1
                    elif fast_ema < slow_ema:
                        bearish_count += 1
            
            if total_comparisons > 0:
                if bullish_count > bearish_count:
                    trend_strength.iloc[i] = bullish_count / total_comparisons
                elif bearish_count > bullish_count:
                    trend_strength.iloc[i] = -bearish_count / total_comparisons
        
        return trend_strength
    
    def get_support_resistance(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Identify potential support and resistance levels using EMAs.
        
        Args:
            data: DataFrame with EMA values and price data
            price_column: Column name for current price
            
        Returns:
            DataFrame with support/resistance levels
        """
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        df = data.copy()
        
        # Initialize support/resistance columns
        for period in self.periods:
            ema_col = f'ema_{period}'
            if ema_col in df.columns:
                # EMA acts as support when price is above it
                df[f'ema_{period}_support'] = np.where(df[price_column] > df[ema_col], df[ema_col], np.nan)
                
                # EMA acts as resistance when price is below it
                df[f'ema_{period}_resistance'] = np.where(df[price_column] < df[ema_col], df[ema_col], np.nan)
        
        return df
    
    def get_column_names(self) -> List[str]:
        """
        Get list of EMA column names that will be created.
        
        Returns:
            List of EMA column names
        """
        return [f'ema_{period}' for period in self.periods]
    
    def __str__(self) -> str:
        """String representation of EMA indicator."""
        return f"EMA(periods={self.periods})"
    
    def __repr__(self) -> str:
        """Detailed representation of EMA indicator."""
        return f"EMA(periods={self.periods})"


def calculate_ema(data: pd.DataFrame, periods: Union[int, List[int]] = [9, 21, 50, 200], column: str = 'close') -> pd.DataFrame:
    """
    Convenience function to calculate EMA.
    
    Args:
        data: DataFrame with price data
        periods: Single period or list of periods
        column: Column name to calculate EMA on
        
    Returns:
        DataFrame with EMA values added
    """
    ema = EMA(periods=periods)
    return ema.calculate(data, column=column)