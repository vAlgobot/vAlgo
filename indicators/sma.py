#!/usr/bin/env python3
"""
Simple Moving Average (SMA) Indicator
=====================================

Simple Moving Average calculates the arithmetic mean of prices over a specific number of periods.
It's one of the most fundamental and widely used technical indicators.

Features:
- Multiple period support (9, 21, 50, 200)
- Excel configuration integration
- Crossover detection
- Trend analysis
- Support/resistance identification

Author: vAlgo Development Team
Created: June 28, 2025
"""

import pandas as pd
import numpy as np
from typing import List, Union, Optional


class SMA:
    """
    Simple Moving Average Calculator
    
    Calculates SMA for single or multiple periods with trend analysis capabilities.
    """
    
    def __init__(self, periods: Union[int, List[int]] = [9, 21, 50, 200]):
        """
        Initialize SMA calculator.
        
        Args:
            periods: Single period or list of periods for SMA calculation
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
        Calculate SMA for specified periods.
        
        Args:
            data: DataFrame with price data
            column: Column name to calculate SMA on (default: 'close')
            
        Returns:
            DataFrame with SMA columns added
            
        Raises:
            ValueError: If specified column is missing
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if data.empty:
            return data
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Calculate SMA for each period
        for period in self.periods:
            if len(df) < period:
                # Not enough data for this period
                df[f'sma_{period}'] = np.nan
            else:
                # Calculate SMA using pandas rolling mean
                df[f'sma_{period}'] = df[column].rolling(window=period).mean()
        
        return df
    
    def calculate_single(self, data: pd.DataFrame, period: int, column: str = 'close') -> pd.Series:
        """
        Calculate SMA for a single period.
        
        Args:
            data: DataFrame with price data
            period: SMA period
            column: Column name to calculate SMA on
            
        Returns:
            Series with SMA values
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if period <= 0:
            raise ValueError("Period must be positive")
        
        if data.empty or len(data) < period:
            return pd.Series(np.nan, index=data.index)
        
        return data[column].rolling(window=period).mean()
    
    def get_crossovers(self, data: pd.DataFrame, fast_period: int, slow_period: int) -> pd.DataFrame:
        """
        Detect SMA crossovers between two periods.
        
        Args:
            data: DataFrame with SMA values calculated
            fast_period: Faster SMA period
            slow_period: Slower SMA period
            
        Returns:
            DataFrame with crossover signals
        """
        fast_col = f'sma_{fast_period}'
        slow_col = f'sma_{slow_period}'
        
        if fast_col not in data.columns or slow_col not in data.columns:
            raise ValueError(f"SMA columns not found. Ensure SMAs are calculated for periods {fast_period} and {slow_period}")
        
        if fast_period >= slow_period:
            raise ValueError("Fast period must be less than slow period")
        
        df = data.copy()
        
        # Calculate crossover signals
        df['sma_diff'] = df[fast_col] - df[slow_col]
        df['sma_diff_prev'] = df['sma_diff'].shift(1)
        
        # Golden cross: fast SMA crosses above slow SMA
        df['golden_cross'] = ((df['sma_diff'] > 0) & (df['sma_diff_prev'] <= 0)).astype(int)
        
        # Death cross: fast SMA crosses below slow SMA
        df['death_cross'] = ((df['sma_diff'] < 0) & (df['sma_diff_prev'] >= 0)).astype(int)
        
        # Overall signal: 1 for bullish, -1 for bearish, 0 for neutral
        df['sma_signal'] = np.where(df['sma_diff'] > 0, 1, np.where(df['sma_diff'] < 0, -1, 0))
        
        return df
    
    def get_trend_strength(self, data: pd.DataFrame, periods: List[int] = None) -> pd.Series:
        """
        Calculate trend strength based on SMA alignment.
        
        Args:
            data: DataFrame with SMA values calculated
            periods: List of periods to check alignment (default: all calculated periods)
            
        Returns:
            Series with trend strength (-1 to 1)
        """
        if periods is None:
            periods = self.periods
        
        # Check if all required SMA columns exist
        sma_cols = [f'sma_{period}' for period in periods]
        missing_cols = [col for col in sma_cols if col not in data.columns]
        
        if missing_cols:
            raise ValueError(f"Missing SMA columns: {missing_cols}")
        
        if len(periods) < 2:
            raise ValueError("At least 2 periods required for trend strength calculation")
        
        # Sort periods for proper comparison
        sorted_periods = sorted(periods)
        
        # Count how many SMAs are in proper order
        trend_strength = pd.Series(0.0, index=data.index)
        
        for i in range(len(data)):
            bullish_count = 0
            bearish_count = 0
            total_comparisons = len(sorted_periods) - 1
            
            for j in range(len(sorted_periods) - 1):
                fast_sma = data[f'sma_{sorted_periods[j]}'].iloc[i]
                slow_sma = data[f'sma_{sorted_periods[j+1]}'].iloc[i]
                
                if pd.notna(fast_sma) and pd.notna(slow_sma):
                    if fast_sma > slow_sma:
                        bullish_count += 1
                    elif fast_sma < slow_sma:
                        bearish_count += 1
            
            if total_comparisons > 0:
                if bullish_count > bearish_count:
                    trend_strength.iloc[i] = bullish_count / total_comparisons
                elif bearish_count > bullish_count:
                    trend_strength.iloc[i] = -bearish_count / total_comparisons
        
        return trend_strength
    
    def get_support_resistance(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Identify potential support and resistance levels using SMAs.
        
        Args:
            data: DataFrame with SMA values and price data
            price_column: Column name for current price
            
        Returns:
            DataFrame with support/resistance levels
        """
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        df = data.copy()
        
        # Initialize support/resistance columns
        for period in self.periods:
            sma_col = f'sma_{period}'
            if sma_col in df.columns:
                # SMA acts as support when price is above it
                df[f'sma_{period}_support'] = np.where(df[price_column] > df[sma_col], df[sma_col], np.nan)
                
                # SMA acts as resistance when price is below it
                df[f'sma_{period}_resistance'] = np.where(df[price_column] < df[sma_col], df[sma_col], np.nan)
        
        return df
    
    def get_price_distance(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Calculate price distance from SMAs as percentage.
        
        Args:
            data: DataFrame with SMA values and price data
            price_column: Column name for current price
            
        Returns:
            DataFrame with price distance percentages
        """
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        df = data.copy()
        
        # Calculate distance for each SMA
        for period in self.periods:
            sma_col = f'sma_{period}'
            if sma_col in df.columns:
                df[f'sma_{period}_distance_pct'] = ((df[price_column] - df[sma_col]) / df[sma_col]) * 100
        
        return df
    
    def get_sma_slopes(self, data: pd.DataFrame, lookback: int = 3) -> pd.DataFrame:
        """
        Calculate SMA slopes to determine trend direction and strength.
        
        Args:
            data: DataFrame with SMA values calculated
            lookback: Number of periods to look back for slope calculation
            
        Returns:
            DataFrame with SMA slopes
        """
        if lookback <= 0:
            raise ValueError("Lookback period must be positive")
        
        df = data.copy()
        
        # Calculate slope for each SMA
        for period in self.periods:
            sma_col = f'sma_{period}'
            if sma_col in df.columns:
                # Calculate slope as percentage change over lookback period
                df[f'sma_{period}_slope'] = df[sma_col].pct_change(periods=lookback) * 100
                
                # Classify slope direction
                conditions = [
                    df[f'sma_{period}_slope'] > 0.1,  # Rising
                    df[f'sma_{period}_slope'] < -0.1,  # Falling
                ]
                choices = ['rising', 'falling']
                df[f'sma_{period}_trend'] = pd.Series(
                    np.select(conditions, choices, default='sideways'), 
                    index=df.index
                )
        
        return df
    
    def get_column_names(self) -> List[str]:
        """
        Get list of SMA column names that will be created.
        
        Returns:
            List of SMA column names
        """
        return [f'sma_{period}' for period in self.periods]
    
    def __str__(self) -> str:
        """String representation of SMA indicator."""
        return f"SMA(periods={self.periods})"
    
    def __repr__(self) -> str:
        """Detailed representation of SMA indicator."""
        return f"SMA(periods={self.periods})"


def calculate_sma(data: pd.DataFrame, periods: Union[int, List[int]] = [9, 21, 50, 200], column: str = 'close') -> pd.DataFrame:
    """
    Convenience function to calculate SMA.
    
    Args:
        data: DataFrame with price data
        periods: Single period or list of periods
        column: Column name to calculate SMA on
        
    Returns:
        DataFrame with SMA values added
    """
    sma = SMA(periods=periods)
    return sma.calculate(data, column=column)