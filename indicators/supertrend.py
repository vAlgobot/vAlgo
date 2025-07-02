#!/usr/bin/env python3
"""
Supertrend Indicator
===================

Supertrend is a trend-following indicator based on Average True Range (ATR).
It provides dynamic support and resistance levels that adapt to market volatility.

Features:
- ATR-based calculation
- Configurable period and multiplier
- Trend direction signals
- Excel configuration integration

Author: vAlgo Development Team
Created: June 28, 2025
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple


class Supertrend:
    """
    Supertrend Indicator Calculator
    
    Calculates Supertrend levels and trend direction based on ATR.
    """
    
    def __init__(self, period: int = 10, multiplier: float = 3.0):
        """
        Initialize Supertrend calculator.
        
        Args:
            period: ATR calculation period (default: 10)
            multiplier: ATR multiplier for Supertrend calculation (default: 3.0)
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        if multiplier <= 0:
            raise ValueError("Multiplier must be positive")
            
        self.period = period
        self.multiplier = multiplier
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Supertrend indicator.
        
        Args:
            data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close']
            
        Returns:
            DataFrame with Supertrend values and signals added
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate input data
        required_columns = ['high', 'low', 'close']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            return data
        
        if len(data) < self.period:
            # Return data with NaN values instead of raising error for insufficient data
            df = data.copy()
            df['atr'] = np.nan
            df['supertrend'] = np.nan
            df['supertrend_signal'] = 0
            return df
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Calculate ATR (Average True Range)
        df = self._calculate_atr(df)
        
        # Calculate Supertrend
        df = self._calculate_supertrend(df)
        
        return df
    
    def _calculate_atr(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Average True Range (ATR)."""
        # Calculate True Range components
        df['tr1'] = df['high'] - df['low']
        df['tr2'] = abs(df['high'] - df['close'].shift(1))
        df['tr3'] = abs(df['low'] - df['close'].shift(1))
        
        # True Range is the maximum of the three components
        df['true_range'] = df[['tr1', 'tr2', 'tr3']].max(axis=1)
        
        # Calculate ATR using Simple Moving Average
        df['atr'] = df['true_range'].rolling(window=self.period).mean()
        
        # Clean up intermediate columns
        df = df.drop(['tr1', 'tr2', 'tr3', 'true_range'], axis=1)
        
        return df
    
    def _calculate_supertrend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Supertrend levels and signals."""
        # Calculate basic upper and lower bands
        hl_avg = (df['high'] + df['low']) / 2
        df['basic_upper'] = hl_avg + (self.multiplier * df['atr'])
        df['basic_lower'] = hl_avg - (self.multiplier * df['atr'])
        
        # Initialize final upper and lower bands
        df['final_upper'] = 0.0
        df['final_lower'] = 0.0
        df['supertrend'] = 0.0
        df['supertrend_signal'] = 0  # 1 for bullish, -1 for bearish
        
        # Reset index to use integer indices for iteration
        df = df.reset_index(drop=False)  # Keep original index as column if needed
        
        # Calculate final bands and Supertrend
        for i in range(len(df)):
            if i == 0:
                # First row
                df.loc[i, 'final_upper'] = df.loc[i, 'basic_upper']
                df.loc[i, 'final_lower'] = df.loc[i, 'basic_lower']
            else:
                # Final Upper Band
                if (df.loc[i, 'basic_upper'] < df.loc[i-1, 'final_upper']) or (df.loc[i-1, 'close'] > df.loc[i-1, 'final_upper']):
                    df.loc[i, 'final_upper'] = df.loc[i, 'basic_upper']
                else:
                    df.loc[i, 'final_upper'] = df.loc[i-1, 'final_upper']
                
                # Final Lower Band
                if (df.loc[i, 'basic_lower'] > df.loc[i-1, 'final_lower']) or (df.loc[i-1, 'close'] < df.loc[i-1, 'final_lower']):
                    df.loc[i, 'final_lower'] = df.loc[i, 'basic_lower']
                else:
                    df.loc[i, 'final_lower'] = df.loc[i-1, 'final_lower']
        
        # Determine Supertrend direction
        for i in range(len(df)):
            if i == 0:
                # First row - assume bullish
                df.loc[i, 'supertrend'] = df.loc[i, 'final_lower']
                df.loc[i, 'supertrend_signal'] = 1
            else:
                prev_supertrend = df.loc[i-1, 'supertrend']
                prev_close = df.loc[i-1, 'close']
                current_close = df.loc[i, 'close']
                
                # Determine current Supertrend
                if (prev_supertrend == df.loc[i-1, 'final_upper']) and (current_close <= df.loc[i, 'final_upper']):
                    df.loc[i, 'supertrend'] = df.loc[i, 'final_upper']
                    df.loc[i, 'supertrend_signal'] = -1
                elif (prev_supertrend == df.loc[i-1, 'final_upper']) and (current_close > df.loc[i, 'final_upper']):
                    df.loc[i, 'supertrend'] = df.loc[i, 'final_lower']
                    df.loc[i, 'supertrend_signal'] = 1
                elif (prev_supertrend == df.loc[i-1, 'final_lower']) and (current_close >= df.loc[i, 'final_lower']):
                    df.loc[i, 'supertrend'] = df.loc[i, 'final_lower']
                    df.loc[i, 'supertrend_signal'] = 1
                elif (prev_supertrend == df.loc[i-1, 'final_lower']) and (current_close < df.loc[i, 'final_lower']):
                    df.loc[i, 'supertrend'] = df.loc[i, 'final_upper']
                    df.loc[i, 'supertrend_signal'] = -1
                else:
                    # Fallback
                    if current_close <= df.loc[i, 'final_upper']:
                        df.loc[i, 'supertrend'] = df.loc[i, 'final_upper']
                        df.loc[i, 'supertrend_signal'] = -1
                    else:
                        df.loc[i, 'supertrend'] = df.loc[i, 'final_lower']
                        df.loc[i, 'supertrend_signal'] = 1
        
        # Clean up intermediate columns
        df = df.drop(['basic_upper', 'basic_lower', 'final_upper', 'final_lower'], axis=1)
        
        # Restore original index if it was saved
        if 'timestamp' in df.columns and pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df = df.set_index('timestamp')
        elif hasattr(data, 'index') and hasattr(data.index, 'name') and data.index.name:
            # Try to restore to original index structure
            df = df.set_index(df.columns[0]) if len(df.columns) > 3 else df
        
        return df
    
    def get_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Get buy/sell signals based on Supertrend crossovers.
        
        Args:
            data: DataFrame with Supertrend calculated
            
        Returns:
            DataFrame with signal columns added
        """
        if 'supertrend_signal' not in data.columns:
            raise ValueError("Supertrend not calculated. Run calculate() first.")
        
        df = data.copy()
        
        # Detect signal changes
        df['signal_change'] = df['supertrend_signal'].diff()
        
        # Buy signal: bearish to bullish (signal changes from -1 to 1)
        df['buy_signal'] = (df['signal_change'] == 2).astype(int)
        
        # Sell signal: bullish to bearish (signal changes from 1 to -1)
        df['sell_signal'] = (df['signal_change'] == -2).astype(int)
        
        return df
    
    def get_trend_direction(self, data: pd.DataFrame) -> pd.Series:
        """
        Get trend direction as string.
        
        Args:
            data: DataFrame with Supertrend calculated
            
        Returns:
            Series with trend direction ('bullish', 'bearish')
        """
        if 'supertrend_signal' not in data.columns:
            raise ValueError("Supertrend not calculated. Run calculate() first.")
        
        return data['supertrend_signal'].map({1: 'bullish', -1: 'bearish'})
    
    def get_support_resistance(self, data: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Get dynamic support and resistance levels.
        
        Args:
            data: DataFrame with Supertrend calculated
            
        Returns:
            Tuple of (support_levels, resistance_levels)
        """
        if 'supertrend' not in data.columns or 'supertrend_signal' not in data.columns:
            raise ValueError("Supertrend not calculated. Run calculate() first.")
        
        # When trend is bullish, Supertrend acts as support
        # When trend is bearish, Supertrend acts as resistance
        support = data['supertrend'].where(data['supertrend_signal'] == 1)
        resistance = data['supertrend'].where(data['supertrend_signal'] == -1)
        
        return support, resistance
    
    def __str__(self) -> str:
        """String representation of Supertrend indicator."""
        return f"Supertrend(period={self.period}, multiplier={self.multiplier})"
    
    def __repr__(self) -> str:
        """Detailed representation of Supertrend indicator."""
        return f"Supertrend(period={self.period}, multiplier={self.multiplier})"


def calculate_supertrend(data: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
    """
    Convenience function to calculate Supertrend.
    
    Args:
        data: DataFrame with OHLC data
        period: ATR calculation period
        multiplier: ATR multiplier
        
    Returns:
        DataFrame with Supertrend values added
    """
    supertrend = Supertrend(period=period, multiplier=multiplier)
    return supertrend.calculate(data)