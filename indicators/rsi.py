#!/usr/bin/env python3
"""
Relative Strength Index (RSI) Indicator
=======================================

RSI is a momentum oscillator that measures the speed and change of price movements.
It oscillates between 0 and 100 and is typically used to identify overbought and oversold conditions.

Features:
- Standard RSI calculation with Wilder's smoothing
- Configurable period (default: 14)
- Overbought/oversold signal detection
- Excel configuration integration

Author: vAlgo Development Team
Created: June 28, 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class RSI:
    """
    Relative Strength Index Calculator
    
    Calculates RSI using Wilder's smoothing method for accurate momentum analysis.
    """
    
    def __init__(self, period: int = 14, overbought: float = 70.0, oversold: float = 30.0):
        """
        Initialize RSI calculator.
        
        Args:
            period: RSI calculation period (default: 14)
            overbought: Overbought threshold (default: 70.0)
            oversold: Oversold threshold (default: 30.0)
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        if not 0 <= oversold < overbought <= 100:
            raise ValueError("Invalid threshold values. Must be: 0 <= oversold < overbought <= 100")
            
        self.period = period
        self.overbought = overbought
        self.oversold = oversold
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Calculate RSI for the given price data.
        
        Args:
            data: DataFrame with price data
            column: Column name to calculate RSI on (default: 'close')
            
        Returns:
            DataFrame with RSI values and signals added
            
        Raises:
            ValueError: If specified column is missing or insufficient data
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if data.empty:
            return data
        
        if len(data) < self.period + 1:
            raise ValueError(f"Insufficient data. Need at least {self.period + 1} periods")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Calculate price changes
        df['price_change'] = df[column].diff()
        
        # Separate gains and losses
        df['gain'] = df['price_change'].where(df['price_change'] > 0, 0)
        df['loss'] = abs(df['price_change'].where(df['price_change'] < 0, 0))
        
        # Calculate RSI using Wilder's smoothing method
        df = self._calculate_rsi_wilder(df)
        
        # Generate signals
        df = self._generate_signals(df)
        
        # Clean up intermediate columns
        df = df.drop(['price_change', 'gain', 'loss', 'avg_gain', 'avg_loss'], axis=1)
        
        return df
    
    def _calculate_rsi_wilder(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate RSI using Wilder's smoothing method.
        Updated for higher accuracy to match reference implementations."""
        
        # Initialize columns
        df['avg_gain'] = np.nan
        df['avg_loss'] = np.nan
        df['rsi'] = np.nan
        
        # Skip first row (no price change available)
        if len(df) < self.period + 1:
            return df
        
        # Calculate initial averages using simple mean (more accurate initial calculation)
        # Use iloc[1:period+1] to get exactly 'period' values, skipping the first NaN
        initial_gains = df['gain'].iloc[1:self.period+1]
        initial_losses = df['loss'].iloc[1:self.period+1]
        
        # First average (simple average for first period)
        first_avg_gain = initial_gains.mean()
        first_avg_loss = initial_losses.mean()
        
        # Avoid division by zero and ensure proper handling
        if first_avg_loss == 0:
            if first_avg_gain == 0:
                first_rsi = 50.0  # Neutral when no movement
            else:
                first_rsi = 100.0  # Maximum when only gains
        else:
            rs = first_avg_gain / first_avg_loss
            first_rsi = 100.0 - (100.0 / (1.0 + rs))
        
        # Set first calculated values at position 'period' (0-indexed)
        df.iloc[self.period, df.columns.get_loc('avg_gain')] = first_avg_gain
        df.iloc[self.period, df.columns.get_loc('avg_loss')] = first_avg_loss
        df.iloc[self.period, df.columns.get_loc('rsi')] = first_rsi
        
        # Calculate subsequent values using Wilder's exponential smoothing
        for i in range(self.period + 1, len(df)):
            # Wilder's smoothing formula with proper precision
            prev_avg_gain = df.iloc[i-1, df.columns.get_loc('avg_gain')]
            prev_avg_loss = df.iloc[i-1, df.columns.get_loc('avg_loss')]
            current_gain = df.iloc[i, df.columns.get_loc('gain')]
            current_loss = df.iloc[i, df.columns.get_loc('loss')]
            
            # Wilder's smoothing: more precise calculation
            current_avg_gain = ((prev_avg_gain * (self.period - 1)) + current_gain) / self.period
            current_avg_loss = ((prev_avg_loss * (self.period - 1)) + current_loss) / self.period
            
            # Store averages
            df.iloc[i, df.columns.get_loc('avg_gain')] = current_avg_gain
            df.iloc[i, df.columns.get_loc('avg_loss')] = current_avg_loss
            
            # Calculate RSI with proper precision
            if current_avg_loss == 0:
                if current_avg_gain == 0:
                    rsi = 50.0
                else:
                    rsi = 100.0
            else:
                rs = current_avg_gain / current_avg_loss
                rsi = 100.0 - (100.0 / (1.0 + rs))
            
            df.iloc[i, df.columns.get_loc('rsi')] = rsi
        
        # Round RSI to 2 decimal places for consistency with reference
        df['rsi'] = df['rsi'].round(2)
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate RSI-based trading signals."""
        # Overbought and oversold conditions
        df['rsi_overbought'] = (df['rsi'] >= self.overbought).astype(int)
        df['rsi_oversold'] = (df['rsi'] <= self.oversold).astype(int)
        
        # Signal generation
        df['rsi_signal'] = 0
        
        # Bullish signal: RSI crosses above oversold level
        df['rsi_buy_signal'] = ((df['rsi'] > self.oversold) & 
                               (df['rsi'].shift(1) <= self.oversold)).astype(int)
        
        # Bearish signal: RSI crosses below overbought level
        df['rsi_sell_signal'] = ((df['rsi'] < self.overbought) & 
                                (df['rsi'].shift(1) >= self.overbought)).astype(int)
        
        # Overall signal: 1 for buy, -1 for sell, 0 for hold
        df.loc[df['rsi_buy_signal'] == 1, 'rsi_signal'] = 1
        df.loc[df['rsi_sell_signal'] == 1, 'rsi_signal'] = -1
        
        return df
    
    def get_divergence_signals(self, data: pd.DataFrame, price_column: str = 'close') -> pd.DataFrame:
        """
        Detect RSI divergence signals.
        
        Args:
            data: DataFrame with RSI and price data
            price_column: Column name for price data
            
        Returns:
            DataFrame with divergence signals
        """
        if 'rsi' not in data.columns:
            raise ValueError("RSI not calculated. Run calculate() first.")
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        df = data.copy()
        
        # Find local highs and lows for both price and RSI
        window = 5  # Look for peaks/troughs in 5-period window
        
        # Price highs and lows
        df['price_high'] = df[price_column].rolling(window=window, center=True).max() == df[price_column]
        df['price_low'] = df[price_column].rolling(window=window, center=True).min() == df[price_column]
        
        # RSI highs and lows
        df['rsi_high'] = df['rsi'].rolling(window=window, center=True).max() == df['rsi']
        df['rsi_low'] = df['rsi'].rolling(window=window, center=True).min() == df['rsi']
        
        # Initialize divergence signals
        df['bullish_divergence'] = 0
        df['bearish_divergence'] = 0
        
        # Simple divergence detection (can be enhanced with more sophisticated algorithms)
        price_highs = df[df['price_high']]
        price_lows = df[df['price_low']]
        rsi_highs = df[df['rsi_high']]
        rsi_lows = df[df['rsi_low']]
        
        # Mark areas where divergence might occur
        # This is a simplified version - production code would need more sophisticated detection
        for i in range(len(df)):
            if df.loc[i, 'price_low'] and df.loc[i, 'rsi_low']:
                # Look for previous lows to compare
                prev_lows = df.loc[:i][df.loc[:i, 'price_low'] & df.loc[:i, 'rsi_low']]
                if len(prev_lows) > 0:
                    last_low_idx = prev_lows.index[-1]
                    # Bullish divergence: price makes lower low, RSI makes higher low
                    if (df.loc[i, price_column] < df.loc[last_low_idx, price_column] and 
                        df.loc[i, 'rsi'] > df.loc[last_low_idx, 'rsi']):
                        df.loc[i, 'bullish_divergence'] = 1
            
            elif df.loc[i, 'price_high'] and df.loc[i, 'rsi_high']:
                # Look for previous highs to compare
                prev_highs = df.loc[:i][df.loc[:i, 'price_high'] & df.loc[:i, 'rsi_high']]
                if len(prev_highs) > 0:
                    last_high_idx = prev_highs.index[-1]
                    # Bearish divergence: price makes higher high, RSI makes lower high
                    if (df.loc[i, price_column] > df.loc[last_high_idx, price_column] and 
                        df.loc[i, 'rsi'] < df.loc[last_high_idx, 'rsi']):
                        df.loc[i, 'bearish_divergence'] = 1
        
        return df
    
    def get_rsi_trend(self, data: pd.DataFrame) -> pd.Series:
        """
        Determine RSI trend direction.
        
        Args:
            data: DataFrame with RSI calculated
            
        Returns:
            Series with trend direction ('bullish', 'bearish', 'neutral')
        """
        if 'rsi' not in data.columns:
            raise ValueError("RSI not calculated. Run calculate() first.")
        
        conditions = [
            data['rsi'] > self.overbought,
            data['rsi'] < self.oversold,
            (data['rsi'] >= 50) & (data['rsi'] < self.overbought),
            (data['rsi'] > self.oversold) & (data['rsi'] < 50)
        ]
        
        choices = ['overbought', 'oversold', 'bullish', 'bearish']
        
        return pd.Series(np.select(conditions, choices, default='neutral'), index=data.index)
    
    def get_column_names(self) -> list:
        """
        Get list of RSI column names that will be created.
        
        Returns:
            List of RSI column names
        """
        return [
            'rsi', 'rsi_overbought', 'rsi_oversold', 
            'rsi_buy_signal', 'rsi_sell_signal', 'rsi_signal'
        ]
    
    def __str__(self) -> str:
        """String representation of RSI indicator."""
        return f"RSI(period={self.period}, overbought={self.overbought}, oversold={self.oversold})"
    
    def __repr__(self) -> str:
        """Detailed representation of RSI indicator."""
        return f"RSI(period={self.period}, overbought={self.overbought}, oversold={self.oversold})"


def calculate_rsi(data: pd.DataFrame, period: int = 14, column: str = 'close', 
                  overbought: float = 70.0, oversold: float = 30.0) -> pd.DataFrame:
    """
    Convenience function to calculate RSI.
    
    Args:
        data: DataFrame with price data
        period: RSI calculation period
        column: Column name to calculate RSI on
        overbought: Overbought threshold
        oversold: Oversold threshold
        
    Returns:
        DataFrame with RSI values added
    """
    rsi = RSI(period=period, overbought=overbought, oversold=oversold)
    return rsi.calculate(data, column=column)