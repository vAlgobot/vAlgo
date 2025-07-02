#!/usr/bin/env python3
"""
Bollinger Bands Indicator
========================

Bollinger Bands consist of a middle line (SMA) and two outer bands that are standard deviations
away from the middle line. They help identify overbought/oversold conditions and volatility.

Features:
- Configurable period and standard deviation multiplier
- Band squeeze detection
- Breakout signals
- %B and Bandwidth calculations
- Excel configuration integration

Author: vAlgo Development Team
Created: June 28, 2025
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class BollingerBands:
    """
    Bollinger Bands Calculator
    
    Calculates Bollinger Bands with additional metrics like %B and Bandwidth.
    """
    
    def __init__(self, period: int = 20, std_dev: float = 2.0):
        """
        Initialize Bollinger Bands calculator.
        
        Args:
            period: Moving average period (default: 20)
            std_dev: Standard deviation multiplier (default: 2.0)
        """
        if period <= 0:
            raise ValueError("Period must be positive")
        if std_dev <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
            
        self.period = period
        self.std_dev = std_dev
    
    def calculate(self, data: pd.DataFrame, column: str = 'close') -> pd.DataFrame:
        """
        Calculate Bollinger Bands for the given price data.
        
        Args:
            data: DataFrame with price data
            column: Column name to calculate bands on (default: 'close')
            
        Returns:
            DataFrame with Bollinger Bands values added
            
        Raises:
            ValueError: If specified column is missing or insufficient data
        """
        if column not in data.columns:
            raise ValueError(f"Column '{column}' not found in data")
        
        if data.empty:
            return data
        
        if len(data) < self.period:
            raise ValueError(f"Insufficient data. Need at least {self.period} periods")
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Calculate middle band (SMA)
        df['bb_middle'] = df[column].rolling(window=self.period).mean()
        
        # Calculate standard deviation
        df['bb_std'] = df[column].rolling(window=self.period).std()
        
        # Calculate upper and lower bands
        df['bb_upper'] = df['bb_middle'] + (self.std_dev * df['bb_std'])
        df['bb_lower'] = df['bb_middle'] - (self.std_dev * df['bb_std'])
        
        # Calculate additional metrics
        df = self._calculate_percent_b(df, column)
        df = self._calculate_bandwidth(df)
        df = self._generate_signals(df, column)
        
        # Clean up intermediate columns
        df = df.drop(['bb_std'], axis=1)
        
        return df
    
    def _calculate_percent_b(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """
        Calculate %B (Percent B) - position of price within the bands.
        
        %B = (Price - Lower Band) / (Upper Band - Lower Band)
        """
        df['bb_percent_b'] = (df[column] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
        
        # Handle division by zero (when bands are too close)
        df['bb_percent_b'] = df['bb_percent_b'].replace([np.inf, -np.inf], np.nan)
        
        return df
    
    def _calculate_bandwidth(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate Bandwidth - measure of band width relative to middle band.
        
        Bandwidth = (Upper Band - Lower Band) / Middle Band
        """
        df['bb_bandwidth'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Generate Bollinger Bands trading signals."""
        # Price position relative to bands
        df['bb_above_upper'] = (df[column] > df['bb_upper']).astype(int)
        df['bb_below_lower'] = (df[column] < df['bb_lower']).astype(int)
        df['bb_within_bands'] = ((df[column] >= df['bb_lower']) & (df[column] <= df['bb_upper'])).astype(int)
        
        # Band breakout signals
        df['bb_upper_breakout'] = ((df[column] > df['bb_upper']) & 
                                  (df[column].shift(1) <= df['bb_upper'].shift(1))).astype(int)
        
        df['bb_lower_breakout'] = ((df[column] < df['bb_lower']) & 
                                  (df[column].shift(1) >= df['bb_lower'].shift(1))).astype(int)
        
        # Mean reversion signals (returning to middle band)
        df['bb_upper_rejection'] = ((df[column] < df['bb_upper']) & 
                                   (df[column].shift(1) >= df['bb_upper'].shift(1)) & 
                                   (df[column].shift(1) > df['bb_middle'].shift(1))).astype(int)
        
        df['bb_lower_bounce'] = ((df[column] > df['bb_lower']) & 
                                (df[column].shift(1) <= df['bb_lower'].shift(1)) & 
                                (df[column].shift(1) < df['bb_middle'].shift(1))).astype(int)
        
        # Overall signal based on %B
        df['bb_signal'] = 0
        
        # Overbought condition: %B > 1 (above upper band)
        df.loc[df['bb_percent_b'] > 1, 'bb_signal'] = -1
        
        # Oversold condition: %B < 0 (below lower band)
        df.loc[df['bb_percent_b'] < 0, 'bb_signal'] = 1
        
        # Neutral zone: 0.2 < %B < 0.8
        neutral_mask = (df['bb_percent_b'] > 0.2) & (df['bb_percent_b'] < 0.8)
        df.loc[neutral_mask, 'bb_signal'] = 0
        
        return df
    
    def get_squeeze_signals(self, data: pd.DataFrame, lookback: int = 20, threshold: float = 0.1) -> pd.DataFrame:
        """
        Detect Bollinger Band squeeze conditions.
        
        Args:
            data: DataFrame with Bollinger Bands calculated
            lookback: Lookback period for squeeze detection
            threshold: Bandwidth threshold for squeeze detection
            
        Returns:
            DataFrame with squeeze signals
        """
        if 'bb_bandwidth' not in data.columns:
            raise ValueError("Bollinger Bands not calculated. Run calculate() first.")
        
        df = data.copy()
        
        # Calculate minimum bandwidth over lookback period
        df['bb_bandwidth_min'] = df['bb_bandwidth'].rolling(window=lookback).min()
        
        # Squeeze condition: current bandwidth is minimum of lookback period
        df['bb_squeeze'] = (df['bb_bandwidth'] == df['bb_bandwidth_min']).astype(int)
        
        # Enhanced squeeze: bandwidth also below threshold
        df['bb_tight_squeeze'] = ((df['bb_squeeze'] == 1) & 
                                 (df['bb_bandwidth'] < threshold)).astype(int)
        
        # Squeeze release: bandwidth expanding after squeeze
        df['bb_squeeze_release'] = ((df['bb_squeeze'].shift(1) == 1) & 
                                   (df['bb_squeeze'] == 0) & 
                                   (df['bb_bandwidth'] > df['bb_bandwidth'].shift(1))).astype(int)
        
        return df
    
    def get_volatility_signals(self, data: pd.DataFrame, high_vol_threshold: float = 0.15, 
                              low_vol_threshold: float = 0.05) -> pd.DataFrame:
        """
        Detect high and low volatility conditions based on bandwidth.
        
        Args:
            data: DataFrame with Bollinger Bands calculated
            high_vol_threshold: Bandwidth threshold for high volatility
            low_vol_threshold: Bandwidth threshold for low volatility
            
        Returns:
            DataFrame with volatility signals
        """
        if 'bb_bandwidth' not in data.columns:
            raise ValueError("Bollinger Bands not calculated. Run calculate() first.")
        
        df = data.copy()
        
        # High volatility: bandwidth above threshold
        df['bb_high_volatility'] = (df['bb_bandwidth'] > high_vol_threshold).astype(int)
        
        # Low volatility: bandwidth below threshold
        df['bb_low_volatility'] = (df['bb_bandwidth'] < low_vol_threshold).astype(int)
        
        # Volatility expansion: bandwidth increasing significantly
        df['bb_vol_expansion'] = ((df['bb_bandwidth'] > df['bb_bandwidth'].shift(1) * 1.2) & 
                                 (df['bb_bandwidth'] > low_vol_threshold)).astype(int)
        
        # Volatility contraction: bandwidth decreasing significantly
        df['bb_vol_contraction'] = ((df['bb_bandwidth'] < df['bb_bandwidth'].shift(1) * 0.8) & 
                                   (df['bb_bandwidth'] < high_vol_threshold)).astype(int)
        
        return df
    
    def get_price_targets(self, data: pd.DataFrame, current_price: float) -> dict:
        """
        Calculate potential price targets based on Bollinger Bands.
        
        Args:
            data: DataFrame with Bollinger Bands calculated
            current_price: Current market price
            
        Returns:
            Dictionary with price targets and levels
        """
        if 'bb_upper' not in data.columns or 'bb_lower' not in data.columns:
            raise ValueError("Bollinger Bands not calculated. Run calculate() first.")
        
        # Get latest band values
        latest = data.iloc[-1]
        
        targets = {
            'upper_band': latest['bb_upper'],
            'middle_band': latest['bb_middle'],
            'lower_band': latest['bb_lower'],
            'current_position': 'unknown'
        }
        
        # Determine current position
        if current_price > latest['bb_upper']:
            targets['current_position'] = 'above_upper'
            targets['immediate_support'] = latest['bb_upper']
            targets['next_support'] = latest['bb_middle']
        elif current_price < latest['bb_lower']:
            targets['current_position'] = 'below_lower'
            targets['immediate_resistance'] = latest['bb_lower']
            targets['next_resistance'] = latest['bb_middle']
        else:
            targets['current_position'] = 'within_bands'
            targets['resistance'] = latest['bb_upper']
            targets['support'] = latest['bb_lower']
        
        # Calculate band width as percentage
        targets['band_width_pct'] = ((latest['bb_upper'] - latest['bb_lower']) / latest['bb_middle']) * 100
        
        return targets
    
    def get_column_names(self) -> list:
        """
        Get list of Bollinger Bands column names that will be created.
        
        Returns:
            List of Bollinger Bands column names
        """
        return [
            'bb_upper', 'bb_middle', 'bb_lower', 'bb_percent_b', 'bb_bandwidth',
            'bb_above_upper', 'bb_below_lower', 'bb_within_bands',
            'bb_upper_breakout', 'bb_lower_breakout', 
            'bb_upper_rejection', 'bb_lower_bounce', 'bb_signal'
        ]
    
    def __str__(self) -> str:
        """String representation of Bollinger Bands indicator."""
        return f"BollingerBands(period={self.period}, std_dev={self.std_dev})"
    
    def __repr__(self) -> str:
        """Detailed representation of Bollinger Bands indicator."""
        return f"BollingerBands(period={self.period}, std_dev={self.std_dev})"


def calculate_bollinger_bands(data: pd.DataFrame, period: int = 20, std_dev: float = 2.0, 
                             column: str = 'close') -> pd.DataFrame:
    """
    Convenience function to calculate Bollinger Bands.
    
    Args:
        data: DataFrame with price data
        period: Moving average period
        std_dev: Standard deviation multiplier
        column: Column name to calculate bands on
        
    Returns:
        DataFrame with Bollinger Bands values added
    """
    bb = BollingerBands(period=period, std_dev=std_dev)
    return bb.calculate(data, column=column)