#!/usr/bin/env python3
"""
Volume Weighted Average Price (VWAP) Indicator
==============================================

VWAP is a trading benchmark that gives the average price a security has traded at throughout the day,
based on both volume and price. It's particularly useful for intraday trading strategies.

Features:
- Standard VWAP calculation
- Session-based VWAP (daily, weekly, monthly)
- VWAP bands using standard deviation
- Excel configuration integration

Author: vAlgo Development Team
Created: June 28, 2025
"""

import pandas as pd
import numpy as np
from typing import Optional, Tuple
from datetime import datetime


class VWAP:
    """
    Volume Weighted Average Price Calculator
    
    Calculates VWAP and related metrics for trading analysis.
    """
    
    def __init__(self, session_type: str = 'daily', std_dev_multiplier: float = 1.0):
        """
        Initialize VWAP calculator.
        
        Args:
            session_type: VWAP session type ('daily', 'weekly', 'monthly', 'continuous')
            std_dev_multiplier: Standard deviation multiplier for VWAP bands
        """
        self.session_type = session_type.lower()
        self.valid_sessions = ['daily', 'weekly', 'monthly', 'continuous']
        
        if self.session_type not in self.valid_sessions:
            raise ValueError(f"Invalid session type: {session_type}. Must be one of {self.valid_sessions}")
        
        if std_dev_multiplier <= 0:
            raise ValueError("Standard deviation multiplier must be positive")
            
        self.std_dev_multiplier = std_dev_multiplier
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate VWAP for the given OHLCV data.
        
        Args:
            data: DataFrame with columns ['timestamp', 'high', 'low', 'close', 'volume']
            
        Returns:
            DataFrame with VWAP values and bands added
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate input data
        required_columns = ['timestamp', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in data.columns]
        
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")
        
        if data.empty:
            return data
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Calculate typical price
        df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
        
        # Handle missing or zero volume case
        has_volume = 'volume' in df.columns and df['volume'].notna().any() and df['volume'].sum() > 0
        
        if not has_volume:
            # Generate synthetic volume based on price volatility for VWAP calculation
            # Higher volatility (larger price moves) gets higher volume
            df['price_change'] = abs(df['close'] - df['open'])
            df['price_volatility'] = df['price_change'].rolling(window=5, min_periods=1).mean()
            
            # Normalize volatility to volume range (1000-50000)
            vol_min, vol_max = 1000, 50000
            vol_normalized = (df['price_volatility'] - df['price_volatility'].min()) / (df['price_volatility'].max() - df['price_volatility'].min())
            df['volume'] = vol_min + (vol_normalized * (vol_max - vol_min))
            
            # Fill any NaN values
            df['volume'] = df['volume'].fillna(10000)
            
            # Clean up temporary columns
            df = df.drop(['price_change', 'price_volatility'], axis=1)
            
            print("Warning: No volume data available. Generated synthetic volume based on price volatility for VWAP calculation.")
            print(f"Synthetic volume range: {df['volume'].min():.0f} - {df['volume'].max():.0f}")
        else:
            # Check for any zero volume periods and handle them
            zero_volume_mask = df['volume'] == 0
            if zero_volume_mask.any():
                # Replace zero volume with rolling average volume
                avg_volume = df.loc[df['volume'] > 0, 'volume'].mean()
                df.loc[zero_volume_mask, 'volume'] = avg_volume
                print(f"Warning: {zero_volume_mask.sum()} zero volume periods detected. Replaced with average volume: {avg_volume:.0f}")
        
        # Calculate VWAP based on session type
        if self.session_type == 'continuous':
            df = self._calculate_continuous_vwap(df)
        else:
            df = self._calculate_session_vwap(df)
        
        # Calculate VWAP bands
        df = self._calculate_vwap_bands(df)
        
        # Generate signals
        df = self._generate_signals(df)
        
        # Clean up intermediate columns
        df = df.drop(['typical_price'], axis=1)
        
        return df
    
    def _calculate_continuous_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate continuous VWAP (from start of data)."""
        # Calculate cumulative values
        df['volume_price'] = df['typical_price'] * df['volume']
        df['cumulative_volume_price'] = df['volume_price'].cumsum()
        df['cumulative_volume'] = df['volume'].cumsum()
        
        # Calculate VWAP
        df['vwap'] = df['cumulative_volume_price'] / df['cumulative_volume']
        
        # Clean up intermediate columns
        df = df.drop(['volume_price', 'cumulative_volume_price', 'cumulative_volume'], axis=1)
        
        return df
    
    def _calculate_session_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate session-based VWAP (daily, weekly, monthly)."""
        # Create session grouping
        if self.session_type == 'daily':
            df['session'] = df['timestamp'].dt.date
        elif self.session_type == 'weekly':
            df['session'] = df['timestamp'].dt.to_period('W')
        elif self.session_type == 'monthly':
            df['session'] = df['timestamp'].dt.to_period('M')
        
        # Calculate VWAP for each session
        df['volume_price'] = df['typical_price'] * df['volume']
        
        # Group by session and calculate cumulative values within each session
        df['session_cumulative_volume_price'] = df.groupby('session')['volume_price'].cumsum()
        df['session_cumulative_volume'] = df.groupby('session')['volume'].cumsum()
        
        # Calculate VWAP
        df['vwap'] = df['session_cumulative_volume_price'] / df['session_cumulative_volume']
        
        # Clean up intermediate columns
        df = df.drop([
            'session', 'volume_price', 
            'session_cumulative_volume_price', 'session_cumulative_volume'
        ], axis=1)
        
        return df
    
    def _calculate_vwap_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate VWAP bands using standard deviation."""
        if self.session_type == 'continuous':
            # For continuous VWAP, calculate rolling standard deviation
            window = min(20, len(df))  # Use 20-period or available data
            df['vwap_std'] = df['typical_price'].rolling(window=window).std()
        else:
            # For session-based VWAP, calculate within-session standard deviation
            if self.session_type == 'daily':
                session_col = df['timestamp'].dt.date
            elif self.session_type == 'weekly':
                session_col = df['timestamp'].dt.to_period('W')
            else:  # monthly
                session_col = df['timestamp'].dt.to_period('M')
            
            # Calculate expanding standard deviation within each session
            df['session_group'] = session_col
            df['vwap_std'] = df.groupby('session_group')['typical_price'].expanding().std().reset_index(level=0, drop=True)
            df = df.drop(['session_group'], axis=1)
        
        # Calculate upper and lower bands
        df['vwap_upper'] = df['vwap'] + (self.std_dev_multiplier * df['vwap_std'])
        df['vwap_lower'] = df['vwap'] - (self.std_dev_multiplier * df['vwap_std'])
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate VWAP-based trading signals."""
        # Price position relative to VWAP
        df['above_vwap'] = (df['close'] > df['vwap']).astype(int)
        df['below_vwap'] = (df['close'] < df['vwap']).astype(int)
        
        # VWAP crossover signals
        df['vwap_bullish_cross'] = ((df['close'] > df['vwap']) & 
                                   (df['close'].shift(1) <= df['vwap'].shift(1))).astype(int)
        
        df['vwap_bearish_cross'] = ((df['close'] < df['vwap']) & 
                                   (df['close'].shift(1) >= df['vwap'].shift(1))).astype(int)
        
        # Band signals (assuming bands exist)
        if 'vwap_upper' in df.columns and 'vwap_lower' in df.columns:
            df['above_upper_band'] = (df['close'] > df['vwap_upper']).astype(int)
            df['below_lower_band'] = (df['close'] < df['vwap_lower']).astype(int)
            
            # Mean reversion signals
            df['upper_band_rejection'] = ((df['close'] < df['vwap_upper']) & 
                                         (df['close'].shift(1) >= df['vwap_upper'].shift(1))).astype(int)
            
            df['lower_band_bounce'] = ((df['close'] > df['vwap_lower']) & 
                                      (df['close'].shift(1) <= df['vwap_lower'].shift(1))).astype(int)
        
        # Overall signal
        df['vwap_signal'] = 0
        df.loc[df['vwap_bullish_cross'] == 1, 'vwap_signal'] = 1
        df.loc[df['vwap_bearish_cross'] == 1, 'vwap_signal'] = -1
        
        return df
    
    def get_volume_profile(self, data: pd.DataFrame, price_levels: int = 20) -> pd.DataFrame:
        """
        Calculate volume profile around VWAP.
        
        Args:
            data: DataFrame with VWAP calculated
            price_levels: Number of price levels for volume profile
            
        Returns:
            DataFrame with volume profile data
        """
        if 'vwap' not in data.columns:
            raise ValueError("VWAP not calculated. Run calculate() first.")
        
        # Create price bins around VWAP
        price_range = data['high'].max() - data['low'].min()
        bin_size = price_range / price_levels
        
        # Create bins
        min_price = data['low'].min()
        bins = [min_price + i * bin_size for i in range(price_levels + 1)]
        
        # Assign each candle to price bins and sum volume
        data_copy = data.copy()
        data_copy['price_bin'] = pd.cut(data_copy['typical_price'], bins=bins, include_lowest=True)
        
        volume_profile = data_copy.groupby('price_bin')['volume'].sum().reset_index()
        volume_profile['price_level'] = volume_profile['price_bin'].apply(lambda x: x.mid)
        volume_profile['volume_percent'] = (volume_profile['volume'] / volume_profile['volume'].sum()) * 100
        
        return volume_profile[['price_level', 'volume', 'volume_percent']]
    
    def get_vwap_deviation(self, data: pd.DataFrame, price_column: str = 'close') -> pd.Series:
        """
        Calculate price deviation from VWAP as percentage.
        
        Args:
            data: DataFrame with VWAP calculated
            price_column: Column name for price data
            
        Returns:
            Series with VWAP deviation percentages
        """
        if 'vwap' not in data.columns:
            raise ValueError("VWAP not calculated. Run calculate() first.")
        
        if price_column not in data.columns:
            raise ValueError(f"Price column '{price_column}' not found in data")
        
        return ((data[price_column] - data['vwap']) / data['vwap']) * 100
    
    def get_column_names(self) -> list:
        """
        Get list of VWAP column names that will be created.
        
        Returns:
            List of VWAP column names
        """
        base_columns = [
            'vwap', 'vwap_std', 'vwap_upper', 'vwap_lower',
            'above_vwap', 'below_vwap', 'vwap_bullish_cross', 'vwap_bearish_cross',
            'vwap_signal'
        ]
        
        # Add band-related columns if using bands
        if self.std_dev_multiplier > 0:
            base_columns.extend([
                'above_upper_band', 'below_lower_band',
                'upper_band_rejection', 'lower_band_bounce'
            ])
        
        return base_columns
    
    def __str__(self) -> str:
        """String representation of VWAP indicator."""
        return f"VWAP(session_type='{self.session_type}', std_dev_multiplier={self.std_dev_multiplier})"
    
    def __repr__(self) -> str:
        """Detailed representation of VWAP indicator."""
        return f"VWAP(session_type='{self.session_type}', std_dev_multiplier={self.std_dev_multiplier})"


def calculate_vwap(data: pd.DataFrame, session_type: str = 'daily', 
                   std_dev_multiplier: float = 1.0) -> pd.DataFrame:
    """
    Convenience function to calculate VWAP.
    
    Args:
        data: DataFrame with OHLCV data
        session_type: VWAP session type
        std_dev_multiplier: Standard deviation multiplier for bands
        
    Returns:
        DataFrame with VWAP values added
    """
    vwap = VWAP(session_type=session_type, std_dev_multiplier=std_dev_multiplier)
    return vwap.calculate(data)