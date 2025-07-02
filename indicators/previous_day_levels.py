#!/usr/bin/env python3
"""
Previous Day High/Low/Close Levels Indicator
===========================================

Calculates previous day's high, low, and close levels which act as important
support and resistance levels for intraday trading strategies.

Features:
- Previous day high/low/close calculation
- Support/resistance level identification
- Breakout signal detection
- Multiple timeframe support (daily, weekly, monthly)
- Excel configuration integration

Author: vAlgo Development Team
Created: June 28, 2025
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Tuple
from datetime import datetime, timedelta


class PreviousDayLevels:
    """
    Previous Day High/Low/Close Levels Calculator
    
    Calculates previous session's key levels for support/resistance analysis.
    """
    
    def __init__(self, timeframe: str = 'daily'):
        """
        Initialize Previous Day Levels calculator.
        
        Args:
            timeframe: Calculation timeframe ('daily', 'weekly', 'monthly')
        """
        self.timeframe = timeframe.lower()
        self.valid_timeframes = ['daily', 'weekly', 'monthly']
        
        if self.timeframe not in self.valid_timeframes:
            raise ValueError(f"Invalid timeframe: {timeframe}. Must be one of {self.valid_timeframes}")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate previous day/week/month levels.
        
        Args:
            data: DataFrame with columns ['timestamp', 'open', 'high', 'low', 'close']
            
        Returns:
            DataFrame with previous levels added
            
        Raises:
            ValueError: If required columns are missing
        """
        # Validate input data
        required_columns = ['timestamp', 'open', 'high', 'low', 'close']
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
        
        # Calculate levels based on timeframe
        if self.timeframe == 'daily':
            df = self._calculate_daily_levels(df)
        elif self.timeframe == 'weekly':
            df = self._calculate_weekly_levels(df)
        elif self.timeframe == 'monthly':
            df = self._calculate_monthly_levels(df)
        
        # Generate signals
        df = self._generate_signals(df)
        
        return df
    
    def _calculate_daily_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate previous day levels."""
        df['date'] = df['timestamp'].dt.date
        
        # Initialize previous day columns
        df['prev_day_high'] = np.nan
        df['prev_day_low'] = np.nan
        df['prev_day_close'] = np.nan
        df['prev_day_open'] = np.nan
        
        # Get unique dates
        unique_dates = sorted(df['date'].unique())
        
        for i, current_date in enumerate(unique_dates):
            if i == 0:  # First day, no previous data
                continue
                
            prev_date = unique_dates[i-1]
            
            # Get previous day's OHLC
            prev_day_data = df[df['date'] == prev_date]
            if not prev_day_data.empty:
                prev_high = prev_day_data['high'].max()
                prev_low = prev_day_data['low'].min()
                prev_close = prev_day_data['close'].iloc[-1]  # Last close of previous day
                prev_open = prev_day_data['open'].iloc[0]     # First open of previous day
                
                # Set previous day values for current day
                current_day_mask = df['date'] == current_date
                df.loc[current_day_mask, 'prev_day_high'] = prev_high
                df.loc[current_day_mask, 'prev_day_low'] = prev_low
                df.loc[current_day_mask, 'prev_day_close'] = prev_close
                df.loc[current_day_mask, 'prev_day_open'] = prev_open
        
        # Clean up temporary column
        df = df.drop(['date'], axis=1)
        
        return df
    
    def _calculate_weekly_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate previous week levels."""
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['year'] = df['timestamp'].dt.year
        df['week_year'] = df['year'].astype(str) + '_' + df['week'].astype(str)
        
        # Initialize previous week columns
        df['prev_week_high'] = np.nan
        df['prev_week_low'] = np.nan
        df['prev_week_close'] = np.nan
        df['prev_week_open'] = np.nan
        
        unique_weeks = sorted(df['week_year'].unique())
        
        for i, current_week in enumerate(unique_weeks):
            if i == 0:  # First week, no previous data
                continue
                
            prev_week = unique_weeks[i-1]
            
            # Get previous week's OHLC
            prev_week_data = df[df['week_year'] == prev_week]
            if not prev_week_data.empty:
                prev_high = prev_week_data['high'].max()
                prev_low = prev_week_data['low'].min()
                prev_close = prev_week_data['close'].iloc[-1]  # Last close of previous week
                prev_open = prev_week_data['open'].iloc[0]     # First open of previous week
                
                # Set previous week values for current week
                current_week_mask = df['week_year'] == current_week
                df.loc[current_week_mask, 'prev_week_high'] = prev_high
                df.loc[current_week_mask, 'prev_week_low'] = prev_low
                df.loc[current_week_mask, 'prev_week_close'] = prev_close
                df.loc[current_week_mask, 'prev_week_open'] = prev_open
        
        # Clean up temporary columns
        df = df.drop(['week', 'year', 'week_year'], axis=1)
        
        return df
    
    def _calculate_monthly_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate previous month levels."""
        df['month'] = df['timestamp'].dt.month
        df['year'] = df['timestamp'].dt.year
        df['month_year'] = df['year'].astype(str) + '_' + df['month'].astype(str).str.zfill(2)
        
        # Initialize previous month columns
        df['prev_month_high'] = np.nan
        df['prev_month_low'] = np.nan
        df['prev_month_close'] = np.nan
        df['prev_month_open'] = np.nan
        
        unique_months = sorted(df['month_year'].unique())
        
        for i, current_month in enumerate(unique_months):
            if i == 0:  # First month, no previous data
                continue
                
            prev_month = unique_months[i-1]
            
            # Get previous month's OHLC
            prev_month_data = df[df['month_year'] == prev_month]
            if not prev_month_data.empty:
                prev_high = prev_month_data['high'].max()
                prev_low = prev_month_data['low'].min()
                prev_close = prev_month_data['close'].iloc[-1]  # Last close of previous month
                prev_open = prev_month_data['open'].iloc[0]     # First open of previous month
                
                # Set previous month values for current month
                current_month_mask = df['month_year'] == current_month
                df.loc[current_month_mask, 'prev_month_high'] = prev_high
                df.loc[current_month_mask, 'prev_month_low'] = prev_low
                df.loc[current_month_mask, 'prev_month_close'] = prev_close
                df.loc[current_month_mask, 'prev_month_open'] = prev_open
        
        # Clean up temporary columns
        df = df.drop(['month', 'year', 'month_year'], axis=1)
        
        return df
    
    def _generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """Generate trading signals based on previous levels."""
        # Determine column prefix based on timeframe
        if self.timeframe == 'daily':
            prefix = 'prev_day'
        elif self.timeframe == 'weekly':
            prefix = 'prev_week'
        else:  # monthly
            prefix = 'prev_month'
        
        high_col = f'{prefix}_high'
        low_col = f'{prefix}_low'
        close_col = f'{prefix}_close'
        
        if high_col not in df.columns:
            return df  # No previous levels calculated
        
        # Breakout signals
        df[f'{prefix}_high_breakout'] = ((df['close'] > df[high_col]) & 
                                        (df['close'].shift(1) <= df[high_col].shift(1))).astype(int)
        
        df[f'{prefix}_low_breakdown'] = ((df['close'] < df[low_col]) & 
                                        (df['close'].shift(1) >= df[low_col].shift(1))).astype(int)
        
        # Support/Resistance tests
        df[f'{prefix}_high_test'] = ((df['high'] >= df[high_col] * 0.998) & 
                                    (df['high'] <= df[high_col] * 1.002) & 
                                    (df['close'] < df[high_col])).astype(int)
        
        df[f'{prefix}_low_test'] = ((df['low'] <= df[low_col] * 1.002) & 
                                   (df['low'] >= df[low_col] * 0.998) & 
                                   (df['close'] > df[low_col])).astype(int)
        
        # Price position relative to previous levels
        df[f'{prefix}_above_high'] = (df['close'] > df[high_col]).astype(int)
        df[f'{prefix}_below_low'] = (df['close'] < df[low_col]).astype(int)
        df[f'{prefix}_between_levels'] = ((df['close'] >= df[low_col]) & 
                                         (df['close'] <= df[high_col])).astype(int)
        
        # Gap signals (if opening above/below previous levels)
        if 'open' in df.columns:
            df[f'{prefix}_gap_up'] = ((df['open'] > df[high_col]) & 
                                     (df['close'].shift(1) < df[high_col])).astype(int)
            
            df[f'{prefix}_gap_down'] = ((df['open'] < df[low_col]) & 
                                       (df['close'].shift(1) > df[low_col])).astype(int)
        
        # Overall signal
        df[f'{prefix}_signal'] = 0
        df.loc[df[f'{prefix}_high_breakout'] == 1, f'{prefix}_signal'] = 1   # Bullish
        df.loc[df[f'{prefix}_low_breakdown'] == 1, f'{prefix}_signal'] = -1  # Bearish
        
        return df
    
    def get_support_resistance_levels(self, data: pd.DataFrame, current_price: float) -> Dict:
        """
        Get current support and resistance levels.
        
        Args:
            data: DataFrame with previous levels calculated
            current_price: Current market price
            
        Returns:
            Dictionary with support/resistance levels and distances
        """
        # Determine column prefix based on timeframe
        if self.timeframe == 'daily':
            prefix = 'prev_day'
        elif self.timeframe == 'weekly':
            prefix = 'prev_week'
        else:  # monthly
            prefix = 'prev_month'
        
        high_col = f'{prefix}_high'
        low_col = f'{prefix}_low'
        close_col = f'{prefix}_close'
        
        if data.empty or high_col not in data.columns:
            return {}
        
        # Get latest levels
        latest = data.iloc[-1]
        
        levels = {
            'timeframe': self.timeframe,
            'high': latest[high_col] if pd.notna(latest[high_col]) else None,
            'low': latest[low_col] if pd.notna(latest[low_col]) else None,
            'close': latest[close_col] if pd.notna(latest[close_col]) else None,
            'current_price': current_price
        }
        
        # Calculate distances and determine key levels
        if pd.notna(latest[high_col]) and pd.notna(latest[low_col]):
            high_val = latest[high_col]
            low_val = latest[low_col]
            
            # Determine current position
            if current_price > high_val:
                levels['position'] = 'above_resistance'
                levels['immediate_support'] = high_val
                levels['support_distance'] = ((current_price - high_val) / current_price) * 100
            elif current_price < low_val:
                levels['position'] = 'below_support'
                levels['immediate_resistance'] = low_val
                levels['resistance_distance'] = ((low_val - current_price) / current_price) * 100
            else:
                levels['position'] = 'between_levels'
                levels['resistance'] = high_val
                levels['support'] = low_val
                levels['resistance_distance'] = ((high_val - current_price) / current_price) * 100
                levels['support_distance'] = ((current_price - low_val) / current_price) * 100
        
        return levels
    
    def get_level_strength(self, data: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """
        Calculate strength of previous levels based on touch count.
        
        Args:
            data: DataFrame with previous levels calculated
            lookback: Number of periods to look back for touches
            
        Returns:
            DataFrame with level strength metrics
        """
        # Determine column prefix based on timeframe
        if self.timeframe == 'daily':
            prefix = 'prev_day'
        elif self.timeframe == 'weekly':
            prefix = 'prev_week'
        else:  # monthly
            prefix = 'prev_month'
        
        high_col = f'{prefix}_high'
        low_col = f'{prefix}_low'
        
        if high_col not in data.columns:
            return data
        
        df = data.copy()
        
        # Count touches of high level (within 0.2% tolerance)
        high_touch_threshold = 0.002  # 0.2%
        df[f'{prefix}_high_touches'] = 0
        
        for i in range(lookback, len(df)):
            if pd.notna(df.loc[i, high_col]):
                high_level = df.loc[i, high_col]
                # Count touches in recent periods
                recent_data = df.loc[i-lookback:i-1]
                touches = ((recent_data['high'] >= high_level * (1 - high_touch_threshold)) & 
                          (recent_data['high'] <= high_level * (1 + high_touch_threshold))).sum()
                df.loc[i, f'{prefix}_high_touches'] = touches
        
        # Count touches of low level
        low_touch_threshold = 0.002  # 0.2%
        df[f'{prefix}_low_touches'] = 0
        
        for i in range(lookback, len(df)):
            if pd.notna(df.loc[i, low_col]):
                low_level = df.loc[i, low_col]
                # Count touches in recent periods
                recent_data = df.loc[i-lookback:i-1]
                touches = ((recent_data['low'] >= low_level * (1 - low_touch_threshold)) & 
                          (recent_data['low'] <= low_level * (1 + low_touch_threshold))).sum()
                df.loc[i, f'{prefix}_low_touches'] = touches
        
        # Calculate level strength (more touches = stronger level)
        df[f'{prefix}_high_strength'] = np.where(df[f'{prefix}_high_touches'] >= 2, 'strong', 
                                                np.where(df[f'{prefix}_high_touches'] == 1, 'moderate', 'weak'))
        
        df[f'{prefix}_low_strength'] = np.where(df[f'{prefix}_low_touches'] >= 2, 'strong', 
                                               np.where(df[f'{prefix}_low_touches'] == 1, 'moderate', 'weak'))
        
        return df
    
    def get_column_names(self) -> list:
        """
        Get list of column names that will be created.
        
        Returns:
            List of column names
        """
        if self.timeframe == 'daily':
            prefix = 'prev_day'
        elif self.timeframe == 'weekly':
            prefix = 'prev_week'
        else:  # monthly
            prefix = 'prev_month'
        
        return [
            f'{prefix}_high', f'{prefix}_low', f'{prefix}_close', f'{prefix}_open',
            f'{prefix}_high_breakout', f'{prefix}_low_breakdown',
            f'{prefix}_high_test', f'{prefix}_low_test',
            f'{prefix}_above_high', f'{prefix}_below_low', f'{prefix}_between_levels',
            f'{prefix}_gap_up', f'{prefix}_gap_down', f'{prefix}_signal'
        ]
    
    def __str__(self) -> str:
        """String representation of Previous Day Levels indicator."""
        return f"PreviousDayLevels(timeframe='{self.timeframe}')"
    
    def __repr__(self) -> str:
        """Detailed representation of Previous Day Levels indicator."""
        return f"PreviousDayLevels(timeframe='{self.timeframe}')"


def calculate_previous_day_levels(data: pd.DataFrame, timeframe: str = 'daily') -> pd.DataFrame:
    """
    Convenience function to calculate previous day/week/month levels.
    
    Args:
        data: DataFrame with OHLC data
        timeframe: Calculation timeframe
        
    Returns:
        DataFrame with previous levels added
    """
    pdl = PreviousDayLevels(timeframe=timeframe)
    return pdl.calculate(data)