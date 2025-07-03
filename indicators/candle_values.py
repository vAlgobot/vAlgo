#!/usr/bin/env python3
"""
Candle Values Indicator
=======================

Dynamic candle-based indicator supporting both backtesting and live trading.
Provides Current Candle, Previous Candle, Current Day, and Previous Day OHLC values.

Features:
- Dynamic mode support (backtesting/live)
- Multiple candle modes (CurrentCandle, PreviousCandle, CurrentDayCandle, PreviousDayCandle)
- Incremental updates for live trading
- Aggregation support for day-level calculations
- Excel configuration integration

Author: vAlgo Development Team
Created: June 30, 2025
"""

import pandas as pd
import numpy as np
from typing import Dict, Optional, List, Union, Any
from datetime import datetime, timedelta

class CandleValues:
    """
    Dynamic Candle Values Calculator
    
    Supports both backtesting (vectorized) and live trading (incremental) modes.
    Calculates various candle-based indicators based on user's requirements.
    """
    
    def __init__(self, mode: str = "PreviousDayCandle", aggregate_day: bool = False, 
                 count: int = 1, direction: str = "backward", timeframe: str = "daily"):
        """
        Initialize CandleValues calculator.
        
        Args:
            mode: Candle calculation mode
                  - "CurrentCandle": Current timestamp candle OHLC
                  - "PreviousCandle": Previous candle from current timestamp  
                  - "CurrentDayCandle": All candles for current day
                  - "PreviousDayCandle": Previous day's complete OHLC
                  - "FetchMultipleCandles": Multiple candles based on count/direction
            aggregate_day: Whether to aggregate intraday data to daily OHLC
            count: Number of candles to fetch (for FetchMultipleCandles mode)
            direction: Direction for multiple candles ("forward" or "backward")
            timeframe: Timeframe for calculations ("daily", "intraday")
        """
        self.mode = mode
        self.aggregate_day = aggregate_day
        self.count = count
        self.direction = direction
        self.timeframe = timeframe
        
        # Validate mode
        valid_modes = ["CurrentCandle", "PreviousCandle", "CurrentDayCandle", 
                      "PreviousDayCandle", "FetchMultipleCandles"]
        if self.mode not in valid_modes:
            raise ValueError(f"Invalid mode: {mode}. Must be one of {valid_modes}")
        
        # Validate direction
        if self.direction not in ["forward", "backward"]:
            raise ValueError("Direction must be 'forward' or 'backward'")
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate candle values for backtesting mode (vectorized).
        
        Args:
            data: DataFrame with timestamp index and OHLC columns
            
        Returns:
            DataFrame with candle indicator columns added
        """
        if data.empty:
            return data
        
        # Create a copy to avoid modifying original data
        df = data.copy()
        
        # Ensure we have timestamp either as index or column
        if 'timestamp' not in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df['timestamp'] = df.index
            else:
                raise ValueError("No timestamp column or datetime index found")
        
        # Ensure timestamp is datetime
        if not pd.api.types.is_datetime64_any_dtype(df['timestamp']):
            df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Sort by timestamp
        df = df.sort_values('timestamp').reset_index(drop=True)
        
        # Apply candle logic based on mode
        if self.mode == "CurrentCandle":
            df = self._calculate_current_candle(df)
        elif self.mode == "PreviousCandle":
            df = self._calculate_previous_candle(df)
        elif self.mode == "CurrentDayCandle":
            df = self._calculate_current_day_candle(df)
        elif self.mode == "PreviousDayCandle":
            df = self._calculate_previous_day_candle(df)
        elif self.mode == "FetchMultipleCandles":
            df = self._calculate_multiple_candles(df)
        
        # Remove temporary timestamp column if it was added
        if df.index.name == 'timestamp' and 'timestamp' in df.columns:
            df = df.drop('timestamp', axis=1)
        
        return df
    
    def _calculate_current_candle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate current candle OHLC values with user-expected naming format."""
        # Current candle with underscore naming format (OHLC only)
        df['CurrentCandle_Open'] = df['open']
        df['CurrentCandle_High'] = df['high']
        df['CurrentCandle_Low'] = df['low']
        df['CurrentCandle_Close'] = df['close']
        
        return df
    
    def _calculate_previous_candle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate previous candle OHLC values with user-expected naming format."""
        # Shift OHLC values by 1 period to get previous candle with underscore naming (OHLC only)
        df['PreviousCandle_Open'] = df['open'].shift(1)
        df['PreviousCandle_High'] = df['high'].shift(1)
        df['PreviousCandle_Low'] = df['low'].shift(1)
        df['PreviousCandle_Close'] = df['close'].shift(1)
        
        return df
    
    def _calculate_current_day_candle(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate current day candle OHLC values with INCREMENTAL high/low for backtesting.
        FIXED: Now calculates cumulative high/low up to current timestamp, not full day.
        """
        df['date'] = df['timestamp'].dt.date
        
        if self.aggregate_day:
            # Aggregate intraday data to daily OHLC
            daily_data = df.groupby('date').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last'
            }).reset_index()
            
            # Merge back to original dataframe
            df = df.merge(daily_data, on='date', suffixes=('', '_day'))
            df['CurrentDayCandle_Open'] = df['open_day']
            df['CurrentDayCandle_High'] = df['high_day']
            df['CurrentDayCandle_Low'] = df['low_day']
            df['CurrentDayCandle_Close'] = df['close_day']
            
            # Clean up temporary columns
            df = df.drop(['open_day', 'high_day', 'low_day', 'close_day'], axis=1)
        else:
            # OPTIMIZED CUMULATIVE approach for backtesting accuracy
            # This calculates high/low UP TO the current timestamp, not for the entire day
            
            # CRITICAL: Ensure proper timestamp ordering within each day for cumulative calculations
            df = df.sort_values(['date', 'timestamp']).reset_index(drop=True)
            
            # PROGRESSIVE CUMULATIVE calculations - exactly as user expects
            df['CurrentDayCandle_Open'] = df.groupby('date')['open'].transform('first')
            
            # FIXED: Progressive cumulative max/min within each day (maintains chronological order)
            df['CurrentDayCandle_High'] = df.groupby('date')['high'].cummax()
            df['CurrentDayCandle_Low'] = df.groupby('date')['low'].cummin()
            
            # Current close is always the current candle's close
            df['CurrentDayCandle_Close'] = df['close']
        
        # Clean up temporary columns
        df = df.drop(['date'], axis=1)
        
        return df
    
    def _calculate_previous_day_candle(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate previous day candle OHLC values with user-expected naming format.
        Removed unwanted signal columns as per user request."""
        df['date'] = df['timestamp'].dt.date
        
        # Get previous day's OHLC for each row (using user-expected naming format, OHLC only)
        df['PreviousDayCandle_Open'] = np.nan
        df['PreviousDayCandle_High'] = np.nan
        df['PreviousDayCandle_Low'] = np.nan
        df['PreviousDayCandle_Close'] = np.nan
        
        # Group by date and get previous day's data
        unique_dates = sorted(df['date'].unique())
        
        for i, current_date in enumerate(unique_dates):
            if i == 0:  # First day, no previous data
                continue
                
            prev_date = unique_dates[i-1]
            
            # Get previous day's OHLC
            prev_day_data = df[df['date'] == prev_date]
            if not prev_day_data.empty:
                if self.aggregate_day:
                    # Aggregate previous day to single OHLC
                    prev_day_open = prev_day_data['open'].iloc[0]
                    prev_day_high = prev_day_data['high'].max()
                    prev_day_low = prev_day_data['low'].min()
                    prev_day_close = prev_day_data['close'].iloc[-1]
                else:
                    # Use last values from previous day (OHLC only)
                    prev_day_open = prev_day_data['open'].iloc[0]
                    prev_day_high = prev_day_data['high'].max()
                    prev_day_low = prev_day_data['low'].min()
                    prev_day_close = prev_day_data['close'].iloc[-1]
                
                # Set previous day values for current day (OHLC only)
                current_day_mask = df['date'] == current_date
                df.loc[current_day_mask, 'PreviousDayCandle_Open'] = prev_day_open
                df.loc[current_day_mask, 'PreviousDayCandle_High'] = prev_day_high
                df.loc[current_day_mask, 'PreviousDayCandle_Low'] = prev_day_low
                df.loc[current_day_mask, 'PreviousDayCandle_Close'] = prev_day_close
        
        # REMOVED: All unwanted signal columns as per user request:
        # - prev_day_above_high, prev_day_below_low, prev_day_between_levels
        # - prev_day_gap_down, prev_day_gap_up, prev_day_high_breakout
        # - prev_day_high_test, prev_day_low_breakdown, prev_day_low_test
        # - prev_day_signal, previous_day_volume
        
        # Clean up temporary columns
        df = df.drop(['date'], axis=1)
        
        return df
    
    def _calculate_multiple_candles(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate multiple candles based on count and direction."""
        if self.direction == "backward":
            # Get previous N candles (OHLC only)
            for i in range(1, self.count + 1):
                df[f'candle_minus_{i}_open'] = df['open'].shift(i)
                df[f'candle_minus_{i}_high'] = df['high'].shift(i)
                df[f'candle_minus_{i}_low'] = df['low'].shift(i)
                df[f'candle_minus_{i}_close'] = df['close'].shift(i)
        else:
            # Get future N candles (for backtesting only, OHLC only)
            for i in range(1, self.count + 1):
                df[f'candle_plus_{i}_open'] = df['open'].shift(-i)
                df[f'candle_plus_{i}_high'] = df['high'].shift(-i)
                df[f'candle_plus_{i}_low'] = df['low'].shift(-i)
                df[f'candle_plus_{i}_close'] = df['close'].shift(-i)
        
        return df
    
    def get_candle_for_timestamp(self, data: pd.DataFrame, target_timestamp: str) -> pd.DataFrame:
        """
        Get specific candle data for a given timestamp (utility method).
        Based on user's get_OHLC_candle_for_specified_Datetime function.
        
        Args:
            data: OHLC DataFrame
            target_timestamp: Target timestamp string
            
        Returns:
            DataFrame with candle data for the specified timestamp
        """
        # Ensure 'timestamp' column exists and is datetime
        df = data.copy()
        if 'timestamp' not in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df.index):
                df['timestamp'] = df.index
            else:
                raise ValueError("No timestamp column or datetime index found")
        
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        
        target = pd.to_datetime(target_timestamp)
        
        if self.mode == "CurrentCandle":
            result = df[df['timestamp'] == target]
        
        elif self.mode == "FetchMultipleCandles":
            idx = df[df['timestamp'] == target].index
            if idx.empty:
                return pd.DataFrame()
            i = idx[0]
            if self.direction == "forward":
                result = df.iloc[i:i + self.count]
            elif self.direction == "backward":
                start = max(0, i - self.count + 1)
                result = df.iloc[start:i + 1]
            else:
                raise ValueError("Invalid direction: Use 'forward' or 'backward'")
        
        elif self.mode == "PreviousCandle":
            idx = df[df['timestamp'] == target].index
            if not idx.empty and idx[0] > 0:
                result = df.iloc[[idx[0] - 1]]
            else:
                return pd.DataFrame()
        
        elif self.mode == "CurrentDayCandle":
            result = df[df['timestamp'].dt.date == target.date()]
            if result.empty:
                return pd.DataFrame()
            if self.aggregate_day:
                return pd.DataFrame({
                    "timestamp": [result['timestamp'].iloc[0].date()],
                    "open": [result['open'].iloc[0]],
                    "high": [result['high'].max()],
                    "low": [result['low'].min()],
                    "close": [result['close'].iloc[-1]],
                    "volume": [result['volume'].sum()]
                })
        
        elif self.mode == "PreviousDayCandle":
            current_day_indices = df[df['timestamp'].dt.date == target.date()].index
            if current_day_indices.empty:
                return pd.DataFrame()
            start_idx = current_day_indices[0]
            if start_idx > 0:
                previous_day = df.loc[start_idx - 1, 'timestamp'].date()
                result = df[df['timestamp'].dt.date == previous_day]
            else:
                return pd.DataFrame()
        
        else:
            raise ValueError(f"Invalid mode. Use: CurrentCandle, PreviousCandle, CurrentDayCandle, PreviousDayCandle, FetchMultipleCandles")
        
        return result.reset_index(drop=True)
    
    def get_indicator_names(self) -> List[str]:
        """
        Get list of indicator column names that will be generated.
        Updated to reflect new naming format and removed unwanted signal columns.
        
        Returns:
            List of column names
        """
        if self.mode == "CurrentCandle":
            return ['CurrentCandle_Open', 'CurrentCandle_High', 'CurrentCandle_Low', 
                   'CurrentCandle_Close']
        
        elif self.mode == "PreviousCandle":
            return ['PreviousCandle_Open', 'PreviousCandle_High', 'PreviousCandle_Low',
                   'PreviousCandle_Close']
        
        elif self.mode == "CurrentDayCandle":
            return ['CurrentDayCandle_Open', 'CurrentDayCandle_High', 'CurrentDayCandle_Low', 
                   'CurrentDayCandle_Close']
        
        elif self.mode == "PreviousDayCandle":
            return ['PreviousDayCandle_Open', 'PreviousDayCandle_High', 'PreviousDayCandle_Low', 
                   'PreviousDayCandle_Close']
        
        elif self.mode == "FetchMultipleCandles":
            indicators = []
            if self.direction == "backward":
                for i in range(1, self.count + 1):
                    indicators.extend([f'candle_minus_{i}_open', f'candle_minus_{i}_high', 
                                     f'candle_minus_{i}_low', f'candle_minus_{i}_close'])
            else:
                for i in range(1, self.count + 1):
                    indicators.extend([f'candle_plus_{i}_open', f'candle_plus_{i}_high',
                                     f'candle_plus_{i}_low', f'candle_plus_{i}_close'])
            return indicators
        
        return []
    
    def __str__(self) -> str:
        """String representation of CandleValues indicator."""
        return f"CandleValues(mode='{self.mode}', aggregate_day={self.aggregate_day})"
    
    def __repr__(self) -> str:
        """Detailed representation of CandleValues indicator."""
        return f"CandleValues(mode='{self.mode}', aggregate_day={self.aggregate_day}, count={self.count}, direction='{self.direction}')"


def calculate_candle_values(data: pd.DataFrame, mode: str = "PreviousDayCandle", 
                          aggregate_day: bool = True, count: int = 1, 
                          direction: str = "backward") -> pd.DataFrame:
    """
    Convenience function to calculate candle values.
    
    Args:
        data: DataFrame with OHLC data
        mode: Candle calculation mode
        aggregate_day: Whether to aggregate intraday data
        count: Number of candles (for FetchMultipleCandles)
        direction: Direction for multiple candles
        
    Returns:
        DataFrame with candle indicators added
    """
    candle_values = CandleValues(mode=mode, aggregate_day=aggregate_day, 
                                count=count, direction=direction)
    return candle_values.calculate(data)