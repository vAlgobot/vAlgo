#!/usr/bin/env python3
"""
Trading Day Validator for vAlgo Trading System
==============================================

Utility functions to identify valid trading days and filter out weekends,
holidays, and non-trading periods.

Author: vAlgo Development Team
Created: July 12, 2025
Version: 1.0.0
"""

import pandas as pd
from datetime import datetime, date, time
from typing import List, Set, Optional, Union
import logging

from utils.logger import get_logger


class TradingDayValidator:
    """
    Validator for trading days and times based on market hours and signal data.
    """
    
    def __init__(self):
        """Initialize the trading day validator."""
        self.logger = get_logger(__name__)
        
        # Standard market hours (IST)
        self.market_open = time(9, 15)  # 9:15 AM
        self.market_close = time(15, 30)  # 3:30 PM
    
    def is_trading_day(self, date_obj: Union[datetime, date]) -> bool:
        """
        Check if a given date is a potential trading day (not weekend).
        
        Args:
            date_obj: Date to check
            
        Returns:
            True if weekday (Monday-Friday), False for weekends
        """
        try:
            if isinstance(date_obj, datetime):
                date_obj = date_obj.date()
            
            # Check if it's a weekday (Monday=0, Sunday=6)
            weekday = date_obj.weekday()
            return weekday < 5  # Monday-Friday are 0-4
            
        except Exception as e:
            self.logger.warning(f"Error checking trading day: {e}")
            return False
    
    def is_trading_time(self, timestamp: datetime) -> bool:
        """
        Check if a given timestamp is within market trading hours.
        
        Args:
            timestamp: Timestamp to check
            
        Returns:
            True if within market hours, False otherwise
        """
        try:
            time_part = timestamp.time()
            return self.market_open <= time_part <= self.market_close
            
        except Exception as e:
            self.logger.warning(f"Error checking trading time: {e}")
            return False
    
    def get_trading_days_from_signal_data(self, signal_data: pd.DataFrame) -> Set[date]:
        """
        Extract actual trading days from signal data.
        
        Args:
            signal_data: DataFrame with timestamp column
            
        Returns:
            Set of dates that have actual trading data
        """
        try:
            if signal_data.empty or 'timestamp' not in signal_data.columns:
                return set()
            
            # Extract unique dates from timestamps
            trading_dates = set()
            for timestamp in signal_data['timestamp']:
                if pd.notna(timestamp):
                    if isinstance(timestamp, str):
                        timestamp = pd.to_datetime(timestamp)
                    trading_dates.add(timestamp.date())
            
            # Filter to only weekdays
            valid_trading_dates = {
                d for d in trading_dates 
                if self.is_trading_day(d)
            }
            
            self.logger.info(f"Found {len(valid_trading_dates)} valid trading days in signal data")
            return valid_trading_dates
            
        except Exception as e:
            self.logger.error(f"Error extracting trading days from signal data: {e}")
            return set()
    
    def filter_to_trading_days(self, df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
        """
        Filter DataFrame to only include rows from valid trading days.
        
        Args:
            df: DataFrame to filter
            date_column: Name of the date column
            
        Returns:
            Filtered DataFrame with only trading day data
        """
        try:
            if df.empty or date_column not in df.columns:
                return df
            
            # Create filter for trading days
            trading_day_filter = df[date_column].apply(
                lambda x: self.is_trading_day(x) if pd.notna(x) else False
            )
            
            filtered_df = df[trading_day_filter].copy()
            
            self.logger.debug(f"Filtered from {len(df)} to {len(filtered_df)} trading day records")
            return filtered_df
            
        except Exception as e:
            self.logger.error(f"Error filtering to trading days: {e}")
            return df
    
    def validate_trade_timestamp(self, trade_data: dict) -> bool:
        """
        Validate that a trade occurred on a valid trading day and time.
        
        Args:
            trade_data: Dictionary containing trade information
            
        Returns:
            True if valid trading timestamp, False otherwise
        """
        try:
            # Check if we have valid trade data
            trade_status = trade_data.get('Trade status', '')
            if trade_status == 'No Trade' or not trade_status:
                return False
            
            # Check date
            trade_date = trade_data.get('Date')
            if not trade_date:
                return False
            
            if not self.is_trading_day(trade_date):
                return False
            
            # Check entry time if available
            entry_time = trade_data.get('Entry time', '')
            if entry_time:
                try:
                    # Parse time string (e.g., "09:15:00 AM")
                    if isinstance(entry_time, str):
                        # Combine date and time for validation
                        datetime_str = f"{trade_date.date()} {entry_time}"
                        entry_datetime = pd.to_datetime(datetime_str)
                        
                        if not self.is_trading_time(entry_datetime):
                            return False
                except:
                    # If time parsing fails, just check the date
                    pass
            
            return True
            
        except Exception as e:
            self.logger.warning(f"Error validating trade timestamp: {e}")
            return False


# Convenience functions for easy use
_validator = None

def get_trading_day_validator() -> TradingDayValidator:
    """Get a singleton instance of TradingDayValidator."""
    global _validator
    if _validator is None:
        _validator = TradingDayValidator()
    return _validator

def is_trading_day(date_obj: Union[datetime, date]) -> bool:
    """Check if a date is a trading day (weekday)."""
    return get_trading_day_validator().is_trading_day(date_obj)

def is_trading_time(timestamp: datetime) -> bool:
    """Check if a timestamp is within market hours."""
    return get_trading_day_validator().is_trading_time(timestamp)

def filter_to_trading_days(df: pd.DataFrame, date_column: str = 'Date') -> pd.DataFrame:
    """Filter DataFrame to only trading days."""
    return get_trading_day_validator().filter_to_trading_days(df, date_column)

def validate_trade_timestamp(trade_data: dict) -> bool:
    """Validate trade timestamp is on a trading day."""
    return get_trading_day_validator().validate_trade_timestamp(trade_data)