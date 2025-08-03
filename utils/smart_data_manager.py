#!/usr/bin/env python3
"""
Smart Data Manager
==================

Database-first data management with intelligent gap detection and API fallback.
Prevents duplicate API calls and ensures data completeness with market calendar awareness.

Author: vAlgo Development Team
Created: July 4, 2025
Version: 1.0.0
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime, timedelta, date
import pandas as pd
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from data_manager.database import DatabaseManager
    from utils.market_data_fetcher import MarketDataFetcher
    from utils.config_cache import get_cached_config
    from utils.logger import get_logger
    DATABASE_AVAILABLE = True
except ImportError as e:
    print(f"[SMART_DATA] Import warning: {e}")
    DATABASE_AVAILABLE = False


class SmartDataManager:
    """
    Intelligent data manager with database-first approach.
    Minimizes API calls and prevents duplicate data storage.
    """
    
    def __init__(self, db_path: str = "data/vAlgo_market_data.db"):
        """Initialize smart data manager."""
        self.db_path = db_path
        self.db_manager = None
        self.market_data_fetcher = None
        self.config_cache = get_cached_config()
        
        try:
            self.logger = get_logger(__name__)
        except:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # Market calendar for Indian markets
        self.market_holidays = {
            # 2025 Indian market holidays (NSE)
            date(2025, 1, 26),  # Republic Day
            date(2025, 3, 14),  # Holi
            date(2025, 3, 29),  # Good Friday
            date(2025, 4, 14),  # Ram Navami
            date(2025, 8, 15),  # Independence Day
            date(2025, 10, 2),  # Gandhi Jayanti
            date(2025, 11, 1),  # Diwali
            date(2025, 11, 15), # Guru Nanak Jayanti
        }
        
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize database and market data fetcher."""
        try:
            if DATABASE_AVAILABLE:
                self.db_manager = DatabaseManager(self.db_path)
                self.market_data_fetcher = MarketDataFetcher()
                self.logger.info("[SMART_DATA] Components initialized successfully")
            else:
                self.logger.warning("[SMART_DATA] Database components not available")
        except Exception as e:
            self.logger.error(f"[SMART_DATA] Failed to initialize components: {e}")
    
    def get_previous_trading_day(self, reference_date: date) -> str:
        """
        Get the previous trading day considering weekends and holidays.
        
        Args:
            reference_date: Date to find previous trading day for
            
        Returns:
            str: Previous trading day in YYYY-MM-DD format
        """
        current_date = reference_date
        
        while True:
            # Go back one day
            current_date -= timedelta(days=1)
            
            # Check if it's a weekend
            if current_date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                continue
                
            # Check if it's a holiday
            if current_date in self.market_holidays:
                continue
                
            # Found a valid trading day
            return current_date.strftime('%Y-%m-%d')
    
    def check_data_completeness(self, symbol: str, exchange: str, 
                              timeframe: str, target_date: str) -> Tuple[bool, int, int]:
        """
        Check if data is complete for a given date.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Time interval (5m, 15m, etc.)
            target_date: Date to check (YYYY-MM-DD)
            
        Returns:
            Tuple[bool, int, int]: (is_complete, existing_count, expected_count)
        """
        try:
            if not self.db_manager:
                return False, 0, 0
            
            # Get existing data for the date
            existing_data = self.db_manager.get_ohlcv_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=target_date,
                end_date=target_date
            )
            
            existing_count = len(existing_data) if existing_data is not None else 0
            
            # Expected count based on timeframe (for a full trading day)
            # NSE trading hours: 9:15 AM to 3:30 PM = 6 hours 15 minutes = 375 minutes
            if timeframe == '5m':
                expected_count = 75  # 375 / 5 = 75 candles
            elif timeframe == '15m':
                expected_count = 25  # 375 / 15 = 25 candles
            elif timeframe == '1h':
                expected_count = 7   # Approximately 6-7 candles
            else:
                expected_count = 75  # Default to 5m
            
            is_complete = existing_count >= expected_count * 0.95  # Allow 5% tolerance
            
            self.logger.debug(f"[COMPLETENESS_CHECK] {symbol} {target_date}: {existing_count}/{expected_count} candles")
            
            return is_complete, existing_count, expected_count
            
        except Exception as e:
            self.logger.error(f"[COMPLETENESS_CHECK] Error checking data completeness: {e}")
            return False, 0, 0
    
    def get_current_day_data(self, symbol: str, exchange: str = 'NSE_INDEX', 
                           timeframe: str = '5m', target_date: str = None) -> pd.DataFrame:
        """
        Get current day data with database-first approach.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Time interval
            target_date: Target date (default: today)
            
        Returns:
            pd.DataFrame: Current day OHLCV data
        """
        if target_date is None:
            target_date = datetime.now().strftime('%Y-%m-%d')
        
        self.logger.info(f"[CURRENT_DAY] Getting data for {symbol} on {target_date}")
        
        # Step 1: Check database first
        is_complete, existing_count, expected_count = self.check_data_completeness(
            symbol, exchange, timeframe, target_date
        )
        
        if is_complete:
            self.logger.info(f"[CURRENT_DAY] ✅ Complete data found in DB: {existing_count}/{expected_count} candles")
            return self.db_manager.get_ohlcv_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=target_date,
                end_date=target_date
            )
        
        # Step 2: Data incomplete or missing - fetch from API
        self.logger.info(f"[CURRENT_DAY] ⚠️ Incomplete data in DB ({existing_count}/{expected_count}), fetching from API")
        
        try:
            if not self.market_data_fetcher or not self.market_data_fetcher.openalgo_client:
                self.logger.error("[CURRENT_DAY] ❌ Market data fetcher not available")
                return pd.DataFrame()
            
            # Fetch from API
            api_data = self.market_data_fetcher.openalgo_client.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                interval=timeframe,
                start_date=target_date,
                end_date=target_date
            )
            
            if api_data is not None and not api_data.empty:
                self.logger.info(f"[CURRENT_DAY] ✅ Fetched {len(api_data)} candles from API")
                
                # Store in database (will handle duplicates)
                self.db_manager.store_ohlcv_data(
                    data=api_data,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                
                return api_data
            else:
                self.logger.warning(f"[CURRENT_DAY] ❌ No data available from API for {target_date}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"[CURRENT_DAY] ❌ Error fetching data: {e}")
            return pd.DataFrame()
    
    def get_previous_day_data(self, symbol: str, exchange: str = 'NSE_INDEX', 
                            timeframe: str = 'D', reference_date: date = None) -> pd.DataFrame:
        """
        Get previous trading day data with database-first approach.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Time interval (D for daily)
            reference_date: Reference date (default: today)
            
        Returns:
            pd.DataFrame: Previous day OHLCV data
        """
        if reference_date is None:
            reference_date = datetime.now().date()
        
        previous_day = self.get_previous_trading_day(reference_date)
        self.logger.info(f"[PREVIOUS_DAY] Getting data for {symbol} on {previous_day}")
        
        # Step 1: Check database first
        is_complete, existing_count, expected_count = self.check_data_completeness(
            symbol, exchange, timeframe, previous_day
        )
        
        if timeframe == 'D':
            # For daily data, we only need 1 candle
            expected_count = 1
            is_complete = existing_count >= 1
        
        if is_complete:
            self.logger.info(f"[PREVIOUS_DAY] ✅ Complete data found in DB: {existing_count} candles")
            return self.db_manager.get_ohlcv_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=previous_day,
                end_date=previous_day
            )
        
        # Step 2: Data missing - fetch from API
        self.logger.info(f"[PREVIOUS_DAY] ⚠️ Data missing in DB, fetching from API")
        
        try:
            if not self.market_data_fetcher or not self.market_data_fetcher.openalgo_client:
                self.logger.error("[PREVIOUS_DAY] ❌ Market data fetcher not available")
                return pd.DataFrame()
            
            # Fetch from API
            api_data = self.market_data_fetcher.openalgo_client.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                interval=timeframe,
                start_date=previous_day,
                end_date=previous_day
            )
            
            if api_data is not None and not api_data.empty:
                self.logger.info(f"[PREVIOUS_DAY] ✅ Fetched {len(api_data)} candles from API")
                
                # Store in database
                self.db_manager.store_ohlcv_data(
                    data=api_data,
                    symbol=symbol,
                    exchange=exchange,
                    timeframe=timeframe
                )
                
                return api_data
            else:
                self.logger.warning(f"[PREVIOUS_DAY] ❌ No data available from API for {previous_day}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"[PREVIOUS_DAY] ❌ Error fetching data: {e}")
            return pd.DataFrame()
    
    def get_unified_dataset(self, symbol: str, exchange: str = 'NSE_INDEX',
                          timeframe: str = '5m', candle_count: int = 500) -> pd.DataFrame:
        """
        Get unified dataset with current day, previous day, and historical data.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            timeframe: Time interval
            candle_count: Number of historical candles needed
            
        Returns:
            pd.DataFrame: Unified dataset with complete OHLCV data
        """
        try:
            self.logger.info(f"[UNIFIED_DATASET] Building dataset for {symbol} with {candle_count} candles")
            
            # Get current day data (ensures today's data is in DB)
            current_day = datetime.now().strftime('%Y-%m-%d')
            current_day_data = self.get_current_day_data(symbol, exchange, timeframe, current_day)
            
            # Get previous day data (ensures previous day is in DB for CPR)
            previous_day_data = self.get_previous_day_data(symbol, exchange, 'D')
            
            # Get historical data from database (now includes current and previous day)
            end_date = datetime.now().date()
            start_date = end_date - timedelta(days=30)  # Get enough days to ensure candle_count
            
            historical_data = self.db_manager.get_ohlcv_data(
                symbol=symbol,
                exchange=exchange,
                timeframe=timeframe,
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if historical_data is None or historical_data.empty:
                self.logger.warning(f"[UNIFIED_DATASET] ❌ No historical data found for {symbol}")
                return pd.DataFrame()
            
            # Take the most recent candles
            if len(historical_data) > candle_count:
                historical_data = historical_data.tail(candle_count)
            
            self.logger.info(f"[UNIFIED_DATASET] ✅ Built dataset with {len(historical_data)} candles")
            self.logger.info(f"[UNIFIED_DATASET] ✅ Range: {historical_data.index.min()} to {historical_data.index.max()}")
            
            return historical_data
            
        except Exception as e:
            self.logger.error(f"[UNIFIED_DATASET] ❌ Error building unified dataset: {e}")
            return pd.DataFrame()
    
    def get_data_statistics(self) -> Dict[str, Any]:
        """Get data management statistics."""
        try:
            stats = {
                "db_path": self.db_path,
                "db_available": self.db_manager is not None,
                "api_available": self.market_data_fetcher is not None,
                "market_holidays_count": len(self.market_holidays),
                "current_date": datetime.now().strftime('%Y-%m-%d'),
                "previous_trading_day": self.get_previous_trading_day(datetime.now().date())
            }
            
            if self.db_manager:
                # Add database statistics if available
                pass
                
            return stats
            
        except Exception as e:
            self.logger.error(f"[STATISTICS] Error getting statistics: {e}")
            return {"error": str(e)}