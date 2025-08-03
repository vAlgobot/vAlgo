#!/usr/bin/env python3
"""
Market Data Fetcher
===================

Real-time market data fetching with multiple data sources and fallback mechanisms.
Provides reliable OHLCV data for live trading with proper validation and error handling.

Features:
- OpenAlgo API integration
- Yahoo Finance API fallback
- Database query fallback
- Real-time data validation
- Multiple source redundancy

Author: vAlgo Development Team
Created: July 2, 2025
Version: 1.0.0 (Production)
"""

import sys
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging
import requests
import time
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.logger import get_logger
    from data_manager.openalgo import OpenAlgo
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


class MarketDataFetcher:
    """
    Multi-source market data fetcher with real-time capabilities.
    Provides redundant data sources for reliable live trading.
    """
    
    def __init__(self, db_path: str = "data/valgo_market_data.db", 
                 openalgo_url: str = None, mode: str = "backtesting"):
        self.db_path = db_path
        
        # Load OpenAlgo configuration from environment variables
        try:
            from utils.env_loader import get_all_config
            env_config = get_all_config()
            
            self.openalgo_url = openalgo_url or env_config.get('openalgo_url', 'http://localhost:5000')
            self.openalgo_api_key = env_config.get('openalgo_api_key')
            
            # Remove trailing slash if present
            if self.openalgo_url.endswith('/'):
                self.openalgo_url = self.openalgo_url.rstrip('/')
                
        except Exception as e:
            # Fallback to direct os.environ access
            import os
            self.openalgo_url = openalgo_url or os.getenv('OPENALGO_URL', 'http://localhost:5000')
            self.openalgo_api_key = os.getenv('OPENALGO_API_KEY')
            
            # Remove trailing slash if present
            if self.openalgo_url.endswith('/'):
                self.openalgo_url = self.openalgo_url.rstrip('/')
                
            print(f"[FALLBACK] Using environment variables directly - URL: {self.openalgo_url}")
        
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
        
        # Initialize data sources based on mode
        self.mode = mode
        self.openalgo_client = None
        
        # LIVE MODE: Only OpenAlgo (no fallbacks to prevent mock data)
        # BACKTESTING MODE: OpenAlgo + Database fallback
        if mode == "live":
            self.data_sources = ['openalgo']  # NO DATABASE FALLBACK in live mode
        else:
            self.data_sources = ['openalgo', 'database']  # Allow database for backtesting
            
        self.last_successful_source = None
        self.cache = {}
        self.cache_timeout = 30  # 30 seconds cache
        
        self._initialize_openalgo()
        self.logger.info(f"[DATA] MarketDataFetcher initialized in {mode} mode with sources: {self.data_sources}")
    
    def _initialize_openalgo(self):
        """Initialize OpenAlgo client if available."""
        try:
            if UTILS_AVAILABLE:
                # Use correct OpenAlgo constructor parameters
                self.openalgo_client = OpenAlgo(host=self.openalgo_url, api_key=self.openalgo_api_key)
                # Test connection using connect() method
                if self.openalgo_client.connect():
                    self.logger.info(f"[OK] OpenAlgo client connected successfully to {self.openalgo_url}")
                else:
                    self.logger.warning(f"[WARNING] OpenAlgo client connection failed to {self.openalgo_url}")
                    self.openalgo_client = None
            else:
                self.logger.warning("[WARNING] OpenAlgo components not available")
        except Exception as e:
            self.logger.warning(f"OpenAlgo initialization failed: {e}")
            self.logger.warning(f"Attempted URL: {self.openalgo_url}, API Key: {'***masked***' if self.openalgo_api_key else 'None'}")
            self.openalgo_client = None
    
    def fetch_real_time_data(self, symbol: str, timeframe: str = "5m") -> Dict[str, Any]:
        """
        Fetch real-time market data with multiple source fallback.
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY')
            timeframe: Data timeframe (e.g., '5m', '1m')
            
        Returns:
            Dictionary with OHLCV data or empty dict if all sources fail
        """
        # LIVE MODE: Fail fast if OpenAlgo not available
        if self.mode == "live" and not self.openalgo_client:
            error_msg = f"[CRITICAL] Live trading requires OpenAlgo connection for {symbol} - no fallbacks available"
            self.logger.error(error_msg)
            raise Exception(error_msg)
        cache_key = f"{symbol}_{timeframe}"
        
        # Check cache first
        if self._is_cache_valid(cache_key):
            self.logger.debug(f"[CACHE] Using cached data for {symbol}")
            return self.cache[cache_key]['data']
        
        # Try each data source in priority order
        for source in self.data_sources:
            try:
                data = self._fetch_from_source(symbol, timeframe, source)
                if data and self._validate_data(data, symbol):
                    self.logger.info(f"[DATA] Successfully fetched {symbol} from {source}")
                    self.last_successful_source = source
                    self._update_cache(cache_key, data)
                    return data
                else:
                    self.logger.warning(f"[WARNING] Invalid data from {source} for {symbol}")
            except Exception as e:
                self.logger.warning(f"[ERROR] {source} fetch failed for {symbol}: {e}")
                continue
        
        # All sources failed - CRITICAL ERROR
        if self.mode == "live":
            error_msg = f"[CRITICAL] Live trading data source failed for {symbol}. OpenAlgo connection lost or broker disconnected."
            self.logger.error(error_msg)
            self.logger.error("[CRITICAL] Live trading cannot continue without real market data - stopping execution")
            raise Exception(f"Live trading data failure for {symbol} - OpenAlgo connection required")
        else:
            self.logger.error(f"[CRITICAL] All real data sources failed for {symbol}. Available sources: {self.data_sources}")
            self.logger.error(f"[CRITICAL] System will NOT provide mock data. Fix data source issues.")
            return {}
    
    def _fetch_from_source(self, symbol: str, timeframe: str, source: str) -> Dict[str, Any]:
        """Fetch data from specific source (PRODUCTION: Only real data sources)."""
        if source == 'openalgo':
            return self._fetch_from_openalgo(symbol, timeframe)
        elif source == 'database':
            return self._fetch_from_database(symbol)
        else:
            self.logger.error(f"[ERROR] Invalid data source: {source}. Only 'openalgo' and 'database' supported.")
            return {}
    
    def _fetch_from_openalgo(self, symbol: str, timeframe: str) -> Dict[str, Any]:
        """Fetch real-time data from OpenAlgo API with comprehensive logging."""
        if not self.openalgo_client:
            self.logger.error("[OPENALGO] Client not initialized")
            return {}
        
        try:
            # Map symbols to exchange (Indian market)
            exchange = 'NSE'  # Default to NSE for NIFTY
            if symbol in ['BANKNIFTY', 'NIFTY']:
                exchange = 'NSE'
            
            self.logger.info(f"[OPENALGO] Fetching {symbol} from {exchange} with timeframe {timeframe}")
            
            # ENHANCED STRATEGY: Try multiple approaches for real-time data
            
            # Strategy 1: Use recent historical data (most reliable for live candles)
            self.logger.info(f"[OPENALGO] Trying historical endpoint for latest {symbol} candle")
            try:
                from datetime import timedelta
                current_time = datetime.now()
                # Get last 10 minutes of data to capture latest candle
                start_time = current_time - timedelta(minutes=10)
                end_date = current_time.strftime('%Y-%m-%d')
                start_date = start_time.strftime('%Y-%m-%d')
                
                recent_data = self.openalgo_client.get_historical_data(
                    symbol=symbol,
                    exchange='NSE_INDEX',
                    interval=timeframe,
                    start_date=start_date,
                    end_date=end_date
                )
                
                if recent_data is not None and not recent_data.empty:
                    # Get the latest candle
                    latest_candle = recent_data.iloc[-1]
                    latest_timestamp = recent_data.index[-1]
                    
                    ohlcv_data = {
                        'timestamp': latest_timestamp,
                        'open': float(latest_candle['open']),
                        'high': float(latest_candle['high']),
                        'low': float(latest_candle['low']),
                        'close': float(latest_candle['close']),
                        'volume': int(latest_candle['volume']),
                        'source': 'openalgo_historical'
                    }
                    
                    self.logger.info(f"[OPENALGO] Latest candle: O:{ohlcv_data['open']:.2f} H:{ohlcv_data['high']:.2f} L:{ohlcv_data['low']:.2f} C:{ohlcv_data['close']:.2f} V:{ohlcv_data['volume']}")
                    
                    # Validate the data is not all zeros
                    if all(ohlcv_data[key] == 0.0 for key in ['open', 'high', 'low', 'close']):
                        self.logger.warning(f"[OPENALGO] Historical data contains zero values - trying quotes endpoint")
                    else:
                        return ohlcv_data
                        
            except Exception as e:
                self.logger.warning(f"[OPENALGO] Historical data fetch failed: {e}")
            
            # Strategy 2: Fallback to quotes endpoint
            self.logger.info(f"[OPENALGO] Trying quotes endpoint for {symbol}")
            quote_response = self.openalgo_client.get_quotes(symbol, 'NSE_INDEX')
            self.logger.info(f"[OPENALGO] Quote API response status: {quote_response.get('status') if quote_response else 'None'}")
            
            if quote_response and quote_response.get('status') == 'success':
                quote_data = quote_response.get('data')
                if quote_data:
                    # Extract price data - check multiple possible field names
                    current_price = float(quote_data.get('ltp', quote_data.get('close', quote_data.get('c', 0))))
                    open_price = float(quote_data.get('open', quote_data.get('o', current_price)))
                    high_price = float(quote_data.get('high', quote_data.get('h', current_price)))
                    low_price = float(quote_data.get('low', quote_data.get('l', current_price)))
                    volume = int(quote_data.get('volume', quote_data.get('v', 0)))
                    
                    ohlcv_data = {
                        'timestamp': datetime.now(),
                        'open': open_price,
                        'high': high_price,
                        'low': low_price,
                        'close': current_price,
                        'volume': volume,
                        'source': 'openalgo_quotes'
                    }
                    
                    # Log the extracted data for debugging
                    self.logger.info(f"[OPENALGO] Quote data: O:{ohlcv_data['open']:.2f} H:{ohlcv_data['high']:.2f} L:{ohlcv_data['low']:.2f} C:{ohlcv_data['close']:.2f} V:{ohlcv_data['volume']}")
                    
                    # Validate the data is meaningful
                    if all(ohlcv_data[key] == 0.0 for key in ['open', 'high', 'low', 'close']):
                        self.logger.error(f"[OPENALGO] Quote API returning zero values - broker may not be connected")
                        return {}
                    
                    return ohlcv_data
            
            # SECONDARY: Try historical data endpoint for latest 5-minute candle
            self.logger.info(f"[OPENALGO] Falling back to historical data API for {timeframe} interval")
            end_date = datetime.now().strftime('%Y-%m-%d')
            start_date = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
            
            # CRITICAL: Log the exact API call being made
            self.logger.info(f"[OPENALGO] Historical API call: symbol={symbol}, exchange={exchange}, interval={timeframe}, start_date={start_date}, end_date={end_date}")
            
            historical_data = self.openalgo_client.get_historical_data(
                symbol=symbol,
                exchange=exchange,
                interval=timeframe,  # This should be '5m' for 5-minute data
                start_date=start_date,
                end_date=end_date
            )
            
            if isinstance(historical_data, pd.DataFrame) and not historical_data.empty:
                latest_row = historical_data.iloc[-1]
                
                ohlcv_data = {
                    'timestamp': latest_row.name if hasattr(latest_row, 'name') else datetime.now(),
                    'open': float(latest_row.get('open', 0)),
                    'high': float(latest_row.get('high', 0)),
                    'low': float(latest_row.get('low', 0)),
                    'close': float(latest_row.get('close', 0)),
                    'volume': int(latest_row.get('volume', 0)),
                    'source': 'openalgo_historical'
                }
                
                self.logger.info(f"[OPENALGO] Historical data: O:{ohlcv_data['open']:.2f} H:{ohlcv_data['high']:.2f} L:{ohlcv_data['low']:.2f} C:{ohlcv_data['close']:.2f} V:{ohlcv_data['volume']}")
                self.logger.info(f"[OPENALGO] Historical dataframe shape: {historical_data.shape}, columns: {historical_data.columns.tolist()}")
                return ohlcv_data
            else:
                self.logger.warning(f"[OPENALGO] Historical data empty or invalid. Type: {type(historical_data)}, Empty: {historical_data.empty if hasattr(historical_data, 'empty') else 'N/A'}")
            
            self.logger.error(f"[OPENALGO] All data fetch methods failed for {symbol}")
            return {}
            
        except Exception as e:
            self.logger.error(f"[OPENALGO] Fetch error for {symbol}: {e}")
            import traceback
            self.logger.error(f"[OPENALGO] Traceback: {traceback.format_exc()}")
            return {}
    
    def _fetch_from_database(self, symbol: str) -> Dict[str, Any]:
        """Fetch latest data from DuckDB database."""
        if not DUCKDB_AVAILABLE:
            return {}
        
        try:
            conn = duckdb.connect(self.db_path)
            
            # Query latest record for symbol
            query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC 
                LIMIT 1
            """
            
            result = conn.execute(query).fetchone()
            conn.close()
            
            if result:
                return {
                    'timestamp': pd.to_datetime(result[0]) if result[0] else datetime.now(),
                    'open': float(result[1]) if result[1] else 0,
                    'high': float(result[2]) if result[2] else 0,
                    'low': float(result[3]) if result[3] else 0,
                    'close': float(result[4]) if result[4] else 0,
                    'volume': int(result[5]) if result[5] else 0,
                    'source': 'database'
                }
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Database fetch error: {e}")
            return {}
    
    # REMOVED: _fetch_from_yahoo() - Not suitable for Indian market real-time data
    
    # REMOVED: _fetch_mock_data() - Production systems use only real market data
    
    def _validate_data(self, data: Dict[str, Any], symbol: str) -> bool:
        """Validate fetched data for sanity."""
        try:
            if not data:
                return False
            
            # Check required fields
            required_fields = ['open', 'high', 'low', 'close', 'volume']
            for field in required_fields:
                if field not in data:
                    return False
            
            # Basic price validation
            ohlc = [data['open'], data['high'], data['low'], data['close']]
            
            # Check for zero or negative prices
            if any(price <= 0 for price in ohlc):
                return False
            
            # Check price relationships
            if data['high'] < max(data['open'], data['close']):
                return False
            if data['low'] > min(data['open'], data['close']):
                return False
            
            # Check for reasonable price ranges (for Indian markets)
            if symbol == 'NIFTY':
                if not (20000 <= data['close'] <= 30000):  # Reasonable NIFTY range
                    self.logger.warning(f"[VALIDATION] NIFTY price {data['close']} outside expected range")
                    return False
            
            # Check volume
            if data['volume'] < 0:
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Data validation error: {e}")
            return False
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self.cache:
            return False
        
        cache_entry = self.cache[cache_key]
        cache_age = (datetime.now() - cache_entry['timestamp']).total_seconds()
        
        return cache_age < self.cache_timeout
    
    def _update_cache(self, cache_key: str, data: Dict[str, Any]):
        """Update cache with new data."""
        self.cache[cache_key] = {
            'timestamp': datetime.now(),
            'data': data
        }
    
    def get_data_source_status(self) -> Dict[str, str]:
        """Get status of all data sources."""
        status = {}
        
        # Check OpenAlgo
        if self.openalgo_client and self.openalgo_client.test_connection():
            status['openalgo'] = 'Available'
        else:
            status['openalgo'] = 'Unavailable'
        
        # Check Database
        if DUCKDB_AVAILABLE:
            try:
                conn = duckdb.connect(self.db_path)
                conn.execute("SELECT 1").fetchone()
                conn.close()
                status['database'] = 'Available'
            except:
                status['database'] = 'Unavailable'
        else:
            status['database'] = 'Module not installed'
        
        # Removed Yahoo Finance and mock data sources for production
        status['last_successful'] = self.last_successful_source or 'None'
        
        return status
    
    def force_refresh(self, symbol: str, timeframe: str = "5m") -> Dict[str, Any]:
        """Force refresh data by clearing cache and fetching fresh data."""
        cache_key = f"{symbol}_{timeframe}"
        if cache_key in self.cache:
            del self.cache[cache_key]
        
        return self.fetch_real_time_data(symbol, timeframe)
    
    def get_multiple_symbols_data(self, symbols: List[str], timeframe: str = "5m") -> Dict[str, Dict[str, Any]]:
        """Fetch data for multiple symbols efficiently."""
        results = {}
        
        for symbol in symbols:
            try:
                data = self.fetch_real_time_data(symbol, timeframe)
                if data:
                    results[symbol] = data
                else:
                    self.logger.warning(f"[WARNING] No data available for {symbol}")
            except Exception as e:
                self.logger.error(f"[ERROR] Failed to fetch data for {symbol}: {e}")
        
        return results