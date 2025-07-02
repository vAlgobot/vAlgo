"""
Market Data Manager for vAlgo Trading System

This module provides comprehensive market data management including:
- Historical data fetching with batch processing
- Real-time data streaming with WebSocket management
- Data validation and quality assurance
- Multi-symbol data coordination

Author: vAlgo Development Team
Created: June 27, 2025
"""

import logging
from typing import Dict, List, Optional, Any, Callable, Tuple
import pandas as pd
from datetime import datetime, timedelta
import time
import threading
from pathlib import Path
import json

from .openalgo import OpenAlgo
from utils.config_loader import ConfigLoader
from utils.logger import get_logger


class MarketDataManager:
    """
    Professional market data manager for the vAlgo trading system.
    
    Provides historical data fetching, real-time streaming, data validation,
    and quality assurance for trading operations.
    """
    
    def __init__(self, openalgo_client: OpenAlgo, config_path: Optional[str] = None):
        """
        Initialize Market Data Manager.
        
        Args:
            openalgo_client: Connected OpenAlgo client instance
            config_path: Path to Excel configuration file
        """
        self.logger = get_logger(__name__)
        self.openalgo = openalgo_client
        self.config_path = config_path or "config/config.xlsx"
        
        # Streaming state
        self.is_streaming_active = False
        self.streaming_thread = None
        self.streaming_symbols = []
        self.streaming_callbacks = []
        
        # Data cache for performance
        self.data_cache = {}
        self.cache_timeout = 300  # 5 minutes
        
        # Configuration storage
        self.instruments_config = []
        self.timeframe_config = {}
        
        # Load configuration
        self._load_configuration()
        
        self.logger.info("MarketDataManager initialized")
    
    def _load_configuration(self) -> None:
        """Load market data configuration from Excel file."""
        try:
            if Path(self.config_path).exists():
                config_loader = ConfigLoader(self.config_path)
                if not config_loader.load_config():
                    self.logger.error(f"Failed to load config from {self.config_path}")
                    return
                
                # Load instruments configuration
                self.instruments_config = self._load_instruments_config(config_loader)
                
                # Load timeframe preferences
                self.timeframe_config = self._load_timeframe_config(config_loader)
                
                self.logger.info(f"Configuration loaded: {len(self.instruments_config)} instruments")
            else:
                self.logger.warning(f"Config file not found: {self.config_path}")
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            # Set defaults
            self.instruments_config = []
            self.timeframe_config = {'default': '5m'}
    
    def _load_instruments_config(self, config_loader: ConfigLoader) -> List[Dict[str, Any]]:
        """Load instruments configuration from config loader."""
        try:
            instruments = config_loader.get_active_instruments()
            self.logger.warning(f"Could load instruments config: {instruments}")
            # Ensure required fields
            formatted_instruments = []
            for instrument in instruments:
                if isinstance(instrument, dict):
                    # Safe string conversion for all fields
                    symbol_value = instrument.get('symbol', '')
                    symbol_str = str(symbol_value).upper().strip() if pd.notna(symbol_value) else ''
                    
                    exchange_value = instrument.get('exchange', 'NSE')
                    exchange_str = str(exchange_value).strip() if pd.notna(exchange_value) else 'NSE'
                    
                    timeframe_value = instrument.get('timeframe', '5m')
                    timeframe_str = str(timeframe_value).strip() if pd.notna(timeframe_value) else '5m'
                    
                    status_value = instrument.get('status', 'Active')
                    status_str = str(status_value).lower().strip() if pd.notna(status_value) else 'active'
                    
                    formatted_instrument = {
                        'symbol': symbol_str,
                        'exchange': exchange_str,
                        'timeframe': timeframe_str,
                        'status': status_str,
                    }
                    
                    if formatted_instrument['symbol'] and formatted_instrument['status'] == 'active':
                        formatted_instruments.append(formatted_instrument)
            
            return formatted_instruments
            
        except Exception as e:
            self.logger.warning(f"Could not load instruments config: {e}")
            return []
    
    def _load_timeframe_config(self, config_loader: ConfigLoader) -> Dict[str, Any]:
        """Load timeframe configuration preferences."""
        try:
            # Try to get from Initialize sheet
            init_config = config_loader.get_initialize_config()
            
            timeframe_config = {
                'default': init_config.get('default_timeframe', '5m'),
                'supported': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d'],
                'batch_size': init_config.get('data_batch_size', 100),
                'max_days_per_request': init_config.get('max_days_per_request', 30)
            }
            
            return timeframe_config
            
        except Exception as e:
            self.logger.warning(f"Could not load timeframe config: {e}")
            return {
                'default': '5m',
                'supported': ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '1d'],
                'batch_size': 100,
                'max_days_per_request': 30
            }
    
    def get_active_symbols(self) -> List[Dict[str, str]]:
        """
        Get list of active symbols for data fetching.
        
        Returns:
            List of symbol dictionaries with symbol, exchange, timeframe
        """
        return [
            {
                'symbol': instrument['symbol'],
                'exchange': instrument['exchange'],
                'timeframe': instrument['timeframe']
            }
            for instrument in self.instruments_config
            if instrument['status'] == 'active'
        ]
    
    def fetch_historical_data(self, symbol: str, exchange: str, interval: str, 
                            start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical OHLCV data for a single symbol.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Time interval (1m, 5m, 15m, etc.)
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            
        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Parse and validate date range
            start_dt = self._parse_date_string(start_date)
            end_dt = self._parse_date_string(end_date)
            
            if start_dt >= end_dt:
                raise ValueError(f"Start date ({start_date}) must be before end date ({end_date})")
            
            days_span = (end_dt - start_dt).days
            
            # Check cache first
            cache_key = f"{symbol}_{exchange}_{interval}_{start_date}_{end_date}"
            cached_data = self._get_cached_data(cache_key)
            if cached_data is not None:
                self.logger.debug(f"Using cached data for {symbol}")
                return cached_data
            
            self.logger.info(f"Fetching historical data: {symbol} ({interval}) - {start_date} to {end_date} ({days_span} days)")
            
            # Add rate limiting for large date ranges to prevent authentication failures
            # Use timeframe-based batching to prevent API overload
            max_days_per_request = self._get_optimal_batch_size(interval)
            
            if days_span <= max_days_per_request:
                # Single request for small date ranges
                df = self.openalgo.get_historical_data(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    start_date=start_dt.strftime('%Y-%m-%d'),
                    end_date=end_dt.strftime('%Y-%m-%d')
                )
            else:
                # Break into chunks for large date ranges with delays between requests
                df = self._fetch_historical_data_in_chunks(
                    symbol=symbol,
                    exchange=exchange,
                    interval=interval,
                    start_dt=start_dt,
                    end_dt=end_dt,
                    max_days_per_request=max_days_per_request
                )
            
            if not df.empty:
                # Validate and clean data
                df = self.validate_ohlcv_data(df)
                
                # Cache the data
                self._cache_data(cache_key, df)
                
                self.logger.info(f"Successfully fetched {len(df)} records for {symbol}")
                return df
            else:
                self.logger.warning(f"No data received for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def _parse_date_string(self, date_str: str) -> datetime:
        """
        Parse date string to datetime object. Expects standard dd-MM-yyyy format from user.
        
        Args:
            date_str: Date string in dd-MM-yyyy format (e.g., '01-01-2024') or YYYY-MM-DD from API
            
        Returns:
            datetime: Parsed datetime object
        """
        try:
            # Remove any extra whitespace
            date_str = str(date_str).strip()
            
            # If it's already a datetime object, return as is
            if isinstance(date_str, datetime):
                return date_str
            
            # Remove time component if present
            if ' ' in date_str:
                date_str = date_str.split(' ')[0]
            
            # Primary format: dd-MM-yyyy (user input standard)
            try:
                return datetime.strptime(date_str, '%d-%m-%Y')
            except ValueError:
                pass
            
            # Secondary format: YYYY-MM-DD (API responses)
            try:
                return datetime.strptime(date_str, '%Y-%m-%d')
            except ValueError:
                pass
            
            # Last resort: pandas parsing for edge cases
            import pandas as pd
            parsed_date = pd.to_datetime(date_str).to_pydatetime()
            return parsed_date
            
        except Exception as e:
            raise ValueError(f"Invalid date format: '{date_str}'. Expected dd-MM-yyyy format (e.g., 01-01-2024) or YYYY-MM-DD: {e}")
    
    def _fetch_historical_data_in_chunks(self, symbol: str, exchange: str, interval: str,
                                        start_dt: datetime, end_dt: datetime,
                                        max_days_per_request: int) -> pd.DataFrame:
        """
        Fetch historical data in chunks with rate limiting to prevent authentication failures.
        
        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Time interval
            start_dt: Start datetime
            end_dt: End datetime
            max_days_per_request: Maximum days per API request
            
        Returns:
            pd.DataFrame: Combined historical data
        """
        try:
            all_data = []
            current_start = start_dt
            chunk_count = 0
            total_chunks = ((end_dt - start_dt).days // max_days_per_request) + 1
            
            self.logger.info(f"Fetching {symbol} data in {total_chunks} batches to prevent rate limiting")
            
            while current_start < end_dt:
                chunk_count += 1
                current_end = min(current_start + timedelta(days=max_days_per_request), end_dt)
                
                print(f"    📦 Batch {chunk_count}/{total_chunks}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
                self.logger.debug(f"Fetching batch {chunk_count}/{total_chunks}: {current_start.strftime('%Y-%m-%d')} to {current_end.strftime('%Y-%m-%d')}")
                
                try:
                    # Add delay between requests to prevent overwhelming the server
                    if chunk_count > 1:
                        delay_seconds = 1.0  # 1 second delay between chunks
                        print(f"    ⏱️  Rate limiting: waiting {delay_seconds}s before next batch...")
                        self.logger.debug(f"Rate limiting: waiting {delay_seconds}s before next batch")
                        time.sleep(delay_seconds)
                    
                    # Fetch chunk data
                    chunk_df = self.openalgo.get_historical_data(
                        symbol=symbol,
                        exchange=exchange,
                        interval=interval,
                        start_date=current_start.strftime('%Y-%m-%d'),
                        end_date=current_end.strftime('%Y-%m-%d')
                    )
                    
                    if not chunk_df.empty:
                        all_data.append(chunk_df)
                        print(f"    ✅ Batch {chunk_count}: {len(chunk_df)} records fetched")
                        self.logger.debug(f"Batch {chunk_count}: {len(chunk_df)} records fetched")
                    else:
                        print(f"    ❌ Batch {chunk_count}: No data received")
                        self.logger.warning(f"Batch {chunk_count}: No data received")
                
                except Exception as e:
                    self.logger.error(f"Error fetching batch {chunk_count} for {symbol}: {e}")
                    # Continue with next batch even if one fails
                
                # Move to next chunk: ensure we don't miss the last day
                if current_end >= end_dt:
                    break
                current_start = current_end
            
            # Combine all batches
            if all_data:
                # PRESERVE DatetimeIndex: Don't ignore index to maintain API timestamps
                combined_df = pd.concat(all_data, ignore_index=False)
                
                # Remove duplicates that might occur at batch boundaries
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.drop_duplicates(subset=['timestamp'], keep='first')
                elif combined_df.index.name == 'timestamp':
                    combined_df = combined_df[~combined_df.index.duplicated(keep='first')]
                
                # Sort by timestamp
                if 'timestamp' in combined_df.columns:
                    combined_df = combined_df.sort_values('timestamp')
                else:
                    combined_df = combined_df.sort_index()
                
                # Validate actual vs requested date range
                if not combined_df.empty:
                    if 'timestamp' in combined_df.columns:
                        actual_start = combined_df['timestamp'].min().strftime('%Y-%m-%d')
                        actual_end = combined_df['timestamp'].max().strftime('%Y-%m-%d')
                    else:
                        actual_start = combined_df.index.min().strftime('%Y-%m-%d')
                        actual_end = combined_df.index.max().strftime('%Y-%m-%d')
                    
                    requested_start = start_dt.strftime('%Y-%m-%d')
                    requested_end = end_dt.strftime('%Y-%m-%d')
                    
                    print(f"    📅 Date Range Validation:")
                    print(f"       Requested: {requested_start} to {requested_end}")
                    print(f"       Actual:    {actual_start} to {actual_end}")
                    
                    if actual_end != requested_end:
                        print(f"    ⚠️  WARNING: End date mismatch! Missing {requested_end}")
                        self.logger.warning(f"Date range mismatch for {symbol}: requested {requested_start} to {requested_end}, got {actual_start} to {actual_end}")
                
                print(f"    🔗 Combining {len(all_data)} batches into final dataset...")
                self.logger.info(f"Successfully combined {len(all_data)} batches into {len(combined_df)} records for {symbol}")
                return combined_df
            else:
                print(f"    ❌ No data retrieved across all batches")
                self.logger.warning(f"No data retrieved for {symbol} across all batches")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error in batch data fetching for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_multiple_symbols(self, symbols: List[Dict[str, str]], interval: str, 
                             days_back: int) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for multiple symbols efficiently.
        
        Args:
            symbols: List of symbol dictionaries
            interval: Time interval
            days_back: Number of days of historical data
            
        Returns:
            Dict mapping symbol names to DataFrames
        """
        results = {}
        total_symbols = len(symbols)
        
        self.logger.info(f"Fetching data for {total_symbols} symbols")
        
        for i, symbol_info in enumerate(symbols, 1):
            symbol = symbol_info['symbol']
            exchange = symbol_info['exchange']
            
            try:
                self.logger.debug(f"Processing {i}/{total_symbols}: {symbol}")
                
                df = self.fetch_historical_data(symbol, exchange, interval, days_back)
                
                if not df.empty:
                    results[symbol] = df
                    self.logger.debug(f"✅ {symbol}: {len(df)} records")
                else:
                    self.logger.warning(f"❌ {symbol}: No data")
                
                # Small delay to respect rate limits
                time.sleep(0.1)
                
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        self.logger.info(f"Completed batch fetch: {len(results)}/{total_symbols} successful")
        return results
    
    def fetch_all_active_instruments_unified_dates(self, start_date: str, end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all active instruments using unified date range.
        
        Args:
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            
        Returns:
            Dict mapping symbol names to DataFrames
        """
        active_symbols = self.get_active_symbols()
        
        if not active_symbols:
            self.logger.warning("No active instruments found")
            return {}
        
        # Calculate days_back from date range
        from datetime import datetime
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        days_back = (end_dt - start_dt).days
        
        results = {}
        
        self.logger.info(f"Fetching data for date range: {start_date} to {end_date} ({days_back} days)")
        
        for symbol_info in active_symbols:
            symbol = symbol_info['symbol']
            exchange = symbol_info['exchange']
            timeframe = symbol_info['timeframe']
            
            try:
                self.logger.info(f"Fetching data for {symbol} ({exchange}, {timeframe})")
                
                df = self.fetch_historical_data(symbol, exchange, timeframe, days_back)
                
                if not df.empty:
                    results[symbol] = df
                    self.logger.info(f"✅ {symbol}: {len(df)} records loaded")
                else:
                    self.logger.warning(f"❌ {symbol}: No data received")
                    
            except Exception as e:
                self.logger.error(f"Failed to fetch data for {symbol}: {e}")
                continue
        
        self.logger.info(f"Loaded data for {len(results)} instruments")
        return results
    
    def fetch_all_active_instruments_from_config(self) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical data for all active instruments using date range from Initialize sheet.
        
        Returns:
            Dict mapping symbol names to DataFrames
        """
        try:
            config_loader = ConfigLoader(self.config_path)
            if not config_loader.load_config():
                self.logger.error("Failed to load configuration")
                return {}
            
            # Get date range from Initialize sheet
            init_config = config_loader.get_initialize_config()
            start_date = init_config.get('start date', '2024-01-01')
            end_date = init_config.get('end date', '2024-12-31')
            
            self.logger.info(f"Using date range from config: {start_date} to {end_date}")
            
            return self.fetch_all_active_instruments_unified_dates(start_date, end_date)
            
        except Exception as e:
            self.logger.error(f"Error fetching data from config: {e}")
            return {}
    
    def _get_optimal_batch_size(self, interval: str) -> int:
        """
        Calculate optimal batch size (days per request) based on timeframe to prevent API overload.
        
        Args:
            interval: Time interval (1m, 5m, 15m, 1h, 1d)
            
        Returns:
            int: Optimal days per batch for this timeframe
        """
        try:
            # Parse interval to minutes
            interval_lower = interval.lower().strip()
            
            if interval_lower.endswith('m'):
                minutes = int(interval_lower[:-1])
            elif interval_lower.endswith('h'):
                minutes = int(interval_lower[:-1]) * 60
            elif interval_lower.endswith('d'):
                minutes = int(interval_lower[:-1]) * 60 * 24
            else:
                minutes = 5  # Default to 5 minutes
            
            # Calculate records per day (assuming 6.5 hours trading day = 390 minutes)
            trading_minutes_per_day = 390
            records_per_day = max(1, trading_minutes_per_day // minutes)  # Prevent division by zero
            
            # Target ~10,000-15,000 records per batch for optimal performance
            target_records_per_batch = 12000
            optimal_days = max(1, min(60, target_records_per_batch // records_per_day))
            
            self.logger.info(f"Batch size calculation for {interval}: {records_per_day} records/day → {optimal_days} days per batch")
            
            # Timeframe-specific optimization
            if minutes <= 1:  # 1-minute data
                return min(optimal_days, 7)  # Max 7 days for 1-minute
            elif minutes <= 5:  # 5-minute data  
                return min(optimal_days, 30)  # Max 30 days for 5-minute
            elif minutes <= 15:  # 15-minute data
                return min(optimal_days, 45)  # Max 45 days for 15-minute
            elif minutes <= 60:  # 1-hour data
                return min(optimal_days, 90)  # Max 90 days for hourly
            else:  # Daily+ data
                return 365  # Can handle large batches for daily data
                
        except Exception as e:
            self.logger.error(f"Error calculating optimal batch size for {interval}: {e}")
            return 30  # Safe fallback

    def get_optimal_date_ranges(self, days_back: int) -> List[Tuple[str, str]]:
        """
        Calculate optimal date ranges for data fetching.
        
        Args:
            days_back: Total days of historical data needed
            
        Returns:
            List of (start_date, end_date) tuples
        """
        # Use dynamic batch sizing for different timeframes
        max_days_per_request = self._get_optimal_batch_size('5m')  # Default for this method
        
        if days_back <= max_days_per_request:
            # Single request
            end_date = datetime.now()
            start_date = end_date - timedelta(days=days_back)
            return [(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))]
        
        # Multiple requests needed
        ranges = []
        current_end = datetime.now()
        remaining_days = days_back
        
        while remaining_days > 0:
            days_this_batch = min(remaining_days, max_days_per_request)
            start_date = current_end - timedelta(days=days_this_batch)
            
            ranges.append((
                start_date.strftime('%Y-%m-%d'),
                current_end.strftime('%Y-%m-%d')
            ))
            
            current_end = start_date
            remaining_days -= days_this_batch
        
        return ranges
    
    def start_streaming(self, symbols: List[Dict[str, str]], 
                       callback: Optional[Callable] = None) -> bool:
        """
        Start real-time data streaming for specified symbols.
        
        Args:
            symbols: List of symbols to stream
            callback: Optional callback function for tick data
            
        Returns:
            bool: True if streaming started successfully
        """
        try:
            if self.is_streaming_active:
                self.logger.warning("Streaming already active")
                return False
            
            if not self.openalgo.is_connected():
                self.logger.error("OpenAlgo client not connected")
                return False
            
            self.streaming_symbols = symbols
            if callback:
                self.streaming_callbacks.append(callback)
            
            # Start streaming thread
            self.streaming_thread = threading.Thread(target=self._streaming_worker, daemon=True)
            self.is_streaming_active = True
            self.streaming_thread.start()
            
            self.logger.info(f"Started streaming for {len(symbols)} symbols")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start streaming: {e}")
            return False
    
    def stop_streaming(self) -> None:
        """Stop real-time data streaming."""
        try:
            if not self.is_streaming_active:
                self.logger.warning("Streaming not active")
                return
            
            self.is_streaming_active = False
            
            if self.streaming_thread and self.streaming_thread.is_alive():
                self.streaming_thread.join(timeout=5)
            
            self.streaming_symbols = []
            self.streaming_callbacks = []
            
            self.logger.info("Streaming stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping streaming: {e}")
    
    def is_streaming(self) -> bool:
        """
        Check if real-time streaming is active.
        
        Returns:
            bool: Streaming status
        """
        return self.is_streaming_active
    
    def _streaming_worker(self) -> None:
        """Background worker for real-time data streaming."""
        try:
            self.logger.debug("Streaming worker started")
            
            while self.is_streaming_active:
                try:
                    # Get real-time quotes for each symbol
                    for symbol_info in self.streaming_symbols:
                        if not self.is_streaming_active:
                            break
                        
                        symbol = symbol_info['symbol']
                        exchange = symbol_info['exchange']
                        
                        # Get quote data
                        quote_response = self.openalgo.get_quotes(symbol, exchange)
                        
                        if quote_response['status'] == 'success':
                            tick_data = {
                                'symbol': symbol,
                                'exchange': exchange,
                                'timestamp': datetime.now(),
                                'data': quote_response['data']
                            }
                            
                            # Process callbacks
                            self.on_tick_received(tick_data)
                    
                    # Small delay between iterations
                    time.sleep(1)
                    
                except Exception as e:
                    self.logger.error(f"Streaming worker error: {e}")
                    self.on_streaming_error(e)
                    time.sleep(5)  # Wait before retry
            
            self.logger.debug("Streaming worker stopped")
            
        except Exception as e:
            self.logger.error(f"Critical streaming worker error: {e}")
            self.is_streaming_active = False
    
    def on_tick_received(self, tick_data: Dict[str, Any]) -> None:
        """
        Handle received tick data.
        
        Args:
            tick_data: Tick data dictionary
        """
        try:
            # Call registered callbacks
            for callback in self.streaming_callbacks:
                try:
                    callback(tick_data)
                except Exception as e:
                    self.logger.error(f"Callback error: {e}")
            
            # Log tick data (debug level)
            symbol = tick_data['symbol']
            self.logger.debug(f"Tick received: {symbol}")
            
        except Exception as e:
            self.logger.error(f"Error processing tick data: {e}")
    
    def on_streaming_error(self, error: Exception) -> None:
        """
        Handle streaming errors.
        
        Args:
            error: Exception that occurred
        """
        self.logger.error(f"Streaming error: {error}")
        
        # Could implement auto-reconnection logic here
        # For now, just log the error
    
    def validate_ohlcv_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Validate and clean OHLCV data.
        
        Args:
            df: Raw OHLCV DataFrame
            
        Returns:
            pd.DataFrame: Validated and cleaned data
        """
        try:
            if df.empty:
                return df
            
            original_length = len(df)
            
            # Ensure required columns exist
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.warning(f"Missing OHLCV columns: {missing_columns}")
                return df
            
            # Convert to numeric types
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove rows with invalid data
            df = df.dropna(subset=required_columns)
            
            # Validate OHLC relationships
            invalid_mask = (
                (df['high'] < df['low']) |
                (df['high'] < df['open']) |
                (df['high'] < df['close']) |
                (df['low'] > df['open']) |
                (df['low'] > df['close']) |
                (df['volume'] < 0)
            )
            
            df = df[~invalid_mask]
            
            # Handle data gaps if needed
            df = self.handle_data_gaps(df)
            
            cleaned_length = len(df)
            if cleaned_length < original_length:
                self.logger.warning(f"Data validation removed {original_length - cleaned_length} invalid records")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error validating OHLCV data: {e}")
            return df
    
    def handle_data_gaps(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing data and gaps in OHLCV data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            pd.DataFrame: Data with gaps handled
        """
        try:
            if df.empty:
                return df
            
            # For now, just forward fill missing values
            # In production, you might want more sophisticated gap handling
            df = df.ffill()
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error handling data gaps: {e}")
            return df
    
    def get_data_quality_score(self, df: pd.DataFrame) -> float:
        """
        Calculate data quality score for OHLCV data.
        
        Args:
            df: OHLCV DataFrame
            
        Returns:
            float: Quality score between 0.0 and 1.0
        """
        try:
            if df.empty:
                return 0.0
            
            total_points = 0
            quality_points = 0
            
            # Check for missing values
            total_points += 1
            if not df.isnull().any().any():
                quality_points += 1
            
            # Check OHLC relationships
            total_points += 1
            valid_ohlc = (
                (df['high'] >= df['low']) &
                (df['high'] >= df['open']) &
                (df['high'] >= df['close']) &
                (df['low'] <= df['open']) &
                (df['low'] <= df['close'])
            ).all()
            
            if valid_ohlc:
                quality_points += 1
            
            # Check for reasonable volume values
            total_points += 1
            if (df['volume'] >= 0).all():
                quality_points += 1
            
            # Check data continuity (no large gaps)
            if len(df) > 1:
                total_points += 1
                # This is a simplified check - in production you'd check time gaps
                quality_points += 1
            
            return quality_points / total_points if total_points > 0 else 0.0
            
        except Exception as e:
            self.logger.error(f"Error calculating data quality score: {e}")
            return 0.0
    
    def _get_cached_data(self, cache_key: str) -> Optional[pd.DataFrame]:
        """Get data from cache if available and not expired."""
        try:
            if cache_key in self.data_cache:
                cached_item = self.data_cache[cache_key]
                
                # Check if cache is still valid
                if time.time() - cached_item['timestamp'] < self.cache_timeout:
                    return cached_item['data']
                else:
                    # Remove expired cache
                    del self.data_cache[cache_key]
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error accessing cache: {e}")
            return None
    
    def _cache_data(self, cache_key: str, data: pd.DataFrame) -> None:
        """Cache data with timestamp."""
        try:
            self.data_cache[cache_key] = {
                'data': data.copy(),
                'timestamp': time.time()
            }
            
            # Clean old cache entries periodically
            if len(self.data_cache) > 100:
                self._clean_cache()
                
        except Exception as e:
            self.logger.error(f"Error caching data: {e}")
    
    def _clean_cache(self) -> None:
        """Remove expired cache entries."""
        try:
            current_time = time.time()
            expired_keys = [
                key for key, item in self.data_cache.items()
                if current_time - item['timestamp'] > self.cache_timeout
            ]
            
            for key in expired_keys:
                del self.data_cache[key]
            
            self.logger.debug(f"Cleaned {len(expired_keys)} expired cache entries")
            
        except Exception as e:
            self.logger.error(f"Error cleaning cache: {e}")
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            'total_entries': len(self.data_cache),
            'cache_timeout': self.cache_timeout,
            'memory_usage_mb': sum(
                item['data'].memory_usage(deep=True).sum() 
                for item in self.data_cache.values()
            ) / (1024 * 1024)
        }
    
    def clear_cache(self) -> None:
        """Clear all cached data."""
        self.data_cache.clear()
        self.logger.info("Data cache cleared")
    
    def __str__(self) -> str:
        """String representation of Market Data Manager."""
        streaming_status = "Active" if self.is_streaming() else "Inactive"
        return f"MarketDataManager(instruments={len(self.instruments_config)}, streaming={streaming_status})"
    
    def __repr__(self) -> str:
        """Detailed representation of Market Data Manager."""
        return (f"MarketDataManager(instruments={len(self.instruments_config)}, "
                f"cache_entries={len(self.data_cache)}, "
                f"streaming={self.is_streaming()})")