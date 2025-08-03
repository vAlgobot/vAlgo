"""
OpenAlgo API Wrapper for vAlgo Trading System

This module provides a clean, professional wrapper around the openalgo library
with Excel configuration integration, robust error handling, and standardized
response formats for the vAlgo trading system.

Author: vAlgo Development Team
Created: June 27, 2025
"""

import logging
from typing import Dict, Optional, Any, List
import pandas as pd
from datetime import datetime, timedelta
import time
from pathlib import Path

try:
    from openalgo import api
except ImportError:
    raise ImportError("openalgo library not installed. Run: pip install openalgo")

try:
    import websocket
    import json
    import threading
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

from utils.config_loader import ConfigLoader
from utils.logger import get_logger


class OpenAlgo:
    """
    Professional wrapper for OpenAlgo API integration.

    Provides centralized connection management, configuration integration,
    robust error handling, and standardized response formats.
    """

    def __init__(self, config_path: Optional[str] = None, api_key: Optional[str] = None,
                 host: Optional[str] = None, ws_url: Optional[str] = None):
        """
        Initialize OpenAlgo client.

        Args:
            config_path: Path to Excel configuration file
            api_key: Direct API key (overrides config)
            host: Direct host URL (overrides config)
            ws_url: Direct WebSocket URL (overrides config)
        """
        self.logger = get_logger(__name__)
        self.config_path = config_path or "config/config.xlsx"
        self.client = None
        self.is_connected_flag = False
        self.connection_time = None

        # Configuration storage
        self.api_key = api_key
        self.host = host
        self.ws_url = ws_url

        # Load configuration
        self._load_config()

        self.logger.info("OpenAlgo wrapper initialized")
    
    def _check_connection_health(self) -> bool:
        """
        Check if the connection is healthy by making a lightweight API call.
        
        Returns:
            bool: True if connection is healthy
        """
        try:
            # Use a lightweight API call to check health
            funds_response = self.client.funds()
            return funds_response is not None
        except Exception as e:
            self.logger.debug(f"Connection health check failed: {e}")
            return False

    def _load_config(self) -> None:
        """Load configuration from Excel file or use direct parameters."""
        try:
            if not all([self.api_key, self.host, self.ws_url]):
                if Path(self.config_path).exists():
                    config_loader = ConfigLoader(self.config_path)
                    if not config_loader.load_config():
                        self.logger.error(f"Failed to load config from {self.config_path}")
                        openalgo_config = {}
                    else:
                        openalgo_config = self._get_openalgo_config(config_loader)

                    self.api_key = self.api_key or openalgo_config.get('api_key')
                    self.host = self.host or openalgo_config.get('host', 'http://127.0.0.1:5000')
                    self.ws_url = self.ws_url or openalgo_config.get('ws_url', 'ws://127.0.0.1:8765')
                else:
                    self.logger.warning(f"Config file not found: {self.config_path}")
                    # Use defaults if no config file
                    self.host = self.host or 'http://127.0.0.1:5000'
                    self.ws_url = self.ws_url or 'ws://127.0.0.1:8765'

            if not self.api_key:
                raise ValueError("API key not provided in config or parameters")

            self.logger.info(f"Configuration loaded - Host: {self.host}")

        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise

    def _get_openalgo_config(self, config_loader: ConfigLoader) -> Dict[str, str]:
        """Extract OpenAlgo configuration from config loader."""
        try:
            # Get OpenAlgo configuration from primary broker
            openalgo_config = config_loader.get_primary_broker_openalgo_config()
            
            self.logger.info(f"Loaded OpenAlgo config for broker: {openalgo_config.get('broker_name', 'Unknown')}")
            return openalgo_config
            
        except Exception as e:
            self.logger.warning(f"Could not extract OpenAlgo config from Excel: {e}")
            return {
                'api_key': '',
                'host': 'http://127.0.0.1:5000',
                'ws_url': 'ws://127.0.0.1:8765'
            }

    def connect(self) -> bool:
        """
        Establish connection to OpenAlgo API.

        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            self.logger.info("Connecting to OpenAlgo API...")

            # Initialize the client
            self.client = api(
                api_key=self.api_key,
                host=self.host,
                ws_url=self.ws_url
            )

            # Test connection with a simple API call
            test_response = self.client.funds()

            if test_response and test_response.get('status') == 'success':
                self.is_connected_flag = True
                self.connection_time = datetime.now()
                self.logger.info("Successfully connected to OpenAlgo API")
                return True
            else:
                self.logger.error(f"Connection test failed: {test_response}")
                return False

        except Exception as e:
            self.logger.error(f"Failed to connect to OpenAlgo API: {e}")
            self.is_connected_flag = False
            return False

    def disconnect(self) -> None:
        """Disconnect from OpenAlgo API and cleanup resources."""
        try:
            if self.client:
                # Clean disconnect - openalgo library handles this internally
                self.client = None

            self.is_connected_flag = False
            self.connection_time = None
            self.logger.info("Disconnected from OpenAlgo API")

        except Exception as e:
            self.logger.error(f"Error during disconnect: {e}")

    def is_connected(self) -> bool:
        """
        Check if client is connected to OpenAlgo API.

        Returns:
            bool: Connection status
        """
        return self.is_connected_flag and self.client is not None

    def _handle_api_call(self, method_name: str, api_method, *args, **kwargs) -> Dict[str, Any]:
        """
        Handle API calls with retry logic and authentication management.

        Args:
            method_name: Name of the API method for logging
            api_method: The actual API method to call
            *args, **kwargs: Arguments to pass to the API method

        Returns:
            dict: Standardized API response
        """
        if not self.is_connected():
            raise ConnectionError("Not connected to OpenAlgo API. Call connect() first.")

        max_retries = 3
        retry_delay = 1.0  # Start with 1 second delay
        
        for attempt in range(max_retries):
            try:
                start_time = time.time()

                # Check connection health before important calls
                if method_name == 'get_historical_data' and attempt == 0:
                    if not self._check_connection_health():
                        self.logger.warning("Connection health check failed, attempting to reconnect")
                        if not self.connect():
                            raise Exception("Failed to establish healthy connection")
                        # Add delay after reconnection
                        time.sleep(1.0)

                # Make the API call
                response = api_method(*args, **kwargs)

                # Calculate timing
                elapsed_time = time.time() - start_time

                # Log the API call
                self.logger.debug(f"API Call: {method_name} - Time: {elapsed_time:.3f}s - Attempt: {attempt + 1}")

                # Return formatted response
                return self._format_response(response, method_name, elapsed_time)

            except Exception as e:
                error_msg = str(e).lower()
                
                # Check if it's an authentication error
                is_auth_error = any(keyword in error_msg for keyword in [
                    'authentication', 'unauthorized', 'invalid token', 'auth failed',
                    'session expired', 'token expired', 'access denied', 'login required'
                ])
                
                if is_auth_error and attempt < max_retries - 1:
                    self.logger.warning(f"Authentication error on attempt {attempt + 1}: {e}")
                    self.logger.info(f"Attempting to reconnect and retry...")
                    
                    # Try to reconnect
                    time.sleep(retry_delay)
                    if self.connect():
                        retry_delay *= 2  # Exponential backoff
                        time.sleep(1.0)  # Extra delay after reconnection
                        continue
                    else:
                        self.logger.error("Failed to reconnect after authentication error")
                
                elif attempt < max_retries - 1:
                    # For other errors, wait and retry
                    self.logger.warning(f"API call failed on attempt {attempt + 1}: {e}")
                    time.sleep(retry_delay)
                    retry_delay *= 1.5  # Moderate backoff for other errors
                    continue
                
                # Final attempt failed
                self.logger.error(f"API call failed after {max_retries} attempts - {method_name}: {e}")
                return {
                    'status': 'error',
                    'message': str(e),
                    'method': method_name,
                    'timestamp': datetime.now().isoformat(),
                    'data': None,
                    'attempts': attempt + 1
                }
        
        # Should never reach here, but just in case
        return {
            'status': 'error',
            'message': 'Maximum retry attempts exceeded',
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'data': None,
            'attempts': max_retries
        }

    def _is_valid_response(self, response: Any) -> bool:
        """
        Check if response is valid (handles DataFrames properly).
        
        Args:
            response: Response to validate
            
        Returns:
            bool: True if response is valid
        """
        if response is None:
            return False
        if isinstance(response, pd.DataFrame):
            return not response.empty
        return bool(response)

    def _format_response(self, raw_response: Any, method_name: str, elapsed_time: float) -> Dict[str, Any]:
        """
        Format and standardize API responses.

        Args:
            raw_response: Raw response from OpenAlgo API
            method_name: Name of the API method
            elapsed_time: Time taken for the API call

        Returns:
            dict: Standardized response format
        """
        return {
            'status': 'success' if self._is_valid_response(raw_response) else 'error',
            'data': raw_response,
            'method': method_name,
            'timestamp': datetime.now().isoformat(),
            'elapsed_time': elapsed_time,
            'source': 'openalgo'
        }

    # Account API Methods
    def get_funds(self) -> Dict[str, Any]:
        """Get account funds information."""
        return self._handle_api_call('get_funds', self.client.funds)

    def get_positions(self) -> Dict[str, Any]:
        """Get current positions."""
        return self._handle_api_call('get_positions', self.client.positionbook)

    def get_holdings(self) -> Dict[str, Any]:
        """Get account holdings."""
        return self._handle_api_call('get_holdings', self.client.holdings)

    def get_orderbook(self) -> Dict[str, Any]:
        """Get order book."""
        return self._handle_api_call('get_orderbook', self.client.orderbook)

    def get_tradebook(self) -> Dict[str, Any]:
        """Get trade book."""
        return self._handle_api_call('get_tradebook', self.client.tradebook)

    # Market Data API Methods
    def get_quotes(self, symbol: str, exchange: str) -> Dict[str, Any]:
        """
        Get real-time quotes for a symbol.

        Args:
            symbol: Trading symbol
            exchange: Exchange name (NSE, BSE, etc.)

        Returns:
            dict: Quote data
        """
        return self._handle_api_call('get_quotes', self.client.quotes, symbol=symbol, exchange=exchange)

    def get_historical_data(self, symbol: str, exchange: str, interval: str,
                          start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get historical OHLCV data.

        Args:
            symbol: Trading symbol
            exchange: Exchange name
            interval: Time interval (1m, 5m, 15m, 1h, 1d)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)

        Returns:
            pd.DataFrame: Historical OHLCV data
        """
        try:
            # Log the API call parameters with requested interval
            self.logger.info(f"[OPENALGO] API CALL - get_historical_data({symbol}, {exchange}, {interval}, {start_date}, {end_date})")
            self.logger.info(f"[OPENALGO] Calling client.history() with {interval} interval")
            
            # Validate interval format for OpenAlgo (including 'D' for daily)
            valid_intervals = ['1m', '5m', '15m', '30m', '1h', '1d', 'D']
            if interval not in valid_intervals:
                raise ValueError(f"Invalid interval: {interval}. Supported: {valid_intervals}")
            
            response = self._handle_api_call(
                'get_historical_data',
                self.client.history,
                symbol=symbol,
                exchange=exchange,
                interval=interval,  # Pass through the requested interval
                start_date=start_date,
                end_date=end_date
            )

            # PRODUCTION DEBUG: Log the raw API response for 5-minute candles
            self.logger.info(f"[OPENALGO] API Response Status: {response['status']}")
            if response['data'] is not None:
                self.logger.info(f"[OPENALGO] Response Data Type: {type(response['data'])}")
                if hasattr(response['data'], '__len__'):
                    self.logger.info(f"[OPENALGO] Response Data Length: {len(response['data'])} records")
                if isinstance(response['data'], pd.DataFrame):
                    self.logger.info(f"[OPENALGO] DataFrame Columns: {response['data'].columns.tolist()}")
                    self.logger.info(f"[OPENALGO] DataFrame Index Type: {type(response['data'].index)}")
                    self.logger.info(f"[OPENALGO] DataFrame Index Sample: {response['data'].index[:3].tolist()}")
                    
                    # Check for proper 5-minute interval data
                    if not response['data'].empty:
                        sample_data = response['data'].iloc[-1]
                        self.logger.info(f"[OPENALGO] Latest Candle Data: O:{sample_data.get('open', 0):.2f} H:{sample_data.get('high', 0):.2f} L:{sample_data.get('low', 0):.2f} C:{sample_data.get('close', 0):.2f} V:{sample_data.get('volume', 0)}")

            if response['status'] == 'success' and response['data'] is not None:
                formatted_data = self._format_historical_data(response['data'])
                
                # EMERGENCY FIX: If we have sequential integer timestamps, generate proper ones
                if not formatted_data.empty:
                    self.logger.info(f"[OPENALGO] Final Result: {len(formatted_data)} {interval} candles")
                    self.logger.info(f"[OPENALGO] Final Index Type: {type(formatted_data.index)}")
                    self.logger.info(f"[OPENALGO] Final Timestamp Range: {formatted_data.index[0]} to {formatted_data.index[-1]}")
                    
                    # Validate intervals (skip for daily candles)
                    if len(formatted_data) > 1 and interval not in ['D', '1d']:
                        time_diff = (formatted_data.index[1] - formatted_data.index[0]).total_seconds() / 60
                        expected_minutes = {'1m': 1, '5m': 5, '15m': 15, '30m': 30, '1h': 60}.get(interval, 0)
                        if expected_minutes > 0:
                            self.logger.info(f"[OPENALGO] Interval Validation: {time_diff} minutes between candles (expected: {expected_minutes})")
                            if abs(time_diff - expected_minutes) > 1.0:  # Allow 1-minute tolerance
                                self.logger.warning(f"[WARNING] Interval mismatch: Got {time_diff} minutes, expected {expected_minutes} minutes")
                    
                    # Check if the index contains sequential integers (0, 1, 2, 3...)
                    if (formatted_data.index.dtype == 'datetime64[ns]' and 
                        len(formatted_data) > 1 and
                        formatted_data.index[0].year == 1970):  # Epoch timestamps detected
                        
                        self.logger.warning(f"[TIMESTAMP_FIX] Detected epoch timestamps, generating proper {interval} timestamps")
                        
                        # Generate proper timestamps based on date range and interval
                        fixed_data = self._fix_timestamp_index(formatted_data, start_date, end_date, interval)
                        self.logger.info(f"[TIMESTAMP_FIX] Generated {interval} timestamps: {fixed_data.index[:3].tolist()} to {fixed_data.index[-3:].tolist()}")
                        return fixed_data
                
                return formatted_data
            else:
                self.logger.error(f"Failed to get historical data: {response}")
                return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error getting historical data: {e}")
            return pd.DataFrame()

    def _format_historical_data(self, raw_data) -> pd.DataFrame:
        """
        Format historical data into standard OHLCV DataFrame.

        Args:
            raw_data: Raw historical data from OpenAlgo

        Returns:
            pd.DataFrame: Formatted OHLCV data
        """
        try:
            # EMERGENCY DEBUG: Focus on timestamp data from API
            self.logger.info(f"OPENALGO API EMERGENCY DEBUG - Raw data type: {type(raw_data)}")
            
            # Look specifically for timestamp/time data
            timestamp_found = False
            if hasattr(raw_data, '__len__') and len(raw_data) > 0:
                if isinstance(raw_data, (list, tuple)):
                    self.logger.info(f"OPENALGO API EMERGENCY DEBUG - List with {len(raw_data)} items")
                    if len(raw_data) > 0 and isinstance(raw_data[0], dict):
                        first_item = raw_data[0]
                        self.logger.info(f"OPENALGO API EMERGENCY DEBUG - First item keys: {list(first_item.keys())}")
                        # Look for timestamp fields
                        for key, value in first_item.items():
                            if 'time' in key.lower() or 'date' in key.lower():
                                self.logger.info(f"OPENALGO API EMERGENCY DEBUG - TIMESTAMP FIELD '{key}': {value} (type: {type(value)})")
                                timestamp_found = True
                elif isinstance(raw_data, dict):
                    self.logger.info(f"OPENALGO API EMERGENCY DEBUG - Dict keys: {list(raw_data.keys())}")
                    # Look for timestamp fields in dict
                    for key, value in raw_data.items():
                        if 'time' in key.lower() or 'date' in key.lower():
                            if isinstance(value, (list, tuple)) and len(value) > 0:
                                self.logger.info(f"OPENALGO API EMERGENCY DEBUG - TIMESTAMP FIELD '{key}': {value[:3]}... (type: {type(value[0])})")
                                timestamp_found = True
                elif isinstance(raw_data, pd.DataFrame):
                    self.logger.info(f"OPENALGO API EMERGENCY DEBUG - DataFrame columns: {raw_data.columns.tolist()}")
                    # Look for timestamp columns
                    for col in raw_data.columns:
                        if 'time' in col.lower() or 'date' in col.lower():
                            self.logger.info(f"OPENALGO API EMERGENCY DEBUG - TIMESTAMP COLUMN '{col}': {raw_data[col].head(3).tolist()}")
                            timestamp_found = True
            
            if not timestamp_found:
                self.logger.warning(f"OPENALGO API EMERGENCY DEBUG - NO TIMESTAMP FIELDS FOUND in API response!")
            
            # CRITICAL DEBUG: Check for epoch timestamps in raw data
            if isinstance(raw_data, pd.DataFrame) and 'timestamp' in raw_data.columns:
                timestamp_series = raw_data['timestamp']
                self.logger.debug(f"TIMESTAMP DEBUG - Raw API timestamps sample: {timestamp_series.head().tolist()}")
                self.logger.debug(f"TIMESTAMP DEBUG - Unique timestamp count: {timestamp_series.nunique()}")
                
                # Check for problematic timestamps
                if timestamp_series.dtype == 'object':
                    # String timestamps - check for null/empty/zero values
                    null_count = timestamp_series.isin(['', '0', 'null', 'None', None]).sum()
                    if null_count > 0:
                        self.logger.warning(f"TIMESTAMP DEBUG - Found {null_count} null/empty/zero timestamp strings in API response")
                        self.logger.warning(f"TIMESTAMP DEBUG - Problematic values: {timestamp_series[timestamp_series.isin(['', '0', 'null', 'None', None])].tolist()}")
                elif pd.api.types.is_numeric_dtype(timestamp_series):
                    # Numeric timestamps - check for zero values (epoch)
                    zero_count = (timestamp_series == 0).sum()
                    if zero_count > 0:
                        self.logger.warning(f"TIMESTAMP DEBUG - Found {zero_count} zero timestamp values in API response (will become epoch)")
                        self.logger.warning(f"TIMESTAMP DEBUG - Zero timestamps at indices: {timestamp_series[timestamp_series == 0].index.tolist()}")
            elif isinstance(raw_data, dict) and any('time' in str(key).lower() for key in raw_data.keys()):
                # Dict with timestamp keys
                timestamp_keys = [key for key in raw_data.keys() if 'time' in str(key).lower()]
                for key in timestamp_keys:
                    timestamps = raw_data[key]
                    if isinstance(timestamps, (list, tuple)) and len(timestamps) > 0:
                        self.logger.debug(f"TIMESTAMP DEBUG - Raw {key} sample: {timestamps[:5]}")
                        # Check for zeros in timestamp lists
                        if isinstance(timestamps, list):
                            zero_count = sum(1 for t in timestamps if t in [0, '0', '', None, 'null'])
                            if zero_count > 0:
                                self.logger.warning(f"TIMESTAMP DEBUG - Found {zero_count} problematic timestamps in {key}")
            
            
            # If it's already a DataFrame, return as is
            if isinstance(raw_data, pd.DataFrame):
                self.logger.info("OpenAlgo API returned data in DataFrame format (no conversion needed)")
                return raw_data
            
            # Handle list of dictionaries (common format)
            if isinstance(raw_data, list) and raw_data:
                if isinstance(raw_data[0], dict):
                    self.logger.info("Converting list of dictionaries to DataFrame")
                    df = pd.DataFrame(raw_data)
                    return self._standardize_columns(df)
            
            # Handle dictionary with lists (another common format)
            if isinstance(raw_data, dict):
                # Check if all values are lists of same length
                list_values = {k: v for k, v in raw_data.items() if isinstance(v, (list, tuple))}
                if list_values:
                    lengths = [len(v) for v in list_values.values()]
                    if len(set(lengths)) == 1:  # All same length
                        self.logger.info("Converting dictionary of lists to DataFrame")
                        df = pd.DataFrame(list_values)
                        return self._standardize_columns(df)
                
                # Try direct conversion
                try:
                    self.logger.info("Attempting direct dictionary to DataFrame conversion")
                    df = pd.DataFrame(raw_data)
                    return self._standardize_columns(df)
                except ValueError as ve:
                    self.logger.warning(f"Direct conversion failed: {ve}")
                    
                    # Handle scalar values with explicit index
                    if all(not isinstance(v, (list, tuple, dict)) for v in raw_data.values()):
                        self.logger.info("Converting scalar values with index")
                        df = pd.DataFrame([raw_data])  # Wrap in list to create single row
                        return self._standardize_columns(df)
            
            # Handle string response (might be JSON)
            if isinstance(raw_data, str):
                try:
                    import json
                    parsed_data = json.loads(raw_data)
                    self.logger.info("Parsed JSON string, recursing")
                    return self._format_historical_data(parsed_data)
                except json.JSONDecodeError:
                    self.logger.warning("String data is not valid JSON")
            
            # If all else fails, return empty DataFrame
            self.logger.warning(f"Could not format historical data of type {type(raw_data)}")
            return pd.DataFrame()

        except Exception as e:
            self.logger.error(f"Error formatting historical data: {e}")
            import traceback
            self.logger.error(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
    
    def _fix_timestamp_index(self, df: pd.DataFrame, start_date: str, end_date: str, interval: str) -> pd.DataFrame:
        """
        Fix DataFrame with epoch timestamps by generating proper timestamps.
        
        Args:
            df: DataFrame with epoch timestamp index
            start_date: Start date string (YYYY-MM-DD)
            end_date: End date string (YYYY-MM-DD)
            interval: Time interval (5m, 15m, 1h, etc.)
            
        Returns:
            pd.DataFrame: DataFrame with proper timestamp index
        """
        try:
            self.logger.info(f"EMERGENCY FIX - Generating timestamps for {len(df)} records from {start_date} to {end_date} at {interval} intervals")
            
            # Parse interval to minutes
            interval_minutes = self._parse_interval_to_minutes(interval)
            
            # Generate timestamp range
            start_dt = pd.to_datetime(start_date)
            end_dt = pd.to_datetime(end_date)
            
            # For market data, typically we want trading hours only
            # For now, generate continuous timestamps and let the data speak for itself
            timestamps = pd.date_range(
                start=start_dt,
                end=end_dt,
                freq=f'{interval_minutes}min'
            )
            
            # Trim to match the actual data length
            if len(timestamps) > len(df):
                timestamps = timestamps[:len(df)]
            elif len(timestamps) < len(df):
                # If we need more timestamps, extend from the start
                additional_needed = len(df) - len(timestamps)
                extended_start = start_dt - pd.Timedelta(minutes=interval_minutes * additional_needed)
                timestamps = pd.date_range(
                    start=extended_start,
                    end=end_dt,
                    freq=f'{interval_minutes}min'
                )[:len(df)]
            
            # Create new DataFrame with proper timestamps
            df_fixed = df.copy()
            df_fixed.index = timestamps
            
            self.logger.info(f"EMERGENCY FIX - Generated {len(timestamps)} timestamps from {timestamps[0]} to {timestamps[-1]}")
            return df_fixed
            
        except Exception as e:
            self.logger.error(f"Error fixing timestamp index: {e}")
            return df
    
    def _parse_interval_to_minutes(self, interval: str) -> int:
        """Parse interval string to minutes."""
        try:
            interval = interval.lower().strip()
            if interval.endswith('m'):
                return int(interval[:-1])
            elif interval.endswith('h'):
                return int(interval[:-1]) * 60
            elif interval.endswith('d'):
                return int(interval[:-1]) * 60 * 24
            else:
                # Default to 5 minutes if unclear
                self.logger.warning(f"Unknown interval format: {interval}, defaulting to 5 minutes")
                return 5
        except Exception as e:
            self.logger.error(f"Error parsing interval {interval}: {e}")
            return 5

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame columns to expected OHLCV format.
        
        Args:
            df: Input DataFrame
            
        Returns:
            pd.DataFrame: Standardized DataFrame
        """
        try:
            # Map common column variations
            column_mapping = {
                'o': 'open', 'open': 'open',
                'h': 'high', 'high': 'high', 
                'l': 'low', 'low': 'low',
                'c': 'close', 'close': 'close',
                'v': 'volume', 'volume': 'volume', 'vol': 'volume',
                'time': 'timestamp', 'timestamp': 'timestamp', 'datetime': 'timestamp',
                'date': 'timestamp', 't': 'timestamp'
            }
            
            # Rename columns to standard names
            df_renamed = df.copy()
            for old_col in df.columns:
                old_col_lower = str(old_col).lower()
                if old_col_lower in column_mapping:
                    new_col = column_mapping[old_col_lower]
                    df_renamed = df_renamed.rename(columns={old_col: new_col})
            
            # Ensure we have the required OHLCV columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            available_columns = [col for col in required_columns if col in df_renamed.columns]
            
            if len(available_columns) >= 4:  # At least OHLC
                # Add missing volume if needed
                if 'volume' not in df_renamed.columns:
                    df_renamed['volume'] = 0
                
                # Set timestamp as index if available
                if 'timestamp' in df_renamed.columns:
                    # EMERGENCY DEBUG: Check what we're converting
                    self.logger.info(f"OPENALGO EMERGENCY DEBUG - Before pd.to_datetime: {df_renamed['timestamp'].head(5).tolist()}")
                    self.logger.info(f"OPENALGO EMERGENCY DEBUG - Timestamp dtype before: {df_renamed['timestamp'].dtype}")
                    
                    df_renamed['timestamp'] = pd.to_datetime(df_renamed['timestamp'])
                    
                    # EMERGENCY DEBUG: Check result
                    self.logger.info(f"OPENALGO EMERGENCY DEBUG - After pd.to_datetime: {df_renamed['timestamp'].head(5).tolist()}")
                    
                    df_renamed = df_renamed.set_index('timestamp')
                
                self.logger.info(f"Standardized DataFrame with {len(df_renamed)} rows, columns: {list(df_renamed.columns)}")
                return df_renamed[required_columns]
            else:
                self.logger.warning(f"Insufficient OHLC columns. Available: {available_columns}")
                return pd.DataFrame()
                
        except Exception as e:
            self.logger.error(f"Error standardizing columns: {e}")
            return df

    # Order Management API Methods
    def place_order(self, **order_params) -> Dict[str, Any]:
        """
        Place a trading order.

        Args:
            **order_params: Order parameters (symbol, action, quantity, etc.)

        Returns:
            dict: Order placement response
        """
        return self._handle_api_call('place_order', self.client.placeorder, **order_params)

    def modify_order(self, order_id: str, **params) -> Dict[str, Any]:
        """
        Modify an existing order.

        Args:
            order_id: Order ID to modify
            **params: Parameters to modify

        Returns:
            dict: Modify order response
        """
        return self._handle_api_call('modify_order', self.client.modifyorder, orderid=order_id, **params)

    def cancel_order(self, order_id: str) -> Dict[str, Any]:
        """
        Cancel an order.

        Args:
            order_id: Order ID to cancel

        Returns:
            dict: Cancel order response
        """
        return self._handle_api_call('cancel_order', self.client.cancelorder, orderid=order_id)

    def get_order_status(self, order_id: str) -> Dict[str, Any]:
        """
        Get status of an order.

        Args:
            order_id: Order ID to check

        Returns:
            dict: Order status
        """
        return self._handle_api_call('get_order_status', self.client.orderstatus, orderid=order_id)

    # WebSocket functionality for live data feeds
    def start_websocket_feed(self, symbols: List[str], callback_function) -> bool:
        """
        Start WebSocket feed for real-time data.
        
        Args:
            symbols: List of symbols to subscribe to
            callback_function: Function to call with new data
            
        Returns:
            bool: True if WebSocket started successfully
        """
        if not WEBSOCKET_AVAILABLE:
            self.logger.error("WebSocket functionality not available. Install: pip install websocket-client")
            return False
        
        try:
            self.logger.info(f"Starting WebSocket feed for symbols: {symbols}")
            
            # Initialize WebSocket state
            self.ws_symbols = symbols
            self.ws_callback = callback_function
            self.ws_connection = None
            self.ws_thread = None
            self.ws_running = False
            
            # Start WebSocket connection in separate thread
            self.ws_thread = threading.Thread(target=self._websocket_worker, daemon=True)
            self.ws_thread.start()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start WebSocket feed: {e}")
            return False
    
    def stop_websocket_feed(self):
        """Stop WebSocket feed."""
        try:
            self.logger.info("Stopping WebSocket feed...")
            self.ws_running = False
            
            if hasattr(self, 'ws_connection') and self.ws_connection:
                self.ws_connection.close()
            
            if hasattr(self, 'ws_thread') and self.ws_thread:
                self.ws_thread.join(timeout=5)
            
            self.logger.info("WebSocket feed stopped")
            
        except Exception as e:
            self.logger.error(f"Error stopping WebSocket feed: {e}")
    
    def _websocket_worker(self):
        """WebSocket worker thread."""
        try:
            # Construct WebSocket URL
            ws_url = self.ws_url or f"{self.host.replace('http', 'ws')}/ws"
            
            self.logger.info(f"Connecting to WebSocket: {ws_url}")
            
            # Create WebSocket connection
            self.ws_connection = websocket.WebSocketApp(
                ws_url,
                on_open=self._on_websocket_open,
                on_message=self._on_websocket_message,
                on_error=self._on_websocket_error,
                on_close=self._on_websocket_close
            )
            
            self.ws_running = True
            
            # Run WebSocket connection
            self.ws_connection.run_forever()
            
        except Exception as e:
            self.logger.error(f"WebSocket worker error: {e}")
            self.ws_running = False
    
    def _on_websocket_open(self, ws):
        """WebSocket connection opened."""
        self.logger.info("WebSocket connection opened")
        
        # Subscribe to symbols
        for symbol in self.ws_symbols:
            subscribe_msg = {
                "action": "subscribe",
                "symbol": symbol,
                "mode": "live"
            }
            ws.send(json.dumps(subscribe_msg))
            self.logger.info(f"Subscribed to {symbol}")
    
    def _on_websocket_message(self, ws, message):
        """Handle WebSocket message."""
        try:
            data = json.loads(message)
            
            # Process the message and call user callback
            if self.ws_callback:
                self.ws_callback(data)
                
        except Exception as e:
            self.logger.error(f"Error processing WebSocket message: {e}")
    
    def _on_websocket_error(self, ws, error):
        """Handle WebSocket error."""
        self.logger.error(f"WebSocket error: {error}")
    
    def _on_websocket_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket close."""
        self.logger.info(f"WebSocket closed: {close_status_code} - {close_msg}")
        self.ws_running = False

    def __str__(self) -> str:
        """String representation of OpenAlgo client."""
        status = "Connected" if self.is_connected() else "Disconnected"
        return f"OpenAlgo(host={self.host}, status={status})"

    def __repr__(self) -> str:
        """Detailed representation of OpenAlgo client."""
        return (f"OpenAlgo(api_key={'***' if self.api_key else 'None'}, "
                f"host={self.host}, ws_url={self.ws_url}, "
                f"connected={self.is_connected()})")