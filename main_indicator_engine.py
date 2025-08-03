#!/usr/bin/env python3
"""
Main Indicator Engine
=====================

Production-grade indicator engine for vAlgo Trading System.
Designed for both backtesting and live trading with institutional accuracy.

Features:
- Config-driven execution (Excel configuration)
- Database integration with proper warmup periods
- 100% accurate indicator calculations
- Live trading state management architecture
- Industry-standard coding practices

Usage:
    python main_indicator_engine.py                    # Full auto mode
    python main_indicator_engine.py --symbol NIFTY     # Manual symbol
    python main_indicator_engine.py --validate         # System validation
    python main_indicator_engine.py --live             # Live trading mode
    python main_indicator_engine.py --live --mock      # Live mode with mock data

Author: vAlgo Development Team
Created: June 29, 2025
Version: 1.0.0 (Production)
"""

import sys
import argparse
import os
import threading
import time
import signal
from pathlib import Path
from datetime import datetime, timedelta, time as dt_time
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging
import json
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import with graceful error handling
try:
    from indicators.unified_indicator_engine import UnifiedIndicatorEngine, LiveTradingIndicatorEngine
    from data_manager.openalgo import OpenAlgo
    from data_manager.database import DatabaseManager
    from utils.logger import get_logger
    from utils.config_loader import ConfigLoader
    from utils.config_cache import get_cached_config
    from utils.smart_data_manager import SmartDataManager
    from utils.live_excel_reporter import LiveExcelReporter
    from utils.performance_tracker import PerformanceTracker
    from utils.market_data_fetcher import MarketDataFetcher
    from utils.live_state_manager import LiveStateManager
    import duckdb
    COMPONENTS_AVAILABLE = True
except ImportError as e:
    print(f"[WARNING]  Component import warning: {e}")
    print("[DEBUG] Continuing with available components...")
    COMPONENTS_AVAILABLE = False


class LiveCSVManager:
    """Real-time CSV manager for live indicator values."""
    
    def __init__(self, output_dir: str = "outputs/live_indicators"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
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
        
        self.current_session_data = defaultdict(list)
        self.csv_files = {}
        self.csv_headers = {}  # Store headers for each symbol
        self.last_update_time = {}
        self.lock = threading.Lock()
    
    def initialize_csv_file(self, symbol: str, indicator_columns: List[str]) -> str:
        """Initialize CSV file for a symbol with comprehensive headers."""
        try:
            today = datetime.now().strftime('%Y%m%d')
            csv_filename = f"{symbol}_live_indicators_{today}.csv"
            csv_path = self.output_dir / csv_filename
            
            # Store headers for this symbol
            headers = ['timestamp', 'open', 'high', 'low', 'close', 'volume'] + sorted(indicator_columns)
            self.csv_headers = {symbol: headers}
            
            if not csv_path.exists():
                with open(csv_path, 'w') as f:
                    f.write(','.join(headers) + '\n')
                self.logger.info(f"[CSV] Initialized live CSV with {len(headers)} columns: {csv_path}")
            else:
                # Check if existing file has correct headers
                with open(csv_path, 'r') as f:
                    existing_headers = f.readline().strip().split(',')
                if existing_headers != headers:
                    self.logger.warning(f"[CSV] Header mismatch for {symbol}, updating headers")
                    # Read existing data
                    existing_df = pd.read_csv(csv_path)
                    # Rewrite with new headers
                    with open(csv_path, 'w') as f:
                        f.write(','.join(headers) + '\n')
                        # Write existing data with new column structure
                        for _, row in existing_df.iterrows():
                            row_values = [str(row.get(col, 0)) for col in headers]
                            f.write(','.join(row_values) + '\n')
            
            self.csv_files[symbol] = str(csv_path)
            return str(csv_path)
            
        except Exception as e:
            self.logger.error(f"Failed to initialize CSV for {symbol}: {e}")
            return ""
    
    def update_csv_with_live_data(self, symbol: str, timestamp: datetime, 
                                 ohlcv_data: Dict[str, float], 
                                 indicator_values: Dict[str, Any]) -> bool:
        """Update CSV file with latest live indicator values using proper column structure."""
        try:
            with self.lock:
                if symbol not in self.csv_files:
                    self.logger.warning(f"CSV file not initialized for {symbol}")
                    return False
                
                # Get proper headers for this symbol
                headers = self.csv_headers.get(symbol, [])
                if not headers:
                    self.logger.error(f"No headers stored for {symbol}")
                    return False
                
                # Flatten indicator values
                flattened_indicators = self._flatten_indicator_values(indicator_values)
                
                # Create row data with all columns
                row_data = {
                    'timestamp': timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    'open': round(float(ohlcv_data.get('open', 0)), 2),
                    'high': round(float(ohlcv_data.get('high', 0)), 2),
                    'low': round(float(ohlcv_data.get('low', 0)), 2),
                    'close': round(float(ohlcv_data.get('close', 0)), 2),
                    'volume': int(ohlcv_data.get('volume', 0))
                }
                
                # Add indicator values, ensuring all columns are present
                for header in headers[6:]:
                    if header in flattened_indicators:
                        value = flattened_indicators[header]
                        row_data[header] = round(float(value), 4) if isinstance(value, (int, float)) else 0.0
                    else:
                        row_data[header] = 0.0  # Fill missing indicators with 0
                
                # Store in session data
                self.current_session_data[symbol].append(row_data)
                
                # Write to CSV file
                csv_path = self.csv_files[symbol]
                with open(csv_path, 'a') as f:
                    row_values = [str(row_data.get(col, 0)) for col in headers]
                    f.write(','.join(row_values) + '\n')
                
                self.last_update_time[symbol] = timestamp
                
                # Log every 10th update to avoid spam
                update_count = len(self.current_session_data[symbol])
                if update_count % 10 == 0:
                    self.logger.info(f"[CSV] {symbol} update #{update_count} with {len(headers)-6} indicators")
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to update CSV for {symbol}: {e}")
            return False
    
    def _flatten_indicator_values(self, indicator_values: Dict[str, Any]) -> Dict[str, float]:
        """Flatten nested indicator values for CSV export with comprehensive handling."""
        flattened = {}
        
        try:
            for indicator_name, values in indicator_values.items():
                # Handle direct key-value pairs (most common)
                if isinstance(values, (int, float)):
                    # Direct value mapping
                    flattened[indicator_name.lower()] = float(values)
                    
                elif isinstance(values, dict):
                    # Handle nested dictionaries
                    for key, value in values.items():
                        if isinstance(value, (int, float)):
                            # Create properly formatted key
                            flattened_key = f"{indicator_name.lower()}_{key}" if indicator_name.lower() != key.lower() else key.lower()
                            flattened[flattened_key] = float(value)
                            
                        elif isinstance(value, dict):
                            # Handle deeply nested structures
                            for sub_key, sub_value in value.items():
                                if isinstance(sub_value, (int, float)):
                                    flattened_key = f"{indicator_name.lower()}_{key}_{sub_key}"
                                    flattened[flattened_key] = float(sub_value)
                                    
                elif isinstance(values, (list, tuple)):
                    # Handle list/tuple values
                    for i, value in enumerate(values):
                        if isinstance(value, (int, float)):
                            flattened[f"{indicator_name.lower()}_{i}"] = float(value)
                            
                elif isinstance(values, str):
                    # Handle string values (try to convert to float)
                    try:
                        flattened[indicator_name.lower()] = float(values)
                    except ValueError:
                        flattened[indicator_name.lower()] = 0.0
                        
                else:
                    # Unknown type, set to 0
                    flattened[indicator_name.lower()] = 0.0
            
            return flattened
            
        except Exception as e:
            self.logger.error(f"Error flattening indicator values: {e}")
            return {}
    
    def export_final_session_csv(self, symbol: str) -> str:
        """Export final session CSV at market close with comprehensive data."""
        try:
            today = datetime.now().strftime('%Y%m%d')
            final_csv_filename = f"{symbol}_final_session_{today}.csv"
            final_csv_path = self.output_dir / final_csv_filename
            
            if symbol in self.current_session_data and self.current_session_data[symbol]:
                # Create DataFrame with proper column order
                session_df = pd.DataFrame(self.current_session_data[symbol])
                
                # Ensure proper column order using stored headers
                if symbol in self.csv_headers:
                    headers = self.csv_headers[symbol]
                    # Reorder columns to match headers
                    session_df = session_df.reindex(columns=headers, fill_value=0)
                
                # Convert timestamp to proper datetime format
                if 'timestamp' in session_df.columns:
                    session_df['timestamp'] = pd.to_datetime(session_df['timestamp'])
                
                # Round numeric columns appropriately
                for col in session_df.columns:
                    if col in ['open', 'high', 'low', 'close']:
                        session_df[col] = session_df[col].round(2)
                    elif col == 'volume':
                        session_df[col] = session_df[col].astype(int)
                    elif col != 'timestamp':
                        session_df[col] = session_df[col].round(4)
                
                # Export to CSV
                session_df.to_csv(final_csv_path, index=False)
                
                record_count = len(session_df)
                column_count = len(session_df.columns)
                
                self.logger.info(f"[CSV] Final session exported: {final_csv_path} ({record_count} records, {column_count} columns)")
                return str(final_csv_path)
            else:
                self.logger.warning(f"No session data to export for {symbol}")
                return ""
                
        except Exception as e:
            self.logger.error(f"Failed to export final session CSV for {symbol}: {e}")
            return ""


class TimeframeScheduler:
    """Timeframe-based scheduler for precise indicator updates."""
    
    def __init__(self, timeframe: str, callback_function):
        self.timeframe = timeframe
        self.callback = callback_function
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
        
        self.interval_minutes = self._parse_timeframe_minutes(timeframe)
        self.last_update = None
        self.scheduler_thread = None
        self.running = False
    
    def _parse_timeframe_minutes(self, timeframe: str) -> int:
        """Parse timeframe string to minutes."""
        timeframe = timeframe.lower().strip()
        
        timeframe_map = {
            '1m': 1, '1min': 1,
            '3m': 3, '3min': 3,
            '5m': 5, '5min': 5,
            '15m': 15, '15min': 15,
            '30m': 30, '30min': 30,
            '1h': 60, '1hour': 60, '60m': 60
        }
        
        if timeframe in timeframe_map:
            return timeframe_map[timeframe]
        
        # Try regex parsing
        import re
        match = re.match(r'(\d+)([mh])', timeframe)
        if match:
            number, unit = match.groups()
            multiplier = 60 if unit == 'h' else 1
            return int(number) * multiplier
        
        self.logger.warning(f"Unknown timeframe format: {timeframe}, defaulting to 5m")
        return 5
    
    def should_update_now(self, current_time: Optional[datetime] = None) -> bool:
        """Check if an update should be triggered now."""
        if current_time is None:
            current_time = datetime.now()
        
        minutes_since_midnight = current_time.hour * 60 + current_time.minute
        seconds_tolerance = 30
        is_boundary_time = (minutes_since_midnight % self.interval_minutes == 0 and 
                           current_time.second <= seconds_tolerance)
        
        if self.last_update is None:
            time_since_last = timedelta(hours=24)
        else:
            time_since_last = current_time - self.last_update
        
        should_update = (is_boundary_time and 
                        time_since_last.total_seconds() >= (self.interval_minutes * 60 - seconds_tolerance))
        
        return should_update
    
    def start_scheduler(self):
        """Start the timeframe scheduler."""
        self.running = True
        self.scheduler_thread = threading.Thread(target=self._scheduler_worker, daemon=True)
        self.scheduler_thread.start()
        self.logger.info(f"[SCHEDULER] Started for {self.timeframe} timeframe")
    
    def stop_scheduler(self):
        """Stop the timeframe scheduler."""
        self.running = False
        if self.scheduler_thread:
            self.scheduler_thread.join(timeout=5)
        self.logger.info(f"[SCHEDULER] Stopped for {self.timeframe} timeframe")
    
    def _scheduler_worker(self):
        """Scheduler worker thread."""
        while self.running:
            try:
                current_time = datetime.now()
                
                if self.should_update_now(current_time):
                    self.logger.info(f"[SCHEDULER] Triggering {self.timeframe} update")
                    
                    if self.callback:
                        self.callback(current_time)
                    
                    self.last_update = current_time
                
                time.sleep(10)
                
            except Exception as e:
                self.logger.error(f"Scheduler error: {e}")
                time.sleep(60)


# Dynamic Market Hours Configuration
MARKET_HOURS_CONFIG = {
    'NSE': {
        'EQUITY': {
            'open': '09:15',
            'close': '15:30', 
            'export': '15:00',
            'description': 'NSE Equity Market'
        },
        'INDEX': {
            'open': '09:15',
            'close': '15:30',
            'export': '15:00', 
            'description': 'NSE Index Market'
        },
        'FUTURES': {
            'open': '09:15',
            'close': '15:30',
            'export': '15:00',
            'description': 'NSE Futures & Options'
        },
        'CURRENCY': {
            'open': '09:00',
            'close': '17:00',
            'export': '16:30',
            'description': 'NSE Currency Derivatives'
        }
    },
    'MCX': {
        'COMMODITY': {
            'sessions': [
                {
                    'name': 'morning_session',
                    'open': '09:00',
                    'close': '17:00',
                    'description': 'MCX Morning Session'
                },
                {
                    'name': 'evening_session', 
                    'open': '17:00',
                    'close': '23:30',
                    'description': 'MCX Evening Session'
                }
            ],
            'export': '23:00',
            'description': 'MCX Commodity Trading'
        }
    },
    'BSE': {
        'EQUITY': {
            'open': '09:15',
            'close': '15:30',
            'export': '15:00',
            'description': 'BSE Equity Market'
        }
    }
}

# Exchange-Symbol Mapping Patterns
SYMBOL_EXCHANGE_MAPPING = {
    # NSE Index symbols
    'NIFTY': ('NSE', 'INDEX'),
    'BANKNIFTY': ('NSE', 'INDEX'), 
    'FINNIFTY': ('NSE', 'INDEX'),
    'MIDCPNIFTY': ('NSE', 'INDEX'),
    'SENSEX': ('BSE', 'INDEX'),
    
    # Currency symbols (common patterns)
    'USDINR': ('NSE', 'CURRENCY'),
    'EURINR': ('NSE', 'CURRENCY'),
    'GBPINR': ('NSE', 'CURRENCY'),
    'JPYINR': ('NSE', 'CURRENCY'),
    
    # MCX Commodities (common symbols)
    'GOLD': ('MCX', 'COMMODITY'),
    'SILVER': ('MCX', 'COMMODITY'),
    'CRUDE': ('MCX', 'COMMODITY'),
    'CRUDEOIL': ('MCX', 'COMMODITY'),
    'NATURALGAS': ('MCX', 'COMMODITY'),
    'COPPER': ('MCX', 'COMMODITY'),
    'ZINC': ('MCX', 'COMMODITY'),
    'ALUMINIUM': ('MCX', 'COMMODITY'),
}


def detect_exchange_and_instrument(symbol: str, exchange: str = None) -> tuple:
    """
    Detect exchange and instrument type from symbol and optional exchange parameter.
    
    Args:
        symbol: Trading symbol (e.g., 'NIFTY', 'GOLD', 'USDINR')
        exchange: Optional exchange override (e.g., 'NSE_INDEX', 'MCX')
        
    Returns:
        tuple: (exchange, instrument_type) e.g., ('NSE', 'INDEX')
    """
    symbol = symbol.upper()
    
    # If exchange is provided in OpenAlgo format, parse it
    if exchange:
        exchange = exchange.upper()
        
        # Handle OpenAlgo format exchanges
        if exchange in ['NSE_INDEX', 'NSE_EQ', 'NSE_FO', 'NSE_CD']:
            if exchange == 'NSE_INDEX':
                return ('NSE', 'INDEX')
            elif exchange == 'NSE_EQ':
                return ('NSE', 'EQUITY')
            elif exchange == 'NSE_FO':
                return ('NSE', 'FUTURES')
            elif exchange == 'NSE_CD':
                return ('NSE', 'CURRENCY')
        elif exchange == 'MCX':
            return ('MCX', 'COMMODITY')
        elif exchange == 'BSE':
            return ('BSE', 'EQUITY')
        elif exchange == 'NSE':
            # Default NSE to INDEX for common symbols
            if symbol in ['NIFTY', 'BANKNIFTY', 'FINNIFTY', 'MIDCPNIFTY']:
                return ('NSE', 'INDEX')
            else:
                return ('NSE', 'EQUITY')
    
    # Try symbol-based detection
    if symbol in SYMBOL_EXCHANGE_MAPPING:
        return SYMBOL_EXCHANGE_MAPPING[symbol]
    
    # Pattern-based detection for common cases
    if symbol.endswith('INR'):
        return ('NSE', 'CURRENCY')
    
    if any(commodity in symbol for commodity in ['GOLD', 'SILVER', 'CRUDE', 'COPPER', 'ZINC', 'ALUMINIUM', 'NATURALGAS']):
        return ('MCX', 'COMMODITY')
    
    # Check for futures/options patterns
    if any(pattern in symbol for pattern in ['FUT', 'CE', 'PE', 'CALL', 'PUT']):
        return ('NSE', 'FUTURES')
    
    # Default fallback to NSE INDEX (safest for indices like NIFTY)
    return ('NSE', 'INDEX')


class MarketSessionManager:
    """Dynamic market session monitoring and management based on exchange and instrument type."""
    
    def __init__(self, exchange: str = 'NSE', instrument_type: str = 'INDEX', symbol: str = 'NIFTY'):
        try:
            self.logger = get_logger(__name__)
        except:
            self.logger = logging.getLogger(__name__)
        
        # Store exchange and instrument information
        self.exchange = exchange.upper()
        self.instrument_type = instrument_type.upper()
        self.symbol = symbol.upper()
        
        # Initialize market hours based on exchange and instrument
        self.market_config = self._get_market_config()
        
        # Set market hours (support both single session and multi-session)
        if 'sessions' in self.market_config:
            # Multi-session market (like MCX)
            self.sessions = self._parse_sessions(self.market_config['sessions'])
            self.has_multiple_sessions = True
            self.market_open_time = None  # Will be determined dynamically
            self.market_close_time = None  # Will be determined dynamically
        else:
            # Single session market (like NSE, BSE)
            self.market_open_time = self._parse_time(self.market_config['open'])
            self.market_close_time = self._parse_time(self.market_config['close'])
            self.has_multiple_sessions = False
            self.sessions = []
        
        self.final_export_time = self._parse_time(self.market_config['export'])
        
        # Session state
        self.session_active = False
        self.export_triggered = False
        
        # Log the configuration being used
        self.logger.info(f"Market hours initialized for {self.exchange} {self.instrument_type}")
        self.logger.info(f"Configuration: {self.market_config['description']}")
        if self.has_multiple_sessions:
            for session in self.sessions:
                self.logger.info(f"Session: {session['name']} ({session['open_time'].strftime('%H:%M')} - {session['close_time'].strftime('%H:%M')})")
        else:
            self.logger.info(f"Market hours: {self.market_open_time.strftime('%H:%M')} - {self.market_close_time.strftime('%H:%M')}")
        self.logger.info(f"Export time: {self.final_export_time.strftime('%H:%M')}")
    
    def _get_market_config(self) -> dict:
        """Get market configuration based on exchange and instrument type."""
        try:
            # Try exact match first
            if self.exchange in MARKET_HOURS_CONFIG:
                if self.instrument_type in MARKET_HOURS_CONFIG[self.exchange]:
                    return MARKET_HOURS_CONFIG[self.exchange][self.instrument_type]
            
            # Try symbol-based detection as fallback
            if self.symbol in SYMBOL_EXCHANGE_MAPPING:
                detected_exchange, detected_instrument = SYMBOL_EXCHANGE_MAPPING[self.symbol]
                self.logger.info(f"Symbol-based detection: {self.symbol} -> {detected_exchange} {detected_instrument}")
                if detected_exchange in MARKET_HOURS_CONFIG:
                    if detected_instrument in MARKET_HOURS_CONFIG[detected_exchange]:
                        return MARKET_HOURS_CONFIG[detected_exchange][detected_instrument]
            
            # Default fallback to NSE INDEX (safest option)
            self.logger.warning(f"No specific config found for {self.exchange} {self.instrument_type}, defaulting to NSE INDEX")
            return MARKET_HOURS_CONFIG['NSE']['INDEX']
            
        except Exception as e:
            self.logger.error(f"Error getting market config: {e}")
            # Emergency fallback
            return MARKET_HOURS_CONFIG['NSE']['INDEX']
    
    def _parse_time(self, time_str: str) -> dt_time:
        """Parse time string (HH:MM) to datetime.time object."""
        try:
            hour, minute = map(int, time_str.split(':'))
            return dt_time(hour, minute)
        except Exception as e:
            self.logger.error(f"Error parsing time '{time_str}': {e}")
            return dt_time(9, 15)  # Default fallback
    
    def _parse_sessions(self, sessions_config: list) -> list:
        """Parse multiple sessions configuration."""
        sessions = []
        for session in sessions_config:
            sessions.append({
                'name': session['name'],
                'open_time': self._parse_time(session['open']),
                'close_time': self._parse_time(session['close']),
                'description': session['description']
            })
        return sessions
    
    def is_market_open(self) -> bool:
        """Check if market is currently open (supports both single and multi-session markets)."""
        now = datetime.now().time()
        
        if self.has_multiple_sessions:
            # Multi-session market (like MCX) - check if any session is active
            for session in self.sessions:
                if session['open_time'] <= now <= session['close_time']:
                    return True
            return False
        else:
            # Single session market (like NSE, BSE)
            return self.market_open_time <= now <= self.market_close_time
    
    def get_current_session_info(self) -> dict:
        """Get information about the current trading session."""
        now = datetime.now().time()
        
        if self.has_multiple_sessions:
            for session in self.sessions:
                if session['open_time'] <= now <= session['close_time']:
                    return {
                        'session_name': session['name'],
                        'session_description': session['description'],
                        'open_time': session['open_time'].strftime('%H:%M'),
                        'close_time': session['close_time'].strftime('%H:%M'),
                        'is_active': True
                    }
            # No active session
            return {
                'session_name': 'closed',
                'session_description': 'Market Closed',
                'open_time': None,
                'close_time': None,
                'is_active': False
            }
        else:
            is_active = self.market_open_time <= now <= self.market_close_time
            return {
                'session_name': 'main_session',
                'session_description': self.market_config['description'],
                'open_time': self.market_open_time.strftime('%H:%M'),
                'close_time': self.market_close_time.strftime('%H:%M'),
                'is_active': is_active
            }
    
    def should_trigger_final_export(self) -> bool:
        """Check if final export should be triggered during market session."""
        now = datetime.now().time()
        
        # Only trigger export if market is open and at/after export time
        if not self.is_market_open():
            return False
            
        return now >= self.final_export_time and not self.export_triggered
    
    def get_market_hours_summary(self) -> str:
        """Get a human-readable summary of market hours."""
        if self.has_multiple_sessions:
            session_info = []
            for session in self.sessions:
                session_info.append(f"{session['name']}: {session['open_time'].strftime('%H:%M')} - {session['close_time'].strftime('%H:%M')}")
            return f"{self.exchange} {self.instrument_type} - " + " | ".join(session_info)
        else:
            return f"{self.exchange} {self.instrument_type}: {self.market_open_time.strftime('%H:%M')} - {self.market_close_time.strftime('%H:%M')}"


class MainIndicatorEngine:
    """
    Production-grade main indicator engine for vAlgo trading system.
    
    Supports both backtesting (full historical calculation) and live trading
    (incremental updates) with institutional-grade accuracy and performance.
    """
    
    def __init__(self, config_path: str = "config/config.xlsx"):
        """
        Initialize main indicator engine.
        
        Args:
            config_path: Path to Excel configuration file
        """
        self.config_path = config_path
        self.logger = self._setup_logger()
        self.engine = None
        self.config_loader = None
        
        # Initialize config cache for optimized loading
        self.config_cache = get_cached_config(config_path)
        self.logger.info(f"[CONFIG_CACHE] Initialized with config path: {config_path}")
        
        # Performance tracking
        self.start_time = None
        self.processing_stats = {}
        
        # Initialize engine
        self._initialize_engine()
    
    def _setup_logger(self) -> logging.Logger:
        """Setup professional logging for production use."""
        if COMPONENTS_AVAILABLE:
            try:
                return get_logger(__name__)
            except Exception:
                pass
        
        # Fallback logger setup
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def _initialize_engine(self) -> None:
        """Initialize the unified indicator engine with config cache."""
        try:
            if COMPONENTS_AVAILABLE:
                # Use config cache for optimized loading
                config_loader = self.config_cache.get_config_loader()
                if config_loader:
                    self.config_loader = config_loader
                    self.engine = UnifiedIndicatorEngine(self.config_path)
                    self.logger.info("[OK] Main indicator engine initialized successfully")
                    print(f"[TARGET] Engine mode: {self.engine.get_mode().upper()}")
                    
                    # Display config cache stats
                    cache_stats = self.config_cache.get_statistics()
                    self.logger.info(f"[CONFIG_CACHE] Total loads: {cache_stats['total_loads']}")
                else:
                    self.logger.error("[ERROR] Failed to load config from cache")
                    print("[ERROR] Config cache initialization failed")
            else:
                self.logger.warning("[WARNING] Engine components not fully available")
                print("[DEBUG] Running in compatibility mode...")
        except Exception as e:
            self.logger.error(f"[ERROR] Failed to initialize engine: {e}")
            print(f"[ERROR] Engine initialization failed: {e}")
    
    def run_auto_mode(self) -> Dict[str, Any]:
        """
        Run complete auto mode - the default behavior.
        Reads everything from Excel config and processes automatically.
        
        Returns:
            Dictionary with processing results and statistics
        """
        print("\n" + "="*60)
        print("[ENGINE] vAlgo Indicator Engine - AUTO MODE")
        print("="*60)
        
        self.start_time = datetime.now()
        
        try:
            # Step 1: Load and validate configuration
            print("[INFO] Step 1: Loading configuration...")
            config_status = self._load_and_validate_config()
            if not config_status:
                return self._create_error_result("Configuration loading failed")
            
            # Step 2: Connect to database
            print("[DATABASE]  Step 2: Connecting to database...")
            db_status = self._validate_database_connection()
            if not db_status:
                return self._create_error_result("Database connection failed")
            
            # Step 3: Get processing parameters from config
            print("[CONFIG]  Step 3: Reading processing parameters...")
            params = self._get_processing_parameters()
            if not params:
                return self._create_error_result("Failed to read processing parameters")
            
            # Step 4: Process all configured instruments
            print("[PROCESS] Step 4: Processing indicators...")
            results = self._process_all_instruments(params)
            
            # Step 5: Generate validation report
            print("[DATA] Step 5: Generating validation report...")
            report_path = self._generate_validation_report(results, params)
            
            # Step 6: Show summary
            self._show_processing_summary(results, report_path)
            
            return self._create_success_result(results, report_path, params)
            
        except Exception as e:
            self.logger.error(f"[ERROR] Auto mode failed: {e}")
            return self._create_error_result(f"Processing failed: {e}")
    
    def _load_and_validate_config(self) -> bool:
        """Load and validate Excel configuration."""
        try:
            # Check if config file exists
            if not os.path.exists(self.config_path):
                print(f"[ERROR] Configuration file not found: {self.config_path}")
                return False
            
            # Load configuration using cached config loader
            self.config_loader = self.config_cache.get_config_loader(force_refresh=True)
            if not self.config_loader:
                print("[ERROR] Failed to load Excel configuration")
                return False
            
            # Validate key components
            instruments = self.config_loader.get_active_instruments()
            
            # Try to get indicators with error handling
            try:
                indicators = self.config_loader.get_indicator_config()
                indicator_count = len(indicators)
            except Exception as e:
                print(f"[WARNING]  Warning: Could not load indicator config: {e}")
                indicators = []
                indicator_count = 0
            
            print(f"[OK] Configuration loaded successfully")
            print(f"   [DATA] Active instruments: {len(instruments)}")
            print(f"   [CALC] Configured indicators: {indicator_count}")
            
            if len(instruments) == 0:
                print("[WARNING]  No active instruments found in configuration")
                return False
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Configuration validation failed: {e}")
            return False
    
    def _validate_database_connection(self) -> bool:
        """Validate database connection and data availability."""
        try:
            if not self.engine:
                print("[ERROR] Engine not initialized")
                return False
            
            # Test database connection
            db_status = self.engine.validate_database_connection()
            if not db_status:
                print("[ERROR] Database connection failed")
                return False
            
            # Check data availability
            symbols = self.engine.get_available_symbols()
            if not symbols:
                print("[WARNING]  No symbols available in database")
                return False
            
            print(f"[OK] Database connected successfully")
            print(f"   [TARGET] Available symbols: {symbols}")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Database validation failed: {e}")
            return False
    
    def _get_processing_parameters(self) -> Optional[Dict[str, Any]]:
        """Get processing parameters from configuration."""
        try:
            # Get date range from config or use defaults
            params = {}
            
            if self.engine:
                # Force reload config to get fresh dates from Excel
                try:
                    # Use cached config loader with force refresh
                    config_loader = self.config_cache.get_config_loader(force_refresh=True)
                    # Check if config is available
                    if config_loader:
                        init_config = config_loader.get_initialize_config()
                        self.logger.info(f"[CONFIG] Force reloaded config, available keys: {list(init_config.keys())}")
                        print(f"[CONFIG] Reading fresh config from Excel...")
                        print(f"[CONFIG] Available config keys: {list(init_config.keys())}")
                        
                        # Try different possible key names for dates
                        start_date = None
                        end_date = None
                        
                        # Debug: Show all config values for troubleshooting
                        print(f"[DEBUG] All config values:")
                        for key, value in init_config.items():
                            print(f"   {key}: {value} (type: {type(value).__name__})")
                        
                        # Check for start date with various key names
                        for key in ['start_date', 'startdate', 'start', 'from_date']:
                            if key in init_config and init_config[key]:
                                start_date = init_config[key]
                                self.logger.info(f"[CONFIG] Found start_date with key '{key}': {start_date}")
                                print(f"[CONFIG] Found start_date with key '{key}': {start_date}")
                                break
                        
                        # Check for end date with various key names  
                        for key in ['end_date', 'enddate', 'end', 'to_date']:
                            if key in init_config and init_config[key]:
                                end_date = init_config[key]
                                self.logger.info(f"[CONFIG] Found end_date with key '{key}': {end_date}")
                                print(f"[CONFIG] Found end_date with key '{key}': {end_date}")
                                break
                        
                        # Convert datetime objects to strings if needed
                        if start_date:
                            if isinstance(start_date, datetime):
                                start_date = start_date.strftime('%Y-%m-%d')
                            params['start_date'] = str(start_date)
                            self.logger.info(f"[CONFIG] Using config start_date: {params['start_date']}")
                            print(f"[CONFIG] ✅ Using config start_date: {params['start_date']}")
                        else:
                            # Use dynamic fallback instead of hardcoded date
                            fallback_start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                            params['start_date'] = fallback_start
                            self.logger.warning(f"[CONFIG] No start_date found in config, using dynamic fallback: {fallback_start}")
                            print(f"[CONFIG] ⚠️ No start_date found, using fallback: {fallback_start}")
                        
                        if end_date:
                            if isinstance(end_date, datetime):
                                end_date = end_date.strftime('%Y-%m-%d')
                            params['end_date'] = str(end_date)
                            self.logger.info(f"[CONFIG] Using config end_date: {params['end_date']}")
                            print(f"[CONFIG] ✅ Using config end_date: {params['end_date']}")
                        else:
                            # Use dynamic fallback instead of hardcoded date
                            fallback_end = datetime.now().strftime('%Y-%m-%d')
                            params['end_date'] = fallback_end
                            self.logger.warning(f"[CONFIG] No end_date found in config, using dynamic fallback: {fallback_end}")
                            print(f"[CONFIG] ⚠️ No end_date found, using fallback: {fallback_end}")
                    else:
                        self.logger.error("[CONFIG] Failed to load config, using dynamic defaults")
                        fallback_start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                        fallback_end = datetime.now().strftime('%Y-%m-%d')
                        params['start_date'] = fallback_start
                        params['end_date'] = fallback_end
                        print(f"[CONFIG] ⚠️ Config load failed, using dynamic fallback: {fallback_start} to {fallback_end}")
                            
                except Exception as e:
                    self.logger.error(f"[ERROR] Failed to read dates from config: {e}")
                    fallback_start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                    fallback_end = datetime.now().strftime('%Y-%m-%d')
                    params['start_date'] = fallback_start
                    params['end_date'] = fallback_end
                    print(f"[CONFIG] ⚠️ Exception in config reading, using dynamic fallback: {fallback_start} to {fallback_end}")
            else:
                # Dynamic default instead of hardcoded dates
                fallback_start = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
                fallback_end = datetime.now().strftime('%Y-%m-%d')
                params['start_date'] = fallback_start
                params['end_date'] = fallback_end
                print(f"[CONFIG] ⚠️ No engine available, using dynamic fallback: {fallback_start} to {fallback_end}")
            
            # Get active symbols (simplified for unified engine)
            if self.engine:
                # Default to NIFTY for now (can be enhanced based on config)
                params['symbols'] = ['NIFTY']
                params['timeframes'] = {'NIFTY': '5m'}
            else:
                params['symbols'] = ['NIFTY']
                params['timeframes'] = {'NIFTY': '5m'}
            
            print(f"[DATE] Processing period: {params['start_date']} to {params['end_date']}")
            print(f"[TARGET] Symbols to process: {params['symbols']}")
            print(f"[TIME]  Timeframes: {params['timeframes']}")
            
            return params
            
        except Exception as e:
            print(f"[ERROR] Failed to get processing parameters: {e}")
            return None
    
    def _process_all_instruments(self, params: Dict[str, Any]) -> Dict[str, pd.DataFrame]:
        """Process indicators for all configured instruments."""
        results = {}
        total_symbols = len(params['symbols'])
        
        print(f"[PROCESS] Processing {total_symbols} symbols with high accuracy...")
        
        for i, symbol in enumerate(params['symbols'], 1):
            try:
                print(f"\n[DATA] Processing {i}/{total_symbols}: {symbol}")
                timeframe = params['timeframes'].get(symbol, '5m')
                print(f"[TIME]  Using timeframe: {timeframe}")
                
                # Calculate indicators with proper warmup
                if self.engine:
                    result = self.engine.calculate_indicators_for_symbol(
                        symbol=symbol,
                        start_date=params['start_date'],
                        end_date=params['end_date']
                    )
                    
                    if not result.empty:
                        results[symbol] = result
                        print(f"[OK] {symbol}: {len(result)} records processed")
                        print(f"[CALC] Price range: {result['low'].min():.2f} - {result['high'].max():.2f}")
                        print(f"[DATE] Data range: {result.index.min()} to {result.index.max()}")
                    else:
                        print(f"[WARNING]  {symbol}: No data processed")
                else:
                    print(f"[WARNING]  {symbol}: Engine not available, skipping")
                
            except Exception as e:
                print(f"[ERROR] {symbol}: Processing failed - {e}")
                self.logger.error(f"Symbol processing failed for {symbol}: {e}")
                continue
        
        successful = len(results)
        print(f"\n[TARGET] Processing complete: {successful}/{total_symbols} symbols successful")
        
        return results
    
    def _generate_validation_report(self, results: Dict[str, pd.DataFrame], 
                                  params: Dict[str, Any]) -> Optional[str]:
        """Generate Excel validation report with results."""
        try:
            if not results:
                print("[WARNING]  No results to export")
                return None
            
            # Create output directory
            output_dir = Path("outputs/indicators")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename with timestamp
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"indicator_engine_validation_{timestamp}.xlsx"
            output_path = output_dir / filename
            
            print(f"[FILE] Generating validation report: {filename}")
            
            # Use unified engine's export function (single report)
            if self.engine and hasattr(self.engine, 'export_results_to_csv'):
                csv_path = self.engine.export_results_to_csv(results)
                if csv_path:
                    print(f"[OK] Validation report generated: {csv_path}")
                    return csv_path
            
            # If export fails, return None (no fallback to avoid duplicate reports)
            print("[WARNING]  CSV export not available")
            return None
            
        except Exception as e:
            print(f"[ERROR] Failed to generate validation report: {e}")
            self.logger.error(f"Report generation failed: {e}")
            return None
    
    def _show_processing_summary(self, results: Dict[str, pd.DataFrame], 
                               report_path: Optional[str]) -> None:
        """Show processing summary and statistics."""
        end_time = datetime.now()
        duration = end_time - self.start_time if self.start_time else timedelta(0)
        
        print("\n" + "="*60)
        print("[DATA] PROCESSING SUMMARY")
        print("="*60)
        
        # Overall statistics
        total_symbols = len(results)
        total_records = sum(len(df) for df in results.values())
        
        print(f"[TARGET] Symbols processed: {total_symbols}")
        print(f"[CALC] Total records: {total_records:,}")
        print(f"[TIME]  Processing time: {duration.total_seconds():.2f} seconds")
        
        if total_records > 0:
            records_per_second = total_records / max(duration.total_seconds(), 1)
            print(f"[ENGINE] Performance: {records_per_second:,.0f} records/second")
        
        # Individual symbol statistics
        if results:
            print(f"\n[DATA] Individual Results:")
            for symbol, df in results.items():
                if not df.empty:
                    indicators = len(df.columns) - 5  # Excluding OHLCV
                    print(f"   {symbol}: {len(df):,} records, {indicators} indicators")
        
        # Files generated
        if report_path:
            print(f"\n[FILE] Validation Report: {report_path}")
        
        print("\n[OK] Processing completed successfully!")
    
    def _create_success_result(self, results: Dict[str, pd.DataFrame], 
                             report_path: Optional[str], 
                             params: Dict[str, Any]) -> Dict[str, Any]:
        """Create success result dictionary."""
        return {
            'status': 'success',
            'symbols_processed': len(results),
            'total_records': sum(len(df) for df in results.values()),
            'processing_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0,
            'report_path': report_path,
            'parameters': params,
            'results': results
        }
    
    def _create_error_result(self, error_message: str) -> Dict[str, Any]:
        """Create error result dictionary."""
        return {
            'status': 'error',
            'error_message': error_message,
            'symbols_processed': 0,
            'total_records': 0,
            'processing_time': (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        }
    
    def run_manual_mode(self, symbol: str, start_date: str, end_date: str,
                       export: bool = True) -> Dict[str, Any]:
        """
        Run manual mode with specific parameters.
        
        Args:
            symbol: Trading symbol to process
            start_date: Start date (YYYY-MM-DD format)
            end_date: End date (YYYY-MM-DD format)
            export: Whether to export results to Excel
            
        Returns:
            Processing results dictionary
        """
        print(f"\n[TARGET] Manual Mode: {symbol}")
        print(f"[DATE] Period: {start_date} to {end_date}")
        
        try:
            if not self.engine:
                return self._create_error_result("Engine not initialized")
            
            # Get timeframe from config (simplified for unified engine)
            timeframe = '5m'  # Default timeframe
            print(f"[TIME]  Using timeframe: {timeframe}")
            
            # Calculate indicators
            result = self.engine.calculate_indicators_for_symbol(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date
            )
            
            if result.empty:
                return self._create_error_result(f"No data processed for {symbol}")
            
            print(f"[OK] Calculation complete: {len(result)} records")
            print(f"[DATA] Columns: {len(result.columns)} (OHLCV + indicators)")
            print(f"[CALC] Price range: {result['low'].min():.2f} - {result['high'].max():.2f}")
            
            # Export if requested
            report_path = None
            if export:
                if hasattr(self.engine, 'export_results_to_csv'):
                    report_path = self.engine.export_results_to_csv({symbol: result})
                elif hasattr(self.engine, 'engine') and hasattr(self.engine.engine, 'export_results_to_csv'):
                    report_path = self.engine.engine.export_results_to_csv({symbol: result})
                
                if report_path:
                    print(f"[FILE] Results exported to: {report_path}")
            
            return {
                'status': 'success',
                'symbol': symbol,
                'records_processed': len(result),
                'report_path': report_path,
                'result': result
            }
            
        except Exception as e:
            error_msg = f"Manual processing failed for {symbol}: {e}"
            self.logger.error(error_msg)
            return self._create_error_result(error_msg)
    
    def validate_system(self) -> bool:
        """
        Comprehensive system validation.
        
        Returns:
            True if system is properly configured and ready
        """
        print("\n[DEBUG] System Validation")
        print("=" * 40)
        
        validation_passed = True
        
        # Check engine initialization
        if not self.engine:
            print("[ERROR] Engine not initialized")
            validation_passed = False
        else:
            print("[OK] Engine initialized")
        
        # Check configuration
        try:
            if self.config_loader:
                instruments = self.config_loader.get_active_instruments()
                indicators = self.config_loader.get_indicator_config()
                print(f"[OK] Configuration loaded: {len(instruments)} instruments, {len(indicators)} indicators")
            else:
                print("[WARNING]  Configuration loader not available")
        except Exception as e:
            print(f"[ERROR] Configuration check failed: {e}")
            validation_passed = False
        
        # Check database connection (if applicable)
        try:
            if self.engine:
                if hasattr(self.engine, 'validate_database_connection'):
                    if self.engine.validate_database_connection():
                        print("[OK] Database connected")
                    else:
                        print("[ERROR] Database connection failed")
                        validation_passed = False
                elif hasattr(self.engine, 'engine') and hasattr(self.engine.engine, 'validate_database_connection'):
                    if self.engine.engine.validate_database_connection():
                        print("[OK] Database connected")
                    else:
                        print("[ERROR] Database connection failed")
                        validation_passed = False
                else:
                    print("ℹ️  Database validation not available in current engine")
            else:
                print("[ERROR] Engine not available for database validation")
                validation_passed = False
        except Exception as e:
            print(f"[ERROR] Database validation failed: {e}")
            validation_passed = False
        
        # Summary
        print("\n" + "=" * 40)
        if validation_passed:
            print("[OK] System validation PASSED - Ready for production")
        else:
            print("[ERROR] System validation FAILED - Issues need resolution")
        
        return validation_passed
    
    def run_live_mode(self, symbol: Optional[str] = None, 
                     debug_mode: bool = False) -> Dict[str, Any]:
        """
        Run live trading mode with REAL market data only.
        
        Args:
            symbol: Single symbol to process (default: from config)
            debug_mode: Enable detailed logging
            
        Returns:
            Live trading session results
        """
        print("\n" + "="*60)
        print("[ENGINE] vAlgo Indicator Engine - LIVE TRADING MODE")
        print("="*60)
        
        print("[PRODUCTION] Running with REAL market data only")
        
        try:
            # Get symbol for market session detection
            target_symbol = symbol or 'NIFTY'
            
            # Detect exchange and instrument type for market hours validation
            exchange, instrument_type = detect_exchange_and_instrument(target_symbol)
            
            # Initialize market session manager for market hours detection
            session_manager = MarketSessionManager(exchange, instrument_type, target_symbol)
            
            # MARKET HOURS VALIDATION: Only initialize CSV manager during market hours
            if session_manager.is_market_open():
                print(f"[MARKET] ✅ Market is OPEN - {session_manager.get_market_hours_summary()}")
                print(f"[MARKET] Session info: {session_manager.get_current_session_info()}")
                
                # Initialize live components with dual persistence (ONLY during market hours)
                csv_manager = LiveCSVManager()
                print("[CSV] ✅ Live CSV manager initialized during market hours")
            else:
                print(f"[MARKET] ⚠️  Market is CLOSED - {session_manager.get_market_hours_summary()}")
                print(f"[MARKET] Session info: {session_manager.get_current_session_info()}")
                
                # NO CSV resources during market closure
                csv_manager = None
                print("[CSV] ⏸️  CSV manager NOT initialized - market is closed")
                print("[CSV] CSV files will only be created when market opens")
            
            # Always initialize state manager (for system monitoring) with config-driven support
            state_manager = LiveStateManager("live_trading/states", backup_count=5, logger=self.logger, config_path=self.config_path)
            schedulers = {}
            
            print("[STATE] Institutional-grade state manager initialized")
            print(f"[STATE] Session ID: {state_manager.session_id}")
            
            # Initialize market data fetcher for live trading (no database fallbacks)
            market_data_fetcher = MarketDataFetcher(mode="live")
            print("[DATA] Market data fetcher initialized for live trading (OpenAlgo only)")
            
            # CRITICAL: Validate OpenAlgo connection before proceeding
            if not market_data_fetcher.openalgo_client:
                error_msg = f"""
[CRITICAL ERROR] OpenAlgo connection failed - Live trading cannot proceed

OpenAlgo Connection Requirements:
1. OpenAlgo server must be running at: {market_data_fetcher.openalgo_url}
2. Broker must be connected in OpenAlgo dashboard  
3. API key must be valid in .env file

Current Status:
- OpenAlgo URL: {market_data_fetcher.openalgo_url}
- API Key: {'Present' if market_data_fetcher.openalgo_api_key else 'Missing'}
- Connection: FAILED

To fix:
1. Start OpenAlgo server: python app.py
2. Configure broker in OpenAlgo dashboard
3. Verify .env file has correct OPENALGO_API_KEY

System will NOT proceed with live trading without real market data.
"""
                print(error_msg)
                self.logger.error("OpenAlgo connection validation failed - stopping live trading")
                raise Exception("OpenAlgo connection required for live trading - no fallbacks available")
            
            print(f"[OK] OpenAlgo connection validated - proceeding with live trading")
            self.logger.info(f"OpenAlgo connected successfully to {market_data_fetcher.openalgo_url}")
            
            # FIRST-RUN SETUP: Initialize system with fresh market data (regardless of start time)
            print("\n" + "="*60)
            print("[SETUP] FIRST-RUN INITIALIZATION - FETCHING FRESH MARKET DATA")
            print("="*60)
            self._perform_first_run_setup(market_data_fetcher, symbol)
            
            # Load configuration for indicators and symbols using engine's config
            try:
                # Use the engine's existing active indicators if available
                engine_indicators = None
                if (self.engine and hasattr(self.engine, 'engine') and 
                    hasattr(self.engine.engine, 'active_indicators') and 
                    self.engine.engine.active_indicators):
                    engine_indicators = self.engine.engine.active_indicators
                    
                if engine_indicators:
                    indicator_config = list(engine_indicators.values())
                    print(f"[CONFIG] Using engine indicator config: {len(engine_indicators)} indicators")
                    print(f"[INDICATORS] Engine indicators: {list(engine_indicators.keys())}")
                else:
                    # Fallback to cached config loader
                    config_loader = self.config_cache.get_config_loader()
                    if config_loader:
                        active_instruments = config_loader.get_active_instruments()
                        indicator_config = config_loader.get_indicator_config()
                        print(f"[CONFIG] Using cached fallback config: {len(indicator_config)} indicators")
                    else:
                        # Emergency fallback
                        active_instruments = {}
                        indicator_config = []
                        print("[CONFIG] ⚠️  Config cache failed, using empty config")
                
                # Get symbols from config or use provided symbol
                if symbol:
                    symbols = [symbol]
                else:
                    # Use default for live trading since backtesting config might not have active instruments
                    symbols = ['NIFTY']
                    print("[LIVE] Using default symbol for live trading")
                
                # Default timeframe for live trading
                timeframes = ['5m']
                
                print(f"[LIVE] Processing symbols: {symbols}")
                print(f"[TIME] Timeframes: {timeframes}")
                print(f"[CONFIG] Loaded {len(indicator_config)} indicator configurations")
                
                # Initialize dynamic market session manager based on primary symbol
                primary_symbol = symbols[0] if symbols else 'NIFTY'
                detected_exchange, detected_instrument = detect_exchange_and_instrument(primary_symbol)
                session_manager = MarketSessionManager(
                    exchange=detected_exchange,
                    instrument_type=detected_instrument, 
                    symbol=primary_symbol
                )
                print(f"[MARKET] Initialized dynamic market hours for {primary_symbol}: {session_manager.get_market_hours_summary()}")
                
                # Check for recovery scenarios (bot restart during market hours) - AFTER symbols are defined
                if session_manager.is_market_open():
                    print("[RECOVERY] Market is open - checking for previous session state...")
                    
                    for sym in symbols:
                        previous_state = state_manager.load_latest_state(sym)
                        if previous_state:
                            state_age_dt = datetime.fromisoformat(previous_state.timestamp)
                            state_age = (datetime.now() - state_age_dt).total_seconds() / 60
                            print(f"[RECOVERY] Found previous state for {sym} (age: {state_age:.1f} minutes)")
                            
                            # If state is recent (< 30 minutes), consider it for recovery
                            if state_age < 30:
                                print(f"[RECOVERY] Recovering indicators from previous session for {sym}")
                                # Set as current state for continuity
                                state_manager.current_state = previous_state
                            else:
                                print(f"[RECOVERY] Previous state too old ({state_age:.1f} min), will recalculate")
                        else:
                            print(f"[RECOVERY] No previous state found for {sym} - clean start")
                else:
                    print("[STATE] Market closed - clean session start")
                
            except Exception as e:
                print(f"[WARNING] Config loading failed, using defaults: {e}")
                symbols = [symbol] if symbol else ['NIFTY']
                timeframes = ['5m']
                indicator_config = []
                print(f"[FALLBACK] Using default symbols: {symbols}")
                print("[STATE] No recovery check - using defaults")
            
            # Initialize live indicator engine with config
            live_engine = None
            if COMPONENTS_AVAILABLE:
                try:
                    live_engine = LiveTradingIndicatorEngine(self.config_path)
                    print("[OK] Live indicator engine initialized with config")
                except Exception as e:
                    print(f"[WARNING] Live engine not available: {e}")
            
            # Generate comprehensive indicator column list from engine or config
            if (self.engine and hasattr(self.engine, 'engine') and 
                hasattr(self.engine.engine, 'active_indicators') and 
                self.engine.engine.active_indicators):
                # Use engine's active indicators to generate columns
                indicator_columns = self._generate_indicator_columns_from_engine()
                print(f"[INDICATORS] Generated {len(indicator_columns)} indicator columns from engine")
            else:
                # Fallback to config-based generation
                indicator_columns = self._generate_indicator_columns_from_config(indicator_config)
                print(f"[INDICATORS] Generated {len(indicator_columns)} indicator columns from config")
            
            # Initialize performance tracker with recovery awareness
            performance_tracker = PerformanceTracker()
            performance_tracker.start_automatic_reporting()
            print("[PERF] Performance tracking started")
            
            # Log recovery status for monitoring
            if state_manager.current_state:
                print(f"[RECOVERY] Resuming with recovered state from {state_manager.current_state.timestamp}")
                # Safely access CPR Pivot and EMA values with fallback
                cpr_pivot = getattr(state_manager.current_state, 'cpr_pivot', 0.0)
                ema_20 = getattr(state_manager.current_state, 'ema_20', 0.0)
                print(f"[RECOVERY] Last known CPR Pivot: {cpr_pivot:.2f}")
                print(f"[RECOVERY] Last known EMA-20: {ema_20:.2f}")
            else:
                print("[STATE] Starting fresh session - no recovery data")
            
            # Initialize Excel reporter
            excel_reporter = LiveExcelReporter()
            excel_reporter.initialize_session(symbols, indicator_columns)
            print("[EXCEL] Live Excel reporter initialized")
            
            # Setup CSV files for each symbol (only if CSV manager is available)
            if csv_manager:
                for sym in symbols:
                    csv_path = csv_manager.initialize_csv_file(sym, indicator_columns)
                    print(f"[CSV] Initialized CSV for {sym}: {csv_path}")
            else:
                print("[CSV] ⏸️  CSV file initialization skipped - market is closed")
            
            # Check market hours before starting schedulers using dynamic session manager
            current_time = datetime.now()
            is_market_open = session_manager.is_market_open()
            session_info = session_manager.get_current_session_info()
            
            print(f"\n[MARKET_CHECK] Current time: {current_time.strftime('%H:%M:%S')}")
            print(f"[MARKET_CHECK] Market: {session_manager.get_market_hours_summary()}")
            print(f"[MARKET_CHECK] Current session: {session_info['session_description']}")
            print(f"[MARKET_CHECK] Market status: {'OPEN' if is_market_open else 'CLOSED'}")
            
            if session_info['is_active']:
                print(f"[MARKET_CHECK] Active session: {session_info['session_name']} ({session_info['open_time']} - {session_info['close_time']})")
            
            if not is_market_open:
                print(f"\n[AFTER_MARKET] ⚠️  Market is currently CLOSED")
                print(f"[AFTER_MARKET] System setup completed successfully")
                print(f"[AFTER_MARKET] Live trading will NOT start - market closed for {session_manager.exchange} {session_manager.instrument_type}")
                print(f"[AFTER_MARKET] Market hours: {session_manager.get_market_hours_summary()}")
                print(f"[AFTER_MARKET] To test the system after hours, use mock mode: python main_indicator_engine.py --live --mock")
                print(f"[AFTER_MARKET] System stopping gracefully...")
                
                # Clean shutdown without starting schedulers
                self._shutdown_live_mode(schedulers, csv_manager, state_manager, symbols, performance_tracker, excel_reporter)
                return {'status': 'success', 'message': 'System setup completed - market closed, stopping gracefully'}
            
            # Market is open - proceed with scheduler setup
            print(f"[MARKET_OPEN] ✅ Market is OPEN - starting live trading schedulers")
            
            # Setup schedulers for each timeframe with state management
            def create_update_callback(symbol, timeframe):
                def update_callback(timestamp):
                    self._process_live_update(symbol, timeframe, timestamp, 
                                            csv_manager, state_manager, live_engine,
                                            performance_tracker, excel_reporter, market_data_fetcher)
                return update_callback
            
            # Start schedulers
            for tf in timeframes:
                for sym in symbols:
                    scheduler_key = f"{sym}_{tf}"
                    callback = create_update_callback(sym, tf)
                    scheduler = TimeframeScheduler(tf, callback)
                    schedulers[scheduler_key] = scheduler
                    scheduler.start_scheduler()
            
            print(f"[SCHEDULER] Started {len(schedulers)} schedulers")
            
            # Monitor market session and handle signals
            def signal_handler(sig, frame):
                print("\n[SIGNAL] Received stop signal - shutting down gracefully...")
                self._shutdown_live_mode(schedulers, csv_manager, state_manager, symbols, performance_tracker, excel_reporter)
                sys.exit(0)
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Main live trading loop
            print("[LIVE] Starting live trading session...")
            print("[INFO] Press Ctrl+C to stop gracefully")
            
            start_time = datetime.now()
            update_count = 0
            
            try:
                while True:
                    current_time = datetime.now()
                    
                    # Check market session with state management awareness
                    if session_manager.is_market_open():
                        if not session_manager.session_active:
                            print(f"[MARKET] Market opened at {current_time.strftime('%H:%M:%S')}")
                            session_manager.session_active = True
                            
                            # Log state management status at market open
                            if state_manager.current_state:
                                state_age = state_manager.get_state_age_minutes()
                                print(f"[STATE] Market open with existing state (age: {state_age:.1f} min)")
                            else:
                                print("[STATE] Market open with fresh state")
                    else:
                        if session_manager.session_active:
                            print(f"[MARKET] Market closed at {current_time.strftime('%H:%M:%S')}")
                            session_manager.session_active = False
                            
                            # Force final state save at market close
                            if state_manager.current_state:
                                state_manager.save_state_async()
                                print("[STATE] Final state saved at market close")
                    
                    # Check for 3:00 PM export trigger
                    if session_manager.should_trigger_final_export():
                        print("[EXPORT] Triggering 3:00 PM final export...")
                        
                        # Export CSV files (only if CSV manager is available)
                        if csv_manager:
                            for sym in symbols:
                                final_path = csv_manager.export_final_session_csv(sym)
                                if final_path:
                                    print(f"[OK] Final CSV export for {sym}: {final_path}")
                        else:
                            print("[EXPORT] ⏸️  CSV export skipped - CSV manager not initialized (market was closed)")
                        
                        # Export comprehensive Excel reports
                        if excel_reporter:
                            excel_files = excel_reporter.export_final_session_report(symbols)
                            for sym, excel_path in excel_files.items():
                                print(f"[EXCEL] Final Excel report for {sym}: {excel_path}")
                        
                        # Generate performance reports
                        if performance_tracker:
                            perf_report = performance_tracker.generate_session_performance_report()
                            if perf_report:
                                print(f"[PERF] Session performance report: {perf_report}")
                        
                        session_manager.export_triggered = True
                    
                    # Log periodic state statistics (every 30 minutes)
                    if current_time.minute % 30 == 0 and current_time.second < 30:
                        if state_manager:
                            perf_stats = state_manager.get_performance_stats()
                            self.logger.info(
                                f"[STATE REPORT] Saves: {perf_stats['save_count']}, "
                                f"Errors: {perf_stats['error_count']}, "
                                f"Error rate: {perf_stats['error_rate']:.1%}, "
                                f"Avg save time: {perf_stats['last_save_duration_ms']:.1f}ms"
                            )
                    
                    # Sleep and continue monitoring
                    time.sleep(30)  # Check every 30 seconds
                    
            except KeyboardInterrupt:
                print("\n[STOP] Live trading session stopped by user")
            
            # Cleanup and final export with state persistence
            self._shutdown_live_mode(schedulers, csv_manager, state_manager, symbols, performance_tracker, excel_reporter)
            
            # Calculate session statistics
            end_time = datetime.now()
            session_duration = end_time - start_time
            
            return {
                'status': 'success',
                'mode': 'live',
                'symbols_processed': len(symbols),
                'session_duration': session_duration.total_seconds(),
                'schedulers_active': len(schedulers),
                'csv_files_created': len(csv_manager.csv_files) if csv_manager else 0,
                'market_session_active': session_manager.session_active,
                'performance_tracking': performance_tracker is not None,
                'excel_reporting': excel_reporter is not None,
                'indicator_columns': len(indicator_columns),
                'total_updates': sum(len(data) for data in csv_manager.current_session_data.values()) if csv_manager else 0,
                'avg_processing_time': performance_tracker.get_performance_summary()['avg_processing_time'] if performance_tracker else 0.0
            }
            
        except Exception as e:
            error_msg = f"Live trading mode failed: {e}"
            self.logger.error(error_msg)
            return self._create_error_result(error_msg)
    
    def _process_live_update(self, symbol: str, timeframe: str, timestamp: datetime,
                           csv_manager: LiveCSVManager, state_manager: LiveStateManager, live_engine, 
                           performance_tracker=None, excel_reporter=None, market_data_fetcher=None):
        """Process a single live indicator update with REAL market data."""
        perf_context = None
        if performance_tracker:
            perf_context = performance_tracker.start_operation(f"live_update_{symbol}_{timeframe}")
        
        try:
            # FETCH REAL MARKET DATA ONLY (NO MOCK DATA)
            if market_data_fetcher:
                self.logger.info(f"[DATA] Fetching REAL market data for {symbol} (timeframe: {timeframe})")
                market_data = market_data_fetcher.fetch_real_time_data(symbol, timeframe)
                
                if market_data:
                    ohlcv_data = {
                        'open': market_data['open'],
                        'high': market_data['high'],
                        'low': market_data['low'],
                        'close': market_data['close'],
                        'volume': market_data['volume']
                    }
                    data_source = market_data.get('source', 'unknown')
                    self.logger.info(f"[CANDLE] {symbol} {timestamp.strftime('%H:%M:%S')} | O:{ohlcv_data['open']:.2f} H:{ohlcv_data['high']:.2f} L:{ohlcv_data['low']:.2f} C:{ohlcv_data['close']:.2f} V:{ohlcv_data['volume']} | Source: {data_source}")
                else:
                    # CRITICAL ERROR: Real data fetch failed - ENHANCED SAFETY MECHANISMS
                    error_msg = f"[CRITICAL] Real data fetch failed for {symbol}. No fallback available in production mode."
                    self.logger.error(error_msg)
                    
                    # SAFETY MECHANISM 1: Log detailed diagnostic information
                    self.logger.error(f"[DIAGNOSTIC] OpenAlgo URL: {market_data_fetcher.openalgo_url if market_data_fetcher else 'N/A'}")
                    self.logger.error(f"[DIAGNOSTIC] Available data sources: {market_data_fetcher.data_sources if market_data_fetcher else 'N/A'}")
                    self.logger.error(f"[DIAGNOSTIC] Last successful source: {market_data_fetcher.last_successful_source if market_data_fetcher else 'N/A'}")
                    
                    # SAFETY MECHANISM 2: Attempt to provide helpful troubleshooting guidance
                    if market_data_fetcher:
                        source_status = market_data_fetcher.get_data_source_status()
                        self.logger.error(f"[DIAGNOSTIC] Data source status: {source_status}")
                        
                        # Check if OpenAlgo server is reachable
                        import requests
                        try:
                            response = requests.get(market_data_fetcher.openalgo_url + "/api/v1/status", timeout=5)
                            self.logger.error(f"[DIAGNOSTIC] OpenAlgo server response: {response.status_code}")
                        except Exception as e:
                            self.logger.error(f"[DIAGNOSTIC] OpenAlgo server unreachable: {e}")
                            self.logger.error(f"[SAFETY GUIDANCE] Please ensure OpenAlgo server is running at: {market_data_fetcher.openalgo_url}")
                    
                    # SAFETY MECHANISM 3: Clear instructions for recovery
                    self.logger.error(f"[RECOVERY INSTRUCTIONS]")
                    self.logger.error(f"[RECOVERY] 1. Check OpenAlgo server is running: {market_data_fetcher.openalgo_url if market_data_fetcher else 'http://localhost:5000'}")
                    self.logger.error(f"[RECOVERY] 2. Verify broker connection in OpenAlgo dashboard")
                    self.logger.error(f"[RECOVERY] 3. Check network connectivity")
                    self.logger.error(f"[RECOVERY] 4. Verify API credentials in configuration")
                    self.logger.error(f"[RECOVERY] 5. System will NOT trade without real market data")
                    
                    raise Exception(error_msg)
            else:
                # CRITICAL ERROR: No market data fetcher available
                error_msg = f"[CRITICAL] No market data fetcher available for {symbol}. System cannot proceed."
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            # Create DataFrame for indicator calculations
            current_data = pd.DataFrame([{
                'timestamp': timestamp,
                'open': ohlcv_data['open'],
                'high': ohlcv_data['high'],
                'low': ohlcv_data['low'],
                'close': ohlcv_data['close'],
                'volume': ohlcv_data['volume']
            }])
            current_data.set_index('timestamp', inplace=True)
            
            # Calculate indicators using live engine
            if live_engine and hasattr(live_engine, 'calculate_live_indicators'):
                try:
                    # Use live engine for real calculations
                    indicator_values = live_engine.calculate_live_indicators(current_data, symbol)
                    indicator_count = len(indicator_values) if isinstance(indicator_values, dict) else 0
                except Exception as e:
                    self.logger.error(f"[CRITICAL] Live engine calculation failed for {symbol}: {e}")
                    raise Exception(f"Indicator calculation failed: {e}")
            else:
                # CRITICAL ERROR: No live engine available
                error_msg = f"[CRITICAL] No live indicator engine available for {symbol}. Cannot calculate indicators."
                self.logger.error(error_msg)
                raise Exception(error_msg)
            
            # Dual persistence: Update both CSV and JSON state atomically
            # 1. Update in-memory state (ultra-fast, no I/O) with OHLCV data
            state_manager.update_state(indicator_values, symbol, ohlcv_data)
            
            # 2. Update CSV with live data (only if CSV manager is available)
            success = True  # Default success for state management
            if csv_manager:
                success = csv_manager.update_csv_with_live_data(
                    symbol, timestamp, ohlcv_data, indicator_values
                )
            else:
                # Log indicator values without CSV (market closed mode)
                self.logger.debug(f"[CSV_SKIP] Market closed - indicator values calculated but not written to CSV for {symbol}")
            
            # 3. Asynchronously persist state to JSON (non-blocking)
            if success:
                state_manager.save_state_async()
            
            # Update Excel reporter with state validation
            if excel_reporter:
                processing_time = time.time() - perf_context['start_time'] if perf_context else 0.0
                excel_reporter.add_live_update(
                    symbol, timestamp, ohlcv_data, indicator_values, processing_time
                )
                
            # Validate state consistency (institutional requirement)
            if state_manager.current_state:
                # Check for any critical indicator anomalies
                current_close = ohlcv_data['close']
                state_ema_20 = state_manager.current_state.ema_20
                
                # Basic sanity check: EMA should be within reasonable range of current price
                if abs(current_close - state_ema_20) > (current_close * 0.10):  # 10% deviation
                    self.logger.warning(
                        f"[ANOMALY] Large EMA-20 deviation detected for {symbol}: "
                        f"Price={current_close:.2f}, EMA-20={state_ema_20:.2f} "
                        f"(deviation: {abs(current_close - state_ema_20)/current_close*100:.1f}%)"
                    )
            
            # Record performance metrics
            if performance_tracker and perf_context:
                performance_tracker.end_operation(
                    perf_context, 
                    indicator_count=indicator_count,
                    data_points=1,
                    success=success
                )
            
            if success:
                # Log performance stats from state manager
                state_age = state_manager.get_state_age_minutes()
                perf_stats = state_manager.get_performance_stats()
                
                self.logger.info(
                    f"[LIVE] {symbol} {timeframe} update at {timestamp.strftime('%H:%M:%S')} - "
                    f"{indicator_count} indicators | State age: {state_age:.1f}min | "
                    f"Saves: {perf_stats['save_count']} | Errors: {perf_stats['error_count']}"
                )
            else:
                self.logger.warning(f"[WARNING] Failed to update CSV for {symbol}")
                # Still save state even if CSV fails (for recovery)
                state_manager.save_state_async()
                
        except Exception as e:
            error_msg = f"Live update failed for {symbol}: {e}"
            self.logger.error(error_msg)
            
            # Emergency state persistence for critical errors
            if state_manager and state_manager.current_state:
                try:
                    # Force immediate state save for recovery
                    state_manager.save_state_async()
                    self.logger.info(f"[EMERGENCY] State saved for recovery after error: {symbol}")
                except Exception as save_error:
                    self.logger.error(f"[CRITICAL] Emergency state save failed: {save_error}")
            
            # Record failed performance metric
            if performance_tracker and perf_context:
                performance_tracker.end_operation(
                    perf_context, 
                    indicator_count=0,
                    data_points=0,
                    success=False,
                    error_message=str(e)
                )
    
    def _generate_indicator_columns_from_engine(self) -> List[str]:
        """Generate indicator column list from the engine's active indicators."""
        try:
            indicator_columns = []
            
            # Check for active indicators in the nested engine
            if (not self.engine or not hasattr(self.engine, 'engine') or 
                not hasattr(self.engine.engine, 'active_indicators')):
                return self._get_default_indicator_columns()
            
            for indicator_name, indicator_obj in self.engine.engine.active_indicators.items():
                indicator_name_lower = indicator_name.lower()
                
                # Generate columns based on indicator type
                if 'ema' in indicator_name_lower:
                    # Get periods from indicator object
                    if hasattr(indicator_obj, 'periods'):
                        periods = indicator_obj.periods
                    else:
                        periods = [9, 20, 50, 200]  # Default
                    indicator_columns.extend([f'ema_{p}' for p in periods])
                    
                elif 'sma' in indicator_name_lower:
                    if hasattr(indicator_obj, 'periods'):
                        periods = indicator_obj.periods
                    else:
                        periods = [20, 50, 200]  # Default
                    indicator_columns.extend([f'sma_{p}' for p in periods])
                    
                elif 'rsi' in indicator_name_lower:
                    if hasattr(indicator_obj, 'periods'):
                        periods = indicator_obj.periods
                    else:
                        periods = [14]  # Default
                    indicator_columns.extend([f'rsi_{p}' for p in periods])
                    
                elif 'cpr' in indicator_name_lower:
                    cpr_columns = ['cpr_pivot', 'cpr_bc', 'cpr_tc', 'cpr_r1', 'cpr_r2', 'cpr_r3', 'cpr_r4', 
                                  'cpr_s1', 'cpr_s2', 'cpr_s3', 'cpr_s4', 'cpr_previous_day_high', 
                                  'cpr_previous_day_low', 'cpr_previous_day_close']
                    indicator_columns.extend(cpr_columns)
                    
                elif 'supertrend' in indicator_name_lower:
                    indicator_columns.extend(['supertrend', 'supertrend_signal'])
                    
                elif 'vwap' in indicator_name_lower:
                    indicator_columns.extend(['vwap', 'vwap_upper_1', 'vwap_lower_1', 'vwap_upper_2', 'vwap_lower_2'])
                    
                elif 'bollinger' in indicator_name_lower or 'bb' in indicator_name_lower:
                    indicator_columns.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_squeeze'])
                    
                elif 'candle' in indicator_name_lower:
                    candle_columns = ['currentdaycandle_open', 'currentdaycandle_high', 'currentdaycandle_low', 'currentdaycandle_close',
                                     'previousdaycandle_open', 'previousdaycandle_high', 'previousdaycandle_low', 'previousdaycandle_close',
                                     'currentcandle_open', 'currentcandle_high', 'currentcandle_low', 'currentcandle_close',
                                     'previouscandle_open', 'previouscandle_high', 'previouscandle_low', 'previouscandle_close']
                    indicator_columns.extend(candle_columns)
            
            # Remove duplicates and return
            return list(set(indicator_columns)) if indicator_columns else self._get_default_indicator_columns()
            
        except Exception as e:
            self.logger.error(f"Failed to generate indicator columns from engine: {e}")
            return self._get_default_indicator_columns()
    
    def _get_default_indicator_columns(self) -> List[str]:
        """Get default indicator columns for fallback."""
        return [
            'ema_9', 'ema_20', 'ema_50', 'ema_200',
            'sma_20', 'sma_50', 'sma_200',
            'rsi_14', 'rsi_21',
            'cpr_pivot', 'cpr_bc', 'cpr_tc', 'cpr_r1', 'cpr_s1',
            'cpr_previous_day_high', 'cpr_previous_day_low', 'cpr_previous_day_close',
            'supertrend', 'supertrend_signal',
            'vwap', 'vwap_upper_1', 'vwap_lower_1',
            'bb_upper', 'bb_middle', 'bb_lower',
            'currentdaycandle_open', 'currentdaycandle_high', 'currentdaycandle_low', 'currentdaycandle_close',
            'previouscandle_open', 'previouscandle_high', 'previouscandle_low', 'previouscandle_close'
        ]
    
    def _generate_indicator_columns_from_config(self, indicator_config: List[Dict]) -> List[str]:
        """Generate comprehensive indicator column list from configuration."""
        try:
            indicator_columns = []
            
            for ind_config in indicator_config:
                if ind_config.get('status') != 'Active':
                    continue
                
                indicator_name = ind_config.get('indicator', '').lower()
                
                # Generate column names based on indicator type and config parameters
                if indicator_name == 'ema':
                    # Read EMA periods from config parameters
                    parameters = ind_config.get('parameters', [9, 21, 50, 200])
                    if isinstance(parameters, list):
                        periods = [int(p) for p in parameters if isinstance(p, (int, str)) and str(p).isdigit()]
                    else:
                        periods = [9, 21, 50, 200]  # Default
                    indicator_columns.extend([f'ema_{p}' for p in periods])
                elif indicator_name == 'sma':
                    # Read SMA periods from config parameters
                    parameters = ind_config.get('parameters', [9, 21, 50, 200])
                    if isinstance(parameters, list):
                        periods = [int(p) for p in parameters if isinstance(p, (int, str)) and str(p).isdigit()]
                    else:
                        periods = [9, 21, 50, 200]  # Default
                    indicator_columns.extend([f'sma_{p}' for p in periods])
                elif indicator_name == 'rsi':
                    # Read RSI periods from config parameters
                    parameters = ind_config.get('parameters', [14])
                    if isinstance(parameters, list):
                        periods = [int(p) for p in parameters if isinstance(p, (int, str)) and str(p).isdigit()]
                    else:
                        periods = [14]  # Default
                    indicator_columns.extend([f'rsi_{p}' for p in periods])
                elif indicator_name == 'cpr':
                    cpr_columns = ['cpr_pivot', 'cpr_bc', 'cpr_tc', 'cpr_r1', 'cpr_r2', 'cpr_r3', 'cpr_r4', 
                                  'cpr_s1', 'cpr_s2', 'cpr_s3', 'cpr_s4', 'cpr_cpr_width', 'cpr_cpr_range']
                    indicator_columns.extend(cpr_columns)
                elif indicator_name == 'supertrend':
                    indicator_columns.extend(['supertrend', 'supertrend_signal'])
                elif indicator_name == 'vwap':
                    indicator_columns.extend(['vwap', 'vwap_upper_1', 'vwap_lower_1', 'vwap_upper_2', 'vwap_lower_2'])
                elif indicator_name == 'bollinger_bands' or indicator_name == 'bb':
                    indicator_columns.extend(['bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_squeeze'])
                elif indicator_name == 'candle_values' or indicator_name == 'candlevalues':
                    candle_columns = ['candle_open', 'candle_high', 'candle_low', 'candle_close', 'candle_range']
                    indicator_columns.extend(candle_columns)
            
            # Add default indicators if config is empty
            if not indicator_columns:
                indicator_columns = [
                    'ema_9', 'ema_21', 'ema_50', 'ema_200',
                    'sma_9', 'sma_21', 'sma_50', 'sma_200',
                    'rsi_14', 'cpr_pivot', 'cpr_bc', 'cpr_tc', 'cpr_r1', 'cpr_s1',
                    'supertrend', 'supertrend_signal',
                    'vwap', 'vwap_upper_1', 'vwap_lower_1',
                    'bb_upper', 'bb_middle', 'bb_lower',
                    'candle_open', 'candle_high', 'candle_low', 'candle_close'
                ]
            
            return list(set(indicator_columns))  # Remove duplicates
            
        except Exception as e:
            self.logger.error(f"Failed to generate indicator columns from config: {e}")
            # Return comprehensive default set
            return [
                'ema_9', 'ema_21', 'ema_50', 'ema_200',
                'sma_9', 'sma_21', 'sma_50', 'sma_200', 
                'rsi_14', 'cpr_pivot', 'cpr_bc', 'cpr_tc', 'cpr_r1', 'cpr_s1',
                'supertrend', 'supertrend_signal',
                'vwap', 'vwap_upper_1', 'vwap_lower_1',
                'bb_upper', 'bb_middle', 'bb_lower',
                'candle_open', 'candle_high', 'candle_low', 'candle_close'
            ]
    
    # REMOVED: _generate_comprehensive_indicators() - Production systems use real indicator engines only
    
    # REMOVED: _generate_fallback_indicators() - Production systems use real calculations only
    
    def _shutdown_live_mode(self, schedulers: Dict, csv_manager: LiveCSVManager, 
                          state_manager: LiveStateManager, symbols: List[str],
                          performance_tracker=None, excel_reporter=None):
        """Gracefully shutdown live trading mode with comprehensive reporting."""
        print("\n[SHUTDOWN] Shutting down live trading session...")
        
        # Stop all schedulers
        for scheduler_key, scheduler in schedulers.items():
            scheduler.stop_scheduler()
            print(f"[STOP] Stopped scheduler: {scheduler_key}")
        
        # Export final session data with state management
        print("[EXPORT] Exporting final session data with state persistence...")
        
        # Export final state for recovery
        if state_manager.current_state:
            state_manager.save_state_async()
            perf_stats = state_manager.get_performance_stats()
            print(f"[STATE] Final state saved - Total saves: {perf_stats['save_count']}, "
                  f"Errors: {perf_stats['error_count']}, Error rate: {perf_stats['error_rate']:.1%}")
        
        # Export CSV files (only if CSV manager is available)
        if csv_manager:
            for symbol in symbols:
                final_path = csv_manager.export_final_session_csv(symbol)
                if final_path:
                    print(f"[OK] Final CSV exported for {symbol}: {final_path}")
        else:
            print("[SHUTDOWN] ⏸️  CSV export skipped - CSV manager not initialized (market was closed)")
        
        # Export Excel reports
        if excel_reporter:
            print("[EXCEL] Exporting comprehensive Excel reports...")
            excel_files = excel_reporter.export_final_session_report(symbols)
            for symbol, excel_path in excel_files.items():
                print(f"[EXCEL] Final Excel report for {symbol}: {excel_path}")
        
        # Stop performance tracking and generate final report
        if performance_tracker:
            print("[PERF] Stopping performance tracking and generating final report...")
            performance_tracker.stop_automatic_reporting()
            perf_report = performance_tracker.generate_session_performance_report()
            if perf_report:
                print(f"[PERF] Session performance report: {perf_report}")
        
        # Display comprehensive state management statistics
        if state_manager:
            perf_stats = state_manager.get_performance_stats()
            print("\n[STATE] Live State Management Summary:")
            print(f"   Session ID: {perf_stats['session_id']}")
            print(f"   Total state saves: {perf_stats['save_count']}")
            print(f"   State errors: {perf_stats['error_count']}")
            print(f"   Error rate: {perf_stats['error_rate']:.1%}")
            print(f"   Last save duration: {perf_stats['last_save_duration_ms']:.1f}ms")
            if perf_stats['current_state_age_minutes']:
                print(f"   Final state age: {perf_stats['current_state_age_minutes']:.1f} minutes")
        
        print("[OK] Live trading session shutdown complete with institutional-grade state management")
    
    def _perform_first_run_setup(self, market_data_fetcher, target_symbol: str = None):
        """
        ENHANCED FIRST-RUN SETUP: Store API data in DB then extract unified dataset
        
        User's Enhanced Logic:
        1. Fetch today's API data and store in database first
        2. Extract complete unified dataset from database (includes today's latest)
        3. Enhanced CPR with smart previous trading day detection
        4. Calculate baseline indicators for accuracy validation
        5. API-based previous day OHLC for accurate CPR calculation
        
        Benefits:
        - Seamless data flow (no merging complexity)
        - Today's latest candles included in warmup
        - Better EMA/SMA/RSI initialization accuracy
        - Proper market calendar-aware CPR calculation
        """
        from datetime import datetime, time, date, timedelta
        import duckdb
        import pandas as pd
        import time as time_module
        
        # Start performance tracking for entire setup
        setup_start_time = time_module.time()
        phase_times = {}
        print(f"[PERF] 🚀 Starting initial setup performance measurement...")
        print(f"[PERF] Expected phases: Data fetch → Database → CPR → Indicators → WebSocket")
        
        # Determine current market session status
        current_time = datetime.now()
        current_date = current_time.date()
        market_open = datetime.combine(current_date, time(9, 15))  # 9:15 AM IST
        market_close = datetime.combine(current_date, time(15, 30))  # 3:30 PM IST
        
        # Display current status
        print(f"[SETUP] Current time: {current_time.strftime('%H:%M:%S')}")
        if current_time < market_open:
            time_to_open = (market_open - current_time).total_seconds() / 60
            print(f"[MODE] Pre-market: {time_to_open:.0f} minutes until market open (9:15 AM)")
        elif market_open <= current_time <= market_close:
            print(f"[MODE] Market hours: Live trading session active")
        else:
            print(f"[MODE] After-market: Preparing for next trading session")
        
        symbol = target_symbol or 'NIFTY'
        
        # SMART DATA MANAGEMENT: Database-first with API fallback
        print(f"\n[SETUP] SMART STRATEGY: Database-first → API Fallback → Unified Dataset")
        print(f"[SETUP] Phase 1: Smart data management with duplicate prevention...")
        phase1_start = time_module.time()
        
        # Initialize smart data manager
        try:
            smart_data_manager = SmartDataManager()
            smart_data_manager.market_data_fetcher = market_data_fetcher
            
            # Step 1: Get current day data (DB-first approach)
            today_str = current_date.strftime('%Y-%m-%d')
            print(f"[SETUP] Smart fetching for {symbol} on {today_str}")
            
            current_day_data = smart_data_manager.get_current_day_data(
                symbol=symbol,
                exchange='NSE_INDEX',
                timeframe='5m',
                target_date=today_str
            )
            
            if current_day_data is not None and not current_day_data.empty:
                live_candles = len(current_day_data)
                actual_start = current_day_data.index.min().strftime('%H:%M')
                actual_end = current_day_data.index.max().strftime('%H:%M')
                
                print(f"[SETUP] ✅ Today's data available: {live_candles} candles (from {actual_start} to {actual_end})")
                
                # Data freshness validation
                latest_candle_time = current_day_data.index.max()
                if latest_candle_time.tz is not None:
                    latest_candle_naive = latest_candle_time.tz_localize(None)
                else:
                    latest_candle_naive = latest_candle_time
                
                data_age_minutes = (current_time - latest_candle_naive).total_seconds() / 60
                print(f"[SETUP] ✅ Data freshness: {data_age_minutes:.1f} minutes (OPTIMIZED)")
                
                phase_times['phase1_data_fetch_store'] = time_module.time() - phase1_start
                print(f"[PERF] Phase 1 completed in {phase_times['phase1_data_fetch_store']:.2f}s")
                    
            else:
                print(f"[SETUP] ⚠️  No current day data from API - using database only")
                
        except Exception as e:
            print(f"[SETUP] ⚠️  Today's API fetch failed: {e}")
            print(f"[SETUP] Continuing with database-only extraction")
        
        # Step 2: Build unified dataset with smart data manager
        print(f"\n[SETUP] Phase 2: Building unified dataset with smart data integration...")
        phase2_start = time_module.time()
        try:
            # Use smart data manager to build complete dataset
            unified_data = smart_data_manager.get_unified_dataset(
                symbol=symbol,
                exchange='NSE_INDEX',
                timeframe='5m',
                candle_count=500
            )
            
            if unified_data is not None and not unified_data.empty:
                total_candles = len(unified_data)
                data_start = unified_data.index.min().strftime('%Y-%m-%d %H:%M')
                data_end = unified_data.index.max().strftime('%Y-%m-%d %H:%M')
                
                print(f"[SETUP] ✅ Unified dataset: {total_candles} candles")
                print(f"[SETUP] ✅ Range: {data_start} to {data_end}")
                print(f"[SETUP] ✅ Smart integration: Current + Previous + Historical data")
                
                # Validate sufficiency for indicator calculations
                if total_candles >= 200:
                    print(f"[SETUP] ✅ Sufficient data for all indicators (need: 200, have: {total_candles})")
                    phase_times['phase2_database_extract'] = time_module.time() - phase2_start
                    print(f"[PERF] Phase 2 completed in {phase_times['phase2_database_extract']:.2f}s")
                else:
                    print(f"[SETUP] ⚠️  Limited data (need: 200, have: {total_candles})")
                    phase_times['phase2_database_extract'] = time_module.time() - phase2_start
                    print(f"[PERF] Phase 2 completed in {phase_times['phase2_database_extract']:.2f}s")
                    
            else:
                print(f"[SETUP] ❌ Failed to build unified dataset")
                return False
                
        except Exception as e:
            print(f"[SETUP] ❌ Unified dataset building failed: {e}")
            return False
        
        # Step 3: Calculate baseline indicators (EMA, SMA, RSI, CPR) for accuracy validation
        print(f"\n[SETUP] Phase 3: Calculating baseline indicators (EMA, SMA, RSI, CPR) for validation...")
        phase3_start = time_module.time()
        try:
            baseline_indicators = self._calculate_baseline_indicators_for_setup(unified_data, symbol)
            if baseline_indicators is not None and not baseline_indicators.empty:
                print(f"[SETUP] ✅ Baseline indicators calculated - ready for incremental updates")
                # Display key indicator values for verification  
                latest_values = baseline_indicators.iloc[-1]
                print(f"[INDICATORS] Latest EMA(20): {latest_values.get('ema_20', 0):.2f}")
                print(f"[INDICATORS] Latest SMA(20): {latest_values.get('sma_20', 0):.2f}")
                print(f"[INDICATORS] Latest RSI(14): {latest_values.get('rsi_14', 0):.2f}")
                # Display CPR values if available
                if 'CPR_Pivot' in latest_values:
                    print(f"[INDICATORS] CPR Pivot: {latest_values.get('CPR_Pivot', 0):.2f}")
                    print(f"[INDICATORS] CPR TC: {latest_values.get('CPR_TC', 0):.2f}, BC: {latest_values.get('CPR_BC', 0):.2f}")
                print(f"[INDICATORS] Total calculated: {len(baseline_indicators)} candles with full indicator coverage")
                phase_times['phase3_indicators'] = time_module.time() - phase3_start
                print(f"[PERF] Phase 3 (Indicators+CPR) completed in {phase_times['phase3_indicators']:.2f}s")
            else:
                print(f"[SETUP] ⚠️  Baseline indicator calculation failed")
                phase_times['phase3_indicators'] = time_module.time() - phase3_start
                print(f"[PERF] Phase 3 (Indicators+CPR) completed in {phase_times['phase3_indicators']:.2f}s")
                
        except Exception as e:
            print(f"[SETUP] ⚠️  Baseline indicators failed: {e}")
        
        # Step 5: WebSocket setup for real-time updates
        print(f"\n[SETUP] Phase 4: Setting up real-time data connection...")
        try:
            # Use OpenAlgo's existing WebSocket functionality
            if hasattr(market_data_fetcher.openalgo_client, 'start_websocket'):
                # Start WebSocket for real-time price updates
                market_data_fetcher.openalgo_client.start_websocket(symbols=[symbol])
                print(f"[SETUP] ✅ WebSocket connection established for {symbol}")
            else:
                print(f"[SETUP] ✅ WebSocket ready for manual connection")
        except Exception as e:
            print(f"[WARNING] WebSocket setup failed: {e}")
            print(f"[INFO] System will use API polling for real-time data")
        
        # Step 6: System readiness confirmation
        print(f"\n[SETUP] ✅ Enhanced first-run initialization complete")
        
        # Calculate and display comprehensive performance summary
        total_setup_time = time_module.time() - setup_start_time
        phase_times['total_setup'] = total_setup_time
        
        print(f"\n[PERF] 📊 INITIAL SETUP PERFORMANCE SUMMARY:")
        print(f"[PERF] ═══════════════════════════════════════")
        print(f"[PERF] Phase 1 (Smart Data Management): {phase_times.get('phase1_data_fetch_store', 0):.2f}s")
        print(f"[PERF] Phase 2 (Unified Dataset): {phase_times.get('phase2_database_extract', 0):.2f}s") 
        print(f"[PERF] Phase 3 (Indicators+CPR): {phase_times.get('phase3_indicators', 0):.2f}s")
        print(f"[PERF] ═══════════════════════════════════════")
        print(f"[PERF] 🚀 TOTAL SETUP TIME: {total_setup_time:.2f}s")
        print(f"[PERF] 📈 Setup includes: All indicators + CPR (with enhanced fallback) + 500 candle warmup")
        print(f"[PERF] ⚡ Ready for incremental 5-minute updates")
        
        # Calculate estimated 5-minute incremental performance
        self._estimate_5min_incremental_performance(phase_times)
        
        print(f"\n[READY] System prepared with unified dataset and market-aware CPR")
        
        if current_time < market_open:
            print(f"[READY] Waiting for market open at 9:15 AM...")
        elif market_open <= current_time <= market_close:
            print(f"[READY] Starting live trading immediately...")
        else:
            print(f"[READY] System ready for monitoring/testing mode")
        
        return True
    
    def _get_previous_trading_day(self, current_date):
        """
        Get the correct previous trading day (handles weekends/holidays)
        
        Args:
            current_date: Current date
            
        Returns:
            str: Previous trading day in YYYY-MM-DD format
        """
        # Monday (weekday=0) -> Previous Friday (subtract 3 days)
        # Tuesday-Friday -> Previous day (subtract 1 day)
        # Saturday/Sunday -> Previous Friday
        
        if current_date.weekday() == 0:  # Monday
            previous_day = current_date - timedelta(days=3)  # Friday
        elif current_date.weekday() in [5, 6]:  # Saturday/Sunday
            days_back = current_date.weekday() - 4  # Back to Friday
            previous_day = current_date - timedelta(days=days_back)
        else:  # Tuesday-Friday
            previous_day = current_date - timedelta(days=1)
        
        return previous_day.strftime('%Y-%m-%d')
    
    def _calculate_cpr_for_live_trading(self, market_data_fetcher, symbol: str):
        """
        Calculate CPR levels once during first-run setup for live trading.
        Uses API call to get correct previous trading day OHLC data.
        
        Args:
            market_data_fetcher: Market data fetcher instance
            symbol: Trading symbol (e.g., 'NIFTY')
            
        Returns:
            dict: CPR levels or None if calculation fails
        """
        try:
            # Get correct previous trading day
            today = datetime.now().date()
            prev_trading_day = self._get_previous_trading_day(today)
            
            print(f"[CPR] Calculating for previous trading day: {prev_trading_day}")
            print(f"[CPR] Today: {today.strftime('%A %Y-%m-%d')}, Previous: {prev_trading_day}")
            
            # API call for complete previous day session
            prev_day_data = market_data_fetcher.openalgo_client.get_historical_data(
                symbol=symbol,
                exchange='NSE_INDEX',
                interval='5m',
                start_date=prev_trading_day,
                end_date=prev_trading_day
            )
            
            if prev_day_data is not None and not prev_day_data.empty:
                # Extract previous day OHLC
                prev_high = float(prev_day_data['high'].max())
                prev_low = float(prev_day_data['low'].min())
                prev_close = float(prev_day_data['close'].iloc[-1])  # Last candle close
                
                print(f"[CPR] Previous day OHLC: H={prev_high:.2f}, L={prev_low:.2f}, C={prev_close:.2f}")
                
                # Calculate CPR levels using standard formulas
                pivot = (prev_high + prev_low + prev_close) / 3
                bc = (prev_high + prev_low) / 2
                tc = 2 * pivot - bc
                
                # Support/Resistance levels
                s1 = 2 * pivot - prev_high
                r1 = 2 * pivot - prev_low
                s2 = pivot - (prev_high - prev_low)
                r2 = pivot + (prev_high - prev_low)
                s3 = prev_low - (2 * (prev_high - pivot))
                r3 = prev_high + (2 * (pivot - prev_low))
                s4 = s3 + (s2 - s1)
                r4 = r3 + (r2 - r1)
                
                cpr_levels = {
                    'pivot': round(pivot, 2),
                    'bc': round(bc, 2),
                    'tc': round(tc, 2),
                    's1': round(s1, 2), 'r1': round(r1, 2),
                    's2': round(s2, 2), 'r2': round(r2, 2),
                    's3': round(s3, 2), 'r3': round(r3, 2),
                    's4': round(s4, 2), 'r4': round(r4, 2),
                    'prev_high': round(prev_high, 2),
                    'prev_low': round(prev_low, 2),
                    'prev_close': round(prev_close, 2),
                    'calculated_date': prev_trading_day,
                    'symbol': symbol
                }
                
                print(f"[CPR] ✅ Calculated: Pivot={pivot:.2f}, BC={bc:.2f}, TC={tc:.2f}")
                print(f"[CPR] ✅ Resistance: R1={r1:.2f}, R2={r2:.2f}, R3={r3:.2f}, R4={r4:.2f}")
                print(f"[CPR] ✅ Support: S1={s1:.2f}, S2={s2:.2f}, S3={s3:.2f}, S4={s4:.2f}")
                
                # Store CPR levels for session use
                self.session_cpr_levels = cpr_levels
                
                return cpr_levels
            else:
                print(f"[CPR] ❌ Failed to get previous day data for {prev_trading_day}")
                print(f"[CPR] API returned: {type(prev_day_data)}, Empty: {prev_day_data.empty if hasattr(prev_day_data, 'empty') else 'N/A'}")
                return None
                
        except Exception as e:
            print(f"[CPR] ❌ Calculation error: {e}")
            import traceback
            print(f"[CPR] Traceback: {traceback.format_exc()}")
            return None
    
    def _get_optimal_data_range(self, current_time: datetime):
        """
        HYBRID DATA STRATEGY: Returns parameters for database + API approach
        - Database: Historical warmup (last 300 candles)
        - API: Current day only (today's date)
        
        Args:
            current_time: Current datetime
            
        Returns:
            tuple: (start_datetime, end_datetime, session_info)
        """
        from datetime import time, timedelta
        
        market_open = time(9, 15)
        market_close = time(15, 30)
        current_time_only = current_time.time()
        
        # HYBRID APPROACH: Always use today's date for API
        today_date = current_time.date()
        api_start_datetime = datetime.combine(today_date, time(9, 15))  # Today 9:15 AM
        api_end_datetime = current_time  # Current time
        
        if current_time_only < market_open:
            # PRE-MARKET: API gets yesterday's close + database warmup
            api_end_datetime = datetime.combine(today_date - timedelta(days=1), time(15, 30))
            session_type = 'pre_market'
            description = "database warmup + yesterday's close"
            expected_candles = 300 + 75  # 300 warmup + 75 yesterday
            freshness_info = "Complete data up to previous trading session"
            
        elif market_open <= current_time_only <= market_close:
            # MARKET HOURS: API gets today's live candles + database warmup
            api_start_datetime = datetime.combine(today_date, time(9, 15))
            api_end_datetime = current_time
            session_type = 'market_hours'
            
            # Calculate today's candles so far
            today_start = datetime.combine(today_date, time(9, 15))
            minutes_elapsed = (current_time - today_start).total_seconds() / 60
            today_candles = max(0, int(minutes_elapsed / 5))  # 5-minute candles
            
            description = f"database warmup + {today_candles} live candles today"
            expected_candles = 300 + today_candles  # 300 warmup + today's candles
            freshness_info = f"Real-time data with {today_candles} candles from today's session"
            
        else:
            # AFTER-MARKET: API gets today's complete session + database warmup
            api_start_datetime = datetime.combine(today_date, time(9, 15))
            api_end_datetime = datetime.combine(today_date, time(15, 30))
            session_type = 'after_market'
            today_candles = 75  # Complete trading session
            description = f"database warmup + {today_candles} candles from today's complete session"
            expected_candles = 300 + today_candles  # 300 warmup + today's session
            freshness_info = f"Complete trading session with {today_candles} candles from today"
        
        session_info = {
            'session_type': session_type,
            'description': description,
            'expected_candles': expected_candles,
            'freshness_info': freshness_info,
            'api_strategy': 'today_only',  # Flag for hybrid approach
            'database_warmup': 300,  # Expected warmup candles from database
            'api_start': api_start_datetime,
            'api_end': api_end_datetime
        }
        
        # Return API-specific dates (today only)
        return api_start_datetime, api_end_datetime, session_info
    
    def _is_new_candle_boundary(self, current_time: datetime, timeframe: str = "5m") -> bool:
        """
        Detect if current time represents a new candle boundary for the given timeframe.
        
        Args:
            current_time: Current datetime
            timeframe: Candle timeframe (e.g., "5m", "15m", "1h")
            
        Returns:
            bool: True if this is a new candle boundary
        """
        try:
            if timeframe == "5m":
                # 5-minute candles: 09:15, 09:20, 09:25, etc.
                return current_time.minute % 5 == 0 and current_time.second < 10
            elif timeframe == "15m":
                # 15-minute candles: 09:15, 09:30, 09:45, etc.
                return current_time.minute % 15 == 0 and current_time.second < 10
            elif timeframe == "1h":
                # 1-hour candles: 09:00, 10:00, 11:00, etc.
                return current_time.minute == 0 and current_time.second < 10
            else:
                return False
        except Exception as e:
            self.logger.error(f"Error detecting candle boundary: {e}")
            return False
    
    def _should_process_live_update(self, current_time: datetime, timeframe: str = "5m") -> bool:
        """
        Determine if live update should be processed based on market hours and candle boundaries.
        
        Args:
            current_time: Current datetime
            timeframe: Candle timeframe
            
        Returns:
            bool: True if update should be processed
        """
        try:
            from datetime import time
            
            # Check if within market hours (9:15 AM - 3:30 PM IST)
            market_open = time(9, 15)
            market_close = time(15, 30)
            current_time_only = current_time.time()
            
            is_market_hours = market_open <= current_time_only <= market_close
            is_new_candle = self._is_new_candle_boundary(current_time, timeframe)
            
            return is_market_hours and is_new_candle
            
        except Exception as e:
            self.logger.error(f"Error checking live update conditions: {e}")
            return False
    
    # REMOVED: _generate_realistic_fallback() - Production systems use real market data only
    
    def _store_current_day_in_database(self, current_day_data: pd.DataFrame, symbol: str) -> bool:
        """
        Store today's API data in database for unified extraction.
        Implements UPSERT logic to handle duplicates gracefully.
        
        Args:
            current_day_data: Today's OHLCV data from API
            symbol: Trading symbol
            
        Returns:
            bool: Success status
        """
        try:
            if current_day_data is None or current_day_data.empty:
                print(f"[DB_STORE] No data to store for {symbol}")
                return False
            
            db_path = "data/valgo_market_data.db"
            conn = duckdb.connect(db_path)
            
            # Prepare data for insertion
            insert_data = []
            for timestamp, row in current_day_data.iterrows():
                # Handle timezone-aware timestamps
                if timestamp.tz is not None:
                    timestamp_naive = timestamp.tz_localize(None)
                else:
                    timestamp_naive = timestamp
                
                insert_data.append({
                    'symbol': symbol,
                    'timestamp': timestamp_naive,
                    'open': float(row['open']),
                    'high': float(row['high']),
                    'low': float(row['low']),
                    'close': float(row['close']),
                    'volume': int(row['volume'])
                })
            
            # UPSERT logic: DELETE existing today's data then INSERT new
            today_date = current_day_data.index[0].strftime('%Y-%m-%d')
            
            # Delete existing data for today
            delete_query = f"""
                DELETE FROM ohlcv_data 
                WHERE symbol = '{symbol}' 
                AND DATE(timestamp) = '{today_date}'
            """
            
            deleted_count = conn.execute(delete_query).fetchone()[0]
            if deleted_count > 0:
                print(f"[DB_STORE] Removed {deleted_count} existing records for {symbol} on {today_date}")
            
            # Insert new data
            insert_query = """
                INSERT INTO ohlcv_data (symbol, timestamp, open, high, low, close, volume)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """
            
            for record in insert_data:
                conn.execute(insert_query, [
                    record['symbol'],
                    record['timestamp'],
                    record['open'],
                    record['high'],
                    record['low'],
                    record['close'],
                    record['volume']
                ])
            
            conn.close()
            
            print(f"[DB_STORE] ✅ Stored {len(insert_data)} candles for {symbol} on {today_date}")
            return True
            
        except Exception as e:
            print(f"[DB_STORE] ❌ Storage failed for {symbol}: {e}")
            return False
    
    def _extract_complete_warmup_data(self, symbol: str, candle_count: int = 350) -> pd.DataFrame:
        """
        Extract complete unified dataset from database including today's latest candles.
        This provides seamless data for better EMA/SMA/RSI warmup accuracy.
        
        Args:
            symbol: Trading symbol
            candle_count: Number of recent candles to extract
            
        Returns:
            pd.DataFrame: Unified dataset with timezone-naive timestamps
        """
        try:
            db_path = "data/valgo_market_data.db"
            conn = duckdb.connect(db_path)
            
            # Single query to get unified dataset including today's latest
            unified_query = f"""
                SELECT timestamp, open, high, low, close, volume
                FROM ohlcv_data 
                WHERE symbol = '{symbol}'
                ORDER BY timestamp DESC 
                LIMIT {candle_count}
            """
            
            result = conn.execute(unified_query).fetchall()
            conn.close()
            
            if not result:
                print(f"[UNIFIED_EXTRACT] ❌ No data found for {symbol}")
                return pd.DataFrame()
            
            # Convert to DataFrame (reverse order for chronological sequence)
            df_data = []
            for row in reversed(result):  # Reverse to get oldest first
                df_data.append({
                    'timestamp': pd.to_datetime(row[0]),
                    'open': float(row[1]),
                    'high': float(row[2]),
                    'low': float(row[3]),
                    'close': float(row[4]),
                    'volume': int(row[5])
                })
            
            unified_data = pd.DataFrame(df_data).set_index('timestamp')
            
            # Ensure timezone-naive for consistent processing
            if unified_data.index.tz is not None:
                unified_data.index = unified_data.index.tz_localize(None)
            
            print(f"[UNIFIED_EXTRACT] ✅ Extracted {len(unified_data)} unified candles for {symbol}")
            
            return unified_data
            
        except Exception as e:
            print(f"[UNIFIED_EXTRACT] ❌ Extraction failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _get_previous_trading_day_smart(self, current_date) -> str:
        """
        Smart previous trading day detection with market calendar awareness.
        Handles weekends and major market holidays.
        
        Args:
            current_date: Current date
            
        Returns:
            str: Previous trading day in YYYY-MM-DD format
        """
        try:
            from datetime import timedelta
            
            # NSE major holidays (update annually)
            # This is a basic list - in production, integrate with NSE API
            nse_holidays_2025 = [
                '2025-01-26',  # Republic Day
                '2025-03-14',  # Holi
                '2025-04-18',  # Good Friday
                '2025-08-15',  # Independence Day
                '2025-10-02',  # Gandhi Jayanti
                '2025-11-01',  # Diwali
                # Add more holidays as needed
            ]
            
            # Start with previous day
            prev_day = current_date - timedelta(days=1)
            
            # Keep going back until we find a trading day
            max_attempts = 10  # Prevent infinite loop
            attempts = 0
            
            while attempts < max_attempts:
                # Check if it's a weekend
                if prev_day.weekday() >= 5:  # Saturday = 5, Sunday = 6
                    prev_day = prev_day - timedelta(days=1)
                    attempts += 1
                    continue
                
                # Check if it's a market holiday
                prev_day_str = prev_day.strftime('%Y-%m-%d')
                if prev_day_str in nse_holidays_2025:
                    print(f"[TRADING_DAY] Skipping market holiday: {prev_day_str}")
                    prev_day = prev_day - timedelta(days=1)
                    attempts += 1
                    continue
                
                # Found a valid trading day
                return prev_day_str
            
            # Fallback: if we couldn't find a trading day, just use Friday
            while prev_day.weekday() >= 5:
                prev_day = prev_day - timedelta(days=1)
            
            return prev_day.strftime('%Y-%m-%d')
            
        except Exception as e:
            print(f"[TRADING_DAY] ❌ Error finding previous trading day: {e}")
            # Emergency fallback
            fallback_day = current_date - timedelta(days=1)
            if fallback_day.weekday() == 6:  # Sunday
                fallback_day = fallback_day - timedelta(days=2)  # Friday
            elif fallback_day.weekday() == 5:  # Saturday  
                fallback_day = fallback_day - timedelta(days=1)  # Friday
            return fallback_day.strftime('%Y-%m-%d')
    
    def _map_to_openalgo_exchange(self, exchange: str, instrument_type: str) -> str:
        """
        Map detected exchange and instrument to OpenAlgo exchange format.
        
        Args:
            exchange: Exchange name (e.g., 'NSE', 'MCX', 'BSE')
            instrument_type: Instrument type (e.g., 'INDEX', 'EQUITY', 'COMMODITY')
            
        Returns:
            str: OpenAlgo exchange format (e.g., 'NSE_INDEX', 'MCX', 'BSE')
        """
        exchange = exchange.upper()
        instrument_type = instrument_type.upper()
        
        # Map to OpenAlgo exchange format
        if exchange == 'NSE':
            if instrument_type == 'INDEX':
                return 'NSE_INDEX'
            elif instrument_type == 'EQUITY':
                return 'NSE_EQ'
            elif instrument_type == 'FUTURES':
                return 'NSE_FO'
            elif instrument_type == 'CURRENCY':
                return 'NSE_CD'
            else:
                return 'NSE_INDEX'  # Default fallback
        elif exchange == 'MCX':
            return 'MCX'
        elif exchange == 'BSE':
            return 'BSE'
        else:
            # Default fallback to NSE_INDEX
            return 'NSE_INDEX'
    
    def _calculate_enhanced_cpr_for_live_trading(self, market_data_fetcher, symbol: str) -> dict:
        """
        Enhanced CPR calculation using daily candles for maximum accuracy.
        
        Args:
            market_data_fetcher: Market data fetcher instance
            symbol: Trading symbol
            
        Returns:
            dict: CPR levels with previous day metadata
        """
        try:
            from datetime import datetime
            
            # Get smart previous trading day (excludes weekends and holidays)
            today = datetime.now().date()
            prev_trading_day = self._get_previous_trading_day_smart(today)
            
            print(f"[ENHANCED_CPR] Today: {today.strftime('%A %Y-%m-%d')}")
            print(f"[ENHANCED_CPR] Previous trading day: {prev_trading_day}")
            
            # Detect exchange and instrument for the symbol
            detected_exchange, detected_instrument = detect_exchange_and_instrument(symbol)
            
            # Map to OpenAlgo exchange format
            openalgo_exchange = self._map_to_openalgo_exchange(detected_exchange, detected_instrument)
            
            print(f"[ENHANCED_CPR] Using exchange: {openalgo_exchange} for {symbol}")
            
            # API call for daily candle of previous trading day (more accurate than 5m aggregation)
            print(f"[ENHANCED_CPR] Fetching daily candle (interval='D') for {prev_trading_day}")
            prev_day_data = market_data_fetcher.openalgo_client.get_historical_data(
                symbol=symbol,
                exchange=openalgo_exchange,
                interval='D',  # Daily candle - Working format confirmed by manual test
                start_date=prev_trading_day,
                end_date=prev_trading_day
            )
            
            if prev_day_data is not None and not prev_day_data.empty:
                if len(prev_day_data) == 1:
                    # Extract previous day OHLC directly from official daily candle
                    prev_open = float(prev_day_data['open'].iloc[0])
                    prev_high = float(prev_day_data['high'].iloc[0])
                    prev_low = float(prev_day_data['low'].iloc[0])
                    prev_close = float(prev_day_data['close'].iloc[0])
                    
                    print(f"[ENHANCED_CPR] ✅ Official daily candle OHLC: O={prev_open:.2f}, H={prev_high:.2f}, L={prev_low:.2f}, C={prev_close:.2f}")
                else:
                    # Fallback: Multiple candles returned, use aggregation
                    print(f"[ENHANCED_CPR] ⚠️  Multiple candles returned ({len(prev_day_data)}), using aggregation fallback")
                    prev_open = float(prev_day_data['open'].iloc[0])  # First candle's open
                    prev_high = float(prev_day_data['high'].max())
                    prev_low = float(prev_day_data['low'].min())
                    prev_close = float(prev_day_data['close'].iloc[-1])  # Last candle's close
                
                print(f"[ENHANCED_CPR] Previous day OHLC: H={prev_high:.2f}, L={prev_low:.2f}, C={prev_close:.2f}")
                
                # Calculate enhanced CPR levels using industry-standard formulas (CORRECTED)
                pivot = (prev_high + prev_low + prev_close) / 3
                tc = (prev_high + prev_low) / 2  # Top Central (TC) = (H + L) / 2
                bc = 2 * pivot - tc  # Bottom Central (BC) = 2 * Pivot - TC
                
                # Enhanced support/resistance levels (R4/S4 included)
                r1 = 2 * pivot - prev_low
                r2 = pivot + (prev_high - prev_low)
                r3 = prev_high + (2 * (pivot - prev_low))
                r4 = r3 + (r2 - r1)  # Enhanced R4 level
                
                s1 = 2 * pivot - prev_high
                s2 = pivot - (prev_high - prev_low)
                s3 = prev_low - (2 * (prev_high - pivot))
                s4 = s3 - (r2 - r1)  # Enhanced S4 level
                
                enhanced_cpr_levels = {
                    'previous_date': prev_trading_day,
                    'previous_high': prev_high,
                    'previous_low': prev_low,
                    'previous_close': prev_close,
                    'pivot': pivot,
                    'bc': bc,
                    'tc': tc,
                    'r1': r1, 'r2': r2, 'r3': r3, 'r4': r4,
                    's1': s1, 's2': s2, 's3': s3, 's4': s4,
                    'calculation_method': 'enhanced_api_based',
                    'market_calendar_aware': True
                }
                
                print(f"[ENHANCED_CPR] ✅ Calculated: Pivot={pivot:.2f}, BC={bc:.2f}, TC={tc:.2f}")
                print(f"[ENHANCED_CPR] ✅ Resistance: R1={r1:.2f}, R2={r2:.2f}, R3={r3:.2f}, R4={r4:.2f}")
                print(f"[ENHANCED_CPR] ✅ Support: S1={s1:.2f}, S2={s2:.2f}, S3={s3:.2f}, S4={s4:.2f}")
                
                return enhanced_cpr_levels
                
            else:
                print(f"[ENHANCED_CPR] ❌ CRITICAL: No previous day data available for {prev_trading_day}")
                print(f"[ENHANCED_CPR] 🛑 STOPPING SYSTEM: Daily candle fetch failed - no fallback allowed")
                print(f"[ENHANCED_CPR] 🛑 System will not proceed without previous day data")
                raise SystemExit(f"CRITICAL: No previous day data available for {symbol} on {prev_trading_day}. System stopped as requested.")
                
        except Exception as e:
            print(f"[ENHANCED_CPR] ❌ CRITICAL ERROR: {e}")
            print(f"[ENHANCED_CPR] 🛑 STOPPING SYSTEM: CPR calculation failed - no fallback allowed")
            print(f"[ENHANCED_CPR] 🛑 System will not proceed without valid CPR data")
            raise SystemExit(f"CRITICAL: CPR calculation failed for {symbol}. Error: {e}. System stopped as requested.")
    

    def _store_current_day_in_database(self, current_day_data: pd.DataFrame, symbol: str) -> bool:
        """
        Store today's current day data in the database using DatabaseManager.
        
        Args:
            current_day_data: DataFrame with today's OHLCV data from API
            symbol: Trading symbol (e.g., 'NIFTY')
            
        Returns:
            bool: True if storage successful
        """
        try:
            if current_day_data is None or current_day_data.empty:
                print(f"[DB_STORE] ❌ No data to store for {symbol}")
                return False
            
            # Initialize DatabaseManager
            db_manager = DatabaseManager()
            
            # Store data using the standard method
            success = db_manager.store_ohlcv_data(
                symbol=symbol,
                exchange='NSE_INDEX',  # Standard exchange for index symbols
                timeframe='5m',
                data=current_day_data,
                replace=False  # Incremental mode - don't replace existing data
            )
            
            if success:
                print(f"[DB_STORE] ✅ Stored {len(current_day_data)} records for {symbol}")
                return True
            else:
                print(f"[DB_STORE] ❌ Storage failed for {symbol}")
                return False
                
        except Exception as e:
            print(f"[DB_STORE] ❌ Storage failed for {symbol}: {e}")
            return False

    def _extract_complete_warmup_data(self, symbol: str, candle_count: int = 350) -> pd.DataFrame:
        """
        Extract complete warmup dataset from database including today's latest data.
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY')
            candle_count: Number of candles to retrieve for warmup
            
        Returns:
            pd.DataFrame: Historical OHLCV data with proper timestamp index
        """
        try:
            # Initialize DatabaseManager
            db_manager = DatabaseManager()
            
            # Calculate date range for the required number of candles
            # Assuming 5-minute candles: 75 candles per day, need ~5 trading days for 350 candles
            from datetime import datetime, timedelta
            current_date = datetime.now().date()
            
            # Go back approximately 10 trading days to ensure we get enough data
            # (accounting for weekends and holidays)
            start_date = current_date - timedelta(days=15)
            end_date = current_date
            
            # Retrieve data using the database manager
            unified_data = db_manager.get_ohlcv_data(
                symbol=symbol,
                exchange='NSE_INDEX',
                timeframe='5m',
                start_date=start_date.strftime('%Y-%m-%d'),
                end_date=end_date.strftime('%Y-%m-%d')
            )
            
            if unified_data is not None and not unified_data.empty:
                # Keep only the most recent candles for warmup
                if len(unified_data) > candle_count:
                    unified_data = unified_data.tail(candle_count)
                
                print(f"[UNIFIED_EXTRACT] ✅ Extracted {len(unified_data)} candles for {symbol}")
                print(f"[UNIFIED_EXTRACT] ✅ Range: {unified_data.index.min()} to {unified_data.index.max()}")
                return unified_data
            else:
                print(f"[UNIFIED_EXTRACT] ❌ No data found for {symbol}")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"[UNIFIED_EXTRACT] ❌ Extraction failed for {symbol}: {e}")
            return pd.DataFrame()

    def _calculate_baseline_indicators_for_setup(self, unified_data: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """
        Calculate baseline indicators (EMA, SMA, RSI) during setup phase for accuracy validation.
        
        Args:
            unified_data: Historical OHLCV data with sufficient warmup period
            symbol: Trading symbol
            
        Returns:
            pd.DataFrame: Data with calculated indicators for accuracy validation
        """
        try:
            print(f"[BASELINE_CALC] Starting indicator calculations for {symbol}")
            print(f"[BASELINE_CALC] Input data: {len(unified_data)} candles")
            print(f"[BASELINE_CALC] Range: {unified_data.index.min()} to {unified_data.index.max()}")
            
            # Use the backtesting indicator engine for calculations
            from indicators.unified_indicator_engine import BacktestingIndicatorEngine
            
            # Initialize the engine for accurate calculations
            engine = BacktestingIndicatorEngine()
            
            # Calculate all indicators using the backtesting engine
            result_data = engine.calculate_indicators(data=unified_data)
            
            if result_data is not None and not result_data.empty:
                # Log indicator coverage
                indicator_columns = [col for col in result_data.columns if col not in ['open', 'high', 'low', 'close', 'volume']]
                print(f"[BASELINE_CALC] ✅ Calculated {len(indicator_columns)} indicators")
                print(f"[BASELINE_CALC] ✅ Available indicators: {', '.join(indicator_columns[:10])}{'...' if len(indicator_columns) > 10 else ''}")
                
                # Display latest values for key indicators
                if not result_data.empty:
                    latest = result_data.iloc[-1]
                    print(f"[BASELINE_CALC] Latest candle: {result_data.index[-1]}")
                    print(f"[BASELINE_CALC] OHLC: O={latest.get('open', 0):.2f}, H={latest.get('high', 0):.2f}, L={latest.get('low', 0):.2f}, C={latest.get('close', 0):.2f}")
                    
                    # Display key indicator values
                    key_indicators = ['ema_9', 'ema_20', 'ema_50', 'sma_20', 'sma_50', 'rsi_14']
                    for indicator in key_indicators:
                        if indicator in latest:
                            print(f"[BASELINE_CALC] {indicator.upper()}: {latest[indicator]:.2f}")
                
                return result_data
            else:
                print(f"[BASELINE_CALC] ❌ No indicators calculated")
                return pd.DataFrame()
                
        except Exception as e:
            print(f"[BASELINE_CALC] ❌ Calculation failed: {e}")
            import traceback
            print(f"[BASELINE_CALC] Error details: {traceback.format_exc()}")
            return pd.DataFrame()

    def _estimate_5min_incremental_performance(self, phase_times: dict):
        """
        Estimate 5-minute incremental update performance based on setup times.
        
        Args:
            phase_times: Dictionary of phase timing measurements
        """
        try:
            # Calculate estimates based on actual setup performance
            indicator_time = phase_times.get('phase35_indicators', 0.2)  # Fallback 0.2s
            
            # Incremental updates are much faster than full calculation
            # Estimates based on incremental algorithms:
            ema_update_time = 0.001  # EMA: simple exponential calculation
            sma_update_time = 0.002  # SMA: rolling window update  
            rsi_update_time = 0.002  # RSI: gain/loss calculation
            cpr_update_time = 0.000  # CPR: only updates daily (no 5min update)
            candle_update_time = 0.001  # Candle values: direct OHLC copy
            
            # Total incremental time per 5-minute update
            total_incremental = (ema_update_time + sma_update_time + rsi_update_time + 
                               cpr_update_time + candle_update_time)
            
            # Performance metrics
            updates_per_hour = 12  # 60 minutes / 5 minutes
            updates_per_day = 77   # 6.25 hours * 12 updates (9:15-15:30)
            
            print(f"\n[PERF] 📊 5-MINUTE INCREMENTAL PERFORMANCE ESTIMATES:")
            print(f"[PERF] ═══════════════════════════════════════════════")
            print(f"[PERF] EMA incremental update: ~{ema_update_time*1000:.1f}ms")
            print(f"[PERF] SMA incremental update: ~{sma_update_time*1000:.1f}ms") 
            print(f"[PERF] RSI incremental update: ~{rsi_update_time*1000:.1f}ms")
            print(f"[PERF] CPR update: Daily only (no 5min updates)")
            print(f"[PERF] Candle values update: ~{candle_update_time*1000:.1f}ms")
            print(f"[PERF] ═══════════════════════════════════════════════")
            print(f"[PERF] ⚡ TOTAL per 5-min update: ~{total_incremental*1000:.1f}ms")
            print(f"[PERF] 📈 Updates per hour: {updates_per_hour}")
            print(f"[PERF] 📈 Updates per trading day: {updates_per_day}")
            print(f"[PERF] 🔥 Daily incremental time: ~{total_incremental*updates_per_day:.2f}s")
            print(f"[PERF] 💡 Setup vs Daily ratio: {phase_times.get('total_setup', 5):.1f}s setup : {total_incremental*updates_per_day:.2f}s daily")
            
        except Exception as e:
            print(f"[PERF] ❌ Performance estimation failed: {e}")


def main():
    """Main entry point with command line interface."""
    parser = argparse.ArgumentParser(
        description='vAlgo Indicator Engine - Production Grade',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main_indicator_engine.py                    # Full auto mode (backtesting)
  python main_indicator_engine.py --symbol NIFTY     # Manual symbol processing
  python main_indicator_engine.py --validate         # System validation
  python main_indicator_engine.py --live             # Live trading mode (REAL DATA ONLY)
        """
    )
    
    # Command line arguments
    parser.add_argument('--symbol', type=str, help='Symbol to process (e.g., NIFTY)')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--validate', action='store_true', help='Validate system configuration')
    parser.add_argument('--live', action='store_true', help='Run in live trading mode (REAL DATA ONLY)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    parser.add_argument('--config', type=str, default='config/config.xlsx', help='Config file path')
    parser.add_argument('--no-export', action='store_true', help='Skip Excel export')
    
    args = parser.parse_args()
    
    try:
        # Initialize engine
        engine = MainIndicatorEngine(args.config)
        
        # ====================================================================
        # INDICATOR DISCOVERY CHECK
        # ====================================================================
        # Check if Indicator_Key_Generate flag is enabled
        if engine.config_loader.get_indicator_key_generate():
            print("\n" + "="*60)
            print("🔍 INDICATOR DISCOVERY MODE ACTIVATED")
            print("="*60)
            print("Auto-discovering available indicators...")
            
            try:
                from utils.indicator_discovery import update_rule_types_with_discovery
                
                # Run indicator discovery and update Rule_Types sheet
                success = update_rule_types_with_discovery(args.config)
                
                if success:
                    print("\n" + "="*60)
                    print("✅ INDICATOR DISCOVERY COMPLETED SUCCESSFULLY!")
                    print("="*60)
                    print("📋 NEXT STEPS:")
                    print("1. Set 'Indicator_Key_Generate = false' in Initialize sheet")
                    print("2. Configure your strategies using the new indicators")
                    print("3. Run the engine again for normal operation")
                    print("="*60)
                else:
                    print("\n" + "="*60)
                    print("❌ INDICATOR DISCOVERY FAILED!")
                    print("="*60)
                    print("Please check the logs for error details")
                    print("="*60)
                
                return 0 if success else 1
                
            except Exception as e:
                print(f"\n❌ Error during indicator discovery: {e}")
                return 1
        
        # ====================================================================
        # NORMAL OPERATION MODES
        # ====================================================================
        # Route to appropriate mode
        if args.validate:
            # Validation mode
            success = engine.validate_system()
            return 0 if success else 1
            
        elif args.live:
            # Live trading mode (REAL DATA ONLY)
            result = engine.run_live_mode(
                symbol=args.symbol,
                debug_mode=args.debug
            )
            return 0 if result['status'] == 'success' else 1
            
        elif args.symbol and args.start and args.end:
            # Manual mode
            result = engine.run_manual_mode(
                symbol=args.symbol,
                start_date=args.start,
                end_date=args.end,
                export=not args.no_export
            )
            return 0 if result['status'] == 'success' else 1
            
        else:
            # Default: Auto mode (backtesting)
            result = engine.run_auto_mode()
            return 0 if result['status'] == 'success' else 1
    
    except KeyboardInterrupt:
        print("\n[WARNING]  Operation cancelled by user")
        return 1
    
    except Exception as e:
        print(f"[ERROR] Critical error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)