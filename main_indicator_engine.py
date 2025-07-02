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
    from utils.logger import get_logger
    from utils.config_loader import ConfigLoader
    from utils.live_excel_reporter import LiveExcelReporter
    from utils.performance_tracker import PerformanceTracker
    from utils.market_data_fetcher import MarketDataFetcher
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


class MarketSessionManager:
    """Market session monitoring and management."""
    
    def __init__(self):
        try:
            self.logger = get_logger(__name__)
        except:
            self.logger = logging.getLogger(__name__)
        
        self.market_open_time = dt_time(9, 15)  # 9:15 AM
        self.market_close_time = dt_time(15, 30)  # 3:30 PM  
        self.final_export_time = dt_time(15, 0)  # 3:00 PM
        
        self.session_active = False
        self.export_triggered = False
    
    def is_market_open(self) -> bool:
        """Check if market is currently open."""
        now = datetime.now().time()
        return self.market_open_time <= now <= self.market_close_time
    
    def should_trigger_final_export(self) -> bool:
        """Check if 3:00 PM final export should be triggered."""
        now = datetime.now().time()
        return now >= self.final_export_time and not self.export_triggered


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
        """Initialize the unified indicator engine."""
        try:
            if COMPONENTS_AVAILABLE:
                self.engine = UnifiedIndicatorEngine(self.config_path)
                self.logger.info("[OK] Main indicator engine initialized successfully")
                print(f"[TARGET] Engine mode: {self.engine.get_mode().upper()}")
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
            
            # Load configuration with force reload to ensure fresh data
            self.config_loader = ConfigLoader(self.config_path)
            if not self.config_loader.force_reload_config():
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
                    from utils.config_loader import ConfigLoader
                    config_loader = ConfigLoader(self.config_path)
                    # Force reload to ensure we get the latest Excel data
                    if config_loader.force_reload_config():
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
            if self.engine and hasattr(self.engine, 'export_results_to_excel'):
                excel_path = self.engine.export_results_to_excel(results)
                if excel_path:
                    print(f"[OK] Validation report generated: {excel_path}")
                    return excel_path
            
            # If export fails, return None (no fallback to avoid duplicate reports)
            print("[WARNING]  Excel export not available")
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
                if hasattr(self.engine, 'export_results_to_excel'):
                    report_path = self.engine.export_results_to_excel({symbol: result})
                elif hasattr(self.engine, 'engine') and hasattr(self.engine.engine, 'export_results_to_excel'):
                    report_path = self.engine.engine.export_results_to_excel({symbol: result})
                
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
            # Initialize live components
            csv_manager = LiveCSVManager()
            session_manager = MarketSessionManager()
            schedulers = {}
            
            # Initialize market data fetcher for real data
            market_data_fetcher = MarketDataFetcher()
            print("[DATA] Market data fetcher initialized")
            
            # Load configuration for indicators and symbols
            try:
                config_loader = ConfigLoader(self.config_path)
                active_instruments = config_loader.get_active_instruments()
                indicator_config = config_loader.get_indicator_config()
                
                # Get symbols from config or use provided symbol
                if symbol:
                    symbols = [symbol]
                else:
                    symbols = [inst['Symbol'] for inst in active_instruments if inst['Status'] == 'Active']
                    if not symbols:
                        symbols = ['NIFTY']  # Fallback
                
                # Get timeframes from config
                timeframes = list(set([inst['Timeframe'] for inst in active_instruments if inst['Status'] == 'Active']))
                if not timeframes:
                    timeframes = ['5m']  # Fallback
                
                print(f"[LIVE] Processing symbols: {symbols}")
                print(f"[TIME] Timeframes: {timeframes}")
                print(f"[CONFIG] Loaded {len(indicator_config)} indicator configurations")
                
            except Exception as e:
                print(f"[WARNING] Config loading failed, using defaults: {e}")
                symbols = [symbol] if symbol else ['NIFTY']
                timeframes = ['5m']
                indicator_config = []
            
            # Initialize live indicator engine with config
            live_engine = None
            if COMPONENTS_AVAILABLE:
                try:
                    live_engine = LiveTradingIndicatorEngine(self.config_path)
                    print("[OK] Live indicator engine initialized with config")
                except Exception as e:
                    print(f"[WARNING] Live engine not available: {e}")
            
            # Generate comprehensive indicator column list from config
            indicator_columns = self._generate_indicator_columns_from_config(indicator_config)
            print(f"[INDICATORS] Generated {len(indicator_columns)} indicator columns")
            
            # Initialize performance tracker
            performance_tracker = PerformanceTracker()
            performance_tracker.start_automatic_reporting()
            print("[PERF] Performance tracking started")
            
            # Initialize Excel reporter
            excel_reporter = LiveExcelReporter()
            excel_reporter.initialize_session(symbols, indicator_columns)
            print("[EXCEL] Live Excel reporter initialized")
            
            # Setup CSV files for each symbol
            for sym in symbols:
                csv_path = csv_manager.initialize_csv_file(sym, indicator_columns)
                print(f"[CSV] Initialized CSV for {sym}: {csv_path}")
            
            # Setup schedulers for each timeframe
            def create_update_callback(symbol, timeframe):
                def update_callback(timestamp):
                    self._process_live_update(symbol, timeframe, timestamp, 
                                            csv_manager, live_engine,
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
                self._shutdown_live_mode(schedulers, csv_manager, symbols)
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
                    
                    # Check market session
                    if session_manager.is_market_open():
                        if not session_manager.session_active:
                            print(f"[MARKET] Market opened at {current_time.strftime('%H:%M:%S')}")
                            session_manager.session_active = True
                    else:
                        if session_manager.session_active:
                            print(f"[MARKET] Market closed at {current_time.strftime('%H:%M:%S')}")
                            session_manager.session_active = False
                    
                    # Check for 3:00 PM export trigger
                    if session_manager.should_trigger_final_export():
                        print("[EXPORT] Triggering 3:00 PM final export...")
                        
                        # Export CSV files
                        for sym in symbols:
                            final_path = csv_manager.export_final_session_csv(sym)
                            if final_path:
                                print(f"[OK] Final CSV export for {sym}: {final_path}")
                        
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
                    
                    # Sleep and continue monitoring
                    time.sleep(30)  # Check every 30 seconds
                    
            except KeyboardInterrupt:
                print("\n[STOP] Live trading session stopped by user")
            
            # Cleanup and final export
            self._shutdown_live_mode(schedulers, csv_manager, symbols, performance_tracker, excel_reporter)
            
            # Calculate session statistics
            end_time = datetime.now()
            session_duration = end_time - start_time
            
            return {
                'status': 'success',
                'mode': 'live',
                'symbols_processed': len(symbols),
                'session_duration': session_duration.total_seconds(),
                'schedulers_active': len(schedulers),
                'csv_files_created': len(csv_manager.csv_files),
                'market_session_active': session_manager.session_active,
                'performance_tracking': performance_tracker is not None,
                'excel_reporting': excel_reporter is not None,
                'indicator_columns': len(indicator_columns),
                'total_updates': sum(len(data) for data in csv_manager.current_session_data.values()),
                'avg_processing_time': performance_tracker.get_performance_summary()['avg_processing_time'] if performance_tracker else 0.0
            }
            
        except Exception as e:
            error_msg = f"Live trading mode failed: {e}"
            self.logger.error(error_msg)
            return self._create_error_result(error_msg)
    
    def _process_live_update(self, symbol: str, timeframe: str, timestamp: datetime,
                           csv_manager: LiveCSVManager, live_engine, 
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
                    # CRITICAL ERROR: Real data fetch failed
                    error_msg = f"[CRITICAL] Real data fetch failed for {symbol}. No fallback available in production mode."
                    self.logger.error(error_msg)
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
            
            # Update CSV with live data
            success = csv_manager.update_csv_with_live_data(
                symbol, timestamp, ohlcv_data, indicator_values
            )
            
            # Update Excel reporter
            if excel_reporter:
                processing_time = time.time() - perf_context['start_time'] if perf_context else 0.0
                excel_reporter.add_live_update(
                    symbol, timestamp, ohlcv_data, indicator_values, processing_time
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
                self.logger.info(f"[LIVE] {symbol} {timeframe} update at {timestamp.strftime('%H:%M:%S')} - {indicator_count} indicators calculated")
            else:
                self.logger.warning(f"[WARNING] Failed to update CSV for {symbol}")
                
        except Exception as e:
            error_msg = f"Live update failed for {symbol}: {e}"
            self.logger.error(error_msg)
            
            # Record failed performance metric
            if performance_tracker and perf_context:
                performance_tracker.end_operation(
                    perf_context, 
                    indicator_count=0,
                    data_points=0,
                    success=False,
                    error_message=str(e)
                )
    
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
    
    def _shutdown_live_mode(self, schedulers: Dict, csv_manager: LiveCSVManager, symbols: List[str],
                          performance_tracker=None, excel_reporter=None):
        """Gracefully shutdown live trading mode with comprehensive reporting."""
        print("\n[SHUTDOWN] Shutting down live trading session...")
        
        # Stop all schedulers
        for scheduler_key, scheduler in schedulers.items():
            scheduler.stop_scheduler()
            print(f"[STOP] Stopped scheduler: {scheduler_key}")
        
        # Export final session data
        print("[EXPORT] Exporting final session data...")
        
        # Export CSV files
        for symbol in symbols:
            final_path = csv_manager.export_final_session_csv(symbol)
            if final_path:
                print(f"[OK] Final CSV exported for {symbol}: {final_path}")
        
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
        
        print("[OK] Live trading session shutdown complete with comprehensive reporting")
    
    # REMOVED: _generate_realistic_fallback() - Production systems use real market data only


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