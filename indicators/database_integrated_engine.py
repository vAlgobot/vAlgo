#!/usr/bin/env python3
"""
Database Integrated Indicator Engine
====================================

Production-grade indicator engine with full database integration and config-driven execution.
Designed for both backtesting and live trading with institutional accuracy.

Key Features:
- Config-driven indicator selection (active indicators only)
- Dynamic parameter loading from Excel configuration
- Timeframe-aware data fetching with proper warmup periods
- Live trading state management architecture
- 100% accurate calculations with industry standards

Author: vAlgo Development Team
Created: June 29, 2025
Version: 1.0.0 (Production)
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import with graceful error handling
try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

try:
    from utils.config_loader import ConfigLoader
    from utils.logger import get_logger
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False
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
    from indicators.warmup_calculator import WarmupCalculator
    from indicators.cpr import CPR
    from indicators.supertrend import Supertrend
    from indicators.ema import EMA
    from indicators.rsi import RSI
    from indicators.vwap import VWAP
    from indicators.sma import SMA
    from indicators.bollinger_bands import BollingerBands
    from indicators.previous_day_levels import PreviousDayLevels
    INDICATORS_AVAILABLE = True
except ImportError as e:
    INDICATORS_AVAILABLE = False


class DatabaseIntegratedEngine:
    """
    Production-grade indicator engine with complete database and configuration integration.
    
    Supports config-driven execution, dynamic parameter loading, and both backtesting
    and live trading modes with institutional-grade accuracy.
    """
    
    def __init__(self, config_path: str = "config/config.xlsx", 
                 db_path: str = "data/valgo_market_data.db",
                 accuracy_level: str = 'good'):
        """
        Initialize database integrated engine.
        
        Args:
            config_path: Path to Excel configuration file
            db_path: Path to DuckDB database file
            accuracy_level: Accuracy level for calculations ('good', 'high', 'max')
        """
        self.config_path = config_path
        self.db_path = db_path
        self.accuracy_level = accuracy_level
        self.logger = get_logger(__name__)
        
        # Core components
        self.db_connection = None
        self.config_loader = None
        self.warmup_calculator = None
        
        # Configuration storage
        self.instruments_config = []
        self.indicators_config = {}
        self.active_indicators = {}
        
        # Live trading state (for future implementation)
        self.indicator_states = {}
        
        # Initialize components
        self._initialize_components()
    
    def _initialize_components(self) -> None:
        """Initialize all engine components."""
        try:
            # Initialize warmup calculator
            if INDICATORS_AVAILABLE:
                self.warmup_calculator = WarmupCalculator(self.accuracy_level)
                self.logger.info("✅ Warmup calculator initialized")
            
            # Load configuration
            self._load_configuration()
            
            # Initialize database
            self._initialize_database()
            
            self.logger.info("✅ Database integrated engine initialized successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Engine initialization failed: {e}")
            raise
    
    def _load_configuration(self) -> None:
        """Load and process Excel configuration."""
        try:
            if not os.path.exists(self.config_path):
                self.logger.warning(f"Config file not found: {self.config_path}, using defaults")
                self._load_default_configuration()
                return
            
            if not CONFIG_AVAILABLE:
                self.logger.warning("ConfigLoader not available, using defaults")
                self._load_default_configuration()
                return
            
            # Load Excel configuration
            self.config_loader = ConfigLoader(self.config_path)
            if not self.config_loader.load_config():
                self.logger.error("Failed to load Excel configuration")
                self._load_default_configuration()
                return
            
            # Load instruments configuration
            self.instruments_config = self.config_loader.get_active_instruments()
            self.logger.info(f"📊 Loaded {len(self.instruments_config)} active instruments")
            
            # Load and process indicators configuration
            self._load_active_indicators_from_excel()
            
            # Log configuration summary
            timeframes = set()
            for instrument in self.instruments_config:
                if 'timeframe' in instrument:
                    # Standardize timeframe format (5min -> 5m)
                    tf = instrument['timeframe']
                    if tf.endswith('min'):
                        tf = tf.replace('min', 'm')
                        instrument['timeframe'] = tf
                    timeframes.add(tf)
            
            self.logger.info(f"⏱️  Configured timeframes: {sorted(timeframes)}")
            self.logger.info(f"📈 Active indicators: {list(self.active_indicators.keys())}")
            
        except Exception as e:
            self.logger.error(f"Configuration loading failed: {e}")
            self._load_default_configuration()
    
    def _load_active_indicators_from_excel(self) -> None:
        """Load only active indicators with their parameters from Excel configuration."""
        try:
            if not self.config_loader:
                self._load_default_indicators()
                return
            
            # Get all indicator configurations
            all_indicators = self.config_loader.get_indicator_config()
            self.active_indicators = {}
            
            for indicator_config in all_indicators:
                indicator_name = indicator_config.get('indicator', '')
                status = indicator_config.get('status', 'inactive').lower()
                
                # Only process active indicators
                if status == 'active' and indicator_name:
                    # Parse parameters
                    parameters = self._parse_indicator_parameters(
                        indicator_name, 
                        indicator_config.get('parameters', [])
                    )
                    
                    self.active_indicators[indicator_name] = parameters
                    self.logger.info(f"📈 Active indicator: {indicator_name} with params: {parameters}")
            
            if not self.active_indicators:
                self.logger.warning("No active indicators found in config, using defaults")
                self._load_default_indicators()
            
        except Exception as e:
            self.logger.error(f"Failed to load active indicators: {e}")
            self._load_default_indicators()
    
    def _parse_indicator_parameters(self, indicator_name: str, parameters: List) -> Dict[str, Any]:
        """Parse indicator parameters from Excel configuration."""
        try:
            # Default parameters for each indicator
            default_params = {
                'EMA': {'periods': [9, 21, 50, 200]},
                'SMA': {'periods': [9, 21, 50, 200]},
                'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
                'BollingerBands': {'period': 20, 'std_dev': 2.0},
                'Supertrend': {'period': 10, 'multiplier': 3.0},
                'VWAP': {'session_type': 'daily', 'std_dev_multiplier': 1.0},
                'CPR': {'timeframe': 'daily'},
                'PreviousDayLevels': {'timeframe': 'daily'}
            }
            
            # Start with defaults
            params = default_params.get(indicator_name, {}).copy()
            
            # Override with Excel parameters if provided
            if parameters:
                if indicator_name in ['EMA', 'SMA']:
                    # For EMA/SMA, parameters are periods like [9, 21, 50]
                    if isinstance(parameters, list) and len(parameters) > 0:
                        params['periods'] = parameters
                elif indicator_name == 'RSI':
                    # For RSI, first parameter is period
                    if isinstance(parameters, list) and len(parameters) > 0:
                        params['period'] = int(parameters[0])
                elif indicator_name == 'BollingerBands':
                    # For BB, first param is period, second is std_dev
                    if isinstance(parameters, list):
                        if len(parameters) > 0:
                            params['period'] = int(parameters[0])
                        if len(parameters) > 1:
                            params['std_dev'] = float(parameters[1])
                elif indicator_name == 'Supertrend':
                    # For Supertrend, first param is period, second is multiplier
                    if isinstance(parameters, list):
                        if len(parameters) > 0:
                            params['period'] = int(parameters[0])
                        if len(parameters) > 1:
                            params['multiplier'] = float(parameters[1])
            
            return params
            
        except Exception as e:
            self.logger.warning(f"Failed to parse parameters for {indicator_name}: {e}")
            return default_params.get(indicator_name, {})
    
    def _load_default_configuration(self) -> None:
        """Load default configuration when Excel config is unavailable."""
        self.instruments_config = [
            {
                'symbol': 'NIFTY',
                'exchange': 'NSE_INDEX',
                'timeframe': '5m',  # Use 5m format
                'status': 'active'
            }
        ]
        
        self._load_default_indicators()
        self.logger.info("✅ Using default configuration")
    
    def _load_default_indicators(self) -> None:
        """Load default active indicators."""
        self.active_indicators = {
            'CPR': {'timeframe': 'daily'},
            'Supertrend': {'period': 10, 'multiplier': 3.0},
            'EMA': {'periods': [9, 21, 50, 200]},
            'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
            'VWAP': {'session_type': 'daily', 'std_dev_multiplier': 1.0},
            'SMA': {'periods': [9, 21, 50, 200]},
            'BollingerBands': {'period': 20, 'std_dev': 2.0},
            'PreviousDayLevels': {'timeframe': 'daily'}
        }
    
    def _initialize_database(self) -> None:
        """Initialize database connection."""
        try:
            if not DUCKDB_AVAILABLE:
                self.logger.error("DuckDB not available")
                return
            
            if not os.path.exists(self.db_path):
                self.logger.error(f"Database file not found: {self.db_path}")
                return
            
            # Test connection
            self.db_connection = duckdb.connect(self.db_path)
            
            # Verify database structure
            tables = self.db_connection.execute("SHOW TABLES").fetchall()
            table_names = [table[0] for table in tables]
            
            if 'ohlcv_data' not in table_names:
                self.logger.error("OHLCV data table not found in database")
                return
            
            # Get database statistics
            count = self.db_connection.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()[0]
            symbols = self.db_connection.execute("SELECT DISTINCT symbol FROM ohlcv_data").fetchall()
            timeframes = self.db_connection.execute("SELECT DISTINCT timeframe FROM ohlcv_data").fetchall()
            
            self.logger.info(f"✅ Database connected: {count:,} records")
            self.logger.info(f"📊 Available symbols: {[s[0] for s in symbols]}")
            self.logger.info(f"⏱️  Available timeframes: {[t[0] for t in timeframes]}")
            
        except Exception as e:
            self.logger.error(f"Database initialization failed: {e}")
            self.db_connection = None
    
    def validate_database_connection(self) -> bool:
        """Validate database connection and data availability."""
        try:
            if not self.db_connection:
                return False
            
            # Test query
            result = self.db_connection.execute("SELECT COUNT(*) FROM ohlcv_data").fetchone()
            return result[0] > 0
            
        except Exception:
            return False
    
    def get_available_symbols(self) -> List[str]:
        """Get list of available symbols from configuration."""
        symbols = []
        for instrument in self.instruments_config:
            if instrument.get('status', '').lower() == 'active':
                symbols.append(instrument['symbol'])
        return symbols
    
    def get_symbol_timeframe(self, symbol: str) -> str:
        """Get timeframe for a specific symbol from configuration."""
        for instrument in self.instruments_config:
            if instrument.get('symbol') == symbol:
                return instrument.get('timeframe', '5m')
        return '5m'  # Default timeframe
    
    def get_date_range_from_config(self) -> Tuple[str, str]:
        """Get date range from Excel Initialize sheet configuration."""
        try:
            if self.config_loader:
                init_config = self.config_loader.get_initialize_config()
                start_date = init_config.get('start_date', '2025-05-01')
                end_date = init_config.get('end_date', '2025-05-31')
                
                # Convert datetime objects to strings if needed
                if isinstance(start_date, datetime):
                    start_date = start_date.strftime('%Y-%m-%d')
                if isinstance(end_date, datetime):
                    end_date = end_date.strftime('%Y-%m-%d')
                
                return str(start_date), str(end_date)
        except Exception:
            pass
        
        # Default to May 2025 (as requested for testing)
        return '2025-05-01', '2025-05-31'
    
    def _fetch_data_with_warmup(self, symbol: str, start_date: str, end_date: str, 
                               timeframe: str) -> pd.DataFrame:
        """Fetch data from database with proper warmup period for accurate calculations."""
        try:
            if not self.db_connection:
                raise Exception("Database connection not available")
            
            # Calculate required warmup period
            max_warmup = 0
            for indicator_name in self.active_indicators.keys():
                if indicator_name == 'EMA':
                    # For EMA, get the largest period and calculate warmup
                    periods = self.active_indicators[indicator_name].get('periods', [200])
                    max_period = max(periods)
                    warmup = max_period * 4  # 4x period for 95% accuracy
                    max_warmup = max(max_warmup, warmup)
                elif indicator_name == 'SMA':
                    # For SMA, need the full period
                    periods = self.active_indicators[indicator_name].get('periods', [200])
                    max_period = max(periods)
                    max_warmup = max(max_warmup, max_period)
                elif indicator_name == 'RSI':
                    # For RSI, need 4x period for accurate calculation
                    period = self.active_indicators[indicator_name].get('period', 14)
                    warmup = period * 4
                    max_warmup = max(max_warmup, warmup)
                elif indicator_name == 'BollingerBands':
                    # For BB, need the period
                    period = self.active_indicators[indicator_name].get('period', 20)
                    max_warmup = max(max_warmup, period)
                elif indicator_name == 'Supertrend':
                    # For Supertrend, need the period
                    period = self.active_indicators[indicator_name].get('period', 10)
                    max_warmup = max(max_warmup, period)
            
            # Add buffer for safety
            warmup_periods = max_warmup + 50
            
            # Calculate warmup start date
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            
            # Estimate days needed based on timeframe
            if timeframe == '1m':
                # 1-min: ~390 candles per day
                warmup_days = (warmup_periods // 390) + 1
            elif timeframe == '5m':
                # 5-min: ~78 candles per day
                warmup_days = (warmup_periods // 78) + 1
            elif timeframe == 'd':
                # Daily: 1 candle per day
                warmup_days = warmup_periods
            else:
                # Default estimation
                warmup_days = (warmup_periods // 100) + 1
            
            # Ensure minimum warmup
            warmup_days = max(warmup_days, 30)  # At least 30 days
            
            warmup_start = start_dt - timedelta(days=warmup_days)
            warmup_start_str = warmup_start.strftime('%Y-%m-%d')
            
            self.logger.info(f"📊 Fetching {symbol} data with warmup:")
            self.logger.info(f"   Target period: {start_date} to {end_date}")
            self.logger.info(f"   Warmup period: {warmup_start_str} to {start_date}")
            self.logger.info(f"   Total warmup: {warmup_periods} candles (~{warmup_days} days)")
            
            # Fetch data from database
            query = """
            SELECT timestamp, open, high, low, close, volume
            FROM ohlcv_data
            WHERE symbol = ? AND timeframe = ?
              AND timestamp >= ? AND timestamp <= ?
            ORDER BY timestamp
            """
            
            data = self.db_connection.execute(
                query, 
                [symbol, timeframe, warmup_start_str, end_date]
            ).fetchall()
            
            if not data:
                raise Exception(f"No data found for {symbol} {timeframe}")
            
            # Convert to DataFrame
            df = pd.DataFrame(data, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Keep timestamp as both column and index for indicator compatibility
            df = df.set_index('timestamp', drop=False)
            
            # Ensure numeric types
            for col in ['open', 'high', 'low', 'close', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            self.logger.info(f"✅ Fetched {len(df)} records with warmup")
            return df
            
        except Exception as e:
            self.logger.error(f"Data fetch failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def _calculate_indicators_on_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Calculate all active indicators on the provided data."""
        try:
            if data.empty or not INDICATORS_AVAILABLE:
                return data
            
            result_df = data.copy()
            
            # Calculate each active indicator
            for indicator_name, params in self.active_indicators.items():
                try:
                    self.logger.info(f"📈 Calculating {indicator_name}...")
                    
                    if indicator_name == 'CPR':
                        indicator = CPR(**params)
                        result_df = indicator.calculate(result_df)
                        
                    elif indicator_name == 'Supertrend':
                        indicator = Supertrend(**params)
                        result_df = indicator.calculate(result_df)
                        
                    elif indicator_name == 'EMA':
                        indicator = EMA(**params)
                        result_df = indicator.calculate(result_df)
                        
                    elif indicator_name == 'RSI':
                        indicator = RSI(**params)
                        result_df = indicator.calculate(result_df)
                        
                    elif indicator_name == 'VWAP':
                        indicator = VWAP(**params)
                        result_df = indicator.calculate(result_df)
                        
                    elif indicator_name == 'SMA':
                        indicator = SMA(**params)
                        result_df = indicator.calculate(result_df)
                        
                    elif indicator_name == 'BollingerBands':
                        indicator = BollingerBands(**params)
                        result_df = indicator.calculate(result_df)
                        
                    elif indicator_name == 'PreviousDayLevels':
                        indicator = PreviousDayLevels(**params)
                        result_df = indicator.calculate(result_df)
                    
                    self.logger.info(f"✅ {indicator_name} calculated successfully")
                    
                except Exception as e:
                    self.logger.error(f"❌ {indicator_name} calculation failed: {e}")
                    continue
            
            return result_df
            
        except Exception as e:
            self.logger.error(f"Indicator calculation failed: {e}")
            return data
    
    def calculate_indicators_for_symbol(self, symbol: str, start_date: str, 
                                      end_date: str) -> pd.DataFrame:
        """
        Calculate indicators for a specific symbol with proper warmup.
        
        Args:
            symbol: Trading symbol
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            DataFrame with OHLCV data and calculated indicators
        """
        try:
            # Get timeframe for symbol
            timeframe = self.get_symbol_timeframe(symbol)
            
            # Fetch data with warmup
            data = self._fetch_data_with_warmup(symbol, start_date, end_date, timeframe)
            if data.empty:
                self.logger.warning(f"No data available for {symbol}")
                return pd.DataFrame()
            
            # Calculate indicators
            result = self._calculate_indicators_on_data(data)
            
            # Filter to target date range (remove warmup period)
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')
            
            mask = (result.index >= start_dt) & (result.index <= end_dt)
            final_result = result[mask].copy()
            
            self.logger.info(f"✅ {symbol}: {len(final_result)} records in target period")
            return final_result
            
        except Exception as e:
            self.logger.error(f"Symbol calculation failed for {symbol}: {e}")
            return pd.DataFrame()
    
    def calculate_indicators_for_all_instruments(self, start_date: str, 
                                               end_date: str) -> Dict[str, pd.DataFrame]:
        """
        Calculate indicators for all active instruments.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            Dictionary mapping symbol names to DataFrames with results
        """
        results = {}
        symbols = self.get_available_symbols()
        
        self.logger.info(f"🔄 Processing {len(symbols)} symbols: {symbols}")
        
        for symbol in symbols:
            try:
                result = self.calculate_indicators_for_symbol(symbol, start_date, end_date)
                if not result.empty:
                    results[symbol] = result
                    self.logger.info(f"✅ {symbol}: {len(result)} records processed")
                else:
                    self.logger.warning(f"⚠️  {symbol}: No data processed")
                    
            except Exception as e:
                self.logger.error(f"❌ {symbol}: Processing failed - {e}")
                continue
        
        self.logger.info(f"🎯 Batch processing complete: {len(results)} symbols successful")
        return results
    
    def export_results_to_excel(self, results: Dict[str, pd.DataFrame]) -> Optional[str]:
        """
        Export calculation results to Excel validation file.
        
        Args:
            results: Dictionary of symbol DataFrames
            
        Returns:
            Path to created Excel file
        """
        try:
            if not results:
                self.logger.warning("No results to export")
                return None
            
            # Create output directory
            output_dir = Path("outputs/indicators")
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Generate filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"indicator_engine_validation_{timestamp}.xlsx"
            output_path = output_dir / filename
            
            self.logger.info(f"📁 Exporting to: {filename}")
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for symbol, df in results.items():
                    indicator_count = len(df.columns) - 5  # Exclude OHLCV
                    summary_data.append({
                        'Symbol': symbol,
                        'Records': len(df),
                        'Start_Date': df.index.min(),
                        'End_Date': df.index.max(),
                        'Price_Low': df['low'].min(),
                        'Price_High': df['high'].max(),
                        'Indicators_Count': indicator_count,
                        'Active_Indicators': ', '.join(self.active_indicators.keys())
                    })
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual symbol sheets (sample data)
                for symbol, df in results.items():
                    # Round numeric values for better readability
                    sample_df = df.head(1000).round(4)
                    
                    # Reset index to include timestamp as column
                    sample_df = sample_df.reset_index()
                    
                    # Create sheet name (Excel limit: 31 characters)
                    sheet_name = symbol[:31]
                    sample_df.to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Configuration sheet
                config_data = []
                config_data.append(['Configuration', 'Value'])
                config_data.append(['Config Path', self.config_path])
                config_data.append(['Database Path', self.db_path])
                config_data.append(['Accuracy Level', self.accuracy_level])
                config_data.append(['Active Indicators', ', '.join(self.active_indicators.keys())])
                
                for symbol in self.get_available_symbols():
                    timeframe = self.get_symbol_timeframe(symbol)
                    config_data.append([f'{symbol} Timeframe', timeframe])
                
                config_df = pd.DataFrame(config_data)
                config_df.to_excel(writer, sheet_name='Configuration', header=False, index=False)
            
            self.logger.info(f"✅ Excel validation file created: {output_path}")
            return str(output_path)
            
        except Exception as e:
            self.logger.error(f"Excel export failed: {e}")
            return None
    
    def get_indicator_state_for_live_trading(self, symbol: str) -> Dict[str, Any]:
        """
        Get current indicator state for live trading updates.
        This will be used for incremental indicator updates.
        """
        # Future implementation for live trading
        # Will store last calculated values and states for each indicator
        return self.indicator_states.get(symbol, {})
    
    def update_indicators_with_new_data(self, symbol: str, new_close: float, 
                                      new_timestamp: datetime) -> Dict[str, Any]:
        """
        Update indicators incrementally with new data point.
        This is the live trading mode - efficient single-value updates.
        """
        # Future implementation for live trading
        # Will update indicators incrementally without full recalculation
        pass
    
    def __str__(self) -> str:
        """String representation of the engine."""
        return (f"DatabaseIntegratedEngine("
               f"symbols={len(self.instruments_config)}, "
               f"indicators={len(self.active_indicators)}, "
               f"db_connected={self.db_connection is not None})")