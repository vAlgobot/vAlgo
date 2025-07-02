#!/usr/bin/env python3
"""
Unified Indicator Calculator
============================

Single entry point for all indicator calculations with high accuracy and database integration.
Provides unified interface for backtesting and live trading with automatic warmup handling.

Core Functions:
1. calculate_indicators_backtest() - Historical data with database integration
2. calculate_indicators_live() - Real-time data processing
3. get_latest_indicator_values() - Cached latest values
4. export_indicators_to_excel() - Excel export with formatting

Author: vAlgo Development Team
Created: June 29, 2025
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Union, Optional, Any, Tuple
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    # Import vAlgo components
    from data_manager.database import DatabaseManager
    from utils.config_loader import ConfigLoader
    from utils.logger import get_logger
    LOGGER_AVAILABLE = True
except ImportError:
    # Create fallback logger
    import logging
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger
    LOGGER_AVAILABLE = False

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
    print(f"Warning: Some imports failed: {e}")
    print("This is expected if running outside the full vAlgo environment")
    INDICATORS_AVAILABLE = False


def create_local_indicator_factory(config_data: dict) -> dict:
    """
    Create indicator instances from configuration (local implementation).
    
    Args:
        config_data: Dictionary with indicator configurations
        
    Returns:
        Dictionary of initialized indicator instances
    """
    indicators = {}
    
    if not INDICATORS_AVAILABLE:
        print("Warning: Indicators not available, returning empty factory")
        return indicators
    
    indicator_classes = {
        'CPR': CPR,
        'Supertrend': Supertrend,
        'EMA': EMA,
        'RSI': RSI,
        'VWAP': VWAP,
        'SMA': SMA,
        'BollingerBands': BollingerBands,
        'PreviousDayLevels': PreviousDayLevels
    }
    
    for indicator_name, params in config_data.items():
        if indicator_name in indicator_classes:
            try:
                indicators[indicator_name] = indicator_classes[indicator_name](**params)
            except Exception as e:
                print(f"Warning: Failed to create {indicator_name} indicator: {e}")
    
    return indicators


class UnifiedIndicatorCalculator:
    """
    Unified calculator for all indicator operations.
    
    Provides high-level interface for backtesting and live trading
    with automatic warmup period handling and database integration.
    """
    
    def __init__(self, config_path: str = "config/config.xlsx", accuracy_level: str = 'good'):
        """
        Initialize unified calculator.
        
        Args:
            config_path: Path to Excel configuration file
            accuracy_level: Accuracy level for warmup calculations
        """
        self.config_path = config_path
        self.accuracy_level = accuracy_level
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.warmup_calculator = WarmupCalculator(accuracy_level)
        self.db_manager = None
        self.config_loader = None
        self.latest_values_cache = {}
        
        # Load configuration if available
        self._load_configuration()
    
    def _load_configuration(self) -> None:
        """Load configuration from Excel file."""
        try:
            if os.path.exists(self.config_path):
                self.config_loader = ConfigLoader(self.config_path)
                self.logger.info(f"Configuration loaded from {self.config_path}")
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
    
    def _get_indicator_configs(self) -> Dict:
        """
        Get indicator configurations from Excel or use defaults.
        
        Returns:
            Dictionary of indicator configurations
        """
        if self.config_loader:
            try:
                # Try to get from Excel configuration
                indicator_configs = self.config_loader.get_indicator_config()
                if indicator_configs:
                    return indicator_configs
            except Exception as e:
                self.logger.warning(f"Failed to load indicator config from Excel: {e}")
        
        # Default configuration
        default_configs = {
            'CPR': {'timeframe': 'daily'},
            'Supertrend': {'period': 10, 'multiplier': 3.0},
            'EMA': {'periods': [9, 21, 50, 200]},
            'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
            'VWAP': {'session_type': 'daily', 'std_dev_multiplier': 1.0},
            'SMA': {'periods': [9, 21, 50, 200]},
            'BollingerBands': {'period': 20, 'std_dev': 2.0},
            'PreviousDayLevels': {'timeframe': 'daily'}
        }
        
        self.logger.info("Using default indicator configurations")
        return default_configs
    
    def _fetch_data_with_fallback(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, timeframe: str = '1min') -> pd.DataFrame:
        """
        Fetch data from database with CSV fallback.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe
            
        Returns:
            OHLCV DataFrame with timestamp index
        """
        data = None
        
        # Try database first
        try:
            if not self.db_manager:
                self.db_manager = DatabaseManager()
            
            # Convert to string format for database query
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            data = self.db_manager.get_ohlcv_data(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                timeframe=timeframe
            )
            
            if data is not None and not data.empty:
                self.logger.info(f"Fetched {len(data)} records from database for {symbol}")
                
                # Keep timestamp as column for indicators that need it
                # Also ensure index is datetime for filtering
                if 'timestamp' in data.columns:
                    data['timestamp'] = pd.to_datetime(data['timestamp'])
                    data.set_index('timestamp', inplace=True, drop=False)  # Keep both index and column
                
                return data
            
        except Exception as e:
            self.logger.warning(f"Database fetch failed for {symbol}: {e}")
        
        # Fallback to CSV files
        try:
            csv_files = [
                f"data/{symbol}_{timeframe}.csv",
                f"test/{symbol}_{timeframe}.csv", 
                f"outputs/data_exports/{symbol}_{timeframe}.csv",
                f"data/{symbol}.csv",
                f"test/{symbol}.csv"
            ]
            
            for csv_file in csv_files:
                if os.path.exists(csv_file):
                    self.logger.info(f"Loading data from CSV: {csv_file}")
                    
                    data = pd.read_csv(csv_file)
                    
                    # Ensure required columns exist
                    required_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
                    if all(col in data.columns for col in required_cols):
                        # Convert timestamp and set as index (keep both)
                        data['timestamp'] = pd.to_datetime(data['timestamp'])
                        data.set_index('timestamp', inplace=True, drop=False)  # Keep both index and column
                        
                        # Filter by date range
                        data = data[(data.index >= start_date) & (data.index <= end_date)]
                        
                        if not data.empty:
                            self.logger.info(f"Loaded {len(data)} records from CSV for {symbol}")
                            return data
            
        except Exception as e:
            self.logger.warning(f"CSV fallback failed: {e}")
        
        # No data available
        raise Exception(f"No data available for {symbol} from {start_date} to {end_date}")
    
    def calculate_indicators_backtest(self, symbol: str, start_date: Union[str, datetime], 
                                    end_date: Union[str, datetime], 
                                    timeframe: str = '1min',
                                    accuracy_level: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate all indicators for backtesting with automatic warmup handling.
        
        Args:
            symbol: Trading symbol (e.g., 'NIFTY')
            start_date: Start date for backtest
            end_date: End date for backtest
            timeframe: Data timeframe ('1min', '5min', '1day', etc.)
            accuracy_level: Override accuracy level for this calculation
            
        Returns:
            DataFrame with OHLCV data and all indicators
        """
        try:
            # Use provided accuracy level or instance default
            accuracy_level = accuracy_level or self.accuracy_level
            warmup_calc = WarmupCalculator(accuracy_level)
            
            # Convert dates
            if isinstance(start_date, str):
                start_date = pd.to_datetime(start_date)
            if isinstance(end_date, str):
                end_date = pd.to_datetime(end_date)
            
            self.logger.info(f"Starting backtest calculation for {symbol} from {start_date.date()} to {end_date.date()}")
            self.logger.info(f"Accuracy level: {accuracy_level} ({warmup_calc.get_accuracy_info()['accuracy']})")
            
            # Get indicator configurations
            indicator_configs = self._get_indicator_configs()
            self.logger.info(f"Configured indicators: {list(indicator_configs.keys())}")
            
            # Calculate required warmup period
            max_warmup_candles = warmup_calc.calculate_max_warmup_needed(indicator_configs)
            self.logger.info(f"Maximum warmup required: {max_warmup_candles} candles")
            
            # Calculate extended start date
            extended_start = warmup_calc.calculate_extended_start_date(
                start_date, max_warmup_candles, timeframe
            )
            self.logger.info(f"Extended fetch period: {extended_start.date()} to {end_date.date()}")
            
            # Fetch extended data
            extended_data = self._fetch_data_with_fallback(
                symbol, extended_start, end_date, timeframe
            )
            
            # Validate data sufficiency
            is_sufficient, message = warmup_calc.validate_data_sufficiency(
                len(extended_data), max_warmup_candles
            )
            self.logger.info(f"Data sufficiency: {message}")
            
            if not is_sufficient:
                self.logger.warning("Proceeding with available data, but accuracy may be reduced")
            
            # Calculate all indicators on extended dataset
            self.logger.info("Calculating indicators...")
            indicators = create_local_indicator_factory(indicator_configs)
            
            result_data = extended_data.copy()
            calculation_summary = {}
            
            for name, indicator in indicators.items():
                try:
                    before_cols = len(result_data.columns)
                    
                    # Prepare data based on indicator requirements
                    if name in ['CPR', 'VWAP', 'PreviousDayLevels']:
                        # These indicators need timestamp as column, not index
                        indicator_data = result_data.reset_index()
                        if 'timestamp' not in indicator_data.columns:
                            indicator_data['timestamp'] = indicator_data.index
                        calculated_data = indicator.calculate(indicator_data)
                        # Restore index and merge results
                        calculated_data.set_index('timestamp', inplace=True)
                        # Copy new columns to result_data
                        new_columns = [col for col in calculated_data.columns if col not in result_data.columns]
                        for col in new_columns:
                            result_data[col] = calculated_data[col]
                    else:
                        # Other indicators work with index-based data
                        result_data = indicator.calculate(result_data)
                    
                    after_cols = len(result_data.columns)
                    new_cols = after_cols - before_cols
                    calculation_summary[name] = {
                        'status': 'Success',
                        'new_columns': new_cols,
                        'column_names': list(result_data.columns[-new_cols:]) if new_cols > 0 else []
                    }
                    
                    self.logger.info(f"✓ {name}: Added {new_cols} columns")
                    
                except Exception as e:
                    calculation_summary[name] = {
                        'status': 'Failed',
                        'error': str(e)
                    }
                    self.logger.error(f"✗ {name}: {e}")
            
            # Filter to requested date range (indicators now accurate)
            final_result = result_data[result_data.index >= start_date].copy()
            
            # Ensure we have data in the final result
            if final_result.empty:
                self.logger.warning("Final result is empty after date filtering. Returning all calculated data.")
                final_result = result_data.copy()
            
            self.logger.info(f"Final result: {len(final_result)} records from {final_result.index.min()} to {final_result.index.max()}")
            self.logger.info(f"Total columns: {len(final_result.columns)} (OHLCV + indicators)")
            
            # Cache latest values for quick access
            if not final_result.empty:
                latest_row = final_result.iloc[-1]
                self.latest_values_cache[symbol] = {
                    'timestamp': latest_row.name,
                    'values': latest_row.to_dict(),
                    'calculation_summary': calculation_summary
                }
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Backtest calculation failed for {symbol}: {e}")
            raise
    
    def calculate_indicators_live(self, symbol: str, data_stream: pd.DataFrame,
                                update_cache: bool = True) -> Dict[str, Any]:
        """
        Calculate indicators for real-time data stream.
        
        Args:
            symbol: Trading symbol
            data_stream: Real-time OHLCV data (single row or multiple rows)
            update_cache: Whether to update the latest values cache
            
        Returns:
            Dictionary with latest indicator values and signals
        """
        try:
            # Ensure data_stream has timestamp index but keep column too
            if 'timestamp' in data_stream.columns:
                data_stream['timestamp'] = pd.to_datetime(data_stream['timestamp'])
                data_stream.set_index('timestamp', inplace=True, drop=False)
            
            # Get indicator configurations
            indicator_configs = self._get_indicator_configs()
            
            # For live calculations, we need historical context
            # This is a simplified version - in production, you'd maintain a rolling buffer
            if len(data_stream) < 200:  # Need more historical data
                self.logger.warning("Insufficient historical data for accurate live calculations")
                self.logger.warning("Consider using calculate_indicators_backtest() first to warm up")
            
            # Calculate indicators
            indicators = create_local_indicator_factory(indicator_configs)
            result_data = data_stream.copy()
            
            for name, indicator in indicators.items():
                try:
                    result_data = indicator.calculate(result_data)
                except Exception as e:
                    self.logger.error(f"Live calculation failed for {name}: {e}")
            
            # Extract latest values
            if not result_data.empty:
                latest_row = result_data.iloc[-1]
                latest_values = {
                    'timestamp': latest_row.name,
                    'ohlcv': {
                        'open': latest_row.get('open'),
                        'high': latest_row.get('high'),
                        'low': latest_row.get('low'),
                        'close': latest_row.get('close'),
                        'volume': latest_row.get('volume')
                    },
                    'indicators': {},
                    'signals': {}
                }
                
                # Extract indicator values
                for col in latest_row.index:
                    if col not in ['open', 'high', 'low', 'close', 'volume']:
                        latest_values['indicators'][col] = latest_row[col]
                
                # Generate basic signals
                latest_values['signals'] = self._generate_basic_signals(latest_row)
                
                # Update cache
                if update_cache:
                    self.latest_values_cache[symbol] = latest_values
                
                return latest_values
            
            return {}
            
        except Exception as e:
            self.logger.error(f"Live calculation failed for {symbol}: {e}")
            return {}
    
    def get_latest_indicator_values(self, symbol: str) -> Dict[str, Any]:
        """
        Get cached latest indicator values for a symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Dictionary with latest indicator values, or empty dict if not available
        """
        return self.latest_values_cache.get(symbol, {})
    
    def export_indicators_to_excel(self, results: pd.DataFrame, 
                                  symbol: str = "Unknown",
                                  output_path: Optional[str] = None) -> str:
        """
        Export indicator results to Excel with professional formatting.
        
        Args:
            results: DataFrame with OHLCV and indicator data
            symbol: Trading symbol for naming
            output_path: Custom output path (optional)
            
        Returns:
            Path to created Excel file
        """
        try:
            # Generate output filename
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"outputs/indicators/{symbol}_indicators_{timestamp}.xlsx"
            
            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare data for export
            export_data = results.copy()
            
            # Round numeric columns for readability
            numeric_columns = export_data.select_dtypes(include=[np.number]).columns
            export_data[numeric_columns] = export_data[numeric_columns].round(4)
            
            # Create Excel writer
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Main data sheet
                export_data.to_excel(writer, sheet_name='Indicator_Data', index=True)
                
                # Summary sheet
                summary_data = {
                    'Metric': [
                        'Symbol',
                        'Date Range',
                        'Total Records',
                        'Total Columns',
                        'Indicator Columns',
                        'First Record',
                        'Last Record',
                        'Accuracy Level'
                    ],
                    'Value': [
                        symbol,
                        f"{export_data.index.min().date()} to {export_data.index.max().date()}",
                        len(export_data),
                        len(export_data.columns),
                        len(export_data.columns) - 5,  # Subtract OHLCV columns
                        export_data.index.min(),
                        export_data.index.max(),
                        self.accuracy_level
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Sample data (first and last 100 rows)
                sample_data = pd.concat([
                    export_data.head(100),
                    export_data.tail(100)
                ]).drop_duplicates()
                
                sample_data.to_excel(writer, sheet_name='Sample_Data', index=True)
            
            file_size = Path(output_path).stat().st_size / 1024  # KB
            self.logger.info(f"Excel file exported: {output_path} ({file_size:.1f} KB)")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Excel export failed: {e}")
            raise
    
    def _generate_basic_signals(self, latest_row: pd.Series) -> Dict[str, str]:
        """
        Generate basic trading signals from latest indicator values.
        
        Args:
            latest_row: Latest row of data with indicators
            
        Returns:
            Dictionary with basic signals
        """
        signals = {}
        
        try:
            # RSI signals
            if 'rsi' in latest_row:
                rsi_value = latest_row['rsi']
                if rsi_value >= 70:
                    signals['rsi'] = 'overbought'
                elif rsi_value <= 30:
                    signals['rsi'] = 'oversold'
                else:
                    signals['rsi'] = 'neutral'
            
            # Supertrend signals
            if 'supertrend_signal' in latest_row:
                st_signal = latest_row['supertrend_signal']
                signals['supertrend'] = 'bullish' if st_signal == 1 else 'bearish'
            
            # EMA trend signals (9 vs 21)
            if 'ema_9' in latest_row and 'ema_21' in latest_row:
                if latest_row['ema_9'] > latest_row['ema_21']:
                    signals['ema_trend'] = 'bullish'
                else:
                    signals['ema_trend'] = 'bearish'
            
            # VWAP signals
            if 'vwap' in latest_row and 'close' in latest_row:
                if latest_row['close'] > latest_row['vwap']:
                    signals['vwap'] = 'above_vwap'
                else:
                    signals['vwap'] = 'below_vwap'
            
        except Exception as e:
            self.logger.warning(f"Signal generation error: {e}")
        
        return signals


# Convenience functions for direct usage
def calculate_indicators_backtest(symbol: str, start_date: Union[str, datetime], 
                                end_date: Union[str, datetime], 
                                timeframe: str = '1min',
                                accuracy_level: str = 'good',
                                config_path: str = "config/config.xlsx") -> pd.DataFrame:
    """
    Convenience function to calculate indicators for backtesting.
    
    Args:
        symbol: Trading symbol
        start_date: Start date for backtest
        end_date: End date for backtest
        timeframe: Data timeframe
        accuracy_level: Accuracy level ('basic', 'good', 'high', 'conservative')
        config_path: Path to configuration file
        
    Returns:
        DataFrame with OHLCV and all indicators
    """
    calculator = UnifiedIndicatorCalculator(config_path, accuracy_level)
    return calculator.calculate_indicators_backtest(symbol, start_date, end_date, timeframe)


def calculate_indicators_live(symbol: str, data_stream: pd.DataFrame,
                            config_path: str = "config/config.xlsx") -> Dict[str, Any]:
    """
    Convenience function to calculate indicators for live data.
    
    Args:
        symbol: Trading symbol
        data_stream: Real-time OHLCV data
        config_path: Path to configuration file
        
    Returns:
        Dictionary with latest indicator values
    """
    calculator = UnifiedIndicatorCalculator(config_path)
    return calculator.calculate_indicators_live(symbol, data_stream)


def get_latest_indicator_values(symbol: str, config_path: str = "config/config.xlsx") -> Dict[str, Any]:
    """
    Convenience function to get latest indicator values.
    
    Args:
        symbol: Trading symbol
        config_path: Path to configuration file
        
    Returns:
        Dictionary with latest values
    """
    calculator = UnifiedIndicatorCalculator(config_path)
    return calculator.get_latest_indicator_values(symbol)


def export_indicators_to_excel(results: pd.DataFrame, symbol: str = "Unknown",
                              output_path: Optional[str] = None) -> str:
    """
    Convenience function to export indicators to Excel.
    
    Args:
        results: DataFrame with indicator data
        symbol: Trading symbol
        output_path: Custom output path
        
    Returns:
        Path to created Excel file
    """
    calculator = UnifiedIndicatorCalculator()
    return calculator.export_indicators_to_excel(results, symbol, output_path)


if __name__ == "__main__":
    # Example usage
    print("🚀 Unified Indicator Calculator")
    print("=" * 50)
    
    try:
        # Example 1: Backtest calculation
        print("\n📊 Example: Backtest Calculation")
        results = calculate_indicators_backtest(
            symbol="NIFTY",
            start_date="2024-06-01", 
            end_date="2024-06-30",
            timeframe="1min",
            accuracy_level="good"
        )
        
        print(f"✓ Calculated indicators for {len(results)} records")
        print(f"✓ Date range: {results.index.min()} to {results.index.max()}")
        print(f"✓ Columns: {len(results.columns)} total")
        
        # Example 2: Export to Excel
        excel_file = export_indicators_to_excel(results, "NIFTY")
        print(f"✓ Exported to: {excel_file}")
        
    except Exception as e:
        print(f"❌ Example failed: {e}")
        print("This is expected if database/CSV data is not available")