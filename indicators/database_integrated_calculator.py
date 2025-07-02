#!/usr/bin/env python3
"""
Database Integrated Indicator Calculator
========================================

Production-ready indicator calculator with full database integration and timeframe support.
Reads timeframe configuration from Excel and fetches appropriate candle data from DuckDB.

Key Features:
- Timeframe-aware data fetching from database
- Excel configuration integration for instruments and timeframes
- High-accuracy calculations with dynamic warmup
- Robust error handling and fallback mechanisms

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
    DATABASE_AVAILABLE = True
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
    DATABASE_AVAILABLE = False

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
    INDICATORS_AVAILABLE = False


class DatabaseIntegratedCalculator:
    """
    Production-ready calculator with full database and configuration integration.
    
    Automatically reads timeframes from Excel config and fetches appropriate
    candle data from DuckDB for accurate indicator calculations.
    """
    
    def __init__(self, config_path: str = "config/config.xlsx", accuracy_level: str = 'good'):
        """
        Initialize database integrated calculator.
        
        Args:
            config_path: Path to Excel configuration file
            accuracy_level: Accuracy level for calculations
        """
        self.config_path = config_path
        self.accuracy_level = accuracy_level
        self.logger = get_logger(__name__)
        
        # Initialize components
        self.warmup_calculator = WarmupCalculator(accuracy_level)
        self.db_manager = None
        self.config_loader = None
        self.instruments_config = {}
        self.indicators_config = {}
        
        # Load configuration
        self._initialize_configuration()
        self._initialize_database()
    
    def _initialize_configuration(self) -> None:
        """Initialize and load configuration from Excel."""
        try:
            if os.path.exists(self.config_path):
                if DATABASE_AVAILABLE:
                    self.config_loader = ConfigLoader(self.config_path)
                    
                    # Load instruments configuration
                    self.instruments_config = self.config_loader.get_active_instruments()
                    self.logger.info(f"Loaded {len(self.instruments_config)} active instruments")
                    
                    # Load indicators configuration
                    self.indicators_config = self.config_loader.get_indicator_config()
                    self.logger.info(f"Loaded configuration for {len(self.indicators_config)} indicators")
                    
                    # Log timeframes found
                    timeframes = set()
                    for instrument in self.instruments_config:
                        if 'timeframe' in instrument:
                            timeframes.add(instrument['timeframe'])
                    
                    self.logger.info(f"Configured timeframes: {sorted(timeframes)}")
                else:
                    self.logger.warning("ConfigLoader not available, using defaults")
                    self._load_default_configuration()
            else:
                self.logger.warning(f"Configuration file not found: {self.config_path}")
                self._load_default_configuration()
                
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            self._load_default_configuration()
    
    def _load_default_configuration(self) -> None:
        """Load default configuration when Excel config is unavailable."""
        self.instruments_config = [
            {
                'symbol': 'NIFTY',
                'exchange': 'NSE_INDEX',
                'timeframe': '1min',
                'status': 'Active'
            }
        ]
        
        self.indicators_config = {
            'CPR': {'timeframe': 'daily'},
            'Supertrend': {'period': 10, 'multiplier': 3.0},
            'EMA': {'periods': [9, 21, 50, 200]},
            'RSI': {'period': 14, 'overbought': 70, 'oversold': 30},
            'VWAP': {'session_type': 'daily', 'std_dev_multiplier': 1.0},
            'SMA': {'periods': [9, 21, 50, 200]},
            'BollingerBands': {'period': 20, 'std_dev': 2.0},
            'PreviousDayLevels': {'timeframe': 'daily'}
        }
        
        self.logger.info("Using default configuration")
    
    def _initialize_database(self) -> None:
        """Initialize database connection."""
        try:
            if DATABASE_AVAILABLE:
                self.db_manager = DatabaseManager()
                self.logger.info("Database connection established")
            else:
                self.logger.warning("Database not available - will use CSV fallback")
        except Exception as e:
            self.logger.error(f"Failed to initialize database: {e}")
            self.db_manager = None
    
    def _get_instrument_timeframe(self, symbol: str) -> str:
        """
        Get timeframe for a specific instrument from config.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Timeframe string (e.g., '1min', '5min', '1day')
        """
        for instrument in self.instruments_config:
            if instrument.get('symbol') == symbol:
                return instrument.get('timeframe', '1min')
        
        # Default timeframe if not found
        return '1min'
    
    def _fetch_data_from_database(self, symbol: str, start_date: datetime, 
                                 end_date: datetime, timeframe: str) -> pd.DataFrame:
        """
        Fetch data from database with proper timeframe handling.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for data
            end_date: End date for data
            timeframe: Data timeframe from config
            
        Returns:
            OHLCV DataFrame with timestamp column and index
        """
        try:
            if not self.db_manager:
                raise Exception("Database manager not available")
            
            # Convert dates to string format for database query
            start_str = start_date.strftime('%Y-%m-%d')
            end_str = end_date.strftime('%Y-%m-%d')
            
            self.logger.info(f"Fetching {symbol} data from {start_str} to {end_str} ({timeframe})")
            
            # Fetch data from database
            data = self.db_manager.get_ohlcv_data(
                symbol=symbol,
                start_date=start_str,
                end_date=end_str,
                timeframe=timeframe
            )
            
            if data is None or data.empty:
                raise Exception(f"No data returned from database for {symbol}")
            
            # Ensure proper data format
            data = self._prepare_database_data(data)
            
            self.logger.info(f"Successfully fetched {len(data)} records from database")
            self.logger.info(f"Date range: {data.index.min()} to {data.index.max()}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Database fetch failed: {e}")
            raise
    
    def _prepare_database_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Prepare database data for indicator calculations.
        
        Args:
            data: Raw data from database
            
        Returns:
            Properly formatted DataFrame
        """
        # Ensure timestamp handling
        if 'timestamp' in data.columns:
            data['timestamp'] = pd.to_datetime(data['timestamp'])
            data.set_index('timestamp', inplace=True, drop=False)
        elif data.index.name == 'timestamp' or isinstance(data.index, pd.DatetimeIndex):
            if 'timestamp' not in data.columns:
                data['timestamp'] = data.index
        else:
            raise ValueError("No timestamp column or index found in data")
        
        # Validate required columns
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        missing_cols = [col for col in required_cols if col not in data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        # Sort by timestamp
        data = data.sort_index()
        
        return data
    
    def _create_indicator_factory(self, config_data: dict) -> dict:
        """
        Create indicator instances from configuration.
        
        Args:
            config_data: Dictionary with indicator configurations
            
        Returns:
            Dictionary of initialized indicator instances
        """
        indicators = {}
        
        if not INDICATORS_AVAILABLE:
            self.logger.warning("Indicators not available")
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
                    self.logger.debug(f"Created {indicator_name} indicator with params: {params}")
                except Exception as e:
                    self.logger.warning(f"Failed to create {indicator_name} indicator: {e}")
        
        return indicators
    
    def calculate_indicators_for_symbol(self, symbol: str, start_date: Union[str, datetime], 
                                       end_date: Union[str, datetime],
                                       accuracy_level: Optional[str] = None) -> pd.DataFrame:
        """
        Calculate indicators for a specific symbol using database and config.
        
        Args:
            symbol: Trading symbol
            start_date: Start date for calculation
            end_date: End date for calculation
            accuracy_level: Override accuracy level
            
        Returns:
            DataFrame with OHLCV and all indicators
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
            
            self.logger.info(f"Starting indicator calculation for {symbol}")
            self.logger.info(f"Period: {start_date.date()} to {end_date.date()}")
            self.logger.info(f"Accuracy level: {accuracy_level}")
            
            # Get timeframe from config
            timeframe = self._get_instrument_timeframe(symbol)
            self.logger.info(f"Using timeframe from config: {timeframe}")
            
            # Calculate required warmup period
            max_warmup_candles = warmup_calc.calculate_max_warmup_needed(self.indicators_config)
            self.logger.info(f"Maximum warmup required: {max_warmup_candles} candles")
            
            # Calculate extended start date for warmup
            extended_start = warmup_calc.calculate_extended_start_date(
                start_date, max_warmup_candles, timeframe
            )
            self.logger.info(f"Extended fetch period: {extended_start.date()} to {end_date.date()}")
            
            # Fetch data from database
            data = self._fetch_data_from_database(symbol, extended_start, end_date, timeframe)
            
            # Validate data sufficiency
            is_sufficient, message = warmup_calc.validate_data_sufficiency(
                len(data), max_warmup_candles
            )
            self.logger.info(f"Data sufficiency: {message}")
            
            if not is_sufficient:
                self.logger.warning("Proceeding with available data, but accuracy may be reduced")
            
            # Calculate indicators
            result_data = self._calculate_all_indicators(data)
            
            # Filter to requested date range
            final_result = result_data[result_data.index >= start_date].copy()
            
            # Ensure we have data in final result
            if final_result.empty:
                self.logger.warning("Final result empty after filtering. Returning all data.")
                final_result = result_data.copy()
            
            self.logger.info(f"Final result: {len(final_result)} records")
            self.logger.info(f"Date range: {final_result.index.min()} to {final_result.index.max()}")
            self.logger.info(f"Total columns: {len(final_result.columns)}")
            
            return final_result
            
        except Exception as e:
            self.logger.error(f"Indicator calculation failed for {symbol}: {e}")
            raise
    
    def _calculate_all_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate all configured indicators on the data.
        
        Args:
            data: OHLCV data
            
        Returns:
            Data with all indicators calculated
        """
        self.logger.info("Calculating all indicators...")
        
        # Create indicator instances
        indicators = self._create_indicator_factory(self.indicators_config)
        self.logger.info(f"Created {len(indicators)} indicators")
        
        result_data = data.copy()
        calculation_summary = {}
        
        for name, indicator in indicators.items():
            try:
                before_cols = len(result_data.columns)
                
                # Handle indicators that need timestamp as column
                if name in ['CPR', 'VWAP', 'PreviousDayLevels']:
                    # These indicators need timestamp as column, not index
                    indicator_data = result_data.reset_index()
                    if 'timestamp' not in indicator_data.columns:
                        indicator_data['timestamp'] = indicator_data.index
                    
                    calculated_data = indicator.calculate(indicator_data)
                    
                    # Restore index and merge results
                    if 'timestamp' in calculated_data.columns:
                        calculated_data.set_index('timestamp', inplace=True)
                    
                    # Copy new columns to result_data
                    new_columns = [col for col in calculated_data.columns 
                                 if col not in result_data.columns]
                    for col in new_columns:
                        if col in calculated_data.columns:
                            result_data[col] = calculated_data[col]
                else:
                    # Other indicators work with index-based data
                    result_data = indicator.calculate(result_data)
                
                after_cols = len(result_data.columns)
                new_cols = after_cols - before_cols
                
                calculation_summary[name] = {
                    'status': 'Success',
                    'new_columns': new_cols
                }
                
                self.logger.info(f"✓ {name}: Added {new_cols} columns")
                
            except Exception as e:
                calculation_summary[name] = {
                    'status': 'Failed',
                    'error': str(e)
                }
                self.logger.error(f"✗ {name}: {e}")
        
        # Log summary
        successful = sum(1 for s in calculation_summary.values() if s['status'] == 'Success')
        total = len(calculation_summary)
        self.logger.info(f"Indicator calculation complete: {successful}/{total} successful")
        
        return result_data
    
    def calculate_indicators_for_all_instruments(self, start_date: Union[str, datetime],
                                               end_date: Union[str, datetime]) -> Dict[str, pd.DataFrame]:
        """
        Calculate indicators for all configured instruments.
        
        Args:
            start_date: Start date for calculation
            end_date: End date for calculation
            
        Returns:
            Dictionary mapping symbols to their indicator DataFrames
        """
        results = {}
        
        # Get active instruments from config
        active_instruments = [inst for inst in self.instruments_config 
                            if inst.get('status', '').upper() == 'ACTIVE']
        
        self.logger.info(f"Processing {len(active_instruments)} active instruments")
        
        for instrument in active_instruments:
            symbol = instrument.get('symbol')
            if not symbol:
                continue
                
            try:
                self.logger.info(f"Processing {symbol}...")
                result = self.calculate_indicators_for_symbol(symbol, start_date, end_date)
                results[symbol] = result
                self.logger.info(f"✓ {symbol}: {len(result)} records processed")
                
            except Exception as e:
                self.logger.error(f"✗ {symbol}: Failed - {e}")
                results[symbol] = pd.DataFrame()  # Empty DataFrame for failed symbols
        
        self.logger.info(f"Batch processing complete: {len(results)} symbols processed")
        return results
    
    def export_results_to_excel(self, results: Dict[str, pd.DataFrame], 
                               output_path: Optional[str] = None) -> str:
        """
        Export calculation results to Excel file.
        
        Args:
            results: Dictionary of symbol -> DataFrame results
            output_path: Custom output path
            
        Returns:
            Path to created Excel file
        """
        try:
            if not output_path:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                output_path = f"outputs/indicators/batch_indicators_{timestamp}.xlsx"
            
            # Create output directory
            Path(output_path).parent.mkdir(parents=True, exist_ok=True)
            
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                # Summary sheet
                summary_data = []
                for symbol, data in results.items():
                    if not data.empty:
                        summary_data.append({
                            'Symbol': symbol,
                            'Records': len(data),
                            'Columns': len(data.columns),
                            'Start_Date': data.index.min(),
                            'End_Date': data.index.max(),
                            'Timeframe': self._get_instrument_timeframe(symbol)
                        })
                
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Individual symbol sheets
                for symbol, data in results.items():
                    if not data.empty:
                        # Limit to first 10000 rows for Excel
                        export_data = data.head(10000).copy()
                        
                        # Round numeric columns
                        numeric_columns = export_data.select_dtypes(include=[np.number]).columns
                        export_data[numeric_columns] = export_data[numeric_columns].round(4)
                        
                        # Create sheet name (max 31 chars for Excel)
                        sheet_name = symbol[:31]
                        export_data.to_excel(writer, sheet_name=sheet_name, index=True)
            
            file_size = Path(output_path).stat().st_size / 1024  # KB
            self.logger.info(f"Excel export complete: {output_path} ({file_size:.1f} KB)")
            
            return output_path
            
        except Exception as e:
            self.logger.error(f"Excel export failed: {e}")
            raise
    
    def get_available_symbols(self) -> List[str]:
        """
        Get list of available symbols from configuration.
        
        Returns:
            List of symbol names
        """
        return [inst.get('symbol') for inst in self.instruments_config 
                if inst.get('symbol') and inst.get('status', '').upper() == 'ACTIVE']
    
    def get_symbol_timeframe(self, symbol: str) -> str:
        """
        Get timeframe for a specific symbol.
        
        Args:
            symbol: Trading symbol
            
        Returns:
            Timeframe string
        """
        return self._get_instrument_timeframe(symbol)
    
    def validate_database_connection(self) -> bool:
        """
        Validate database connection and data availability.
        
        Returns:
            True if database is accessible and has data
        """
        try:
            if not self.db_manager:
                return False
            
            # Try to get database stats
            stats = self.db_manager.get_database_stats()
            
            if stats and stats.get('total_records', 0) > 0:
                self.logger.info(f"Database validation successful: {stats['total_records']} records available")
                return True
            else:
                self.logger.warning("Database accessible but no records found")
                return False
                
        except Exception as e:
            self.logger.error(f"Database validation failed: {e}")
            return False


# Convenience functions for easy usage
def calculate_indicators_with_database(symbol: str, start_date: Union[str, datetime],
                                     end_date: Union[str, datetime], 
                                     config_path: str = "config/config.xlsx",
                                     accuracy_level: str = 'good') -> pd.DataFrame:
    """
    Convenience function to calculate indicators with database integration.
    
    Args:
        symbol: Trading symbol
        start_date: Start date
        end_date: End date
        config_path: Path to configuration file
        accuracy_level: Accuracy level
        
    Returns:
        DataFrame with indicators
    """
    calculator = DatabaseIntegratedCalculator(config_path, accuracy_level)
    return calculator.calculate_indicators_for_symbol(symbol, start_date, end_date)


def calculate_indicators_batch_processing(start_date: Union[str, datetime],
                                        end_date: Union[str, datetime],
                                        config_path: str = "config/config.xlsx",
                                        accuracy_level: str = 'good') -> Dict[str, pd.DataFrame]:
    """
    Convenience function for batch processing all configured instruments.
    
    Args:
        start_date: Start date
        end_date: End date
        config_path: Path to configuration file
        accuracy_level: Accuracy level
        
    Returns:
        Dictionary of results
    """
    calculator = DatabaseIntegratedCalculator(config_path, accuracy_level)
    return calculator.calculate_indicators_for_all_instruments(start_date, end_date)


if __name__ == "__main__":
    # Example usage and testing
    print("🗄️ Database Integrated Indicator Calculator")
    print("=" * 60)
    
    try:
        # Initialize calculator
        calculator = DatabaseIntegratedCalculator()
        
        # Validate database
        if calculator.validate_database_connection():
            print("✅ Database connection validated")
            
            # Get available symbols
            symbols = calculator.get_available_symbols()
            print(f"📊 Available symbols: {symbols}")
            
            if symbols:
                # Test with first symbol
                test_symbol = symbols[0]
                timeframe = calculator.get_symbol_timeframe(test_symbol)
                print(f"🎯 Testing with {test_symbol} ({timeframe})")
                
                # Calculate indicators
                result = calculator.calculate_indicators_for_symbol(
                    test_symbol, "2024-06-01", "2024-06-30"
                )
                
                print(f"✅ Calculation successful: {len(result)} records")
                print(f"📅 Date range: {result.index.min()} to {result.index.max()}")
                print(f"📊 Columns: {len(result.columns)}")
                
                # Export to Excel
                excel_file = calculator.export_results_to_excel({test_symbol: result})
                print(f"📁 Excel exported: {excel_file}")
                
            else:
                print("⚠️ No active symbols found in configuration")
        else:
            print("❌ Database connection failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print("This is expected if database is not available or configured properly")