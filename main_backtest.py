#!/usr/bin/env python3
"""
Enhanced Backtesting Script for vAlgo Trading System
==================================================

Institutional-grade backtesting with enhanced features:
- Real money options integration with actual strike prices
- Parallel tracking of equity and options data
- Enhanced exit analysis with method comparison
- Comprehensive 8-sheet Excel reporting
- 156x performance optimization

Usage:
    python main_backtest.py                          # Run with config-driven settings
    python main_backtest.py --symbol NIFTY           # Specific symbol
    python main_backtest.py --start-date 2024-01-01 --end-date 2024-12-31
    python main_backtest.py --validate-data          # Check data availability first
"""

import sys
import argparse
import os
import time
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import logging
import pandas as pd
import datetime as dt_module

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import enhanced components
try:
    from backtest_engine.enhanced_backtest_integration import EnhancedBacktestIntegration
    from utils.logger import get_logger, setup_logger
    COMPONENTS_AVAILABLE = True
    print("Enhanced components loaded successfully")
except ImportError as e:
    print(f"[ERROR] Failed to import enhanced components: {e}")
    COMPONENTS_AVAILABLE = False

# Import supporting components conditionally
if COMPONENTS_AVAILABLE:
    try:
        from backtest_engine.options_real_money_integrator import OptionsRealMoneyIntegrator
        from utils.parallel_chart_tracker import ParallelChartTracker
        from backtest_engine.enhanced_exit_analyzer import EnhancedExitAnalyzer
        from backtest_engine.report_generator import ReportGenerator
        from utils.config_cache import get_cached_config
        from data_manager.database import DatabaseManager
        from indicators.unified_indicator_engine import UnifiedIndicatorEngine
    except ImportError as e:
        print(f"[WARNING] Some enhanced components not available: {e}")
        print("Running in basic enhanced mode")


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="vAlgo Enhanced Backtesting Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Symbol/Strategy selection
    parser.add_argument('--symbol', type=str, help='Symbol to backtest (e.g., NIFTY)')
    parser.add_argument('--symbols', type=str, help='Comma-separated symbols (e.g., NIFTY,BANKNIFTY)')
    parser.add_argument('--strategy', type=str, help='Strategy name to test')
    
    # Date range - CONFIG-DRIVEN by default
    parser.add_argument('--start-date', type=str, 
                       help='Start date (YYYY-MM-DD) - requires --override-config-dates flag')
    parser.add_argument('--end-date', type=str, 
                       help='End date (YYYY-MM-DD) - requires --override-config-dates flag')
    parser.add_argument('--override-config-dates', action='store_true',
                       help='Override config dates with command line dates (use with --start-date/--end-date)')
    parser.add_argument('--show-config-dates', action='store_true',
                       help='Display date range from config without running backtest')
    
    # Configuration
    parser.add_argument('--config', type=str, default='config/config.xlsx', help='Config file path')
    parser.add_argument('--capital', type=float, default=100000.0, help='Initial capital')
    
    # Enhanced features
    parser.add_argument('--enable-real-money', action='store_true', default=True, 
                       help='Enable real money analysis (default: True)')
    parser.add_argument('--disable-real-money', action='store_true', 
                       help='Disable real money analysis')
    
    # Output options
    parser.add_argument('--output-dir', type=str, default='outputs/reports', help='Output directory')
    parser.add_argument('--no-excel', action='store_true', help='Skip Excel report generation')
    parser.add_argument('--export-detailed', action='store_true', default=True,
                       help='Export detailed analysis reports')
    
    # Logging
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose logging')
    parser.add_argument('--quiet', '-q', action='store_true', help='Minimal output')
    
    # Info commands
    parser.add_argument('--info', action='store_true', help='Show engine information')
    parser.add_argument('--validate', action='store_true', help='Validate configuration only')
    parser.add_argument('--validate-data', action='store_true', help='Validate data availability before running')
    
    return parser.parse_args()


def setup_logging(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Setup logging configuration"""
    if quiet:
        log_level = "WARNING"
    elif verbose:
        log_level = "DEBUG"
    else:
        log_level = "INFO"
    
    logger = setup_logger("enhanced_backtest", log_level=log_level)
    return logger


def parse_symbol_list(symbols_str: str) -> List[str]:
    """Parse comma-separated symbol list"""
    if not symbols_str:
        return []
    return [s.strip().upper() for s in symbols_str.split(',') if s.strip()]


def determine_symbols(args: argparse.Namespace, integration: Any) -> List[str]:
    """Determine which symbols to backtest"""
    if args.symbol:
        return [args.symbol.upper()]
    elif args.symbols:
        return parse_symbol_list(args.symbols)
    else:
        # Default to NIFTY
        return ['NIFTY']


def determine_strategies(args: argparse.Namespace, integration: Any) -> List[str]:
    """Determine which strategies to test using enhanced strategy selection"""
    try:
        if args.strategy:
            # Validate that requested strategy is available
            active_strategies = list(integration.config['active_strategies'].keys())
            if args.strategy not in active_strategies:
                available_strategies = ', '.join(active_strategies) if active_strategies else 'None'
                raise ValueError(f"Requested strategy '{args.strategy}' not found in active strategies. Available: {available_strategies}")
            return [args.strategy]
        else:
            # Get all active strategies from enhanced system (already filtered by Run_Strategy)
            active_strategies = list(integration.config['active_strategies'].keys())
            
            if not active_strategies:
                # Check if there are strategies in config but none selected in Run_Strategy
                strategy_summary = integration.config.get('strategy_summary', {})
                total_in_config = strategy_summary.get('total_strategies_in_config', 0)
                total_selected = strategy_summary.get('active_strategies_in_run_strategy', 0)
                
                if total_in_config > 0 and total_selected == 0:
                    raise ValueError(f"No strategies selected in Run_Strategy sheet. Found {total_in_config} strategies in config but all are disabled in Run_Strategy")
                else:
                    raise ValueError("No active strategies found after enhanced filtering")
            
            # Return all active strategies (enhanced system already filtered them)
            return active_strategies
            
    except Exception as e:
        # Strict error handling - no fallbacks
        raise ValueError(f"Strategy determination failed: {e}")


def determine_date_range(args: argparse.Namespace, integration: Any, symbol: str = 'NIFTY') -> tuple:
    """
    Determine start and end dates for backtesting - STRICTLY CONFIG-DRIVEN
    
    Always reads from config first. Command line dates only used with explicit override flag.
    No fallback mechanisms to database or hardcoded dates.
    """
    
    # STEP 1: Always try to read from config first (config-driven approach)
    config_start_date = None
    config_end_date = None
    
    try:
        # Load config through cached config to avoid repetitive loading
        config_cache = get_cached_config(integration.config_path)
        config_loader = config_cache.get_config_loader()
        init_config = config_loader.get_initialize_config()
        
        # Extract start_date from config
        config_start = init_config.get('start_date')
        if config_start and str(config_start) != 'nan':
            config_start_date = _parse_config_date(config_start, 'start_date', integration.logger)
        
        # Extract end_date from config
        config_end = init_config.get('end_date')
        if config_end and str(config_end) != 'nan':
            config_end_date = _parse_config_date(config_end, 'end_date', integration.logger)
            
        # Log what we found in config
        if config_start_date and config_end_date:
            integration.logger.info(f"[CONFIG] Found date range in config: {config_start_date} to {config_end_date}")
        else:
            integration.logger.warning(f"[CONFIG] Config missing dates - start_date: {config_start_date}, end_date: {config_end_date}")
            
    except Exception as e:
        integration.logger.error(f"[CONFIG] Error reading dates from config: {e}")
        config_start_date = None
        config_end_date = None
    
    # STEP 2: Check if command line override is requested
    use_command_line_dates = getattr(args, 'override_config_dates', False)
    
    if use_command_line_dates:
        # Command line override explicitly requested
        if args.start_date and args.end_date:
            integration.logger.info(f"[OVERRIDE] Using command line dates (override requested): {args.start_date} to {args.end_date}")
            return args.start_date, args.end_date
        else:
            integration.logger.error(f"[OVERRIDE] Override flag set but command line dates incomplete: start={args.start_date}, end={args.end_date}")
            raise ValueError("When using --override-config-dates, both --start-date and --end-date must be provided")
    
    # STEP 3: Use config dates (strict config-driven approach)
    if config_start_date and config_end_date:
        integration.logger.info(f"[CONFIG-DRIVEN] Using config dates: {config_start_date} to {config_end_date}")
        return config_start_date, config_end_date
    
    # STEP 4: FAIL FAST - No fallbacks allowed
    integration.logger.error(f"[ERROR] MISSING DATE CONFIGURATION IN CONFIG FILE")
    integration.logger.error("   - start_date and end_date must be properly configured in Initialize sheet")
    integration.logger.error("   - Command line dates ignored unless --override-config-dates flag is used")
    integration.logger.error("   - No fallback mechanisms available - system is strictly config-driven")
    
    if args.start_date or args.end_date:
        integration.logger.error(f"   - NOTE: Command line dates detected ({args.start_date}, {args.end_date}) but ignored")
        integration.logger.error(f"   - Use --override-config-dates flag to use command line dates instead of config")
    
    raise ValueError("Date configuration missing in config file. System is strictly config-driven - no fallbacks available.")


def _parse_config_date(date_value: Any, date_name: str, logger: Any) -> Optional[str]:
    """Parse date value from config, supporting multiple formats"""
    try:
        # Handle pandas Timestamp objects
        if hasattr(date_value, 'strftime'):
            return date_value.strftime('%Y-%m-%d')
        
        # Handle string values
        date_str = str(date_value).strip()
        if not date_str or date_str.lower() in ['nan', 'nat', 'none', '']:
            return None
            
        # Handle DD-MM-YYYY format from Excel config
        if '-' in date_str and len(date_str.split('-')) == 3:
            try:
                # Try DD-MM-YYYY format first
                parsed_date = dt_module.datetime.strptime(date_str, '%d-%m-%Y')
                result = parsed_date.strftime('%Y-%m-%d')
                logger.info(f"[CONFIG] Converted {date_name} {date_str} (DD-MM-YYYY) to {result}")
                return result
            except ValueError:
                try:
                    # Try YYYY-MM-DD format as fallback
                    parsed_date = dt_module.datetime.strptime(date_str, '%Y-%m-%d')
                    logger.info(f"[CONFIG] Using {date_name} {date_str} (YYYY-MM-DD format)")
                    return date_str
                except ValueError:
                    logger.error(f"[CONFIG] Invalid date format for {date_name}: {date_str}")
                    return None
        else:
            logger.warning(f"[CONFIG] Unexpected date format for {date_name}: {date_str}")
            return date_str if date_str else None
            
    except Exception as e:
        logger.error(f"[CONFIG] Error parsing {date_name} '{date_value}': {e}")
        return None


def _determine_exchange_for_symbol(symbol: str, db_manager: Any, logger: Any) -> str:
    """Determine the correct exchange for a symbol for data loading"""
    try:
        # Query available exchanges for this symbol using direct connection
        result = db_manager.connection.execute("""
            SELECT DISTINCT exchange, COUNT(*) as record_count
            FROM ohlcv_data 
            WHERE symbol = ? 
            GROUP BY exchange 
            ORDER BY record_count DESC
        """, [symbol.upper()]).fetchall()
        
        if result:
            best_exchange = result[0][0]
            logger.info(f"Found data for {symbol} in exchange: {best_exchange} ({result[0][1]:,} records)")
            return best_exchange
        else:
            # Get available symbols for user reference
            available_symbols = db_manager.connection.execute("""
                SELECT DISTINCT symbol, exchange, COUNT(*) as records
                FROM ohlcv_data 
                GROUP BY symbol, exchange 
                ORDER BY symbol, exchange
            """).fetchall()
            
            logger.error(f"[ERROR] NO DATA FOUND for symbol '{symbol}' in database")
            logger.error("📊 Available symbols in database:")
            for sym, exch, count in available_symbols:
                logger.error(f"   {sym} ({exch}): {count:,} records")
            
            raise ValueError(f"No data available for symbol '{symbol}'. Check available symbols above.")
                
    except Exception as e:
        if "No data available for symbol" in str(e):
            raise e  # Re-raise our specific error
        logger.error(f"[ERROR] Database error while checking symbol '{symbol}': {e}")
        raise ValueError(f"Database error while validating symbol '{symbol}': {e}")


def _validate_data_availability(symbol: str, exchange: str, start_date: str, end_date: str, 
                               db_manager: Any, logger: Any) -> None:
    """Validate that required data is available for the specified parameters"""
    try:
        # Check 5-minute data availability
        result_5m = db_manager.connection.execute("""
            SELECT COUNT(*) as count, MIN(timestamp) as min_date, MAX(timestamp) as max_date
            FROM ohlcv_data 
            WHERE symbol = ? AND exchange = ? AND timeframe = '5m'
            AND timestamp >= ? AND timestamp <= ?
        """, [symbol.upper(), exchange, start_date, end_date + ' 23:59:59']).fetchone()
        
        # Check 1-minute data availability  
        result_1m = db_manager.connection.execute("""
            SELECT COUNT(*) as count, MIN(timestamp) as min_date, MAX(timestamp) as max_date
            FROM ohlcv_data 
            WHERE symbol = ? AND exchange = ? AND timeframe = '1m'
            AND timestamp >= ? AND timestamp <= ?
        """, [symbol.upper(), exchange, start_date, end_date + ' 23:59:59']).fetchone()
        
        # Validate 5-minute data
        if not result_5m or result_5m[0] == 0:
            logger.error(f"[ERROR] NO 5-MINUTE DATA for {symbol} ({exchange}) in range {start_date} to {end_date}")
            _log_available_data_summary(symbol, exchange, db_manager, logger)
            raise ValueError(f"No 5-minute data found for {symbol} ({exchange}) in specified date range")
        
        # Validate 1-minute data
        if not result_1m or result_1m[0] == 0:
            logger.error(f"[ERROR] NO 1-MINUTE DATA for {symbol} ({exchange}) in range {start_date} to {end_date}")
            _log_available_data_summary(symbol, exchange, db_manager, logger)
            raise ValueError(f"No 1-minute data found for {symbol} ({exchange}) in specified date range")
        
        # Log validation success
        logger.info(f"[OK] Data validation passed:")
        logger.info(f"   5m data: {result_5m[0]:,} records ({str(result_5m[1])[:10]} to {str(result_5m[2])[:10]})")
        logger.info(f"   1m data: {result_1m[0]:,} records ({str(result_1m[1])[:10]} to {str(result_1m[2])[:10]})")
        
    except Exception as e:
        if "No 5-minute data found" in str(e) or "No 1-minute data found" in str(e):
            raise e  # Re-raise our specific errors
        logger.error(f"[ERROR] Error during data validation: {e}")
        raise ValueError(f"Data validation failed for {symbol}: {e}")


def _log_available_data_summary(symbol: str, exchange: str, db_manager: Any, logger: Any) -> None:
    """Log summary of available data for troubleshooting"""
    try:
        # Get available data for this symbol/exchange
        result = db_manager.connection.execute("""
            SELECT timeframe, COUNT(*) as records, MIN(timestamp) as start_date, MAX(timestamp) as end_date
            FROM ohlcv_data 
            WHERE symbol = ? AND exchange = ?
            GROUP BY timeframe
            ORDER BY timeframe
        """, [symbol.upper(), exchange]).fetchall()
        
        if result:
            logger.error(f"[DATA] Available data for {symbol} ({exchange}):")
            for tf, count, start, end in result:
                logger.error(f"   {tf}: {count:,} records from {str(start)[:10]} to {str(end)[:10]}")
        else:
            logger.error(f"[DATA] No data found for {symbol} ({exchange}) in any timeframe")
            
    except Exception as e:
        logger.error(f"Error getting data summary: {e}")


def load_enhanced_data(symbol: str, start_date: str, end_date: str, 
                      integration: Any) -> tuple:
    """Load signal and execution data for enhanced backtesting - STRICT VALIDATION"""
    try:
        # Initialize database manager with optimized connection
        db_manager = DatabaseManager()
        
        # Strictly determine the correct exchange for the symbol (will fail if no data)
        exchange = _determine_exchange_for_symbol(symbol, db_manager, integration.logger)
        integration.logger.info(f"[OK] Using exchange '{exchange}' for symbol {symbol}")
        
        # Validate data availability before loading
        _validate_data_availability(symbol, exchange, start_date, end_date, db_manager, integration.logger)
        
        # Load additional historical data for proper EMA warm-up (200 periods for EMA_200)
        # Calculate start date for warm-up (go back 200 trading days for EMA_200)
        start_date_parsed = dt_module.datetime.strptime(start_date, '%Y-%m-%d')
        warmup_start_date = (start_date_parsed - dt_module.timedelta(days=400)).strftime('%Y-%m-%d')  # ~200 trading days
        
        integration.logger.info(f"Loading data with warm-up period: {warmup_start_date} to {end_date}")
        
        # Load 5-minute data for signal generation with warm-up
        signal_data_with_warmup = db_manager.get_ohlcv_data(
            symbol=symbol,
            exchange=exchange,
            start_date=warmup_start_date, 
            end_date=end_date, 
            timeframe='5m'
        )
        
        # Load 1-minute data for execution precision (only requested range)
        execution_data = db_manager.get_ohlcv_data(
            symbol=symbol,
            exchange=exchange,
            start_date=start_date, 
            end_date=end_date, 
            timeframe='1m'
        )
        
        # Final verification - data should not be empty after validation
        if signal_data_with_warmup.empty:
            integration.logger.error(f"[ERROR] UNEXPECTED: 5-minute data is empty after validation")
            raise ValueError(f"5-minute data unexpectedly empty for {symbol} ({exchange})")
            
        if execution_data.empty:
            integration.logger.error(f"[ERROR] UNEXPECTED: 1-minute data is empty after validation")
            raise ValueError(f"1-minute data unexpectedly empty for {symbol} ({exchange})")
        
        # Success - log data loaded
        integration.logger.info(f"[OK] Data loaded successfully:")
        integration.logger.info(f"   Signal data (5m): {len(signal_data_with_warmup):,} records")
        integration.logger.info(f"   Execution data (1m): {len(execution_data):,} records")
        
        # Calculate indicators on signal data using BacktestingIndicatorEngine
        # Use same initialization method as main_indicator_engine.py for consistency
        from indicators.unified_indicator_engine import BacktestingIndicatorEngine
        
        indicator_engine = BacktestingIndicatorEngine()
        
        # Log data sample before indicator calculation for debugging
        if not signal_data_with_warmup.empty:
            integration.logger.info(f"[DEBUG] Signal data sample before indicators:")
            # Check if timestamp is in columns or index
            if 'timestamp' in signal_data_with_warmup.columns:
                sample_cols = ['timestamp', 'open', 'high', 'low', 'close']
                available_cols = [col for col in sample_cols if col in signal_data_with_warmup.columns]
                integration.logger.info(f"   First 3 rows: {signal_data_with_warmup[available_cols].head(3).to_string()}")
            else:
                # Timestamp is likely in index
                basic_cols = ['open', 'high', 'low', 'close']
                available_cols = [col for col in basic_cols if col in signal_data_with_warmup.columns]
                sample_data = signal_data_with_warmup[available_cols].head(3)
                sample_data.index.name = 'timestamp'
                integration.logger.info(f"   First 3 rows: {sample_data.to_string()}")
            integration.logger.info(f"   Data types: {signal_data_with_warmup.dtypes.to_dict()}")
            integration.logger.info(f"   Index name: {signal_data_with_warmup.index.name}")
            integration.logger.info(f"   Shape: {signal_data_with_warmup.shape}")
        
        # Add indicators to signal data with warmup
        signal_data_with_indicators = indicator_engine.calculate_indicators(
            data=signal_data_with_warmup
        )
        
        # Now filter the data to the requested date range (removing warmup data)
        start_date_parsed = dt_module.datetime.strptime(start_date, '%Y-%m-%d')
        end_date_parsed = dt_module.datetime.strptime(end_date, '%Y-%m-%d')
        
        # CRITICAL DEBUG: Log the exact filtering parameters
        integration.logger.info(f"[DATE_FILTER] Filtering data to range: {start_date} to {end_date}")
        integration.logger.info(f"[DATE_FILTER] Parsed dates: {start_date_parsed.date()} to {end_date_parsed.date()}")
        integration.logger.info(f"[DATE_FILTER] Signal data before filtering: {len(signal_data_with_indicators)} records")
        
        # Filter signal data to requested range
        if 'timestamp' in signal_data_with_indicators.columns:
            signal_data_with_indicators['timestamp'] = pd.to_datetime(signal_data_with_indicators['timestamp'])
            
            # DEBUG: Log data range before filtering
            min_date = signal_data_with_indicators['timestamp'].min()
            max_date = signal_data_with_indicators['timestamp'].max()
            integration.logger.info(f"[DATE_FILTER] Data range before filtering: {min_date} to {max_date}")
            
            signal_data = signal_data_with_indicators[
                (signal_data_with_indicators['timestamp'].dt.date >= start_date_parsed.date()) &
                (signal_data_with_indicators['timestamp'].dt.date <= end_date_parsed.date())
            ].copy()
            
            # DEBUG: Log filtering results
            if not signal_data.empty:
                min_filtered = signal_data['timestamp'].min()
                max_filtered = signal_data['timestamp'].max()
                unique_dates = signal_data['timestamp'].dt.date.nunique()
                integration.logger.info(f"[DATE_FILTER] Data range after filtering: {min_filtered} to {max_filtered}")
                integration.logger.info(f"[DATE_FILTER] Unique trading days found: {unique_dates}")
                integration.logger.info(f"[DATE_FILTER] Records after filtering: {len(signal_data)}")
            else:
                integration.logger.error(f"[DATE_FILTER] ERROR: No data remains after filtering!")
                integration.logger.error(f"[DATE_FILTER] Filter criteria: {start_date_parsed.date()} to {end_date_parsed.date()}")
        else:
            # Timestamp is in index
            signal_data_with_indicators.index = pd.to_datetime(signal_data_with_indicators.index)
            
            # DEBUG: Log data range before filtering
            min_date = signal_data_with_indicators.index.min()
            max_date = signal_data_with_indicators.index.max()
            integration.logger.info(f"[DATE_FILTER] Index range before filtering: {min_date} to {max_date}")
            
            signal_data = signal_data_with_indicators[
                (signal_data_with_indicators.index.date >= start_date_parsed.date()) &
                (signal_data_with_indicators.index.date <= end_date_parsed.date())
            ].copy()
            
            # DEBUG: Log filtering results
            if not signal_data.empty:
                min_filtered = signal_data.index.min()
                max_filtered = signal_data.index.max()
                unique_dates = pd.Series(signal_data.index.date).nunique()
                integration.logger.info(f"[DATE_FILTER] Index range after filtering: {min_filtered} to {max_filtered}")
                integration.logger.info(f"[DATE_FILTER] Unique trading days found: {unique_dates}")
                integration.logger.info(f"[DATE_FILTER] Records after filtering: {len(signal_data)}")
            else:
                integration.logger.error(f"[DATE_FILTER] ERROR: No data remains after filtering!")
                integration.logger.error(f"[DATE_FILTER] Filter criteria: {start_date_parsed.date()} to {end_date_parsed.date()}")
        
        # Log data sample after indicator calculation for debugging  
        if not signal_data.empty:
            integration.logger.info(f"[DEBUG] Signal data sample after indicators:")
            ema_columns = [col for col in signal_data.columns if 'ema' in col.lower()]
            if ema_columns:
                # Check if timestamp is in columns or index
                if 'timestamp' in signal_data.columns:
                    sample_cols = ['timestamp', 'close'] + ema_columns
                    available_cols = [col for col in sample_cols if col in signal_data.columns]
                    sample_data = signal_data[available_cols].head(3)
                else:
                    # Timestamp is in index
                    sample_cols = ['close'] + ema_columns
                    available_cols = [col for col in sample_cols if col in signal_data.columns]
                    sample_data = signal_data[available_cols].head(3)
                    sample_data.index.name = 'timestamp'
                integration.logger.info(f"   EMA values: {sample_data.to_string()}")
            else:
                integration.logger.warning(f"   No EMA columns found in signal data. Available columns: {list(signal_data.columns)}")
        
        # Validate EMA calculation consistency
        if not signal_data.empty and 'ema_9' in signal_data.columns:
            # Check if EMA values are realistic (should have variation)
            ema_9_std = signal_data['ema_9'].std()
            close_std = signal_data['close'].std()
            
            if ema_9_std > 0 and close_std > 0:
                ema_variation_ratio = ema_9_std / close_std
                integration.logger.info(f"[DEBUG] EMA variation check: EMA_9 std={ema_9_std:.2f}, Close std={close_std:.2f}, Ratio={ema_variation_ratio:.4f}")
                
                # EMA should have less variation than close but not too little
                if ema_variation_ratio < 0.1:
                    integration.logger.warning(f"[WARNING] EMA values may be too smooth (variation ratio: {ema_variation_ratio:.4f})")
                elif ema_variation_ratio > 0.9:
                    integration.logger.warning(f"[WARNING] EMA values may be too volatile (variation ratio: {ema_variation_ratio:.4f})")
            else:
                integration.logger.warning(f"[WARNING] EMA or Close values have zero variation - possible data issue")
        
        # Convert index to timestamp column for enhanced backtest integration
        if not signal_data.empty and 'timestamp' not in signal_data.columns:
            signal_data = signal_data.reset_index()
            if 'index' in signal_data.columns:
                signal_data = signal_data.rename(columns={'index': 'timestamp'})
        
        if not execution_data.empty and 'timestamp' not in execution_data.columns:
            execution_data = execution_data.reset_index()
            if 'index' in execution_data.columns:
                execution_data = execution_data.rename(columns={'index': 'timestamp'})
        
        integration.logger.info(f"[OK] Enhanced data loading completed for {symbol}")
        integration.logger.info(f"   Signal data columns: {list(signal_data.columns)[:10]}...")
        integration.logger.info(f"   Execution data columns: {list(execution_data.columns)[:5]}...")
        
        # Debug: Show actual date range in loaded data
        if not signal_data.empty:
            min_date = signal_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
            max_date = signal_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            integration.logger.info(f"   Signal data date range: {min_date} to {max_date}")
            
        if not execution_data.empty:
            min_date = execution_data['timestamp'].min().strftime('%Y-%m-%d %H:%M:%S')
            max_date = execution_data['timestamp'].max().strftime('%Y-%m-%d %H:%M:%S')
            integration.logger.info(f"   Execution data date range: {min_date} to {max_date}")
            
        integration.logger.info(f"   Requested date range: {start_date} to {end_date}")
        
        # CRITICAL VALIDATION: Ensure data spans the full requested range
        if not signal_data.empty and not execution_data.empty:
            # Get actual date ranges
            signal_start = signal_data['timestamp'].min().date()
            signal_end = signal_data['timestamp'].max().date()
            execution_start = execution_data['timestamp'].min().date()
            execution_end = execution_data['timestamp'].max().date()
            
            # Get expected date ranges
            expected_start = start_date_parsed.date()
            expected_end = end_date_parsed.date()
            expected_days = (expected_end - expected_start).days + 1
            
            # Check signal data coverage
            signal_days = (signal_end - signal_start).days + 1
            if signal_start != expected_start or signal_end != expected_end:
                integration.logger.warning(f"[DATE_VALIDATION] Signal data range mismatch:")
                integration.logger.warning(f"   Expected: {expected_start} to {expected_end} ({expected_days} days)")
                integration.logger.warning(f"   Actual: {signal_start} to {signal_end} ({signal_days} days)")
            else:
                integration.logger.info(f"[DATE_VALIDATION] [OK] Signal data spans full {expected_days}-day range")
            
            # Check execution data coverage
            execution_days = (execution_end - execution_start).days + 1
            if execution_start != expected_start or execution_end != expected_end:
                integration.logger.warning(f"[DATE_VALIDATION] Execution data range mismatch:")
                integration.logger.warning(f"   Expected: {expected_start} to {expected_end} ({expected_days} days)")
                integration.logger.warning(f"   Actual: {execution_start} to {execution_end} ({execution_days} days)")
            else:
                integration.logger.info(f"[DATE_VALIDATION] [OK] Execution data spans full {expected_days}-day range")
        
        return signal_data, execution_data
        
    except Exception as e:
        # Re-raise specific validation errors
        if any(phrase in str(e) for phrase in ["No data available for symbol", "No 5-minute data found", 
                                               "No 1-minute data found", "Data validation failed"]):
            raise e
        # Log and re-raise unexpected errors
        integration.logger.error(f"[ERROR] Unexpected error loading data for {symbol}: {e}")
        raise ValueError(f"Failed to load data for {symbol}: {e}")


def run_enhanced_symbol_backtest(symbol: str, start_date: str, end_date: str,
                                integration: Any,
                                strategy_name: Optional[str] = None) -> Dict[str, Any]:
    """Run enhanced backtest for a single symbol"""
    try:
        integration.logger.info(f"Starting enhanced backtest for {symbol}")
        
        # Load enhanced data with strict validation
        try:
            signal_data, execution_data = load_enhanced_data(symbol, start_date, end_date, integration)
        except Exception as e:
            integration.logger.error(f"[ERROR] Data loading failed for {symbol}: {e}")
            return {'error': str(e), 'symbol': symbol}
        
        # Run enhanced backtesting with real money integration and strict error handling
        try:
            if integration.enable_real_money:
                integration.logger.info(f"[BACKTEST] Running real money backtest for {symbol}")
                results = integration.run_real_money_backtest(
                    signal_data=signal_data,
                    execution_data=execution_data,
                    symbol=symbol,
                    strategy_name=strategy_name
                )
            else:
                integration.logger.info(f"[BACKTEST] Running enhanced backtest for {symbol}")
                results = integration.run_enhanced_backtest(
                    symbol=symbol,
                    start_date=start_date,
                    end_date=end_date,
                    signal_data=signal_data,
                    execution_data=execution_data,
                    strategy_name=strategy_name
                )
            
            # Validate results
            if not results:
                raise ValueError("Backtest returned empty results")
            
            integration.logger.info(f"[BACKTEST] Enhanced backtest completed successfully for {symbol}")
            return results
            
        except ValueError as e:
            # Strategy or configuration errors
            integration.logger.error(f"[BACKTEST_ERROR] Configuration/Strategy error for {symbol}: {e}")
            return {'error': f"Configuration error: {e}", 'symbol': symbol}
        except Exception as e:
            # Unexpected errors during backtesting
            integration.logger.error(f"[BACKTEST_ERROR] Unexpected error during backtest for {symbol}: {e}")
            return {'error': f"Backtest execution error: {e}", 'symbol': symbol}
        
    except Exception as e:
        # Top-level errors (data loading, initialization, etc.)
        integration.logger.error(f"[SYMBOL_ERROR] Critical error for {symbol}: {e}")
        return {'error': f"Critical error: {e}", 'symbol': symbol}


def generate_enhanced_summary(results: Dict[str, Any], symbols: List[str], 
                            execution_time: float, logger: logging.Logger) -> None:
    """Generate enhanced summary with institutional metrics"""
    try:
        print("\n" + "=" * 80)
        print("ENHANCED BACKTESTING SUMMARY WITH REAL MONEY ANALYSIS")
        print("=" * 80)
        
        # Overall execution metrics
        total_symbols = len(results)
        successful_results = sum(1 for result in results.values() 
                               if isinstance(result, dict) and 'error' not in result)
        
        print(f"Symbols Processed: {successful_results}/{total_symbols}")
        print(f"Execution Time: {execution_time:.2f} seconds")
        
        # Calculate processing speed
        total_records = 0
        total_trades = 0
        total_pnl = 0.0
        overall_stats = {
            'total_trades': 0, 'tp_hits': 0, 'sl_hits': 0,
            'call_trades': 0, 'put_trades': 0, 'equity_exits': 0, 'options_exits': 0
        }
        
        for symbol, result in results.items():
            if isinstance(result, dict) and 'error' not in result:
                # Extract metrics from enhanced results
                if 'performance_summary' in result:
                    perf = result['performance_summary']
                    total_trades += perf.get('total_trades', 0)
                    total_pnl += perf.get('total_pnl', 0)
                    overall_stats['tp_hits'] += perf.get('winning_trades', 0)
                    overall_stats['sl_hits'] += perf.get('losing_trades', 0)
                    overall_stats['equity_exits'] += perf.get('equity_exits', 0)
                    overall_stats['options_exits'] += perf.get('options_exits', 0)
                
                if 'backtest_info' in result:
                    total_records += result['backtest_info'].get('total_trading_days', 0)
        
        if execution_time > 0:
            processing_speed = total_records / execution_time
            print(f"Processing Speed: {processing_speed:,.0f} records/second")
        
        # Enhanced performance metrics
        print(f"\n[PERF] ENHANCED PERFORMANCE ANALYSIS:")
        print(f"  ├─ Total Trades Executed: {total_trades}")
        print(f"  ├─ Profitable Trades (TP): {overall_stats['tp_hits']}")
        print(f"  ├─ Loss Trades (SL): {overall_stats['sl_hits']}")
        
        if total_trades > 0:
            win_rate = (overall_stats['tp_hits'] / total_trades) * 100
            print(f"  ├─ Overall Win Rate: {win_rate:.1f}%")
        
        print(f"  ├─ Total P&L: ${total_pnl:.2f}")
        
        # Enhanced exit analysis
        total_exits = overall_stats['equity_exits'] + overall_stats['options_exits']
        if total_exits > 0:
            equity_pct = (overall_stats['equity_exits'] / total_exits) * 100
            options_pct = (overall_stats['options_exits'] / total_exits) * 100
            print(f"  ├─ Exit Method Analysis:")
            print(f"  │   ├─ Equity-based Exits: {overall_stats['equity_exits']} ({equity_pct:.1f}%)")
            print(f"  │   └─ Options-based Exits: {overall_stats['options_exits']} ({options_pct:.1f}%)")
        
        # Real money analysis summary
        for symbol, result in results.items():
            if isinstance(result, dict) and 'real_money_analysis' in result:
                real_analysis = result['real_money_analysis']
                print(f"\n[DATA] {symbol} REAL MONEY INTEGRATION:")
                print(f"  ├─ Database Integration: {'[ACTIVE]' if real_analysis.get('database_integration') else '[INACTIVE]'}")
                print(f"  ├─ Real Strike Selections: {real_analysis.get('real_strike_selections', 0)}")
                print(f"  └─ Parallel Tracking Points: {real_analysis.get('parallel_tracking_points', 0)}")
        
        print("=" * 80)
        
    except Exception as e:
        logger.error(f"Error generating enhanced summary: {e}")
        # Fallback to basic summary
        print(f"\nBasic Summary: {len(results)} symbols processed in {execution_time:.2f} seconds")


def main():
    """Main execution function with enhanced backtesting"""
    
    # Check component availability
    if not COMPONENTS_AVAILABLE:
        print("[ERROR] Enhanced backtesting components not available")
        return 1
    
    try:
        # Parse arguments
        args = parse_arguments()
        
        # Setup logging
        logger = setup_logging(args.verbose, args.quiet)
        
        # Determine real money mode
        enable_real_money = args.enable_real_money and not args.disable_real_money
        
        # Initialize enhanced backtesting integration with STRICT error handling
        logger.info("Initializing Enhanced Backtesting Integration...")
        try:
            integration = EnhancedBacktestIntegration(
                config_path=args.config,
                enable_real_money=enable_real_money
            )
        except ValueError as e:
            logger.error(f"[INIT_ERROR] Enhanced integration initialization failed: {e}")
            logger.error("[ACTION_REQUIRED] Please check your configuration:")
            logger.error("  1. Ensure Run_Strategy sheet exists and is not empty")
            logger.error("  2. Verify at least one strategy is enabled in Run_Strategy")
            logger.error("  3. Check that Strategy_Config has matching active strategies")
            logger.error("  4. No fallbacks available - configuration must be correct")
            return 1
        except Exception as e:
            logger.error(f"[INIT_ERROR] Unexpected error during enhanced integration init: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
        
        # Enhanced strategy selection validation with detailed error reporting
        logger.info("Validating enhanced strategy selection...")
        try:
            strategy_summary = integration.config.get('strategy_summary', {})
            logger.info(f"Strategy Selection Summary:")
            logger.info(f"  Strategies in Strategy_Config: {strategy_summary.get('total_strategies_in_config', 0)}")
            logger.info(f"  Strategies in Run_Strategy: {strategy_summary.get('total_strategies_in_run_strategy', 0)}")
            logger.info(f"  Active strategies selected: {strategy_summary.get('active_strategies_in_run_strategy', 0)}")
            
            active_strategies = list(integration.config['active_strategies'].keys())
            if not active_strategies:
                logger.error("[STRATEGY_ERROR] No active strategies found after enhanced filtering")
                logger.error("[ACTION_REQUIRED] Check the following:")
                logger.error("  1. Run_Strategy sheet: Ensure strategies have Run_Status = True/Active")
                logger.error("  2. Strategy_Config sheet: Ensure strategies have Active = True")
                logger.error("  3. Strategy name matching: Names must match exactly between sheets")
                logger.error("  4. No 'nan' or empty strategy names allowed")
                raise ValueError("No active strategies available for backtesting")
            
            # Log selected strategies with details
            logger.info(f"Successfully selected {len(active_strategies)} strategies for backtesting:")
            for strategy_name in active_strategies:
                strategy_config = integration.config['active_strategies'][strategy_name]
                entry_rules = strategy_config.get('entry_rules', [])
                exit_rules = strategy_config.get('exit_rules', [])
                entry_rule = entry_rules[0] if entry_rules else 'Unknown'
                exit_rule = exit_rules[0] if exit_rules else 'Unknown'
                logger.info(f"  [OK] {strategy_name}: {entry_rule} -> {exit_rule}")
            
        except ValueError as e:
            # Re-raise ValueError for strategy issues
            logger.error(f"[STRATEGY_ERROR] {e}")
            return 1
        except Exception as e:
            logger.error(f"[STRATEGY_ERROR] Enhanced strategy validation failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
        
        # Handle info and validation commands
        if args.info:
            logger.info("Enhanced Backtesting Engine Information:")
            logger.info(f"Active Strategies: {list(integration.config['active_strategies'].keys())}")
            logger.info(f"Real Money Integration: {'Enabled' if enable_real_money else 'Disabled'}")
            logger.info(f"Initial Capital: ${args.capital:,.2f}")
            return 0
        
        if args.validate:
            logger.info("Enhanced configuration validation...")
            
            # Validate Run_Strategy configuration
            validation_issues = integration.enhanced_rule_engine.condition_loader.validate_run_strategy_configuration()
            if validation_issues:
                logger.warning("Run_Strategy validation issues found:")
                for issue in validation_issues:
                    logger.warning(f"  - {issue}")
                if any("not found in Strategy_Config" in issue for issue in validation_issues):
                    logger.error("Critical validation errors found - backtesting may fail")
                    return 1
            else:
                logger.info("Run_Strategy validation passed")
            
            logger.info("Enhanced configuration validation completed")
            return 0
        
        if args.show_config_dates:
            logger.info("Displaying config date range...")
            try:
                # Show date range from config
                config_cache = get_cached_config(integration.config_path)
                config_loader = config_cache.get_config_loader()
                init_config = config_loader.get_initialize_config()
                
                config_start = init_config.get('start_date')
                config_end = init_config.get('end_date')
                
                if config_start and config_end:
                    start_parsed = _parse_config_date(config_start, 'start_date', logger)
                    end_parsed = _parse_config_date(config_end, 'end_date', logger)
                    
                    logger.info(f"[CONFIG] Date range in config file:")
                    logger.info(f"   Raw start_date: {config_start}")
                    logger.info(f"   Raw end_date: {config_end}")
                    logger.info(f"   Parsed start_date: {start_parsed}")
                    logger.info(f"   Parsed end_date: {end_parsed}")
                    
                    if start_parsed and end_parsed:
                        start_dt = dt_module.datetime.strptime(start_parsed, '%Y-%m-%d')
                        end_dt = dt_module.datetime.strptime(end_parsed, '%Y-%m-%d')
                        days_diff = (end_dt - start_dt).days + 1
                        logger.info(f"   Date range spans: {days_diff} days")
                    
                else:
                    logger.error(f"[CONFIG] Date configuration missing in Initialize sheet")
                    logger.error(f"   start_date: {config_start}")
                    logger.error(f"   end_date: {config_end}")
                    
                return 0
                
            except Exception as e:
                logger.error(f"[ERROR] Error reading config dates: {e}")
                return 1

        if args.validate_data:
            logger.info("Validating data availability...")
            try:
                # Determine execution parameters for validation
                symbols = determine_symbols(args, integration)
                primary_symbol = symbols[0] if symbols else 'NIFTY'
                start_date, end_date = determine_date_range(args, integration, primary_symbol)
                
                # Validate data for each symbol
                for symbol in symbols:
                    logger.info(f"Validating data for {symbol}...")
                    db_manager = DatabaseManager()
                    exchange = _determine_exchange_for_symbol(symbol, db_manager, logger)
                    _validate_data_availability(symbol, exchange, start_date, end_date, db_manager, logger)
                
                logger.info("[OK] Data validation passed for all symbols")
                return 0
                
            except Exception as e:
                logger.error(f"[ERROR] Data validation failed: {e}")
                return 1
        
        # Determine execution parameters
        symbols = determine_symbols(args, integration)
        strategies = determine_strategies(args, integration)
        # Use first symbol for date range determination
        primary_symbol = symbols[0] if symbols else 'NIFTY'
        start_date, end_date = determine_date_range(args, integration, primary_symbol)
        
        if not symbols:
            logger.error("No symbols specified for backtesting")
            return 1
        
        if not strategies:
            logger.error("No strategies available for backtesting")
            return 1
        
        # Log execution parameters
        if not args.quiet:
            logger.info(f"Symbols: {symbols}")
            logger.info(f"Strategies: {strategies}")
            logger.info(f"Date range: {start_date} to {end_date}")
            logger.info(f"Real Money Mode: {'Enabled' if enable_real_money else 'Disabled'}")
        
        # Execute enhanced backtest with strict validation
        start_time = time.time()
        results = {}
        failed_symbols = []
        
        for symbol in symbols:
            logger.info(f"Running enhanced backtest for {symbol}...")
            
            # Run enhanced backtest for this symbol
            symbol_result = run_enhanced_symbol_backtest(
                symbol=symbol,
                start_date=start_date,
                end_date=end_date,
                integration=integration,
                strategy_name=strategies[0] if strategies else None
            )
            
            # Check for data validation failures
            if isinstance(symbol_result, dict) and 'error' in symbol_result:
                failed_symbols.append(symbol)
                logger.error(f"[ERROR] Backtest failed for {symbol}: {symbol_result['error']}")
            
            results[symbol] = symbol_result
        
        # Stop execution if any symbols failed due to data issues
        if failed_symbols:
            logger.error(f"[ERROR] BACKTESTING STOPPED - Data validation failed for symbols: {', '.join(failed_symbols)}")
            logger.error("[ACTIONS] REQUIRED ACTIONS:")
            logger.error("   1. Verify symbol names are correct")
            logger.error("   2. Check date range has available data")
            logger.error("   3. Load missing data using data loading process")
            logger.error("   4. Use --validate-data flag to check data availability")
            return 1
        
        execution_time = time.time() - start_time
        
        # Generate enhanced summary
        if not args.quiet:
            generate_enhanced_summary(results, symbols, execution_time, logger)
        
        # Generate enhanced reports
        if not args.no_excel:
            logger.info("Generating enhanced reports...")
            
            try:
                # Create output directory with date-based organization
                output_dir = Path(args.output_dir)
                current_date = datetime.now().strftime("%Y-%m-%d")
                dated_output_dir = output_dir / current_date
                dated_output_dir.mkdir(parents=True, exist_ok=True)
                
                # Export comprehensive reports for each symbol
                for symbol, result in results.items():
                    if isinstance(result, dict) and 'error' not in result:
                        try:
                            # Export enhanced analysis reports
                            if args.export_detailed:
                                exported_files = integration.export_comprehensive_report(
                                    str(dated_output_dir)
                                )
                                
                                if exported_files:
                                    logger.info(f"Enhanced reports for {symbol}:")
                                    for report_type, file_path in exported_files.items():
                                        logger.info(f"  {report_type.upper()}: {file_path}")
                        
                        except Exception as e:
                            logger.warning(f"Error generating enhanced reports for {symbol}: {e}")
                
                logger.info(f"All enhanced reports saved in: {dated_output_dir}")
                
            except Exception as e:
                logger.error(f"Error generating enhanced reports: {e}")
        
        # High-level execution summary
        successful_results = sum(1 for result in results.values() 
                               if isinstance(result, dict) and 'error' not in result)
        
        logger.info("=" * 60)
        logger.info("ENHANCED BACKTESTING EXECUTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Symbols Processed: {successful_results}/{len(symbols)}")
        logger.info(f"Execution Time: {execution_time:.2f} seconds")
        logger.info(f"Real Money Integration: {'[ACTIVE]' if enable_real_money else '[INACTIVE]'}")
        logger.info(f"Enhanced Features: [ACTIVE] Parallel Tracking, Exit Analysis, Options P&L")
        logger.info("=" * 60)
        
        logger.info("Enhanced backtesting completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Enhanced backtesting interrupted by user")
        return 1
    except Exception as e:
        if 'logger' in locals():
            logger.error(f"Enhanced backtesting failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        else:
            print(f"[ERROR] Enhanced backtesting failed: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)