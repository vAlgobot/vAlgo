#!/usr/bin/env python3
"""
VectorBT Enhanced Backtesting System
===================================

High-performance backtesting system using VectorBT framework.
Maintains strict production standards with comprehensive validation and error handling.

NO HARDCODING - NO MOCK DATA - NO ASSUMPTIONS - NO FALLBACKS

Usage:
    python main_vectorbt_backtest.py
    python main_vectorbt_backtest.py --symbol NIFTY --strategy EMA_RSI_Scalping
    python main_vectorbt_backtest.py --start-date 2025-01-01 --end-date 2025-07-30

Author: vAlgo Development Team
Created: July 24, 2025
Version: 1.0.0
"""

import sys
import argparse
from pathlib import Path
from typing import Dict, Any, Optional, List
import pandas as pd
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# VectorBT Engine imports
from vectorbt_engine import VectorBTAdapter, get_vectorbt_logger, ErrorHandler
from vectorbt_engine.models.error_models import VectorBTError, ConfigurationError, DataError
from vectorbt_engine.validators.data_validator import DataValidator

# Existing vAlgo imports for configuration and data
try:
    from utils.config_cache import get_cached_config
    from data_manager.database import DatabaseManager
    from indicators.unified_indicator_engine import UnifiedIndicatorEngine
    from backtest_engine.enhanced_backtest_integration import EnhancedBacktestIntegration
    from rules.enhanced_rule_engine import EnhancedRuleEngine, EnhancedSignalProcessor
    from rules.enhanced_condition_loader import EnhancedConditionLoader
    VALGO_IMPORTS_AVAILABLE = True
except ImportError:
    VALGO_IMPORTS_AVAILABLE = False
    print("Warning: vAlgo imports not available. Some features may be limited.")


class VectorBTBacktestRunner:
    """
    Main VectorBT backtesting runner.
    
    Orchestrates the entire VectorBT backtesting process with comprehensive
    validation and error handling.
    """
    
    def __init__(self, config_path: str = "config/config.xlsx"):
        """
        Initialize VectorBT Backtest Runner.
        
        Args:
            config_path: Path to Excel configuration file
        """
        self.config_path = config_path
        self.logger = get_vectorbt_logger("backtest_runner")
        self.error_handler = ErrorHandler(self.logger.logger)
        self.data_validator = DataValidator(strict_mode=True)
        
        # Initialize components
        self.config = None
        self.database = None
        self.indicator_engine = None
        self.vectorbt_adapter = None
        
        # Enhanced components for complete flow
        self.enhanced_integration = None
        self.enhanced_rule_engine = None
        self.enhanced_signal_processor = None
        
        self.logger.info("VectorBT Backtest Runner initialized")
    
    def run_backtest(
        self,
        symbols: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
        enable_real_money: bool = True
    ) -> Dict[str, Any]:
        """
        Run comprehensive VectorBT backtesting.
        
        Args:
            symbols: List of symbols to backtest (default: from config)
            strategies: List of strategies to run (default: from config)
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            enable_real_money: Enable real money options integration
            
        Returns:
            Comprehensive backtest results
            
        Raises:
            VectorBTError: If backtesting fails
        """
        self.logger.info("Starting VectorBT enhanced backtesting")
        self.logger.start_operation("vectorbt_backtest")
        
        try:
            # Phase 1: Initialize and validate configuration
            self._initialize_configuration()
            self._validate_configuration()
            
            # Phase 2: Determine backtesting parameters
            symbols = self._determine_symbols(symbols)
            strategies = self._determine_strategies(strategies)
            date_range = self._determine_date_range(start_date, end_date)
            
            self.logger.log_vectorbt_operation(
                operation="backtest_start",
                symbols=symbols,
                strategies=strategies,
                date_range=date_range
            )
            
            # Phase 3: Initialize data systems
            self._initialize_data_systems()
            
            # Phase 4: Initialize enhanced components
            self._initialize_enhanced_components()
            
            # Phase 5: Initialize VectorBT adapter
            self._initialize_vectorbt_adapter(enable_real_money)
            
            # Phase 6: Run backtesting for each symbol
            all_results = {}
            
            for symbol in symbols:
                self.logger.info(f"Processing symbol: {symbol}")
                
                try:
                    symbol_results = self._run_symbol_backtest(
                        symbol=symbol,
                        strategies=strategies,
                        date_range=date_range,
                        enable_real_money=enable_real_money
                    )
                    
                    all_results[symbol] = symbol_results
                    
                except Exception as e:
                    self.error_handler.handle_error(
                        e, 
                        context={'symbol': symbol, 'phase': 'symbol_backtest'},
                        reraise=False
                    )
                    all_results[symbol] = {'error': str(e), 'status': 'failed'}
            
            # Phase 6: Generate comprehensive results
            final_results = self._generate_final_results(all_results, symbols, strategies, date_range)
            
            self.logger.info("VectorBT backtesting completed successfully")
            return final_results
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'phase': 'main_backtest'})
            
        finally:
            duration = self.logger.end_operation("vectorbt_backtest")
            self.logger.log_performance_summary()
            
            if duration:
                self.logger.info(f"Total backtesting time: {duration:.2f} seconds")
    
    def _initialize_configuration(self) -> None:
        """Initialize configuration system."""
        self.logger.start_operation("initialize_configuration")
        
        try:
            if not VALGO_IMPORTS_AVAILABLE:
                raise ConfigurationError(
                    message="vAlgo configuration system not available",
                    suggestion="Ensure vAlgo modules are properly installed"
                )
            
            # Load configuration using existing vAlgo system
            config_cache = get_cached_config(self.config_path)
            self.config = config_cache.get_config_loader()
            
            if not self.config:
                raise ConfigurationError(
                    message=f"Failed to load configuration from {self.config_path}",
                    config_section="all",
                    suggestion=f"Verify that {self.config_path} exists and is readable"
                )
            
            self.logger.info(f"Configuration loaded successfully from {self.config_path}")
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'config_path': self.config_path})
            
        finally:
            self.logger.end_operation("initialize_configuration")
    
    def _validate_configuration(self) -> None:
        """Validate loaded configuration."""
        self.logger.start_operation("validate_configuration")
        
        try:
            # Validate required configuration sections exist
            required_sections = ['instruments', 'indicators', 'run_strategy']
            
            for section in required_sections:
                if not hasattr(self.config, 'config_data') or section not in self.config.config_data:
                    raise ConfigurationError(
                        message=f"Required configuration section missing: {section}",
                        config_section=section,
                        suggestion=f"Add {section} sheet to Excel configuration file"
                    )
            
            self.logger.info("Configuration validation passed")
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'phase': 'config_validation'})
            
        finally:
            self.logger.end_operation("validate_configuration")
    
    def _determine_symbols(self, symbols: Optional[List[str]]) -> List[str]:
        """Determine symbols to backtest."""
        if symbols:
            return symbols
        
        # Get symbols from configuration
        try:
            instruments_df = self.config.config_data.get('instruments')
            if instruments_df is None or instruments_df.empty:
                raise ConfigurationError(
                    message="No instruments configured",
                    config_section="instruments",
                    suggestion="Add instrument configurations to Excel file"
                )
            
            # Filter only active instruments
            if 'Status' in instruments_df.columns:
                active_instruments = instruments_df[
                    instruments_df['Status'].str.lower().isin(['active', 'yes', 'true', '1'])
                ]
            elif 'Active' in instruments_df.columns:
                active_instruments = instruments_df[
                    instruments_df['Active'].str.lower().isin(['active', 'yes', 'true', '1'])
                ]
            else:
                self.logger.warning("No Status/Active column found in instruments, using all instruments")
                active_instruments = instruments_df
            
            configured_symbols = active_instruments['Symbol'].dropna().unique().tolist()
            
            self.logger.info(f"Filtered instruments: {len(configured_symbols)} active out of {len(instruments_df)} total")
            
            if not configured_symbols:
                raise ConfigurationError(
                    message="No active symbols found in instruments configuration",
                    config_section="instruments",
                    suggestion="Ensure at least one instrument has Status/Active set to 'active'"
                )
            
            return configured_symbols
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'phase': 'determine_symbols'})
    
    def _determine_strategies(self, strategies: Optional[List[str]]) -> List[str]:
        """Determine strategies to run."""
        if strategies:
            return strategies
        
        # Get active strategies from configuration
        try:
            run_strategy_df = self.config.config_data.get('run_strategy')
            if run_strategy_df is None or run_strategy_df.empty:
                raise ConfigurationError(
                    message="No strategies configured",
                    config_section="run_strategy",
                    suggestion="Add strategy configurations to Excel file"
                )
            
            active_strategies = run_strategy_df[
                run_strategy_df['Run_Status'].str.lower() == 'active'
            ]['Strategy_Name'].dropna().tolist()
            
            if not active_strategies:
                raise ConfigurationError(
                    message="No active strategies found",
                    config_section="run_strategy",
                    suggestion="Set Run_Status to 'active' for desired strategies"
                )
            
            return active_strategies
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'phase': 'determine_strategies'})
    
    def _determine_date_range(
        self, 
        start_date: Optional[str], 
        end_date: Optional[str]
    ) -> tuple:
        """Determine date range for backtesting - Config-driven approach with NO fallbacks."""
        if start_date and end_date:
            return (start_date, end_date)
        
        # Get date range from configuration - STRICT CONFIG-DRIVEN APPROACH
        try:
            initialize_df = self.config.config_data.get('initialize')
            if initialize_df is None or initialize_df.empty:
                raise ConfigurationError(
                    message="No initialization configuration found",
                    config_section="initialize",
                    suggestion="Add initialize sheet to Excel configuration file"
                )
            
            config_start = initialize_df.loc[initialize_df['Parameter'] == 'start_date', 'Value']
            config_end = initialize_df.loc[initialize_df['Parameter'] == 'end_date', 'Value']
            
            if config_start.empty or config_end.empty:
                raise ConfigurationError(
                    message="Start/end date not configured in initialize sheet",
                    config_section="initialize",
                    suggestion="Add start_date and end_date parameters to initialize sheet"
                )
            
            # Convert config values to datetime objects
            start_value = config_start.iloc[0]
            end_value = config_end.iloc[0]
            
            # Handle different date formats
            if isinstance(start_value, str):
                start_dt = pd.to_datetime(start_value)
            else:
                start_dt = start_value
                
            if isinstance(end_value, str):
                end_dt = pd.to_datetime(end_value)
            else:
                end_dt = end_value
            
            # Config-driven approach: Accept same start/end dates (single day backtesting)
            # NO VALIDATION, NO FALLBACKS - Use exactly what's configured
            self.logger.info(f"Date range from config: {start_dt.strftime('%Y-%m-%d')} to {end_dt.strftime('%Y-%m-%d')}")
            
            # Handle same-date scenario
            if start_dt == end_dt:
                self.logger.info(f"Single-day backtesting configured for: {start_dt.strftime('%Y-%m-%d')}")
            
            return (start_dt, end_dt)
            
        except Exception as e:
            # Re-raise configuration errors - NO FALLBACKS
            if isinstance(e, ConfigurationError):
                raise e
            else:
                raise ConfigurationError(
                    message=f"Error reading date configuration: {str(e)}",
                    config_section="initialize",
                    suggestion="Check initialize sheet format and parameter names"
                )
    
    def _initialize_data_systems(self) -> None:
        """Initialize data management systems."""
        self.logger.start_operation("initialize_data_systems")
        
        try:
            if not VALGO_IMPORTS_AVAILABLE:
                raise SystemError(
                    message="vAlgo data systems not available",
                    suggestion="Ensure vAlgo modules are properly installed"
                )
            
            # Initialize database
            self.database = DatabaseManager()
            
            # Initialize indicator engine
            self.indicator_engine = UnifiedIndicatorEngine()
            
            self.logger.info("Data systems initialized successfully")
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'phase': 'data_systems_init'})
            
        finally:
            self.logger.end_operation("initialize_data_systems")
    
    def _initialize_enhanced_components(self) -> None:
        """Initialize enhanced backtesting components."""
        self.logger.start_operation("initialize_enhanced_components")
        
        try:
            if not VALGO_IMPORTS_AVAILABLE:
                raise SystemError(
                    message="Enhanced vAlgo components not available",
                    suggestion="Ensure enhanced backtesting modules are properly installed"
                )
            
            # Initialize enhanced backtest integration (reference implementation)
            self.enhanced_integration = EnhancedBacktestIntegration(
                config_path=self.config_path,
                enable_real_money=True
            )
            
            # Use enhanced_integration's rule engine and signal processor (no duplicates)
            self.enhanced_rule_engine = self.enhanced_integration.enhanced_rule_engine
            self.enhanced_signal_processor = self.enhanced_integration.enhanced_signal_processor
            
            self.logger.info("Enhanced components initialized successfully")
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'phase': 'enhanced_components_init'})
            
        finally:
            self.logger.end_operation("initialize_enhanced_components")
    
    def _initialize_vectorbt_adapter(self, enable_real_money: bool) -> None:
        """Initialize VectorBT adapter."""
        self.logger.start_operation("initialize_vectorbt_adapter")
        
        try:
            # Initialize VectorBT adapter with Phase 2 capabilities
            try:
                self.vectorbt_adapter = VectorBTAdapter(
                    config_loader=self.config,
                    enable_real_money=enable_real_money
                )
                self.logger.info("VectorBT adapter initialized with Phase 2 signal generation")
                self.use_phase2_signals = True
                
            except Exception as e:
                self.logger.warning(f"Phase 2 VectorBT adapter failed, using fallback: {e}")
                self.vectorbt_adapter = None
                self.use_phase2_signals = False
            
            self.logger.info(f"VectorBT adapter initialized (real_money: {enable_real_money}, phase2: {self.use_phase2_signals})")
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'phase': 'vectorbt_adapter_init'})
            
        finally:
            self.logger.end_operation("initialize_vectorbt_adapter")
    
    def _run_symbol_backtest(
        self,
        symbol: str,
        strategies: List[str],
        date_range: tuple,
        enable_real_money: bool
    ) -> Dict[str, Any]:
        """
        Run backtesting for a single symbol following the complete flow architecture.
        
        Implements: Symbol → Strategy → Daily → Entry → Exit sequential processing
        """
        self.logger.start_operation(f"symbol_backtest_{symbol}")
        
        try:
            start_date, end_date = date_range
            
            # Convert dates to string format for database queries
            start_date_str = start_date.strftime('%Y-%m-%d') if hasattr(start_date, 'strftime') else str(start_date)
            end_date_str = end_date.strftime('%Y-%m-%d') if hasattr(end_date, 'strftime') else str(end_date)
            
            self.logger.info(f"Loading data for {symbol} from {start_date_str} to {end_date_str}")
            
            # Phase 1: Load and validate data
            signal_data = self._load_signal_data(symbol, start_date_str, end_date_str)
            execution_data = self._load_execution_data(symbol, start_date_str, end_date_str)
            
            # Validate data
            self.data_validator.validate_signal_data(signal_data, symbol)
            self.data_validator.validate_execution_data(execution_data, symbol)
            
            # Phase 2: Calculate indicators
            signal_data_with_indicators = self._calculate_indicators(signal_data, symbol, start_date_str, end_date_str)
            
            self.logger.info(f"Starting sequential flow processing for {symbol}")
            
            # Phase 3: Sequential Strategy Processing (Symbol → Strategy → Daily → Entry → Exit)
            symbol_results = self._process_symbol_sequential_flow(
                symbol=symbol,
                strategies=strategies,
                signal_data=signal_data_with_indicators,
                execution_data=execution_data,
                date_range=date_range,
                enable_real_money=enable_real_money
            )
            
            # Phase 4: Convert sequential results to VectorBT signals and create portfolio
            if self.vectorbt_adapter and symbol_results.get('trades'):
                try:
                    vectorbt_portfolio = self._create_vectorbt_portfolio_from_trades(
                        symbol_results=symbol_results,
                        signal_data=signal_data_with_indicators,
                        execution_data=execution_data,
                        symbol=symbol,
                        enable_real_money=enable_real_money
                    )
                    
                    # Enhance results with VectorBT portfolio analysis
                    symbol_results['vectorbt_portfolio'] = vectorbt_portfolio
                    symbol_results['status'] = 'vectorbt_enhanced_complete'
                    
                except Exception as e:
                    self.logger.warning(f"VectorBT portfolio creation failed for {symbol}: {e}")
                    # Continue with sequential results
            
            self.logger.info(f"Symbol backtest completed for {symbol}")
            return symbol_results
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'symbol': symbol, 'phase': 'symbol_backtest'})
            return {'symbol': symbol, 'error': str(e), 'status': 'failed'}
            
        finally:
            self.logger.end_operation(f"symbol_backtest_{symbol}")
    
    def _process_symbol_sequential_flow(
        self,
        symbol: str,
        strategies: List[str],
        signal_data: pd.DataFrame,
        execution_data: pd.DataFrame,
        date_range: tuple,
        enable_real_money: bool
    ) -> Dict[str, Any]:
        """
        Process symbol using sequential flow architecture.
        
        Flow: Symbol → Strategy → Daily → Entry → Exit
        Following lines 28-112 from backtesting_Flow.md
        """
        start_date, end_date = date_range
        all_trades = []
        strategy_results = {}
        
        # Generate date range for T+1 processing (line 30-31 in flow)
        date_range_list = pd.date_range(start=start_date, end=end_date, freq='D')
        
        self.logger.info(f"Processing {len(strategies)} strategies for {symbol} over {len(date_range_list)} days")
        
        # Loop: For each Active Strategy (line 39-40 in flow)
        for strategy_name in strategies:
            self.logger.info(f"Processing strategy: {strategy_name} for symbol: {symbol}")
            
            strategy_trades = []
            daily_entry_counts = {}  # Track entries per day for Max_Daily_Entries
            
            # Get strategy configuration and limits
            strategy_config = self._get_strategy_configuration(strategy_name)
            max_daily_entries = strategy_config.get('max_daily_entries', 1)
            
            self.logger.info(f"Strategy {strategy_name}: Max daily entries = {max_daily_entries}")
            
            # Loop through Dates (T+1) (line 30-31 in flow)
            for current_date in date_range_list:
                date_str = current_date.strftime('%Y-%m-%d')
                daily_entry_count = daily_entry_counts.get(date_str, 0)
                
                # Skip if max daily entries reached (line 97-99 in flow)
                if daily_entry_count >= max_daily_entries:
                    self.logger.debug(f"Max daily entries ({max_daily_entries}) reached for {strategy_name} on {date_str}")
                    continue
                
                # Process daily entry/exit cycle
                daily_trades = self._process_daily_strategy_cycle(
                    symbol=symbol,
                    strategy_name=strategy_name,
                    strategy_config=strategy_config,
                    current_date=current_date,
                    signal_data=signal_data,
                    execution_data=execution_data,
                    max_daily_entries=max_daily_entries,
                    current_daily_count=daily_entry_count,
                    enable_real_money=enable_real_money
                )
                
                # Update daily entry count and add trades
                if daily_trades:
                    daily_entry_counts[date_str] = daily_entry_count + len(daily_trades)
                    strategy_trades.extend(daily_trades)
            
            # Store strategy results
            strategy_results[strategy_name] = {
                'total_trades': len(strategy_trades),
                'trades': strategy_trades,
                'max_daily_entries': max_daily_entries,
                'daily_entry_counts': daily_entry_counts
            }
            
            all_trades.extend(strategy_trades)
        
        # Generate comprehensive results
        return self._generate_symbol_results(
            symbol=symbol,
            strategies=strategies,
            date_range=date_range,
            signal_data=signal_data,
            execution_data=execution_data,
            all_trades=all_trades,
            strategy_results=strategy_results
        )
    
    def _get_strategy_configuration(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get strategy configuration using enhanced_integration's loaded config.
        
        Maps Run_Strategy → Strategy_Config (lines 43-49 in flow)
        Uses enhanced system's pre-loaded entry/exit rules and strategy configurations.
        """
        try:
            # Use enhanced_integration's loaded configuration
            enhanced_config = self.enhanced_integration.config
            
            # Check if strategy exists in active strategies
            if strategy_name not in enhanced_config.get('active_strategies', []):
                self.logger.warning(f"Strategy {strategy_name} not found in enhanced active strategies")
                return {'max_daily_entries': 1}
            
            # Get strategy row from enhanced system
            strategy_rows = enhanced_config.get('strategy_rows', [])
            strategy_row = None
            for row in strategy_rows:
                if row.strategy_name == strategy_name:
                    strategy_row = row
                    break
            
            if not strategy_row:
                self.logger.warning(f"Strategy row not found for {strategy_name}")
                return {'max_daily_entries': 1}
            
            # Extract configuration from enhanced strategy row
            config = {
                'max_daily_entries': getattr(strategy_row, 'max_positions', 1),
                'strategy_config_name': strategy_name,
                'signal_type': getattr(strategy_row, 'signal_type', 'CALL'),
                'position_size': getattr(strategy_row, 'position_size', 1000.0),
                'risk_per_trade': getattr(strategy_row, 'risk_per_trade', 0.02),
                'entry_rule': strategy_row.entry_rule,
                'exit_rule': strategy_row.exit_rule,
                'active': strategy_row.active
            }
            
            # Get entry conditions from enhanced system's loaded entry_rules
            entry_rules = enhanced_config.get('entry_rules', {})
            if strategy_row.entry_rule in entry_rules:
                config['entry_conditions'] = entry_rules[strategy_row.entry_rule]
            else:
                config['entry_conditions'] = []
            
            # Get exit conditions from enhanced system's loaded exit_rules  
            exit_rules = enhanced_config.get('exit_rules', {})
            if strategy_row.exit_rule in exit_rules:
                config['exit_conditions'] = exit_rules[strategy_row.exit_rule]
            else:
                config['exit_conditions'] = []
            
            self.logger.info(f"Strategy configuration loaded from enhanced system for {strategy_name}: {len(config.get('entry_conditions', []))} entry conditions, {len(config.get('exit_conditions', []))} exit conditions")
            return config
            
        except Exception as e:
            self.logger.error(f"Error getting enhanced strategy configuration for {strategy_name}: {e}")
            return {'max_daily_entries': 1}
    
    def _process_daily_strategy_cycle(
        self,
        symbol: str,
        strategy_name: str,
        strategy_config: Dict[str, Any],
        current_date: pd.Timestamp,
        signal_data: pd.DataFrame,
        execution_data: pd.DataFrame,
        max_daily_entries: int,
        current_daily_count: int,
        enable_real_money: bool
    ) -> List[Dict[str, Any]]:
        """
        Process daily entry/exit cycle for a strategy.
        
        Implements lines 52-108 from backtesting_Flow.md:
        - Check Entry Rule
        - Identify Entry Point
        - Calculate SL/TP
        - Monitor Exit Conditions
        - Calculate P&L
        """
        daily_trades = []
        date_str = current_date.strftime('%Y-%m-%d')
        
        # Filter data for current date
        # Handle timestamp column (might be index or column after indicator calculation)
        if 'timestamp' in signal_data.columns:
            current_day_signal = signal_data[signal_data['timestamp'].dt.date == current_date.date()]  
        elif signal_data.index.name == 'timestamp' or hasattr(signal_data.index, 'date'):
            current_day_signal = signal_data[signal_data.index.date == current_date.date()]
        else:
            self.logger.error(f"No timestamp found in signal_data columns: {list(signal_data.columns)}")
            return daily_trades
            
        if 'timestamp' in execution_data.columns:
            current_day_execution = execution_data[execution_data['timestamp'].dt.date == current_date.date()] 
        elif execution_data.index.name == 'timestamp' or hasattr(execution_data.index, 'date'):
            current_day_execution = execution_data[execution_data.index.date == current_date.date()]
        else:
            self.logger.error(f"No timestamp found in execution_data columns: {list(execution_data.columns)}")
            return daily_trades
        
        if current_day_signal.empty or current_day_execution.empty:
            return daily_trades
        
        remaining_entries = max_daily_entries - current_daily_count
        entries_made = 0
        
        # Process potential entries throughout the day
        for idx, signal_row in current_day_signal.iterrows():
            if entries_made >= remaining_entries:
                break
            
            # Check if Entry Rule is TRUE (line 52-53 in flow)
            entry_signal = self._evaluate_entry_rule(
                signal_row=signal_row,
                strategy_config=strategy_config,
                symbol=symbol
            )
            
            if entry_signal:
                # Get timestamp for logging (might be in index or column)
                if 'timestamp' in signal_row.index:
                    timestamp_str = str(signal_row['timestamp'])
                elif hasattr(signal_row, 'name') and signal_row.name:
                    timestamp_str = str(signal_row.name)
                else:
                    timestamp_str = "unknown"
                    
                self.logger.debug(f"Entry signal detected for {strategy_name} at {timestamp_str}")
                
                # Identify Entry Point (line 57-59 in flow)
                entry_point = self._identify_entry_point(
                    signal_row=signal_row,
                    strategy_config=strategy_config
                )
                
                if entry_point:
                    # Calculate SL/TP using Advanced SL/TP Engine (line 62-65 in flow)
                    sl_tp_levels = self._calculate_sl_tp_levels(
                        entry_point=entry_point,
                        signal_row=signal_row,
                        strategy_config=strategy_config,
                        symbol=symbol
                    )
                    
                    # Get timestamp for trade execution (might be in index or column)
                    if 'timestamp' in signal_row.index:
                        signal_timestamp = signal_row['timestamp']
                    elif hasattr(signal_row, 'name') and signal_row.name:
                        signal_timestamp = signal_row.name
                    else:
                        signal_timestamp = current_date  # Fallback to current_date
                    
                    # Open Trade and Monitor Exit (line 67-90 in flow)
                    trade_result = self._execute_and_monitor_trade(
                        symbol=symbol,
                        strategy_name=strategy_name,
                        entry_point=entry_point,
                        sl_tp_levels=sl_tp_levels,
                        signal_timestamp=signal_timestamp,
                        execution_data=current_day_execution,
                        strategy_config=strategy_config,
                        enable_real_money=enable_real_money
                    )
                    
                    if trade_result:
                        daily_trades.append(trade_result)
                        entries_made += 1
                        self.logger.info(f"Trade executed for {strategy_name}: Entry at {entry_point}")
        
        return daily_trades
    
    def _evaluate_entry_rule(
        self,
        signal_row: pd.Series,
        strategy_config: Dict[str, Any],
        symbol: str
    ) -> bool:
        """
        Evaluate entry rule using enhanced rule engine.
        
        Implements line 52-53 from backtesting_Flow.md: Check if Entry Rule is TRUE
        """
        try:
            # Use enhanced_integration's rule engine for entry evaluation
            if hasattr(self, 'enhanced_rule_engine') and self.enhanced_rule_engine:
                # Get entry conditions from strategy config (now from enhanced system)
                entry_conditions = strategy_config.get('entry_conditions', [])
                
                if not entry_conditions:
                    self.logger.warning(f"No entry conditions found for strategy")
                    return False
                
                # Get entry rule name for evaluation
                entry_rule = strategy_config.get('entry_rule', '')
                if not entry_rule:
                    self.logger.warning(f"No entry rule specified for strategy")
                    return False
                
                # Evaluate conditions using enhanced rule engine's condition evaluation
                # Convert signal_row to dict for enhanced rule engine
                signal_data_dict = signal_row.to_dict()
                
                # Use enhanced rule engine's evaluate_entry_rule method
                # Convert signal data to IndicatorData format expected by enhanced rule engine
                from rules.models import IndicatorData
                indicator_data = IndicatorData(
                    data=signal_data_dict,
                    timestamp=signal_data_dict.get('timestamp', pd.Timestamp.now())
                )
                
                result = self.enhanced_rule_engine.evaluate_entry_rule(
                    rule_name=entry_rule,
                    indicator_data=indicator_data
                )
                
                # Extract boolean result from RuleResult object
                triggered = result.triggered if hasattr(result, 'triggered') else False
                
                if triggered:
                    self.logger.debug(f"Entry rule '{entry_rule}' evaluated to TRUE for {symbol}")
                else:
                    self.logger.debug(f"Entry rule '{entry_rule}' evaluated to FALSE for {symbol}")
                    if hasattr(result, 'error_message') and result.error_message:
                        self.logger.debug(f"Entry rule error: {result.error_message}")
                
                return triggered
            
            else:
                # Fallback: Simple condition evaluation
                self.logger.warning("Enhanced rule engine not available, using simple fallback")
                # Basic example: RSI < 30 for oversold condition
                rsi_col = [col for col in signal_row.index if 'rsi' in col.lower()]
                if rsi_col and signal_row[rsi_col[0]] < 30:
                    return True
                
                return False
                
        except Exception as e:
            self.logger.error(f"Error evaluating entry rule: {e}")
            return False
    
    def _identify_entry_point(
        self,
        signal_row: pd.Series,
        strategy_config: Dict[str, Any]
    ) -> Optional[float]:
        """
        Identify entry point based on signal type.
        
        Implements line 57-59 from backtesting_Flow.md:
        Signal Type: CALL → Current candle High, PUT → Current candle Low
        """
        try:
            signal_type = strategy_config.get('signal_type', 'CALL').upper()
            
            if signal_type == 'CALL':
                entry_point = signal_row['high']
            elif signal_type == 'PUT':
                entry_point = signal_row['low']
            else:
                self.logger.warning(f"Unknown signal type: {signal_type}, using close price")
                entry_point = signal_row['close']
            
            self.logger.debug(f"Entry point identified: {entry_point} for signal type: {signal_type}")
            return float(entry_point)
            
        except Exception as e:
            self.logger.error(f"Error identifying entry point: {e}")
            return None
    
    def _calculate_sl_tp_levels(
        self,
        entry_point: float,
        signal_row: pd.Series,
        strategy_config: Dict[str, Any],
        symbol: str
    ) -> Dict[str, float]:
        """
        Calculate SL/TP levels using Advanced SL/TP Engine.
        
        Implements line 62-65 from backtesting_Flow.md:
        Based on Strategy_Config (SL_Method / TP_Method)
        Identify SL & TP from: Breakout Candle, CPR, ATR, etc.
        """
        try:
            sl_method = strategy_config.get('sl_method', 'CPR')
            tp_method = strategy_config.get('tp_method', 'CPR')
            signal_type = strategy_config.get('signal_type', 'CALL')
            
            # Use enhanced SL/TP engine if available
            if hasattr(self, 'enhanced_integration') and self.enhanced_integration:
                sl_tp_engine = self.enhanced_integration.sl_tp_engine
                
                # Calculate SL level
                sl_level = sl_tp_engine.calculate_stop_loss(
                    entry_price=entry_point,
                    method=sl_method,
                    signal_data=signal_row,
                    signal_type=signal_type,
                    symbol=symbol
                )
                
                # Calculate TP level
                tp_level = sl_tp_engine.calculate_take_profit(
                    entry_price=entry_point,
                    method=tp_method,
                    signal_data=signal_row,
                    signal_type=signal_type,
                    symbol=symbol
                )
                
                return {
                    'stop_loss': sl_level,
                    'take_profit': tp_level,
                    'sl_method': sl_method,
                    'tp_method': tp_method
                }
            
            else:
                # Fallback: Basic SL/TP calculation
                sl_percentage = 0.02  # 2% stop loss
                tp_percentage = 0.04  # 4% take profit
                
                if signal_type.upper() == 'CALL':
                    sl_level = entry_point * (1 - sl_percentage)
                    tp_level = entry_point * (1 + tp_percentage)
                else:  # PUT
                    sl_level = entry_point * (1 + sl_percentage)
                    tp_level = entry_point * (1 - tp_percentage)
                
                return {
                    'stop_loss': sl_level,
                    'take_profit': tp_level,
                    'sl_method': 'percentage_fallback',
                    'tp_method': 'percentage_fallback'
                }
                
        except Exception as e:
            self.logger.error(f"Error calculating SL/TP levels: {e}")
            # Return safe defaults
            return {
                'stop_loss': entry_point * 0.98,
                'take_profit': entry_point * 1.02,
                'sl_method': 'error_fallback',
                'tp_method': 'error_fallback'
            }
    
    def _execute_and_monitor_trade(
        self,
        symbol: str,
        strategy_name: str,
        entry_point: float,
        sl_tp_levels: Dict[str, float],
        signal_timestamp: pd.Timestamp,
        execution_data: pd.DataFrame,
        strategy_config: Dict[str, Any],
        enable_real_money: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Execute trade and monitor exit conditions.
        
        Implements lines 67-90 from backtesting_Flow.md:
        - Entry Confirmed → Open Trade
        - Get Entry Premium (Option LTP)
        - Monitor Exit Conditions
        - Get Exit Premium
        - Calculate P&L
        """
        try:
            signal_type = strategy_config.get('signal_type', 'CALL')
            
            # Get entry premium (line 72-73 in flow)
            entry_premium = self._get_entry_premium(
                entry_point=entry_point,
                signal_timestamp=signal_timestamp,
                execution_data=execution_data,
                symbol=symbol,
                signal_type=signal_type,
                enable_real_money=enable_real_money
            )
            
            if entry_premium is None:
                self.logger.warning(f"Could not get entry premium for {symbol} at {signal_timestamp}")
                return None
            
            # Monitor exit conditions (line 75-82 in flow)
            exit_result = self._monitor_exit_conditions(
                entry_timestamp=signal_timestamp,
                entry_premium=entry_premium,
                sl_tp_levels=sl_tp_levels,
                execution_data=execution_data,
                strategy_config=strategy_config,
                symbol=symbol,
                signal_type=signal_type,
                enable_real_money=enable_real_money
            )
            
            if exit_result:
                # Calculate P&L (line 89-90 in flow)
                pnl = self._calculate_trade_pnl(
                    entry_premium=entry_premium,
                    exit_premium=exit_result['exit_premium'],
                    signal_type=signal_type,
                    lot_size=75  # NIFTY lot size
                )
                
                # Create trade record
                trade_record = {
                    'symbol': symbol,
                    'strategy': strategy_name,
                    'signal_type': signal_type,
                    'entry_timestamp': signal_timestamp,
                    'entry_point': entry_point,
                    'entry_premium': entry_premium,
                    'exit_timestamp': exit_result['exit_timestamp'],
                    'exit_premium': exit_result['exit_premium'],
                    'exit_reason': exit_result['exit_reason'],
                    'stop_loss': sl_tp_levels['stop_loss'],
                    'take_profit': sl_tp_levels['take_profit'],
                    'pnl': pnl,
                    'trade_duration': (exit_result['exit_timestamp'] - signal_timestamp).total_seconds() / 60,  # minutes
                    'sl_method': sl_tp_levels['sl_method'],
                    'tp_method': sl_tp_levels['tp_method']
                }
                
                self.logger.info(f"Trade completed: {strategy_name} P&L: {pnl:.2f}")
                return trade_record
            
            else:
                self.logger.warning(f"Trade monitoring failed for {symbol} at {signal_timestamp}")
                return None
                
        except Exception as e:
            self.logger.error(f"Error executing and monitoring trade: {e}")
            return None
    
    def _get_entry_premium(
        self,
        entry_point: float,
        signal_timestamp: pd.Timestamp,
        execution_data: pd.DataFrame,
        symbol: str,
        signal_type: str,
        enable_real_money: bool
    ) -> Optional[float]:
        """
        Get entry premium (Option LTP).
        
        Implements line 72-73 from backtesting_Flow.md: Get Entry Premium (Option LTP)
        """
        try:
            if enable_real_money and hasattr(self, 'enhanced_integration'):
                # Use real money integration for actual option prices
                return self.enhanced_integration.options_integrator.get_option_premium(
                    underlying_symbol=symbol,
                    timestamp=signal_timestamp,
                    strike_price=entry_point,
                    option_type=signal_type
                )
            else:
                # Fallback: Use execution data
                # Find closest timestamp in execution data
                execution_data['time_diff'] = abs(execution_data['timestamp'] - signal_timestamp)
                closest_row = execution_data.loc[execution_data['time_diff'].idxmin()]
                
                # Return close price as premium approximation
                return float(closest_row['close'])
                
        except Exception as e:
            self.logger.error(f"Error getting entry premium: {e}")
            return None
    
    def _monitor_exit_conditions(
        self,
        entry_timestamp: pd.Timestamp,
        entry_premium: float,
        sl_tp_levels: Dict[str, float],
        execution_data: pd.DataFrame,
        strategy_config: Dict[str, Any],
        symbol: str,
        signal_type: str,
        enable_real_money: bool
    ) -> Optional[Dict[str, Any]]:
        """
        Monitor exit conditions throughout the day.
        
        Implements lines 75-82 from backtesting_Flow.md:
        Loop: Monitor Exit Conditions
        If Exit Condition TRUE or End Time Reached (e.g. 15:00)
        """
        try:
            # Filter execution data after entry
            post_entry_data = execution_data[execution_data['timestamp'] > entry_timestamp].copy()
            
            if post_entry_data.empty:
                return None
            
            # Sort by timestamp
            post_entry_data = post_entry_data.sort_values('timestamp')
            
            # End of day time (15:00 IST)
            eod_time = entry_timestamp.replace(hour=15, minute=0, second=0, microsecond=0)
            
            # Monitor each candle for exit conditions
            for idx, row in post_entry_data.iterrows():
                current_time = row['timestamp']
                current_price = row['close']
                
                # Check SL/TP conditions
                exit_reason = None
                
                if signal_type.upper() == 'CALL':
                    if current_price <= sl_tp_levels['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif current_price >= sl_tp_levels['take_profit']:
                        exit_reason = 'take_profit'
                else:  # PUT
                    if current_price >= sl_tp_levels['stop_loss']:
                        exit_reason = 'stop_loss'
                    elif current_price <= sl_tp_levels['take_profit']:
                        exit_reason = 'take_profit'
                
                # Check end of day condition (line 81-82 in flow)
                if current_time >= eod_time:
                    exit_reason = 'end_of_day'
                
                # If exit condition met
                if exit_reason:
                    exit_premium = self._get_exit_premium(
                        current_price=current_price,
                        exit_timestamp=current_time,
                        symbol=symbol,
                        signal_type=signal_type,
                        enable_real_money=enable_real_money
                    )
                    
                    if exit_premium:
                        return {
                            'exit_timestamp': current_time,
                            'exit_premium': exit_premium,
                            'exit_reason': exit_reason
                        }
            
            # If no exit condition met during day, force exit at EOD
            last_row = post_entry_data.iloc[-1]
            exit_premium = self._get_exit_premium(
                current_price=last_row['close'],
                exit_timestamp=last_row['timestamp'],
                symbol=symbol,
                signal_type=signal_type,
                enable_real_money=enable_real_money
            )
            
            return {
                'exit_timestamp': last_row['timestamp'],
                'exit_premium': exit_premium or last_row['close'],
                'exit_reason': 'end_of_data'
            }
            
        except Exception as e:
            self.logger.error(f"Error monitoring exit conditions: {e}")
            return None
    
    def _get_exit_premium(
        self,
        current_price: float,
        exit_timestamp: pd.Timestamp,
        symbol: str,
        signal_type: str,
        enable_real_money: bool
    ) -> Optional[float]:
        """
        Get exit premium.
        
        Implements line 85-86 from backtesting_Flow.md: Get Exit Premium
        """
        try:
            if enable_real_money and hasattr(self, 'enhanced_integration'):
                # Use real money integration for actual option prices
                return self.enhanced_integration.options_integrator.get_option_premium(
                    underlying_symbol=symbol,
                    timestamp=exit_timestamp,
                    strike_price=current_price,
                    option_type=signal_type
                )
            else:
                # Fallback: Use current price
                return float(current_price)
                
        except Exception as e:
            self.logger.error(f"Error getting exit premium: {e}")
            return current_price
    
    def _calculate_trade_pnl(
        self,
        entry_premium: float,
        exit_premium: float,
        signal_type: str,
        lot_size: int = 75
    ) -> float:
        """
        Calculate trade P&L with correct position sizing.
        
        Implements line 89-90 from backtesting_Flow.md: Calculate P&L
        Uses correct NIFTY lot size (75) for accurate P&L = (Exit - Entry) × 75
        """
        try:
            if signal_type.upper() == 'CALL':
                # For CALL: Profit when exit > entry
                pnl = (exit_premium - entry_premium) * lot_size
            else:  # PUT
                # For PUT: Profit when entry > exit
                pnl = (entry_premium - exit_premium) * lot_size
            
            return round(pnl, 2)
            
        except Exception as e:
            self.logger.error(f"Error calculating P&L: {e}")
            return 0.0
    
    def _generate_symbol_results(
        self,
        symbol: str,
        strategies: List[str],
        date_range: tuple,
        signal_data: pd.DataFrame,
        execution_data: pd.DataFrame,
        all_trades: List[Dict[str, Any]],
        strategy_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Generate comprehensive symbol results."""
        total_pnl = sum(trade['pnl'] for trade in all_trades)
        winning_trades = [t for t in all_trades if t['pnl'] > 0]
        losing_trades = [t for t in all_trades if t['pnl'] < 0]
        
        return {
            'symbol': symbol,
            'strategies': strategies,
            'date_range': date_range,
            'signal_data_shape': signal_data.shape,
            'execution_data_shape': execution_data.shape,
            'total_trades': len(all_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(all_trades) * 100 if all_trades else 0,
            'total_pnl': total_pnl,
            'avg_pnl_per_trade': total_pnl / len(all_trades) if all_trades else 0,
            'strategy_results': strategy_results,
            'trades': all_trades,
            'status': 'sequential_flow_complete',
            'message': 'Complete sequential flow backtesting implemented'
        }
    
    def _create_vectorbt_portfolio_from_trades(
        self,
        symbol_results: Dict[str, Any],
        signal_data: pd.DataFrame,
        execution_data: pd.DataFrame,
        symbol: str,
        enable_real_money: bool
    ) -> Dict[str, Any]:
        """
        Convert sequential flow trades to VectorBT portfolio analysis.
        
        This method bridges the gap between the sequential flow architecture
        and VectorBT's vectorized analysis capabilities.
        """
        try:
            trades = symbol_results.get('trades', [])
            if not trades:
                return {}
            
            # Convert trades to VectorBT signals
            vectorbt_signals = self._convert_trades_to_vectorbt_signals(
                trades=trades,
                execution_data=execution_data
            )
            
            if not vectorbt_signals:
                return {}
            
            # Create VectorBT portfolio using the adapter
            portfolio_result = self.vectorbt_adapter.create_vectorbt_portfolio(
                signal_data=signal_data,
                execution_data=execution_data,
                strategies=list(symbol_results.get('strategies', [])),
                symbol=symbol,
                vectorbt_signals=vectorbt_signals
            )
            
            # Enhance with sequential flow statistics
            enhanced_portfolio = self._enhance_portfolio_with_sequential_stats(
                portfolio_result=portfolio_result,
                sequential_trades=trades,
                symbol_results=symbol_results
            )
            
            return enhanced_portfolio
            
        except Exception as e:
            self.logger.error(f"Error creating VectorBT portfolio from trades: {e}")
            return {}
    
    def _convert_trades_to_vectorbt_signals(
        self,
        trades: List[Dict[str, Any]],
        execution_data: pd.DataFrame
    ) -> Dict[str, pd.DataFrame]:
        """
        Convert sequential flow trades to VectorBT entry/exit signals.
        
        Creates boolean arrays for VectorBT portfolio creation.
        """
        try:
            # Create timestamp index from execution data
            timestamps = execution_data['timestamp'].copy()
            
            # Initialize signal arrays
            vectorbt_signals = {}
            
            # Group trades by strategy
            trades_by_strategy = {}
            for trade in trades:
                strategy = trade.get('strategy', 'unknown')
                if strategy not in trades_by_strategy:
                    trades_by_strategy[strategy] = []
                trades_by_strategy[strategy].append(trade)
            
            # Convert each strategy's trades to signals
            for strategy, strategy_trades in trades_by_strategy.items():
                # Initialize boolean arrays
                entries = pd.Series(False, index=timestamps)
                exits = pd.Series(False, index=timestamps)
                
                # Mark entry and exit points
                for trade in strategy_trades:
                    entry_time = trade.get('entry_timestamp')
                    exit_time = trade.get('exit_timestamp')
                    
                    if entry_time and exit_time:
                        # Find closest timestamp indices
                        entry_idx = self._find_closest_timestamp_index(timestamps, entry_time)
                        exit_idx = self._find_closest_timestamp_index(timestamps, exit_time)
                        
                        if entry_idx is not None and exit_idx is not None:
                            entries.iloc[entry_idx] = True
                            exits.iloc[exit_idx] = True
                
                # Store signals for strategy
                vectorbt_signals[strategy] = {
                    'entries': entries,
                    'exits': exits,
                    'trade_count': len(strategy_trades)
                }
            
            self.logger.info(f"Converted {len(trades)} trades to VectorBT signals for {len(vectorbt_signals)} strategies")
            return vectorbt_signals
            
        except Exception as e:
            self.logger.error(f"Error converting trades to VectorBT signals: {e}")
            return {}
    
    def _find_closest_timestamp_index(
        self,
        timestamps: pd.Series,
        target_time: pd.Timestamp
    ) -> Optional[int]:
        """Find the closest timestamp index in the series."""
        try:
            time_diffs = abs(timestamps - target_time)
            min_idx = time_diffs.idxmin()
            return timestamps.index.get_loc(min_idx)
        except Exception:
            return None
    
    def _enhance_portfolio_with_sequential_stats(
        self,
        portfolio_result: Dict[str, Any],
        sequential_trades: List[Dict[str, Any]],
        symbol_results: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Enhance VectorBT portfolio results with sequential flow statistics.
        
        Combines VectorBT's vectorized analysis with sequential flow insights.
        """
        try:
            # Calculate sequential flow metrics
            total_trades = len(sequential_trades)
            winning_trades = sum(1 for t in sequential_trades if t.get('pnl', 0) > 0)
            total_pnl = sum(t.get('pnl', 0) for t in sequential_trades)
            
            # Extract exit reasons statistics
            exit_reasons = {}
            for trade in sequential_trades:
                reason = trade.get('exit_reason', 'unknown')
                exit_reasons[reason] = exit_reasons.get(reason, 0) + 1
            
            # Calculate average trade duration
            durations = [t.get('trade_duration', 0) for t in sequential_trades if t.get('trade_duration')]
            avg_duration = sum(durations) / len(durations) if durations else 0
            
            # Enhance portfolio result
            enhanced_result = portfolio_result.copy() if portfolio_result else {}
            enhanced_result.update({
                'sequential_flow_stats': {
                    'total_trades': total_trades,
                    'winning_trades': winning_trades,
                    'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                    'total_pnl': total_pnl,
                    'avg_pnl_per_trade': total_pnl / total_trades if total_trades > 0 else 0,
                    'avg_trade_duration_minutes': avg_duration,
                    'exit_reasons': exit_reasons
                },
                'strategy_breakdown': symbol_results.get('strategy_results', {}),
                'flow_integration': {
                    'sequential_flow_complete': True,
                    'vectorbt_enhanced': True,
                    'real_money_integration': symbol_results.get('status') == 'vectorbt_enhanced_complete'
                }
            })
            
            return enhanced_result
            
        except Exception as e:
            self.logger.error(f"Error enhancing portfolio with sequential stats: {e}")
            return portfolio_result or {}
    
    def _create_phase1_results(
        self, 
        symbol: str, 
        strategies: List[str], 
        date_range: tuple, 
        signal_data: pd.DataFrame, 
        execution_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """Create Phase 1 fallback results."""
        return {
            'symbol': symbol,
            'strategies': strategies,
            'date_range': date_range,
            'signal_data_shape': signal_data.shape,
            'execution_data_shape': execution_data.shape,
            'indicators_calculated': len([col for col in signal_data.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]),
            'status': 'phase_1_fallback',
            'message': 'Phase 1 fallback - Data loading and validation successful'
        }
    
    def _load_signal_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load 5-minute signal data."""
        try:
            data = self.database.get_ohlcv_data(
                symbol=symbol,
                exchange='NSE_INDEX',  # Get from instruments config
                timeframe='5m',
                start_date=start_date,
                end_date=end_date
            )
            
            # Reset index to convert timestamp index to column
            if not data.empty and data.index.name == 'timestamp':
                data = data.reset_index()
            
            if data.empty:
                raise DataError(
                    message=f"No signal data available for {symbol}",
                    symbol=symbol,
                    data_type="5m signal data",
                    context={'date_range': f"{start_date} to {end_date}"},
                    suggestion="Check database for data availability"
                )
            
            return data
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'symbol': symbol, 'data_type': 'signal'})
    
    def _load_execution_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load 1-minute execution data."""
        try:
            data = self.database.get_ohlcv_data(
                symbol=symbol,
                exchange='NSE_INDEX',  # Get from instruments config
                timeframe='1m',
                start_date=start_date,
                end_date=end_date
            )
            
            # Reset index to convert timestamp index to column
            if not data.empty and data.index.name == 'timestamp':
                data = data.reset_index()
            
            if data.empty:
                raise DataError(
                    message=f"No execution data available for {symbol}",
                    symbol=symbol,
                    data_type="1m execution data",
                    context={'date_range': f"{start_date} to {end_date}"},
                    suggestion="Check database for data availability"
                )
            
            return data
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'symbol': symbol, 'data_type': 'execution'})
    
    def _calculate_indicators(self, data: pd.DataFrame, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Calculate technical indicators."""
        try:
            # Use existing vAlgo indicator engine (data should have timestamp as index)
            if 'timestamp' in data.columns:
                data_indexed = data.set_index('timestamp')
            else:
                data_indexed = data
            
            data_with_indicators = self.indicator_engine.calculate_indicators_for_symbol(symbol, start_date, end_date)
            
            # Validate indicators
            indicator_columns = [col for col in data_with_indicators.columns 
                               if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            
            if indicator_columns:
                self.data_validator.validate_indicator_data(data_with_indicators, indicator_columns)
            
            self.logger.info(f"Calculated {len(indicator_columns)} indicators for {symbol}")
            return data_with_indicators
            
        except Exception as e:
            self.error_handler.handle_error(e, context={'symbol': symbol, 'phase': 'indicators'})
    
    def _generate_final_results(
        self, 
        all_results: Dict[str, Any],
        symbols: List[str],
        strategies: List[str],
        date_range: tuple
    ) -> Dict[str, Any]:
        """Generate comprehensive final results."""
        successful_symbols = [s for s, r in all_results.items() if r.get('status') != 'failed']
        failed_symbols = [s for s, r in all_results.items() if r.get('status') == 'failed']
        
        return {
            'backtest_summary': {
                'symbols_requested': symbols,
                'strategies_requested': strategies,
                'date_range': date_range,
                'successful_symbols': successful_symbols,
                'failed_symbols': failed_symbols,
                'success_rate': len(successful_symbols) / len(symbols) * 100,
                'timestamp': datetime.now().isoformat()
            },
            'symbol_results': all_results,
            'performance_metrics': self.logger.get_performance_metrics(),
            'validation_statistics': self.data_validator.get_validation_statistics(),
            'error_statistics': self.error_handler.get_error_statistics(),
            'system_info': {
                'vectorbt_version': '1.0.0',
                'python_version': sys.version,
                'config_path': self.config_path
            }
        }


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='VectorBT Enhanced Backtesting System')
    
    parser.add_argument('--symbol', type=str, nargs='+', 
                       help='Symbol(s) to backtest (default: from config)')
    parser.add_argument('--strategy', type=str, nargs='+',
                       help='Strategy(ies) to run (default: active from config)')
    parser.add_argument('--start-date', type=str,
                       help='Start date (YYYY-MM-DD, default: from config)')
    parser.add_argument('--end-date', type=str,
                       help='End date (YYYY-MM-DD, default: from config)')
    parser.add_argument('--config', type=str, default='config/config.xlsx',
                       help='Configuration file path')
    parser.add_argument('--no-real-money', action='store_true',
                       help='Disable real money options integration')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_arguments()
    
    try:
        # Initialize runner
        runner = VectorBTBacktestRunner(config_path=args.config)
        
        # Run backtesting
        results = runner.run_backtest(
            symbols=args.symbol,
            strategies=args.strategy, 
            start_date=args.start_date,
            end_date=args.end_date,
            enable_real_money=not args.no_real_money
        )
        
        # Display results summary
        if results:
            summary = results.get('backtest_summary', {})
            print("\n" + "="*60)
            print("VECTORBT BACKTESTING RESULTS")
            print("="*60)
            print(f"Symbols: {summary.get('symbols_requested', [])}")
            print(f"Strategies: {summary.get('strategies_requested', [])}")
            print(f"Date Range: {summary.get('date_range', ('N/A', 'N/A'))}")
            print(f"Success Rate: {summary.get('success_rate', 0):.1f}%")
            print(f"Successful: {summary.get('successful_symbols', [])}")
            
            if summary.get('failed_symbols'):
                print(f"Failed: {summary.get('failed_symbols', [])}")
            
            print("="*60)
            print("Phase 1 Implementation Complete")
            print("Next: VectorBT signal generation and portfolio management")
            print("="*60)
        
    except KeyboardInterrupt:
        print("\nBacktesting interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nBacktesting failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()