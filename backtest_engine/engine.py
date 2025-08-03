"""
Main Backtesting Engine for the vAlgo Trading System.
Orchestrates the complete backtesting process with integration to existing infrastructure.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from data_manager.database import DatabaseManager
from backtest_engine.models import BacktestResult, PerformanceMetrics
from backtest_engine.strategy_runner import StrategyRunner
from backtest_engine.trade_evaluator import TradeEvaluator
from backtest_engine.utils import BacktestUtils
from indicators.unified_indicator_engine import UnifiedIndicatorEngine


class BacktestEngine:
    """
    Main backtesting engine that coordinates all components.
    Integrates with existing vAlgo infrastructure for data and configuration.
    """
    
    def __init__(self, config_file: str = None, initial_capital: float = 100000.0):
        """
        Initialize the backtest engine
        
        Args:
            config_file: Path to configuration file
            initial_capital: Starting capital for backtesting
        """
        self.logger = get_logger(__name__)
        self.initial_capital = initial_capital
        
        # Initialize configuration using cached config
        try:
            from utils.config_cache import get_cached_config
            config_cache = get_cached_config(config_file)
            self.config_loader = config_cache.get_config_loader()
            if not self.config_loader:
                raise ValueError("Failed to load configuration from cache")
        except Exception as e:
            self.logger.error(f"Error initializing configuration: {e}")
            raise
        
        # Initialize components
        self.utils = BacktestUtils(self.config_loader)
        self.database_manager = None
        
        # Initialize database connection
        try:
            self.database_manager = DatabaseManager()
            self.logger.info("Database connection established")
        except Exception as e:
            self.logger.warning(f"Database connection failed: {e}")
        
        # Load and validate configuration
        self.config = self.utils.get_backtest_configuration()
        self.validation_issues = self.utils.validate_backtest_config(self.config)
        
        if self.validation_issues:
            self.logger.warning(f"Configuration validation issues: {self.validation_issues}")
        
        self.logger.info("BacktestEngine initialized successfully")
        self.utils.log_configuration_summary(self.config)
    
    def run_backtest(self, 
                    symbols: List[str] = None,
                    start_date: str = None,
                    end_date: str = None,
                    strategy_names: List[str] = None,
                    parallel_execution: bool = True,
                    max_workers: int = 4) -> Dict[str, BacktestResult]:
        """
        Run complete backtest with specified parameters
        
        Args:
            symbols: List of symbols to backtest (None for all active)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            strategy_names: List of strategy names (None for all active)
            parallel_execution: Whether to run symbols in parallel
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping symbols to BacktestResult objects
        """
        start_time = time.time()
        
        try:
            self.logger.info("=" * 60)
            self.logger.info("STARTING BACKTEST EXECUTION")
            self.logger.info("=" * 60)
            
            # Validate configuration
            if self.validation_issues:
                self.logger.error(f"Configuration validation failed: {self.validation_issues}")
                raise ValueError("Configuration validation failed")
            
            # Determine symbols to backtest
            target_symbols = self._determine_target_symbols(symbols)
            if not target_symbols:
                raise ValueError("No symbols available for backtesting")
            
            # Determine strategies to test
            target_strategies = self._determine_target_strategies(strategy_names)
            if not target_strategies:
                raise ValueError("No active strategies available for backtesting")
            
            self.logger.info(f"Backtest parameters:")
            self.logger.info(f"  Symbols: {target_symbols}")
            self.logger.info(f"  Strategies: {list(target_strategies.keys())}")
            self.logger.info(f"  Date range: {start_date} to {end_date}")
            self.logger.info(f"  Initial capital: ${self.initial_capital:,.2f}")
            self.logger.info(f"  Parallel execution: {parallel_execution}")
            
            # Load market data for all symbols
            symbol_data_map = self._load_market_data(target_symbols, start_date, end_date)
            
            if not symbol_data_map:
                raise ValueError("No market data loaded for specified symbols and date range")
            
            # Run backtest (parallel or sequential)
            if parallel_execution and len(symbol_data_map) > 1:
                results = self._run_parallel_backtest(symbol_data_map, target_strategies, max_workers)
            else:
                results = self._run_sequential_backtest(symbol_data_map, target_strategies)
            
            # Calculate execution time
            execution_time = time.time() - start_time
            
            # Update results with execution metadata
            for symbol, result in results.items():
                if result:
                    result.execution_time = execution_time
            
            self.logger.info("=" * 60)
            self.logger.info("BACKTEST EXECUTION COMPLETED")
            self.logger.info(f"Total execution time: {execution_time:.2f} seconds")
            self.logger.info(f"Results generated for {len([r for r in results.values() if r])} symbols")
            self.logger.info("=" * 60)
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in backtest execution: {e}")
            raise
    
    def _determine_target_symbols(self, symbols: List[str] = None) -> List[str]:
        """
        Determine which symbols to backtest
        
        Args:
            symbols: Requested symbols (None for all active)
            
        Returns:
            List of target symbols
        """
        if symbols:
            return symbols
        
        # Get active instruments from configuration
        active_instruments = self.config.get('instruments', [])
        return [inst['symbol'] for inst in active_instruments if inst.get('status') == 'active']
    
    def _determine_target_strategies(self, strategy_names: List[str] = None) -> Dict[str, Any]:
        """
        Determine which strategies to test
        
        Args:
            strategy_names: Requested strategy names (None for all active)
            
        Returns:
            Dictionary of target strategies
        """
        all_strategies = self.utils.get_active_strategies_for_backtest(self.config)
        
        if strategy_names:
            return {name: config for name, config in all_strategies.items() 
                   if name in strategy_names}
        
        return all_strategies
    
    def _load_market_data(self, symbols: List[str], start_date: str = None, 
                         end_date: str = None) -> Dict[str, pd.DataFrame]:
        """
        Load market data with indicators for specified symbols and date range
        
        Args:
            symbols: List of symbols to load
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            Dictionary mapping symbols to their market data
        """
        symbol_data_map = {}
        
        try:
            self.logger.info(f"Loading market data for {len(symbols)} symbols...")
            
            for symbol in symbols:
                try:
                    # Load data using existing infrastructure
                    market_data = self._load_symbol_data(symbol, start_date, end_date)
                    
                    if market_data is not None and not market_data.empty:
                        # Validate data
                        from backtest_engine.utils import validate_market_data
                        validation_issues = validate_market_data(market_data)
                        if validation_issues:
                            self.logger.warning(f"Data validation issues for {symbol}: {validation_issues}")
                        
                        symbol_data_map[symbol] = market_data
                        self.logger.info(f"Loaded {len(market_data)} records for {symbol}")
                    else:
                        self.logger.warning(f"No data available for {symbol}")
                        
                except Exception as e:
                    self.logger.error(f"Error loading data for {symbol}: {e}")
            
            return symbol_data_map
            
        except Exception as e:
            self.logger.error(f"Error loading market data: {e}")
            return {}
    
    def _load_symbol_data(self, symbol: str, start_date: str = None, 
                         end_date: str = None) -> Optional[pd.DataFrame]:
        """
        Load data for a single symbol with indicators calculated using UnifiedIndicatorEngine
        
        Args:
            symbol: Symbol to load
            start_date: Start date filter
            end_date: End date filter
            
        Returns:
            DataFrame with OHLCV data and calculated indicators
        """
        try:
            # Initialize UnifiedIndicatorEngine with same config as main system
            indicator_engine = UnifiedIndicatorEngine(self.config_loader.config_file)
            
            # Use UnifiedIndicatorEngine to get data WITH indicators
            df = indicator_engine.calculate_indicators_for_symbol(
                symbol=symbol,
                start_date=start_date or "2024-01-01",  # Default start if None
                end_date=end_date or "2024-12-31"       # Default end if None
            )
            
            if df.empty:
                self.logger.warning(f"No data with indicators found for {symbol}")
                return None
            
            # Reset index to make timestamp a column if needed
            if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                df = df.reset_index()
            elif 'timestamp' not in df.columns and hasattr(df.index, 'name'):
                # Handle case where index is datetime but not named 'timestamp'
                df = df.reset_index()
                df = df.rename(columns={df.columns[0]: 'timestamp'})
            
            # Ensure timestamp is datetime
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Map lowercase indicator names to uppercase for compatibility with strategy rules
            indicator_mapping = {}
            for col in df.columns:
                if col.startswith('ema_'):
                    uppercase_col = col.replace('ema_', 'EMA_')
                    indicator_mapping[col] = uppercase_col
                elif col.startswith('rsi_'):
                    uppercase_col = col.replace('rsi_', 'RSI_')
                    indicator_mapping[col] = uppercase_col
                elif col.startswith('sma_'):
                    uppercase_col = col.replace('sma_', 'SMA_')
                    indicator_mapping[col] = uppercase_col
            
            # Check what indicators are actually needed by strategies
            available_cols = set(df.columns)
            
            # Only generate missing indicators if they're actually required by active strategies
            required_indicators = self._get_required_indicators_for_strategies()
            
            # Generate any missing required indicators
            for required_indicator in required_indicators:
                if required_indicator not in available_cols and required_indicator.lower() not in available_cols:
                    # Check if we can generate this indicator
                    if required_indicator == 'EMA_21' and 'close' in df.columns:
                        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
                        indicator_mapping['ema_21'] = 'EMA_21'
                        self.logger.info(f"Generated required indicator {required_indicator} from close prices")
                    # Add more indicator generation logic here if needed
            
            # Rename columns to match strategy rule expectations
            if indicator_mapping:
                df = df.rename(columns=indicator_mapping)
                self.logger.info(f"Mapped {len(indicator_mapping)} indicator column names for strategy compatibility")
            
            # Log successful indicator loading
            indicator_columns = [col for col in df.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
            self.logger.info(f"Loaded {len(df)} records for {symbol} with {len(indicator_columns)} indicators")
            self.logger.debug(f"Available indicators: {indicator_columns}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading data with indicators for {symbol}: {e}")
            # Fallback to basic OHLCV data if indicator engine fails
            try:
                if self.database_manager:
                    self.logger.info(f"Falling back to basic OHLCV data for {symbol}")
                    df = self.database_manager.get_ohlcv_data(
                        symbol=symbol,
                        exchange="NSE_INDEX",
                        timeframe="1m",
                        start_date=start_date,
                        end_date=end_date
                    )
                    if not df.empty:
                        # Reset index to make timestamp a column
                        if 'timestamp' not in df.columns and df.index.name == 'timestamp':
                            df = df.reset_index()
                        if 'timestamp' in df.columns:
                            df['timestamp'] = pd.to_datetime(df['timestamp'])
                        return df
                return None
            except Exception as fallback_error:
                self.logger.error(f"Fallback data loading also failed for {symbol}: {fallback_error}")
                return None
    
    def _run_sequential_backtest(self, symbol_data_map: Dict[str, pd.DataFrame],
                                target_strategies: Dict[str, Any]) -> Dict[str, BacktestResult]:
        """
        Run backtest sequentially for each symbol
        
        Args:
            symbol_data_map: Map of symbols to their data
            target_strategies: Strategies to test
            
        Returns:
            Dictionary of backtest results
        """
        results = {}
        
        try:
            for symbol, market_data in symbol_data_map.items():
                self.logger.info(f"Running sequential backtest for {symbol}")
                
                # Run for each strategy or combined
                if len(target_strategies) == 1:
                    strategy_name = list(target_strategies.keys())[0]
                    runner = StrategyRunner(self.config_loader, self.initial_capital)
                    result = runner.run_strategy_backtest(symbol, market_data, strategy_name)
                    results[symbol] = result
                else:
                    # Run all strategies combined
                    runner = StrategyRunner(self.config_loader, self.initial_capital)
                    result = runner.run_strategy_backtest(symbol, market_data)
                    results[symbol] = result
                    
        except Exception as e:
            self.logger.error(f"Error in sequential backtest: {e}")
            
        return results
    
    def _run_parallel_backtest(self, symbol_data_map: Dict[str, pd.DataFrame],
                              target_strategies: Dict[str, Any], 
                              max_workers: int) -> Dict[str, BacktestResult]:
        """
        Run backtest in parallel across symbols
        
        Args:
            symbol_data_map: Map of symbols to their data
            target_strategies: Strategies to test
            max_workers: Maximum parallel workers
            
        Returns:
            Dictionary of backtest results
        """
        results = {}
        
        try:
            self.logger.info(f"Running parallel backtest with {max_workers} workers")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_symbol = {}
                
                for symbol, market_data in symbol_data_map.items():
                    # Create runner for this symbol
                    runner = StrategyRunner(self.config_loader, self.initial_capital)
                    
                    # Determine strategy to run
                    strategy_name = None
                    if len(target_strategies) == 1:
                        strategy_name = list(target_strategies.keys())[0]
                    
                    # Submit task
                    future = executor.submit(runner.run_strategy_backtest, symbol, market_data, strategy_name)
                    future_to_symbol[future] = symbol
                
                # Collect results
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                        self.logger.info(f"Completed parallel backtest for {symbol}")
                    except Exception as e:
                        self.logger.error(f"Error in parallel backtest for {symbol}: {e}")
                        results[symbol] = None
                        
        except Exception as e:
            self.logger.error(f"Error in parallel backtest execution: {e}")
            
        return results
    
    def get_aggregated_results(self, results: Dict[str, BacktestResult]) -> Dict[str, Any]:
        """
        Get aggregated results across all symbols
        
        Args:
            results: Dictionary of symbol results
            
        Returns:
            Aggregated performance metrics
        """
        try:
            # Filter successful results
            valid_results = {k: v for k, v in results.items() if v is not None}
            
            if not valid_results:
                return {}
            
            # Aggregate metrics
            total_trades = sum(len(result.all_trades) for result in valid_results.values())
            total_pnl = sum(result.overall_performance.total_pnl 
                           for result in valid_results.values() 
                           if result.overall_performance)
            
            winning_trades = sum(result.overall_performance.winning_trades
                               for result in valid_results.values()
                               if result.overall_performance)
            
            aggregated = {
                'symbols_tested': len(valid_results),
                'total_trades': total_trades,
                'total_pnl': total_pnl,
                'winning_trades': winning_trades,
                'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
                'initial_capital': self.initial_capital,
                'final_capital': self.initial_capital + total_pnl,
                'total_return_percentage': (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0,
                'symbol_results': {symbol: result.overall_performance.to_dict() 
                                 for symbol, result in valid_results.items() 
                                 if result.overall_performance}
            }
            
            return aggregated
            
        except Exception as e:
            self.logger.error(f"Error aggregating results: {e}")
            return {}
    
    def validate_backtest_setup(self) -> Tuple[bool, List[str]]:
        """
        Validate backtest setup and configuration
        
        Returns:
            Tuple of (is_valid, list_of_issues)
        """
        issues = []
        
        # Configuration validation
        issues.extend(self.validation_issues)
        
        # Database validation
        if not self.database_manager:
            issues.append("Database connection not available")
        
        # Strategy validation
        active_strategies = self.utils.get_active_strategies_for_backtest(self.config)
        if not active_strategies:
            issues.append("No active strategies configured")
        
        # Instrument validation
        active_instruments = self.config.get('instruments', [])
        if not active_instruments:
            issues.append("No active instruments configured")
        
        is_valid = len(issues) == 0
        return is_valid, issues
    
    def _get_required_indicators_for_strategies(self) -> set:
        """
        Analyze active strategies to determine which indicators are actually required
        
        Returns:
            Set of required indicator names
        """
        required_indicators = set()
        
        try:
            # Get active strategies
            active_strategies = self.utils.get_active_strategies_for_backtest(self.config)
            
            # Get entry and exit rules
            entry_rules = self.config.get('entry_rules', {})
            exit_rules = self.config.get('exit_rules', {})
            
            for strategy_name, strategy_config in active_strategies.items():
                # Check entry rule requirements
                entry_rule_name = strategy_config.get('entry_rule', '')
                if entry_rule_name in entry_rules:
                    entry_rule = entry_rules[entry_rule_name]
                    required_indicators.update(self._extract_indicators_from_rule(entry_rule))
                
                # Check exit rule requirements
                exit_rule_names = strategy_config.get('exit_rules', [])
                for exit_rule_name in exit_rule_names:
                    if exit_rule_name in exit_rules:
                        exit_rule = exit_rules[exit_rule_name]
                        required_indicators.update(self._extract_indicators_from_rule(exit_rule))
            
        except Exception as e:
            self.logger.warning(f"Error analyzing required indicators: {e}")
        
        return required_indicators
    
    def _extract_indicators_from_rule(self, rule_data: Dict[str, Any]) -> set:
        """
        Extract indicator names from a rule definition
        
        Args:
            rule_data: Rule data from config
            
        Returns:
            Set of indicator names used in this rule
        """
        indicators = set()
        
        try:
            # Handle entry rules with condition_groups
            if 'condition_groups' in rule_data:
                for group in rule_data['condition_groups']:
                    for condition in group.get('conditions', []):
                        indicator = condition.get('indicator', '')
                        if indicator:
                            indicators.add(indicator)
            
            # Handle exit rules with conditions
            elif 'conditions' in rule_data:
                for condition in rule_data['conditions']:
                    indicator = condition.get('indicator', '')
                    if indicator:
                        indicators.add(indicator)
        
        except Exception as e:
            self.logger.warning(f"Error extracting indicators from rule: {e}")
        
        return indicators

    def get_engine_info(self) -> Dict[str, Any]:
        """
        Get information about the engine configuration
        
        Returns:
            Dictionary with engine information
        """
        return {
            'initial_capital': self.initial_capital,
            'active_strategies': len(self.utils.get_active_strategies_for_backtest(self.config)),
            'active_instruments': len(self.config.get('instruments', [])),
            'validation_issues': len(self.validation_issues),
            'database_available': self.database_manager is not None,
            'config_file': getattr(self.config_loader, 'config_file', 'Unknown')
        }