"""
Strategy Runner for the vAlgo Backtesting Engine.
Orchestrates trade execution and position management for backtesting strategies.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import pandas as pd
import logging
from collections import defaultdict
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from backtest_engine.models import Trade, Position, TradeType, TradeStatus, BacktestResult, ExitReason
from backtest_engine.trade_evaluator import TradeEvaluator
from backtest_engine.utils import BacktestUtils


class StrategyRunner:
    """
    Strategy runner that executes backtesting strategies using existing ConfigLoader system.
    Handles position management, trade lifecycle, and parallel execution.
    """
    
    def __init__(self, config_loader: ConfigLoader, initial_capital: float = 100000.0):
        """
        Initialize strategy runner
        
        Args:
            config_loader: Configured ConfigLoader instance
            initial_capital: Starting capital for backtesting
        """
        self.config_loader = config_loader
        self.logger = get_logger(__name__)
        self.initial_capital = initial_capital
        
        # Initialize components
        self.trade_evaluator = TradeEvaluator(config_loader)
        self.utils = BacktestUtils(config_loader)
        
        # Load configuration
        self.config = self.utils.get_backtest_configuration()
        self.active_strategies = self.utils.get_active_strategies_for_backtest(self.config)
        
        # Track positions and trades
        self.positions: Dict[str, Position] = {}
        self.open_trades: Dict[str, List[Trade]] = defaultdict(list)
        self.closed_trades: List[Trade] = []
        
        # Performance tracking
        self.current_capital = initial_capital
        self.trade_counter = 0
        
        # Thread safety
        self.lock = threading.Lock()
        
        self.logger.info(f"StrategyRunner initialized with {len(self.active_strategies)} strategies")
        self.logger.info(f"Initial capital: ${initial_capital:,.2f}")
    
    def run_strategy_backtest(self, symbol: str, market_data: pd.DataFrame, 
                             strategy_name: str = None) -> BacktestResult:
        """
        Run backtest for a specific symbol and strategy
        
        Args:
            symbol: Trading symbol
            market_data: Historical market data with indicators
            strategy_name: Specific strategy to test (None for all)
            
        Returns:
            BacktestResult object with complete results
        """
        try:
            self.logger.info(f"Starting backtest for {symbol} with {len(market_data)} data points")
            
            # Validate market data
            from backtest_engine.utils import validate_market_data
            validation_issues = validate_market_data(market_data)
            if validation_issues:
                self.logger.warning(f"Market data validation issues: {validation_issues}")
            
            # Initialize result container
            start_date = market_data['timestamp'].min()
            end_date = market_data['timestamp'].max()
            
            result = BacktestResult(
                strategy_name=strategy_name or "ALL_STRATEGIES",
                symbols=[symbol],
                start_date=start_date,
                end_date=end_date,
                initial_capital=self.initial_capital
            )
            
            # Filter strategies if specific strategy requested
            strategies_to_run = self.active_strategies
            if strategy_name and strategy_name in self.active_strategies:
                strategies_to_run = {strategy_name: self.active_strategies[strategy_name]}
            
            # Process each row of market data
            for idx, row in market_data.iterrows():
                current_timestamp = self.utils.format_timestamp(row['timestamp'])
                
                # Process each strategy
                for strat_name, strat_config in strategies_to_run.items():
                    self._process_strategy_signals(symbol, strat_name, row, current_timestamp, result)
            
            # Close any remaining open trades at the end
            self._close_remaining_trades(symbol, market_data.iloc[-1], result)
            
            # Calculate final performance
            result.calculate_performance()
            result.data_points_processed = len(market_data)
            
            self.logger.info(f"Backtest completed for {symbol}: "
                           f"{len(result.all_trades)} trades generated")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error running backtest for {symbol}: {e}")
            raise
    
    def _process_strategy_signals(self, symbol: str, strategy_name: str, 
                                 market_row: pd.Series, timestamp: datetime,
                                 result: BacktestResult) -> None:
        """
        Process entry and exit signals for a strategy at current timestamp
        
        Args:
            symbol: Trading symbol
            strategy_name: Strategy name
            market_row: Current market data row
            timestamp: Current timestamp
            result: BacktestResult to update
        """
        try:
            position_key = f"{symbol}_{strategy_name}"
            
            # Get current open trades for this strategy-symbol combination
            current_open_trades = [
                trade for trade in self.open_trades[position_key]
                if trade.status == TradeStatus.OPEN
            ]
            
            # Evaluate exit signals first (for open trades)
            if current_open_trades:
                exit_signals = self.trade_evaluator.evaluate_exit_signals(
                    current_open_trades, market_row, timestamp
                )
                
                for exit_signal in exit_signals:
                    self._execute_exit_signal(exit_signal, result)
            
            # Evaluate entry signals
            entry_signals = self.trade_evaluator.evaluate_entry_signals(
                symbol, market_row, timestamp
            )
            
            # Filter entry signals for current strategy
            strategy_entry_signals = [
                signal for signal in entry_signals
                if signal['strategy_name'] == strategy_name
            ]
            
            # Execute entry signals (respecting position limits)
            for entry_signal in strategy_entry_signals:
                if self._can_open_new_position(strategy_name, symbol):
                    self._execute_entry_signal(entry_signal, result)
                    
        except Exception as e:
            self.logger.error(f"Error processing signals for {strategy_name}/{symbol}: {e}")
    
    def _execute_entry_signal(self, signal: Dict[str, Any], result: BacktestResult) -> None:
        """
        Execute entry signal and create new trade
        
        Args:
            signal: Entry signal dictionary
            result: BacktestResult to update
        """
        try:
            with self.lock:
                # Create trade from signal
                trade = self.trade_evaluator.create_trade_from_signal(signal)
                
                # Update position tracking
                position_key = f"{trade.symbol}_{trade.strategy_name}"
                if position_key not in self.positions:
                    self.positions[position_key] = Position(
                        symbol=trade.symbol,
                        strategy_name=trade.strategy_name
                    )
                
                # Add trade to tracking
                self.open_trades[position_key].append(trade)
                self.positions[position_key].add_trade(trade)
                result.add_trade(trade)
                
                self.trade_counter += 1
                
                self.logger.debug(f"Executed entry: {trade.trade_id}")
                
        except Exception as e:
            self.logger.error(f"Error executing entry signal: {e}")
    
    def _execute_exit_signal(self, signal: Dict[str, Any], result: BacktestResult) -> None:
        """
        Execute exit signal and close trade
        
        Args:
            signal: Exit signal dictionary
            result: BacktestResult to update
        """
        try:
            with self.lock:
                # Close trade
                trade = self.trade_evaluator.close_trade_from_signal(signal)
                
                # Update position tracking
                position_key = f"{trade.symbol}_{trade.strategy_name}"
                if position_key in self.positions:
                    self.positions[position_key].close_trade(trade)
                
                # Move from open to closed
                if trade in self.open_trades[position_key]:
                    self.open_trades[position_key].remove(trade)
                    self.closed_trades.append(trade)
                
                self.logger.debug(f"Executed exit: {trade.trade_id} P&L: {trade.net_pnl:.2f}")
                
        except Exception as e:
            self.logger.error(f"Error executing exit signal: {e}")
    
    def _can_open_new_position(self, strategy_name: str, symbol: str) -> bool:
        """
        Check if new position can be opened based on strategy limits
        
        Args:
            strategy_name: Strategy name
            symbol: Trading symbol
            
        Returns:
            True if new position can be opened
        """
        try:
            strategy_config = self.active_strategies.get(strategy_name, {})
            max_positions = strategy_config.get('max_positions', 1)
            
            position_key = f"{symbol}_{strategy_name}"
            current_open_count = len(self.open_trades[position_key])
            
            return current_open_count < max_positions
            
        except Exception as e:
            self.logger.error(f"Error checking position limits: {e}")
            return False
    
    def _close_remaining_trades(self, symbol: str, final_market_data: pd.Series, 
                               result: BacktestResult) -> None:
        """
        Close any remaining open trades at the end of backtest
        
        Args:
            symbol: Trading symbol
            final_market_data: Final market data row
            result: BacktestResult to update
        """
        try:
            final_timestamp = self.utils.format_timestamp(final_market_data['timestamp'])
            final_price = final_market_data.get('close', 0)
            
            for position_key, open_trades_list in self.open_trades.items():
                if symbol not in position_key:
                    continue
                    
                for trade in open_trades_list.copy():
                    if trade.status == TradeStatus.OPEN:
                        # Force close with market price
                        trade.close_trade(
                            exit_timestamp=final_timestamp,
                            exit_price=final_price,
                            exit_rule="MARKET_CLOSE",
                            exit_reason=ExitReason.TIME_EXIT,
                            exit_conditions=["End of backtest period"]
                        )
                        
                        # Update tracking
                        position_key = f"{trade.symbol}_{trade.strategy_name}"
                        if position_key in self.positions:
                            self.positions[position_key].close_trade(trade)
                        
                        self.closed_trades.append(trade)
                        self.logger.debug(f"Force closed trade {trade.trade_id} at market close")
            
            # Clear open trades
            for key in list(self.open_trades.keys()):
                if symbol in key:
                    self.open_trades[key].clear()
                    
        except Exception as e:
            self.logger.error(f"Error closing remaining trades: {e}")
    
    def run_parallel_backtest(self, symbol_data_map: Dict[str, pd.DataFrame],
                             strategy_name: str = None, max_workers: int = 4) -> Dict[str, BacktestResult]:
        """
        Run backtest in parallel across multiple symbols
        
        Args:
            symbol_data_map: Dictionary mapping symbols to their market data
            strategy_name: Specific strategy to test (None for all)
            max_workers: Maximum number of parallel workers
            
        Returns:
            Dictionary mapping symbols to their BacktestResult
        """
        results = {}
        
        try:
            self.logger.info(f"Starting parallel backtest for {len(symbol_data_map)} symbols")
            
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                # Submit backtest tasks
                future_to_symbol = {}
                for symbol, market_data in symbol_data_map.items():
                    # Create separate runner instance for thread safety
                    runner = StrategyRunner(self.config_loader, self.initial_capital)
                    future = executor.submit(runner.run_strategy_backtest, symbol, market_data, strategy_name)
                    future_to_symbol[future] = symbol
                
                # Collect results
                for future in as_completed(future_to_symbol):
                    symbol = future_to_symbol[future]
                    try:
                        result = future.result()
                        results[symbol] = result
                        self.logger.info(f"Completed backtest for {symbol}: {len(result.all_trades)} trades")
                    except Exception as e:
                        self.logger.error(f"Error in backtest for {symbol}: {e}")
                        results[symbol] = None
            
            self.logger.info(f"Parallel backtest completed for {len(results)} symbols")
            return results
            
        except Exception as e:
            self.logger.error(f"Error in parallel backtest: {e}")
            return results
    
    def get_current_positions(self) -> Dict[str, Position]:
        """
        Get current positions
        
        Returns:
            Dictionary of current positions
        """
        return self.positions.copy()
    
    def get_trade_statistics(self) -> Dict[str, Any]:
        """
        Get current trade statistics
        
        Returns:
            Dictionary with trade statistics
        """
        total_trades = len(self.closed_trades)
        winning_trades = len([t for t in self.closed_trades if t.net_pnl and t.net_pnl > 0])
        
        return {
            'total_trades': total_trades,
            'open_trades': sum(len(trades) for trades in self.open_trades.values()),
            'closed_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': total_trades - winning_trades,
            'win_rate': (winning_trades / total_trades * 100) if total_trades > 0 else 0,
            'current_capital': self.current_capital
        }
    
    def reset_runner(self) -> None:
        """Reset runner state for new backtest"""
        with self.lock:
            self.positions.clear()
            self.open_trades.clear()
            self.closed_trades.clear()
            self.current_capital = self.initial_capital
            self.trade_counter = 0
            self.logger.info("StrategyRunner state reset")
    
    def validate_strategy_configuration(self) -> List[str]:
        """
        Validate strategy configuration for backtesting
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Use existing validation
        config_issues = self.utils.validate_backtest_config(self.config)
        issues.extend(config_issues)
        
        # Check required indicators
        required_indicators = self.trade_evaluator.get_required_indicators()
        if not required_indicators:
            issues.append("No indicators required by strategies - check rule configuration")
        
        return issues