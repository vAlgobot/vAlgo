"""
Trade Evaluator for the vAlgo Backtesting Engine.
Handles rule matching and condition evaluation using existing ConfigLoader methods.
"""

import sys
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Union
from datetime import datetime
import pandas as pd
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import ConfigLoader
from utils.logger import get_logger
from backtest_engine.models import Trade, TradeType, TradeStatus, ExitReason
from backtest_engine.utils import BacktestUtils

# Import SL/TP engine
try:
    from utils.advanced_sl_tp_engine import AdvancedSLTPEngine, create_sl_tp_engine
    SLTP_ENGINE_AVAILABLE = True
except ImportError:
    SLTP_ENGINE_AVAILABLE = False
    AdvancedSLTPEngine = None
    create_sl_tp_engine = None


class TradeEvaluator:
    """
    Trade evaluator that uses existing ConfigLoader for rule evaluation.
    Handles entry/exit signal generation and trade management.
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize trade evaluator with ConfigLoader
        
        Args:
            config_loader: Configured ConfigLoader instance
        """
        self.config_loader = config_loader
        self.logger = get_logger(__name__)
        self.utils = BacktestUtils(config_loader)
        
        # Initialize SL/TP engine if available
        if SLTP_ENGINE_AVAILABLE:
            try:
                self.sl_tp_engine = create_sl_tp_engine()
                self.logger.info("SL/TP engine initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize SL/TP engine: {e}")
                self.sl_tp_engine = None
        else:
            self.sl_tp_engine = None
            self.logger.warning("SL/TP engine not available")
        
        # Load configuration using existing methods
        self.entry_rules = config_loader.get_entry_conditions_enhanced()
        self.exit_rules = config_loader.get_exit_conditions_enhanced()
        self.strategies = config_loader.get_strategy_configs_enhanced()
        
        # Active strategies only
        self.active_strategies = {
            name: config for name, config in self.strategies.items()
            if config.get('status', '').lower() == 'active'
        }
        
        self.logger.info(f"TradeEvaluator initialized with {len(self.active_strategies)} active strategies")
        self.logger.info(f"Available entry rules: {list(self.entry_rules.keys())}")
        self.logger.info(f"Available exit rules: {list(self.exit_rules.keys())}")
    
    def evaluate_entry_signals(self, symbol: str, current_data: pd.Series, 
                              timestamp: datetime) -> List[Dict[str, Any]]:
        """
        Evaluate entry signals for all active strategies
        
        Args:
            symbol: Trading symbol
            current_data: Current market data row with indicators
            timestamp: Current timestamp
            
        Returns:
            List of entry signals with strategy and rule information
        """
        entry_signals = []
        
        try:
            for strategy_name, strategy_config in self.active_strategies.items():
                entry_rule_name = strategy_config.get('entry_rule', '')
                
                if not entry_rule_name or entry_rule_name not in self.entry_rules:
                    continue
                
                entry_rule_data = self.entry_rules[entry_rule_name]
                
                # Evaluate entry rule using existing ConfigLoader format
                rule_passed, matched_conditions = self.utils.evaluate_entry_rule(
                    entry_rule_name, entry_rule_data, current_data
                )
                
                if rule_passed:
                    # Determine trade type (default to BUY for now)
                    trade_type = TradeType.BUY  # Future: make configurable
                    
                    signal = {
                        'symbol': symbol,
                        'strategy_name': strategy_name,
                        'entry_rule': entry_rule_name,
                        'trade_type': trade_type,
                        'timestamp': timestamp,
                        'price': current_data.get('close', 0),
                        'matched_conditions': matched_conditions,
                        'rule_description': self.utils.build_rule_description(
                            entry_rule_name, entry_rule_data, "entry"
                        ),
                        'strategy_config': strategy_config
                    }
                    
                    entry_signals.append(signal)
                    
                    self.logger.debug(f"Entry signal generated: {strategy_name} for {symbol} "
                                    f"at {timestamp} - Rule: {entry_rule_name}")
                    
        except Exception as e:
            self.logger.error(f"Error evaluating entry signals for {symbol}: {e}")
        
        return entry_signals
    
    def calculate_sl_tp_for_signal(self, signal: Dict[str, Any], market_data: pd.DataFrame,
                                  pivot_levels: List[float] = None) -> Optional[Any]:
        """
        Calculate SL/TP levels for an entry signal
        
        Args:
            signal: Entry signal dictionary
            market_data: Recent market data for calculations
            pivot_levels: List of pivot levels (CPR data)
            
        Returns:
            SLTPResult object or None if calculation fails
        """
        if not self.sl_tp_engine:
            return None
        
        try:
            # Use CPR pivot levels if available, otherwise empty list
            if pivot_levels is None:
                pivot_levels = []
            
            # Calculate SL/TP levels
            sl_tp_result = self.sl_tp_engine.calculate_sl_tp(
                symbol=signal['symbol'],
                strategy_name=signal['strategy_name'],
                entry_price=signal['price'],
                pivot_levels=pivot_levels,
                market_data=market_data,
                position_size=signal.get('strategy_config', {}).get('position_size', 1000)
            )
            
            self.logger.info(f"SL/TP calculated for {signal['symbol']}: "
                           f"SL={sl_tp_result.stop_loss.price:.2f}, "
                           f"TPs={[tp.price for tp in sl_tp_result.take_profits]}")
            
            return sl_tp_result
            
        except Exception as e:
            self.logger.error(f"Error calculating SL/TP for {signal['symbol']}: {e}")
            return None
    
    def evaluate_exit_signals(self, open_trades: List[Trade], current_data: pd.Series, 
                             timestamp: datetime) -> List[Dict[str, Any]]:
        """
        Enhanced exit signal evaluation including SL/TP monitoring
        
        Args:
            open_trades: List of open trades to evaluate
            current_data: Current market data row with indicators
            timestamp: Current timestamp
            
        Returns:
            List of exit signals with trade and rule information
        """
        exit_signals = []
        current_price = current_data.get('close', 0)
        
        try:
            for trade in open_trades:
                # Check SL/TP hits first (higher priority than rule-based exits)
                sl_tp_hits = trade.check_sl_tp_hit(current_price, timestamp)
                
                if sl_tp_hits:
                    # Process SL/TP hits
                    for hit in sl_tp_hits:
                        signal = {
                            'trade': trade,
                            'exit_rule': f"SL/TP_{hit['type']}",
                            'timestamp': timestamp,
                            'price': hit['price'],
                            'matched_conditions': [f"{hit['type']} level hit"],
                            'exit_reason': hit['reason'],
                            'rule_description': f"Stop Loss/Take Profit {hit['type']} triggered",
                            'is_sl_tp': True,
                            'sl_tp_info': hit
                        }
                        
                        exit_signals.append(signal)
                        
                        self.logger.info(f"SL/TP triggered: {trade.strategy_name} for {trade.symbol} "
                                       f"at {timestamp} - {hit['type']} at {hit['price']}")
                    
                    # Skip rule-based exits if SL/TP triggered
                    continue
                
                # Update unrealized PnL for open positions
                trade.update_total_pnl(current_price)
                
                # Evaluate rule-based exits if no SL/TP triggered
                strategy_config = self.active_strategies.get(trade.strategy_name)
                if not strategy_config:
                    continue
                
                exit_rules_list = strategy_config.get('exit_rules', [])
                
                for exit_rule_name in exit_rules_list:
                    if exit_rule_name not in self.exit_rules:
                        continue
                    
                    exit_rule_data = self.exit_rules[exit_rule_name]
                    
                    # Evaluate exit rule using existing ConfigLoader format
                    rule_passed, matched_conditions = self.utils.evaluate_exit_rule(
                        exit_rule_name, exit_rule_data, current_data
                    )
                    
                    if rule_passed:
                        signal = {
                            'trade': trade,
                            'exit_rule': exit_rule_name,
                            'timestamp': timestamp,
                            'price': current_price,
                            'matched_conditions': matched_conditions,
                            'exit_reason': ExitReason.SIGNAL,
                            'rule_description': self.utils.build_rule_description(
                                exit_rule_name, exit_rule_data, "exit"
                            ),
                            'is_sl_tp': False
                        }
                        
                        exit_signals.append(signal)
                        
                        self.logger.debug(f"Exit signal generated: {trade.strategy_name} for {trade.symbol} "
                                        f"at {timestamp} - Rule: {exit_rule_name}")
                        
                        # Only one exit signal per trade per timestamp
                        break
                        
        except Exception as e:
            self.logger.error(f"Error evaluating exit signals: {e}")
        
        return exit_signals
    
    def process_sl_tp_exit(self, trade: Trade, exit_signal: Dict[str, Any], 
                          commission_rate: float = 0.01) -> bool:
        """
        Process a SL/TP exit for a trade (supports partial exits)
        
        Args:
            trade: Trade to exit
            exit_signal: Exit signal with SL/TP information
            commission_rate: Commission rate for the exit
            
        Returns:
            True if trade was completely closed, False if partially closed
        """
        try:
            if not exit_signal.get('is_sl_tp', False):
                return False
            
            sl_tp_info = exit_signal['sl_tp_info']
            
            # Calculate commission for this exit
            commission = sl_tp_info['quantity'] * sl_tp_info['price'] * commission_rate / 100
            
            # Execute the partial exit
            partial_exit = trade.execute_partial_exit(sl_tp_info, commission)
            
            self.logger.info(f"Partial exit executed: {trade.symbol} {sl_tp_info['type']} "
                           f"- Qty: {partial_exit.quantity_closed}, "
                           f"PnL: {partial_exit.net_pnl:.2f}, "
                           f"Remaining: {trade.remaining_quantity}")
            
            # Return True if trade is completely closed
            return trade.status == TradeStatus.CLOSED
            
        except Exception as e:
            self.logger.error(f"Error processing SL/TP exit for {trade.symbol}: {e}")
            return False
    
    def create_trade_from_signal(self, signal: Dict[str, Any]) -> Trade:
        """
        Create Trade object from entry signal
        
        Args:
            signal: Entry signal dictionary
            
        Returns:
            Trade object
        """
        try:
            strategy_config = signal.get('strategy_config', {})
            
            # Calculate position size
            position_size = self.utils.calculate_position_size(
                strategy_config, 
                signal['price'], 
                100000  # Default capital - will be managed by strategy runner
            )
            
            # Generate trade ID
            trade_id = self.utils.generate_trade_id(
                signal['symbol'],
                signal['strategy_name'],
                signal['timestamp'],
                signal['trade_type'].value
            )
            
            trade = Trade(
                trade_id=trade_id,
                symbol=signal['symbol'],
                strategy_name=signal['strategy_name'],
                trade_type=signal['trade_type'],
                entry_timestamp=signal['timestamp'],
                entry_price=signal['price'],
                quantity=position_size,
                entry_rule=signal['entry_rule'],
                entry_conditions=signal['matched_conditions'],
                status=TradeStatus.OPEN
            )
            
            self.logger.info(f"Created trade {trade_id}: {signal['trade_type'].value} "
                           f"{position_size} {signal['symbol']} at {signal['price']}")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error creating trade from signal: {e}")
            raise
    
    def close_trade_from_signal(self, signal: Dict[str, Any]) -> Trade:
        """
        Close trade based on exit signal
        
        Args:
            signal: Exit signal dictionary
            
        Returns:
            Updated Trade object
        """
        try:
            trade = signal['trade']
            
            trade.close_trade(
                exit_timestamp=signal['timestamp'],
                exit_price=signal['price'],
                exit_rule=signal['exit_rule'],
                exit_reason=signal['exit_reason'],
                exit_conditions=signal['matched_conditions']
            )
            
            self.logger.info(f"Closed trade {trade.trade_id}: "
                           f"P&L = {trade.net_pnl:.2f}, "
                           f"Duration = {trade.trade_duration:.1f} min")
            
            return trade
            
        except Exception as e:
            self.logger.error(f"Error closing trade from signal: {e}")
            raise
    
    def get_strategy_configuration(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Strategy configuration dictionary
        """
        return self.active_strategies.get(strategy_name, {})
    
    def get_entry_rule_info(self, rule_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an entry rule
        
        Args:
            rule_name: Name of the entry rule
            
        Returns:
            Entry rule information dictionary
        """
        if rule_name not in self.entry_rules:
            return {}
        
        rule_data = self.entry_rules[rule_name]
        
        return {
            'rule_name': rule_name,
            'status': rule_data.get('status', ''),
            'condition_groups': rule_data.get('condition_groups', []),
            'total_conditions': rule_data.get('total_conditions', 0),
            'description': self.utils.build_rule_description(rule_name, rule_data, "entry")
        }
    
    def get_exit_rule_info(self, rule_name: str) -> Dict[str, Any]:
        """
        Get detailed information about an exit rule
        
        Args:
            rule_name: Name of the exit rule
            
        Returns:
            Exit rule information dictionary
        """
        if rule_name not in self.exit_rules:
            return {}
        
        rule_data = self.exit_rules[rule_name]
        
        return {
            'rule_name': rule_name,
            'status': rule_data.get('status', ''),
            'exit_type': rule_data.get('exit_type', ''),
            'conditions': rule_data.get('conditions', []),
            'description': self.utils.build_rule_description(rule_name, rule_data, "exit")
        }
    
    def validate_signal_data(self, current_data: pd.Series, required_indicators: List[str]) -> bool:
        """
        Validate that current data contains required indicators
        
        Args:
            current_data: Current market data row
            required_indicators: List of required indicator names
            
        Returns:
            True if all required indicators are present
        """
        missing_indicators = []
        
        for indicator in required_indicators:
            if indicator not in current_data.index or pd.isna(current_data[indicator]):
                missing_indicators.append(indicator)
        
        if missing_indicators:
            self.logger.warning(f"Missing indicators: {missing_indicators}")
            return False
        
        return True
    
    def get_required_indicators(self) -> List[str]:
        """
        Get list of all indicators required by active strategies
        
        Returns:
            List of required indicator names
        """
        required_indicators = set()
        
        # From entry rules
        for rule_data in self.entry_rules.values():
            for group in rule_data.get('condition_groups', []):
                for condition in group.get('conditions', []):
                    indicator = condition.get('indicator', '')
                    if indicator:
                        required_indicators.add(indicator)
        
        # From exit rules
        for rule_data in self.exit_rules.values():
            for condition in rule_data.get('conditions', []):
                indicator = condition.get('indicator', '')
                if indicator:
                    required_indicators.add(indicator)
        
        return list(required_indicators)
    
    def log_evaluation_summary(self, symbol: str, timestamp: datetime, 
                              entry_signals: List[Dict], exit_signals: List[Dict]) -> None:
        """
        Log summary of signal evaluation
        
        Args:
            symbol: Trading symbol
            timestamp: Current timestamp
            entry_signals: Generated entry signals
            exit_signals: Generated exit signals
        """
        if entry_signals or exit_signals:
            self.logger.debug(f"Signal evaluation for {symbol} at {timestamp}: "
                            f"{len(entry_signals)} entry, {len(exit_signals)} exit")
            
            for signal in entry_signals:
                self.logger.debug(f"  Entry: {signal['strategy_name']} - {signal['entry_rule']}")
            
            for signal in exit_signals:
                self.logger.debug(f"  Exit: {signal['trade'].strategy_name} - {signal['exit_rule']}")
    
    def get_evaluation_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the evaluation process
        
        Returns:
            Dictionary with evaluation statistics
        """
        return {
            'active_strategies': len(self.active_strategies),
            'entry_rules': len(self.entry_rules),
            'exit_rules': len(self.exit_rules),
            'required_indicators': len(self.get_required_indicators()),
            'strategy_names': list(self.active_strategies.keys()),
            'entry_rule_names': list(self.entry_rules.keys()),
            'exit_rule_names': list(self.exit_rules.keys()),
            'required_indicator_list': self.get_required_indicators()
        }