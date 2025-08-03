"""
Utility functions for the vAlgo Backtesting Engine.
Leverages existing ConfigLoader methods for maximum code reuse.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, Tuple
from datetime import datetime, timedelta
import pandas as pd
import logging
import operator

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.config_loader import ConfigLoader
from utils.logger import get_logger


class BacktestUtils:
    """Utility class for backtesting operations leveraging existing ConfigLoader"""
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize with ConfigLoader instance
        
        Args:
            config_loader: Existing ConfigLoader instance
        """
        self.config_loader = config_loader
        self.logger = get_logger(__name__)
        
        # Operator mapping for condition evaluation
        self.operators = {
            '>': operator.gt,
            '<': operator.lt,
            '>=': operator.ge,
            '<=': operator.le,
            '=': operator.eq,
            '==': operator.eq,
            '!=': operator.ne,
            '<>': operator.ne
        }
    
    def get_backtest_configuration(self) -> Dict[str, Any]:
        """
        Get complete backtest configuration using existing ConfigLoader methods
        
        Returns:
            Dictionary with all backtest configuration
        """
        try:
            config = {
                'entry_rules': self.config_loader.get_entry_conditions_enhanced(),
                'exit_rules': self.config_loader.get_exit_conditions_enhanced(),
                'strategies': self.config_loader.get_strategy_configs_enhanced(),
                'instruments': self.config_loader.get_active_instruments(),
                'indicators': self.config_loader.get_indicator_config(),
                'initial_capital': self.config_loader.get_capital(),
                'trading_mode': self.config_loader.get_trading_mode(),
                'validation_issues': self.config_loader.validate_enhanced_config()
            }
            
            self.logger.info(f"Loaded backtest configuration: "
                           f"{len(config['strategies'])} strategies, "
                           f"{len(config['instruments'])} instruments, "
                           f"{len(config['entry_rules'])} entry rules, "
                           f"{len(config['exit_rules'])} exit rules")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error loading backtest configuration: {e}")
            return {}
    
    def validate_backtest_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate backtest configuration using existing validation methods
        
        Args:
            config: Backtest configuration dictionary
            
        Returns:
            List of validation issues
        """
        issues = []
        
        # Use existing validation
        config_issues = config.get('validation_issues', [])
        issues.extend(config_issues)
        
        # Additional backtest-specific validation
        if not config.get('strategies'):
            issues.append("No active strategies found for backtesting")
        
        if not config.get('instruments'):
            issues.append("No active instruments found for backtesting")
        
        # Validate strategy-rule linkages
        entry_rules = config.get('entry_rules', {})
        exit_rules = config.get('exit_rules', {})
        strategies = config.get('strategies', {})
        
        for strategy_name, strategy_data in strategies.items():
            if strategy_data.get('status', '').lower() != 'active':
                continue
                
            entry_rule = strategy_data.get('entry_rule', '')
            if entry_rule and entry_rule not in entry_rules:
                issues.append(f"Strategy '{strategy_name}' references non-existent entry rule '{entry_rule}'")
            
            exit_rule_list = strategy_data.get('exit_rules', [])
            for exit_rule in exit_rule_list:
                if exit_rule and exit_rule not in exit_rules:
                    issues.append(f"Strategy '{strategy_name}' references non-existent exit rule '{exit_rule}'")
        
        return issues
    
    def build_rule_description(self, rule_name: str, rule_data: Dict[str, Any], 
                              rule_type: str = "entry") -> str:
        """
        Build human-readable rule description using existing ConfigLoader method
        
        Args:
            rule_name: Name of the rule
            rule_data: Rule data from ConfigLoader
            rule_type: Type of rule ("entry" or "exit")
            
        Returns:
            Human-readable rule description
        """
        try:
            if rule_type == "entry" and 'condition_groups' in rule_data:
                return self.config_loader.build_rule_expression(rule_data['condition_groups'])
            elif rule_type == "exit" and 'conditions' in rule_data:
                conditions = rule_data['conditions']
                if len(conditions) == 1:
                    cond = conditions[0]
                    return f"{cond['indicator']} {cond['operator']} {cond['value']}"
                else:
                    cond_strs = []
                    for cond in conditions:
                        cond_strs.append(f"{cond['indicator']} {cond['operator']} {cond['value']}")
                    return " AND ".join(cond_strs)
            else:
                return f"Rule: {rule_name}"
                
        except Exception as e:
            self.logger.warning(f"Error building rule description for {rule_name}: {e}")
            return f"Rule: {rule_name}"
    
    def evaluate_condition(self, indicator_value: Any, operator_str: str, 
                          target_value: Any, market_data: pd.Series = None, 
                          execution_data: pd.Series = None, trade_record: Dict[str, Any] = None) -> bool:
        """
        Evaluate a single condition with support for LTP_* indicators and TP/SL substitution
        
        Args:
            indicator_value: Current indicator value
            operator_str: Operator string (>, <, =, etc.)
            target_value: Target value to compare against (can be a number, indicator name, or TP/SL)
            market_data: Market data for indicator-to-indicator comparisons
            execution_data: Execution data for LTP_* indicators (1-minute candle data)
            trade_record: Trade record for TP/SL value substitution
            
        Returns:
            Boolean result of condition evaluation
        """
        try:
            # Handle TP/SL value substitution - CRITICAL FIX: Check market_data first
            if isinstance(target_value, str):
                if target_value.upper() == 'TP':
                    # First check market_data (where we properly add TP values)
                    if market_data is not None and 'TP' in market_data.index:
                        target_value = market_data['TP']
                        self.logger.info(f"[TP_FIX] TP value substituted from market_data: {target_value}")
                    elif trade_record:
                        # Fallback to trade_record for backward compatibility
                        target_value = trade_record.get('take_profit', trade_record.get('tp', target_value))
                        self.logger.info(f"[TP_FIX] TP value substituted from trade_record: {target_value}")
                elif target_value.upper() == 'SL':
                    # First check market_data (where we properly add SL values)
                    if market_data is not None and 'SL' in market_data.index:
                        target_value = market_data['SL']
                        self.logger.info(f"[SL_FIX] SL value substituted from market_data: {target_value}")
                    elif trade_record:
                        # Fallback to trade_record for backward compatibility
                        target_value = trade_record.get('stop_loss', trade_record.get('sl', target_value))
                        self.logger.info(f"[SL_FIX] SL value substituted from trade_record: {target_value}")
            
            # Check if target_value is an indicator name (indicator-to-indicator comparison)
            if market_data is not None and isinstance(target_value, str):
                # Check if target_value looks like an indicator name
                if any(indicator_pattern in target_value.upper() for indicator_pattern in ['EMA_', 'SMA_', 'RSI_', 'MACD', 'BB_', 'LTP_']):
                    # Try to get the target indicator value from appropriate data source
                    target_indicator_value = None
                    
                    # Check if it's an LTP indicator
                    if target_value.upper().startswith('LTP_') and execution_data is not None:
                        # Map LTP indicators to execution data columns
                        ltp_mapping = {
                            'LTP_OPEN': 'open',
                            'LTP_HIGH': 'high', 
                            'LTP_LOW': 'low',
                            'LTP_CLOSE': 'close'
                        }
                        
                        ltp_key = target_value.upper()
                        if ltp_key in ltp_mapping:
                            execution_column = ltp_mapping[ltp_key]
                            if hasattr(execution_data, execution_column):
                                target_indicator_value = getattr(execution_data, execution_column)
                                self.logger.debug(f"LTP indicator found: {target_value} -> {target_indicator_value}")
                            else:
                                self.logger.warning(f"LTP indicator {target_value} column '{execution_column}' not found in execution data")
                                return False
                        else:
                            self.logger.warning(f"Unknown LTP indicator: {target_value}")
                            return False
                    else:
                        # Regular indicator - try exact match first
                        if target_value in market_data.index:
                            target_indicator_value = market_data[target_value]
                        else:
                            # Try alternative names
                            alt_names = [
                                target_value.upper(),
                                target_value.lower(),
                                target_value.replace('EMA_', 'ema_'),
                                target_value.replace('SMA_', 'sma_'),
                                target_value.replace('RSI_', 'rsi_')
                            ]
                            
                            for alt_name in alt_names:
                                if alt_name in market_data.index:
                                    target_indicator_value = market_data[alt_name]
                                    break
                    
                    if target_indicator_value is not None:
                        target_value = target_indicator_value
                        self.logger.debug(f"Indicator comparison: {indicator_value} {operator_str} {target_value}")
                    else:
                        self.logger.warning(f"Target indicator {target_value} not found in market data")
                        return False
            
            # Convert values to numeric if possible
            try:
                indicator_val = float(indicator_value)
                target_val = float(target_value)
            except (ValueError, TypeError):
                # Keep as strings for comparison
                indicator_val = str(indicator_value)
                target_val = str(target_value)
            
            # Get operator function
            op_func = self.operators.get(operator_str.strip())
            if not op_func:
                self.logger.warning(f"Unknown operator: {operator_str}")
                return False
            
            # Evaluate condition
            result = op_func(indicator_val, target_val)
            return bool(result)
            
        except Exception as e:
            self.logger.warning(f"Error evaluating condition {indicator_value} {operator_str} {target_value}: {e}")
            return False
    
    def evaluate_condition_group(self, condition_group: Dict[str, Any], 
                                market_data: pd.Series, execution_data: pd.Series = None, 
                                trade_record: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        Evaluate a group of conditions with AND/OR logic and LTP support
        
        Args:
            condition_group: Group of conditions from ConfigLoader format
            market_data: Current market data row with indicators
            execution_data: Execution data for LTP_* indicators (1-minute candle data)
            trade_record: Trade record for TP/SL value substitution
            
        Returns:
            Tuple of (group_result, matched_conditions)
        """
        try:
            conditions = condition_group.get('conditions', [])
            if not conditions:
                return False, []
            
            results = []
            debug_conditions = []
            matched_conditions = []
            
            for condition in conditions:
                indicator = condition.get('indicator', '')
                operator_str = condition.get('operator', '')
                target_value = condition.get('value', '')
                
                # CRITICAL FIX: Check if indicator is LTP_* type - check both market_data and execution_data
                if indicator.upper().startswith('LTP_'):
                    # First try to get LTP value from market_data (which has our added LTP indicators)
                    if indicator in market_data.index:
                        indicator_value = market_data[indicator]
                        result = self.evaluate_condition(indicator_value, operator_str, target_value, market_data, execution_data, trade_record)
                        results.append(result)
                        debug_conditions.append(f"{indicator}={indicator_value} {operator_str} {target_value} -> {result}")
                        
                        # CRITICAL DEBUG: Add detailed logging for LTP_High > TP condition
                        if indicator == 'LTP_High' and operator_str == '>' and target_value == 'TP':
                            tp_value = market_data.get('TP', 0)
                            self.logger.info(f"[CRITICAL_DEBUG] LTP_High condition: {indicator_value} > {tp_value} = {result}")
                            self.logger.info(f"[CRITICAL_DEBUG] Raw comparison: {indicator_value} > {tp_value} = {indicator_value > tp_value}")
                        
                        # Track actually matched conditions
                        if result:
                            matched_conditions.append(f"{indicator} {operator_str} {target_value}")
                        continue
                    
                    # Fallback to execution_data if not in market_data
                    if execution_data is None:
                        self.logger.warning(f"LTP indicator {indicator} not in market_data and execution_data is None")
                        results.append(False)
                        debug_conditions.append(f"{indicator}=NO_DATA -> False")
                        continue
                    
                    # Map LTP indicators to execution data columns
                    ltp_mapping = {
                        'LTP_OPEN': 'open',
                        'LTP_HIGH': 'high',
                        'LTP_High': 'high',  # Support mixed case
                        'LTP_LOW': 'low',
                        'LTP_Low': 'low',    # Support mixed case
                        'LTP_CLOSE': 'close',
                        'LTP_Close': 'close' # Support mixed case
                    }
                    
                    # CRITICAL FIX: Check both original case and uppercase
                    ltp_key = indicator  # Check original case first
                    if ltp_key not in ltp_mapping:
                        ltp_key = indicator.upper()  # Try uppercase if original fails
                    
                    if ltp_key in ltp_mapping:
                        execution_column = ltp_mapping[ltp_key]
                        if hasattr(execution_data, execution_column):
                            indicator_value = getattr(execution_data, execution_column)
                            result = self.evaluate_condition(indicator_value, operator_str, target_value, market_data, execution_data, trade_record)
                            results.append(result)
                            debug_conditions.append(f"{indicator}={indicator_value} {operator_str} {target_value} -> {result}")
                            
                            # Track actually matched conditions
                            if result:
                                matched_conditions.append(f"{indicator} {operator_str} {target_value}")
                        else:
                            self.logger.warning(f"LTP indicator {indicator} column '{execution_column}' not found in execution data")
                            results.append(False)
                            debug_conditions.append(f"{indicator}=NOT_FOUND -> False")
                    else:
                        self.logger.warning(f"Unknown LTP indicator: {indicator}")
                        results.append(False)
                        debug_conditions.append(f"{indicator}=UNKNOWN_LTP -> False")
                # Get indicator value from market data
                elif indicator in market_data.index:
                    indicator_value = market_data[indicator]
                    result = self.evaluate_condition(indicator_value, operator_str, target_value, market_data, execution_data, trade_record)
                    results.append(result)
                    debug_conditions.append(f"{indicator}={indicator_value} {operator_str} {target_value} -> {result}")
                    
                    # Track actually matched conditions
                    if result:
                        matched_conditions.append(f"{indicator} {operator_str} {target_value}")
                else:
                    # Try alternative indicator names (case variations)
                    alt_indicators = [
                        indicator.upper(),
                        indicator.lower(), 
                        indicator.replace('_', '').lower(),
                        indicator.replace('EMA_', 'ema_'),
                        indicator.replace('RSI_', 'rsi_'),
                        indicator.replace('SMA_', 'sma_')
                    ]
                    
                    found = False
                    for alt_indicator in alt_indicators:
                        if alt_indicator in market_data.index:
                            indicator_value = market_data[alt_indicator]
                            result = self.evaluate_condition(indicator_value, operator_str, target_value, market_data, execution_data, trade_record)
                            results.append(result)
                            debug_conditions.append(f"{indicator}({alt_indicator})={indicator_value} {operator_str} {target_value} -> {result}")
                            
                            # Track actually matched conditions
                            if result:
                                matched_conditions.append(f"{indicator} {operator_str} {target_value}")
                            found = True
                            break
                    
                    if not found:
                        # Check if it's an LTP indicator we missed
                        if indicator.upper().startswith('LTP_'):
                            self.logger.warning(f"LTP indicator {indicator} not found - missing execution data")
                            debug_conditions.append(f"{indicator}=LTP_MISSING_EXECUTION -> False")
                        else:
                            available_indicators = [col for col in market_data.index if any(x in col.lower() for x in ['ema', 'rsi', 'sma'])]
                            self.logger.warning(f"Indicator {indicator} not found. Available: {available_indicators[:10]}")
                            debug_conditions.append(f"{indicator}=NOT_FOUND -> False")
                        results.append(False)
            
            # For conditions within a group, use OR logic (as per ConfigLoader design)
            group_result = any(results)
            
            # Debug log for EMA conditions
            if any('EMA' in str(cond) for cond in debug_conditions):
                timestamp = market_data.get('timestamp', 'unknown')
                if hasattr(timestamp, 'minute') and timestamp.minute % 10 == 0:  # Log every 10 minutes
                    self.logger.debug(f"Condition Group @ {timestamp}: {debug_conditions} -> {group_result}")
            
            return group_result, matched_conditions
            
        except Exception as e:
            self.logger.error(f"Error evaluating condition group: {e}")
            return False, []
    
    def evaluate_entry_rule(self, rule_name: str, rule_data: Dict[str, Any], 
                           market_data: pd.Series, execution_data: pd.Series = None) -> Tuple[bool, List[str]]:
        """
        Evaluate entry rule using ConfigLoader enhanced format with LTP support
        
        Args:
            rule_name: Name of the entry rule
            rule_data: Rule data from get_entry_conditions_enhanced()
            market_data: Current market data row with indicators
            execution_data: Execution data for LTP_* indicators (1-minute candle data)
            
        Returns:
            Tuple of (rule_passed, matched_conditions)
        """
        try:
            # Check if rule is active
            if rule_data.get('status', '').lower() != 'active':
                self.logger.debug(f"Rule {rule_name} is not active: status={rule_data.get('status')}")
                return False, []
            
            condition_groups = rule_data.get('condition_groups', [])
            if not condition_groups:
                self.logger.debug(f"Rule {rule_name} has no condition groups")
                return False, []
            
            group_results = []
            matched_conditions = []
            debug_info = []
            
            # Evaluate each condition group
            for i, group in enumerate(condition_groups):
                group_result, group_matched_conditions = self.evaluate_condition_group(group, market_data, execution_data)
                group_results.append(group_result)
                
                # Debug logging for each group
                group_conditions = group.get('conditions', [])
                debug_info.append(f"Group {i+1}: {group_result} (conditions: {len(group_conditions)})")
                
                if group_result:
                    # Add only the actually matched conditions to list
                    matched_conditions.extend(group_matched_conditions)
            
            # Between groups, use AND logic (as per ConfigLoader design)
            rule_passed = all(group_results)
            
            # Enhanced debugging for EMA_Crossover_Long
            if rule_name == "EMA_Crossover_Long":
                # Only log every 1000th evaluation to avoid spam
                timestamp = market_data.get('timestamp', 'unknown')
                if hasattr(timestamp, 'minute') and timestamp.minute % 10 == 0:  # Log every 10 minutes
                    self.logger.debug(f"[{rule_name}] @ {timestamp}: Groups={debug_info}, Result={rule_passed}")
                    
                    # Log actual EMA values for debugging
                    try:
                        ema_9 = market_data.get('EMA_9', market_data.get('ema_9', 'N/A'))
                        ema_20 = market_data.get('EMA_20', market_data.get('ema_20', 'N/A'))
                        ema_50 = market_data.get('EMA_50', market_data.get('ema_50', 'N/A'))
                        ema_200 = market_data.get('EMA_200', market_data.get('ema_200', 'N/A'))
                        self.logger.debug(f"[{rule_name}] EMAs: 9={ema_9}, 20={ema_20}, 50={ema_50}, 200={ema_200}")
                    except Exception:
                        pass
            
            return rule_passed, matched_conditions if rule_passed else []
            
        except Exception as e:
            self.logger.error(f"Error evaluating entry rule {rule_name}: {e}")
            return False, []
    
    def evaluate_exit_rule(self, rule_name: str, rule_data: Dict[str, Any], 
                          market_data: pd.Series, execution_data: pd.Series = None, 
                          trade_record: Dict[str, Any] = None) -> Tuple[bool, List[str]]:
        """
        Evaluate exit rule using ConfigLoader enhanced format with LTP support and condition groups
        
        Args:
            rule_name: Name of the exit rule
            rule_data: Rule data from get_exit_conditions_enhanced()
            market_data: Current market data row with indicators
            execution_data: Execution data for LTP_* indicators (1-minute candle data)
            trade_record: Trade record for TP/SL value substitution
            
        Returns:
            Tuple of (rule_passed, matched_conditions)
        """
        try:
            # Check if rule is active
            if rule_data.get('status', '').lower() != 'active':
                return False, []
            
            # Check if new condition_groups structure is available
            condition_groups = rule_data.get('condition_groups', [])
            if condition_groups:
                # Use new sophisticated condition_groups structure (like entry conditions)
                group_results = []
                matched_conditions = []
                debug_info = []
                
                # Evaluate each condition group
                for i, group in enumerate(condition_groups):
                    group_result, group_matched_conditions = self.evaluate_condition_group(group, market_data, execution_data, trade_record)
                    group_results.append(group_result)
                    
                    # Debug logging for each group
                    group_conditions = group.get('conditions', [])
                    debug_info.append(f"Group {i+1}: {group_result} (conditions: {len(group_conditions)})")
                    
                    if group_result:
                        # Add only the actually matched conditions to list
                        matched_conditions.extend(group_matched_conditions)
                
                # For exit conditions, use OR logic between groups (any group can trigger exit)
                rule_passed = any(group_results)
                
                # Debug logging for multi-condition exit rules
                if len(condition_groups) > 1:
                    timestamp = market_data.get('timestamp', 'unknown')
                    self.logger.debug(f"[{rule_name}] Multi-condition exit @ {timestamp}: Groups={debug_info}, Result={rule_passed}")
                
                return rule_passed, matched_conditions if rule_passed else []
                
            else:
                # Fallback to old conditions structure for backward compatibility
                conditions = rule_data.get('conditions', [])
                if not conditions:
                    return False, []
                
                results = []
                matched_conditions = []
                
                # Evaluate each condition
                for condition in conditions:
                    indicator = condition.get('indicator', '')
                    operator_str = condition.get('operator', '')
                    target_value = condition.get('value', '')
                    
                    # Check if indicator is LTP_* type
                    if indicator.upper().startswith('LTP_'):
                        if execution_data is None:
                            self.logger.warning(f"LTP indicator {indicator} requested but execution_data is None")
                            results.append(False)
                            continue
                        
                        # Map LTP indicators to execution data columns
                        ltp_mapping = {
                            'LTP_OPEN': 'open',
                            'LTP_HIGH': 'high',
                            'LTP_High': 'high',  # Support mixed case
                            'LTP_LOW': 'low',
                            'LTP_Low': 'low',    # Support mixed case
                            'LTP_CLOSE': 'close',
                            'LTP_Close': 'close' # Support mixed case
                        }
                        
                        # CRITICAL FIX: Check both original case and uppercase
                        ltp_key = indicator  # Check original case first
                        if ltp_key not in ltp_mapping:
                            ltp_key = indicator.upper()  # Try uppercase if original fails
                        
                        if ltp_key in ltp_mapping:
                            execution_column = ltp_mapping[ltp_key]
                            if hasattr(execution_data, execution_column):
                                indicator_value = getattr(execution_data, execution_column)
                                result = self.evaluate_condition(indicator_value, operator_str, target_value, market_data, execution_data, trade_record)
                                results.append(result)
                                
                                if result:
                                    cond_str = f"{indicator} {operator_str} {target_value}"
                                    matched_conditions.append(cond_str)
                            else:
                                self.logger.warning(f"LTP indicator {indicator} column '{execution_column}' not found in execution data")
                                self.logger.warning(f"Execution data attributes: {dir(execution_data) if execution_data is not None else 'None'}")
                                results.append(False)
                        else:
                            self.logger.warning(f"Unknown LTP indicator: {indicator}")
                            results.append(False)
                    # Get indicator value from market data
                    elif indicator in market_data.index:
                        indicator_value = market_data[indicator]
                        result = self.evaluate_condition(indicator_value, operator_str, target_value, market_data, execution_data, trade_record)
                        results.append(result)
                        
                        if result:
                            cond_str = f"{indicator} {operator_str} {target_value}"
                            matched_conditions.append(cond_str)
                    else:
                        # Check if it's an LTP indicator we missed
                        if indicator.upper().startswith('LTP_'):
                            self.logger.warning(f"LTP indicator {indicator} not found - missing execution data")
                        else:
                            available_indicators = [col for col in market_data.index if any(x in col.lower() for x in ['ema', 'rsi', 'sma'])]
                            self.logger.warning(f"Indicator {indicator} not found. Available: {available_indicators[:10]}")
                        results.append(False)
                
                # For exit conditions, use OR logic (any condition can trigger exit)
                rule_passed = any(results)
                
                return rule_passed, matched_conditions if rule_passed else []
            
        except Exception as e:
            self.logger.error(f"Error evaluating exit rule {rule_name}: {e}")
            return False, []
    
    def get_active_strategies_for_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """
        Get active strategies suitable for backtesting
        
        Args:
            config: Backtest configuration dictionary
            
        Returns:
            Dictionary of active strategies
        """
        strategies = config.get('strategies', {})
        active_strategies = {}
        
        for strategy_name, strategy_data in strategies.items():
            if strategy_data.get('status', '').lower() == 'active':
                # Ensure strategy has required components
                if (strategy_data.get('entry_rule') and 
                    strategy_data.get('exit_rules')):
                    active_strategies[strategy_name] = strategy_data
                else:
                    self.logger.warning(f"Strategy {strategy_name} missing entry or exit rules")
        
        self.logger.info(f"Found {len(active_strategies)} active strategies for backtesting")
        return active_strategies
    
    def calculate_position_size(self, strategy_config: Dict[str, Any], 
                               current_price: float, available_capital: float) -> int:
        """
        Calculate position size based on strategy configuration
        
        Args:
            strategy_config: Strategy configuration from ConfigLoader
            current_price: Current market price
            available_capital: Available capital for trading
            
        Returns:
            Position size (number of shares/contracts)
        """
        try:
            position_size = strategy_config.get('position_size', 1000)
            
            # For now, use fixed position size
            # Future enhancement: implement percentage-based and risk-based sizing
            return int(position_size)
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1000  # Default fallback
    
    def format_timestamp(self, timestamp: Union[str, datetime, pd.Timestamp]) -> datetime:
        """
        Format timestamp to datetime object
        
        Args:
            timestamp: Timestamp in various formats
            
        Returns:
            datetime object
        """
        try:
            if isinstance(timestamp, datetime):
                return timestamp
            elif isinstance(timestamp, pd.Timestamp):
                return timestamp.to_pydatetime()
            elif isinstance(timestamp, str):
                return pd.to_datetime(timestamp).to_pydatetime()
            else:
                return pd.to_datetime(timestamp).to_pydatetime()
        except Exception as e:
            self.logger.error(f"Error formatting timestamp {timestamp}: {e}")
            return datetime.now()
    
    def generate_trade_id(self, symbol: str, strategy_name: str, 
                         timestamp: datetime, trade_type: str) -> str:
        """
        Generate unique trade ID
        
        Args:
            symbol: Trading symbol
            strategy_name: Strategy name
            timestamp: Trade timestamp
            trade_type: BUY or SELL
            
        Returns:
            Unique trade ID string
        """
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")
        return f"{symbol}_{strategy_name}_{trade_type}_{timestamp_str}"
    
    def log_configuration_summary(self, config: Dict[str, Any]) -> None:
        """
        Log summary of backtest configuration
        
        Args:
            config: Backtest configuration dictionary
        """
        try:
            self.logger.info("=== Backtest Configuration Summary ===")
            self.logger.info(f"Initial Capital: ${config.get('initial_capital', 0):,.2f}")
            self.logger.info(f"Active Strategies: {len(config.get('strategies', {}))}")
            self.logger.info(f"Active Instruments: {len(config.get('instruments', []))}")
            self.logger.info(f"Entry Rules: {len(config.get('entry_rules', {}))}")
            self.logger.info(f"Exit Rules: {len(config.get('exit_rules', {}))}")
            
            # Log validation issues if any
            issues = config.get('validation_issues', [])
            if issues:
                self.logger.warning(f"Configuration Issues Found: {len(issues)}")
                for issue in issues:
                    self.logger.warning(f"  - {issue}")
            else:
                self.logger.info("Configuration validation: PASSED")
                
        except Exception as e:
            self.logger.error(f"Error logging configuration summary: {e}")


def create_backtest_utils(config_file: str = None) -> BacktestUtils:
    """
    Factory function to create BacktestUtils with ConfigLoader
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        BacktestUtils instance
    """
    try:
        from utils.config_cache import get_cached_config
        config_cache = get_cached_config(config_file)
        config_loader = config_cache.get_config_loader()
        if config_loader:
            return BacktestUtils(config_loader)
        else:
            raise ValueError("Failed to load configuration from cache")
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error creating BacktestUtils: {e}")
        raise


def validate_market_data(df: pd.DataFrame) -> List[str]:
    """
    Validate market data DataFrame for backtesting
    
    Args:
        df: Market data DataFrame
        
    Returns:
        List of validation issues
    """
    issues = []
    
    # Check required columns
    required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing required columns: {missing_columns}")
    
    # Check data types and values
    if 'timestamp' in df.columns:
        try:
            pd.to_datetime(df['timestamp'])
        except:
            issues.append("Invalid timestamp format")
    
    price_columns = ['open', 'high', 'low', 'close']
    for col in price_columns:
        if col in df.columns:
            if (df[col] <= 0).any():
                issues.append(f"Non-positive values found in {col}")
    
    if 'volume' in df.columns:
        if (df['volume'] < 0).any():
            issues.append("Negative values found in volume")
    
    # Check OHLC relationship
    if all(col in df.columns for col in price_columns):
        invalid_ohlc = (
            (df['high'] < df['low']) |
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close'])
        ).any()
        
        if invalid_ohlc:
            issues.append("Invalid OHLC relationships found")
    
    return issues