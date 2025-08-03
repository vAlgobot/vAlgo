"""
Enhanced Rule Engine for vAlgo Trading System

Integrates all enhanced components for Excel-driven strategy configuration:
- Row-by-row strategy processing (1:1 entry-exit pairing)
- Active status filtering and condition_order logic
- AND/OR logic evaluation within rules
- Entry-exit state machine for tracking trades
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Union, Callable
from datetime import datetime, time
from pathlib import Path

from .models import RuleResult, IndicatorData
from .enhanced_condition_loader import EnhancedConditionLoader, StrategyRow, RuleCondition
from .condition_group_evaluator import ConditionGroupEvaluator, evaluate_conditions_with_logic
from .entry_exit_state_machine import EntryExitStateMachine, create_state_machine_from_config
from .operator_resolver import OperatorResolver
from utils.initialize_config_loader import InitializeConfigLoader


class EnhancedRuleEngine:
    """
    Enhanced rule engine with Excel-driven strategy configuration
    
    Features:
    - Row-by-row strategy processing from strategy_config
    - 1:1 entry-exit rule pairing per strategy row
    - Active status filtering within rules
    - condition_order based AND/OR logic evaluation
    - State machine for tracking entry-exit lifecycle
    - Complete configuration-driven approach
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize enhanced rule engine
        
        Args:
            config_path: Path to Excel config file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path or "config/config.xlsx"
        
        # Load initialization configuration for daily limits and trade hours
        self.init_config = InitializeConfigLoader(self.config_path)
        self.init_config.load_config()
        self.daily_entry_limit = self.init_config.get_parameter('daily_entry_limit', 1)
        
        # PHASE 1 FIX: Load trade hours configuration
        trade_start_str = self.init_config.get_parameter('trade_start_time', '09:20:00')
        trade_end_str = self.init_config.get_parameter('trade_end_time', '15:00:00')
        
        # Parse trade hours (format: HH:MM:SS or time object)
        try:
            from datetime import time
            
            # Handle if already time objects or need string parsing
            if isinstance(trade_start_str, time):
                self.trade_start_time = trade_start_str
            else:
                start_parts = str(trade_start_str).split(':')
                self.trade_start_time = time(int(start_parts[0]), int(start_parts[1]), int(start_parts[2]) if len(start_parts) > 2 else 0)
            
            if isinstance(trade_end_str, time):
                self.trade_end_time = trade_end_str
            else:
                end_parts = str(trade_end_str).split(':')
                self.trade_end_time = time(int(end_parts[0]), int(end_parts[1]), int(end_parts[2]) if len(end_parts) > 2 else 0)
            
            self.logger.info(f"Enhanced rule engine configured:")
            self.logger.info(f"  Daily_Entry_Limit: {self.daily_entry_limit}")
            self.logger.info(f"  Trade_Start_Time: {self.trade_start_time}")
            self.logger.info(f"  Trade_End_Time: {self.trade_end_time}")
            
        except Exception as e:
            self.logger.error(f"Error parsing trade hours: {e}")
            # Fallback to default values
            from datetime import time
            self.trade_start_time = time(9, 20, 0)  # 09:20:00
            self.trade_end_time = time(15, 0, 0)    # 15:00:00
            self.logger.warning(f"Using default trade hours: {self.trade_start_time} - {self.trade_end_time}")
        
        # Initialize components
        self.condition_loader = EnhancedConditionLoader(config_path)
        self.operator_resolver = OperatorResolver()
        self.condition_evaluator = ConditionGroupEvaluator(self.operator_resolver)
        self.state_machine = EntryExitStateMachine()
        
        # Cache for loaded data
        self.strategy_rows: List[StrategyRow] = []
        self.entry_rule_conditions: Dict[str, List[RuleCondition]] = {}
        self.exit_rule_conditions: Dict[str, List[RuleCondition]] = {}
        
        # Daily entry limit tracking
        self.daily_entry_counts: Dict[str, int] = {}
        self.current_date: Optional[str] = None
        self.trade_history: List[Dict[str, Any]] = []
        
        # Strategy-specific cumulative daily counters for output tracking
        self.strategy_entries_today: Dict[str, int] = {}
        self.strategy_exits_today: Dict[str, int] = {}
        
        # Configuration state
        self._loaded = False
        
        # Load configuration on initialization
        self._load_configuration()
    
    def _load_configuration(self) -> bool:
        """Load complete configuration from Excel"""
        try:
            if not self.condition_loader.load_config():
                self.logger.error("Failed to load configuration")
                return False
            
            # Load strategy rows
            self.strategy_rows = self.condition_loader.load_strategy_rows()
            if not self.strategy_rows:
                self.logger.error("No active strategy rows found")
                return False
            
            # Load rule conditions for all referenced rules
            self._load_rule_conditions()
            
            # Initialize state machine
            self.state_machine.initialize_from_strategy_rows(self.strategy_rows)
            
            self._loaded = True
            self.logger.info(f"Enhanced rule engine loaded: {len(self.strategy_rows)} strategy rows, "
                           f"{len(self.entry_rule_conditions)} entry rules, "
                           f"{len(self.exit_rule_conditions)} exit rules")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading enhanced configuration: {e}")
            return False
    
    def _load_rule_conditions(self) -> None:
        """Load conditions for all entry and exit rules"""
        # Get unique rule names
        entry_rules = set()
        exit_rules = set()
        
        for row in self.strategy_rows:
            entry_rules.add(row.entry_rule)
            exit_rules.add(row.exit_rule)
        
        # Load entry rule conditions
        for rule_name in entry_rules:
            conditions = self.condition_loader.load_rule_conditions(rule_name, "entry")
            if conditions:
                self.entry_rule_conditions[rule_name] = conditions
        
        # Load exit rule conditions
        for rule_name in exit_rules:
            conditions = self.condition_loader.load_rule_conditions(rule_name, "exit")
            if conditions:
                self.exit_rule_conditions[rule_name] = conditions
    
    def reload_configuration(self) -> bool:
        """Reload configuration from Excel file"""
        self.logger.info("Reloading enhanced configuration")
        
        # Clear caches
        self.strategy_rows.clear()
        self.entry_rule_conditions.clear()
        self.exit_rule_conditions.clear()
        self.state_machine.reset_state()
        self._loaded = False
        
        # Reload
        return self._load_configuration()
    
    def evaluate_entry_rule(self, rule_name: str, indicator_data: IndicatorData, 
                           timestamp: Optional[datetime] = None) -> RuleResult:
        """
        Evaluate an entry rule with enhanced logic
        
        Args:
            rule_name: Name of the entry rule
            indicator_data: Current indicator values
            timestamp: Timestamp for evaluation (optional)
            
        Returns:
            RuleResult with detailed evaluation
        """
        if not self._loaded:
            return RuleResult(
                rule_name=rule_name,
                triggered=False,
                error_message="Configuration not loaded"
            )
        
        if rule_name not in self.entry_rule_conditions:
            return RuleResult(
                rule_name=rule_name,
                triggered=False,
                error_message=f"Entry rule '{rule_name}' not found or has no active conditions"
            )
        
        try:
            conditions = self.entry_rule_conditions[rule_name]
            result = self.condition_evaluator.evaluate_rule_conditions(
                conditions, indicator_data, rule_name
            )
            result.timestamp = timestamp or datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating entry rule '{rule_name}': {e}")
            return RuleResult(
                rule_name=rule_name,
                triggered=False,
                error_message=f"Evaluation error: {e}",
                timestamp=timestamp or datetime.now()
            )
    
    def evaluate_exit_rule(self, rule_name: str, indicator_data: IndicatorData, 
                          trade_data: Optional[Dict[str, Any]] = None,
                          timestamp: Optional[datetime] = None) -> RuleResult:
        """
        Evaluate an exit rule with enhanced logic
        
        Args:
            rule_name: Name of the exit rule
            indicator_data: Current indicator values
            trade_data: Trade data (SL, TP, etc.) - not used in this implementation
            timestamp: Timestamp for evaluation (optional)
            
        Returns:
            RuleResult with detailed evaluation
        """
        if not self._loaded:
            return RuleResult(
                rule_name=rule_name,
                triggered=False,
                error_message="Configuration not loaded"
            )
        
        if rule_name not in self.exit_rule_conditions:
            return RuleResult(
                rule_name=rule_name,
                triggered=False,
                error_message=f"Exit rule '{rule_name}' not found or has no active conditions"
            )
        
        try:
            conditions = self.exit_rule_conditions[rule_name]
            result = self.condition_evaluator.evaluate_rule_conditions(
                conditions, indicator_data, rule_name
            )
            result.timestamp = timestamp or datetime.now()
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error evaluating exit rule '{rule_name}': {e}")
            return RuleResult(
                rule_name=rule_name,
                triggered=False,
                error_message=f"Evaluation error: {e}",
                timestamp=timestamp or datetime.now()
            )
    
    def process_signal_data_enhanced(self, indicator_data: IndicatorData, 
                                   timestamp: Optional[datetime] = None) -> Dict[str, Any]:
        """
        Process signal data using enhanced strategy logic
        
        Args:
            indicator_data: Current indicator values
            timestamp: Timestamp for processing (optional)
            
        Returns:
            Dictionary with processing results
        """
        if not self._loaded:
            return {
                'error': 'Configuration not loaded',
                'timestamp': timestamp or datetime.now()
            }
        
        processing_timestamp = timestamp or datetime.now()
        
        # Update daily tracking
        self._update_daily_tracking(processing_timestamp)
        
        # Define evaluator functions for state machine with daily limit enforcement
        def entry_evaluator(rule_name: str, data: IndicatorData) -> RuleResult:
            # PHASE 2 FIX: Check trade hours, daily limit, and position state
            if not self._can_enter_new_trade(rule_name, processing_timestamp):
                # Determine the specific reason for blocking
                if not self._is_within_trade_hours(processing_timestamp):
                    error_msg = f"Outside trade hours ({self.trade_start_time} - {self.trade_end_time})"
                elif self.daily_entry_counts.get(self._get_strategy_key_for_rule(rule_name), 0) >= self.daily_entry_limit:
                    strategy_name = self._get_strategy_key_for_rule(rule_name)
                    error_msg = f"Daily entry limit ({self.daily_entry_limit}) reached for strategy '{strategy_name}'"
                else:
                    error_msg = "Active position exists"
                
                return RuleResult(
                    rule_name=rule_name,
                    triggered=False,
                    error_message=error_msg,
                    timestamp=processing_timestamp
                )
            
            # Evaluate the entry rule normally
            result = self.evaluate_entry_rule(rule_name, data, processing_timestamp)
            
            # If entry is triggered, increment daily count
            if result.triggered:
                self._increment_daily_entry_count(rule_name)
            
            return result
        
        def exit_evaluator(rule_name: str, data: IndicatorData) -> RuleResult:
            return self.evaluate_exit_rule(rule_name, data, None, processing_timestamp)
        
        # Process through state machine
        self.logger.debug(f"Processing signal data at {processing_timestamp}")
        results = self.state_machine.process_signal_data(
            indicator_data, processing_timestamp, entry_evaluator, exit_evaluator
        )
        
        # Increment strategy-specific cumulative counters based on results
        entries_triggered = results.get('entries_triggered', [])
        exits_triggered = results.get('exits_triggered', [])
        
        # Process entry signals per strategy
        for entry in entries_triggered:
            strategy_name = entry.get('strategy_name', 'Unknown')
            if strategy_name not in self.strategy_entries_today:
                self.strategy_entries_today[strategy_name] = 0
            self.strategy_entries_today[strategy_name] += 1
            self.logger.debug(f"Incremented {strategy_name} entries to {self.strategy_entries_today[strategy_name]}")
        
        # Process exit signals per strategy
        for exit in exits_triggered:
            strategy_name = exit.get('strategy_name', 'Unknown')
            if strategy_name not in self.strategy_exits_today:
                self.strategy_exits_today[strategy_name] = 0
            self.strategy_exits_today[strategy_name] += 1
            self.logger.debug(f"Incremented {strategy_name} exits to {self.strategy_exits_today[strategy_name]}")
        
        # Add strategy-specific counters to results for export
        results['strategy_entries_today'] = self.strategy_entries_today.copy()
        results['strategy_exits_today'] = self.strategy_exits_today.copy()
        
        # Add engine state information
        results['engine_state'] = self.get_engine_state()
        
        return results
    
    def get_strategy_rows(self) -> List[StrategyRow]:
        """Get loaded strategy rows"""
        return self.strategy_rows.copy()
    
    def get_entry_rule_names(self) -> List[str]:
        """Get all loaded entry rule names"""
        return list(self.entry_rule_conditions.keys())
    
    def get_exit_rule_names(self) -> List[str]:
        """Get all loaded exit rule names"""
        return list(self.exit_rule_conditions.keys())
    
    def get_rule_conditions(self, rule_name: str, rule_type: str) -> Optional[List[RuleCondition]]:
        """
        Get conditions for a specific rule
        
        Args:
            rule_name: Name of the rule
            rule_type: "entry" or "exit"
            
        Returns:
            List of RuleCondition objects or None
        """
        if rule_type == "entry":
            return self.entry_rule_conditions.get(rule_name)
        elif rule_type == "exit":
            return self.exit_rule_conditions.get(rule_name)
        else:
            return None
    
    def get_engine_state(self) -> Dict[str, Any]:
        """Get current engine state"""
        return {
            'loaded': self._loaded,
            'config_path': self.config_path,
            'strategy_rows_count': len(self.strategy_rows),
            'entry_rules_count': len(self.entry_rule_conditions),
            'exit_rules_count': len(self.exit_rule_conditions),
            'state_machine_state': self.state_machine.get_current_state()
        }
    
    def get_strategy_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all strategy rows"""
        summary = []
        
        for row in self.strategy_rows:
            entry_conditions_count = len(self.entry_rule_conditions.get(row.entry_rule, []))
            exit_conditions_count = len(self.exit_rule_conditions.get(row.exit_rule, []))
            
            summary.append({
                'strategy_name': row.strategy_name,
                'entry_rule': row.entry_rule,
                'exit_rule': row.exit_rule,
                'active': row.active,
                'position_size': row.position_size,
                'entry_conditions_count': entry_conditions_count,
                'exit_conditions_count': exit_conditions_count
            })
        
        return summary
    
    def validate_configuration(self) -> List[str]:
        """Validate the entire configuration"""
        issues = []
        
        try:
            # Validate basic configuration
            config_issues = self.condition_loader.validate_configuration()
            issues.extend(config_issues)
            
            # Validate rule conditions against available indicators
            # (This would require knowing available indicators, which could come from signal data)
            
            # Validate state machine initialization
            if not self.strategy_rows:
                issues.append("No strategy rows loaded for state machine")
            
            self.logger.info(f"Configuration validation completed with {len(issues)} issues")
            
        except Exception as e:
            issues.append(f"Validation error: {e}")
        
        return issues
    
    def get_trade_lifecycle_report(self) -> Dict[str, Any]:
        """Get comprehensive trade lifecycle report"""
        return self.state_machine.get_trade_lifecycle_report()
    
    def get_completed_trades(self) -> List[Dict[str, Any]]:
        """Get all completed trades"""
        return self.state_machine.get_completed_trades_summary()
    
    def get_active_trades(self) -> List[Dict[str, Any]]:
        """Get all active trades"""
        return self.state_machine.get_active_trades_summary()
    
    def reset_state_machine(self) -> None:
        """Reset the state machine to initial state"""
        self.state_machine.reset_state()
        if self.strategy_rows:
            self.state_machine.initialize_from_strategy_rows(self.strategy_rows)
    
    def _update_daily_tracking(self, timestamp: datetime) -> None:
        """
        PHASE 3 FIX: Update daily tracking based on Trade_Start_Time instead of just date change
        Reset daily counters when new trading day starts at Trade_Start_Time
        """
        current_date = timestamp.strftime('%Y-%m-%d')
        current_time = timestamp.time()
        
        # Determine if this is a new trading day
        # A new trading day starts when we reach Trade_Start_Time on a new date
        # OR when we reach Trade_Start_Time on same date but after Trade_End_Time
        is_new_trading_day = False
        
        if self.current_date != current_date:
            # Date changed - definitely a new trading day if we're at or after start time
            if current_time >= self.trade_start_time:
                is_new_trading_day = True
        else:
            # Same date - check if we've crossed into new trading day
            # This handles overnight processing or multi-day backtesting
            if (hasattr(self, '_last_processed_time') and 
                self._last_processed_time is not None and
                self._last_processed_time < self.trade_start_time and 
                current_time >= self.trade_start_time):
                is_new_trading_day = True
        
        # Reset counts if new trading day detected
        if is_new_trading_day:
            if self.current_date is not None:
                self.logger.info(f"New trading day detected: {current_date} at {current_time} - resetting daily entry counts")
            
            self.current_date = current_date
            self.daily_entry_counts = {}
            
            # Reset strategy-specific cumulative counters for new trading day
            self.strategy_entries_today = {}
            self.strategy_exits_today = {}
            
            self.logger.info(f"Daily entry tracking initialized for trading day {current_date} (starts at {self.trade_start_time}, limit: {self.daily_entry_limit})")
        
        # Store last processed time for next iteration
        self._last_processed_time = current_time
    
    def _is_within_trade_hours(self, timestamp: datetime) -> bool:
        """
        PHASE 2 FIX: Check if timestamp is within configured trade hours
        
        Args:
            timestamp: Current timestamp to check
            
        Returns:
            True if within trade hours, False otherwise
        """
        current_time = timestamp.time()
        
        # Check if current time is between trade start and end times
        is_within_hours = self.trade_start_time <= current_time <= self.trade_end_time
        
        if not is_within_hours:
            self.logger.debug(f"[TRADE_HOURS] {timestamp} is outside trade hours ({self.trade_start_time} - {self.trade_end_time})")
        else:
            self.logger.debug(f"[TRADE_HOURS] {timestamp} is within trade hours ({self.trade_start_time} - {self.trade_end_time})")
        
        return is_within_hours
    
    def _is_within_strategy_hours(self, timestamp: datetime, start_time: time, end_time: time) -> bool:
        """
        Check if timestamp is within strategy-specific trade hours
        
        Args:
            timestamp: Current timestamp to check
            start_time: Strategy-specific start time
            end_time: Strategy-specific end time
            
        Returns:
            True if within strategy hours, False otherwise
        """
        current_time = timestamp.time()
        
        # Check if current time is between strategy start and end times
        is_within_hours = start_time <= current_time <= end_time
        
        if not is_within_hours:
            self.logger.debug(f"[STRATEGY_HOURS] {timestamp} is outside strategy hours ({start_time} - {end_time})")
        else:
            self.logger.debug(f"[STRATEGY_HOURS] {timestamp} is within strategy hours ({start_time} - {end_time})")
        
        return is_within_hours
    
    def _can_enter_new_trade(self, rule_name: str, timestamp: datetime) -> bool:
        """
        Enhanced: Check if a new entry is allowed based on strategy-specific trade hours, daily limits and active position state
        
        Args:
            rule_name: Name of the entry rule
            timestamp: Current timestamp for validation
            
        Returns:
            True if entry is allowed, False otherwise
        """
        strategy_key = self._get_strategy_key_for_rule(rule_name)
        
        # Check 1: Strategy-specific trade hours validation (if Run_Strategy configured)
        run_config = self.condition_loader.get_run_strategy_config(strategy_key)
        if run_config and run_config.start_time and run_config.end_time:
            # Use strategy-specific time window
            if not self._is_within_strategy_hours(timestamp, run_config.start_time, run_config.end_time):
                self.logger.debug(f"Entry blocked for {rule_name}: outside strategy-specific trade hours ({run_config.start_time} - {run_config.end_time})")
                return False
        else:
            # Use global trade hours
            if not self._is_within_trade_hours(timestamp):
                self.logger.debug(f"Entry blocked for {rule_name}: outside global trade hours")
                return False
        
        # Check 2: Strategy-specific daily entry limit (if configured) or global limit
        current_count = self.daily_entry_counts.get(strategy_key, 0)
        
        if run_config and run_config.max_daily_entries is not None:
            # Use strategy-specific daily limit
            daily_limit = run_config.max_daily_entries
        else:
            # Use global daily limit
            daily_limit = self.daily_entry_limit
        
        if current_count >= daily_limit:
            self.logger.debug(f"Daily entry limit reached for strategy '{strategy_key}': {current_count}/{daily_limit}")
            return False
        
        # Check 3: Active position state
        if self._has_active_position(rule_name):
            strategy_name = self._get_strategy_name_for_rule(rule_name)
            self.logger.debug(f"Entry blocked for {strategy_name}: already has active position")
            return False
        
        return True
    
    def _increment_daily_entry_count(self, rule_name: str) -> None:
        """Increment daily entry count for a strategy"""
        strategy_key = self._get_strategy_key_for_rule(rule_name)
        
        current_count = self.daily_entry_counts.get(strategy_key, 0)
        self.daily_entry_counts[strategy_key] = current_count + 1
        
        # Determine which daily limit to show in logging
        run_config = self.condition_loader.get_run_strategy_config(strategy_key)
        if run_config and run_config.max_daily_entries is not None:
            daily_limit = run_config.max_daily_entries
            limit_type = "strategy-specific"
        else:
            daily_limit = self.daily_entry_limit
            limit_type = "global"
        
        self.logger.info(f"Daily entry count incremented for strategy '{strategy_key}' (rule: {rule_name}): {current_count + 1}/{daily_limit} ({limit_type} limit)")
    
    def _get_strategy_key_for_rule(self, rule_name: str) -> str:
        """
        PHASE 1 FIX: Get strategy key for tracking daily limits at STRATEGY level, not rule level
        
        Args:
            rule_name: Name of the entry rule
            
        Returns:
            Strategy name (not rule name) for shared daily limit counting
        """
        # Find which strategy uses this rule
        for row in self.strategy_rows:
            if row.entry_rule == rule_name:
                # Return ONLY strategy name - all rules within strategy share same daily limit
                return row.strategy_name
        
        # Fallback to rule name if strategy not found
        return rule_name
    
    def _get_strategy_name_for_rule(self, rule_name: str) -> str:
        """Get strategy name for a given rule"""
        # Find which strategy uses this rule
        for row in self.strategy_rows:
            if row.entry_rule == rule_name:
                return row.strategy_name
        
        # Fallback to rule name if strategy not found
        return rule_name
    
    def _has_active_position(self, rule_name: str) -> bool:
        """Check if strategy already has an active position"""
        strategy_name = self._get_strategy_name_for_rule(rule_name)
        
        # Check state machine for active trades for this strategy
        for tracker_id, tracker in self.state_machine.active_trades.items():
            if tracker.strategy_row.strategy_name == strategy_name:
                self.logger.debug(f"Active position found for {strategy_name}: {tracker_id}")
                return True
        
        return False
    
    def get_daily_entry_summary(self) -> Dict[str, Any]:
        """Get summary of daily entry counts"""
        return {
            'current_date': self.current_date,
            'daily_entry_limit': self.daily_entry_limit,
            'daily_entry_counts': self.daily_entry_counts.copy(),
            'total_entries_today': sum(self.daily_entry_counts.values()),
            'trade_history_count': len(self.trade_history)
        }


class EnhancedSignalProcessor:
    """
    Enhanced signal processor for batch processing signal data
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize enhanced signal processor
        
        Args:
            config_path: Path to Excel config file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.rule_engine = EnhancedRuleEngine(config_path)
        self.processing_results: List[Dict[str, Any]] = []
    
    def process_signal_data_batch(self, signal_data_df: pd.DataFrame, 
                                 timestamp_column: str = 'timestamp') -> bool:
        """
        Process a batch of signal data
        
        Args:
            signal_data_df: DataFrame with signal data
            timestamp_column: Name of timestamp column
            
        Returns:
            True if successful, False otherwise
        """
        try:
            self.processing_results.clear()
            
            for idx, row in signal_data_df.iterrows():
                # Get timestamp
                if timestamp_column in row.index:
                    timestamp = pd.to_datetime(row[timestamp_column])
                else:
                    timestamp = datetime.now()
                
                # Convert row to IndicatorData
                indicator_data = IndicatorData.from_dict(row.to_dict(), timestamp)
                
                # Process signal
                result = self.rule_engine.process_signal_data_enhanced(indicator_data, timestamp)
                result['row_index'] = idx
                
                self.processing_results.append(result)
            
            self.logger.info(f"Processed {len(signal_data_df)} signal data rows")
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing signal data batch: {e}")
            return False
    
    def get_processing_results(self) -> List[Dict[str, Any]]:
        """Get all processing results"""
        return self.processing_results.copy()
    
    def export_results(self, output_path: str, format: str = 'csv') -> bool:
        """Export processing results to file"""
        try:
            if not self.processing_results:
                self.logger.warning("No results to export")
                return False
            
            # Flatten results for export
            export_data = []
            
            for result in self.processing_results:
                # Get strategy counters from this result
                strategy_entries_today = result.get('strategy_entries_today', {})
                strategy_exits_today = result.get('strategy_exits_today', {})
                
                base_row = {
                    'timestamp': result.get('timestamp'),
                    'errors_count': len(result.get('errors', []))
                }
                
                # Add entry details with strategy-specific counts
                for entry in result.get('entries_triggered', []):
                    strategy_name = entry['strategy_name']
                    export_row = base_row.copy()
                    export_row.update({
                        'signal_type': 'entry',
                        'strategy_name': strategy_name,
                        'strategy_entries_count': strategy_entries_today.get(strategy_name, 0),
                        'strategy_exits_count': strategy_exits_today.get(strategy_name, 0),
                        'rule_name': entry['entry_rule'],
                        'matched_conditions': ', '.join(entry['matched_conditions'])
                    })
                    export_data.append(export_row)
                
                # Add exit details with strategy-specific counts
                for exit in result.get('exits_triggered', []):
                    strategy_name = exit['strategy_name']
                    export_row = base_row.copy()
                    export_row.update({
                        'signal_type': 'exit',
                        'strategy_name': strategy_name,
                        'strategy_entries_count': strategy_entries_today.get(strategy_name, 0),
                        'strategy_exits_count': strategy_exits_today.get(strategy_name, 0),
                        'rule_name': exit['exit_rule'],
                        'matched_conditions': ', '.join(exit['matched_conditions'])
                    })
                    export_data.append(export_row)
                
                # Add row even if no signals (for completeness)
                if not result.get('entries_triggered') and not result.get('exits_triggered'):
                    base_row['signal_type'] = 'no_signal'
                    export_data.append(base_row)
            
            # Export to file
            df = pd.DataFrame(export_data)
            
            if format.lower() == 'csv':
                df.to_csv(output_path, index=False)
            elif format.lower() in ['excel', 'xlsx']:
                df.to_excel(output_path, index=False)
            else:
                raise ValueError(f"Unsupported format: {format}")
            
            self.logger.info(f"Exported {len(export_data)} result rows to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return False


# Convenience functions
def create_enhanced_rule_engine(config_path: Optional[str] = None) -> EnhancedRuleEngine:
    """Create enhanced rule engine instance"""
    return EnhancedRuleEngine(config_path)


def process_signal_data_enhanced(config_path: Optional[str] = None, 
                               signal_data_df: Optional[pd.DataFrame] = None) -> EnhancedSignalProcessor:
    """Process signal data using enhanced rule engine"""
    processor = EnhancedSignalProcessor(config_path)
    
    if signal_data_df is not None:
        processor.process_signal_data_batch(signal_data_df)
    
    return processor