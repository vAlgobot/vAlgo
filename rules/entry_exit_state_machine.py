"""
Entry-Exit State Machine for vAlgo Rule Engine

Tracks entry rule triggers and manages corresponding exit rule evaluations.
Maintains 1:1 entry-exit pairing as defined in strategy_config rows.
"""

import logging
from typing import Dict, List, Any, Optional, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum

from .models import RuleResult, IndicatorData
from .enhanced_condition_loader import StrategyRow


class TradeState(Enum):
    """Trade states for tracking entry-exit lifecycle"""
    WAITING_ENTRY = "waiting_entry"
    ENTRY_TRIGGERED = "entry_triggered"
    EXIT_TRIGGERED = "exit_triggered"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass
class TradeTracker:
    """Tracks a single entry-exit pair from strategy_config row"""
    strategy_row: StrategyRow
    entry_rule: str
    exit_rule: str
    state: TradeState = TradeState.WAITING_ENTRY
    entry_timestamp: Optional[datetime] = None
    exit_timestamp: Optional[datetime] = None
    entry_result: Optional[RuleResult] = None
    exit_result: Optional[RuleResult] = None
    entry_signal_data: Optional[Dict[str, Any]] = None
    exit_signal_data: Optional[Dict[str, Any]] = None
    error_message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for output"""
        return {
            'strategy_name': self.strategy_row.strategy_name,
            'entry_rule': self.entry_rule,
            'exit_rule': self.exit_rule,
            'state': self.state.value,
            'entry_timestamp': self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'entry_triggered': self.entry_result.triggered if self.entry_result else False,
            'exit_triggered': self.exit_result.triggered if self.exit_result else False,
            'entry_matched_conditions': self.entry_result.matched_conditions if self.entry_result else [],
            'exit_matched_conditions': self.exit_result.matched_conditions if self.exit_result else [],
            'position_size': self.strategy_row.position_size,
            'error_message': self.error_message
        }
    
    def get_trade_summary(self) -> Dict[str, Any]:
        """Get completed trade summary for history tracking"""
        return {
            'strategy_name': self.strategy_row.strategy_name,
            'entry_rule': self.entry_rule,
            'exit_rule': self.exit_rule,
            'entry_timestamp': self.entry_timestamp.isoformat() if self.entry_timestamp else None,
            'exit_timestamp': self.exit_timestamp.isoformat() if self.exit_timestamp else None,
            'entry_matched_conditions': self.entry_result.matched_conditions if self.entry_result else [],
            'exit_matched_conditions': self.exit_result.matched_conditions if self.exit_result else [],
            'position_size': self.strategy_row.position_size,
            'trade_duration_minutes': self._calculate_trade_duration_minutes()
        }
    
    def _calculate_trade_duration_minutes(self) -> Optional[float]:
        """Calculate trade duration in minutes"""
        if self.entry_timestamp and self.exit_timestamp:
            duration = self.exit_timestamp - self.entry_timestamp
            return duration.total_seconds() / 60.0
        return None
    
    def reset_for_new_cycle(self) -> None:
        """Reset tracker for a new entry-exit cycle while preserving strategy config"""
        # Store previous trade info before reset (if needed for advanced tracking)
        previous_entry = self.entry_timestamp
        previous_exit = self.exit_timestamp
        
        # Reset trade-specific data
        self.state = TradeState.WAITING_ENTRY
        self.entry_timestamp = None
        self.exit_timestamp = None
        self.entry_result = None
        self.exit_result = None
        self.entry_signal_data = None
        self.exit_signal_data = None
        self.error_message = ""
        
        # Keep strategy_row, entry_rule, exit_rule (these define the tracker)


@dataclass
class StateSnapshot:
    """Snapshot of state machine at a point in time"""
    timestamp: datetime
    active_trades: List[TradeTracker] = field(default_factory=list)
    completed_trades: List[TradeTracker] = field(default_factory=list)
    waiting_entries: List[TradeTracker] = field(default_factory=list)
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary statistics"""
        return {
            'timestamp': self.timestamp.isoformat(),
            'active_trades_count': len(self.active_trades),
            'completed_trades_count': len(self.completed_trades),
            'waiting_entries_count': len(self.waiting_entries),
            'total_trackers': len(self.active_trades) + len(self.completed_trades) + len(self.waiting_entries)
        }


class EntryExitStateMachine:
    """
    State machine for tracking entry-exit rule pairs
    
    Features:
    - 1:1 entry-exit pairing from strategy_config rows
    - State tracking for each trade lifecycle
    - Historical tracking and reporting
    - Concurrent trade support (multiple strategies)
    """
    
    def __init__(self):
        """Initialize the state machine"""
        self.logger = logging.getLogger(__name__)
        
        # Trade trackers organized by state
        self.waiting_entries: Dict[str, TradeTracker] = {}  # key: strategy_row_id
        self.active_trades: Dict[str, TradeTracker] = {}    # key: strategy_row_id
        self.completed_trades: List[TradeTracker] = []
        
        # Trade history for completed cycles (for reporting and analytics)
        self.trade_history: List[Dict[str, Any]] = []
        
        # Historical snapshots
        self.state_history: List[StateSnapshot] = []
        
        # Configuration
        self.max_history_size = 1000
    
    def initialize_from_strategy_rows(self, strategy_rows: List[StrategyRow]) -> None:
        """
        Initialize state machine with strategy rows
        
        Args:
            strategy_rows: List of StrategyRow objects from config
        """
        self.logger.info(f"Initializing state machine with {len(strategy_rows)} strategy rows")
        
        for i, row in enumerate(strategy_rows):
            tracker_id = f"{row.strategy_name}_{row.entry_rule}_{row.exit_rule}_{i}"
            
            tracker = TradeTracker(
                strategy_row=row,
                entry_rule=row.entry_rule,
                exit_rule=row.exit_rule,
                state=TradeState.WAITING_ENTRY
            )
            
            self.waiting_entries[tracker_id] = tracker
        
        self.logger.info(f"Initialized {len(self.waiting_entries)} trade trackers")
    
    def process_signal_data(self, indicator_data: IndicatorData, timestamp: datetime,
                           entry_evaluator, exit_evaluator) -> Dict[str, Any]:
        """
        Process signal data against all tracked entry-exit pairs
        
        Args:
            indicator_data: Current indicator values
            timestamp: Current timestamp
            entry_evaluator: Function to evaluate entry rules
            exit_evaluator: Function to evaluate exit rules
            
        Returns:
            Dictionary with processing results
        """
        results = {
            'timestamp': timestamp,
            'entries_triggered': [],
            'exits_triggered': [],
            'errors': []
        }
        
        try:
            self.logger.debug(f"Processing signal data: {len(self.waiting_entries)} waiting, {len(self.active_trades)} active")
            
            # Process waiting entries
            self._process_waiting_entries(indicator_data, timestamp, entry_evaluator, results)
            
            # Process active trades (check for exits)
            self._process_active_trades(indicator_data, timestamp, exit_evaluator, results)
            
            # Take snapshot
            self._take_snapshot(timestamp)
            
        except Exception as e:
            self.logger.error(f"Error processing signal data: {e}")
            results['errors'].append(f"Processing error: {e}")
        
        return results
    
    def _process_waiting_entries(self, indicator_data: IndicatorData, timestamp: datetime,
                                entry_evaluator, results: Dict[str, Any]) -> None:
        """Process waiting entry signals"""
        entries_to_activate = []
        
        for tracker_id, tracker in self.waiting_entries.items():
            try:
                # Evaluate entry rule
                self.logger.debug(f"Evaluating entry rule '{tracker.entry_rule}' for strategy '{tracker.strategy_row.strategy_name}'")
                entry_result = entry_evaluator(tracker.entry_rule, indicator_data)
                tracker.entry_result = entry_result
                
                if entry_result.triggered:
                    # Entry signal triggered
                    tracker.state = TradeState.ENTRY_TRIGGERED
                    tracker.entry_timestamp = timestamp
                    tracker.entry_signal_data = indicator_data.data.copy()
                    
                    entries_to_activate.append(tracker_id)
                    results['entries_triggered'].append({
                        'strategy_name': tracker.strategy_row.strategy_name,
                        'entry_rule': tracker.entry_rule,
                        'exit_rule': tracker.exit_rule,
                        'timestamp': timestamp.isoformat(),
                        'matched_conditions': entry_result.matched_conditions
                    })
                    
                    self.logger.info(f"Entry triggered: {tracker.strategy_row.strategy_name} - {tracker.entry_rule}")
                
            except Exception as e:
                tracker.state = TradeState.ERROR
                tracker.error_message = f"Entry evaluation error: {e}"
                results['errors'].append(f"Entry error for {tracker_id}: {e}")
        
        # Move triggered entries to active trades
        for tracker_id in entries_to_activate:
            tracker = self.waiting_entries.pop(tracker_id)
            self.active_trades[tracker_id] = tracker
    
    def _process_active_trades(self, indicator_data: IndicatorData, timestamp: datetime,
                              exit_evaluator, results: Dict[str, Any]) -> None:
        """Process active trades for exit signals and time-based exits"""
        exits_to_complete = []
        signal_exits = []
        
        # PHASE 1 FIX: Check signal-based exits FIRST (before time-based exits)
        for tracker_id, tracker in self.active_trades.items():
            try:
                # Evaluate exit rule with detailed logging
                self.logger.debug(f"[EXIT_EVAL] Evaluating exit rule '{tracker.exit_rule}' for {tracker.strategy_row.strategy_name} at {timestamp}")
                exit_result = exit_evaluator(tracker.exit_rule, indicator_data)
                tracker.exit_result = exit_result
                
                if exit_result.triggered:
                    # Signal-based exit triggered
                    tracker.state = TradeState.EXIT_TRIGGERED
                    tracker.exit_timestamp = timestamp
                    tracker.exit_signal_data = indicator_data.data.copy()
                    
                    signal_exits.append(tracker_id)
                    exits_to_complete.append(tracker_id)
                    results['exits_triggered'].append({
                        'strategy_name': tracker.strategy_row.strategy_name,
                        'entry_rule': tracker.entry_rule,
                        'exit_rule': tracker.exit_rule,
                        'exit_type': 'Signal_exit',
                        'entry_timestamp': tracker.entry_timestamp.isoformat() if tracker.entry_timestamp else None,
                        'exit_timestamp': timestamp.isoformat(),
                        'matched_conditions': exit_result.matched_conditions
                    })
                    
                    self.logger.info(f"✅ Signal exit triggered: {tracker.strategy_row.strategy_name} - {tracker.exit_rule}")
                else:
                    # Log why exit didn't trigger for debugging
                    self.logger.debug(f"[EXIT_EVAL] Exit rule '{tracker.exit_rule}' not triggered - Conditions: {exit_result.matched_conditions}")
                
            except Exception as e:
                tracker.state = TradeState.ERROR
                tracker.error_message = f"Exit evaluation error: {e}"
                results['errors'].append(f"Exit error for {tracker_id}: {e}")
        
        # PHASE 1 FIX: Check time-based exits ONLY for trades that didn't get signal-based exits
        if len(signal_exits) == 0:  # Only if no signal exits occurred
            time_exits = self._check_time_based_exits(timestamp, results)
            exits_to_complete.extend(time_exits)
        else:
            self.logger.info(f"[EXIT_PRIORITY] {len(signal_exits)} signal-based exits found - skipping time-based exit check")
        
        # Handle completed exits - reset for continuous cycles
        for tracker_id in exits_to_complete:
            tracker = self.active_trades.pop(tracker_id)
            
            # Store completed trade in history for reporting
            trade_summary = tracker.get_trade_summary()
            self.trade_history.append(trade_summary)
            
            self.logger.debug(f"Trade completed: {tracker.strategy_row.strategy_name} "
                            f"({trade_summary.get('trade_duration_minutes', 0):.1f} minutes)")
            
            # Reset tracker for new cycle and move back to waiting entries
            tracker.reset_for_new_cycle()
            self.waiting_entries[tracker_id] = tracker
            
            self.logger.debug(f"Tracker reset for new cycle: {tracker_id}")
            
            # Also store in completed_trades for state machine snapshots (optional)
            # We keep this for backward compatibility with existing reporting
            completed_tracker = TradeTracker(
                strategy_row=tracker.strategy_row,
                entry_rule=tracker.entry_rule,
                exit_rule=tracker.exit_rule,
                state=TradeState.COMPLETED
            )
            self.completed_trades.append(completed_tracker)
    
    def _check_time_based_exits(self, timestamp: datetime, results: Dict[str, Any]) -> List[str]:
        """Check for time-based exits at market close (15:00)"""
        exits_to_complete = []
        
        # Market close time: 15:00 (3:00 PM) IST
        market_close_time = timestamp.replace(hour=15, minute=0, second=0, microsecond=0)
        
        # Force exit all active positions at or after market close
        if timestamp >= market_close_time and self.active_trades:
            self.logger.info(f"Market close reached at {timestamp.strftime('%H:%M:%S')} - forcing time-based exits for {len(self.active_trades)} active trades")
            
            for tracker_id, tracker in self.active_trades.items():
                # Force time-based exit
                tracker.state = TradeState.EXIT_TRIGGERED
                tracker.exit_timestamp = timestamp
                tracker.exit_signal_data = {'time_exit': 'market_close', 'close_time': '15:00'}
                
                exits_to_complete.append(tracker_id)
                results['exits_triggered'].append({
                    'strategy_name': tracker.strategy_row.strategy_name,
                    'entry_rule': tracker.entry_rule,
                    'exit_rule': 'TIME_EXIT_15:00',
                    'exit_type': 'Time_exit',
                    'entry_timestamp': tracker.entry_timestamp.isoformat() if tracker.entry_timestamp else None,
                    'exit_timestamp': timestamp.isoformat(),
                    'matched_conditions': ['Market close at 15:00 IST']
                })
                
                self.logger.info(f"Time exit triggered: {tracker.strategy_row.strategy_name} - Market close at {timestamp.strftime('%H:%M:%S')}")
        
        return exits_to_complete
    
    def _take_snapshot(self, timestamp: datetime) -> None:
        """Take a snapshot of current state"""
        snapshot = StateSnapshot(
            timestamp=timestamp,
            active_trades=list(self.active_trades.values()),
            completed_trades=self.completed_trades.copy(),
            waiting_entries=list(self.waiting_entries.values())
        )
        
        self.state_history.append(snapshot)
        
        # Limit history size
        if len(self.state_history) > self.max_history_size:
            self.state_history = self.state_history[-self.max_history_size:]
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current state summary"""
        return {
            'waiting_entries_count': len(self.waiting_entries),
            'active_trades_count': len(self.active_trades),
            'completed_trades_count': len(self.completed_trades),
            'total_trackers': len(self.waiting_entries) + len(self.active_trades) + len(self.completed_trades),
            'strategies_summary': self._get_strategies_summary()
        }
    
    def _get_strategies_summary(self) -> Dict[str, Any]:
        """Get summary by strategy"""
        strategy_stats = {}
        
        # Count by strategy
        for tracker in list(self.waiting_entries.values()) + list(self.active_trades.values()) + self.completed_trades:
            strategy_name = tracker.strategy_row.strategy_name
            if strategy_name not in strategy_stats:
                strategy_stats[strategy_name] = {
                    'waiting_entries': 0,
                    'active_trades': 0,
                    'completed_trades': 0
                }
            
            if tracker.state == TradeState.WAITING_ENTRY:
                strategy_stats[strategy_name]['waiting_entries'] += 1
            elif tracker.state == TradeState.ENTRY_TRIGGERED:
                strategy_stats[strategy_name]['active_trades'] += 1
            elif tracker.state == TradeState.COMPLETED:
                strategy_stats[strategy_name]['completed_trades'] += 1
        
        return strategy_stats
    
    def get_completed_trades_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all completed trades"""
        return [tracker.to_dict() for tracker in self.completed_trades]
    
    def get_active_trades_summary(self) -> List[Dict[str, Any]]:
        """Get summary of all active trades"""
        return [tracker.to_dict() for tracker in self.active_trades.values()]
    
    def reset_state(self) -> None:
        """Reset state machine to initial state"""
        self.logger.info("Resetting state machine")
        
        self.waiting_entries.clear()
        self.active_trades.clear()
        self.completed_trades.clear()
        self.state_history.clear()
    
    def get_trade_lifecycle_report(self) -> Dict[str, Any]:
        """Get comprehensive trade lifecycle report"""
        total_trackers = len(self.waiting_entries) + len(self.active_trades) + len(self.completed_trades)
        
        # Calculate success rates
        entries_triggered = len(self.active_trades) + len(self.completed_trades)
        exits_triggered = len(self.completed_trades)
        
        entry_rate = entries_triggered / total_trackers if total_trackers > 0 else 0
        exit_rate = exits_triggered / entries_triggered if entries_triggered > 0 else 0
        completion_rate = exits_triggered / total_trackers if total_trackers > 0 else 0
        
        # Get strategy breakdown
        strategy_breakdown = {}
        all_trackers = list(self.waiting_entries.values()) + list(self.active_trades.values()) + self.completed_trades
        
        for tracker in all_trackers:
            strategy = tracker.strategy_row.strategy_name
            if strategy not in strategy_breakdown:
                strategy_breakdown[strategy] = {
                    'total': 0, 'entries': 0, 'exits': 0, 'completed': 0
                }
            
            strategy_breakdown[strategy]['total'] += 1
            
            if tracker.state in [TradeState.ENTRY_TRIGGERED, TradeState.COMPLETED]:
                strategy_breakdown[strategy]['entries'] += 1
            
            if tracker.state == TradeState.COMPLETED:
                strategy_breakdown[strategy]['exits'] += 1
                strategy_breakdown[strategy]['completed'] += 1
        
        return {
            'total_trackers': total_trackers,
            'waiting_entries': len(self.waiting_entries),
            'active_trades': len(self.active_trades),
            'completed_trades': len(self.completed_trades),
            'entry_trigger_rate': entry_rate,
            'exit_trigger_rate': exit_rate,
            'completion_rate': completion_rate,
            'strategy_breakdown': strategy_breakdown,
            'state_history_count': len(self.state_history)
        }


# Convenience functions
def create_state_machine_from_config(strategy_rows: List[StrategyRow]) -> EntryExitStateMachine:
    """
    Create and initialize state machine from strategy configuration
    
    Args:
        strategy_rows: List of StrategyRow objects
        
    Returns:
        Initialized EntryExitStateMachine
    """
    state_machine = EntryExitStateMachine()
    state_machine.initialize_from_strategy_rows(strategy_rows)
    return state_machine