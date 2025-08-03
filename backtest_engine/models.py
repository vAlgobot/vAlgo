"""
Core data models for the vAlgo Backtesting Engine.
Defines Trade, Position, and Performance tracking structures.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from enum import Enum
import pandas as pd

# Import SL/TP models
try:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))
    from utils.advanced_sl_tp_engine import SLTPResult, SLTPLevel, ZoneType
    SLTP_AVAILABLE = True
except ImportError:
    SLTP_AVAILABLE = False
    SLTPResult = None
    SLTPLevel = None
    ZoneType = None


class TradeType(Enum):
    """Trade direction enumeration"""
    BUY = "BUY"
    SELL = "SELL"


class TradeStatus(Enum):
    """Trade status enumeration"""
    OPEN = "OPEN"
    WAITING_FOR_BREAKOUT = "WAITING_FOR_BREAKOUT"
    CLOSED = "CLOSED"
    CANCELLED = "CANCELLED"


class ExitReason(Enum):
    """Exit reason enumeration"""
    SIGNAL = "SIGNAL"
    STOP_LOSS = "STOP_LOSS"
    TAKE_PROFIT = "TAKE_PROFIT"
    TAKE_PROFIT_1 = "TAKE_PROFIT_1"
    TAKE_PROFIT_2 = "TAKE_PROFIT_2"
    TAKE_PROFIT_3 = "TAKE_PROFIT_3"
    TIME_EXIT = "TIME_EXIT"
    MANUAL = "MANUAL"
    TRAILING_STOP = "TRAILING_STOP"
    NO_BREAKOUT_BREAKEVEN = "NO_BREAKOUT_BREAKEVEN"


@dataclass
class PartialExit:
    """Partial exit information for multi-target trades"""
    exit_timestamp: datetime
    exit_price: float
    quantity_closed: int
    exit_reason: ExitReason
    target_level: str  # 'TP1', 'TP2', 'TP3', 'SL'
    pnl: float = 0.0
    commission: float = 0.0
    net_pnl: float = 0.0
    
    def __post_init__(self):
        """Calculate net PnL after commission"""
        self.net_pnl = self.pnl - self.commission


@dataclass
class Trade:
    """Enhanced individual trade record with SL/TP support"""
    trade_id: str
    symbol: str
    strategy_name: str
    trade_type: TradeType
    entry_timestamp: datetime
    entry_price: float
    quantity: int
    entry_rule: str
    entry_conditions: List[str] = field(default_factory=list)
    
    # SL/TP Information
    sl_tp_result: Optional['SLTPResult'] = None
    current_stop_loss: Optional[float] = None
    current_take_profits: List[Dict[str, Any]] = field(default_factory=list)
    zone_type: Optional['ZoneType'] = None
    risk_reward_ratio: float = 0.0
    
    # Multi-target support
    remaining_quantity: int = 0
    partial_exits: List[PartialExit] = field(default_factory=list)
    is_multi_target: bool = False
    
    # Exit information (for final/complete exit)
    exit_timestamp: Optional[datetime] = None
    exit_price: Optional[float] = None
    exit_rule: Optional[str] = None
    exit_reason: Optional[ExitReason] = None
    exit_conditions: List[str] = field(default_factory=list)
    
    # Performance metrics
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None
    commission: float = 0.0
    net_pnl: Optional[float] = None
    trade_duration: Optional[float] = None  # in minutes
    
    # Enhanced performance tracking
    total_pnl: float = 0.0  # Sum of all partial exits + remaining position
    total_commission: float = 0.0
    realized_pnl: float = 0.0  # From partial exits
    unrealized_pnl: float = 0.0  # From remaining position
    
    # Status
    status: TradeStatus = TradeStatus.OPEN
    
    # Additional metadata
    timeframe: str = "1min"
    market_data: Dict[str, Any] = field(default_factory=dict)
    pivot_levels_used: List[float] = field(default_factory=list)
    
    def __post_init__(self):
        """Calculate derived fields after initialization"""
        # Set remaining quantity to initial quantity if not set
        if self.remaining_quantity == 0:
            self.remaining_quantity = self.quantity
            
        # Initialize SL/TP levels from result if available
        if self.sl_tp_result and SLTP_AVAILABLE:
            self.current_stop_loss = self.sl_tp_result.stop_loss.price
            self.zone_type = self.sl_tp_result.zone_type
            self.risk_reward_ratio = self.sl_tp_result.risk_reward_ratio
            self.pivot_levels_used = self.sl_tp_result.pivot_levels_used.copy()
            
            # Set up take profit levels
            self.current_take_profits = []
            for tp in self.sl_tp_result.take_profits:
                self.current_take_profits.append({
                    'price': tp.price,
                    'type': tp.type,
                    'percentage': tp.percentage,
                    'triggered': tp.triggered,
                    'remaining_quantity': int(self.quantity * tp.percentage / 100)
                })
            
            # Enable multi-target mode if multiple TPs
            self.is_multi_target = len(self.current_take_profits) > 1
            
        # Calculate PnL if trade is closed
        if self.exit_timestamp and self.exit_price:
            self.calculate_pnl()
            self.calculate_duration()
            if self.status == TradeStatus.OPEN:
                self.status = TradeStatus.CLOSED
    
    def calculate_pnl(self) -> None:
        """Calculate P&L for the trade"""
        if self.exit_price is None:
            return
            
        if self.trade_type == TradeType.BUY:
            self.pnl = (self.exit_price - self.entry_price) * self.quantity
        else:  # SELL
            self.pnl = (self.entry_price - self.exit_price) * self.quantity
            
        self.pnl_percentage = (self.pnl / (self.entry_price * self.quantity)) * 100
        self.net_pnl = self.pnl - self.commission
    
    def calculate_duration(self) -> None:
        """Calculate trade duration in minutes"""
        if self.exit_timestamp:
            duration = self.exit_timestamp - self.entry_timestamp
            self.trade_duration = duration.total_seconds() / 60
    
    def set_sl_tp_levels(self, sl_tp_result: 'SLTPResult') -> None:
        """Set SL/TP levels from calculation result"""
        if not SLTP_AVAILABLE:
            return
            
        self.sl_tp_result = sl_tp_result
        self.current_stop_loss = sl_tp_result.stop_loss.price
        self.zone_type = sl_tp_result.zone_type
        self.risk_reward_ratio = sl_tp_result.risk_reward_ratio
        self.pivot_levels_used = sl_tp_result.pivot_levels_used.copy()
        
        # Set up take profit levels
        self.current_take_profits = []
        for tp in sl_tp_result.take_profits:
            self.current_take_profits.append({
                'price': tp.price,
                'type': tp.type,
                'percentage': tp.percentage,
                'triggered': False,
                'remaining_quantity': int(self.quantity * tp.percentage / 100)
            })
        
        # Enable multi-target mode if multiple TPs
        self.is_multi_target = len(self.current_take_profits) > 1
    
    def check_sl_tp_hit(self, current_price: float, timestamp: datetime) -> List[Dict[str, Any]]:
        """
        Check if any SL/TP levels are hit by current price
        
        Returns:
            List of hit levels with details
        """
        hit_levels = []
        
        # Check stop loss
        if self.current_stop_loss and self.remaining_quantity > 0:
            sl_hit = False
            
            if self.trade_type == TradeType.BUY:
                sl_hit = current_price <= self.current_stop_loss
            else:  # SELL
                sl_hit = current_price >= self.current_stop_loss
            
            if sl_hit:
                hit_levels.append({
                    'type': 'SL',
                    'price': self.current_stop_loss,
                    'quantity': self.remaining_quantity,
                    'reason': ExitReason.STOP_LOSS,
                    'timestamp': timestamp
                })
        
        # Check take profits (only if SL not hit)
        if not hit_levels and self.current_take_profits:
            for tp in self.current_take_profits:
                if tp['triggered'] or tp['remaining_quantity'] <= 0:
                    continue
                
                tp_hit = False
                
                if self.trade_type == TradeType.BUY:
                    tp_hit = current_price >= tp['price']
                else:  # SELL
                    tp_hit = current_price <= tp['price']
                
                if tp_hit:
                    # Map TP type to exit reason
                    exit_reason = ExitReason.TAKE_PROFIT
                    if tp['type'] == 'TP1':
                        exit_reason = ExitReason.TAKE_PROFIT_1
                    elif tp['type'] == 'TP2':
                        exit_reason = ExitReason.TAKE_PROFIT_2
                    elif tp['type'] == 'TP3':
                        exit_reason = ExitReason.TAKE_PROFIT_3
                    
                    hit_levels.append({
                        'type': tp['type'],
                        'price': tp['price'],
                        'quantity': tp['remaining_quantity'],
                        'reason': exit_reason,
                        'timestamp': timestamp,
                        'percentage': tp['percentage']
                    })
        
        return hit_levels
    
    def execute_partial_exit(self, exit_info: Dict[str, Any], commission: float = 0.0) -> PartialExit:
        """
        Execute a partial exit at SL/TP level
        
        Args:
            exit_info: Exit information from check_sl_tp_hit
            commission: Commission for this partial exit
            
        Returns:
            PartialExit object
        """
        quantity_to_close = min(exit_info['quantity'], self.remaining_quantity)
        
        # Calculate PnL for this partial exit
        if self.trade_type == TradeType.BUY:
            pnl = (exit_info['price'] - self.entry_price) * quantity_to_close
        else:  # SELL
            pnl = (self.entry_price - exit_info['price']) * quantity_to_close
        
        # Create partial exit record
        partial_exit = PartialExit(
            exit_timestamp=exit_info['timestamp'],
            exit_price=exit_info['price'],
            quantity_closed=quantity_to_close,
            exit_reason=exit_info['reason'],
            target_level=exit_info['type'],
            pnl=pnl,
            commission=commission
        )
        
        # Update trade state
        self.partial_exits.append(partial_exit)
        self.remaining_quantity -= quantity_to_close
        self.realized_pnl += partial_exit.net_pnl
        self.total_commission += commission
        
        # Mark TP as triggered if it's a take profit
        if exit_info['type'].startswith('TP'):
            for tp in self.current_take_profits:
                if tp['type'] == exit_info['type']:
                    tp['triggered'] = True
                    break
        
        # Update total PnL
        self.update_total_pnl()
        
        # Check if trade is completely closed
        if self.remaining_quantity <= 0:
            self.status = TradeStatus.CLOSED
            self.exit_timestamp = exit_info['timestamp']
            self.exit_price = exit_info['price']
            self.exit_reason = exit_info['reason']
            self.exit_rule = f"SL/TP_{exit_info['type']}"
        
        return partial_exit
    
    def update_total_pnl(self, current_price: Optional[float] = None) -> None:
        """Update total PnL including realized and unrealized"""
        # Start with realized PnL from partial exits
        self.total_pnl = self.realized_pnl
        
        # Add unrealized PnL from remaining position
        if self.remaining_quantity > 0 and current_price:
            if self.trade_type == TradeType.BUY:
                self.unrealized_pnl = (current_price - self.entry_price) * self.remaining_quantity
            else:  # SELL
                self.unrealized_pnl = (self.entry_price - current_price) * self.remaining_quantity
            
            self.total_pnl += self.unrealized_pnl
        
        # Update legacy PnL field for compatibility
        self.pnl = self.total_pnl
        self.net_pnl = self.total_pnl - self.total_commission
    
    def update_trailing_stop(self, current_price: float, trailing_distance: float) -> bool:
        """
        Update trailing stop loss based on current price
        
        Args:
            current_price: Current market price
            trailing_distance: Distance for trailing stop
            
        Returns:
            True if stop loss was updated
        """
        if not self.current_stop_loss or self.remaining_quantity <= 0:
            return False
        
        updated = False
        
        if self.trade_type == TradeType.BUY:
            # For BUY trades, trail stop loss upward
            new_sl = current_price - trailing_distance
            if new_sl > self.current_stop_loss:
                self.current_stop_loss = new_sl
                updated = True
        else:  # SELL
            # For SELL trades, trail stop loss downward
            new_sl = current_price + trailing_distance
            if new_sl < self.current_stop_loss:
                self.current_stop_loss = new_sl
                updated = True
        
        return updated
    
    def close_trade(self, exit_timestamp: datetime, exit_price: float, 
                   exit_rule: str, exit_reason: ExitReason,
                   exit_conditions: List[str] = None) -> None:
        """Close the trade with exit information (for complete/final exit)"""
        self.exit_timestamp = exit_timestamp
        self.exit_price = exit_price
        self.exit_rule = exit_rule
        self.exit_reason = exit_reason
        self.exit_conditions = exit_conditions or []
        self.status = TradeStatus.CLOSED
        
        # If there's remaining quantity, close it
        if self.remaining_quantity > 0:
            if self.trade_type == TradeType.BUY:
                final_pnl = (exit_price - self.entry_price) * self.remaining_quantity
            else:
                final_pnl = (self.entry_price - exit_price) * self.remaining_quantity
            
            self.realized_pnl += final_pnl
            self.remaining_quantity = 0
            self.unrealized_pnl = 0.0
        
        self.calculate_pnl()
        self.calculate_duration()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert trade to dictionary for reporting"""
        base_dict = {
            'trade_id': self.trade_id,
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'trade_type': self.trade_type.value,
            'entry_timestamp': self.entry_timestamp,
            'entry_price': self.entry_price,
            'quantity': self.quantity,
            'entry_rule': self.entry_rule,
            'entry_conditions': ', '.join(self.entry_conditions),
            'exit_timestamp': self.exit_timestamp,
            'exit_price': self.exit_price,
            'exit_rule': self.exit_rule,
            'exit_reason': self.exit_reason.value if self.exit_reason else None,
            'exit_conditions': ', '.join(self.exit_conditions),
            'pnl': self.pnl,
            'pnl_percentage': self.pnl_percentage,
            'commission': self.commission,
            'net_pnl': self.net_pnl,
            'trade_duration': self.trade_duration,
            'status': self.status.value,
            'timeframe': self.timeframe,
            
            # Enhanced SL/TP fields
            'stop_loss': self.current_stop_loss,
            'zone_type': self.zone_type.value if self.zone_type else None,
            'risk_reward_ratio': self.risk_reward_ratio,
            'is_multi_target': self.is_multi_target,
            'remaining_quantity': self.remaining_quantity,
            'total_pnl': self.total_pnl,
            'realized_pnl': self.realized_pnl,
            'unrealized_pnl': self.unrealized_pnl,
            'total_commission': self.total_commission,
            'partial_exits_count': len(self.partial_exits),
            'take_profits': [
                f"{tp['type']}:{tp['price']}" for tp in self.current_take_profits
            ] if self.current_take_profits else [],
            'pivot_levels_used': self.pivot_levels_used
        }
        
        # Add partial exits details if any
        if self.partial_exits:
            base_dict['partial_exits'] = [
                {
                    'timestamp': pe.exit_timestamp,
                    'price': pe.exit_price,
                    'quantity': pe.quantity_closed,
                    'target': pe.target_level,
                    'pnl': pe.net_pnl,
                    'reason': pe.exit_reason.value
                }
                for pe in self.partial_exits
            ]
        
        return base_dict


@dataclass
class Position:
    """Position tracking for a strategy-symbol combination"""
    symbol: str
    strategy_name: str
    current_quantity: int = 0
    average_price: float = 0.0
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    total_commission: float = 0.0
    open_trades: List[Trade] = field(default_factory=list)
    closed_trades: List[Trade] = field(default_factory=list)
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the position"""
        if trade.status == TradeStatus.OPEN:
            self.open_trades.append(trade)
            self.update_position_from_trade(trade)
        else:
            self.closed_trades.append(trade)
            if trade.net_pnl:
                self.realized_pnl += trade.net_pnl
        
        self.total_commission += trade.commission
    
    def update_position_from_trade(self, trade: Trade) -> None:
        """Update position metrics from a new trade"""
        if trade.trade_type == TradeType.BUY:
            new_quantity = self.current_quantity + trade.quantity
            if new_quantity != 0:
                self.average_price = (
                    (self.average_price * self.current_quantity + 
                     trade.entry_price * trade.quantity) / new_quantity
                )
            self.current_quantity = new_quantity
        else:  # SELL
            self.current_quantity -= trade.quantity
            if self.current_quantity < 0:
                # Position flipped, update average price
                self.average_price = trade.entry_price
    
    def close_trade(self, trade: Trade) -> None:
        """Move trade from open to closed"""
        if trade in self.open_trades:
            self.open_trades.remove(trade)
            self.closed_trades.append(trade)
            if trade.net_pnl:
                self.realized_pnl += trade.net_pnl
    
    def calculate_unrealized_pnl(self, current_price: float) -> None:
        """Calculate unrealized P&L based on current price"""
        if self.current_quantity == 0:
            self.unrealized_pnl = 0.0
            return
        
        if self.current_quantity > 0:  # Long position
            self.unrealized_pnl = (current_price - self.average_price) * self.current_quantity
        else:  # Short position
            self.unrealized_pnl = (self.average_price - current_price) * abs(self.current_quantity)
    
    def get_total_pnl(self) -> float:
        """Get total P&L (realized + unrealized)"""
        return self.realized_pnl + self.unrealized_pnl
    
    def is_flat(self) -> bool:
        """Check if position is flat (no open quantity)"""
        return self.current_quantity == 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary for reporting"""
        return {
            'symbol': self.symbol,
            'strategy_name': self.strategy_name,
            'current_quantity': self.current_quantity,
            'average_price': self.average_price,
            'unrealized_pnl': self.unrealized_pnl,
            'realized_pnl': self.realized_pnl,
            'total_pnl': self.get_total_pnl(),
            'total_commission': self.total_commission,
            'open_trades_count': len(self.open_trades),
            'closed_trades_count': len(self.closed_trades),
            'is_flat': self.is_flat()
        }


@dataclass
class PerformanceMetrics:
    """Performance metrics for backtesting results"""
    strategy_name: str
    symbol: str = "ALL"
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None
    
    # Trade statistics
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    break_even_trades: int = 0
    
    # P&L statistics
    total_pnl: float = 0.0
    gross_profit: float = 0.0
    gross_loss: float = 0.0
    net_profit: float = 0.0
    
    # Performance ratios
    win_rate: float = 0.0
    average_win: float = 0.0
    average_loss: float = 0.0
    profit_factor: float = 0.0
    
    # Drawdown statistics
    max_drawdown: float = 0.0
    max_drawdown_percentage: float = 0.0
    
    # Additional metrics
    largest_win: float = 0.0
    largest_loss: float = 0.0
    average_trade_duration: float = 0.0
    total_commission: float = 0.0
    
    # Capital metrics
    initial_capital: float = 100000.0
    final_capital: float = 100000.0
    total_return: float = 0.0
    total_return_percentage: float = 0.0
    
    def calculate_metrics(self, trades: List[Trade]) -> None:
        """Calculate all performance metrics from trade list"""
        if not trades:
            return
        
        self.total_trades = len(trades)
        
        # Filter closed trades for calculations
        closed_trades = [t for t in trades if t.status == TradeStatus.CLOSED and t.net_pnl is not None]
        
        if not closed_trades:
            return
        
        # Calculate basic statistics
        pnl_values = [t.net_pnl for t in closed_trades]
        self.total_pnl = sum(pnl_values)
        self.net_profit = self.total_pnl
        self.total_commission = sum(t.commission for t in closed_trades)
        
        # Winning/losing trades
        winning_pnl = [pnl for pnl in pnl_values if pnl > 0]
        losing_pnl = [pnl for pnl in pnl_values if pnl < 0]
        breakeven_pnl = [pnl for pnl in pnl_values if pnl == 0]
        
        self.winning_trades = len(winning_pnl)
        self.losing_trades = len(losing_pnl)
        self.break_even_trades = len(breakeven_pnl)
        
        # Calculate ratios
        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
        
        if winning_pnl:
            self.average_win = sum(winning_pnl) / len(winning_pnl)
            self.gross_profit = sum(winning_pnl)
            self.largest_win = max(winning_pnl)
        
        if losing_pnl:
            self.average_loss = sum(losing_pnl) / len(losing_pnl)
            self.gross_loss = abs(sum(losing_pnl))
            self.largest_loss = min(losing_pnl)
        
        # Profit factor
        if self.gross_loss > 0:
            self.profit_factor = self.gross_profit / self.gross_loss
        
        # Trade duration
        durations = [t.trade_duration for t in closed_trades if t.trade_duration is not None]
        if durations:
            self.average_trade_duration = sum(durations) / len(durations)
        
        # Capital calculations
        self.final_capital = self.initial_capital + self.total_pnl
        self.total_return = self.total_pnl
        if self.initial_capital > 0:
            self.total_return_percentage = (self.total_return / self.initial_capital) * 100
        
        # Calculate drawdown
        self.calculate_drawdown(closed_trades)
    
    def calculate_drawdown(self, trades: List[Trade]) -> None:
        """Calculate maximum drawdown from trade sequence"""
        if not trades:
            return
        
        # Sort trades by exit timestamp
        sorted_trades = sorted(trades, key=lambda x: x.exit_timestamp or datetime.min)
        
        running_pnl = 0.0
        peak_pnl = 0.0
        max_dd = 0.0
        
        for trade in sorted_trades:
            if trade.net_pnl is not None:
                running_pnl += trade.net_pnl
                peak_pnl = max(peak_pnl, running_pnl)
                
                current_dd = peak_pnl - running_pnl
                max_dd = max(max_dd, current_dd)
        
        self.max_drawdown = max_dd
        if peak_pnl > 0:
            self.max_drawdown_percentage = (max_dd / peak_pnl) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to dictionary for reporting"""
        return {
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'total_trades': self.total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'break_even_trades': self.break_even_trades,
            'win_rate': round(self.win_rate, 2),
            'total_pnl': round(self.total_pnl, 2),
            'gross_profit': round(self.gross_profit, 2),
            'gross_loss': round(self.gross_loss, 2),
            'net_profit': round(self.net_profit, 2),
            'average_win': round(self.average_win, 2),
            'average_loss': round(self.average_loss, 2),
            'profit_factor': round(self.profit_factor, 2),
            'max_drawdown': round(self.max_drawdown, 2),
            'max_drawdown_percentage': round(self.max_drawdown_percentage, 2),
            'largest_win': round(self.largest_win, 2),
            'largest_loss': round(self.largest_loss, 2),
            'average_trade_duration': round(self.average_trade_duration, 2),
            'total_commission': round(self.total_commission, 2),
            'initial_capital': round(self.initial_capital, 2),
            'final_capital': round(self.final_capital, 2),
            'total_return': round(self.total_return, 2),
            'total_return_percentage': round(self.total_return_percentage, 2)
        }


@dataclass
class BacktestResult:
    """Complete backtest result container"""
    strategy_name: str
    symbols: List[str]
    start_date: datetime
    end_date: datetime
    initial_capital: float
    
    # Trade data
    all_trades: List[Trade] = field(default_factory=list)
    positions: Dict[str, Position] = field(default_factory=dict)
    
    # Performance metrics
    overall_performance: Optional[PerformanceMetrics] = None
    symbol_performance: Dict[str, PerformanceMetrics] = field(default_factory=dict)
    
    # Execution metadata
    execution_time: float = 0.0
    data_points_processed: int = 0
    
    def add_trade(self, trade: Trade) -> None:
        """Add a trade to the backtest result"""
        self.all_trades.append(trade)
        
        # Update position
        position_key = f"{trade.symbol}_{trade.strategy_name}"
        if position_key not in self.positions:
            self.positions[position_key] = Position(
                symbol=trade.symbol,
                strategy_name=trade.strategy_name
            )
        
        self.positions[position_key].add_trade(trade)
    
    def calculate_performance(self) -> None:
        """Calculate overall and per-symbol performance metrics"""
        # Overall performance
        self.overall_performance = PerformanceMetrics(
            strategy_name=self.strategy_name,
            symbol="ALL",
            start_date=self.start_date,
            end_date=self.end_date,
            initial_capital=self.initial_capital
        )
        self.overall_performance.calculate_metrics(self.all_trades)
        
        # Per-symbol performance
        for symbol in self.symbols:
            symbol_trades = [t for t in self.all_trades if t.symbol == symbol]
            if symbol_trades:
                symbol_perf = PerformanceMetrics(
                    strategy_name=self.strategy_name,
                    symbol=symbol,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    initial_capital=self.initial_capital / len(self.symbols)  # Distribute capital
                )
                symbol_perf.calculate_metrics(symbol_trades)
                self.symbol_performance[symbol] = symbol_perf
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert backtest result to dictionary"""
        return {
            'strategy_name': self.strategy_name,
            'symbols': self.symbols,
            'start_date': self.start_date,
            'end_date': self.end_date,
            'initial_capital': self.initial_capital,
            'total_trades': len(self.all_trades),
            'execution_time': self.execution_time,
            'data_points_processed': self.data_points_processed,
            'overall_performance': self.overall_performance.to_dict() if self.overall_performance else {},
            'symbol_performance': {k: v.to_dict() for k, v in self.symbol_performance.items()},
            'positions': {k: v.to_dict() for k, v in self.positions.items()}
        }