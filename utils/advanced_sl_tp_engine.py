#!/usr/bin/env python3
"""
Advanced Stop Loss / Take Profit Engine
======================================

Production-grade SL/TP management system for vAlgo Trading System.
Implements intelligent zone detection, dynamic risk management, and multi-target systems.

Features:
- Config-driven SL/TP calculation
- Zone-aware (Support/Resistance) algorithms
- Multiple target system with partial exits
- ATR-based dynamic adjustments
- Fibonacci integration
- Real-time monitoring and updates

Author: vAlgo Development Team
Created: July 10, 2025
Version: 1.0.0 (Production)
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Union, NamedTuple
from datetime import datetime
from enum import Enum
from dataclasses import dataclass, field
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.config_cache import get_cached_config
    from utils.logger import get_logger
    CONFIG_AVAILABLE = True
except ImportError:
    CONFIG_AVAILABLE = False


class ZoneType(Enum):
    """Zone type enumeration for SL/TP calculation"""
    SUPPORT = "PUT"
    RESISTANCE = "CALL"
    NEUTRAL = "NEUTRAL"


class SLMethod(Enum):
    """Stop Loss calculation methods"""
    ATR = "ATR"
    BREAKOUT = "BREAKOUT"
    BREAKOUT_CANDLE = "BREAKOUT_CANDLE"
    PIVOT = "PIVOT"
    FIXED = "FIXED"


class TPMethod(Enum):
    """Take Profit calculation methods"""
    PIVOT = "PIVOT"
    FIBONACCI = "FIBONACCI"
    HYBRID = "HYBRID"
    RATIO = "RATIO"
    BREAKOUT_CANDLE = "BREAKOUT_CANDLE"


@dataclass
class SLTPLevel:
    """Individual SL/TP level definition"""
    price: float
    type: str  # 'SL', 'TP1', 'TP2', 'TP3'
    percentage: float = 100.0  # Percentage of position to close
    triggered: bool = False
    zone_strength: float = 1.0
    calculation_method: str = ""
    
    def __post_init__(self):
        """Validate level data"""
        if self.percentage <= 0 or self.percentage > 100:
            raise ValueError(f"Invalid percentage: {self.percentage}")
        if self.price <= 0:
            raise ValueError(f"Invalid price: {self.price}")


@dataclass
class SLTPResult:
    """Complete SL/TP calculation result"""
    symbol: str
    strategy_name: str
    entry_price: float
    zone_type: ZoneType
    
    # Stop Loss levels
    stop_loss: SLTPLevel
    
    # Take Profit levels (multiple targets)
    take_profits: List[SLTPLevel] = field(default_factory=list)
    
    # Risk metrics
    risk_amount: float = 0.0
    risk_percentage: float = 0.0
    risk_reward_ratio: float = 0.0
    
    # Calculation metadata
    calculation_timestamp: datetime = field(default_factory=datetime.now)
    pivot_levels_used: List[float] = field(default_factory=list)
    atr_value: Optional[float] = None
    zone_strength: float = 1.0
    
    def get_total_risk_reward(self) -> float:
        """Calculate total risk-reward ratio across all targets"""
        if not self.take_profits or self.risk_amount == 0:
            return 0.0
            
        total_reward = sum(
            (tp.price - self.entry_price) * (tp.percentage / 100) 
            for tp in self.take_profits
        )
        return abs(total_reward / self.risk_amount) if self.risk_amount != 0 else 0.0


class ZoneDetector:
    """Intelligent zone detection for Support/Resistance identification"""
    
    def __init__(self, config_cache=None):
        self.config_cache = config_cache
        self.logger = get_logger(__name__) if CONFIG_AVAILABLE else logging.getLogger(__name__)
    
    def detect_zone_type(self, entry_price: float, pivot_levels: List[float], 
                        recent_high: float, recent_low: float) -> Tuple[ZoneType, float]:
        """
        Detect zone type based on entry price and pivot levels
        
        Args:
            entry_price: Entry price level
            pivot_levels: List of pivot points (S1-S4, R1-R4, etc.)
            recent_high: Recent high for context
            recent_low: Recent low for context
            
        Returns:
            Tuple of (ZoneType, zone_strength)
        """
        if not pivot_levels:
            return ZoneType.NEUTRAL, 1.0
        
        # Sort pivot levels
        sorted_pivots = sorted(pivot_levels)
        
        # Find nearest pivot levels
        lower_pivots = [p for p in sorted_pivots if p < entry_price]
        upper_pivots = [p for p in sorted_pivots if p > entry_price]
        
        # Calculate zone strength based on proximity to pivots
        zone_strength = self._calculate_zone_strength(
            entry_price, lower_pivots, upper_pivots, recent_high, recent_low
        )
        
        # Determine zone type
        if entry_price > np.median(sorted_pivots):
            # Above median pivots - likely resistance breakout
            return ZoneType.RESISTANCE, zone_strength
        elif entry_price < np.median(sorted_pivots):
            # Below median pivots - likely support breakout  
            return ZoneType.SUPPORT, zone_strength
        else:
            return ZoneType.NEUTRAL, zone_strength
    
    def _calculate_zone_strength(self, entry_price: float, lower_pivots: List[float],
                               upper_pivots: List[float], recent_high: float, 
                               recent_low: float) -> float:
        """Calculate zone strength based on pivot proximity and recent price action"""
        
        base_strength = 1.0
        
        # Strength from pivot proximity
        if lower_pivots:
            nearest_lower = max(lower_pivots)
            lower_distance = abs(entry_price - nearest_lower) / entry_price
            base_strength += max(0, 1 - lower_distance * 10)  # Closer = stronger
        
        if upper_pivots:
            nearest_upper = min(upper_pivots)
            upper_distance = abs(nearest_upper - entry_price) / entry_price
            base_strength += max(0, 1 - upper_distance * 10)  # Closer = stronger
        
        # Strength from recent price action context
        range_position = (entry_price - recent_low) / (recent_high - recent_low) if recent_high != recent_low else 0.5
        
        # Extreme positions (near highs/lows) are stronger zones
        if range_position > 0.8 or range_position < 0.2:
            base_strength += 0.5
        
        return min(base_strength, 3.0)  # Cap at 3.0


class SLCalculator:
    """Advanced Stop Loss calculation algorithms"""
    
    def __init__(self, config_cache=None):
        self.config_cache = config_cache
        self.logger = get_logger(__name__) if CONFIG_AVAILABLE else logging.getLogger(__name__)
    
    def calculate_stop_loss(self, entry_price: float, zone_type: ZoneType,
                          method: SLMethod,
                          market_data: pd.DataFrame, config: Dict[str, Any]) -> SLTPLevel:
        """
        Calculate stop loss based on specified method
        
        Args:
            entry_price: Entry price
            zone_type: Support/Resistance zone
            method: SL calculation method
            pivot_levels: Available pivot levels
            market_data: Recent market data for ATR calculation
            config: Strategy configuration
            
        Returns:
            SLTPLevel object with stop loss information
        """
        
        if method == SLMethod.ATR:
            return self._calculate_atr_sl(entry_price, zone_type, market_data, config)
        elif method == SLMethod.BREAKOUT:
            return self._calculate_breakout_sl(entry_price, zone_type, market_data, config)
        elif method == SLMethod.BREAKOUT_CANDLE:
            return self._calculate_breakout_candle_sl(entry_price, zone_type, market_data, config)
        elif method == SLMethod.PIVOT:
            return self._calculate_pivot_sl(entry_price, zone_type, [123,321,5432], config)
        elif method == SLMethod.FIXED:
            return self._calculate_fixed_sl(entry_price, zone_type, config)
        else:
            # Default to ATR
            return self._calculate_breakout_candle_sl(entry_price, zone_type, market_data, config)
    
    def _calculate_atr_sl(self, entry_price: float, zone_type: ZoneType,
                         market_data: pd.DataFrame, config: Dict[str, Any]) -> SLTPLevel:
        """Calculate ATR-based stop loss"""
        
        atr_period = config.get('atr_period', 14)
        atr_multiplier = config.get('atr_multiplier', 1.5)
        max_sl_points = config.get('max_sl_points', 50)
        
        # Calculate ATR
        if len(market_data) < atr_period:
            # Fallback to fixed SL if insufficient data
            return self._calculate_fixed_sl(entry_price, zone_type, config)
        
        high_low = market_data['high'] - market_data['low']
        high_close = abs(market_data['high'] - market_data['close'].shift(1))
        low_close = abs(market_data['low'] - market_data['close'].shift(1))
        
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        atr_value = true_range.rolling(window=atr_period).mean().iloc[-1]
        
        # Calculate SL distance
        sl_distance = atr_value * atr_multiplier
        
        # Apply max SL constraint
        if sl_distance > max_sl_points:
            sl_distance = max_sl_points
            self.logger.warning(f"ATR SL distance {atr_value * atr_multiplier:.2f} "
                              f"capped to max {max_sl_points}")
        
        # Calculate SL price based on zone type
        if zone_type == ZoneType.RESISTANCE:
            sl_price = entry_price - sl_distance
        else:  # SUPPORT or NEUTRAL
            sl_price = entry_price + sl_distance
        
        return SLTPLevel(
            price=round(sl_price, 2),
            type='SL',
            percentage=100.0,
            calculation_method=f"ATR({atr_period}) * {atr_multiplier}"
        )
    
    def _calculate_breakout_sl(self, entry_price: float, zone_type: ZoneType,
                              market_data: pd.DataFrame, config: Dict[str, Any]) -> SLTPLevel:
        """Calculate breakout candle-based stop loss"""
        
        max_sl_points = config.get('max_sl_points', 50)
        breakout_buffer = config.get('breakout_buffer', 5)  # Additional buffer
        
        if len(market_data) < 2:
            return self._calculate_fixed_sl(entry_price, zone_type, config)
        
        # Get the breakout candle (previous candle)
        breakout_candle = market_data.iloc[-2]
        candle_range = breakout_candle['high'] - breakout_candle['low']
        
        # Calculate SL distance
        sl_distance = candle_range + breakout_buffer
        
        # Apply max SL constraint
        if sl_distance > max_sl_points:
            sl_distance = max_sl_points
        
        # Calculate SL price based on zone type
        if zone_type == ZoneType.RESISTANCE:
            sl_price = entry_price - sl_distance
        else:  # SUPPORT or NEUTRAL
            sl_price = entry_price + sl_distance
        
        return SLTPLevel(
            price=round(sl_price, 2),
            type='SL',
            percentage=100.0,
            calculation_method=f"Breakout_Candle + {breakout_buffer}"
        )
    
    def _calculate_breakout_candle_sl(self, entry_price: float, zone_type: ZoneType,
                                     market_data: pd.DataFrame, config: Dict[str, Any]) -> SLTPLevel:
        """Calculate breakout candle-based stop loss using signal candle range"""
        
        max_sl_points = config.get('max_sl_points', 45)
        breakout_buffer = config.get('breakout_buffer', 5)
        
        if len(market_data) < 1:
            return self._calculate_fixed_sl(entry_price, zone_type, config)
        
        # Get signal candle data (last candle in market_data)
        signal_candle = market_data.iloc[-1]
        
        # Calculate signal candle range (high - low)
        signal_candle_range = signal_candle['high'] - signal_candle['low']
        
        # Use signal candle range for SL calculation instead of absolute distance
        # This aligns with the logged signal candle range (54.90)
        sl_distance = signal_candle_range + breakout_buffer
        
        # Validate SL distance doesn't exceed max points
        if sl_distance > max_sl_points:
            self.logger.warning(f"Signal candle range {signal_candle_range:.2f} + buffer {breakout_buffer} = {sl_distance:.2f} exceeds max {max_sl_points}")
            sl_distance = max_sl_points
        
        # Calculate SL price based on signal candle range
        if zone_type == ZoneType.RESISTANCE:  # CALL trades
            sl_price = entry_price - sl_distance
        else:  # SUPPORT or NEUTRAL (PUT trades)
            sl_price = entry_price + sl_distance
        
        self.logger.info(f"Breakout candle SL: signal_range={signal_candle_range:.2f}, buffer={breakout_buffer}, sl_distance={sl_distance:.2f}")
        
        return SLTPLevel(
            price=round(sl_price, 2),
            type='SL',
            percentage=100.0,
            calculation_method=f"Breakout_Candle_Range_{sl_distance:.2f}"
        )
    
    def _calculate_pivot_sl(self, entry_price: float, zone_type: ZoneType,
                           pivot_levels: List[float], config: Dict[str, Any]) -> SLTPLevel:
        """Calculate pivot-based stop loss"""
        
        max_sl_points = config.get('max_sl_points', 50)
        pivot_buffer = config.get('pivot_buffer', 2)  # Buffer beyond pivot
        
        if not pivot_levels:
            return self._calculate_fixed_sl(entry_price, zone_type, config)
        
        sorted_pivots = sorted(pivot_levels)
        
        # Find appropriate pivot for SL
        if zone_type == ZoneType.RESISTANCE:
            # For resistance, use nearest pivot below entry
            support_pivots = [p for p in sorted_pivots if p < entry_price]
            if support_pivots:
                sl_price = max(support_pivots) - pivot_buffer
            else:
                sl_price = entry_price - max_sl_points
        else:  # SUPPORT or NEUTRAL
            # For support, use nearest pivot above entry
            resistance_pivots = [p for p in sorted_pivots if p > entry_price]
            if resistance_pivots:
                sl_price = min(resistance_pivots) + pivot_buffer
            else:
                sl_price = entry_price + max_sl_points
        
        # Validate SL distance
        sl_distance = abs(entry_price - sl_price)
        if sl_distance > max_sl_points:
            if zone_type == ZoneType.RESISTANCE:
                sl_price = entry_price - max_sl_points
            else:
                sl_price = entry_price + max_sl_points
        
        return SLTPLevel(
            price=round(sl_price, 2),
            type='SL',
            percentage=100.0,
            calculation_method=f"Pivot + {pivot_buffer}"
        )
    
    def _calculate_fixed_sl(self, entry_price: float, zone_type: ZoneType,
                           config: Dict[str, Any]) -> SLTPLevel:
        """Calculate fixed point stop loss"""
        
        max_sl_points = config.get('max_sl_points', 50)
        
        if zone_type == ZoneType.RESISTANCE:
            sl_price = entry_price - max_sl_points
        else:  # SUPPORT or NEUTRAL
            sl_price = entry_price + max_sl_points
        
        return SLTPLevel(
            price=round(sl_price, 2),
            type='SL',
            percentage=100.0,
            calculation_method="Fixed_Points"
        )


class TPCalculator:
    """Advanced Take Profit calculation algorithms"""
    
    def __init__(self, config_cache=None):
        self.config_cache = config_cache
        self.logger = get_logger(__name__) if CONFIG_AVAILABLE else logging.getLogger(__name__)
        
        # Fibonacci levels for TP calculation
        self.fibonacci_levels = [0.618, 1.0, 1.618, 2.618]
    
    def calculate_take_profits(self, entry_price: float, sl_price: float,
                             zone_type: ZoneType, method: TPMethod,
                             pivot_levels: List[float], config: Dict[str, Any]) -> List[SLTPLevel]:
        """
        Calculate multiple take profit levels
        
        Args:
            entry_price: Entry price
            sl_price: Stop loss price
            zone_type: Support/Resistance zone
            method: TP calculation method
            pivot_levels: Available pivot levels
            config: Strategy configuration
            
        Returns:
            List of SLTPLevel objects for take profits
        """
        
        if method == TPMethod.PIVOT:
            return self._calculate_pivot_tp(entry_price, sl_price, zone_type, pivot_levels, config)
        elif method == TPMethod.FIBONACCI:
            return self._calculate_fibonacci_tp(entry_price, sl_price, zone_type, config)
        elif method == TPMethod.HYBRID:
            return self._calculate_hybrid_tp(entry_price, sl_price, zone_type, pivot_levels, config)
        elif method == TPMethod.RATIO:
            return self._calculate_ratio_tp(entry_price, sl_price, zone_type, config)
        elif method == TPMethod.BREAKOUT_CANDLE:
            return self._calculate_breakout_candle_tp(entry_price, sl_price, zone_type, config)
        else:
            # Default to ratio-based
            return self._calculate_ratio_tp(entry_price, sl_price, zone_type, config)
    
    def _calculate_pivot_tp(self, entry_price: float, sl_price: float,
                           zone_type: ZoneType, pivot_levels: List[float],
                           config: Dict[str, Any]) -> List[SLTPLevel]:
        """Calculate pivot-based take profit levels"""
        
        if not pivot_levels:
            return self._calculate_ratio_tp(entry_price, sl_price, zone_type, config)
        
        risk_distance = abs(entry_price - sl_price)
        min_ratio = config.get('min_risk_reward_ratio', 1.5)
        max_ratio = config.get('max_risk_reward_ratio', 3.0)
        
        sorted_pivots = sorted(pivot_levels)
        tp_levels = []
        
        # Generate pivot + midpoint levels
        pivot_and_midpoints = self._generate_pivot_midpoints(sorted_pivots)
        
        if zone_type == ZoneType.RESISTANCE:
            # For resistance breakout, target pivots above entry
            target_pivots = [p for p in pivot_and_midpoints if p > entry_price]
            
            for i, pivot in enumerate(target_pivots[:3]):  # Max 3 targets
                tp_distance = pivot - entry_price
                ratio = tp_distance / risk_distance if risk_distance > 0 else 0
                
                if min_ratio <= ratio <= max_ratio:
                    tp_levels.append(SLTPLevel(
                        price=round(pivot, 2),
                        type=f'TP{i+1}',
                        percentage=self._get_target_percentage(i),
                        calculation_method="Pivot_Level"
                    ))
        
        else:  # SUPPORT or NEUTRAL
            # For support breakout, target pivots below entry
            target_pivots = [p for p in reversed(pivot_and_midpoints) if p < entry_price]
            
            for i, pivot in enumerate(target_pivots[:3]):  # Max 3 targets
                tp_distance = entry_price - pivot
                ratio = tp_distance / risk_distance if risk_distance > 0 else 0
                
                if min_ratio <= ratio <= max_ratio:
                    tp_levels.append(SLTPLevel(
                        price=round(pivot, 2),
                        type=f'TP{i+1}',
                        percentage=self._get_target_percentage(i),
                        calculation_method="Pivot_Level"
                    ))
        
        # If no pivot levels found, fallback to ratio
        if not tp_levels:
            return self._calculate_ratio_tp(entry_price, sl_price, zone_type, config)
        
        return tp_levels
    
    def _calculate_fibonacci_tp(self, entry_price: float, sl_price: float,
                               zone_type: ZoneType, config: Dict[str, Any]) -> List[SLTPLevel]:
        """Calculate Fibonacci-based take profit levels"""
        
        risk_distance = abs(entry_price - sl_price)
        tp_levels = []
        
        for i, fib_level in enumerate(self.fibonacci_levels):
            if i >= 3:  # Max 3 targets
                break
                
            tp_distance = risk_distance * fib_level
            
            if zone_type == ZoneType.RESISTANCE:
                tp_price = entry_price + tp_distance
            else:  # SUPPORT or NEUTRAL
                tp_price = entry_price - tp_distance
            
            tp_levels.append(SLTPLevel(
                price=round(tp_price, 2),
                type=f'TP{i+1}',
                percentage=self._get_target_percentage(i),
                calculation_method=f"Fibonacci_{fib_level}"
            ))
        
        return tp_levels
    
    def _calculate_hybrid_tp(self, entry_price: float, sl_price: float,
                            zone_type: ZoneType, pivot_levels: List[float],
                            config: Dict[str, Any]) -> List[SLTPLevel]:
        """Calculate hybrid pivot + Fibonacci take profit levels"""
        
        # Start with pivot levels
        pivot_tps = self._calculate_pivot_tp(entry_price, sl_price, zone_type, pivot_levels, config)
        
        # If we have pivot levels, use them for first targets
        if pivot_tps:
            # Add Fibonacci extension for final target
            risk_distance = abs(entry_price - sl_price)
            fib_extension = 2.618  # Fibonacci extension level
            
            if zone_type == ZoneType.RESISTANCE:
                fib_tp_price = entry_price + (risk_distance * fib_extension)
            else:
                fib_tp_price = entry_price - (risk_distance * fib_extension)
            
            # Add as final target if it's beyond the last pivot target
            if len(pivot_tps) < 3:
                last_pivot_price = pivot_tps[-1].price if pivot_tps else entry_price
                
                if ((zone_type == ZoneType.RESISTANCE and fib_tp_price > last_pivot_price) or
                    (zone_type != ZoneType.RESISTANCE and fib_tp_price < last_pivot_price)):
                    
                    pivot_tps.append(SLTPLevel(
                        price=round(fib_tp_price, 2),
                        type=f'TP{len(pivot_tps)+1}',
                        percentage=self._get_target_percentage(len(pivot_tps)),
                        calculation_method="Fibonacci_Extension"
                    ))
            
            return pivot_tps
        
        else:
            # Fallback to Fibonacci if no pivots
            return self._calculate_fibonacci_tp(entry_price, sl_price, zone_type, config)
    
    def _calculate_ratio_tp(self, entry_price: float, sl_price: float,
                           zone_type: ZoneType, config: Dict[str, Any]) -> List[SLTPLevel]:
        """Calculate ratio-based take profit levels"""
        
        risk_distance = abs(entry_price - sl_price)
        risk_ratios = config.get('risk_reward_ratios', [1.5, 2.0, 3.0])
        target_percentages = config.get('target_percentages', [50, 30, 20])
        
        tp_levels = []
        
        for i, ratio in enumerate(risk_ratios):
            if i >= len(target_percentages):
                break
                
            tp_distance = risk_distance * ratio
            
            if zone_type == ZoneType.RESISTANCE:
                tp_price = entry_price + tp_distance
            else:  # SUPPORT or NEUTRAL
                tp_price = entry_price - tp_distance
            
            tp_levels.append(SLTPLevel(
                price=round(tp_price, 2),
                type=f'TP{i+1}',
                percentage=target_percentages[i],
                calculation_method=f"Ratio_{ratio}"
            ))
        
        return tp_levels
    
    def _calculate_breakout_candle_tp(self, entry_price: float, sl_price: float,
                                     zone_type: ZoneType, config: Dict[str, Any]) -> List[SLTPLevel]:
        """Calculate breakout candle-based take profit using signal candle range"""
        
        # Get signal candle range from config (passed from market data)
        signal_candle_range = config.get('signal_candle_range', 0)
        buffer_sl_tp = config.get('buffer_sl_tp', 5)
        
        # CRITICAL FIX: Add detailed logging for debugging TP calculation
        self.logger.info(f"[TP_DEBUG] Entry price: {entry_price:.2f}")
        self.logger.info(f"[TP_DEBUG] Signal candle range from config: {signal_candle_range:.2f}")
        self.logger.info(f"[TP_DEBUG] Buffer: {buffer_sl_tp:.2f}")
        self.logger.info(f"[TP_DEBUG] Zone type: {zone_type}")
        
        # If no signal candle range provided, fallback to ratio-based
        if signal_candle_range <= 0:
            self.logger.warning(f"[TP_DEBUG] No signal candle range provided, falling back to ratio-based TP")
            return self._calculate_ratio_tp(entry_price, sl_price, zone_type, config)
        
        # Calculate TP: Entry + Signal Candle Range + Buffer
        tp_distance = signal_candle_range + buffer_sl_tp
        
        self.logger.info(f"[TP_DEBUG] Calculated TP distance: {tp_distance:.2f}")
        
        if zone_type == ZoneType.RESISTANCE:  # CALL trades
            tp_price = entry_price + tp_distance
        else:  # SUPPORT or NEUTRAL (PUT trades)
            tp_price = entry_price - tp_distance
        
        self.logger.info(f"[TP_DEBUG] Final TP price: {tp_price:.2f}")
        
        tp_levels = [
            SLTPLevel(
                price=round(tp_price, 2),
                type='TP1',
                percentage=100.0,
                calculation_method=f"Breakout_Candle_TP_{tp_distance:.2f}"
            )
        ]
        
        return tp_levels
    
    def _generate_pivot_midpoints(self, pivot_levels: List[float]) -> List[float]:
        """Generate pivot levels with midpoints between them"""
        
        if len(pivot_levels) < 2:
            return pivot_levels
        
        pivot_and_midpoints = []
        
        for i in range(len(pivot_levels) - 1):
            pivot_and_midpoints.append(pivot_levels[i])
            # Add midpoint
            midpoint = (pivot_levels[i] + pivot_levels[i + 1]) / 2
            pivot_and_midpoints.append(midpoint)
        
        # Add the last pivot
        pivot_and_midpoints.append(pivot_levels[-1])
        
        return sorted(pivot_and_midpoints)
    
    def _get_target_percentage(self, target_index: int) -> float:
        """Get position percentage for target based on index"""
        
        default_percentages = [50.0, 30.0, 20.0]
        
        if target_index < len(default_percentages):
            return default_percentages[target_index]
        
        return 20.0  # Default for additional targets


class AdvancedSLTPEngine:
    """
    Main SL/TP calculation engine with advanced algorithms and config integration
    """
    
    def __init__(self, config_path: str = "config/config.xlsx"):
        """Initialize the advanced SL/TP engine"""
        
        self.config_path = config_path
        self.logger = self._setup_logger()
        
        # Initialize config cache
        if CONFIG_AVAILABLE:
            self.config_cache = get_cached_config(config_path)
        else:
            self.config_cache = None
            
        # Initialize calculation components
        self.zone_detector = ZoneDetector(self.config_cache)
        self.sl_calculator = SLCalculator(self.config_cache)
        self.tp_calculator = TPCalculator(self.config_cache)
        
        self.logger.info("AdvancedSLTPEngine initialized successfully")
    
    def _setup_logger(self) -> logging.Logger:
        """Setup logger for the engine"""
        
        if CONFIG_AVAILABLE:
            try:
                return get_logger(__name__)
            except Exception:
                pass
        
        # Fallback logger
        logger = logging.getLogger(__name__)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        
        return logger
    
    def calculate_sl_tp(self, 
                       symbol: str,
                       strategy_name: str,
                       entry_price: float,
                       pivot_levels: List[float],
                       market_data: pd.DataFrame,
                       position_size: Optional[float] = None,
                       custom_config: Optional[Dict[str, Any]] = None) -> SLTPResult:
        """
        Calculate complete SL/TP levels for a trade entry
        
        Args:
            symbol: Trading symbol
            strategy_name: Strategy name for config lookup
            entry_price: Entry price level
            pivot_levels: List of pivot points (S1-S4, R1-R4, etc.)
            market_data: Recent market data for calculations
            position_size: Position size for risk calculation
            custom_config: Override configuration parameters
            
        Returns:
            SLTPResult with complete SL/TP information
        """
        
        try:
            # Load configuration
            config = self._load_strategy_config(strategy_name, custom_config)
            
            # Validate inputs
            if entry_price <= 0:
                raise ValueError(f"Invalid entry price: {entry_price}")
            
            if len(market_data) == 0:
                raise ValueError("Empty market data provided")
            
            # Get recent high/low for zone detection
            recent_data = market_data.tail(20)  # Last 20 bars for context
            recent_high = recent_data['high'].max()
            recent_low = recent_data['low'].min()
            
            # Detect zone type and strength
            zone_type, zone_strength = self.zone_detector.detect_zone_type(
                entry_price, pivot_levels, recent_high, recent_low
            )
            
            self.logger.info(f"Zone detected: {zone_type.value} (strength: {zone_strength:.2f})")
            
            # Calculate stop loss
            sl_method = SLMethod(config.get('sl_method', 'Breakout_Candle'))
            stop_loss = self.sl_calculator.calculate_stop_loss(
                entry_price, zone_type, sl_method, pivot_levels, market_data, config
            )
            
            # Calculate take profits
            tp_method = TPMethod(config.get('tp_method', 'Breakout_Candle'))
            take_profits = self.tp_calculator.calculate_take_profits(
                entry_price, stop_loss.price, zone_type, tp_method, pivot_levels, config
            )
            
            # Calculate risk metrics
            risk_amount = abs(entry_price - stop_loss.price)
            risk_percentage = (risk_amount / entry_price) * 100
            
            # Calculate overall risk-reward ratio
            total_reward = sum(
                abs(tp.price - entry_price) * (tp.percentage / 100) 
                for tp in take_profits
            )
            risk_reward_ratio = total_reward / risk_amount if risk_amount > 0 else 0
            
            # Create result
            result = SLTPResult(
                symbol=symbol,
                strategy_name=strategy_name,
                entry_price=entry_price,
                zone_type=zone_type,
                stop_loss=stop_loss,
                take_profits=take_profits,
                risk_amount=risk_amount,
                risk_percentage=risk_percentage,
                risk_reward_ratio=risk_reward_ratio,
                pivot_levels_used=pivot_levels.copy(),
                zone_strength=zone_strength
            )
            
            self.logger.info(f"SL/TP calculated: SL={stop_loss.price:.2f}, "
                           f"TPs={[tp.price for tp in take_profits]}, "
                           f"RR={risk_reward_ratio:.2f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating SL/TP for {symbol}: {e}")
            raise
    
    def _load_strategy_config(self, strategy_name: str, 
                             custom_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Load strategy-specific configuration with SL/TP integration"""
        
        # Default configuration
        default_config = {
            'max_sl_points': 50,
            'sl_method': 'ATR',
            'tp_method': 'HYBRID',
            'atr_period': 14,
            'atr_multiplier': 1.5,
            'min_risk_reward_ratio': 1.5,
            'max_risk_reward_ratio': 3.0,
            'risk_reward_ratios': [1.5, 2.0, 3.0],
            'target_percentages': [50, 30, 20],
            'breakout_buffer': 5,
            'pivot_buffer': 2,
            'volatility_adjustment': True,
            'multi_target_mode': True
        }
        
        # Load from config cache if available
        if self.config_cache:
            try:
                config_loader = self.config_cache.get_config_loader()
                
                # Load strategy-specific SL/TP configuration
                sl_tp_config = config_loader.get_strategy_sl_tp_config(strategy_name)
                
                # Map configuration keys (remove 'default_' prefix)
                for key, value in sl_tp_config.items():
                    if key.startswith('default_'):
                        clean_key = key.replace('default_', '')
                        default_config[clean_key] = value
                
                # Load general strategy configuration
                strategies = config_loader.get_strategy_configs_enhanced()
                if strategy_name in strategies:
                    strategy_config = strategies[strategy_name]
                    
                    # Extract additional SL/TP specific settings
                    strategy_mapping = {
                        'position_size': 'position_size',
                        'risk_per_trade': 'risk_per_trade',
                        'max_positions': 'max_positions'
                    }
                    
                    for strategy_field, config_field in strategy_mapping.items():
                        if strategy_field in strategy_config:
                            default_config[config_field] = strategy_config[strategy_field]
                
                self.logger.info(f"Loaded SL/TP config for strategy {strategy_name}: "
                               f"SL_method={default_config['sl_method']}, "
                               f"TP_method={default_config['tp_method']}, "
                               f"Max_SL={default_config['max_sl_points']}")
                    
            except Exception as e:
                self.logger.warning(f"Could not load strategy SL/TP config: {e}")
        
        # Apply custom overrides
        if custom_config:
            default_config.update(custom_config)
        
        return default_config
    
    def update_sl_tp_levels(self, current_result: SLTPResult, 
                           current_price: float, 
                           market_data: pd.DataFrame) -> SLTPResult:
        """
        Update SL/TP levels based on current market conditions (for trailing stops, etc.)
        
        Args:
            current_result: Current SL/TP result
            current_price: Current market price
            market_data: Updated market data
            
        Returns:
            Updated SLTPResult
        """
        
        # For now, return the same result (future: implement trailing stops)
        # This is where trailing stop logic would be implemented
        
        self.logger.debug(f"SL/TP levels updated for {current_result.symbol}")
        return current_result
    
    def validate_sl_tp_levels(self, result: SLTPResult) -> List[str]:
        """
        Validate SL/TP levels for safety and consistency
        
        Args:
            result: SL/TP calculation result
            
        Returns:
            List of validation warnings/errors
        """
        
        warnings = []
        
        # Check SL direction
        if result.zone_type == ZoneType.RESISTANCE:
            if result.stop_loss.price >= result.entry_price:
                warnings.append("Stop loss should be below entry for resistance breakout")
        elif result.zone_type == ZoneType.SUPPORT:
            if result.stop_loss.price <= result.entry_price:
                warnings.append("Stop loss should be above entry for support breakout")
        
        # Check TP directions
        for tp in result.take_profits:
            if result.zone_type == ZoneType.RESISTANCE:
                if tp.price <= result.entry_price:
                    warnings.append(f"Take profit {tp.type} should be above entry for resistance")
            elif result.zone_type == ZoneType.SUPPORT:
                if tp.price >= result.entry_price:
                    warnings.append(f"Take profit {tp.type} should be below entry for support")
        
        # Check risk-reward ratio
        if result.risk_reward_ratio < 1.0:
            warnings.append(f"Low risk-reward ratio: {result.risk_reward_ratio:.2f}")
        
        # Check position percentages sum to 100%
        total_percentage = sum(tp.percentage for tp in result.take_profits)
        if abs(total_percentage - 100.0) > 0.01:
            warnings.append(f"Target percentages don't sum to 100%: {total_percentage}%")
        
        return warnings


# Convenience function for easy access
def create_sl_tp_engine(config_path: str = "config/config.xlsx") -> AdvancedSLTPEngine:
    """Create and return SL/TP engine instance"""
    return AdvancedSLTPEngine(config_path)


# Example usage
if __name__ == "__main__":
    # Example usage of the SL/TP engine
    engine = create_sl_tp_engine()
    
    # Sample data
    sample_pivots = [380, 390, 410, 440, 450, 490, 545]  # S4, S3, S2, S1, R1, R2, R3
    sample_data = pd.DataFrame({
        'high': [420, 425, 430, 428, 435],
        'low': [415, 418, 425, 422, 430],
        'close': [418, 422, 428, 425, 433],
        'volume': [1000, 1200, 1100, 900, 1300]
    })
    
    # Calculate SL/TP
    result = engine.calculate_sl_tp(
        symbol="NIFTY",
        strategy_name="EMA_RSI_Scalping",
        entry_price=430.0,
        pivot_levels=sample_pivots,
        market_data=sample_data
    )
    
    print(f"Zone: {result.zone_type.value}")
    print(f"Stop Loss: {result.stop_loss.price}")
    print(f"Take Profits: {[(tp.type, tp.price, f'{tp.percentage}%') for tp in result.take_profits]}")
    print(f"Risk-Reward Ratio: {result.risk_reward_ratio:.2f}")