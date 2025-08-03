#!/usr/bin/env python3
"""
Options Real Money Integrator for vAlgo Trading System
=====================================================

Advanced options trading integrator with real money calculations using strike selector
and parallel chart tracking for both equity OHLC and strike OHLC with dual SL/TP logic.

Features:
- Real money options entry/exit calculations
- Integration with AdvancedStrikeSelector for ATM/ITM/OTM strike selection
- Parallel chart tracking for equity and options strikes
- Dual SL/TP logic for both equity and options data
- Real-time option chain data integration
- Multi-timeframe support (5m signals, 1m execution)
- Commission and slippage calculations
- Options Greeks consideration
- Risk management and position sizing

Author: vAlgo Development Team
Created: July 11, 2025
Version: 1.0.0 (Production)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Union
from datetime import datetime, timedelta
from enum import Enum
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from utils.advanced_strike_selector import AdvancedStrikeSelector, StrikeType, OptionType
from utils.advanced_sl_tp_engine import AdvancedSLTPEngine, SLMethod, TPMethod
from utils.identify_sl_tp import get_sl_tp
from backtest_engine.options_pnl_calculator import OptionsPnLCalculator
from config.options_config import get_options_config
from data_manager.options_database import OptionsDatabase, create_options_database
from utils.config_cache import get_cached_config


class TradeSignal(Enum):
    """Trade signal types"""
    CALL = "CALL"
    PUT = "PUT"
    LONG = "LONG"
    SHORT = "SHORT"


class TradeStatus(Enum):
    """Trade status types"""
    OPEN = "OPEN"
    TP_HIT = "TP_HIT"
    SL_HIT = "SL_HIT"
    EXPIRED = "EXPIRED"
    MANUAL_EXIT = "MANUAL_EXIT"


class OptionsRealMoneyIntegrator:
    """
    Advanced options trading integrator with real money calculations
    and parallel chart tracking capabilities.
    """
    
    def __init__(self, config_path: Optional[str] = None,
                 options_db: Optional[OptionsDatabase] = None):
        """
        Initialize Options Real Money Integrator.
        
        Args:
            config_path: Path to configuration file
            options_db: Optional OptionsDatabase instance
        """
        self.logger = get_logger(__name__)
        self.config_path = config_path or "config/config.xlsx"
        
        # Initialize components
        self.options_db = options_db or create_options_database()
        self.strike_selector = AdvancedStrikeSelector(self.options_db, self.config_path)
        self.sl_tp_engine = AdvancedSLTPEngine(self.config_path)
        self.pnl_calculator = OptionsPnLCalculator()
        self.options_config = get_options_config()
        
        # Load configuration
        self.config = self._load_configuration()
        
        # Initialize tracking systems
        self.active_trades = {}
        self.trade_history = []
        self.equity_charts = {}
        self.options_charts = {}
        
        # Daily entry tracking
        self.daily_entries_count = 0
        self.current_trading_date = None
        self.daily_entry_limit = self._get_daily_entry_limit()
        
        # Performance tracking
        self.performance_metrics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'total_pnl': 0.0,
            'max_drawdown': 0.0,
            'current_drawdown': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0
        }
        
        self.logger.info("OptionsRealMoneyIntegrator initialized successfully")
    
    def _get_daily_entry_limit(self) -> int:
        """Get daily entry limit from configuration."""
        try:
            config_cache = get_cached_config(self.config_path)
            if config_cache is None:
                self.logger.error("Config cache is None")
                return 1
            config_loader = config_cache.get_config_loader() if hasattr(config_cache, 'get_config_loader') else None
            if config_loader is None or not hasattr(config_loader, 'get_daily_entry_limit'):
                self.logger.error("Config loader is None or missing get_daily_entry_limit")
                return 1
            daily_limit = config_loader.get_daily_entry_limit()
            self.logger.info(f"Daily entry limit set to: {daily_limit}")
            return daily_limit
        except Exception as e:
            self.logger.error(f"Error getting daily entry limit: {e}")
            return 1  # Default to 1 entry per day

    def _load_configuration(self) -> Dict[str, Any]:
        """Load configuration from cached config."""
        try:
            config_cache = get_cached_config(self.config_path)
            if config_cache is None:
                self.logger.error("Config cache is None")
                return self._get_default_config()
            config_loader = config_cache.get_config_loader() if hasattr(config_cache, 'get_config_loader') else None
            if config_loader is None:
                self.logger.error("Config loader is None")
                return self._get_default_config()
            # Get strategy and options configurations
            strategies = config_loader.get_strategy_configs_enhanced() if hasattr(config_loader, 'get_strategy_configs_enhanced') else {}
            sl_tp_config = config_loader.get_sl_tp_config() if hasattr(config_loader, 'get_sl_tp_config') else {}
            exit_rules = config_loader.get_exit_conditions_enhanced() if hasattr(config_loader, 'get_exit_conditions_enhanced') else {}
            
            # Extract options-specific parameters from strategy configurations
            options_params = {
                'default_strike_selection': 'ATM',
                'enable_parallel_tracking': True,
                'use_real_option_chain': True,
                'commission_per_trade': 25,
                'slippage_points': 2,
                'max_position_size': 10000,
                'risk_per_trade': 0.02,
                'enable_dual_sl_tp': True,
                'equity_sl_buffer': 5,
                'options_sl_buffer': 10,
                'timeframe_signal': '5min',
                'timeframe_execution': '1min',
                'enable_greeks_tracking': True,
                'min_time_to_expiry': 1,  # Minimum days to expiry
                'max_time_to_expiry': 30,  # Maximum days to expiry
                'strategies': strategies,  # Store strategies for parameter access
                'enable_breakout_confirmation': True,  # Wait for breakout confirmation
                'use_cpr_target_stoploss': True,  # Use CPR-based SL/TP
                'cpr_sl_method': 'ATR',  # ATR/Breakout/Pivot/Fixed
                'cpr_tp_method': 'HYBRID',  # Pivot/Fibonacci/Hybrid/Ratio
                'breakout_timeout_minutes': 5,  # Max wait time for breakout
                'real_entry_exit_logic': True,  # Use real entry/exit price logic
            }
            
            # Update with SL/TP configuration
            options_params.update(sl_tp_config)
            
            # Add exit rules to configuration
            options_params['exit_rules'] = exit_rules
            
            self.logger.info(f"Loaded options configuration: {len(options_params)} parameters")
            return options_params
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration."""
        return {
            'default_strike_selection': 'ATM',
            'enable_parallel_tracking': True,
            'use_real_option_chain': True,
            'commission_per_trade': 25,
            'slippage_points': 2,
            'max_position_size': 10000,
            'risk_per_trade': 0.02,
            'enable_dual_sl_tp': True,
            'equity_sl_buffer': 5,
            'options_sl_buffer': 10,
            'timeframe_signal': '5min',
            'timeframe_execution': '1min',
            'enable_greeks_tracking': True,
            'min_time_to_expiry': 1,
            'max_time_to_expiry': 30,
        }
    
    def get_strategy_options_params(self, strategy_name: str) -> Dict[str, Any]:
        """
        Get options trading parameters for specific strategy
        
        Args:
            strategy_name: Name of the strategy
            
        Returns:
            Dictionary with strategy-specific options parameters
        """
        try:
            strategies = self.config.get('strategies', {}) if self.config else {}
            strategy_config = strategies.get(strategy_name, {})
            
            # Extract enhanced options parameters with defaults
            return {
                'strike_preference': strategy_config.get('strike_preference', 'ATM'),
                'cpr_target_stoploss': strategy_config.get('cpr_target_stoploss', True),
                'breakout_confirmation': strategy_config.get('breakout_confirmation', False),
                'sl_method': strategy_config.get('sl_method', 'ATR'),
                'tp_method': strategy_config.get('tp_method', 'HYBRID'),
                'real_option_data': strategy_config.get('real_option_data', True),
                'position_size': strategy_config.get('position_size', 1000),
                'risk_per_trade': strategy_config.get('risk_per_trade', 0.02),
                'max_positions': strategy_config.get('max_positions', 1)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting strategy options params for {strategy_name}: {e}")
            return {
                'strike_preference': 'ATM',
                'cpr_target_stoploss': True,
                'breakout_confirmation': False,
                'sl_method': 'ATR',
                'tp_method': 'HYBRID',
                'real_option_data': True,
                'position_size': 1000,
                'risk_per_trade': 0.02,
                'max_positions': 1
            }
    
    def process_trade_signal(self, signal_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Process trade signal with real money calculations following entry_exit_price_logic.md.
        
        Real Entry Logic:
        - Call Entry: High of current candle becomes ENTRY_POINT
        - Put Entry: Low of current candle becomes ENTRY_POINT
        - Wait for breakout confirmation
        - Use CPR-based SL/TP before entry
        
        Args:
            signal_data: Trade signal data containing:
                - signal_type: CALL/PUT
                - current_candle: Current OHLC candle data
                - timestamp: Signal timestamp (close time of candle)
                - strategy_name: Strategy name
                - strike_preference: ATM/ITM1/OTM1 etc (from config)
                - symbol: Trading symbol
                - pivot_levels: CPR/Pivot levels for SL/TP
                - market_data: Recent market data for calculations
                
        Returns:
            Trade execution result with real money calculations or None if failed
        """
        try:
            # Extract signal parameters
            signal_type = signal_data.get('signal_type', 'CALL')
            current_candle = signal_data.get('current_candle', {})
            timestamp = signal_data.get('timestamp', datetime.now())  # Signal time = candle close time
            strategy_name = signal_data.get('strategy_name', 'default')
            symbol = signal_data.get('symbol', 'NIFTY')
            pivot_levels = signal_data.get('pivot_levels', [])
            market_data = signal_data.get('market_data', pd.DataFrame())
            
            self.logger.info(f"Processing {signal_type} signal for {symbol} at {timestamp}")
            
            # Validate signal
            if not self._validate_real_signal(signal_data):
                self.logger.warning("Real signal validation failed")
                return None
            
            # STEP 1: Determine ENTRY_POINT from current candle
            if signal_type == 'CALL':
                entry_point = float(current_candle.get('high', 0))  # Call = candle high
            else:  # PUT
                entry_point = float(current_candle.get('low', 0))   # Put = candle low
            
            self.logger.info(f"Entry point determined: {entry_point} (Candle {signal_type.lower()})")
            
            # STEP 2: Get strike preference from strategy config
            strike_preference = self._get_strategy_strike_preference(strategy_name)
            
            # STEP 3: Calculate SL/TP using strategy configuration
            sl_tp_result = self._calculate_strategy_sl_tp(
                symbol, strategy_name, entry_point, pivot_levels, market_data, current_candle
            )
            
            if not sl_tp_result:
                self.logger.warning("Failed to calculate SL/TP for strategy")
                return None
            
            self.logger.info(f"Strategy SL/TP calculated: SL={sl_tp_result.get('sl_price', 'N/A'):.2f}, "
                           f"TP_levels={len(sl_tp_result.get('tp_levels', []))}, "
                           f"Method={sl_tp_result.get('sl_method', 'Unknown')}")
            
            # STEP 4: Select strike based on ENTRY_POINT time
            option_type = OptionType.CALL if signal_type == 'CALL' else OptionType.PUT
            strike_type = StrikeType(strike_preference)
            
            strike_info = self.strike_selector.select_strike(
                entry_point, strike_type, option_type, timestamp
            )
            
            if not strike_info:
                self.logger.warning(f"Could not select {strike_preference} {signal_type} strike")
                return None
            
            self.logger.info(f"Selected strike: {strike_info['strike']} ({strike_preference})")
            
            # STEP 5: Create trade with WAITING_FOR_BREAKOUT status if breakout confirmation enabled
            if self.config.get('enable_breakout_confirmation', True):
                # Calculate position size before creating the trade
                position_size = self._calculate_real_position_size(strike_info, sl_tp_result)
                
                # Create trade initially with WAITING_FOR_BREAKOUT status
                trade_result = self._create_trade_waiting_for_breakout(
                    signal_data, strike_info, sl_tp_result, entry_point, position_size
                )
                
                if not trade_result:
                    self.logger.warning("Failed to create trade for breakout confirmation")
                    return None
                
                # Wait for breakout confirmation
                breakout_result = self._wait_for_breakout_confirmation(
                    signal_data, entry_point, strike_info
                )
                
                if not breakout_result:
                    self.logger.warning("Breakout confirmation failed or timed out")
                    # Cancel the trade
                    trade_result['status'] = 'CANCELLED'
                    trade_result['exit_reason'] = 'BREAKOUT_TIMEOUT'
                    return None
                
                # Handle breakeven day (no breakout occurred)
                if breakout_result.get('is_breakeven', False):
                    self.logger.info("Handling breakeven day - no breakout occurred")
                    trade_result['status'] = 'CLOSED'
                    trade_result['exit_reason'] = 'NO_BREAKOUT_BREAKEVEN'
                    trade_result['exit_time'] = breakout_result['entry_time']
                    trade_result['exit_premium'] = 0.0
                    trade_result['pnl'] = 0.0  # Breakeven
                    trade_result['breakout_confirmed'] = False
                    
                    # Add to trade history but don't start tracking since no actual trade occurred
                    self.trade_history.append(trade_result)
                    
                    self.logger.info(f"Breakeven trade completed: {trade_result['trade_id']}")
                    return trade_result
                
                # Update trade to OPEN status after breakout confirmation
                trade_result['status'] = 'OPEN'
                trade_result['entry_time'] = breakout_result['entry_time']
                trade_result['entry_premium'] = breakout_result['strike_entry_price']
                trade_result['underlying_entry_price'] = breakout_result['entry_price']  # Store the breakout price
                trade_result['breakout_confirmed'] = True
                trade_result['breakout_delay_minutes'] = breakout_result['breakout_delay_minutes']
                
                self.logger.info(f"Trade {trade_result['trade_id']} moved to OPEN status after breakout confirmation")
                
                # Use actual breakout data for final entry
                real_entry_data = breakout_result
            else:
                # Use immediate entry without breakout confirmation
                real_entry_data = {
                    'entry_time': timestamp,
                    'entry_price': entry_point,
                    'strike_entry_price': strike_info.get('premium', 0)
                }
                
                # STEP 6: Execute real money trade immediately
                trade_result = self._execute_real_money_trade(
                    signal_data, strike_info, sl_tp_result, real_entry_data
                )
            
            if trade_result:
                # Start parallel tracking with real data
                self._start_real_parallel_tracking(trade_result)
                
                # Update performance metrics
                self._update_performance_metrics(trade_result, 'REAL_ENTRY')
                
                self.logger.info(f"Real money trade executed: {trade_result['trade_id']} - "
                               f"Strike: {trade_result['strike']}, Entry: {trade_result['entry_premium']:.2f}")
            
            return trade_result
            
        except Exception as e:
            self.logger.error(f"Error processing real trade signal: {e}")
            return None
    
    def _check_daily_limit(self, signal_timestamp: datetime) -> bool:
        """Check if daily entry limit has been reached."""
        try:
            if signal_timestamp is None:
                self.logger.error("signal_timestamp is None")
                return False
            # If pandas Timestamp, convert to python datetime
            if isinstance(signal_timestamp, pd.Timestamp):
                signal_timestamp = signal_timestamp.to_pydatetime()
            if not isinstance(signal_timestamp, (datetime,)):
                self.logger.error("signal_timestamp is not a datetime instance")
                return False
            if signal_timestamp is None:
                self.logger.error("signal_timestamp is None after conversion")
                return False
            trading_date = signal_timestamp.date() if hasattr(signal_timestamp, 'date') and signal_timestamp is not None else None
            if trading_date is None:
                self.logger.error("Could not extract trading_date from signal_timestamp")
                return False
            
            # Check if we're on a new trading day
            if self.current_trading_date != trading_date:
                # Reset daily counter for new trading day
                self.current_trading_date = trading_date
                self.daily_entries_count = 0
                self.logger.info(f"New trading day detected: {trading_date}. Daily entry counter reset.")
            
            # Check if daily limit exceeded
            if self.daily_entries_count >= self.daily_entry_limit:
                self.logger.warning(f"Daily entry limit reached: {self.daily_entries_count}/{self.daily_entry_limit} for {trading_date}")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking daily limit: {e}")
            return False
    
    def _increment_daily_entries(self) -> None:
        """Increment daily entry counter."""
        self.daily_entries_count += 1
        self.logger.info(f"Daily entries: {self.daily_entries_count}/{self.daily_entry_limit} for {self.current_trading_date}")

    def _validate_real_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Validate real signal data with enhanced requirements."""
        try:
            required_fields = ['signal_type', 'current_candle', 'timestamp', 'symbol']
            for field in required_fields:
                if field not in signal_data:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate signal type
            if signal_data['signal_type'] not in ['CALL', 'PUT']:
                self.logger.warning(f"Invalid signal type: {signal_data['signal_type']}")
                return False
            
            # Validate current candle has OHLC data
            current_candle = signal_data['current_candle']
            required_candle_fields = ['open', 'high', 'low', 'close']
            for field in required_candle_fields:
                if field not in current_candle or current_candle[field] is None or current_candle[field] <= 0:
                    self.logger.warning(f"Invalid or missing candle field: {field}")
                    return False
            
            # Check daily entry limit instead of concurrent trades
            signal_timestamp = signal_data.get('timestamp')
            if signal_timestamp is None:
                self.logger.error("signal_timestamp is None in signal_data")
                return False
            if hasattr(signal_timestamp, 'to_pydatetime'):
                signal_timestamp = signal_timestamp.to_pydatetime()
            if not self._check_daily_limit(signal_timestamp):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating real signal: {e}")
            return False
    
    def _get_strategy_strike_preference(self, strategy_name: str) -> str:
        """Get strike preference from strategy configuration."""
        try:
            config_cache = get_cached_config(self.config_path)
            config_loader = config_cache.get_config_loader() if config_cache and hasattr(config_cache, 'get_config_loader') else None
            strategies = config_loader.get_strategy_configs_enhanced() if config_loader and hasattr(config_loader, 'get_strategy_configs_enhanced') else {}
            if strategy_name in strategies:
                strategy_config = strategies[strategy_name]
                strike_preference = strategy_config.get('strike_preference', 'ATM')
                self.logger.debug(f"Strike preference for {strategy_name}: {strike_preference}")
                return strike_preference
            # Default fallback
            return self.config.get('default_strike_selection', 'ATM') if self.config else 'ATM'
        except Exception as e:
            self.logger.error(f"Error getting strike preference: {e}")
            return 'ATM'
    
    def _calculate_cpr_sl_tp(self, symbol: str, strategy_name: str, entry_price: float,
                           pivot_levels: List[float], market_data: pd.DataFrame) -> Optional[Dict[str, Any]]:
        """Calculate CPR-based SL/TP using advanced SL/TP engine."""
        try:
            if market_data.empty:
                self.logger.warning("No market data for CPR SL/TP calculation")
                return None
            
            # Use advanced SL/TP engine for CPR calculation
            sl_tp_result = self.sl_tp_engine.calculate_sl_tp(
                symbol=symbol,
                strategy_name=strategy_name,
                entry_price=entry_price,
                pivot_levels=pivot_levels,
                market_data=market_data
            )
            
            # Convert to format expected by rest of system
            cpr_result = {
                'sl_price': sl_tp_result.stop_loss.price,
                'sl_method': sl_tp_result.stop_loss.calculation_method,
                'tp_levels': [
                    {
                        'price': tp.price,
                        'percentage': tp.percentage,
                        'type': tp.type,
                        'method': tp.calculation_method
                    }
                    for tp in sl_tp_result.take_profits
                ],
                'zone_type': sl_tp_result.zone_type.value,
                'risk_reward_ratio': sl_tp_result.risk_reward_ratio,
                'zone_strength': sl_tp_result.zone_strength,
                'pivot_levels_used': sl_tp_result.pivot_levels_used
            }
            
            self.logger.info(f"CPR SL/TP calculated: SL={cpr_result['sl_price']:.2f}, "
                           f"TPs={len(cpr_result['tp_levels'])}, Zone={cpr_result['zone_type']}")
            
            return cpr_result
            
        except Exception as e:
            self.logger.error(f"Error calculating CPR SL/TP: {e}")
            return None
    
    def _calculate_strategy_sl_tp(self, symbol: str, strategy_name: str, entry_price: float,
                                pivot_levels: List[float], market_data: pd.DataFrame, 
                                current_candle: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Calculate SL/TP using strategy configuration (SL_Method and TP_Method)."""
        try:
            # Get strategy configuration
            config_cache = get_cached_config(self.config_path)
            config_loader = config_cache.get_config_loader()
            strategies = config_loader.get_strategy_configs_enhanced()
            
            if strategy_name not in strategies:
                self.logger.warning(f"Strategy {strategy_name} not found in config")
                return None
            
            strategy_config = strategies[strategy_name]
            
            # Get SL/TP methods from strategy configuration
            sl_method = strategy_config.get('sl_method', 'ATR')
            tp_method = strategy_config.get('tp_method', 'HYBRID')
            buffer_sl_tp = strategy_config.get('buffer_sl_tp', 5)
            max_sl_points = strategy_config.get('max_sl_points', 45)
            
            self.logger.info(f"Using strategy config: SL_Method={sl_method}, TP_Method={tp_method}, "
                           f"Buffer={buffer_sl_tp}, Max_SL={max_sl_points}")
            
            # Prepare market data for SL/TP calculation
            if market_data.empty:
                # Create minimal market data from current candle
                market_data = pd.DataFrame([current_candle])
            
            # For Breakout_Candle method, calculate signal candle range
            signal_candle_range = 0
            if sl_method == 'BREAKOUT_CANDLE' or tp_method == 'BREAKOUT_CANDLE':
                signal_candle_range = current_candle.get('high', 0) - current_candle.get('low', 0)
                self.logger.info(f"Signal candle range: {signal_candle_range:.2f}")
            
            # Create configuration for SL/TP engine
            sl_tp_config = {
                'max_sl_points': max_sl_points,
                'buffer_sl_tp': buffer_sl_tp,
                'breakout_buffer': buffer_sl_tp,
                'signal_candle_range': signal_candle_range,
                'sl_method': sl_method,
                'tp_method': tp_method
            }
            
            # Use advanced SL/TP engine
            sl_tp_result = self.sl_tp_engine.calculate_sl_tp(
                symbol=symbol,
                strategy_name=strategy_name,
                entry_price=entry_price,
                pivot_levels=pivot_levels,
                market_data=market_data,
                custom_config=sl_tp_config
            )
            
            # Convert to format expected by rest of system
            result = {
                'sl_price': sl_tp_result.stop_loss.price,
                'sl_method': sl_tp_result.stop_loss.calculation_method,
                'tp_levels': [
                    {
                        'price': tp.price,
                        'percentage': tp.percentage,
                        'type': tp.type,
                        'method': tp.calculation_method
                    }
                    for tp in sl_tp_result.take_profits
                ],
                'zone_type': sl_tp_result.zone_type.value,
                'risk_reward_ratio': sl_tp_result.risk_reward_ratio,
                'zone_strength': sl_tp_result.zone_strength,
                'pivot_levels_used': sl_tp_result.pivot_levels_used,
                # Required fields for parallel tracking compatibility
                'use_dual_logic': True,
                'primary_exit_method': 'strategy_config',
                'secondary_exit_method': 'strategy_config',
                'options_sl': sl_tp_result.stop_loss.price,
                'options_tp': sl_tp_result.take_profits[0].price if sl_tp_result.take_profits else 0,
                'equity_sl': sl_tp_result.stop_loss.price,
                'equity_tp': sl_tp_result.take_profits[0].price if sl_tp_result.take_profits else 0
            }
            
            self.logger.info(f"Strategy SL/TP calculated: SL={result['sl_price']:.2f}, "
                           f"TPs={len(result['tp_levels'])}, Method={sl_method}/{tp_method}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error calculating strategy SL/TP: {e}")
            return None
    
    def _calculate_basic_sl_tp(self, entry_price: float, signal_type: str,
                             pivot_levels: List[float]) -> Dict[str, Any]:
        """Calculate basic SL/TP using identify_sl_tp module."""
        try:
            # Use existing identify_sl_tp logic as fallback
            zone_type = "Resistance" if signal_type == 'CALL' else "Support"
            
            # Mock SL values (these would come from candle analysis)
            sl1 = 30  # Breakout candle range
            sl2 = 40  # Alternative SL
            max_sl_points = self.config.get('max_sl_points', 50)
            ratio = "1:2"  # Risk-reward ratio
            
            first_sl, first_tp = get_sl_tp(
                zone_type=zone_type,
                ltp=entry_price,
                pivots=pivot_levels,
                sl1=sl1,
                sl2=sl2,
                max_sl_points=max_sl_points,
                ratio=ratio
            )
            
            basic_result = {
                'sl_price': first_sl,
                'sl_method': 'Basic_Pivot',
                'tp_levels': [
                    {
                        'price': first_tp,
                        'percentage': 100.0,
                        'type': 'TP1',
                        'method': 'Basic_Pivot'
                    }
                ],
                'zone_type': zone_type,
                'risk_reward_ratio': 2.0,
                'zone_strength': 1.0,
                'pivot_levels_used': pivot_levels
            }
            
            self.logger.info(f"Basic SL/TP calculated: SL={first_sl:.2f}, TP={first_tp:.2f}")
            
            return basic_result
            
        except Exception as e:
            self.logger.error(f"Error calculating basic SL/TP: {e}")
            return {
                'sl_price': entry_price * 0.98 if signal_type == 'CALL' else entry_price * 1.02,
                'tp_levels': [{'price': entry_price * 1.04 if signal_type == 'CALL' else entry_price * 0.96,
                              'percentage': 100.0, 'type': 'TP1', 'method': 'Default'}],
                'zone_type': 'Unknown',
                'risk_reward_ratio': 2.0
            }
    
    def _wait_for_breakout_confirmation(self, signal_data: Dict[str, Any], entry_point: float,
                                      strike_info: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Wait for breakout confirmation using real 1-minute execution data."""
        try:
            # Import pandas explicitly to avoid scope issues
            import pandas as pd
            if not strike_info:
                self.logger.error("strike_info is None in _wait_for_breakout_confirmation")
                return {}
            signal_type = signal_data.get('signal_type')
            signal_timestamp = signal_data.get('timestamp')
            
            # Get market data (1-minute execution data) from signal_data
            market_data = signal_data.get('market_data')
            if market_data is None or market_data.empty:
                self.logger.warning("No market data available for breakout confirmation")
                return {}
            
            self.logger.info(f"Waiting for {signal_type} breakout confirmation above/below {entry_point}")
            self.logger.info(f"Signal timestamp: {signal_timestamp}")
            self.logger.info(f"Timeout: End of trading day (infinite loop until breakout)")
            
            # Filter market data to only look at candles after signal time
            market_data_after_signal = market_data[
                market_data['timestamp'] > signal_timestamp
            ].sort_values('timestamp')
            
            if market_data_after_signal.empty:
                self.logger.warning("No market data available after signal time")
                return {}
            
            self.logger.info(f"Found {len(market_data_after_signal)} execution candles after signal time")
            
            # Get end of trading day (15:30 IST) for the signal date
            signal_date = signal_timestamp.date() if signal_timestamp is not None else None
            end_of_day = pd.Timestamp(signal_date).replace(hour=15, minute=30, second=0, microsecond=0) if signal_date is not None else None
            
            # Look for breakout until end of trading day (no timeout limit)
            market_data_until_eod = market_data_after_signal[
                market_data_after_signal['timestamp'] <= end_of_day
            ]
            
            breakout_found = False
            breakout_candle = None
            
            for _, candle in market_data_until_eod.iterrows():
                if signal_type == 'CALL':
                    # For CALL: Look for 1-minute high > entry_point
                    self.logger.debug(f"Checking CALL breakout: 1m high {candle['high']:.2f} vs entry_point {entry_point:.2f} at {candle['timestamp']}")
                    if candle['high'] > entry_point:
                        breakout_found = True
                        breakout_candle = candle
                        self.logger.info(f"CALL breakout confirmed: 1m high {candle['high']:.2f} > entry_point {entry_point:.2f} at {candle['timestamp']}")
                        break
                        
                elif signal_type == 'PUT':
                    # For PUT: Look for 1-minute low < entry_point
                    self.logger.debug(f"Checking PUT breakout: 1m low {candle['low']:.2f} vs entry_point {entry_point:.2f} at {candle['timestamp']}")
                    if candle['low'] < entry_point:
                        breakout_found = True
                        breakout_candle = candle
                        self.logger.info(f"PUT breakout confirmed: 1m low {candle['low']:.2f} < entry_point {entry_point:.2f} at {candle['timestamp']}")
                        break
            
            if not breakout_found:
                self.logger.warning(f"No {signal_type} breakout found for entire trading day - marking as breakeven")
                
                # Handle breakeven day - no breakout occurred
                breakeven_result = {
                    'entry_time': end_of_day,
                    'entry_price': entry_point,
                    'strike_entry_price': 0.0,  # No entry occurred
                    'breakout_confirmed': False,
                    'breakout_delay_minutes': 0.0,
                    'exit_reason': 'NO_BREAKOUT_BREAKEVEN',
                    'breakout_candle': None,
                    'is_breakeven': True
                }
                
                self.logger.info(f"Breakeven day result: No breakout for {signal_type} signal")
                return breakeven_result
            
            # Get strike price at breakout time
            breakout_candle_ts = None
            if breakout_candle is not None:
                if isinstance(breakout_candle, pd.Series) and 'timestamp' in breakout_candle.index:
                    breakout_candle_ts = breakout_candle['timestamp']
                elif isinstance(breakout_candle, dict) and 'timestamp' in breakout_candle:
                    breakout_candle_ts = breakout_candle['timestamp']
            if breakout_candle_ts is None:
                self.logger.error("breakout_candle or its timestamp is None in _wait_for_breakout_confirmation")
                return {}
            # Ensure breakout_candle_ts is a datetime
            import pandas as pd
            if isinstance(breakout_candle_ts, pd.DatetimeIndex):
                if len(breakout_candle_ts) > 0:
                    breakout_candle_ts = breakout_candle_ts[0]
                else:
                    self.logger.error("breakout_candle_ts is an empty DatetimeIndex")
                    return {}
            if not isinstance(breakout_candle_ts, (pd.Timestamp, datetime)):
                try:
                    breakout_candle_ts = pd.to_datetime(breakout_candle_ts)
                except Exception as e:
                    self.logger.error(f"Could not convert breakout_candle_ts to datetime: {e}")
                    return {}
            if isinstance(breakout_candle_ts, pd.Timestamp):
                breakout_candle_ts = breakout_candle_ts.to_pydatetime()
            if not isinstance(breakout_candle_ts, datetime):
                self.logger.error(f"breakout_candle_ts is not a datetime after conversion: {type(breakout_candle_ts)}")
                return {}
            strike_entry_price = self._get_strike_price_at_time(
                strike_info, breakout_candle_ts
            )
            
            # Calculate breakout delay
            breakout_delay_minutes = 0
            if breakout_candle is not None and signal_timestamp is not None:
                if isinstance(breakout_candle, pd.Series) and 'timestamp' in breakout_candle.index:
                    breakout_delay_minutes = (breakout_candle['timestamp'] - signal_timestamp).total_seconds() / 60
                elif isinstance(breakout_candle, dict) and 'timestamp' in breakout_candle:
                    breakout_delay_minutes = (breakout_candle['timestamp'] - signal_timestamp).total_seconds() / 60
            breakout_result = {
                'entry_time': breakout_candle_ts,
                'entry_price': entry_point,
                'strike_entry_price': strike_entry_price,
                'breakout_confirmed': True,
                'breakout_delay_minutes': breakout_delay_minutes,
                'breakout_candle': {
                    'open': breakout_candle['open'] if breakout_candle is not None and ((isinstance(breakout_candle, pd.Series) and 'open' in breakout_candle.index) or (isinstance(breakout_candle, dict) and 'open' in breakout_candle)) else None,
                    'high': breakout_candle['high'] if breakout_candle is not None and ((isinstance(breakout_candle, pd.Series) and 'high' in breakout_candle.index) or (isinstance(breakout_candle, dict) and 'high' in breakout_candle)) else None,
                    'low': breakout_candle['low'] if breakout_candle is not None and ((isinstance(breakout_candle, pd.Series) and 'low' in breakout_candle.index) or (isinstance(breakout_candle, dict) and 'low' in breakout_candle)) else None,
                    'close': breakout_candle['close'] if breakout_candle is not None and ((isinstance(breakout_candle, pd.Series) and 'close' in breakout_candle.index) or (isinstance(breakout_candle, dict) and 'close' in breakout_candle)) else None,
                    'timestamp': breakout_candle_ts
                }
            }
            
            self.logger.info(f"[ENTRY] CONFIRMED: {breakout_candle_ts} @ Premium: {strike_entry_price:.2f} (Breakout after {breakout_delay_minutes:.1f} minutes)")
            
            return breakout_result
            
        except Exception as e:
            self.logger.error(f"Error in _wait_for_breakout_confirmation: {e}")
            return {}
    
    def _create_trade_waiting_for_breakout(self, signal_data: Dict[str, Any], 
                                          strike_info: Dict[str, Any], 
                                          sl_tp_result: Dict[str, Any], 
                                          entry_point: float, 
                                          position_size: int = 1) -> Dict[str, Any]:
        """Create a trade with WAITING_FOR_BREAKOUT status."""
        try:
            if not strike_info:
                self.logger.error("strike_info is None in _create_trade_waiting_for_breakout")
                return None
            
            # Generate meaningful trade ID with symbol, strategy, and signal type
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = signal_data.get('symbol', 'NIFTY')
            strategy_name = signal_data.get('strategy_name', 'Unknown')
            signal_type = signal_data.get('signal_type', 'CALL')
            trade_id = f"{symbol}_{strategy_name}_{signal_type}_{timestamp_str}"
            
            self.logger.info(f"[TRADE_ID] Generated meaningful trade ID: {trade_id}")
            
            # Debug: Log the position size passed to this method
            lot_size = self.options_config.get_lot_size(signal_data.get('symbol', 'NIFTY'))
            self.logger.info(f"[DEBUG] _create_trade_waiting_for_breakout: position_size={position_size}, lot_size={lot_size}")
            
            # Debug: Log the exit rule assignment
            exit_rule = signal_data.get('exit_rule')
            self.logger.info(f"[DEBUG] Exit rule for trade {trade_id}: {exit_rule}")
            
            # Create trade structure
            trade_data = {
                'trade_id': trade_id,
                'symbol': signal_data.get('symbol', 'NIFTY'),
                'strategy_name': signal_data.get('strategy_name', 'Unknown'),
                'signal_type': signal_data.get('signal_type', 'CALL'),
                'option_type': signal_data.get('signal_type', 'CALL'),
                'strike': strike_info['strike'],
                'strike_preference': signal_data.get('strike_preference', 'ATM'),
                'entry_point': entry_point,
                'entry_time': signal_data.get('timestamp'),
                'entry_premium': 0.0,  # Will be updated after breakout
                'status': 'WAITING_FOR_BREAKOUT',
                'breakout_confirmed': False,
                'breakout_delay_minutes': 0.0,
                'position_size': position_size,
                'lot_size': self.options_config.get_lot_size(signal_data.get('symbol', 'NIFTY')),
                'commission': self.config.get('commission', 20),
                'slippage': self.config.get('slippage', 0.1),
                'sl_tp_levels': sl_tp_result,
                'exit_reason': None,
                'exit_time': None,
                'exit_premium': None,
                'pnl': 0.0,
                # Add exit rule from strategy configuration
                'exit_rule': "EMA_Signal_Long_Exit",
                # Add signal candle close price for Trade Summary
                'signal_close': signal_data.get('current_candle', {}).get('close', 0),
                'underlying_price': signal_data.get('underlying_price', signal_data.get('current_candle', {}).get('close', 0))
            }
            
            # Add to active trades
            self.active_trades[trade_id] = trade_data
            
            self.logger.info(f"Created trade {trade_id} with WAITING_FOR_BREAKOUT status")
            
            return trade_data
            
        except Exception as e:
            self.logger.error(f"Error creating trade waiting for breakout: {e}")
            return None
    
    def _get_strike_price_at_time(self, strike_info: Dict[str, Any], 
                                 target_time: datetime) -> float:
        """Get strike price from database at specific time."""
        try:
            symbol = strike_info.get('symbol', 'NIFTY')
            strike = strike_info.get('strike')
            option_type = strike_info.get('option_type')
            expiry_date = strike_info.get('expiry_date')
            
            # Convert expiry_date to string format if it's a pandas Timestamp
            if isinstance(expiry_date, pd.Timestamp):
                expiry_date = expiry_date.strftime('%Y-%m-%d')
            elif hasattr(expiry_date, 'date'):
                expiry_date = expiry_date.strftime('%Y-%m-%d') if hasattr(expiry_date, 'strftime') else str(expiry_date.date())
            elif expiry_date is not None and not isinstance(expiry_date, str):
                expiry_date = str(expiry_date)
            
            # Debug log all query parameters
            self.logger.debug(f"Querying option premium: symbol={symbol}, strike={strike}, option_type={option_type}, expiry_date={expiry_date}, target_time={target_time}")
            # Type checks for DB query
            if symbol is None or strike is None or option_type is None or expiry_date is None:
                self.logger.error(f"Missing required fields for DB query: symbol={symbol}, strike={strike}, option_type={option_type}, expiry_date={expiry_date}")
                return 0
            if not isinstance(strike, int) or not isinstance(option_type, str) or not isinstance(expiry_date, str):
                self.logger.error(f"Invalid types for DB query: symbol={symbol}, strike={strike} ({type(strike)}), option_type={option_type} ({type(option_type)}), expiry_date={expiry_date} ({type(expiry_date)})")
                return 0
            result = self.options_db.get_option_premium_at_time(
                symbol=symbol,
                strike=strike,
                option_type=option_type,
                timestamp=target_time,
                expiry_date=expiry_date
            )
            if result and 'premium' in result and result['premium'] is not None:
                self.logger.debug(f"[DB] {symbol} {option_type} {strike} {expiry_date} premium at {target_time}: {result['premium']}")
                return float(result['premium'])
            else:
                self.logger.warning(f"No premium found in DB for {symbol} {option_type} {strike} {expiry_date} at {target_time}")
                return 0
        except Exception as e:
            self.logger.error(f"Error getting strike price at time from DB: {e}")
            return 0
    
    def _execute_real_money_trade(self, signal_data: Dict[str, Any], strike_info: Dict[str, Any],
                                sl_tp_result: Dict[str, Any], real_entry_data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute real money trade with actual strike prices."""
        try:
            # Generate meaningful trade ID with symbol, strategy, and signal type
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = signal_data.get('symbol', 'NIFTY')
            strategy_name = signal_data.get('strategy_name', 'Unknown')
            signal_type = signal_data.get('signal_type', 'CALL')
            trade_id = f"{symbol}_{strategy_name}_{signal_type}_{timestamp_str}"
            
            self.logger.info(f"[TRADE_ID] Generated meaningful trade ID: {trade_id}")
            
            # Calculate position size
            position_size = self._calculate_real_position_size(strike_info, sl_tp_result)
            
            # Calculate real money parameters
            entry_premium = real_entry_data.get('strike_entry_price', 0)
            commission = self.config.get('commission_per_trade', 25)
            lot_size = self.options_config.get_lot_size(signal_data.get('symbol', 'NIFTY'))
            
            # Debug: Log the calculated values
            self.logger.info(f"[DEBUG] Position size calculation: position_size={position_size}, lot_size={lot_size}")
            
            trade_value = entry_premium * lot_size * position_size
            slippage_cost = self.config.get('slippage_points', 2) * lot_size * position_size
            total_cost = trade_value + commission + slippage_cost
            
            # Debug: Log values before creating trade record
            self.logger.info(f"[DEBUG] Before creating trade record: position_size={position_size}, lot_size={lot_size}")
            
            # Create comprehensive trade record
            trade_record = {
                'trade_id': trade_id,
                'timestamp': real_entry_data.get('entry_time'),
                'symbol': signal_data.get('symbol'),
                'strategy_name': signal_data.get('strategy_name'),
                'signal_type': signal_data.get('signal_type'),
                'underlying_entry_price': real_entry_data.get('entry_price'),
                'strike': strike_info.get('strike'),
                'option_type': strike_info.get('option_type'),
                'expiry_date': strike_info.get('expiry_date'),
                'entry_premium': entry_premium,
                'real_entry_premium': entry_premium,  # Keep for backward compatibility
                'position_size': position_size,
                'lot_size': lot_size,
                'trade_value': trade_value,
                'commission': commission,
                'slippage_cost': slippage_cost,
                'total_cost': total_cost,
                'cpr_sl_tp_levels': sl_tp_result,
                'entry_method': 'breakout_confirmation' if real_entry_data.get('breakout_confirmed') else 'immediate',
                'breakout_delay': real_entry_data.get('breakout_delay_minutes', 0),
                'greeks': {
                    'delta': strike_info.get('delta'),
                    'iv': strike_info.get('iv'),
                    'intrinsic_value': strike_info.get('intrinsic_value'),
                    'time_value': strike_info.get('time_value'),
                    'dte': strike_info.get('dte'),
                },
                'status': TradeStatus.OPEN.value,
                'entry_time': real_entry_data.get('entry_time'),
                'exit_time': None,
                'exit_premium': None,
                'real_pnl': 0.0,
                'exit_reason': None,
                'zone_type': sl_tp_result.get('zone_type'),
                'risk_reward_ratio': sl_tp_result.get('risk_reward_ratio'),
                # Add exit rule from strategy configuration
                'exit_rule': signal_data.get('exit_rule'),
                # Add signal candle close price for Trade Summary
                'signal_close': signal_data.get('current_candle', {}).get('close', 0),
                'underlying_price': signal_data.get('underlying_price', signal_data.get('current_candle', {}).get('close', 0)),
                'tracking_data': {
                    'equity_updates': 0,
                    'options_updates': 0,
                    'last_update': datetime.now(),
                    'max_profit': 0.0,
                    'max_loss': 0.0,
                    'equity_high': real_entry_data.get('entry_price', 0),  # Initialize with entry price
                    'equity_low': real_entry_data.get('entry_price', 0),   # Initialize with entry price
                    'options_high': entry_premium,  # Initialize with entry premium
                    'options_low': entry_premium,   # Initialize with entry premium
                    'updates_count': 0
                }
            }
            
            # Add to active trades
            self.active_trades[trade_id] = trade_record
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            
            # Increment daily entry counter
            self._increment_daily_entries()
            
            self.logger.info(f"Real money trade executed: {trade_id} - Strike {strike_info.get('strike')} "
                           f"@ {entry_premium:.2f} (Value: {trade_value:.2f})")
            
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Error executing real money trade: {e}")
            return None
    
    def _calculate_real_position_size(self, strike_info: Dict[str, Any], 
                                    sl_tp_result: Dict[str, Any]) -> int:
        """Calculate position size - fixed at 1 lot for consistent P&L calculation."""
        try:
            # Fixed position size: 1 lot (75 units for NIFTY)
            # P&L = (Exit Premium - Entry Premium) × 75 × 1 = (Exit Premium - Entry Premium) × 75
            position_size = 1
            
            entry_premium = strike_info.get('premium', 100)
            lot_size = self.options_config.get_lot_size(strike_info.get('symbol', 'NIFTY'))
            
            self.logger.info(f"[DEBUG] _calculate_real_position_size: position_size={position_size}, lot_size={lot_size}, premium={entry_premium:.2f}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating real position size: {e}")
            return 1
    
    def _start_real_parallel_tracking(self, trade_record: Dict[str, Any]) -> None:
        """Start real parallel tracking for equity and option strike."""
        try:
            # Skip parallel tracking for backtesting - not needed for core functionality
            return
            trade_id = trade_record['trade_id']
            
            # Initialize real tracking with actual strike data
            self.equity_charts[trade_id] = {
                'timestamps': [],
                'prices': [],
                'high': [],
                'low': [],
                'volume': [],
                'sl_levels': [],
                'tp_levels': [],
                'real_data': True
            }
            
            self.options_charts[trade_id] = {
                'timestamps': [],
                'strike_premiums': [],  # Real strike prices from database
                'strike': trade_record['strike'],
                'option_type': trade_record['option_type'],
                'expiry_date': trade_record['expiry_date'],
                'iv': [],
                'delta': [],
                'intrinsic_values': [],
                'time_values': [],
                'sl_levels': [],
                'tp_levels': [],
                'real_data': True
            }
            
            self.logger.debug(f"Started real parallel tracking for {trade_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting real parallel tracking: {e}")

    def _validate_signal(self, signal_data: Dict[str, Any]) -> bool:
        """Validate trade signal data."""
        try:
            required_fields = ['signal_type', 'underlying_price', 'timestamp', 'symbol']
            for field in required_fields:
                if field not in signal_data:
                    self.logger.warning(f"Missing required field: {field}")
                    return False
            
            # Validate signal type
            if signal_data['signal_type'] not in ['CALL', 'PUT']:
                self.logger.warning(f"Invalid signal type: {signal_data['signal_type']}")
                return False
            
            # Validate underlying price
            if signal_data['underlying_price'] <= 0:
                self.logger.warning(f"Invalid underlying price: {signal_data['underlying_price']}")
                return False
            
            # Check if we have too many open positions
            if len(self.active_trades) >= 10:  # Max 10 concurrent trades
                self.logger.warning("Maximum concurrent trades reached")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error validating signal: {e}")
            return False
    
    def _calculate_position_size(self, underlying_price: float, strike_info: Dict[str, Any],
                               strategy_name: str) -> int:
        """Calculate position size based on risk management rules."""
        try:
            # Get risk parameters
            risk_per_trade = self.config.get('risk_per_trade', 0.02)
            max_position_size = self.config.get('max_position_size', 10000)
            
            # Calculate position size based on premium and risk
            premium = strike_info.get('premium', 100)
            lot_size = self.options_config.get_lot_size(strike_info.get('symbol', 'NIFTY'))
            
            # Risk-based position sizing
            capital_at_risk = max_position_size * risk_per_trade
            max_lots = int(capital_at_risk / (premium * lot_size))
            
            # Ensure minimum 1 lot, maximum based on risk
            position_size = max(1, min(max_lots, 5))  # Max 5 lots per trade
            
            self.logger.debug(f"Calculated position size: {position_size} lots "
                            f"(Premium: {premium}, Risk: {capital_at_risk:.2f})")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"Error calculating position size: {e}")
            return 1  # Default to 1 lot
    
    def _calculate_entry_parameters(self, strike_info: Dict[str, Any], position_size: int,
                                  underlying_price: float, equity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate entry parameters for the trade."""
        try:
            premium = strike_info.get('premium', 100)
            strike = strike_info.get('strike', 0)
            option_type = strike_info.get('option_type', 'CALL')
            lot_size = self.options_config.get_lot_size(strike_info.get('symbol', 'NIFTY'))
            
            # Calculate trade value
            trade_value = premium * lot_size * position_size
            
            # Calculate commission
            commission = self.config.get('commission_per_trade', 25)
            
            # Calculate slippage
            slippage_points = self.config.get('slippage_points', 2)
            slippage_cost = slippage_points * lot_size * position_size
            
            # Calculate SL/TP levels for both equity and options
            sl_tp_levels = self._calculate_dual_sl_tp_levels(
                strike_info, underlying_price, equity_data
            )
            
            entry_params = {
                'premium': premium,
                'strike': strike,
                'option_type': option_type,
                'position_size': position_size,
                'lot_size': lot_size,
                'trade_value': trade_value,
                'commission': commission,
                'slippage_cost': slippage_cost,
                'total_cost': trade_value + commission + slippage_cost,
                'cpr_sl_tp_levels': sl_tp_levels,
                'intrinsic_value': strike_info.get('intrinsic_value', 0),
                'time_value': strike_info.get('time_value', 0),
                'iv': strike_info.get('iv', 0),
                'delta': strike_info.get('delta', 0),
                'dte': strike_info.get('dte', 0),
            }
            
            self.logger.debug(f"Entry parameters calculated: Trade value {trade_value:.2f}, "
                            f"Total cost {entry_params['total_cost']:.2f}")
            
            return entry_params
            
        except Exception as e:
            self.logger.error(f"Error calculating entry parameters: {e}")
            return {}
    
    def _calculate_dual_sl_tp_levels(self, strike_info: Dict[str, Any], underlying_price: float,
                                   equity_data: Dict[str, Any]) -> Dict[str, Any]:
        """Calculate SL/TP levels for both equity and options."""
        try:
            option_type = strike_info.get('option_type', 'CALL')
            premium = strike_info.get('premium', 100)
            strike = strike_info.get('strike', 0)
            
            # Equity-based SL/TP
            equity_sl_buffer = self.config.get('equity_sl_buffer', 5)
            equity_tp_buffer = self.config.get('equity_tp_buffer', 15)
            
            if option_type == 'CALL':
                equity_sl = underlying_price - equity_sl_buffer
                equity_tp = underlying_price + equity_tp_buffer
            else:  # PUT
                equity_sl = underlying_price + equity_sl_buffer
                equity_tp = underlying_price - equity_tp_buffer
            
            # Options-based SL/TP
            options_sl_buffer = self.config.get('options_sl_buffer', 10)
            options_tp_multiplier = self.config.get('options_tp_multiplier', 2.0)
            
            options_sl = premium * (1 - options_sl_buffer / 100)  # 10% stop loss
            options_tp = premium * options_tp_multiplier  # 2x target profit
            
            sl_tp_levels = {
                'equity_sl': equity_sl,
                'equity_tp': equity_tp,
                'options_sl': options_sl,
                'options_tp': options_tp,
                'use_dual_logic': self.config.get('enable_dual_sl_tp', True),
                'primary_exit_method': 'options',  # Primary exit based on options prices
                'secondary_exit_method': 'equity',  # Secondary exit based on equity prices
            }
            
            self.logger.debug(f"Dual SL/TP levels: Equity SL {equity_sl:.2f}, TP {equity_tp:.2f}, "
                            f"Options SL {options_sl:.2f}, TP {options_tp:.2f}")
            
            return sl_tp_levels
            
        except Exception as e:
            self.logger.error(f"Error calculating dual SL/TP levels: {e}")
            return {}
    
    def _execute_options_trade(self, signal_data: Dict[str, Any], strike_info: Dict[str, Any],
                             entry_params: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Execute the options trade."""
        try:
            # Generate meaningful trade ID with symbol, strategy, and signal type
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            symbol = signal_data.get('symbol', 'NIFTY')
            strategy_name = signal_data.get('strategy_name', 'Unknown')
            signal_type = signal_data.get('signal_type', 'CALL')
            trade_id = f"{symbol}_{strategy_name}_{signal_type}_{timestamp_str}"
            
            self.logger.info(f"[TRADE_ID] Generated meaningful trade ID: {trade_id}")
            
            # Create trade record
            trade_record = {
                'trade_id': trade_id,
                'timestamp': signal_data.get('timestamp', datetime.now()),
                'symbol': signal_data.get('symbol', 'NIFTY'),
                'strategy_name': signal_data.get('strategy_name', 'default'),
                'signal_type': signal_data.get('signal_type'),
                'underlying_price': signal_data.get('underlying_price'),
                'strike': strike_info.get('strike'),
                'option_type': strike_info.get('option_type'),
                'expiry_date': strike_info.get('expiry_date'),
                'entry_premium': entry_params.get('premium'),
                'position_size': entry_params.get('position_size'),
                'lot_size': entry_params.get('lot_size'),
                'trade_value': entry_params.get('trade_value'),
                'commission': entry_params.get('commission'),
                'slippage_cost': entry_params.get('slippage_cost'),
                'total_cost': entry_params.get('total_cost'),
                'cpr_sl_tp_levels': entry_params.get('sl_tp_levels'),
                'greeks': {
                    'delta': strike_info.get('delta'),
                    'iv': strike_info.get('iv'),
                    'intrinsic_value': strike_info.get('intrinsic_value'),
                    'time_value': strike_info.get('time_value'),
                    'dte': strike_info.get('dte'),
                },
                'status': TradeStatus.OPEN.value,
                'entry_time': datetime.now(),
                'exit_time': None,
                'exit_premium': None,
                'pnl': 0.0,
                'exit_reason': None,
                'equity_data_at_entry': signal_data.get('equity_data', {}),
                # Add exit rule from strategy configuration
                'exit_rule': signal_data.get('exit_rule'),
                # Add signal candle close price for Trade Summary
                'signal_close': signal_data.get('current_candle', {}).get('close', 0),
                'max_profit': 0.0,
                'max_loss': 0.0,
                'tracking_data': {
                    'equity_high': signal_data.get('underlying_price'),
                    'equity_low': signal_data.get('underlying_price'),
                    'options_high': entry_params.get('premium'),
                    'options_low': entry_params.get('premium'),
                    'updates_count': 0,
                    'last_update': datetime.now(),
                }
            }
            
            # Add to active trades
            self.active_trades[trade_id] = trade_record
            
            # Update performance metrics
            self.performance_metrics['total_trades'] += 1
            
            self.logger.info(f"Options trade executed: {trade_id} - {strike_info.get('option_type')} "
                           f"{strike_info.get('strike')} @ {entry_params.get('premium'):.2f}")
            
            return trade_record
            
        except Exception as e:
            self.logger.error(f"Error executing options trade: {e}")
            return None
    
    def _start_parallel_tracking(self, trade_record: Dict[str, Any]) -> None:
        """Start parallel tracking for equity and options data."""
        try:
            # Skip parallel tracking for backtesting - not needed for core functionality
            return
            trade_id = trade_record['trade_id']
            
            # Initialize tracking data structures
            self.equity_charts[trade_id] = {
                'timestamps': [],
                'prices': [],
                'high': [],
                'low': [],
                'volume': [],
                'sl_levels': [],
                'tp_levels': [],
            }
            
            self.options_charts[trade_id] = {
                'timestamps': [],
                'premiums': [],
                'iv': [],
                'delta': [],
                'intrinsic_values': [],
                'time_values': [],
                'sl_levels': [],
                'tp_levels': [],
            }
            
            self.logger.debug(f"Started parallel tracking for trade {trade_id}")
            
        except Exception as e:
            self.logger.error(f"Error starting parallel tracking: {e}")
    
    def update_parallel_tracking(self, trade_id: str, equity_data: Dict[str, Any],
                               options_data: Optional[Dict[str, Any]] = None, 
                               execution_data: Optional[Dict[str, Any]] = None) -> None:
        """Update parallel tracking data for active trade."""
        try:
            # Skip parallel tracking for backtesting - but still check exit conditions
            if trade_id not in self.active_trades:
                return
                
            # Check for exit conditions (important for backtesting!)
            self._check_exit_conditions(trade_id, equity_data, options_data, execution_data)
            return
            
            # Rest of parallel tracking code disabled for backtesting
            if trade_id not in self.active_trades:
                return
            
            timestamp = datetime.now()
            
            # Update equity chart
            if trade_id in self.equity_charts:
                equity_chart = self.equity_charts[trade_id]
                equity_chart['timestamps'].append(timestamp)
                equity_chart['prices'].append(equity_data.get('close', 0))
                equity_chart['high'].append(equity_data.get('high', 0))
                equity_chart['low'].append(equity_data.get('low', 0))
                equity_chart['volume'].append(equity_data.get('volume', 0))
                
                # Add SL/TP levels
                sl_tp_levels = self.active_trades[trade_id]['cpr_sl_tp_levels']
                equity_chart['sl_levels'].append(sl_tp_levels.get('equity_sl', 0))
                equity_chart['tp_levels'].append(sl_tp_levels.get('equity_tp', 0))
            
            # Update options chart if data available
            if options_data and trade_id in self.options_charts:
                options_chart = self.options_charts[trade_id]
                options_chart['timestamps'].append(timestamp)
                options_chart['premiums'].append(options_data.get('premium', 0))
                options_chart['iv'].append(options_data.get('iv', 0))
                options_chart['delta'].append(options_data.get('delta', 0))
                options_chart['intrinsic_values'].append(options_data.get('intrinsic_value', 0))
                options_chart['time_values'].append(options_data.get('time_value', 0))
                
                # Add SL/TP levels
                sl_tp_levels = self.active_trades[trade_id]['cpr_sl_tp_levels']
                options_chart['sl_levels'].append(sl_tp_levels.get('options_sl', 0))
                options_chart['tp_levels'].append(sl_tp_levels.get('options_tp', 0))
            
            # Update trade tracking data
            trade_record = self.active_trades[trade_id]
            tracking_data = trade_record['tracking_data']
            
            current_price = equity_data.get('close', 0)
            tracking_data['equity_high'] = max(tracking_data['equity_high'], current_price)
            tracking_data['equity_low'] = min(tracking_data['equity_low'], current_price)
            
            if options_data:
                current_premium = options_data.get('premium', 0)
                tracking_data['options_high'] = max(tracking_data['options_high'], current_premium)
                tracking_data['options_low'] = min(tracking_data['options_low'], current_premium)
            
            tracking_data['updates_count'] += 1
            tracking_data['last_update'] = timestamp
            
            # Check for exit conditions
            self._check_exit_conditions(trade_id, equity_data, options_data)
            
        except Exception as e:
            self.logger.error(f"Error updating parallel tracking: {e}")
    
    def _get_real_exit_premium(self, trade_record: Dict[str, Any], exit_timestamp: datetime, exit_reason: str = None) -> tuple[float, datetime]:
        """Calculate real exit premium using database query - NO MOCK DATA. Returns (premium, adjusted_timestamp)."""
        try:
            # Extract trade details
            strike = trade_record.get('strike', 0)
            option_type = trade_record.get('option_type', 'CALL')
            symbol = trade_record.get('symbol', 'NIFTY')
            expiry_date = trade_record.get('expiry_date', '2025-06-12')  # Default weekly expiry
            
            if not strike:
                self.logger.debug(f"Missing strike ({strike}) for exit premium calculation")
                return 0
            
            # For 5-minute exits, adjust timing to next 5-minute boundary for premium fetching
            adjusted_exit_time = self._adjust_5m_exit_time(exit_timestamp, trade_record, exit_reason)
            
            # Query options database directly - same as entry premium logic
            if hasattr(self, 'options_db') and self.options_db:
                try:
                    # Use the real database query method
                    premium_data = self.options_db.get_option_premium_at_time(
                        symbol=symbol,
                        strike=strike,
                        option_type=option_type,
                        timestamp=adjusted_exit_time,
                        expiry_date=expiry_date
                    )
                    
                    if premium_data and 'premium' in premium_data:
                        exit_premium = premium_data['premium']
                        self.logger.info(f"[OK] Real exit premium from database: {exit_premium:.2f} for strike {strike} {option_type} at {adjusted_exit_time}")
                        return exit_premium, adjusted_exit_time
                    else:
                        self.logger.warning(f"[ERROR] No premium data found in database for strike {strike} {option_type} at {adjusted_exit_time}")
                        
                except Exception as e:
                    self.logger.error(f"[ERROR] Error querying options database for exit premium: {e}")
            else:
                self.logger.warning(f"[ERROR] Options database not available for exit premium calculation")
            
            # NO FALLBACK - Return 0 if database query fails
            self.logger.error(f"[ERROR] Failed to get real exit premium from database - no mock data used")
            return 0, exit_timestamp
            
        except Exception as e:
            self.logger.error(f"Error calculating real exit premium: {e}")
            return 0, exit_timestamp
    
    def _adjust_5m_exit_time(self, exit_timestamp: datetime, trade_record: Dict[str, Any], exit_reason: str = None) -> datetime:
        """Adjust exit time for 5-minute exits to next 5-minute boundary for premium fetching."""
        try:
            # Check if this is a 5-minute indicator exit using the passed exit_reason
            # Fixed: Check for any 5M_ pattern instead of hardcoded '5M_INDICATOR_EXIT'
            if exit_reason and exit_reason.startswith('5M_'):
                # For 5-minute exits, fetch premium from next 5-minute candle
                # Example: Exit at 12:15 → fetch premium at 12:20
                minutes = exit_timestamp.minute
                next_5m_minute = ((minutes // 5) + 1) * 5
                
                if next_5m_minute >= 60:
                    # Roll over to next hour
                    adjusted_time = exit_timestamp.replace(hour=exit_timestamp.hour + 1, minute=0, second=0, microsecond=0)
                else:
                    adjusted_time = exit_timestamp.replace(minute=next_5m_minute, second=0, microsecond=0)
                
                self.logger.info(f"[5M_TIMING] Exit timing adjustment for {exit_reason}: {exit_timestamp} -> {adjusted_time}")
                return adjusted_time
            
            # For 1-minute exits and other exits, use exact time
            self.logger.debug(f"[1M_TIMING] Using exact exit time for {exit_reason}: {exit_timestamp}")
            return exit_timestamp
            
        except Exception as e:
            self.logger.debug(f"Error adjusting 5m exit time: {e}")
            return exit_timestamp
    
    def _check_exit_conditions(self, trade_id: str, equity_data: Dict[str, Any],
                              options_data: Optional[Dict[str, Any]] = None,
                              execution_data: Optional[Dict[str, Any]] = None) -> None:
        """Check exit conditions for active trade with two-tier monitoring."""
        try:
            if trade_id not in self.active_trades:
                return
            
            trade_record = self.active_trades[trade_id]
            sl_tp_levels = trade_record.get('cpr_sl_tp_levels', {})
            
            # CRITICAL FIX: Only check exit conditions AFTER actual trade entry time
            current_timestamp = equity_data.get('timestamp') or equity_data.get('execution_timestamp', datetime.now())
            actual_entry_time = trade_record.get('entry_time')
            
            if actual_entry_time is None:
                self.logger.debug(f"Trade {trade_id} has no entry_time yet, skipping exit checks")
                return
            
            # Convert timestamps for comparison if needed
            if isinstance(actual_entry_time, str):
                actual_entry_time = datetime.fromisoformat(actual_entry_time.replace('Z', '+00:00'))
            if isinstance(current_timestamp, str):
                current_timestamp = datetime.fromisoformat(current_timestamp.replace('Z', '+00:00'))
            
            # Skip exit checks if current time is before or equal to actual entry time
            if current_timestamp <= actual_entry_time:
                self.logger.debug(f"Skipping exit check for trade {trade_id}: current time {current_timestamp} <= entry time {actual_entry_time}")
                return
            
            # Development logging with detailed condition values
            self.logger.info(f"[EXIT_CHECK] Trade {trade_id} - Status: {trade_record.get('status')} at {current_timestamp}")
            
            current_equity_price = equity_data.get('close', 0)
            current_options_premium = options_data.get('premium', 0) if options_data else 0
            
            # Development logging: Show current market data
            self.logger.info(f"[EXIT_CHECK] Current equity price: {current_equity_price:.2f}, Entry price: {trade_record.get('underlying_price', 0):.2f}")
            
            # CRITICAL FIX: Calculate real exit premium using option chain data
            # Update trade record with current underlying price for premium calculation
            trade_record['current_underlying_price'] = current_equity_price
            exit_premium, _ = self._get_real_exit_premium(trade_record, current_timestamp, None)  # Will be updated with actual exit_reason below
            if exit_premium == 0:
                exit_premium = current_options_premium  # Fallback to passed premium
            
            # Development logging: Show premium calculation
            self.logger.info(f"[EXIT_CHECK] Exit premium calculated: {exit_premium:.2f} (from database: {exit_premium > 0})")
            
            # Initialize exit tracking variables
            exit_triggered = False
            exit_reason = None
            
            # CONFIG-DRIVEN EXIT MONITORING SYSTEM
            
            # Check config-driven exit conditions - SIMPLIFIED approach
            exit_result = self._check_config_driven_exit_conditions(trade_record, equity_data, current_timestamp, execution_data)
            
            if exit_result['exit_triggered']:
                exit_triggered = True
                exit_reason = exit_result['exit_reason']
                
                # SIMPLIFIED: Always use current timestamp for premium calculation
                exit_premium, adjusted_exit_time = self._get_real_exit_premium(trade_record, current_timestamp, exit_reason)
                if exit_premium == 0:
                    exit_premium = current_options_premium  # Fallback to passed premium
                    adjusted_exit_time = current_timestamp  # Use current timestamp as fallback
                
                self.logger.info(f"Config-driven exit triggered for trade {trade_id}: {exit_reason}")
                self.logger.info(f"Matched conditions: {exit_result['matched_conditions']}")
            
            # 3. Check dual SL/TP logic (existing logic) - ONLY if config-driven exit didn't trigger
            elif sl_tp_levels.get('use_dual_logic', True) and sl_tp_levels:
                # Primary exit method (options)
                if sl_tp_levels.get('primary_exit_method') == 'options' and current_options_premium > 0:
                    if current_options_premium <= sl_tp_levels.get('options_sl', 0):
                        exit_triggered = True
                        exit_reason = 'OPTIONS_SL'
                    elif current_options_premium >= sl_tp_levels.get('options_tp', 0):
                        exit_triggered = True
                        exit_reason = 'OPTIONS_TP'
                
                # Secondary exit method (equity)
                elif sl_tp_levels.get('secondary_exit_method') == 'equity':
                    option_type = trade_record['option_type']
                    if option_type == 'CALL':
                        if current_equity_price <= sl_tp_levels.get('equity_sl', 0):
                            exit_triggered = True
                            exit_reason = 'EQUITY_SL'
                        elif current_equity_price >= sl_tp_levels.get('equity_tp', 0):
                            exit_triggered = True
                            exit_reason = 'EQUITY_TP'
                    else:  # PUT
                        if current_equity_price >= sl_tp_levels.get('equity_sl', 0):
                            exit_triggered = True
                            exit_reason = 'EQUITY_SL'
                        elif current_equity_price <= sl_tp_levels.get('equity_tp', 0):
                            exit_triggered = True
                            exit_reason = 'EQUITY_TP'
                
            # No time-based exit - using only 1m and 5m exit conditions implemented above
            
            # Always check for expiry regardless of logic type
            if trade_record.get('greeks', {}).get('dte', 1) <= 0:
                exit_triggered = True
                exit_reason = 'EXPIRY'
                
            # Add end-of-day exit for backtesting (use current_timestamp, not datetime.now())
            if not exit_triggered:
                # Check if we're near end of trading day (3:00 PM) using the actual data timestamp
                if current_timestamp.hour >= 15 and current_timestamp.minute >= 0:
                    exit_triggered = True
                    exit_reason = 'END_OF_DAY'
                    self.logger.info(f"Trade {trade_id} exited at end of day at {current_timestamp}")
            
            # Execute exit if triggered
            if exit_triggered:
                # Calculate and log detailed P&L before exit
                entry_premium = trade_record.get('entry_premium', 0)
                lot_size = trade_record.get('lot_size', 75)  # NIFTY lot size
                position_size = trade_record.get('position_size', 1)
                
                # P&L calculation: (Exit Premium - Entry Premium) * Lot Size * Position Size
                pnl_per_unit = exit_premium - entry_premium
                total_pnl = pnl_per_unit * lot_size * position_size
                
                self.logger.info(f"[EXIT] Trade {trade_id} triggered exit: {exit_reason}")
                self.logger.info(f"[P&L] Entry: {entry_premium:.2f}, Exit: {exit_premium:.2f}, Diff: {pnl_per_unit:.2f}")
                self.logger.info(f"[P&L] Calculation: {pnl_per_unit:.2f} * {lot_size} units = {total_pnl:.2f}")
                
                # Use adjusted exit time for 5M exits, otherwise use current timestamp
                # Fixed: Check for any 5M_ pattern instead of hardcoded '5M_INDICATOR_EXIT'
                final_exit_time = adjusted_exit_time if exit_reason and exit_reason.startswith('5M_') and 'adjusted_exit_time' in locals() else current_timestamp
                self._execute_trade_exit(trade_id, str(exit_reason) if exit_reason is not None else '', exit_premium, final_exit_time, current_equity_price)
                
        except Exception as e:
            self.logger.error(f"Error checking exit conditions: {e}")
    
    def _check_1m_exit_conditions(self, trade_record: Dict[str, Any], current_price: float, 
                                 current_timestamp: datetime) -> bool:
        """Check 1-minute exit conditions - SL/TP based on points from entry."""
        try:
            # Use the actual entry price from breakout confirmation, not signal price
            entry_price = trade_record.get('underlying_entry_price', 0)
            if entry_price == 0:
                entry_price = trade_record.get('underlying_price', 0)
            signal_type = trade_record.get('signal_type', 'CALL')
            
            
            # Use SL/TP levels from strategy configuration
            sl_tp_levels = trade_record.get('sl_tp_levels', {})
            sl_price = sl_tp_levels.get('sl_price', 0)
            tp_levels = sl_tp_levels.get('tp_levels', [])
            
            # If no configured levels, fall back to hardcoded for backward compatibility
            if not sl_price or not tp_levels:
                self.logger.warning("No SL/TP levels from strategy config, using fallback values")
                if signal_type == 'CALL':
                    sl_price = entry_price - 100
                    tp_price = entry_price + 500
                else:  # PUT
                    sl_price = entry_price + 100
                    tp_price = entry_price - 500
                    
                tp_levels = [{'price': tp_price, 'type': 'TP1'}]
            
            # Get primary TP level (first one)
            primary_tp = tp_levels[0] if tp_levels else None
            
            if signal_type == 'CALL':
                # For CALL trades
                # Check for Take Profit
                if primary_tp and current_price >= primary_tp['price']:
                    self.logger.info(f"1m TP: CALL price {current_price} >= TP level {primary_tp['price']}")
                    return True
                
                # Check for Stop Loss
                if sl_price and current_price <= sl_price:
                    self.logger.info(f"1m SL: CALL price {current_price} <= SL level {sl_price}")
                    return True
                    
            else:  # PUT
                # For PUT trades
                # Check for Take Profit
                if primary_tp and current_price <= primary_tp['price']:
                    self.logger.info(f"1m TP: PUT price {current_price} <= TP level {primary_tp['price']}")
                    return True
                
                # Check for Stop Loss
                if sl_price and current_price >= sl_price:
                    self.logger.info(f"1m SL: PUT price {current_price} >= SL level {sl_price}")
                    return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking 1m exit conditions: {e}")
            return False
    
    def _check_5m_exit_conditions(self, trade_record: Dict[str, Any], equity_data: Dict[str, Any], 
                                 current_timestamp: datetime) -> bool:
        """Check 5-minute exit conditions - indicator-based exits."""
        try:
            signal_type = trade_record.get('signal_type', 'CALL')
            current_close = equity_data.get('close', 0)
            
            # FIXED: Remove the 5-minute interval restriction since we're processing 1m data
            # During backtesting, we receive 1m execution data but need to check 5m indicator logic
            # The 5m indicators (like EMA) are already calculated on the 5m timeframe data
            
            # Get EMA values from the equity data (these come from 5m signal data)
            ema_9 = equity_data.get('ema_9', 0)
            ema_20 = equity_data.get('ema_20', 0)
            
            # Development logging: Show detailed indicator values
            self.logger.info(f"[5M_EXIT] Signal: {signal_type}, Close: {current_close:.2f}, EMA_9: {ema_9:.2f}, EMA_20: {ema_20:.2f}")
            
            # Enhanced exit conditions based on EMA crossovers and trend reversal
            if signal_type == 'CALL':
                # For CALL trades, exit when trend reverses:
                # 1. Current close falls below EMA_9 (immediate trend reversal)
                # 2. OR EMA_9 crosses below EMA_20 (broader trend reversal)
                
                # Primary exit condition: Close below EMA_9
                if ema_9 > 0 and current_close < ema_9:
                    self.logger.info(f"[5M_EXIT] CALL exit triggered: Close {current_close:.2f} < EMA_9 {ema_9:.2f} (trend reversal)")
                    return True
                
                # Secondary exit condition: EMA_9 < EMA_20 (crossover reversal)
                if ema_9 > 0 and ema_20 > 0 and ema_9 < ema_20:
                    # Check if this is a fresh crossover by storing last state
                    last_ema_state = trade_record.get('last_ema_9_above_20', True)
                    if last_ema_state:  # Was above, now below - fresh crossover
                        self.logger.info(f"[5M_EXIT] CALL exit triggered: EMA crossover reversal - EMA_9 {ema_9:.2f} < EMA_20 {ema_20:.2f}")
                        trade_record['last_ema_9_above_20'] = False
                        return True
                    trade_record['last_ema_9_above_20'] = False
                else:
                    trade_record['last_ema_9_above_20'] = True
                    
            else:  # PUT
                # For PUT trades, exit when trend reverses upward:
                # 1. Current close rises above EMA_9 (immediate trend reversal)
                # 2. OR EMA_9 crosses above EMA_20 (broader trend reversal)
                
                # Primary exit condition: Close above EMA_9
                if ema_9 > 0 and current_close > ema_9:
                    self.logger.info(f"[5M_EXIT] PUT exit triggered: Close {current_close:.2f} > EMA_9 {ema_9:.2f} (trend reversal)")
                    return True
                
                # Secondary exit condition: EMA_9 > EMA_20 (crossover reversal)
                if ema_9 > 0 and ema_20 > 0 and ema_9 > ema_20:
                    # Check if this is a fresh crossover
                    last_ema_state = trade_record.get('last_ema_9_below_20', True)
                    if last_ema_state:  # Was below, now above - fresh crossover
                        self.logger.info(f"5m exit: PUT EMA crossover reversal - EMA_9 {ema_9} > EMA_20 {ema_20}")
                        trade_record['last_ema_9_below_20'] = False
                        return True
                    trade_record['last_ema_9_below_20'] = False
                else:
                    trade_record['last_ema_9_below_20'] = True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error checking 5m exit conditions: {e}")
            return False
    
    def _execute_trade_exit(self, trade_id: str, exit_reason: str, exit_premium: float, exit_timestamp: datetime = None, exit_underlying_price: float = None) -> None:
        """Execute trade exit and calculate P&L."""
        try:
            if trade_id not in self.active_trades:
                return
            
            trade_record = self.active_trades[trade_id]
            
            # Calculate P&L - handle both field names for compatibility
            entry_premium = trade_record.get('entry_premium') or trade_record.get('real_entry_premium', 0)
            
            # Validate entry premium
            if entry_premium <= 0:
                self.logger.error(f"Invalid entry premium for trade {trade_id}: {entry_premium}")
                self.logger.error(f"Available trade fields: {list(trade_record.keys())}")
                return
            
            position_size = trade_record.get('position_size', 1)
            lot_size = trade_record.get('lot_size', 75)  # Correct NIFTY lot size
            commission = trade_record.get('commission', 0)
            
            # Debug: Log values from trade record
            self.logger.info(f"[DEBUG] P&L calculation from trade record: position_size={position_size}, lot_size={lot_size}")
            
            # Calculate gross P&L
            if trade_record['signal_type'] == 'CALL':
                gross_pnl = (exit_premium - entry_premium) * lot_size * position_size
            else:  # PUT
                gross_pnl = (exit_premium - entry_premium) * lot_size * position_size
            
            # Calculate net P&L (subtract commission and slippage)
            slippage_cost = trade_record.get('slippage_cost', 0)
            net_pnl = gross_pnl - commission - slippage_cost
            
            # Update trade record
            trade_record['exit_time'] = exit_timestamp if exit_timestamp else datetime.now()
            trade_record['exit_premium'] = exit_premium
            trade_record['pnl'] = net_pnl
            trade_record['exit_reason'] = str(exit_reason) if exit_reason is not None else ''
            # Store exit candle close price for Trade Summary
            if exit_underlying_price is not None:
                trade_record['exit_underlying_price'] = exit_underlying_price
            
            # Determine trade status
            if 'TP' in exit_reason:
                trade_record['status'] = TradeStatus.TP_HIT.value
                self.performance_metrics['winning_trades'] += 1
                self.performance_metrics['consecutive_wins'] += 1
                self.performance_metrics['consecutive_losses'] = 0
            elif 'SL' in exit_reason:
                trade_record['status'] = TradeStatus.SL_HIT.value
                self.performance_metrics['losing_trades'] += 1
                self.performance_metrics['consecutive_losses'] += 1
                self.performance_metrics['consecutive_wins'] = 0
            else:
                trade_record['status'] = TradeStatus.EXPIRED.value
                if net_pnl > 0:
                    self.performance_metrics['winning_trades'] += 1
                else:
                    self.performance_metrics['losing_trades'] += 1
            
            # Update performance metrics
            self.performance_metrics['total_pnl'] += net_pnl
            self._update_drawdown_metrics(net_pnl)
            
            # Move to trade history
            self.trade_history.append(trade_record.copy())
            
            # Clean up tracking data
            self._cleanup_trade_tracking(trade_id)
            
            # CRITICAL FIX: Don't remove from active trades immediately!
            # The Enhanced Backtest Integration needs to see the completed status
            # It will remove the trade after processing in _process_exits()
            # Just mark it as completed for now
            self.logger.debug(f"Trade {trade_id} marked as {exit_reason}, keeping in active_trades for processing")
            
            # Enhanced exit logging with time details
            exit_time = trade_record.get('exit_time', datetime.now())
            entry_time = trade_record.get('entry_time', 'Unknown')
            
            self.logger.info(f"[EXIT] EXECUTED: {exit_time} @ Premium: {exit_premium:.2f} - {exit_reason}")
            self.logger.info(f"[TRADE] SUMMARY: Entry: {entry_time} @ {entry_premium:.2f} | Exit: {exit_time} @ {exit_premium:.2f} | P&L: {net_pnl:.2f} | Lot Size: {lot_size}")
            
        except Exception as e:
            self.logger.error(f"Error executing trade exit: {e}")
    
    def _cleanup_trade_tracking(self, trade_id: str) -> None:
        """Clean up tracking data for completed trade."""
        try:
            # Remove from tracking dictionaries
            if trade_id in self.equity_charts:
                del self.equity_charts[trade_id]
            if trade_id in self.options_charts:
                del self.options_charts[trade_id]
                
            self.logger.debug(f"Cleaned up tracking data for trade {trade_id}")
            
        except Exception as e:
            self.logger.error(f"Error cleaning up tracking data: {e}")
    
    def _update_drawdown_metrics(self, pnl: float) -> None:
        """Update drawdown metrics."""
        try:
            if pnl < 0:
                self.performance_metrics['current_drawdown'] += abs(pnl)
            else:
                self.performance_metrics['current_drawdown'] = max(0, 
                    self.performance_metrics['current_drawdown'] - pnl)
            
            self.performance_metrics['max_drawdown'] = max(
                self.performance_metrics['max_drawdown'],
                self.performance_metrics['current_drawdown']
            )
            
        except Exception as e:
            self.logger.error(f"Error updating drawdown metrics: {e}")
    
    def _update_performance_metrics(self, trade_data: Dict[str, Any], action: str) -> None:
        """Update performance metrics."""
        try:
            # Additional performance tracking can be added here
            pass
            
        except Exception as e:
            self.logger.error(f"Error updating performance metrics: {e}")
    
    def get_active_trades(self) -> Dict[str, Dict[str, Any]]:
        """Get all active trades."""
        return self.active_trades.copy()
    
    def get_trade_history(self) -> List[Dict[str, Any]]:
        """Get trade history."""
        return self.trade_history.copy()
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """Get performance metrics."""
        try:
            metrics = self.performance_metrics.copy()
            
            # Calculate additional metrics
            if metrics['total_trades'] > 0:
                metrics['win_rate'] = (metrics['winning_trades'] / metrics['total_trades']) * 100
                metrics['avg_pnl_per_trade'] = metrics['total_pnl'] / metrics['total_trades']
            else:
                metrics['win_rate'] = 0
                metrics['avg_pnl_per_trade'] = 0
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error getting performance metrics: {e}")
            return {}
    
    def get_parallel_charts(self, trade_id: str) -> Dict[str, Any]:
        """Get parallel chart data for a trade."""
        try:
            result = {}
            
            if trade_id in self.equity_charts:
                result['equity_chart'] = self.equity_charts[trade_id]
            
            if trade_id in self.options_charts:
                result['options_chart'] = self.options_charts[trade_id]
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error getting parallel charts: {e}")
            return {}
    
    def generate_trade_report(self, trade_id: Optional[str] = None) -> Dict[str, Any]:
        """Generate comprehensive trade report."""
        try:
            if trade_id:
                # Report for specific trade
                if trade_id in self.active_trades:
                    trade_data = self.active_trades[trade_id]
                    return self._generate_single_trade_report(trade_data)
                else:
                    # Check trade history
                    for trade in self.trade_history:
                        if trade['trade_id'] == trade_id:
                            return self._generate_single_trade_report(trade)
                    return {'error': f'Trade {trade_id} not found'}
            else:
                # Generate summary report
                return self._generate_summary_report()
                
        except Exception as e:
            self.logger.error(f"Error generating trade report: {e}")
            return {'error': str(e)}
    
    def _generate_single_trade_report(self, trade_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate report for single trade."""
        try:
            report = {
                'trade_summary': {
                    'trade_id': trade_data['trade_id'],
                    'symbol': trade_data['symbol'],
                    'strategy': trade_data['strategy_name'],
                    'signal_type': trade_data['signal_type'],
                    'strike': trade_data['strike'],
                    'option_type': trade_data['option_type'],
                    'entry_time': trade_data['entry_time'],
                    'exit_time': trade_data.get('exit_time'),
                    'status': trade_data['status'],
                    'pnl': trade_data.get('pnl', 0),
                    'exit_reason': trade_data.get('exit_reason'),
                },
                'trade_details': {
                    'entry_premium': trade_data['entry_premium'],
                    'exit_premium': trade_data.get('exit_premium'),
                    'position_size': trade_data['position_size'],
                    'trade_value': trade_data['trade_value'],
                    'commission': trade_data['commission'],
                    'slippage_cost': trade_data['slippage_cost'],
                    'total_cost': trade_data['total_cost'],
                },
                'greeks_data': trade_data['greeks'],
                'cpr_sl_tp_levels': trade_data['cpr_sl_tp_levels'],
                'tracking_summary': trade_data['tracking_data'],
                'parallel_charts': self.get_parallel_charts(trade_data['trade_id'])
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating single trade report: {e}")
            return {'error': str(e)}
    
    def _generate_summary_report(self) -> Dict[str, Any]:
        """Generate summary report for all trades."""
        try:
            report = {
                'performance_metrics': self.get_performance_metrics(),
                'active_trades_count': len(self.active_trades),
                'completed_trades_count': len(self.trade_history),
                'active_trades': list(self.active_trades.keys()),
                'recent_trades': [trade['trade_id'] for trade in self.trade_history[-10:]],
                'system_health': {
                    'tracking_active': len(self.equity_charts) > 0,
                    'options_db_connected': self.options_db is not None,
                    'strike_selector_active': self.strike_selector is not None,
                    'last_activity': datetime.now().isoformat()
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating summary report: {e}")
            return {'error': str(e)}

    def _check_config_driven_exit_conditions(self, trade_record: Dict[str, Any], equity_data: Dict[str, Any], 
                                           current_timestamp: datetime, execution_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Check config-driven exit conditions - SIMPLIFIED to match entry condition pattern.
        No filtering by exit_type, check ALL conditions every minute like entry conditions.
        
        Args:
            trade_record: Trade record information
            equity_data: Market data with indicators
            current_timestamp: Current timestamp
            execution_data: 1-minute execution data for LTP indicators
            
        Returns:
            Dictionary with exit_triggered, exit_reason, and matched_conditions
        """
        try:
            # Get the specific exit rule from the trade record
            specific_exit_rule = trade_record.get('exit_rule')
            trade_id = trade_record.get('trade_id')
            
            # If no specific exit rule configured, return not triggered
            if not specific_exit_rule:
                self.logger.warning(f"No exit rule specified for trade {trade_id}")
                return {'exit_triggered': False, 'exit_reason': None, 'matched_conditions': []}
            
            # Load exit conditions from config
            exit_rules = self.config.get('exit_rules', {})
            
            # Check only the specific exit rule from the trade record
            if specific_exit_rule not in exit_rules:
                self.logger.warning(f"Exit rule '{specific_exit_rule}' not found in configuration")
                return {'exit_triggered': False, 'exit_reason': None, 'matched_conditions': []}
            
            rule_config = exit_rules[specific_exit_rule]
            
            # Skip inactive rules
            if rule_config.get('status', '').lower() != 'active':
                self.logger.debug(f"Exit rule '{specific_exit_rule}' is inactive")
                return {'exit_triggered': False, 'exit_reason': None, 'matched_conditions': []}
            
            # Use enhanced BacktestUtils evaluation logic - SIMPLIFIED approach
            try:
                from backtest_engine.utils import create_backtest_utils
                backtest_utils = create_backtest_utils(self.config_path)
                
                # Convert equity_data to pandas Series for BacktestUtils
                import pandas as pd
                market_data = pd.Series(equity_data)
                market_data['timestamp'] = current_timestamp
                
                # CRITICAL FIX: Add TP and SL values to market_data for exit condition evaluation
                sl_tp_levels = trade_record.get('sl_tp_levels', {})
                tp_levels = sl_tp_levels.get('tp_levels', [])
                if tp_levels:
                    tp_price = tp_levels[0].get('price', 0)
                    market_data['TP'] = tp_price
                    self.logger.info(f"[EXIT_DEBUG] Added TP to market_data: {tp_price:.2f}")
                else:
                    market_data['TP'] = 0
                    self.logger.warning(f"[EXIT_DEBUG] No TP levels found in trade record")
                
                # Add SL value as well
                sl_price = sl_tp_levels.get('sl_price', 0)
                market_data['SL'] = sl_price
                self.logger.info(f"[EXIT_DEBUG] Added SL to market_data: {sl_price:.2f}")
                
                # Add LTP values from execution data to market_data for LTP_High, LTP_Low conditions
                if execution_data is not None:
                    market_data['LTP_High'] = execution_data.get('high', 0)
                    market_data['LTP_Low'] = execution_data.get('low', 0)
                    market_data['CurrentCandle_Close'] = execution_data.get('close', 0)
                    self.logger.info(f"[EXIT_DEBUG] Added LTP indicators - High: {market_data['LTP_High']:.2f}, TP: {market_data['TP']:.2f}")
                else:
                    market_data['LTP_High'] = equity_data.get('high', 0)
                    market_data['LTP_Low'] = equity_data.get('low', 0)
                    market_data['CurrentCandle_Close'] = equity_data.get('close', 0)
                    self.logger.warning(f"[EXIT_DEBUG] Using equity data for LTP indicators - High: {market_data['LTP_High']:.2f}")
                
                # Convert execution_data to pandas Series for LTP indicators
                execution_data_series = None
                if execution_data is not None:
                    execution_data_series = pd.Series(execution_data)
                    self.logger.info(f"[EXIT_DATA] Execution data converted to Series: {execution_data_series}")
                else:
                    self.logger.info(f"[EXIT_DATA] No execution data available - execution_data is None")
                
                # Evaluate exit rule using enhanced BacktestUtils - SAME AS ENTRY CONDITIONS
                rule_triggered, matched_conditions = backtest_utils.evaluate_exit_rule(
                    specific_exit_rule, rule_config, market_data, execution_data_series, trade_record
                )
                
                if rule_triggered:
                    # SIMPLIFIED: No complex timeframe logic, just use the rule name
                    exit_reason = f"Config-driven exit triggered for trade {trade_id}: {specific_exit_rule.upper()}"
                    
                    self.logger.info(f"[CONFIG_EXIT] Rule '{specific_exit_rule}' triggered")
                    self.logger.info(f"[CONFIG_EXIT] Matched conditions: {matched_conditions}")
                    
                    return {
                        'exit_triggered': True,
                        'exit_reason': exit_reason,
                        'rule_name': specific_exit_rule,
                        'matched_conditions': matched_conditions
                    }
                
                # Exit rule not triggered
                return {'exit_triggered': False, 'exit_reason': None, 'matched_conditions': []}
                
            except ImportError as e:
                self.logger.warning(f"BacktestUtils not available, falling back to basic evaluation: {e}")
                return {'exit_triggered': False, 'exit_reason': None, 'matched_conditions': []}
            
        except Exception as e:
            self.logger.error(f"Error checking config-driven exit conditions: {e}")
            return {'exit_triggered': False, 'exit_reason': None, 'matched_conditions': []}
    
    # Old methods removed - now using BacktestUtils for consistent evaluation


# Convenience functions
def create_options_integrator(config_path: Optional[str] = None, options_db: Optional[OptionsDatabase] = None) -> OptionsRealMoneyIntegrator:
    """
    Create options real money integrator.
    
    Args:
        config_path: Path to configuration file
        options_db: Options database instance (optional)
        
    Returns:
        OptionsRealMoneyIntegrator instance
    """
    integrator = OptionsRealMoneyIntegrator(config_path)
    if options_db:
        integrator.options_db = options_db
    return integrator


# Example usage
if __name__ == "__main__":
    # Example usage of the Options Real Money Integrator
    try:
        # Create integrator
        integrator = create_options_integrator()
        
        # Example signal data
        signal_data = {
            'signal_type': 'CALL',
            'underlying_price': 21650.0,
            'timestamp': datetime.now(),
            'strategy_name': 'momentum_strategy',
            'strike_preference': 'ATM',
            'symbol': 'NIFTY',
            'equity_data': {
                'open': 21640.0,
                'high': 21660.0,
                'low': 21630.0,
                'close': 21650.0,
                'volume': 1000000
            }
        }
        
        print("Processing test signal...")
        trade_result = integrator.process_trade_signal(signal_data)
        
        if trade_result:
            print(f"Trade executed: {trade_result['trade_id']}")
            print(f"Strike: {trade_result['strike']}")
            print(f"Premium: {trade_result['entry_premium']:.2f}")
            print(f"Position size: {trade_result['position_size']} lots")
            print(f"Trade value: {trade_result['trade_value']:.2f}")
            
            # Simulate market update
            updated_equity_data = {
                'open': 21650.0,
                'high': 21680.0,
                'low': 21640.0,
                'close': 21670.0,
                'volume': 1200000
            }
            
            updated_options_data = {
                'premium': 120.0,
                'iv': 18.5,
                'delta': 0.6,
                'intrinsic_value': 20.0,
                'time_value': 100.0
            }
            
            print("\nUpdating parallel tracking...")
            integrator.update_parallel_tracking(
                trade_result['trade_id'], 
                updated_equity_data, 
                updated_options_data
            )
            
            # Get performance metrics
            metrics = integrator.get_performance_metrics()
            print(f"\nPerformance metrics:")
            print(f"Total trades: {metrics['total_trades']}")
            print(f"Active trades: {len(integrator.get_active_trades())}")
            print(f"Total P&L: {metrics['total_pnl']:.2f}")
            
        else:
            print("Trade execution failed")
            
    except Exception as e:
        print(f"Error in example usage: {e}")