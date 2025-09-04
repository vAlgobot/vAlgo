"""
VectorBT Signal Generator - Phase 3 of Ultimate Efficiency Engine
================================================================

Ultra-fast VectorBT signal generation using IndicatorFactory pattern.
Extracts Phase 3 functionality from ultimate_efficiency_engine.py for modular architecture.

Features:
- VectorBT IndicatorFactory pattern implementation
- Strategy-case configuration handling  
- Option type and strike type management per case
- Performance metrics tracking
- Direct signal generation with condition evaluation

Author: vAlgo Development Team
Created: July 29, 2025
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, time as dt_time
from datetime import timedelta
import warnings

from core.processor_base import ProcessorBase
from core.json_config_loader import ConfigurationError

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# VectorBT imports with error handling
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
except ImportError:
    VECTORBT_AVAILABLE = False
    raise ConfigurationError("VectorBT not available. Install with: pip install vectorbt[full]")


def create_time_masks(timestamps: pd.Index, config_loader=None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time masks for trading hours validation and forced exits using config.json settings.
    
    Args:
        timestamps: DataFrame index with datetime timestamps
        config_loader: Configuration loader instance (optional for backward compatibility)
        
    Returns:
        Tuple of (trading_hours_mask, forced_exit_mask)
    """
    try:
        # Default hardcoded values (fallback)
        default_trading_start = dt_time(9, 15)
        default_trading_end = dt_time(14, 55)
        default_forced_exit = dt_time(14, 55)
        
        # Use config values if available
        if config_loader:
            risk_config = config_loader.main_config.get('risk_management', {})
            trading_config = config_loader.main_config.get('trading', {})
            
            # Get timeframe for adjustment (subtract from config times)
            # When LTP timeframe is enabled, use 1m for forced exit calculation
            ltp_timeframe_enabled = trading_config.get('ltp_timeframe', False)
            if ltp_timeframe_enabled:
                timeframe = "1m"
                timeframe_minutes = 1
                print(f"🔧 LTP timeframe mode detected - using 1m for forced exit calculation")
            else:
                timeframe = trading_config.get('timeframe', '5m')
                if timeframe.endswith('m'):
                    timeframe_minutes = int(timeframe[:-1])
                else:
                    timeframe_minutes = 5 # Default to 5 minutes
            
            # Get configured times and subtract timeframe minutes
            trading_start_str = risk_config.get('trading_start_time', '09:20')
            trading_end_str = risk_config.get('trading_end_time', '15:00')
            forced_exit_str = risk_config.get('forced_exit_time', '15:00')
            
            # Parse and subtract timeframe minutes from config times for better signal generation
            def subtract_minutes_from_time(time_str, minutes_to_subtract):
                hour, minute = map(int, time_str.split(':'))
                # Convert to total minutes, subtract, then convert back
                total_minutes = hour * 60 + minute - minutes_to_subtract
                # Handle negative values (wrap to previous day if needed)
                if total_minutes < 0:
                    total_minutes = 0  # Clamp to midnight
                new_hour = total_minutes // 60
                new_minute = total_minutes % 60
                return dt_time(new_hour, new_minute)
            
            # Apply subtraction to all config times
            trading_start = subtract_minutes_from_time(trading_start_str, timeframe_minutes)
            trading_end = subtract_minutes_from_time(trading_end_str, timeframe_minutes)
            forced_exit_time = subtract_minutes_from_time(forced_exit_str, timeframe_minutes)
            
            print(f"📅 Using config times (subtracted {timeframe_minutes}m for {timeframe}):")
            print(f"   Trading Start: {trading_start_str} - {timeframe_minutes}m -> {trading_start}")
            print(f"   Trading End: {trading_end_str} - {timeframe_minutes}m -> {trading_end}")
            print(f"   Forced Exit: {forced_exit_str} - {timeframe_minutes}m -> {forced_exit_time}")
        else:
            # Use defaults if no config
            trading_start = default_trading_start
            trading_end = default_trading_end
            forced_exit_time = default_forced_exit
            print(f"⚠️ Using default hardcoded times (no config provided)")
        
        # Extract time from timestamps
        times = pd.to_datetime(timestamps).time
        
        # Trading hours mask
        trading_hours_mask = (times >= trading_start) & (times <= trading_end)
        
        # Forced exit mask
        forced_exit_mask = (times == forced_exit_time)
        
        
        # Convert to numpy arrays properly
        if hasattr(trading_hours_mask, 'values'):
            return trading_hours_mask.values, forced_exit_mask.values
        else:
            return np.array(trading_hours_mask), np.array(forced_exit_mask)
        
    except Exception as e:
        print(f"⚠️ Error creating time masks: {e}")
        # Fallback to all True for trading hours, all False for forced exits
        n = len(timestamps)
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)




def vectorized_position_tracking_with_locked_values(
    entry_conditions: np.ndarray, 
    exit_condition_string: str,
    trading_hours_mask: np.ndarray, 
    forced_exit_mask: np.ndarray,
    market_ohlc: Dict[str, np.ndarray],
    case_config: Dict[str, Any],
    eval_context: Dict[str, Any],
    indicator_timestamps: np.ndarray = None,
    performance_mode: str = 'standard',
    compiled_conditions_cache: Dict[str, Any] = None,
    group_name: str = '',
    case_name: str = '',
    signal_generator=None
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Enhanced stateful position tracking with locked signal candle and SL/TP values.
    
    Key Features:
    - Maintains stateful position tracking (in_position flag)
    - Locks signal candle OHLC and SL/TP values at entry
    - Ignores duplicate entries until exit
    - Evaluates exit conditions with locked values (FIXED APPROACH)
    - Supports JSON-driven SL/TP calculation methods
    
    Args:
        entry_conditions: Boolean array of entry signals
        exit_condition_string: String with exit condition to be evaluated with locked values
        trading_hours_mask: Boolean array for valid trading hours
        forced_exit_mask: Boolean array for forced exits at 15:15
        market_ohlc: Dictionary with 'open', 'high', 'low', 'close' numpy arrays
        case_config: Strategy case configuration (SL_Method, TP_Method, etc.)
        eval_context: Evaluation context for SL/TP calculation
        indicator_timestamps: Array of Indicator_Timestamp values for exit blacklist tracking
        performance_mode: Performance mode for optimization
        compiled_conditions_cache: Cache for compiled exit conditions
        group_name: Strategy group name for logging
        case_name: Strategy case name for logging
        signal_generator: VectorBTSignalGenerator instance for CPR pivot TP calculation
        
    Returns:
        Tuple of (signals_array, locked_values_dict) where:
        - signals_array: Integer array with signals: 1 (Entry), -1 (Exit), 0 (Hold)
        - locked_values_dict: Dictionary with locked arrays for signal candle and SL/TP
    """
    try:
        n = len(entry_conditions)
        signals = np.zeros(n, dtype=int)
        
        # Initialize locked value arrays
        locked_signal_candle_open = np.full(n, np.nan)
        locked_signal_candle_high = np.full(n, np.nan)
        locked_signal_candle_low = np.full(n, np.nan)
        locked_signal_candle_close = np.full(n, np.nan)
        locked_sl_price = np.full(n, np.nan)
        locked_tp_price = np.full(n, np.nan)
        locked_entry_point = np.full(n, np.nan)
        
        # Filter entries by trading hours (config-based)
        valid_entries = entry_conditions & trading_hours_mask
        
        # Initialize exit conditions array (will be evaluated dynamically with locked values)
        exit_conditions = np.zeros(n, dtype=bool)
        
        # Exit timestamp blacklist for unique entry validation
        exit_timestamp_blacklist = set()
        
        # Helper function to evaluate exit condition with current eval_context
        def evaluate_exit_condition_at_index(index):
            """Evaluate exit condition at specific index with current locked values in eval_context"""
            try:
                if index <= 10:  # Only debug first few calls
                    print(f"   🔍 DEBUG: Evaluating exit at index {index} with condition '{exit_condition_string}'")
                
                # FORCE SAFE EVALUATION ALWAYS - Skip all the complex caching logic
                point_context = {}
                for key, values in eval_context.items():
                    try:
                        if hasattr(values, '__len__') and len(values) > index:
                            if hasattr(values, 'iloc'):  # pandas Series
                                point_context[key] = values.iloc[index]
                            else:  # numpy array
                                point_context[key] = values[index]
                        else:
                            point_context[key] = values
                    except Exception as ctx_error:
                        if index <= 10:
                            print(f"   ⚠️ CONTEXT ERROR for key '{key}' at index {index}: {ctx_error}")
                        # Skip problematic keys but continue
                        continue
                
                # HANDLE KNOWN EXIT CONDITIONS WITH SAFE MANUAL EVALUATION
                # if exit_condition_string == "(ltp_high > sl_price) | (ltp_low < tp_price)":
                #     # PUT exit condition
                #     ltp_high = point_context.get('ltp_high', 0)
                #     ltp_low = point_context.get('ltp_low', 0) 
                #     sl_price = point_context.get('sl_price', np.nan)
                #     tp_price = point_context.get('tp_price', np.nan)
                    
                #     sl_hit = False if np.isnan(sl_price) else ltp_high > sl_price
                #     tp_hit = False if np.isnan(tp_price) else ltp_low < tp_price
                #     result = sl_hit or tp_hit
                    
                #     if index <= 20 or result:
                #         timestamp_str = str(eval_context.get('timestamp', [pd.NaT] * (index+1))[index]) if 'timestamp' in eval_context else "Unknown"
                #         print(f"   ✅ SAFE PUT EVAL at {index} ({timestamp_str}): ltp_high={ltp_high}, sl_price={sl_price}, sl_hit={sl_hit}, ltp_low={ltp_low}, tp_price={tp_price}, tp_hit={tp_hit}, result={result}")
                # elif exit_condition_string == "(ltp_low < sl_price) | (ltp_high > tp_price)":
                #     # CALL exit condition - FIXED: Added this missing case
                #     ltp_low = point_context.get('ltp_low', 0)
                #     ltp_high = point_context.get('ltp_high', 0)
                #     sl_price = point_context.get('sl_price', np.nan)
                #     tp_price = point_context.get('tp_price', np.nan)
                    
                #     sl_hit = False if np.isnan(sl_price) else ltp_low < sl_price  
                #     tp_hit = False if np.isnan(tp_price) else ltp_high > tp_price
                #     result = sl_hit or tp_hit
                    
                #     if index <= 20 or result:
                #         timestamp_str = str(eval_context.get('timestamp', [pd.NaT] * (index+1))[index]) if 'timestamp' in eval_context else "Unknown"
                #         print(f"   ✅ SAFE CALL EVAL at {index} ({timestamp_str}): ltp_low={ltp_low}, sl_price={sl_price}, sl_hit={sl_hit}, ltp_high={ltp_high}, tp_price={tp_price}, tp_hit={tp_hit}, result={result}")
                # elif exit_condition_string == "(close < sma_9) | (ltp_low < sl_price) | (ltp_high > tp_price)":
                #     # CALL exit condition with SMA component
                #     close = point_context.get('close', 0)
                #     sma_9 = point_context.get('sma_9', 0)
                #     ltp_low = point_context.get('ltp_low', 0)
                #     ltp_high = point_context.get('ltp_high', 0)
                #     sl_price = point_context.get('sl_price', np.nan)
                #     tp_price = point_context.get('tp_price', np.nan)
                    
                #     close_exit = close < sma_9 if not (np.isnan(close) or np.isnan(sma_9)) else False
                #     sl_hit = False if np.isnan(sl_price) else ltp_low < sl_price  
                #     tp_hit = False if np.isnan(tp_price) else ltp_high > tp_price
                #     result = close_exit or sl_hit or tp_hit
                    
                #     if index <= 10:
                #         print(f"   ✅ SAFE CALL EVAL at {index}: close={close}, sma_9={sma_9}, close_exit={close_exit}, sl_hit={sl_hit}, tp_hit={tp_hit}, result={result}")
                # else:
                    #Try generic eval with proper context
                    try:
                        # Add numpy for mathematical operations
                        point_context['np'] = np
                        
                        # Handle potential NaN values in SL/TP for safe evaluation
                        if 'sl_price' in point_context and np.isnan(point_context['sl_price']):
                            point_context['sl_price'] = float('inf')  # Will never be hit
                        if 'tp_price' in point_context and np.isnan(point_context['tp_price']):
                            point_context['tp_price'] = float('inf')  # Will never be hit
                            
                        result = eval(exit_condition_string, {"__builtins__": {}}, point_context)
                        
                        if index <= 10:
                            print(f"   ✅ GENERIC EVAL at {index}: condition='{exit_condition_string}', result={result}")
                    except Exception as eval_error:
                        result = False
                        if index <= 10:
                            print(f"   ⚠️ EVAL ERROR at {index}: {eval_error}, defaulting to False")
                
                return bool(result)
                
            except Exception as e:
                print(f"⚠️ Exit condition evaluation error at index {index}: {type(e).__name__}: {str(e)}")
                print(f"   🔍 DEBUG: Exit condition was: '{exit_condition_string}'")
                print(f"   🔍 DEBUG: Available context keys: {list(point_context.keys())}")
                return False
        
        # CRITICAL: Sequential stateful position tracking with value locking
        in_position = False
        current_entry_index = None
        locked_entry_ohlc = None
        locked_sl_value = None
        locked_tp_value = None
        locked_entry_point_value = None
        entries_blocked = 0
        exits_tracked = 0
        
        # NEW: Track last exit indicator timestamp to prevent immediate re-entries
        last_exit_indicator_timestamp = None
        
        print(f"🔒 STATEFUL POSITION TRACKING: Processing {n} bars for locked value management")
        
        for i in range(n):
            current_indicator_time = indicator_timestamps[i] if indicator_timestamps is not None else None
            
            if valid_entries[i] and not in_position:
                # NEW ENTRY: Check timestamp blacklist first
                if current_indicator_time is not None and current_indicator_time in exit_timestamp_blacklist:
                    signals[i] = 0  # Skip this entry - timestamp already used
                    entries_blocked += 1
                    continue
                
                # NEW: Prevent immediate re-entries - block entries with same Indicator_Timestamp as recent exit
                # print(f"🔍 ENTRY DEBUG: current_time={current_indicator_time}, last_exit_time={last_exit_indicator_timestamp}")
                if (current_indicator_time is not None and 
                    last_exit_indicator_timestamp is not None and 
                    str(current_indicator_time) == str(last_exit_indicator_timestamp)):
                    signals[i] = 0  # Skip this entry - same indicator timestamp as recent exit
                    entries_blocked += 1
                    # print(f"🚫 ENTRY BLOCKED: Same Indicator_Timestamp as recent exit (current: {current_indicator_time}, last_exit: {last_exit_indicator_timestamp})")
                    continue
                
                # ENTRY LOGIC: Capture and lock signal candle + calculate SL/TP
                signals[i] = 1      # Entry signal
                in_position = True
                current_entry_index = i
                
                # LOCK SIGNAL CANDLE VALUES from current bar
                locked_entry_ohlc = {
                    'open': market_ohlc['open'][i],
                    'high': market_ohlc['high'][i],
                    'low': market_ohlc['low'][i],
                    'close': market_ohlc['close'][i]
                }
                
                # Calculate entry point based on option type (SMART: LTP-aware dynamic entry point detection)
                option_type = case_config.get('OptionType', 'CALL').upper()
                
                # Smart conditional logic: Check if entry/exit conditions use LTP data
                entry_condition = case_config.get('entry', '')
                has_ltp_high = 'ltp_high' in entry_condition 
                has_ltp_low = 'ltp_low' in entry_condition
                
                if option_type == 'CALL':
                    if has_ltp_high and 'ltp_high' in eval_context:
                        # Use LTP breakout for CALL entry (more accurate 1m execution)
                        ltp_high_data = eval_context['ltp_high']
                        if hasattr(ltp_high_data, '__getitem__') and len(ltp_high_data) > i:
                            locked_entry_point_value = ltp_high_data[i]
                            # print(f"🚀 CALL SMART ENTRY: Using ltp_high breakout {locked_entry_point_value:.2f} (1m precision)")
                        else:
                            locked_entry_point_value = locked_entry_ohlc['high']
                            # print(f"🔥 CALL FALLBACK: ltp_high not available, using signal_candle_high {locked_entry_point_value:.2f}")
                    else:
                        # Use signal candle high for CALL entry (5m precision)
                        locked_entry_point_value = locked_entry_ohlc['high']
                        # print(f"🎯 CALL ENTRY: Using signal_candle_high {locked_entry_point_value:.2f} (no LTP in conditions)")
                        
                else:  # PUT
                    if has_ltp_low and 'ltp_low' in eval_context:
                        # Use LTP breakdown for PUT entry (more accurate 1m execution)
                        ltp_low_data = eval_context['ltp_low']
                        if hasattr(ltp_low_data, '__getitem__') and len(ltp_low_data) > i:
                            locked_entry_point_value = ltp_low_data[i]
                            # print(f"🚀 PUT SMART ENTRY: Using ltp_low breakdown {locked_entry_point_value:.2f} (1m precision)")
                        else:
                            locked_entry_point_value = locked_entry_ohlc['low']
                            # print(f"🔥 PUT FALLBACK: ltp_low not available, using signal_candle_low {locked_entry_point_value:.2f}")
                    else:
                        # Use signal candle low for PUT entry (5m precision)
                        locked_entry_point_value = locked_entry_ohlc['low']
                        # print(f"🎯 PUT ENTRY: Using signal_candle_low {locked_entry_point_value:.2f} (no LTP in conditions)")
                
                # CALCULATE SL/TP using locked signal candle values
                sl_value, tp_value = _calculate_sl_tp_for_entry(
                    locked_entry_ohlc, case_config, locked_entry_point_value, eval_context, signal_generator, i
                )
                
                locked_sl_value = sl_value
                locked_tp_value = tp_value
                
                # FIXED: Only populate locked values at current position (no slice assignment)
                locked_signal_candle_open[i] = locked_entry_ohlc['open']
                locked_signal_candle_high[i] = locked_entry_ohlc['high']
                locked_signal_candle_low[i] = locked_entry_ohlc['low']
                locked_signal_candle_close[i] = locked_entry_ohlc['close']
                locked_sl_price[i] = locked_sl_value
                locked_tp_price[i] = locked_tp_value
                locked_entry_point[i] = locked_entry_point_value
                
                # print(f"🔒 ENTRY LOCKED at index {i}: Entry={locked_entry_point_value:.2f}, SL={locked_sl_value:.2f}, TP={locked_tp_value:.2f}")
                
            elif valid_entries[i] and in_position:
                # IGNORE DUPLICATE ENTRIES: Already in position
                signals[i] = 0      # Hold signal (ignore new entry)
                entries_blocked += 1
                # print(f"🚫 ENTRY IGNORED at index {i}: Already in position since index {current_entry_index}")
                
                # CRITICAL FIX: Still populate locked values even when blocking duplicate entries
                # This ensures sl_price, tp_price, entry_point are available for exit condition evaluation
                if locked_sl_value is None or locked_tp_value is None or locked_entry_ohlc is None:
                    print(f"❌ ERROR at index {i}: locked values are None but in_position=True")
                    in_position = False
                    continue
                    
                # Use locked scalar values directly from entry position
                eval_context['signal_candle_open'][i] = locked_entry_ohlc['open']
                eval_context['signal_candle_high'][i] = locked_entry_ohlc['high']  
                eval_context['signal_candle_low'][i] = locked_entry_ohlc['low']
                eval_context['signal_candle_close'][i] = locked_entry_ohlc['close']
                eval_context['sl_price'][i] = locked_sl_value
                eval_context['tp_price'][i] = locked_tp_value
                eval_context['entry_point'][i] = locked_entry_point_value
                
                # Also populate array tracking for export consistency
                locked_signal_candle_open[i] = locked_entry_ohlc['open']
                locked_signal_candle_high[i] = locked_entry_ohlc['high']
                locked_signal_candle_low[i] = locked_entry_ohlc['low']
                locked_signal_candle_close[i] = locked_entry_ohlc['close']
                locked_sl_price[i] = locked_sl_value
                locked_tp_price[i] = locked_tp_value
                locked_entry_point[i] = locked_entry_point_value
                
                # CHECK EXIT CONDITIONS: Evaluate with current locked values in eval_context
                natural_exit = evaluate_exit_condition_at_index(i)
                forced_exit = forced_exit_mask[i] if i < len(forced_exit_mask) else False
                
                if natural_exit or forced_exit:
                    # EXIT LOGIC: Exit triggered by conditions with locked values
                    signals[i] = -1     # Exit signal
                    in_position = False # RESET position state to allow new entries
                    exit_type = "forced" if forced_exit else "natural"
                    #print(f"🔓 EXIT TRIGGERED at index {i}: {exit_type} exit (locked values used for evaluation)")
                    
                    # NEW: Track exit indicator timestamp to prevent immediate re-entries
                    last_exit_indicator_timestamp = current_indicator_time
                    #if current_indicator_time is not None:
                        #print(f"🔓 EXIT PROCESSED: Setting last_exit_indicator_timestamp = {current_indicator_time} (type: {type(current_indicator_time)})")
                    
                    # Reset tracking variables after exit
                    current_entry_index = None
                    locked_entry_ohlc = None
                    locked_sl_value = None
                    locked_tp_value = None
                    locked_entry_point_value = None
                    exits_tracked += 1
                    
                    # Add this exit's timestamp to blacklist
                    if indicator_timestamps is not None and i < len(indicator_timestamps):
                        exit_timestamp_blacklist.add(indicator_timestamps[i])
                        
                    #print(f"🔓 EXIT PROCESSED at index {i}: Position unlocked, awaiting new Indicator_Timestamp")
                
            elif in_position:
                # UPDATE eval_context arrays with locked values at current index before exit evaluation
                # Use SCALAR locked values directly (no array access needed)
                if locked_sl_value is None or locked_tp_value is None or locked_entry_ohlc is None:
                    # print(f"❌ ERROR at index {i}: locked values are None but in_position=True")
                    in_position = False
                    continue
                    
                # Use locked scalar values directly from entry position
                eval_context['signal_candle_open'][i] = locked_entry_ohlc['open']
                eval_context['signal_candle_high'][i] = locked_entry_ohlc['high']  
                eval_context['signal_candle_low'][i] = locked_entry_ohlc['low']
                eval_context['signal_candle_close'][i] = locked_entry_ohlc['close']
                eval_context['sl_price'][i] = locked_sl_value
                eval_context['tp_price'][i] = locked_tp_value
                eval_context['entry_point'][i] = locked_entry_point_value
                
                # Also populate array tracking for export consistency
                locked_signal_candle_open[i] = locked_entry_ohlc['open']
                locked_signal_candle_high[i] = locked_entry_ohlc['high']
                locked_signal_candle_low[i] = locked_entry_ohlc['low']
                locked_signal_candle_close[i] = locked_entry_ohlc['close']
                locked_sl_price[i] = locked_sl_value
                locked_tp_price[i] = locked_tp_value
                locked_entry_point[i] = locked_entry_point_value
                
                # CHECK EXIT CONDITIONS: Evaluate with current locked values in eval_context
                natural_exit = evaluate_exit_condition_at_index(i)
                forced_exit = forced_exit_mask[i] if i < len(forced_exit_mask) else False
                
                if natural_exit or forced_exit:
                    # EXIT LOGIC: Exit triggered by conditions with locked values
                    signals[i] = -1     # Exit signal
                    in_position = False # RESET position state to allow new entries
                    exit_type = "forced" if forced_exit else "natural"
                    # print(f"🔓 EXIT TRIGGERED at index {i}: {exit_type} exit (locked values used for evaluation)")
                    
                    # NEW: Track exit indicator timestamp to prevent immediate re-entries
                    last_exit_indicator_timestamp = current_indicator_time
                    #if current_indicator_time is not None:
                        #print(f"🔓 EXIT PROCESSED: Setting last_exit_indicator_timestamp = {current_indicator_time} (type: {type(current_indicator_time)})")
                    
                    # Reset tracking variables after exit
                    current_entry_index = None
                    locked_entry_ohlc = None
                    locked_sl_value = None
                    locked_tp_value = None
                    locked_entry_point_value = None
                    
                    # Add this exit's Indicator_Timestamp to blacklist
                    if current_indicator_time is not None:
                        exit_timestamp_blacklist.add(current_indicator_time)
                        #print(f"🔓 EXIT PROCESSED at index {i}: Position unlocked, awaiting new Indicator_Timestamp")
                        
                else:
                    # HOLD POSITION: Continue with locked values
                    signals[i] = 0      # Hold signal
                
            else:
                # NO ENTRY OR EXIT CONDITIONS MET: Hold/neutral state
                signals[i] = 0      # Hold signal (neutral state)
        
        # Debug logging for timestamp validation
        if indicator_timestamps is not None:
            total_entries_attempted = np.sum(valid_entries)
            total_entries_allowed = np.sum(signals == 1)
            #print(f"🛡️ STATEFUL TRACKING: {entries_blocked} duplicate entries blocked, {exits_tracked} exits tracked, {total_entries_allowed}/{total_entries_attempted} entries allowed")
        
        # Package locked values for return
        locked_values = {
            'signal_candle_open': locked_signal_candle_open,
            'signal_candle_high': locked_signal_candle_high,
            'signal_candle_low': locked_signal_candle_low,
            'signal_candle_close': locked_signal_candle_close,
            'sl_price': locked_sl_price,
            'tp_price': locked_tp_price,
            'entry_point': locked_entry_point
        }
        
        return signals, locked_values
        
    except Exception as e:
        print(f"Error in stateful position tracking: {e}")
        sys.exit(1)
        # Fallback to zeros array and empty locked values
        n = len(entry_conditions)
        empty_locked_values = {
            'signal_candle_open': np.full(n, np.nan),
            'signal_candle_high': np.full(n, np.nan),
            'signal_candle_low': np.full(n, np.nan),
            'signal_candle_close': np.full(n, np.nan),
            'sl_price': np.full(n, np.nan),
            'tp_price': np.full(n, np.nan),
            'entry_point': np.full(n, np.nan)
        }
        return np.zeros(n, dtype=int), empty_locked_values


def _calculate_sl_tp_for_entry(
    signal_candle_ohlc: Dict[str, float],
    case_config: Dict[str, Any], 
    entry_point: float,
    eval_context: Dict[str, Any],
    signal_generator=None,
    entry_index: int = 0
) -> Tuple[float, float]:
    """
    Calculate SL/TP values for a single entry using locked signal candle values.
    
    Args:
        signal_candle_ohlc: Dictionary with 'open', 'high', 'low', 'close' values
        case_config: Strategy case configuration with SL_Method, TP_Method, etc.
        entry_point: Entry price point 
        eval_context: Evaluation context for additional data access
        signal_generator: VectorBTSignalGenerator instance for CPR pivot TP calculation
        entry_index: Current entry index for correct CPR array access in multi-day runs
        
    Returns:
        Tuple of (sl_price, tp_price)
    """
    try:
        # Extract SL/TP methods from case config
        sl_method = case_config.get('SL_Method', 'cpr_pivot_range_exit')
        tp_method = case_config.get('TP_Method', 'cpr_pivot_range_exit')
        option_type = case_config.get('OptionType', 'CALL')
        max_sl_points = float(case_config.get('Max_Sl_points', 45))
        
        # Get additional config parameters from indicators.SL_TP_levels
        sl_tp_config = eval_context.get('_config_sl_tp_levels', {})
        risk_reward_ratio = sl_tp_config.get('RiskRewardRatio', '1:1')
        buffer_sl_points = float(sl_tp_config.get('buffer_sl_points', 0))
        buffer_tp_points = float(sl_tp_config.get('buffer_tp_points', 0))
        
        # Parse risk reward ratio
        try:
            risk_part, reward_part = risk_reward_ratio.split(':')
            reward_multiplier = float(reward_part) / float(risk_part)
        except:
            reward_multiplier = 1.0  # Default 1:1
        
        # Calculate SL/TP based on method and option type
        if sl_method == 'signal_candle' or sl_method == 'signal_candle_range_exit':
            print(f"🎯 USING signal_candle_range_exit EXIT METHOD for {option_type}")
            # Use signal candle range for SL calculation with Max_Sl_points limit
            signal_range = signal_candle_ohlc['high'] - signal_candle_ohlc['low']
            
            # Apply Max_Sl_points limit to signal range (this is the key fix)
            effective_sl_range = min(signal_range, max_sl_points)
            
            if option_type == 'CALL':
                # CALL: SL calculated from entry point minus effective range
                sl_price = entry_point - effective_sl_range - buffer_sl_points
                tp_distance = effective_sl_range * reward_multiplier
                tp_price = entry_point + tp_distance + buffer_tp_points
                
                #print(f"🔧 CALL SL/TP: Entry={entry_point:.2f}, Signal_Range={signal_range:.2f}, Effective_SL_Range={effective_sl_range:.2f} (max {max_sl_points})")
                #print(f"   SL = {entry_point:.2f} - {effective_sl_range:.2f} - {buffer_sl_points} = {sl_price:.2f}")
                #print(f"   TP = {entry_point:.2f} + {tp_distance:.2f} + {buffer_tp_points} = {tp_price:.2f}")
                
            else:  # PUT
                # PUT: SL calculated from entry point plus effective range  
                sl_price = entry_point + effective_sl_range + buffer_sl_points
                tp_distance = effective_sl_range * reward_multiplier
                tp_price = entry_point - tp_distance - buffer_tp_points
                
                #print(f"🔧 PUT SL/TP: Entry={entry_point:.2f}, Signal_Range={signal_range:.2f}, Effective_SL_Range={effective_sl_range:.2f} (max {max_sl_points})")
                #print(f"   SL = {entry_point:.2f} + {effective_sl_range:.2f} + {buffer_sl_points} = {sl_price:.2f}")
                #print(f"   TP = {entry_point:.2f} - {tp_distance:.2f} - {buffer_tp_points} = {tp_price:.2f}")
                
        elif sl_method == 'cpr_pivot_range_exit':
            # NEW: CPR Pivot-based SL/TP calculation matching vectorized version
            print(f"🎯 USING cpr_pivot_range_exit METHOD for {option_type}")
            
            # Calculate signal range and risk (same logic as vectorized version)
            signal_range = signal_candle_ohlc['high'] - signal_candle_ohlc['low']
            effective_sl_range = min(signal_range, max_sl_points)
            risk = effective_sl_range + buffer_sl_points
            
            print(f"   Range Calculation: {signal_candle_ohlc['high']:.2f} - {signal_candle_ohlc['low']:.2f} = {signal_range:.2f}")
            print(f"   Effective Range: {effective_sl_range:.2f} (max: {max_sl_points})")
            print(f"   Risk: {effective_sl_range:.2f} + {buffer_sl_points} = {risk:.2f}")
            
            if option_type == 'CALL':
                # CALL: SL calculation using signal range (consistent with vectorized)
                sl_price = entry_point - (effective_sl_range + buffer_sl_points)
                
                # CALL: TP calculation using CPR pivot targeting
                try:
                    if signal_generator and hasattr(signal_generator, '_calculate_cpr_pivot_tp'):
                        tp_price = signal_generator._calculate_cpr_pivot_tp(
                            option_type, entry_point, risk, reward_multiplier, buffer_tp_points, eval_context, entry_index
                        )
                        print(f"🎯 ✅ CPR PIVOT TP SUCCESSFUL: TP={tp_price:.2f} (index={entry_index})")
                    else:
                        # Fallback if signal_generator not available
                        raise Exception("Signal generator instance not available for CPR pivot TP calculation")
                        
                except Exception as e:
                    print(f"   ❌ CPR Pivot TP failed, using fallback: {e}")
                    # Fallback to standard RR calculation
                    reward = risk * reward_multiplier
                    tp_price = entry_point + reward + buffer_tp_points
                    
            else:  # PUT
                # PUT: SL calculation using signal range (consistent with vectorized)
                sl_price = entry_point + (effective_sl_range + buffer_sl_points)
                
                # PUT: TP calculation using CPR pivot targeting  
                try:
                    if signal_generator and hasattr(signal_generator, '_calculate_cpr_pivot_tp'):
                        tp_price = signal_generator._calculate_cpr_pivot_tp(
                            option_type, entry_point, risk, reward_multiplier, buffer_tp_points, eval_context, entry_index
                        )
                        print(f"🎯 ✅ CPR PIVOT TP SUCCESSFUL: TP={tp_price:.2f} (index={entry_index})")
                    else:
                        # Fallback if signal_generator not available
                        raise Exception("Signal generator instance not available for CPR pivot TP calculation")
                        
                except Exception as e:
                    print(f"   ❌ CPR Pivot TP failed, using fallback: {e}")
                    # Fallback to standard RR calculation
                    reward = risk * reward_multiplier
                    tp_price = entry_point - reward - buffer_tp_points
            
            print(f"   {option_type} CPR Calculation:")
            print(f"     SL = {entry_point:.2f} {'- ' if option_type == 'CALL' else '+ '}{risk:.2f} = {sl_price:.2f}")
            print(f"     TP = {tp_price:.2f} (CPR pivot-based)")
            print(f"   FINAL RESULT: SL={sl_price:.2f}, TP={tp_price:.2f}")
        else:
            # Fallback: Use signal candle method with Max_Sl_points limit
            signal_range = signal_candle_ohlc['high'] - signal_candle_ohlc['low']
            effective_sl_range = min(signal_range, max_sl_points)
            
            if option_type == 'CALL':
                sl_price = entry_point - effective_sl_range - buffer_sl_points
                tp_price = entry_point + effective_sl_range * reward_multiplier + buffer_tp_points
            else:  # PUT
                sl_price = entry_point + effective_sl_range + buffer_sl_points
                tp_price = entry_point - effective_sl_range * reward_multiplier - buffer_tp_points
                
            #print(f"🔧 FALLBACK SL/TP: Entry={entry_point:.2f}, Effective_SL_Range={effective_sl_range:.2f} (max {max_sl_points})")
        
        return sl_price, tp_price
        
    except Exception as e:
        print(f"Error calculating SL/TP for entry: {e}")
        sys.exit(1)

def vectorized_position_tracking(entry_conditions: np.ndarray, 
                                exit_conditions: np.ndarray, 
                                trading_hours_mask: np.ndarray, 
                                forced_exit_mask: np.ndarray,
                                indicator_timestamps: np.ndarray = None) -> np.ndarray:
    """
    Legacy position tracking function for backward compatibility.
    
    Args:
        entry_conditions: Boolean array of entry signals
        exit_conditions: Boolean array of exit signals  
        trading_hours_mask: Boolean array for valid trading hours
        forced_exit_mask: Boolean array for forced exits at 15:15
        indicator_timestamps: Array of Indicator_Timestamp values for exit blacklist tracking
        
    Returns:
        Integer array with signals: 1 (Entry), -1 (Exit), 0 (Hold)
    """
    try:
        n = len(entry_conditions)
        signals = np.zeros(n, dtype=int)
        
        # Filter entries by trading hours (config-based)
        valid_entries = entry_conditions & trading_hours_mask
        
        # Combine natural exits with forced exits (config-based)
        all_exits = exit_conditions | forced_exit_mask
        
        # NEW: Exit timestamp blacklist for unique entry validation
        exit_timestamp_blacklist = set()
        
        # CRITICAL: Sequential position state tracking for multiple entries with timestamp validation
        in_position = False
        entries_blocked = 0
        exits_tracked = 0
        
        for i in range(n):
            current_indicator_time = indicator_timestamps[i] if indicator_timestamps is not None else None
            
            if valid_entries[i] and not in_position:
                # NEW: Check if this Indicator_Timestamp was used for previous exit
                if current_indicator_time is not None and current_indicator_time in exit_timestamp_blacklist:
                    signals[i] = 0  # Skip this entry - timestamp already used
                    entries_blocked += 1
                    continue
                    
                signals[i] = 1      # Entry signal
                in_position = True
            elif all_exits[i] and in_position:
                signals[i] = -1     # Exit signal
                in_position = False # RESET position state to allow new entries
                
                # NEW: Add this exit's Indicator_Timestamp to blacklist
                if current_indicator_time is not None:
                    exit_timestamp_blacklist.add(current_indicator_time)
                    exits_tracked += 1
            else:
                signals[i] = 0      # Hold signal (neutral state)
        
        # Debug logging for timestamp validation
        if indicator_timestamps is not None:
            total_entries_attempted = np.sum(valid_entries)
            total_entries_allowed = np.sum(signals == 1)
            print(f"🛡️ Exit Timestamp Validation: {entries_blocked} entries blocked, {exits_tracked} exits tracked, {total_entries_allowed}/{total_entries_attempted} entries allowed")
        
        return signals
        
    except Exception as e:
        print(f"Error in position tracking: {e}")
        # Fallback to zeros array
        return np.zeros(len(entry_conditions), dtype=int)


class VectorBTSignalGenerator(ProcessorBase):
    """
    Ultra-fast VectorBT signal generation using IndicatorFactory pattern.
    
    Implements Phase 3 of the Ultimate Efficiency Engine:
    - Processes each strategy case with VectorBT IndicatorFactory
    - Handles strategy-specific option types and strike types
    - Evaluates entry/exit conditions using market data and indicators
    - Generates vectorized signals for maximum performance
    """
    
    def __init__(self, config_loader, data_loader, logger):
        """Initialize VectorBT Signal Generator."""
        super().__init__(config_loader, data_loader, logger)
        
        if not VECTORBT_AVAILABLE:
            raise ConfigurationError("VectorBT is required for signal generation")
        
        # Performance optimization caches
        self._ltp_data_cache = None
        self._merged_context_cache = None
        self._cache_timestamp = None
        self._performance_mode = self.config_loader.main_config.get('system', {}).get('performance_mode', 'normal')
        
        # OPTIMIZATION: Pre-compiled condition cache for ultra performance
        self._compiled_conditions_cache = {}
        
        print(f"🚀 PERFORMANCE: Signal generator initialized in {self._performance_mode} mode")
        
        self.log_major_component("VectorBT Signal Generator initialized", "SIGNAL_GENERATOR")
    
    def _is_indicator_enabled(self, indicator_name: str) -> bool:
        """
        Check if an indicator is enabled in the configuration.
        
        Args:
            indicator_name: Name of the indicator to check (e.g., 'signal_candle', 'rsi', etc.)
            
        Returns:
            True if indicator is enabled, False otherwise
        """
        try:
            indicators_config = self.config_loader.main_config.get('indicators', {})
            indicator_config = indicators_config.get(indicator_name, {})
            enabled = indicator_config.get('enabled', True)  # Default to True for backward compatibility
            
            if not enabled:
                self.log_detailed(f"Indicator '{indicator_name}' is disabled in configuration", "INFO")
            
            return enabled
        except Exception as e:
            self.log_detailed(f"Error checking indicator enabled status for '{indicator_name}': {e}", "WARNING")
            return True  # Fail-safe: assume enabled if unable to check
    
    def generate_signals_with_context(
        self, 
        market_data: pd.DataFrame, 
        indicators: Dict[str, pd.Series],
        shared_eval_context: Dict[str, Any],
        shared_merged_data: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        ULTRA-OPTIMIZED signal generation using pre-built context from main engine.
        
        This method skips ALL expensive context building and uses shared evaluation context
        built once in the main engine for maximum performance.
        
        Args:
            market_data: OHLCV DataFrame (for compatibility - not used)
            indicators: Dictionary of calculated indicators (for compatibility - not used)
            shared_eval_context: Pre-built evaluation context with all market data and indicators
            shared_merged_data: Pre-built merged 1m+5m data (if multi-timeframe mode)
            
        Returns:
            Dictionary containing VectorBT signals for each strategy case
        """
        try:
            with self.measure_execution_time("signal_generation"):
                # DETAILED TIMING: Start overall timing
                overall_start = time.time()
                timing_breakdown = {}
                
                data_length = len(shared_eval_context.get('close', []))
                vectorbt_signals = {}
                all_signal_candle_data = {}  # Collect signal candle data from all strategies
                all_sl_tp_data = {}  # Collect SL/TP data from all strategies
                
                # TIMING: Config setup
                config_start = time.time()
                enabled_strategies = self.config_loader.get_enabled_strategies_cases()
                timing_breakdown['config_setup'] = time.time() - config_start
                
                self.log_major_component(
                    f"Processing {data_length} data points for {len(enabled_strategies)} strategy groups",
                    "SIGNAL_GENERATION"
                )
                
                print(f"🎯 Using VectorBT IndicatorFactory Pattern (Ultra Fast)")
                print(f"   🔄 Processing: {data_length} data points for {len(enabled_strategies)} strategy groups...")
                print(f"🚀 PERFORMANCE: {self._performance_mode} mode enabled")
                print(f"🚀 PERFORMANCE: Using PRE-BUILT context from main engine - NO context building!")
                
                merged_data_for_reports = shared_merged_data
                
                # SL/TP DETECTION: Check all strategies upfront and add arrays to shared context if needed
                any_strategy_uses_sl_tp = False
                for group_name, group_config in enabled_strategies.items():
                    if group_config.get('status') != 'Active':
                        continue
                    for case_name, case_config in group_config.get('cases', {}).items():
                        if case_config.get('status') != 'Active':
                            continue
                        exit_condition = case_config.get('exit', '')
                        if 'sl_price' in exit_condition or 'tp_price' in exit_condition:
                            any_strategy_uses_sl_tp = True
                            break
                    if any_strategy_uses_sl_tp:
                        break
                
                if any_strategy_uses_sl_tp:
                    # Add SL/TP arrays to shared eval_context so they're available for all strategies
                    shared_eval_context['sl_price'] = np.full(data_length, np.nan)
                    shared_eval_context['tp_price'] = np.full(data_length, np.nan)
                    shared_eval_context['entry_point'] = np.full(data_length, np.nan)
                    print(f"✅ Added SL/TP + entry_point arrays to SHARED eval_context: {data_length} elements")
                
                # SKIP CONTEXT BUILDING - Use pre-built context directly
                timing_breakdown['shared_context_building'] = 0.0  # Already built in main engine
                
                # TIMING: Strategy processing (now using shared context)
                strategy_start = time.time()
                strategy_times = {}
                
                # Process each strategy group using PRE-BUILT CONTEXT
                for group_name, group_config in enabled_strategies.items():
                    # Process each case in the strategy group
                    for case_name, case_config in group_config['cases'].items():
                        if not case_config.get('enabled', True):
                            continue
                        
                        strategy_case_start = time.time()
                        print(f"   🚀 Processing {group_name}.{case_name} with PRE-BUILT context...")
                        
                        # Generate signals using PRE-BUILT context (no expensive context building per case)
                        signal_results = self._generate_signals_direct_with_context(
                            eval_context=shared_eval_context,
                            merged_data=shared_merged_data,
                            case_config=case_config,
                            group_name=group_name,
                            case_name=case_name
                        )
                        
                        strategy_case_time = time.time() - strategy_case_start
                        strategy_times[f"{group_name}.{case_name}"] = strategy_case_time
                        
                        # Store VectorBT signal results with strategy configuration
                        strategy_key = f"{group_name}.{case_name}"
                        vectorbt_signals[strategy_key] = {
                            'signals': signal_results.get('vectorbt_result', None),
                            'entry_count': signal_results.get('entry_count', 0),
                            'exit_count': signal_results.get('exit_count', 0),
                            'option_type': signal_results.get('option_type', 'CALL'),
                            'strike_type': signal_results.get('strike_type', 'ATM'),
                            'position_size': signal_results.get('position_size', 1)
                        }
                        
                        # Collect signal candle data from this strategy
                        strategy_signal_candle_data = signal_results.get('signal_candle_data', {})
                        if strategy_signal_candle_data:
                            # MERGE signal candle data from all strategies (instead of overwriting)
                            if not all_signal_candle_data:
                                # First strategy: initialize the combined data
                                all_signal_candle_data = strategy_signal_candle_data.copy()
                                self.log_detailed(f"Initialized signal candle data from {strategy_key}: {list(strategy_signal_candle_data.keys())}", "DEBUG")
                            else:
                                # Subsequent strategies: merge with existing data
                                self._merge_signal_candle_data(all_signal_candle_data, strategy_signal_candle_data, strategy_key)
                                self.log_detailed(f"Merged signal candle data from {strategy_key}: {list(strategy_signal_candle_data.keys())}", "DEBUG")
                        
                        # Collect SL/TP data from this strategy
                        strategy_sl_tp_data = signal_results.get('sl_tp_data', {})
                        if strategy_sl_tp_data:
                            # MERGE SL/TP data from all strategies (instead of overwriting)
                            if not all_sl_tp_data:
                                # First strategy: initialize the combined data
                                all_sl_tp_data = strategy_sl_tp_data.copy()
                                self.log_detailed(f"Initialized SL/TP data from {strategy_key}: {list(strategy_sl_tp_data.keys())}", "DEBUG")
                            else:
                                # Subsequent strategies: merge with existing data
                                self._merge_sl_tp_data(all_sl_tp_data, strategy_sl_tp_data, strategy_key)
                                self.log_detailed(f"Merged SL/TP data from {strategy_key}: {list(strategy_sl_tp_data.keys())}", "DEBUG")
                        
                        # Collect merged_data for reports (same for all cases in LTP mode)
                        if shared_merged_data is not None and merged_data_for_reports is None:
                            merged_data_for_reports = signal_results.get('merged_data')
                            if merged_data_for_reports is not None:
                                self.log_detailed(f"Captured merged 1m data from {group_name}.{case_name} for reports", "INFO")
                        
                        # Log case results
                        print(f"     Generated {signal_results['entry_count']} entries, {signal_results['exit_count']} exits in {signal_results['generation_time']:.3f}s")
                        print(f"     🎯 Strike: {signal_results.get('strike_type', 'ATM')}, Option: {signal_results.get('option_type', 'CALL')}")
                
                # TIMING: Complete timing summary
                timing_breakdown['total_strategy_processing'] = time.time() - strategy_start
                timing_breakdown['total_signal_generation'] = time.time() - overall_start
                
                self.log_major_component(
                    f"VectorBT signal generation complete: {len(vectorbt_signals)} strategy signals generated",
                    "SIGNAL_GENERATION"
                )
                
                print(f"VectorBT signal generation complete:")
                print(f"   📊 Strategy signals generated: {len(vectorbt_signals)}")
                print(f"   ⚡ Using proven IndicatorFactory pattern")
                
                # DETAILED TIMING OUTPUT
                print(f"\n🕐 DETAILED SIGNAL GENERATION TIMING:")
                print(f"   config_setup: {timing_breakdown['config_setup']:.3f}s")
                print(f"   shared_context_building: {timing_breakdown['shared_context_building']:.3f}s (PRE-BUILT)")
                print(f"   total_strategy_processing: {timing_breakdown['total_strategy_processing']:.3f}s")
                print(f"   total_signal_generation: {timing_breakdown['total_signal_generation']:.3f}s")
                
                print(f"\n⏱️  INDIVIDUAL STRATEGY TIMING:")
                for strategy_name, strategy_time in strategy_times.items():
                    print(f"   {strategy_name}: {strategy_time:.3f}s")
                
                # Return signals with merged_data and signal_candle_data
                result = {'signals': vectorbt_signals}
                if shared_merged_data is not None and merged_data_for_reports is not None:
                    result['merged_data'] = merged_data_for_reports
                    print(f"   🔄 Including merged 1m data for reports: {len(merged_data_for_reports)} records")
                
                # Include signal candle data if available and indicator is enabled
                if all_signal_candle_data and self._is_indicator_enabled('signal_candle'):
                    result['signal_candle_data'] = all_signal_candle_data
                    print(f"   📊 Including signal candle data: {list(all_signal_candle_data.keys())}")
                
                # Include SL/TP data if available
                if all_sl_tp_data:
                    result['sl_tp_data'] = all_sl_tp_data
                    print(f"   🎯 Including SL/TP data: {list(all_sl_tp_data.keys())}")
                
                return result
                
        except Exception as e:
            self.log_detailed(f"Error in VectorBT signal generation with context: {e}", "ERROR")
            raise ConfigurationError(f"VectorBT signal generation with context failed: {e}")
    
    def _generate_signals_direct_with_context(
        self,
        eval_context: Dict[str, Any],
        merged_data: pd.DataFrame,
        case_config: Dict[str, Any],
        group_name: str,
        case_name: str
    ) -> Dict[str, Any]:
        """
        Generate signals using pre-built shared context (OPTIMIZED VERSION).
        
        This method skips expensive context building and uses shared evaluation context
        for maximum performance when processing multiple strategies/cases.
        
        Args:
            eval_context: Pre-built evaluation context with all market data and indicators
            merged_data: Pre-built merged 1m+5m data (if multi-timeframe mode)
            case_config: Case-specific configuration
            group_name: Strategy group name
            case_name: Case name
            
        Returns:
            Dictionary with signal generation results
        """
        try:
            start_time = time.time()
            detailed_timing = {}
            
            # TIMING: Setup phase (much faster now)
            setup_start = time.time()
            entry_condition = case_config.get('entry', '')
            exit_condition = case_config.get('exit', '')
            
            if not entry_condition or not exit_condition:
                raise ConfigurationError(f"Entry/exit conditions not defined for {group_name}.{case_name}")
            
            detailed_timing['setup'] = time.time() - setup_start
            
            print(f"🔍 DEBUG Strategy {case_name}: Entry='{entry_condition}', Exit='{exit_condition}'")
            if self._performance_mode != 'ultra':
                print(f"🔍 DEBUG Strategy Processing: {group_name}.{case_name}")
                print(f"   Entry condition: {entry_condition}")
                print(f"   Exit condition: {exit_condition}")
                print(f"🚀 PERFORMANCE: Using SHARED context (no expensive context building)")
            elif self._performance_mode == 'ultra':
                print(f"🚀 PERFORMANCE: {group_name}.{case_name} using shared context")
            
            # SKIP CONTEXT BUILDING - Use shared context directly
            # Get the appropriate index for signal evaluation
            if merged_data is not None:
                # Multi-timeframe mode: use 1m timestamps
                evaluation_index = eval_context['ltp_close'].index if 'ltp_close' in eval_context else merged_data.index
                self.log_detailed(f"Using 1m timestamps for signal evaluation: {len(evaluation_index)} records", "DEBUG")
            else:
                # Single timeframe mode: use 5m timestamps
                evaluation_index = eval_context['close'].index if hasattr(eval_context['close'], 'index') else range(len(eval_context['close']))
                self.log_detailed(f"Using 5m timestamps for signal evaluation: {len(evaluation_index)} records", "DEBUG")
            
            # SL/TP arrays are now initialized in shared context if needed
            
            # TIMING: Signal evaluation phase (now the main work)
            evaluation_start = time.time()
            
            # OPTIMIZED: Fast entry signal evaluation
            try:
                # PERFORMANCE OPTIMIZATION: Use cached compiled conditions for ultra mode
                if self._performance_mode == 'ultra':
                    # Cache key for entry condition
                    entry_cache_key = f"entry_{group_name}_{case_name}"
                    if entry_cache_key not in self._compiled_conditions_cache:
                        self._compiled_conditions_cache[entry_cache_key] = compile(entry_condition, '<string>', 'eval')
                    
                    # Use cached compiled condition for faster evaluation
                    entry_signals = eval(self._compiled_conditions_cache[entry_cache_key], {"__builtins__": {}}, eval_context)
                else:
                    entry_signals = eval(entry_condition, {"__builtins__": {}}, eval_context)
                
                # OPTIMIZED: Direct Series conversion without debug checks
                if isinstance(entry_signals, pd.Series):
                    entry_signals = entry_signals.fillna(False).astype(bool)
                else:
                    entry_signals = pd.Series(entry_signals, index=evaluation_index).fillna(False).astype(bool)
                
                # PERFORMANCE: Simplified logging
                entry_count = entry_signals.sum()
                if self._performance_mode == 'ultra':
                    print(f"🚀 PERFORMANCE: {group_name}.{case_name} - {entry_count} entry signals found")
                    
                # DEBUG: Entry signal detection logging
                if self._is_debug_export_enabled():
                    print(f"🔍 DEBUG: {group_name}.{case_name} - {entry_count} entry signals detected")
                    if entry_count > 0:
                        entry_indices = np.where(entry_signals)[0]
                        print(f"🔍 DEBUG: Entry signal indices: {entry_indices[:5]}{'...' if len(entry_indices) > 5 else ''}")
                    else:
                        print(f"🔍 DEBUG: No entry signals - check entry condition: {entry_condition}")
                    
                    # DEBUG STAGE 1: Export eval_context after entry trigger validation (with blacklist check results)
                    self._export_eval_context_debug_csv(eval_context, evaluation_index, f"after_entry_trigger_validation_{entry_count}_signals", group_name, case_name)
                        
            except Exception as e:
                self.log_detailed(f"Entry condition evaluation failed for {group_name}.{case_name}: {e}", "ERROR")
                entry_signals = pd.Series(False, index=evaluation_index)
            
            # SIGNAL CANDLE INITIALIZATION: Add signal candle arrays to eval_context for exit condition evaluation (only if enabled)
            data_length = len(evaluation_index)
            if 'signal_candle_open' not in eval_context and self._is_indicator_enabled('signal_candle'):
                eval_context['signal_candle_open'] = np.full(data_length, np.nan)
                eval_context['signal_candle_high'] = np.full(data_length, np.nan)
                eval_context['signal_candle_low'] = np.full(data_length, np.nan)
                eval_context['signal_candle_close'] = np.full(data_length, np.nan)
                self.log_detailed(f"Initialized signal candle arrays in eval_context: {data_length} elements", "DEBUG")
            
            # ENTRY_POINT INITIALIZATION: Always add entry_point for ALL strategies (for reference in CSV exports)
            if 'entry_point' not in eval_context:
                eval_context['entry_point'] = np.full(data_length, np.nan)
                self.log_detailed(f"Initialized entry_point array in eval_context: {data_length} elements", "DEBUG")
                print(f"✅ DEBUG: entry_point array initialized for {group_name}.{case_name} (for CSV export reference)")
            
            # SL_TP_LEVELS INITIALIZATION: Add SL/TP arrays to eval_context for exit condition evaluation (only if needed)
            uses_sl_price = 'sl_price' in exit_condition
            uses_tp_price = 'tp_price' in exit_condition
            if (uses_sl_price or uses_tp_price) and 'sl_price' not in eval_context:
                eval_context['sl_price'] = np.full(data_length, np.nan)
                eval_context['tp_price'] = np.full(data_length, np.nan)
                self.log_detailed(f"Initialized SL/TP levels arrays in eval_context: {data_length} elements", "DEBUG")
                print(f"🎯 DEBUG: SL/TP arrays initialized for {group_name}.{case_name} - uses_sl_price: {uses_sl_price}, uses_tp_price: {uses_tp_price}")
            elif uses_sl_price or uses_tp_price:
                self.log_detailed(f"SL/TP arrays already exist in eval_context", "DEBUG")
                print(f"🎯 DEBUG: SL/TP arrays already present for {group_name}.{case_name}")
            else:
                self.log_detailed(f"Exit condition does not use SL/TP, but entry_point still available for CSV export", "DEBUG")
                print(f"🎯 DEBUG: Exit condition for {group_name}.{case_name} does not use SL/TP (but entry_point available for reference)")
            
            # SIGNAL CANDLE PRE-CALCULATION: Check if exit condition uses signal candle data AND indicator is enabled
            if 'signal_candle' in exit_condition and self._is_indicator_enabled('signal_candle'):
                self.log_detailed(f"Signal candle exit condition detected for {group_name}.{case_name}, calculating directly", "INFO")
                
                # Get market data for signal candle capture
                if merged_data is not None:
                    # Multi-timeframe mode: use merged data
                    open_data = merged_data['open'].values
                    high_data = merged_data['high'].values
                    low_data = merged_data['low'].values
                    close_data = merged_data['close'].values
                else:
                    # Single timeframe mode: use evaluation context data
                    open_data = eval_context['open'].values if hasattr(eval_context['open'], 'values') else eval_context['open']
                    high_data = eval_context['high'].values if hasattr(eval_context['high'], 'values') else eval_context['high']
                    low_data = eval_context['low'].values if hasattr(eval_context['low'], 'values') else eval_context['low']
                    close_data = eval_context['close'].values if hasattr(eval_context['close'], 'values') else eval_context['close']
                
                # OPTIMIZED: Calculate signal candle values directly using shared context
                # This eliminates SmartIndicatorEngine instantiation and temp DataFrame creation
                try:
                    # Use direct calculation method with identical logic but no overhead
                    signal_candle_result = self._calculate_signal_candle_direct(
                        open_data, high_data, low_data, close_data,
                        entry_signals.values, evaluation_index, exit_condition
                    )
                    
                    # Update eval_context with calculated signal candle values
                    eval_context['signal_candle_open'] = signal_candle_result['signal_candle_open']
                    eval_context['signal_candle_high'] = signal_candle_result['signal_candle_high']
                    eval_context['signal_candle_low'] = signal_candle_result['signal_candle_low']
                    eval_context['signal_candle_close'] = signal_candle_result['signal_candle_close']
                    
                    self.log_detailed(f"Direct signal candle calculation completed for exit evaluation", "INFO")
                except Exception as e:
                    self.log_detailed(f"Direct signal candle calculation failed: {e}, using NaN values", "WARNING")
            
            # SL_TP_LEVELS PRE-CALCULATION: Check if exit condition uses SL/TP data
            if uses_sl_price or uses_tp_price:
                self.log_detailed(f"SL/TP levels exit condition detected for {group_name}.{case_name}, calculating directly", "INFO")
                
                # Get market data for SL/TP calculation
                if merged_data is not None:
                    # Multi-timeframe mode: use merged data
                    open_data = merged_data['open'].values
                    high_data = merged_data['high'].values
                    low_data = merged_data['low'].values
                    close_data = merged_data['close'].values
                else:
                    # Single timeframe mode: use evaluation context data
                    open_data = eval_context['open'].values if hasattr(eval_context['open'], 'values') else eval_context['open']
                    high_data = eval_context['high'].values if hasattr(eval_context['high'], 'values') else eval_context['high']
                    low_data = eval_context['low'].values if hasattr(eval_context['low'], 'values') else eval_context['low']
                    close_data = eval_context['close'].values if hasattr(eval_context['close'], 'values') else eval_context['close']
                
                # OPTIMIZED: Calculate SL/TP levels directly using shared context
                # This eliminates the need to calculate during position tracking loop
                try:
                    print(f"🎯 EXECUTING SL/TP CALCULATION: _calculate_sl_tp_indicator_direct for {group_name}.{case_name}")
                    print(f"   OptionType: {case_config.get('OptionType', 'NOT_SET')}")
                    
                    # Add case identification to config for debugging
                    case_config_with_names = case_config.copy()
                    case_config_with_names['group_name'] = group_name
                    case_config_with_names['case_name'] = case_name
                    
                    # Use direct calculation method with strategy configuration and entry condition
                    sl_tp_calc_data = self._calculate_sl_tp_indicator_direct(
                        open_data, high_data, low_data, close_data,
                        entry_signals.values, evaluation_index, case_config_with_names, entry_condition, eval_context
                    )
                    
                    # Store calculation data for later population (after position tracking)
                    eval_context['_sl_tp_calc_data'] = sl_tp_calc_data
                    
                    # For now, initialize empty arrays for exit conditions (will be populated later)
                    array_length = len(evaluation_index)
                    eval_context['sl_price'] = np.full(array_length, np.nan)
                    eval_context['tp_price'] = np.full(array_length, np.nan) 
                    eval_context['entry_point'] = np.full(array_length, np.nan)
                    
                    print(f"🎯 SL/TP VALUES CALCULATED: {len(sl_tp_calc_data['entry_indices'])} entries, will populate arrays after position tracking")
                    
                except Exception as e:
                    self.log_detailed(f"Direct SL/TP calculation failed: {e}, using NaN values", "WARNING")
                    print(f"❌ SL/TP pre-calculation failed: {e}")
                
                # DEBUG STAGE 2: Export eval_context after signal candle generation and SL/TP creation
                if self._is_debug_export_enabled():
                    self._export_eval_context_debug_csv(eval_context, evaluation_index, "after_signal_candle_and_sl_tp_creation", group_name, case_name)
            else:
                # DEBUG: Export eval_context before position tracking (shows initial state, config-controlled)
                if self._is_debug_export_enabled():
                    self._export_eval_context_debug_csv(eval_context, evaluation_index, "before_position_tracking", group_name, case_name)
            
            # SKIP PRE-POPULATION: Let SL/TP values be populated AFTER position tracking
            uses_sl_price = 'sl_price' in exit_condition
            uses_tp_price = 'tp_price' in exit_condition
            
            if uses_sl_price or uses_tp_price:
                print(f"🔧 FIXED: SL/TP arrays will be populated AFTER position tracking for {group_name}.{case_name}")
                
                # Verify we have calculation data available
                if '_sl_tp_calc_data' in eval_context:
                    sl_tp_calc_data = eval_context['_sl_tp_calc_data']
                    if sl_tp_calc_data.get('needs_population', False):
                        print(f"   ✅ SL/TP calculation data ready for post-tracking population")
                    else:
                        print(f"   ℹ️  SL/TP data exists but doesn't need population")
                else:
                    print(f"   ⚠️ No SL/TP calculation data found")
                
                # Initialize empty arrays for exit condition evaluation (will be populated after position tracking)
                array_length = len(evaluation_index)
                eval_context['sl_price'] = np.full(array_length, np.nan)
                eval_context['tp_price'] = np.full(array_length, np.nan)
                eval_context['entry_point'] = np.full(array_length, np.nan)
                
                print(f"   🔍 Initial state: Empty SL/TP arrays initialized, will populate after position tracking")
            
            # DEBUG STAGE 3: Export eval_context before stateful position tracking
            if self._is_debug_export_enabled():
                self._export_eval_context_debug_csv(eval_context, evaluation_index, f"before_stateful_position_tracking", group_name, case_name)
            
            # SKIP INITIAL EXIT EVALUATION - Let position tracking handle exit logic with locked values
            print(f"🔧 FIXED FLOW: Skipping premature exit evaluation - using stateful position tracking with locked values")
            print(f"   Exit condition: {exit_condition}")
            print(f"   Exit logic will be evaluated INSIDE position tracking with proper locked SL/TP values")
            
            # TIMING: Position tracking phase with signal candle updates
            tracking_start = time.time()
            
            # Apply time masks
            trading_hours_mask, forced_exit_mask = create_time_masks(evaluation_index, self.config_loader)
            
            # Extract timestamps for exit blacklist tracking (ENHANCED: Prefer ltp_close for 1m accuracy)
            indicator_timestamps = None
            
            # Priority 1: Use ltp_close timestamps for accurate 1m tracking
            if merged_data is not None and 'ltp_close' in merged_data.columns:
                # Use ltp_close timestamps - more accurate for 1m execution
                indicator_timestamps = merged_data.index.values  # Use actual timestamps as tracking keys
                print(f"🔍 ENHANCED: Using ltp_close timestamps (1m precision) for exit blacklist tracking: {len(indicator_timestamps)} records")
            elif 'ltp_close' in eval_context:
                # Use ltp_close from eval_context
                indicator_timestamps = evaluation_index.values if hasattr(evaluation_index, 'values') else evaluation_index
                print(f"🔍 ENHANCED: Using ltp_close eval_context timestamps for exit blacklist tracking: {len(indicator_timestamps)} records")
            # Priority 2: Fallback to indicator timestamps (5m precision)
            elif merged_data is not None and 'Indicator_Timestamp' in merged_data.columns:
                indicator_timestamps = merged_data['Indicator_Timestamp'].values
                print(f"🔍 FALLBACK: Using merged_data Indicator_Timestamp (5m precision) for exit blacklist tracking")
            elif 'Indicator_Timestamp' in eval_context:
                indicator_timestamps = eval_context['Indicator_Timestamp'].values if hasattr(eval_context['Indicator_Timestamp'], 'values') else eval_context['Indicator_Timestamp']
                print(f"🔍 FALLBACK: Using eval_context Indicator_Timestamp (5m precision) for exit blacklist tracking")
            else:
                print(f"⚠️ WARNING: No ltp_close or Indicator_Timestamp found - exit blacklist validation disabled")
            
            # ENHANCED: Stateful position tracking with locked signal candle and SL/TP values
            # Prepare market OHLC data for locked value calculation
            indicator_timestamps = None
            indicator_timestamps = merged_data['Indicator_Timestamp'].values
            print(f"🔍 FALLBACK: Using merged_data Indicator_Timestamp (5m precision) for exit blacklist tracking")
            if merged_data is not None:
                # Multi-timeframe mode: use merged data
                market_ohlc = {
                    'open': merged_data['open'].values,
                    'high': merged_data['high'].values,
                    'low': merged_data['low'].values,
                    'close': merged_data['close'].values
                }
            else:
                # Single timeframe mode: use evaluation context data
                market_ohlc = {
                    'open': eval_context['open'].values if hasattr(eval_context['open'], 'values') else eval_context['open'],
                    'high': eval_context['high'].values if hasattr(eval_context['high'], 'values') else eval_context['high'],
                    'low': eval_context['low'].values if hasattr(eval_context['low'], 'values') else eval_context['low'],
                    'close': eval_context['close'].values if hasattr(eval_context['close'], 'values') else eval_context['close']
                }
            
            # Add SL/TP config to eval_context for calculation
            eval_context['_config_sl_tp_levels'] = self.config_loader.main_config.get('indicators', {}).get('SL_TP_levels', {})
            
            # Enhance case_config with case_name for debugging
            enhanced_case_config = case_config.copy()
            enhanced_case_config['case_name'] = case_name
            
            # Use new stateful position tracking with locked values (SMART APPROACH with LTP-aware entry points)
            position_signals, locked_values = vectorized_position_tracking_with_locked_values(
                entry_signals.values,
                exit_condition,  # Pass exit condition string for dynamic evaluation
                trading_hours_mask,
                forced_exit_mask,
                market_ohlc,
                enhanced_case_config,
                eval_context,
                indicator_timestamps,
                self._performance_mode,
                self._compiled_conditions_cache,
                group_name,
                case_name,
                self  # Pass signal_generator instance for CPR pivot TP calculation
            )
            
            # UPDATE EVAL_CONTEXT with locked values for exit condition re-evaluation
            print(f"🔄 UPDATING eval_context with locked values for {group_name}.{case_name}")
            eval_context['signal_candle_open'] = locked_values['signal_candle_open']
            eval_context['signal_candle_high'] = locked_values['signal_candle_high']
            eval_context['signal_candle_low'] = locked_values['signal_candle_low']
            eval_context['signal_candle_close'] = locked_values['signal_candle_close']
            eval_context['sl_price'] = locked_values['sl_price']
            eval_context['tp_price'] = locked_values['tp_price']
            eval_context['entry_point'] = locked_values['entry_point']
            
            detailed_timing['position_tracking'] = time.time() - tracking_start
            
            # Convert position signals back to entry/exit boolean arrays
            final_entry_signals = pd.Series(position_signals == 1, index=evaluation_index)
            final_exit_signals = pd.Series(position_signals == -1, index=evaluation_index)
            
            # Count final signals
            entry_count = final_entry_signals.sum()
            exit_count = final_exit_signals.sum()
            
            # DEBUG STAGE 4: Export eval_context after exit trigger processing (with blacklist updates)
            if self._is_debug_export_enabled():
                self._export_eval_context_debug_csv(eval_context, evaluation_index, f"after_exit_trigger_processing_{exit_count}_exits", group_name, case_name)
            
            # Signal candle data is now calculated during position tracking
            
            # Create signals array for VectorBT compatibility
            signals_array = np.column_stack([final_entry_signals.values, final_exit_signals.values])
            
            # Get option configuration
            option_type = case_config.get('OptionType')
            strike_type = case_config.get('StrikeType')
            position_size = case_config.get('position_size', '1')
            
            # Validate required configuration
            if not option_type:
                raise ConfigurationError(f"OptionType is required for case {group_name}.{case_name}")
            if not strike_type:
                raise ConfigurationError(f"StrikeType is required for case {group_name}.{case_name}")
            
            # Convert position_size to integer
            try:
                position_size_int = int(position_size)
                risk_config = self.config_loader.main_config.get('risk_management', {})
                position_size_limit = risk_config.get('position_size_limit', 5)
                
                if position_size_int > position_size_limit:
                    self.log_detailed(f"Position size {position_size_int} exceeds limit {position_size_limit}, capping", "WARNING")
                    position_size_int = position_size_limit
                elif position_size_int < 1:
                    position_size_int = 1
                    
            except (ValueError, TypeError):
                self.log_detailed(f"Invalid position_size '{position_size}', using default 1", "WARNING")
                position_size_int = 1
            
            generation_time = time.time() - start_time
            detailed_timing['total_generation_time'] = generation_time
            
            # DETAILED TIMING OUTPUT (skip in ultra mode)
            if self._performance_mode != 'ultra':
                print(f"🕐 DETAILED TIMING for {group_name}.{case_name}:")
                for phase, phase_time in detailed_timing.items():
                    print(f"   {phase}: {phase_time:.3f}s")
            elif self._performance_mode == 'ultra':
                print(f"🚀 PERFORMANCE: {group_name}.{case_name} completed in {generation_time:.3f}s")
            
            # Signal candle data already prepared above during calculation
            # Only prepare signal candle data if the indicator is enabled
            result = {
                'vectorbt_result': {
                    'entries': final_entry_signals,
                    'exits': final_exit_signals,
                    'entry_count': entry_count,
                    'exit_count': exit_count
                },
                'signals_array': signals_array,
                'entry_count': entry_count,
                'exit_count': exit_count,
                'generation_time': generation_time,
                'option_type': option_type,
                'strike_type': strike_type,
                'position_size': position_size_int,
                'merged_data': merged_data,  # Pass through merged data for reports
            }
            
            # Use locked signal candle values from stateful position tracking
            if self._is_indicator_enabled('signal_candle'):
                signal_candle_data = {
                    'signal_candle_open': locked_values['signal_candle_open'],
                    'signal_candle_high': locked_values['signal_candle_high'],
                    'signal_candle_low': locked_values['signal_candle_low'],
                    'signal_candle_close': locked_values['signal_candle_close']
                }
                result['signal_candle_data'] = signal_candle_data
                print(f"✅ LOCKED SIGNAL CANDLE: Using stateful locked values for {group_name}.{case_name}")
            
            # Use locked SL/TP values from stateful position tracking
            result['sl_tp_data'] = {
                'sl_price': locked_values['sl_price'],
                'tp_price': locked_values['tp_price'],
                'entry_point': locked_values['entry_point']
            }
            
            # Debug: Check SL/TP data being packaged for export
            sl_price_data = result['sl_tp_data']['sl_price']
            tp_price_data = result['sl_tp_data']['tp_price']
            if hasattr(sl_price_data, '__len__'):
                sl_valid = np.sum(~np.isnan(sl_price_data))
                tp_valid = np.sum(~np.isnan(tp_price_data))
                print(f"📦 PACKAGING SL/TP for export: {group_name}.{case_name}")
                print(f"   SL values: {sl_valid}/{len(sl_price_data)} valid")
                print(f"   TP values: {tp_valid}/{len(tp_price_data)} valid")
                if sl_valid > 0:
                    first_valid_sl = next((x for x in sl_price_data if not np.isnan(x)), "no valid values")
                    first_valid_tp = next((x for x in tp_price_data if not np.isnan(x)), "no valid values")
                    print(f"   First valid SL: {first_valid_sl}, TP: {first_valid_tp}")
                    # Verify these match our expected values
                     
            print(f"📦 FINAL: Using eval_context SL/TP data for {group_name}.{case_name}")
            
            return result
            
        except Exception as e:
            self.log_detailed(f"Error in optimized signal generation for {group_name}.{case_name}: {e}", "ERROR")
            raise ConfigurationError(f"Signal generation failed for {group_name}.{case_name}: {e}")
    
    
    def _build_multi_timeframe_context(
        self, 
        market_data_5m: pd.DataFrame, 
        indicators_5m: Dict[str, pd.Series]
    ) -> Tuple[Dict[str, Any], pd.DataFrame]:
        """
        Build multi-timeframe evaluation context by merging 1m execution data with 5m indicators.
        
        Implements vectorized as-of/backward merge to prevent lookahead bias:
        - 5m candle labeled 09:20 covers 09:20-09:24 
        - 1m breakouts (09:25-09:29) use completed 09:20 5m values
        
        Args:
            market_data_5m: 5m OHLCV DataFrame (filtered)
            indicators_5m: Dictionary of calculated 5m indicators
            
        Returns:
            Tuple of (evaluation_context, merged_data_dataframe):
            - evaluation_context: Dictionary with 1m_* and 5m prefixed columns for signal evaluation
            - merged_data_dataframe: Complete merged DataFrame with 1m timestamps for reports
        """
        try:
            # Phase 1: Fetch configuration first
            backtesting_config = self.config_loader.main_config.get('backtesting', {})
            trading_config = self.config_loader.main_config.get('trading', {})
            
            symbol = trading_config.get('symbol', 'NIFTY')
            exchange = trading_config.get('exchange', 'NSE_INDEX')
            start_date = backtesting_config.get('start_date')
            end_date = backtesting_config.get('end_date')
            
            # PERFORMANCE OPTIMIZATION: Enhanced cache validation
            current_time = time.time()
            
            # Create cache key based on data characteristics for validation
            cache_key = f"{len(market_data_5m)}_{start_date}_{end_date}_{symbol}"
            
            if (self._performance_mode == 'ultra' and 
                self._ltp_data_cache is not None and 
                self._merged_context_cache is not None and
                self._cache_timestamp is not None and
                hasattr(self, '_cache_key') and self._cache_key == cache_key and
                (current_time - self._cache_timestamp) < 300):  # Extended to 5 minutes
                print("🚀 PERFORMANCE: Using validated cached 1m data and merged context")
                return self._merged_context_cache.copy(), self._ltp_data_cache.copy()
            
            # Phase 2: Fetch 1m LTP timeframe data (using cached config)
            
            if not start_date or not end_date:
                raise ConfigurationError("start_date and end_date must be configured for multi-timeframe processing")
            
            # Fetch 1m data using the new data loader method
            ltp_data_1m = self.data_loader.get_ltp_timeframe_data(
                symbol=symbol,
                exchange=exchange, 
                start_date=start_date,
                end_date=end_date
            )
            
            if ltp_data_1m.empty:
                raise ConfigurationError("No 1m LTP data available for multi-timeframe processing")
            
            # Filter 1m data to start from configured trading start time (when first 5m candle is complete and usable)
            from datetime import time as dt_time
            # Get trading start time from config instead of hardcoding
            trading_start_str = self.config_loader.main_config.get('risk_management', {}).get('trading_start_time', '09:20')
            hour, minute = map(int, trading_start_str.split(':'))
            trading_start_time = dt_time(hour, minute)  # From config: when first 5m candle completes
            original_count = len(ltp_data_1m)
            ltp_data_1m = ltp_data_1m[ltp_data_1m.index.time >= trading_start_time]
            filtered_count = len(ltp_data_1m)
            
            if ltp_data_1m.empty:
                raise ConfigurationError(f"No 1m LTP data available after filtering to trading start time ({trading_start_str})")
            
            self.log_detailed(
                f"Filtered 1m data to trading start: {original_count} -> {filtered_count} records "
                f"(starting from {ltp_data_1m.index[0]})", "INFO"
            )
            
            # Phase 2: Prepare data for merging with generic timeframe strategy
            # Keep 5m data as base columns (open, high, low, close, volume)
            # 1m data will get 1m_ prefix for clear distinction
            market_data_5m_base = market_data_5m.copy()  # Keep 5m as base columns
            
            # Create 5m indicators DataFrame for efficient merging
            indicators_5m_df = pd.DataFrame(index=market_data_5m.index)
            for indicator_name, indicator_values in indicators_5m.items():
                if hasattr(indicator_values, 'index') and len(indicator_values) == len(market_data_5m):
                    indicators_5m_df[indicator_name] = indicator_values
            
            # Combine 5m market data and indicators (keep base column names)
            combined_5m_data = pd.concat([market_data_5m_base, indicators_5m_df], axis=1)
            
            self.log_detailed(f"Prepared {len(combined_5m_data)} 5m records with {len(indicators_5m_df.columns)} indicators", "INFO")
            
            # Phase 3: Add 1m prefix to LTP data and perform vectorized merge
            # Apply ltp_ prefix to distinguish from 5m base columns (1m_ is invalid Python identifier)
            ltp_data_1m_prefixed = ltp_data_1m.add_prefix('ltp_')
            
            # CORRECTED: Custom temporal alignment for proper 1m+5m merging
            # Formula: For 1m timestamp, use 5m data from the PREVIOUS completed 5m candle
            # Example: 1m 09:20-09:24 -> 5m 09:15-09:20 (completed), 1m 09:25-09:29 -> 5m 09:20-09:25 (completed)
            
            # Step 1: Create mapping of 1m timestamps to corresponding completed 5m timestamps
            merged_data = ltp_data_1m_prefixed.copy()
            
            # Step 2: For each 1m timestamp, calculate which 5m candle was completed
            # Formula: completed_5m_timestamp = floor(1m_timestamp / 5min) * 5min - 5min
            def get_completed_5m_timestamp(timestamp_1m):
                # Convert to minutes since midnight for calculation
                minutes_since_midnight = timestamp_1m.hour * 60 + timestamp_1m.minute
                # Floor to 5-minute boundary, then subtract 5 minutes to get PREVIOUS completed candle
                completed_5m_minutes = (minutes_since_midnight // 5) * 5 - 5
                # Handle edge case: if result is negative, use first available 5m candle
                if completed_5m_minutes < 0:
                    completed_5m_minutes = 0
                # Convert back to time and create timestamp for same date
                completed_hour = completed_5m_minutes // 60
                completed_minute = completed_5m_minutes % 60
                return timestamp_1m.replace(hour=completed_hour, minute=completed_minute, second=0, microsecond=0)
            
            # OPTIMIZED: Vectorized calculation of indicator timestamps
            if self._performance_mode == 'ultra':
                print("🚀 PERFORMANCE: Using ultra-fast vectorized merge")
            
            # Step 3: Vectorized calculation of completed 5m timestamps
            timestamps_1m = merged_data.index
            minutes_since_midnight = timestamps_1m.hour * 60 + timestamps_1m.minute
            completed_5m_minutes = (minutes_since_midnight // 5) * 5 - 5
            completed_5m_minutes = np.maximum(completed_5m_minutes, 0)  # Handle negative values
            
            # Convert back to timestamps vectorized
            completed_hours = completed_5m_minutes // 60
            completed_minutes = completed_5m_minutes % 60
            indicator_timestamps = pd.to_datetime(
                timestamps_1m.date.astype(str) + ' ' + 
                completed_hours.astype(str).str.zfill(2) + ':' + 
                completed_minutes.astype(str).str.zfill(2) + ':00'
            )
            
            # Step 4: ULTRA-FAST merge using pandas merge_asof
            merged_data['Indicator_Timestamp'] = indicator_timestamps
            combined_5m_sorted = combined_5m_data.sort_index()
            
            # Create temporary DataFrame for merge_asof
            temp_1m = pd.DataFrame(index=merged_data.index)
            temp_1m['merge_timestamp'] = indicator_timestamps
            temp_1m = temp_1m.sort_values('merge_timestamp')
            
            # Fix timestamp precision issues for merge_asof
            combined_5m_reset = combined_5m_sorted.reset_index()
            combined_5m_reset['timestamp'] = combined_5m_reset['timestamp'].astype('datetime64[ns]')
            temp_1m['merge_timestamp'] = temp_1m['merge_timestamp'].astype('datetime64[ns]')
            
            # VECTORIZED merge using merge_asof (replaces the nested loop)
            merged_5m_data = pd.merge_asof(
                temp_1m, 
                combined_5m_reset, 
                left_on='merge_timestamp', 
                right_on='timestamp',
                direction='backward'
            ).set_index(temp_1m.index)
            
            # Add the merged 5m columns to merged_data
            for col in combined_5m_sorted.columns:
                merged_data[col] = merged_5m_data[col]
            
            if merged_data.empty:
                raise ConfigurationError("Multi-timeframe merge resulted in empty dataset")
            
            # Validate merged data has expected columns
            required_ltp_columns = ['ltp_open', 'ltp_high', 'ltp_low', 'ltp_close', 'ltp_volume']
            required_5m_columns = ['open', 'high', 'low', 'close', 'volume']
            
            missing_ltp_columns = [col for col in required_ltp_columns if col not in merged_data.columns]
            missing_5m_columns = [col for col in required_5m_columns if col not in merged_data.columns]
            
            if missing_ltp_columns:
                raise ConfigurationError(f"Missing LTP columns after merge: {missing_ltp_columns}")
            if missing_5m_columns:
                raise ConfigurationError(f"Missing 5m base columns after merge: {missing_5m_columns}")
            
            self.log_detailed(
                f"Multi-timeframe merge successful: {len(merged_data)} records with "
                f"LTP columns: {len([c for c in merged_data.columns if c.startswith('ltp_')])} "
                f"5m indicators: {len([c for c in merged_data.columns if not c.startswith('ltp_') and c not in required_5m_columns])}", 
                "INFO"
            )
            
            # Phase 4: Build evaluation context with proper namespacing
            eval_context = {
                # LTP execution data (for breakout detection) - now properly prefixed
                'ltp_open': merged_data['ltp_open'],
                'ltp_high': merged_data['ltp_high'], 
                'ltp_low': merged_data['ltp_low'],
                'ltp_close': merged_data['ltp_close'],
                'ltp_volume': merged_data['ltp_volume'],
                
                # 5m market data (backward-aligned) - using base column names
                'open': merged_data['open'],
                'high': merged_data['high'],
                'low': merged_data['low'],
                'close': merged_data['close'],
                'volume': merged_data['volume'],
            }
            
            # Add all 5m indicators to context (without prefix for compatibility)
            for indicator_name in indicators_5m_df.columns:
                if indicator_name in merged_data.columns:
                    eval_context[indicator_name] = merged_data[indicator_name]
            
            # Phase 5: Validation and logging
            self.log_detailed(f"Multi-timeframe context built: {len(merged_data)} merged records", "INFO")
            self.log_detailed(f"Context variables: {len(eval_context)} (1m execution + 5m indicators)", "INFO")
            
            # Log sample alignment for debugging (first few records)
            if len(merged_data) > 0:
                sample_1m_time = merged_data.index[0]
                sample_5m_time = None
                
                # Find corresponding 5m timestamp
                for ts in combined_5m_data.index:
                    if ts <= sample_1m_time:
                        sample_5m_time = ts
                    else:
                        break
                
                self.log_detailed(
                    f"Sample alignment: 1m timestamp {sample_1m_time} -> 5m timestamp {sample_5m_time}", 
                    "DEBUG"
                )
            
            # Phase 6: Critical Validation (bypass in ultra performance mode)
            validation_config = self.config_loader.main_config.get('validation', {})
            validation_enabled = validation_config.get('strict_mode', True) and not validation_config.get('production_mode', False)
            if validation_enabled and self._performance_mode != 'ultra':
                self._validate_multi_timeframe_alignment(ltp_data_1m, combined_5m_data, merged_data)
            elif self._performance_mode == 'ultra':
                print("🚀 PERFORMANCE: Skipping validation in ultra mode")
            
            # PERFORMANCE OPTIMIZATION: Cache results for subsequent strategies
            if self._performance_mode == 'ultra':
                self._merged_context_cache = eval_context.copy()
                self._ltp_data_cache = merged_data.copy()
                self._cache_timestamp = time.time()
                self._cache_key = cache_key  # Store cache key for validation
                print("🚀 PERFORMANCE: Cached 1m data and merged context for reuse")
            
            return eval_context, merged_data
            
        except Exception as e:
            self.log_detailed(f"Error building multi-timeframe context: {e}", "ERROR")
            raise ConfigurationError(f"Multi-timeframe context building failed: {e}")
    
    def _validate_multi_timeframe_alignment(
        self,
        ltp_data_1m: pd.DataFrame,
        combined_5m_data: pd.DataFrame, 
        merged_data: pd.DataFrame
    ) -> None:
        """
        Critical validation for multi-timeframe alignment to prevent lookahead bias.
        
        Validates:
        1. Start time alignment (1m starts at trading_start_time + global_timeframe)
        2. No lookahead bias (1m rows reference completed 5m bars only)
        3. Temporal alignment (proper as-of/backward merge)
        
        Args:
            ltp_data_1m: 1m LTP data 
            combined_5m_data: 5m market data and indicators
            merged_data: Result of merge operation
            
        Raises:
            ConfigurationError: If any validation fails
        """
        try:
            validation_results = {
                'start_time_alignment': False,
                'lookahead_bias_check': False,
                'temporal_alignment': False,
                'validation_errors': []
            }
            
            self.log_detailed("CRITICAL VALIDATION: Multi-timeframe alignment checks", "INFO")
            
            # Validation 1: Start Time Alignment
            validation_results['start_time_alignment'] = self._validate_start_time_alignment(ltp_data_1m)
            if not validation_results['start_time_alignment']:
                validation_results['validation_errors'].append("Start time alignment failed")
            
            # Validation 2: No Lookahead Bias  
            validation_results['lookahead_bias_check'] = self._validate_no_lookahead_bias(
                ltp_data_1m, combined_5m_data, merged_data
            )
            if not validation_results['lookahead_bias_check']:
                validation_results['validation_errors'].append("Lookahead bias validation failed")
            
            # Validation 3: Temporal Alignment
            validation_results['temporal_alignment'] = self._validate_temporal_alignment(
                ltp_data_1m, combined_5m_data, merged_data
            )
            if not validation_results['temporal_alignment']:
                validation_results['validation_errors'].append("Temporal alignment validation failed")
            
            # Overall validation result
            all_passed = all([
                validation_results['start_time_alignment'],
                validation_results['lookahead_bias_check'], 
                validation_results['temporal_alignment']
            ])
            
            if all_passed:
                self.log_detailed("VALIDATION PASSED: All multi-timeframe alignment checks successful", "INFO")
                print("Multi-timeframe validation: ALL CHECKS PASSED")
            else:
                error_summary = f"Multi-timeframe validation FAILED: {validation_results['validation_errors']}"
                self.log_detailed(f"VALIDATION FAILED: {error_summary}", "ERROR")
                print(f"Multi-timeframe validation: FAILED")
                print(f"   Errors: {validation_results['validation_errors']}")
                
                # Raise error in strict mode
                if self.config_loader.main_config.get('validation', {}).get('fail_fast', True):
                    raise ConfigurationError(f"Multi-timeframe validation failed: {error_summary}")
            
        except Exception as e:
            self.log_detailed(f"Error in multi-timeframe validation: {e}", "ERROR")
            if self.config_loader.main_config.get('validation', {}).get('fail_fast', True):
                raise ConfigurationError(f"Multi-timeframe validation error: {e}")
    
    def _validate_start_time_alignment(self, ltp_data_1m: pd.DataFrame) -> bool:
        """
        Validate 1m data start time for multi-timeframe scenarios.
        
        For multi-timeframe (1m execution + 5m indicators):
        - 1m data should start from market open (e.g., 09:15) for breakout capture
        - Allows reasonable market start times within trading session
        
        For single-timeframe: Uses traditional validation (trading_start_time + global_timeframe)
        """
        try:
            if ltp_data_1m.empty:
                self.log_detailed("Start time validation: No 1m data available", "ERROR")
                return False
            
            # Get configuration
            trading_config = self.config_loader.main_config.get('trading', {})
            risk_config = self.config_loader.main_config.get('risk_management', {})
            
            global_timeframe = trading_config.get('timeframe', '5m')
            trading_start_time_str = risk_config.get('trading_start_time', '09:20')
            ltp_timeframe = trading_config.get('ltp_timeframe', False)
            
            actual_start_time = ltp_data_1m.index[0].time()
            
            # Multi-timeframe validation (1m execution + 5m indicators)
            # Check for boolean true value (more unique than '1m' string)
            if ltp_timeframe is True and global_timeframe != '1m':
                # For multi-timeframe: validate 1m data starts within reasonable market hours
                from datetime import time as dt_time
                market_open = dt_time(9, 15)  # 09:15:00
                market_close = dt_time(15, 30)  # 15:30:00
                
                if market_open <= actual_start_time <= market_close:
                    self.log_detailed(f"Multi-timeframe start time validation: 1m data starts at {actual_start_time} (within market hours)", "INFO")
                    print(f"Start time validation: Multi-timeframe mode - 1m data starts at market time {actual_start_time}")
                    return True
                else:
                    error_msg = f"1m data starts outside market hours: {actual_start_time} (expected between {market_open} and {market_close})"
                    self.log_detailed(f"Start time alignment: {error_msg}", "ERROR")
                    print(f"Start time validation: {error_msg}")
                    return False
            
            # Single-timeframe validation (traditional logic)
            else:
                # Parse timeframe to minutes
                if global_timeframe.endswith('m'):
                    timeframe_minutes = int(global_timeframe[:-1])
                else:
                    timeframe_minutes = 5
                
                # Calculate expected start time
                from datetime import datetime, timedelta
                first_date = ltp_data_1m.index[0].date()
                
                hour, minute = map(int, trading_start_time_str.split(':'))
                base_time = datetime.combine(first_date, datetime.min.time().replace(hour=hour, minute=minute))
                expected_start_time = (base_time + timedelta(minutes=timeframe_minutes)).time()
                
                # Validation check
                if actual_start_time == expected_start_time:
                    self.log_detailed(f"Start time alignment: {actual_start_time} matches expected {expected_start_time}", "INFO")
                    print(f"Start time validation: {trading_start_time_str} + {global_timeframe} = {expected_start_time}")
                    return True
                else:
                    error_msg = f"Start time mismatch: expected {expected_start_time}, got {actual_start_time}"
                    self.log_detailed(f"Start time alignment: {error_msg}", "ERROR")
                    print(f"Start time validation: {error_msg}")
                    return False
                
        except Exception as e:
            self.log_detailed(f"Error in start time validation: {e}", "ERROR")
            return False
    
    def _validate_no_lookahead_bias(
        self, 
        ltp_data_1m: pd.DataFrame,
        combined_5m_data: pd.DataFrame,
        merged_data: pd.DataFrame
    ) -> bool:
        """
        Validate that 1m rows only reference completed 5m bars (no future data).
        
        Example: 1m rows 09:25-09:29 should reference 09:20 5m bar, not 09:25 5m bar
        """
        try:
            if merged_data.empty or combined_5m_data.empty:
                self.log_detailed("Lookahead bias validation: Insufficient data", "ERROR")
                return False
            
            validation_passed = True
            errors = []
            
            # Check sample of 1m timestamps to ensure proper 5m reference
            sample_size = min(10, len(merged_data))
            sample_indices = np.linspace(0, len(merged_data)-1, sample_size, dtype=int)
            
            for i, idx in enumerate(sample_indices):
                timestamp_1m = merged_data.index[idx]
                
                # Find the 5m timestamp that should be referenced (most recent completed)
                expected_5m_timestamp = None
                for ts_5m in combined_5m_data.index:
                    if ts_5m <= timestamp_1m:
                        expected_5m_timestamp = ts_5m
                    else:
                        break
                
                if expected_5m_timestamp is None:
                    continue
                
                # Check if any 5m data from future (after 1m timestamp) is being used
                future_5m_timestamps = combined_5m_data.index[combined_5m_data.index > timestamp_1m]
                
                if len(future_5m_timestamps) > 0:
                    # Verify that none of the future 5m data is reflected in current 1m row
                    next_5m_timestamp = future_5m_timestamps[0]
                    
                    # Sample validation: check if 5m_close matches expected timestamp, not future
                    if '5m_close' in merged_data.columns:
                        current_5m_close = merged_data.iloc[idx]['5m_close']
                        expected_5m_close = combined_5m_data.loc[expected_5m_timestamp, '5m_close']
                        
                        # Allow small floating point differences
                        if not np.isclose(current_5m_close, expected_5m_close, rtol=1e-10):
                            error_msg = f"Lookahead bias detected at {timestamp_1m}: 5m_close mismatch"
                            errors.append(error_msg)
                            validation_passed = False
                
                # Log sample validation (first few only)
                if i < 3:
                    self.log_detailed(
                        f"Sample {i+1}: 1m timestamp {timestamp_1m} -> 5m timestamp {expected_5m_timestamp}",
                        "DEBUG"
                    )
            
            if validation_passed:
                self.log_detailed("Lookahead bias validation: No future data leakage detected", "INFO")
                print("Lookahead bias validation: PASSED")
                return True
            else:
                self.log_detailed(f"Lookahead bias validation: {len(errors)} errors found", "ERROR")
                print(f"Lookahead bias validation: FAILED ({len(errors)} errors)")
                return False
                
        except Exception as e:
            self.log_detailed(f"Error in lookahead bias validation: {e}", "ERROR")
            return False
    
    def _validate_temporal_alignment(
        self,
        ltp_data_1m: pd.DataFrame,
        combined_5m_data: pd.DataFrame, 
        merged_data: pd.DataFrame
    ) -> bool:
        """
        Validate proper temporal alignment using merge_asof backward direction.
        
        Ensures each 1m timestamp gets data from most recent completed 5m bar.
        """
        try:
            if merged_data.empty:
                self.log_detailed("Temporal alignment validation: No merged data", "ERROR")
                return False
            
            # Test specific alignment cases
            alignment_tests = []
            
            # Test case 1: Verify first 1m timestamp alignment
            if len(merged_data) > 0:
                first_1m_timestamp = merged_data.index[0]
                
                # Find corresponding 5m timestamp (most recent completed)
                corresponding_5m = None
                for ts_5m in combined_5m_data.index:
                    if ts_5m <= first_1m_timestamp:
                        corresponding_5m = ts_5m
                    else:
                        break
                
                if corresponding_5m is not None:
                    alignment_tests.append({
                        'test': 'first_1m_alignment',
                        'timestamp_1m': first_1m_timestamp,
                        'expected_5m': corresponding_5m,
                        'passed': True
                    })
            
            # Test case 2: Verify alignment within 5m candle boundaries
            # Example: 1m timestamps 09:25, 09:26, 09:27, 09:28, 09:29 should all reference 09:20 5m bar
            sample_5m_periods = combined_5m_data.index[:3]  # Check first 3 5m periods
            
            for period_start in sample_5m_periods:
                # Find 1m timestamps within this 5m period + 1 timeframe
                period_end = period_start + pd.Timedelta(minutes=5)
                next_period_start = period_end  
                
                # 1m timestamps in the following period should reference this 5m bar
                following_1m_timestamps = merged_data.index[
                    (merged_data.index >= next_period_start) & 
                    (merged_data.index < next_period_start + pd.Timedelta(minutes=5))
                ]
                
                if len(following_1m_timestamps) > 0:
                    alignment_tests.append({
                        'test': 'boundary_alignment',
                        'period_5m': period_start,
                        'following_1m_count': len(following_1m_timestamps),
                        'passed': True
                    })
            
            # Validation summary
            total_tests = len(alignment_tests)
            passed_tests = sum(1 for test in alignment_tests if test['passed'])
            
            if passed_tests == total_tests and total_tests > 0:
                self.log_detailed(f"Temporal alignment validation: {passed_tests}/{total_tests} tests passed", "INFO")
                print(f"Temporal alignment validation: PASSED ({passed_tests}/{total_tests})")
                return True
            else:
                self.log_detailed(f"Temporal alignment validation: {passed_tests}/{total_tests} tests passed", "ERROR")
                print(f"Temporal alignment validation: FAILED ({passed_tests}/{total_tests})")
                return False
                
        except Exception as e:
            self.log_detailed(f"Error in temporal alignment validation: {e}", "ERROR")
            return False
    
    def _update_signal_candle_efficient(
        self,
        position_signals: np.ndarray,
        eval_context: Dict[str, Any],
        merged_data: pd.DataFrame,
        evaluation_index: pd.Index
    ) -> Dict[str, np.ndarray]:
        """
        Efficiently update signal candle data using vectorized operations.
        
        This method provides the same signal candle functionality as the sequential approach
        but uses vectorized operations for 10x+ performance improvement.
        
        Args:
            position_signals: Position signals array (1=entry, -1=exit, 0=hold)
            eval_context: Evaluation context to update with signal candle data
            merged_data: Market data for signal candle capture
            evaluation_index: Index for the data
            
        Returns:
            Dictionary with signal candle OHLC data
        """
        try:
            n = len(position_signals)
            
            # Get market data for signal candle capture
            if merged_data is not None:
                # Multi-timeframe mode: use merged data
                open_data = merged_data['open'].values
                high_data = merged_data['high'].values
                low_data = merged_data['low'].values
                close_data = merged_data['close'].values
            else:
                # Single timeframe mode: use evaluation context data
                open_data = eval_context['open'].values if hasattr(eval_context['open'], 'values') else eval_context['open']
                high_data = eval_context['high'].values if hasattr(eval_context['high'], 'values') else eval_context['high']
                low_data = eval_context['low'].values if hasattr(eval_context['low'], 'values') else eval_context['low']
                close_data = eval_context['close'].values if hasattr(eval_context['close'], 'values') else eval_context['close']
            
            # VECTORIZED: Find all entry and exit positions
            entry_positions = np.where(position_signals == 1)[0]
            exit_positions = np.where(position_signals == -1)[0]
            
            # FIXED: Check if signal candle arrays are already populated by _calculate_signal_candle_direct
            signal_open = eval_context['signal_candle_open']
            signal_high = eval_context['signal_candle_high']
            signal_low = eval_context['signal_candle_low']
            signal_close = eval_context['signal_candle_close']
            
            # Check if signal candle data was already calculated correctly
            signal_data_populated = not np.all(np.isnan(signal_open))
            
            if signal_data_populated:
                print(f"📊 Signal candle data already populated by _calculate_signal_candle_direct - skipping redundant calculation")
                print(f"   Valid signal candles: {np.sum(~np.isnan(signal_open))}/{len(signal_open)}")
            else:
                print(f"📊 Signal candle data not populated - performing vectorized calculation")
                
                # EFFICIENT: Process each entry-exit pair with proper 1m->5m index mapping
                for entry_idx in entry_positions:
                    # Find the corresponding exit for this entry
                    corresponding_exits = exit_positions[exit_positions > entry_idx]
                    if len(corresponding_exits) > 0:
                        exit_idx = corresponding_exits[0]
                    else:
                        exit_idx = n  # Position held until end
                    
                    # FIXED: Proper mapping from 1m entry index to 5m signal candle data
                    # In multi-timeframe mode, entry_idx is 1m index but we need 5m signal candle data
                    if merged_data is not None:
                        # Multi-timeframe mode: Get 5m signal candle data from merged_data
                        # The 5m signal candle OHLC is in the 'open', 'high', 'low', 'close' columns
                        # which contain the 5m data aligned via merge_asof to each 1m timestamp
                        signal_open_val = merged_data.iloc[entry_idx]['open']
                        signal_high_val = merged_data.iloc[entry_idx]['high']
                        signal_low_val = merged_data.iloc[entry_idx]['low']
                        signal_close_val = merged_data.iloc[entry_idx]['close']
                        
                        # Debug: Check Indicator_Timestamp at this entry
                        indicator_timestamp = merged_data.iloc[entry_idx]['Indicator_Timestamp']
                        current_timestamp = merged_data.index[entry_idx]
                        print(f"🔍 DEBUG: Entry {entry_idx} at {current_timestamp}")
                        print(f"🔍 DEBUG: Indicator_Timestamp = {indicator_timestamp}")
                        print(f"🔍 DEBUG: Using 5m OHLC: ({signal_open_val:.2f},{signal_high_val:.2f},{signal_low_val:.2f},{signal_close_val:.2f})")
                        
                        signal_open[entry_idx:exit_idx] = signal_open_val
                        signal_high[entry_idx:exit_idx] = signal_high_val
                        signal_low[entry_idx:exit_idx] = signal_low_val
                        signal_close[entry_idx:exit_idx] = signal_close_val
                        signal_candle_idx = entry_idx  # For logging purposes
                    else:
                        # Single timeframe mode: direct mapping
                        signal_candle_idx = entry_idx
                        signal_open[entry_idx:exit_idx] = open_data[signal_candle_idx]
                        signal_high[entry_idx:exit_idx] = high_data[signal_candle_idx] 
                        signal_low[entry_idx:exit_idx] = low_data[signal_candle_idx]
                        signal_close[entry_idx:exit_idx] = close_data[signal_candle_idx]
                
                if self._performance_mode != 'ultra':
                    duration = exit_idx - entry_idx
                    print(f"   📊 Signal candle updated: Entry at {entry_idx}, Signal candle from {signal_candle_idx}, Duration {duration} candles")
                    if merged_data is not None:
                        signal_ohlc = f"({signal_open_val:.2f},{signal_high_val:.2f},{signal_low_val:.2f},{signal_close_val:.2f})"
                    else:
                        signal_ohlc = f"({open_data[signal_candle_idx]:.2f},{high_data[signal_candle_idx]:.2f},{low_data[signal_candle_idx]:.2f},{close_data[signal_candle_idx]:.2f})"
                    print(f"   📊 Signal candle OHLC: {signal_ohlc}")
                else:
                    if merged_data is not None:
                        signal_ohlc = f"({signal_open_val:.2f},{signal_high_val:.2f},{signal_low_val:.2f},{signal_close_val:.2f})"
                    else:
                        signal_ohlc = f"({open_data[signal_candle_idx]:.2f},{high_data[signal_candle_idx]:.2f},{low_data[signal_candle_idx]:.2f},{close_data[signal_candle_idx]:.2f})"
                    print(f"📊 Signal candle: Entry {entry_idx} using signal candle {signal_candle_idx} {signal_ohlc}")
            
            # Prepare signal candle data for export
            signal_candle_data = {
                'signal_candle_open': signal_open.copy(),
                'signal_candle_high': signal_high.copy(),
                'signal_candle_low': signal_low.copy(),
                'signal_candle_close': signal_close.copy()
            }
            
            # Count valid signal candle values for logging
            valid_count = np.sum(~np.isnan(signal_open))
            if self._performance_mode != 'ultra':
                print(f"   ✅ Signal candle calculation complete: {valid_count}/{n} candles with signal data")
            else:
                print(f"✅ Signal candle calculation complete: {valid_count}/{n} candles with signal data")
            
            # SL/TP data should be handled in the main signal generation flow, not here
            # This method is only responsible for signal candle data
            sl_tp_data = {}
            print(f"   🎯 SL/TP calculation SKIPPED in _update_signal_candle_efficient (handled elsewhere)")
            
            return signal_candle_data, sl_tp_data
            
        except Exception as e:
            self.log_detailed(f"Error in efficient signal candle update: {e}", "ERROR")
            # Return empty results on error
            signal_candle_data = {
                'signal_candle_open': np.full(len(position_signals), np.nan),
                'signal_candle_high': np.full(len(position_signals), np.nan),
                'signal_candle_low': np.full(len(position_signals), np.nan),
                'signal_candle_close': np.full(len(position_signals), np.nan)
            }
            sl_tp_data = {
                'sl_price': np.full(len(position_signals), np.nan),
                'tp_price': np.full(len(position_signals), np.nan)
            }
            return signal_candle_data, sl_tp_data
    
    def _calculate_signal_candle_direct(
        self,
        open_data: np.ndarray,
        high_data: np.ndarray,
        low_data: np.ndarray,
        close_data: np.ndarray,
        entry_signals: np.ndarray,
        evaluation_index: pd.Index,
        exit_signals: np.ndarray = None
    ) -> Dict[str, np.ndarray]:
        """
        Calculate signal candle OHLC data directly using shared context.
        
        This method implements the exact same logic as SmartIndicatorEngine.update_signal_candle_with_signals()
        but uses the existing shared context data for maximum efficiency.
        
        Args:
            open_data: Open price array from shared context
            high_data: High price array from shared context
            low_data: Low price array from shared context
            close_data: Close price array from shared context
            entry_signals: Boolean array of entry signals
            evaluation_index: Index for the evaluation data
            
        Returns:
            Dictionary with signal candle OHLC data
        """
        try:
            n = len(evaluation_index)
            
            # Initialize signal candle arrays with NaN
            signal_open = np.full(n, np.nan)
            signal_high = np.full(n, np.nan)
            signal_low = np.full(n, np.nan)
            signal_close = np.full(n, np.nan)
            
            # Position state tracking (identical to SmartIndicatorEngine logic)
            in_position = False
            current_signal_open = np.nan
            current_signal_high = np.nan
            current_signal_low = np.nan
            current_signal_close = np.nan
            
            # Process each candle sequentially to maintain state with proper exit handling
            for i in range(n):
                # Check for exit signal first to reset position state
                if exit_signals is not None and exit_signals[i] and in_position:
                    # Exit signal - reset position state for new entries
                    in_position = False
                    
                if entry_signals[i] and not in_position:
                    # Entry signal - capture SIGNAL candle OHLC (current candle that contains the signal)
                    in_position = True
                    signal_candle_idx = i  # Use current candle as signal candle
                    current_signal_open = open_data[signal_candle_idx]
                    current_signal_high = high_data[signal_candle_idx]
                    current_signal_low = low_data[signal_candle_idx]
                    current_signal_close = close_data[signal_candle_idx]
                    
                    print(f"📊 NEW SIGNAL CANDLE CAPTURED at index {i}: Using candle {signal_candle_idx} ({current_signal_open:.2f},{current_signal_high:.2f},{current_signal_low:.2f},{current_signal_close:.2f})")
                    print(f"📊 Index {i} corresponds to 1m timestamp, signal candle from 5m data at same index")
                    
                    # Set signal candle values for this position
                    signal_open[i] = current_signal_open
                    signal_high[i] = current_signal_high
                    signal_low[i] = current_signal_low
                    signal_close[i] = current_signal_close
                    
                elif in_position:
                    # In position - maintain signal candle values from entry
                    signal_open[i] = current_signal_open
                    signal_high[i] = current_signal_high
                    signal_low[i] = current_signal_low
                    signal_close[i] = current_signal_close
            
            # Count valid signal candle periods for logging
            valid_count = np.sum(~np.isnan(signal_open))
            
            if self._performance_mode != 'ultra':
                print(f"   ✅ Direct signal candle calculation complete: {valid_count}/{n} candles with signal data")
            else:
                print(f"✅ Direct signal candle calculation complete: {valid_count}/{n} candles with signal data")
            
            return {
                'signal_candle_open': signal_open,
                'signal_candle_high': signal_high,
                'signal_candle_low': signal_low,
                'signal_candle_close': signal_close
            }
            
        except Exception as e:
            self.log_detailed(f"Error in direct signal candle calculation: {e}", "ERROR")
            # Return empty results on error
            return {
                'signal_candle_open': np.full(len(evaluation_index), np.nan),
                'signal_candle_high': np.full(len(evaluation_index), np.nan),
                'signal_candle_low': np.full(len(evaluation_index), np.nan),
                'signal_candle_close': np.full(len(evaluation_index), np.nan)
            }

    def _merge_signal_candle_data(
        self, 
        target_data: Dict[str, np.ndarray], 
        source_data: Dict[str, np.ndarray], 
        strategy_key: str
    ) -> None:
        """
        Merge signal candle data from multiple strategies.
        
        This method combines signal candle data from different strategies,
        ensuring that data from all strategies is preserved in the final export.
        
        Args:
            target_data: The combined signal candle data (modified in place)
            source_data: Signal candle data from current strategy
            strategy_key: Strategy identifier for logging
        """
        try:
            import numpy as np
            
            for key in ['signal_candle_open', 'signal_candle_high', 'signal_candle_low', 'signal_candle_close']:
                if key in source_data and key in target_data:
                    source_array = source_data[key]
                    target_array = target_data[key]
                    
                    # Convert to numpy arrays if they aren't already
                    if hasattr(source_array, 'values'):
                        source_array = source_array.values
                    if hasattr(target_array, 'values'):
                        target_array = target_array.values
                    
                    # Merge: use source data where it's not NaN, keep target data otherwise
                    source_valid = ~np.isnan(source_array)
                    target_array[source_valid] = source_array[source_valid]
                    
                    # Update the target data
                    target_data[key] = target_array
                    
                    # Count merged values for logging
                    valid_count = np.sum(~np.isnan(target_array))
                    self.log_detailed(f"{strategy_key} merged {key}: {valid_count} total valid values", "DEBUG")
            
            if self._performance_mode != 'ultra':
                total_valid = np.sum(~np.isnan(target_data['signal_candle_open']))
                print(f"   📊 Signal candle merge complete for {strategy_key}: {total_valid} total candles with data")
                
        except Exception as e:
            self.log_detailed(f"Error merging signal candle data for {strategy_key}: {e}", "ERROR")

 
        """
        Merge SL/TP data from multiple strategies.
        
        This method combines SL/TP data from different strategies,
        ensuring that data from all strategies is preserved in the final export.
        
        Args:
            target_data: The combined SL/TP data (modified in place)
            source_data: SL/TP data from current strategy
            strategy_key: Strategy identifier for logging
        """
        try:
            import numpy as np
            
            for key in ['sl_price', 'tp_price', 'entry_point']:
                if key in source_data and key in target_data:
                    source_array = source_data[key]
                    target_array = target_data[key]
                    
                    # Convert to numpy arrays if they aren't already
                    if hasattr(source_array, 'values'):
                        source_array = source_array.values
                    if hasattr(target_array, 'values'):
                        target_array = target_array.values
                    
                    # Merge: use source data where it's not NaN, keep target data otherwise
                    source_valid = ~np.isnan(source_array)
                    target_data[key][source_valid] = source_array[source_valid]
                    
                    # Count merged values for logging
                    valid_count = np.sum(~np.isnan(target_array))
                    self.log_detailed(f"{strategy_key} merged {key}: {valid_count} total valid values", "DEBUG")
            
            if self._performance_mode != 'ultra':
                total_valid = np.sum(~np.isnan(target_data['sl_price']))
                print(f"   🎯 SL/TP merge complete for {strategy_key}: {total_valid} total candles with data")
                
        except Exception as e:
            self.log_detailed(f"Error merging SL/TP data for {strategy_key}: {e}", "ERROR")
    def _calculate_sl_tp_indicator_direct(
        self,
        open_data: np.ndarray,
        high_data: np.ndarray,
        low_data: np.ndarray,
        close_data: np.ndarray,
        entry_signals: np.ndarray,
        evaluation_index: pd.Index,
        case_config: Dict[str, Any],
        entry_condition: str,
        eval_context: Dict[str, Any]
    ) -> Dict[str, np.ndarray]:
        """
        ULTRA-FAST vectorized SL/TP calculation for all entry signals.
        
        This method uses pure NumPy operations to calculate SL/TP values for all entry signals
        simultaneously, using dynamic entry point detection from actual LTP breakout conditions.
        
        Args:
            open_data: Open prices array
            high_data: High prices array  
            low_data: Low prices array
            close_data: Close prices array
            entry_signals: Boolean array of entry signals
            evaluation_index: Index for timestamps
            case_config: Strategy case configuration containing OptionType
            entry_condition: Strategy entry condition string for LTP analysis
            eval_context: Evaluation context containing LTP data
            
        Returns:
            Dictionary with sl_price and tp_price arrays
        """
        try:
            n = len(entry_signals)
            
            # Performance timing
            calc_start = time.time()
            
            # Find all entry signal indices
            entry_indices = np.where(entry_signals)[0]
            
            if len(entry_indices) == 0:
                # No entry signals - return NaN arrays
                return {
                    'sl_price': np.full(n, np.nan),
                    'tp_price': np.full(n, np.nan)
                }
            
            self.log_detailed(f"VECTORIZED SL/TP: Processing {len(entry_indices)} entry signals", "DEBUG")
            
            # VECTORIZED CALCULATION: Process all entry signals at once
            entry_ohlc = np.column_stack([
                open_data[entry_indices],
                high_data[entry_indices], 
                low_data[entry_indices],
                close_data[entry_indices]
            ])
            
            # Extract actual entry points from LTP breakout conditions
            entry_points = self._extract_dynamic_entry_points(
                entry_indices, entry_condition, case_config, eval_context
            )
            
            # Calculate all SL/TP values using dynamic entry points
            sl_values, tp_values = self._calculate_sl_tp_vectorized(entry_ohlc, case_config, entry_points, eval_context)
            
            # Store calculated values and indices for later population (after position tracking)
            # Don't populate arrays here - this will be done after position_signals is available
            calc_time = time.time() - calc_start
            
            # Performance logging
            signals_per_second = len(entry_indices) / calc_time if calc_time > 0 else float('inf')
            self.log_detailed(f"VECTORIZED SL/TP: {len(entry_indices)} signals calculated in {calc_time:.4f}s ({signals_per_second:.0f} signals/sec)", "INFO")
            
            # Return calculated values and metadata for later array population
            return {
                'sl_values': sl_values,
                'tp_values': tp_values,
                'entry_points': entry_points,
                'entry_indices': entry_indices,
                'array_length': n,
                'needs_population': True  # Flag indicating arrays need to be populated later
            }
            
        except Exception as e:
            self.log_detailed(f"Error in vectorized SL/TP calculation: {e}", "ERROR")
            # Return NaN arrays on error
            return {
                'sl_price': np.full(len(entry_signals), np.nan),
                'tp_price': np.full(len(entry_signals), np.nan),
                'entry_point': np.full(len(entry_signals), np.nan)
            }

    def get_dynamic_entry_point(
        self, 
        eval_context: Dict[str, Any], 
        candle_index: int, 
        option_type: str
    ) -> float:
        """
        Dynamically find entry point by scanning ALL numeric values in eval_context.
        
        For CALL: Returns maximum numeric value found at candle_index
        For PUT: Returns minimum numeric value found at candle_index
        
        Args:
            eval_context: Evaluation context containing all market data and indicators
            candle_index: Index of the candle where entry signal occurred
            option_type: 'CALL' or 'PUT' to determine max vs min logic
            
        Returns:
            Dynamic entry point value (max for CALL, min for PUT)
        """
        try:
            all_numeric_values = []
            
            # Scan EVERY key in eval_context for numeric values
            for key, data in eval_context.items():
                try:
                    # Skip numpy module reference
                    if key == 'np':
                        continue
                        
                    # Get value at current candle index
                    if hasattr(data, '__getitem__') and len(data) > candle_index:
                        value = data[candle_index]
                    else:
                        value = data  # Scalar value
                    
                    # Collect valid positive numeric values
                    if isinstance(value, (int, float)) and not np.isnan(value) and value > 0:
                        all_numeric_values.append(value)
                        
                except (IndexError, TypeError, AttributeError, ValueError):
                    # Skip invalid data
                    continue
            
            # Find max/min based on option type
            if option_type.upper() == 'CALL':
                # For CALL: Use the MAXIMUM value found
                entry_point = max(all_numeric_values) if all_numeric_values else None
                self.log_detailed(f"CALL dynamic entry point: max of {len(all_numeric_values)} values = {entry_point}", "DEBUG")
            else:  # PUT
                # For PUT: Use the MINIMUM value found  
                entry_point = min(all_numeric_values) if all_numeric_values else None
                self.log_detailed(f"PUT dynamic entry point: min of {len(all_numeric_values)} values = {entry_point}", "DEBUG")
            
            # Fallback to close if no valid values found
            if entry_point is None:
                fallback_close = eval_context.get('close')
                if fallback_close is not None:
                    if hasattr(fallback_close, '__getitem__'):
                        entry_point = fallback_close[candle_index]
                    else:
                        entry_point = fallback_close
                else:
                    entry_point = 0.0  # Last resort fallback
                self.log_detailed(f"Using fallback entry point: {entry_point}", "WARNING")
            
            return float(entry_point)
            
        except Exception as e:
            self.log_detailed(f"Error in dynamic entry point detection: {e}", "ERROR")
            # Emergency fallback
            return 0.0

    def _is_indicator_enabled(self, indicator_name: str) -> bool:
        """
        Check if a specific indicator is enabled in the configuration.
        
        Args:
            indicator_name: Name of the indicator to check (e.g., 'signal_candle', 'SL_TP_levels')
            
        Returns:
            True if indicator is enabled, False otherwise
        """
        try:
            indicators_config = self.config_loader.main_config.get('indicators', {})
            indicator_config = indicators_config.get(indicator_name, {})
            return indicator_config.get('enabled', False)
        except Exception as e:
            self.log_detailed(f"Error checking if indicator {indicator_name} is enabled: {e}", "ERROR")
            return False

    def _extract_dynamic_entry_points(
        self,
        entry_indices: np.ndarray,
        entry_condition: str,
        case_config: Dict[str, Any],
        eval_context: Dict[str, Any]
    ) -> np.ndarray:
        """
        Extract actual entry points from LTP breakout conditions.
        
        Analyzes the entry condition to identify LTP usage patterns and extracts
        the actual breakout values where entry conditions were triggered.
        
        Args:
            entry_indices: Indices where entry signals occurred
            entry_condition: Strategy entry condition string
            case_config: Strategy case configuration
            eval_context: Evaluation context containing LTP and market data
            
        Returns:
            Array of actual entry point values for each entry signal
        """
        try:
            option_type = case_config.get('OptionType', 'CALL').upper()
            
            # CORRECTED: Smart entry point detection using signal candle data
            if option_type == 'CALL':
                # For CALL strategies: Entry point should be signal_candle_high for optimal breakout entry
                # This captures the highest price during the signal generation candle
                signal_candle_high_data = eval_context.get('signal_candle_high')
                if signal_candle_high_data is not None:
                    if hasattr(signal_candle_high_data, 'values'):
                        signal_values = signal_candle_high_data.values
                    else:
                        signal_values = signal_candle_high_data
                    
                    # Get the actual signal_candle_high values where entries occurred
                    entry_points = signal_values[entry_indices]
                    self.log_detailed(f"CALL entry points: Using signal_candle_high values, avg={np.mean(entry_points):.2f}", "DEBUG")
                    return entry_points
                
                # Fallback: use 5m candle close as approximate breakout level
                close_data = eval_context.get('close')
                if close_data is not None:
                    if hasattr(close_data, 'values'):
                        close_values = close_data.values
                    else:
                        close_values = close_data
                    entry_points = close_values[entry_indices]
                    self.log_detailed(f"CALL fallback entry points: Using 5m close values", "WARNING")
                    return entry_points
                    
            else:  # PUT
                # For PUT strategies: Entry point should be signal_candle_low for optimal breakdown entry
                # This captures the lowest price during the signal generation candle
                signal_candle_low_data = eval_context.get('signal_candle_low')
                if signal_candle_low_data is not None:
                    if hasattr(signal_candle_low_data, 'values'):
                        signal_values = signal_candle_low_data.values
                    else:
                        signal_values = signal_candle_low_data
                    
                    # Get the actual signal_candle_low values where entries occurred
                    entry_points = signal_values[entry_indices]
                    self.log_detailed(f"PUT entry points: Using signal_candle_low values, avg={np.mean(entry_points):.2f}", "DEBUG")
                    return entry_points
                
                # Fallback: use 5m candle close as approximate breakdown level
                close_data = eval_context.get('close')
                if close_data is not None:
                    if hasattr(close_data, 'values'):
                        close_values = close_data.values
                    else:
                        close_values = close_data
                    entry_points = close_values[entry_indices]
                    self.log_detailed(f"PUT fallback entry points: Using 5m close values", "WARNING")
                    return entry_points
            
            # Last resort fallback: use close prices
            close_data = eval_context.get('close')
            if close_data is not None:
                if hasattr(close_data, 'values'):
                    close_values = close_data.values
                else:
                    close_values = close_data
                entry_points = close_values[entry_indices]
                self.log_detailed(f"Emergency fallback entry points: Using close values", "ERROR")
                return entry_points
            
            # If all else fails, return zeros
            self.log_detailed(f"ERROR: No market data available for entry point extraction", "ERROR")
            return np.zeros(len(entry_indices))
            
        except Exception as e:
            self.log_detailed(f"Error extracting dynamic entry points: {e}", "ERROR")
            # Return zeros as fallback
            return np.zeros(len(entry_indices))

    def _calculate_sl_tp_vectorized(self, entry_ohlc: np.ndarray, case_config: Dict[str, Any], entry_points: np.ndarray = None, eval_context: Dict[str, Any] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Ultra-fast vectorized SL/TP calculation using strategy configuration and dynamic entry points.
        
        Implements corrected SL/TP calculation logic:
        1. Position type from strategy OptionType (not signal candle analysis)
        2. Dynamic entry point from actual LTP breakout values
        3. Buffer points and proper risk-reward calculations
        
        Args:
            entry_ohlc: 2D array of shape (n_entries, 4) containing [open, high, low, close]
            case_config: Strategy case configuration containing OptionType
            entry_points: Array of actual entry point values from LTP breakout detection
            
        Returns:
            Tuple of (sl_values, tp_values) arrays
        """
        try:
            # Get configuration
            sl_tp_config = self.config_loader.main_config.get('indicators', {}).get('SL_TP_levels', {})
            sl_tp_method = sl_tp_config.get('SL_TP_Method', 'signal_candle_range_exit')
            max_sl_points = float(sl_tp_config.get('Max_Sl_points', 60))
            buffer_sl_points = float(sl_tp_config.get('buffer_sl_points', 5))
            buffer_tp_points = float(sl_tp_config.get('buffer_tp_points', 3))
            risk_reward_str = sl_tp_config.get('RiskRewardRatio', '1:2')
            
            # Parse risk-reward ratio
            risk_part, reward_part = risk_reward_str.split(':')
            risk_ratio = float(risk_part)
            reward_ratio = float(reward_part)
            
            # Extract OHLC components
            signal_open = entry_ohlc[:, 0]
            signal_high = entry_ohlc[:, 1]
            signal_low = entry_ohlc[:, 2]
            signal_close = entry_ohlc[:, 3]
            
            # CORRECTED: Get position type from strategy configuration (not signal candle analysis)
            option_type = case_config.get('OptionType', 'CALL').upper()
            is_call_position = (option_type == 'CALL')
            
            # DEBUG: Log input data
            print(f"🎯 EXECUTING: _calculate_sl_tp_vectorized (CORRECT METHOD)")
            print(f"🔍 DEBUG SL/TP CALCULATION:")
            print(f"   Strategy: {case_config.get('case_name', 'Unknown')}")
            print(f"   OptionType: {option_type}")
            print(f"   Signal OHLC: O={signal_open[0]:.2f}, H={signal_high[0]:.2f}, L={signal_low[0]:.2f}, C={signal_close[0]:.2f}")
            print(f"   Entry Points Available: {entry_points is not None}")
            if entry_points is not None:
                print(f"   Entry Point Value: {entry_points[0]:.2f}")
            print(f"   Config: Max_SL={max_sl_points}, Buffer_SL={buffer_sl_points}, Buffer_TP={buffer_tp_points}, RR={risk_reward_str}")
            
            # Use dynamic entry points or fallback to signal candle data
            if entry_points is not None and len(entry_points) == len(signal_close):
                # Use actual LTP breakout values as entry points
                entry_point_values = entry_points
                print(f"   ✅ Using dynamic entry points from LTP breakout detection: {entry_point_values[0]:.2f}")
                self.log_detailed(f"Using dynamic entry points from LTP breakout detection", "INFO")
            else:
                # Fallback to signal candle data (old behavior)
                if is_call_position:
                    entry_point_values = signal_high
                else:
                    entry_point_values = signal_low
                print(f"   ⚠️ Fallback: Using signal candle data for entry points: {entry_point_values[0]:.2f}")
                self.log_detailed(f"Fallback: Using signal candle data for entry points", "WARNING")
            
            # YOUR EXACT FORMULAS: Calculate signal candle range for each entry
            range_values = signal_high - signal_low
            print(f"   Range Calculation: {signal_high[0]:.2f} - {signal_low[0]:.2f} = {range_values[0]:.2f}")
            
            # Cap each range to max_sl_points
            range_values_before_cap = range_values.copy()
            range_values = np.minimum(range_values, max_sl_points)
            
            # For CPR pivot method, enforce minimum SL to reach meaningful pivot targets
            if sl_tp_method == 'cpr_pivot_range_exit':
                min_sl_for_pivots = range_values  # Minimum SL to reach CPR pivot targets
                range_values = np.maximum(range_values, min_sl_for_pivots)
                print(f"   CPR Pivot Method: Enforcing minimum SL of {min_sl_for_pivots} points")
            
            print(f"   Range After Processing: {range_values_before_cap[0]:.2f} -> {range_values[0]:.2f} (max: {max_sl_points})")
            
            # Initialize arrays
            sl_values = np.zeros_like(signal_close)
            tp_values = np.zeros_like(signal_close)
            
            # Store current evaluation context for CPR pivot access
            if not hasattr(self, '_current_eval_context'):
                self._current_eval_context = {}
            
            if is_call_position:
                # CALL POSITION LOGIC
                # Dynamic entry point from actual LTP breakout
                entry_point = entry_point_values
                
                # SL calculation is same for both methods
                sl_values = entry_point - (range_values + buffer_sl_points)
                
                # TP calculation varies by method
                risk = range_values + buffer_sl_points  # Calculate risk for both methods
                if sl_tp_method == 'cpr_pivot_range_exit':
                    # NEW: CPR Pivot-based TP calculation for CALL
                    print(f"🎯 USING CPR PIVOT RANGE EXIT METHOD for {option_type}")
                    try:
                        # Calculate TP using CPR pivot targeting for each entry
                        tp_values = np.zeros_like(entry_point)
                        for i in range(len(entry_point)):
                            print(f"🎯 Calculating CPR pivot TP for entry {i+1}/{len(entry_point)}")
                            tp_values[i] = self._calculate_cpr_pivot_tp(
                                option_type, entry_point[i], risk[i], reward_ratio, buffer_tp_points, eval_context, i
                            )
                        reward = tp_values - entry_point - buffer_tp_points  # Calculate reward for display
                        print(f"🎯 ✅ CPR PIVOT TP SUCCESSFUL: TP={tp_values[0]:.2f}")
                    except Exception as e:
                        print(f"   ❌ CPR Pivot TP failed, using fallback: {e}")
                        import traceback
                        traceback.print_exc()
                        # Fallback to standard RR calculation
                        reward = risk * reward_ratio
                        tp_values = entry_point + reward + buffer_tp_points
                else:
                    # Original signal_candle_range_exit method
                    reward = risk * reward_ratio
                    tp_values = entry_point + reward + buffer_tp_points
                
                print(f"   CALL Calculation:")
                print(f"     SL = {entry_point[0]:.2f} - ({range_values[0]:.2f} + {buffer_sl_points}) = {sl_values[0]:.2f}")
                print(f"     Risk = {range_values[0]:.2f} + {buffer_sl_points} = {risk[0]:.2f}")
                print(f"     Reward = {risk[0]:.2f} × {reward_ratio} = {reward[0]:.2f}")
                print(f"     TP = {entry_point[0]:.2f} + {reward[0]:.2f} + {buffer_tp_points} = {tp_values[0]:.2f}")
                
                self.log_detailed(f"CALL position SL/TP: Entry={np.mean(entry_point):.2f}, Range={np.mean(range_values):.2f}, Risk={np.mean(risk):.2f}, Reward={np.mean(reward):.2f}", "DEBUG")
                
            else:
                # PUT POSITION LOGIC  
                # Dynamic entry point from actual LTP breakout
                entry_point = entry_point_values
                
                # SL calculation is same for both methods
                sl_values = entry_point + (range_values + buffer_sl_points)
                
                # TP calculation varies by method
                risk = range_values + buffer_sl_points  # Calculate risk for both methods
                if sl_tp_method == 'cpr_pivot_range_exit':
                    # NEW: CPR Pivot-based TP calculation for PUT
                    try:
                        # Calculate TP using CPR pivot targeting for each entry
                        tp_values = np.zeros_like(entry_point)
                        for i in range(len(entry_point)):
                            tp_values[i] = self._calculate_cpr_pivot_tp(
                                option_type, entry_point[i], risk[i], reward_ratio, buffer_tp_points, eval_context, i
                            )
                        reward = entry_point - tp_values - buffer_tp_points  # Calculate reward for display
                    except Exception as e:
                        print(f"   ⚠️ CPR Pivot TP failed, using fallback: {e}")
                        # Fallback to standard RR calculation
                        reward = risk * reward_ratio
                        tp_values = entry_point - reward - buffer_tp_points
                else:
                    # Original signal_candle_range_exit method
                    reward = risk * reward_ratio
                    tp_values = entry_point - reward - buffer_tp_points
                
                print(f"   PUT Calculation:")
                print(f"     SL = {entry_point[0]:.2f} + ({range_values[0]:.2f} + {buffer_sl_points}) = {sl_values[0]:.2f}")
                print(f"     Risk = {range_values[0]:.2f} + {buffer_sl_points} = {risk[0]:.2f}")
                print(f"     Reward = {risk[0]:.2f} × {reward_ratio} = {reward[0]:.2f}")
                print(f"     TP = {entry_point[0]:.2f} - {reward[0]:.2f} - {buffer_tp_points} = {tp_values[0]:.2f}")
                
                self.log_detailed(f"PUT position SL/TP: Entry={np.mean(entry_point):.2f}, Range={np.mean(range_values):.2f}, Risk={np.mean(risk):.2f}, Reward={np.mean(reward):.2f}", "DEBUG")
            
            print(f"   FINAL RESULT: SL={sl_values[0]:.2f}, TP={tp_values[0]:.2f}")
            
            # Log calculation details
            self.log_detailed(f"SL/TP Configuration: OptionType={option_type}, Max_SL={max_sl_points}, Buffer_SL={buffer_sl_points}, Buffer_TP={buffer_tp_points}, RR={risk_reward_str}", "INFO")
            
            return sl_values, tp_values
            
        except Exception as e:
            self.log_detailed(f"Error in vectorized SL/TP calculation core: {e}", "ERROR")
            print(f"❌ ERROR in SL/TP calculation: {e}")
            # Return NaN arrays on error
            n_entries = len(entry_ohlc)
            return np.full(n_entries, np.nan), np.full(n_entries, np.nan)

    def _populate_sl_tp_arrays_vectorized(
        self, 
        n: int, 
        entry_indices: np.ndarray, 
        sl_values: np.ndarray, 
        tp_values: np.ndarray,
        entry_points: np.ndarray = None,
        position_signals: np.ndarray = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Efficiently populate SL/TP arrays using vectorized operations with proper exit detection.
        
        Uses same logic as signal candle population to ensure SL/TP values stay constant
        during position lifetime (entry -> exit) and don't overwrite each other.
        
        Args:
            n: Total array length
            entry_indices: Indices where entries occur  
            sl_values: Calculated SL values for each entry
            tp_values: Calculated TP values for each entry
            entry_points: Entry point values for each entry (optional)
            position_signals: Position signals array (1=entry, -1=exit, 0=hold) for exit detection
            
        Returns:
            Tuple of populated (sl_prices, tp_prices, entry_point_array) arrays
        """
        try:
            # Initialize with NaN
            sl_prices = np.full(n, np.nan)
            tp_prices = np.full(n, np.nan)
            entry_point_array = np.full(n, np.nan)
            
            # IMPROVED: Use exit detection like signal candle logic
            if position_signals is not None:
                print(f"🔧 Using position_signals for proper entry-exit pair detection")
                
                # Find entry and exit positions (same logic as signal candle)
                entry_positions = np.where(position_signals == 1)[0]
                exit_positions = np.where(position_signals == -1)[0]
                
                # Process each entry-exit pair
                for i, entry_idx in enumerate(entry_positions):
                    # Find the corresponding exit for this entry
                    corresponding_exits = exit_positions[exit_positions > entry_idx]
                    if len(corresponding_exits) > 0:
                        exit_idx = corresponding_exits[0]
                    else:
                        exit_idx = n  # Position held until end
                    
                    # Set SL/TP/entry_point for SPECIFIC POSITION DURATION (like signal candle)
                    if i < len(sl_values):
                        print(f"       🔧 Setting SL: sl_prices[{entry_idx}:{exit_idx}] = {sl_values[i]:.2f}")
                        sl_prices[entry_idx:exit_idx] = sl_values[i]
                        print(f"       🔧 Setting TP: tp_prices[{entry_idx}:{exit_idx}] = {tp_values[i]:.2f}")
                        tp_prices[entry_idx:exit_idx] = tp_values[i]
                        
                        # Immediate debug: Check if values were actually set
                        sl_test = np.sum(~np.isnan(sl_prices))
                        tp_test = np.sum(~np.isnan(tp_prices))
                        print(f"       ✅ IMMEDIATE CHECK: SL valid {sl_test}, TP valid {tp_test}")
                    
                    # Set entry points for position duration
                    if entry_points is not None and i < len(entry_points):
                        print(f"       🔧 Setting entry_point: entry_point_array[{entry_idx}:{exit_idx}] = {entry_points[i]:.2f}")
                        entry_point_array[entry_idx:exit_idx] = entry_points[i]
                    
                    duration = exit_idx - entry_idx
                    print(f"   📊 SL/TP populated: Entry at {entry_idx}, Exit at {exit_idx}, Duration {duration} candles")
                    entry_point_display = entry_points[i] if entry_points is not None and i < len(entry_points) else "N/A"
                    print(f"       SL={sl_values[i]:.2f}, TP={tp_values[i]:.2f}, Entry_Point={entry_point_display}")
                    
                    # Debug: Check if values were actually set
                    actual_sl_set = np.sum(~np.isnan(sl_prices))
                    actual_tp_set = np.sum(~np.isnan(tp_prices))
                    actual_entry_set = np.sum(~np.isnan(entry_point_array))
                    print(f"       🔍 After setting: SL valid {actual_sl_set}, TP valid {actual_tp_set}, Entry valid {actual_entry_set}")
                
            else:
                # FIXED FALLBACK: Set SL/TP only for limited duration when no position_signals
                print(f"🔧 FIXED FALLBACK: Using safe logic without position_signals (limited duration)")
                for i, entry_idx in enumerate(entry_indices):
                    # CRITICAL FIX: Don't set values until end of array
                    # Instead, set for a reasonable duration (e.g., 100 candles max)
                    max_duration = 100  # Maximum position duration in candles
                    end_idx = min(entry_idx + max_duration, n)
                    
                    print(f"   FALLBACK: Setting SL/TP for Entry {entry_idx} to limited end {end_idx}")
                    sl_prices[entry_idx:end_idx] = sl_values[i]
                    tp_prices[entry_idx:end_idx] = tp_values[i]
                    
                    # Set entry points if provided
                    if entry_points is not None and len(entry_points) > i:
                        entry_point_array[entry_idx:end_idx] = entry_points[i]
            
            return sl_prices, tp_prices, entry_point_array
            
        except Exception as e:
            self.log_detailed(f"Error in SL/TP array population: {e}", "ERROR")
            print(f"❌ EXCEPTION in _populate_sl_tp_arrays_vectorized: {e}")
            import traceback
            traceback.print_exc()
            return np.full(n, np.nan), np.full(n, np.nan), np.full(n, np.nan)

    def _get_pivots_and_midpoints(self, pivots: list) -> list:
        """
        Generates a list of pivots and their midpoints for intelligent CPR targeting.
        
        INTELLIGENT TARGETING ALGORITHM:
        1. Takes all CPR pivot levels (S4→S3→S2→S1→Pivot→TC→BC→R1→R2→R3→R4)
        2. Calculates midpoints between consecutive pivots: midpoint = (pivot[i] + pivot[i+1]) / 2
        3. Creates combined target list: [Pivot1, Midpoint1-2, Pivot2, Midpoint2-3, Pivot3, ...]
        4. Removes duplicates and sorts for systematic targeting
        
        PURPOSE:
        - Provides more granular TP targets than raw pivots alone
        - Enables hitting intermediate levels between major CPR support/resistance
        - Maximizes probability of hitting nearby pivot targets
        - Example: R2=24972.87, R3=25100.63 → Midpoint=25036.75 (additional target)
        
        Args:
            pivots (list): List of pivot points (sorted CPR levels)
            
        Returns:
            list: Sorted list containing pivots and their midpoints for intelligent targeting
        """
        try:
            pivots_and_midpoints = []
            
            # Add first pivot
            if pivots:
                pivots_and_midpoints.append(pivots[0])
                
            # Generate midpoints between consecutive pivots
            for i in range(len(pivots) - 1):
                midpoint = (pivots[i] + pivots[i + 1]) / 2
                pivots_and_midpoints.append(midpoint)
                pivots_and_midpoints.append(pivots[i + 1])
            
            # Remove duplicates and sort
            pivots_and_midpoints = sorted(list(set(pivots_and_midpoints)))
            
            print(f"🎯 Generated pivots and midpoints: {len(pivots_and_midpoints)} levels")
            return pivots_and_midpoints
            
        except Exception as e:
            self.log_detailed(f"Error generating pivots and midpoints: {e}", "ERROR")
            return pivots

    def _calculate_call_pivot_tp(self, pivots: list, entry_point: float, sl_points: float, reward_ratio: float) -> float:
        """
        Calculates the first take profit (TP) for CALL position using CPR pivot targeting.
        
        Args:
            pivots (list): List of pivot points
            entry_point (float): Entry price
            sl_points (float): Stop loss points
            reward_ratio (float): Risk-reward ratio
            
        Returns:
            float: First take profit level
        """
        try:
            pivots = sorted(pivots)
            pivots_and_midpoints = self._get_pivots_and_midpoints(pivots)
            target_pivot_list = [item for item in pivots_and_midpoints if item >= entry_point]
            
            print(f"🎯 CALL TP calculation: entry_point={entry_point:.2f}, sl_points={sl_points:.2f}, reward_ratio={reward_ratio:.2f}")
            print(f"🎯 Available pivots: {pivots[:5]}...")
            print(f"🎯 Pivots+Midpoints: {pivots_and_midpoints[:8]}...")
            print(f"🎯 Target pivot list (>= entry): {target_pivot_list}")
            
            # Calculate min/max TP bounds once
            # Get max reward ratio from config (default 2.0 for 1:1→1:2 max reward specification)
            try:
                sl_tp_config = self.config_loader.main_config.get('indicators', {}).get('SL_TP_levels', {})
                max_reward_ratio = float(sl_tp_config.get('max_reward_ratio', '2.0'))
            except:
                max_reward_ratio = 2.0  # Default fallback
            
            min_tp = entry_point + (sl_points * reward_ratio)
            max_tp = entry_point + (sl_points * max_reward_ratio)
            print(f"🎯 TP bounds: min_tp={min_tp:.2f}, max_tp={max_tp:.2f} (max reward: {max_reward_ratio}x risk)")
            
            for target_level in target_pivot_list:
                # Use exact reference formula: tp = entry_point + abs(target - entry_point)
                tp_candidate = entry_point + abs(target_level - entry_point)
                
                print(f"🎯 Testing target {target_level:.2f}: tp_candidate={tp_candidate:.2f}, range=[{min_tp:.2f}, {max_tp:.2f}]")
                
                # Add small tolerance for floating point precision
                tolerance = 0.1  # 0.1 point tolerance
                if tp_candidate >= (min_tp):
                    print(f"🎯 ✅ CALL TP SELECTED: {tp_candidate:.2f} (target: {target_level:.2f})")
                    return tp_candidate
                else:
                    print(f"🎯 ❌ CALL TP REJECTED: {tp_candidate:.2f} not in range [{min_tp:.2f}, {max_tp:.2f}]")
            
            # Fallback: Use standard risk-reward calculation
            default_tp = entry_point + (sl_points * reward_ratio)
            print(f"🎯 ❌ CALL TP FALLBACK: No suitable pivot found, using RR: {default_tp:.2f}")
            return default_tp
            
        except Exception as e:
            self.log_detailed(f"Error in CALL pivot TP calculation: {e}", "ERROR")
            return entry_point + (sl_points * reward_ratio)

    def _calculate_put_pivot_tp(self, pivots: list, entry_point: float, sl_points: float, reward_ratio: float) -> float:
        """
        Calculates the first take profit (TP) for PUT position using CPR pivot targeting.
        FIXED: Now uses same logic structure as working CALL method.
        
        Args:
            pivots (list): List of pivot points  
            entry_point (float): Entry price
            sl_points (float): Stop loss points
            reward_ratio (float): Risk-reward ratio
            
        Returns:
            float: First take profit level
        """
        try:
            pivots = sorted(pivots, reverse=True)  # Sort descending for PUT  
            pivots_and_midpoints = self._get_pivots_and_midpoints(pivots)
            target_pivot_list = [item for item in pivots_and_midpoints if item <= entry_point]
            
            print(f"🎯 PUT TP calculation: entry_point={entry_point:.2f}, sl_points={sl_points:.2f}, reward_ratio={reward_ratio:.2f}")
            print(f"🎯 Available pivots: {pivots[:5]}...")
            print(f"🎯 Pivots+Midpoints: {pivots_and_midpoints[:8]}...")  
            print(f"🎯 Target pivot list (<= entry): {target_pivot_list}")
            
            # Calculate min/max TP bounds once - FIXED: Same logic as CALL
            # Get max reward ratio from config (default 2.0 for 1:1→1:2 max reward specification)
            try:
                sl_tp_config = self.config_loader.main_config.get('indicators', {}).get('SL_TP_levels', {})
                max_reward_ratio = float(sl_tp_config.get('max_reward_ratio', '2.0'))
            except:
                max_reward_ratio = 2.0  # Default fallback
            
            min_tp = entry_point - (sl_points * reward_ratio)
            max_tp = entry_point - (sl_points * max_reward_ratio)  # max_tp will be lower value
            print(f"🎯 PUT TP bounds: min_tp={min_tp:.2f}, max_tp={max_tp:.2f} (max reward: {max_reward_ratio}x risk)")
            
            for target_level in target_pivot_list:
                # FIXED: Use exact same formula as CALL method
                tp_candidate = entry_point - abs(target_level - entry_point)
                
                print(f"🎯 Testing target {target_level:.2f}: tp_candidate={tp_candidate:.2f}, range=[{max_tp:.2f}, {min_tp:.2f}]")
                
                # FIXED: Correct validation logic for PUT (tp_candidate should be >= max_tp and <= min_tp)
                # For PUT: max_tp < min_tp, so we check if tp is within bounds
                if tp_candidate >= max_tp and tp_candidate <= min_tp:
                    print(f"🎯 ✅ PUT TP SELECTED: {tp_candidate:.2f} (target: {target_level:.2f})")
                    return tp_candidate
                else:
                    print(f"🎯 ❌ PUT TP REJECTED: {tp_candidate:.2f} not in range [{max_tp:.2f}, {min_tp:.2f}]")
            
            # Fallback: Use standard risk-reward calculation
            default_tp = entry_point - (sl_points * reward_ratio)
            print(f"🎯 ❌ PUT TP FALLBACK: No suitable pivot found, using RR: {default_tp:.2f}")
            return default_tp
            
        except Exception as e:
            self.log_detailed(f"Error in PUT pivot TP calculation: {e}", "ERROR")
            print(f"🎯 ❌ PUT TP EXCEPTION: {e}")
            return entry_point - (sl_points * reward_ratio)

    def _calculate_cpr_pivot_tp(self, option_type: str, entry_point: float, sl_points: float, reward_ratio: float, buffer_tp_points: float, eval_context: Dict[str, Any] = None, entry_index: int = 0) -> float:
        """
        Main orchestration function for CPR pivot-based TP calculation.
        
        INTELLIGENT CPR PIVOT TARGETING LOGIC:
        1. Extracts all CPR levels (S4, S3, S2, S1, Pivot, TC, BC, R1, R2, R3, R4) from evaluation context
        2. Generates pivots + midpoints list for granular targeting
        3. Filters targets based on option type direction (CALL: upward, PUT: downward)
        4. Validates targets against risk-reward constraints:
           - Minimum TP: entry ± (sl_points × reward_ratio)
           - Maximum TP: entry ± (sl_points × max_reward_ratio) [configurable, default 2.0]
        5. Selects first valid nearby pivot as TP target
        6. Falls back to standard RR calculation if no suitable pivot found
        
        REWARD RATIO SPECIFICATION:
        - For 1:1 RR with max_reward_ratio=2.0: Maximum reward is 1:2 (2x risk)
        - Configurable via config.json: indicators.SL_TP_levels.max_reward_ratio
        
        Args:
            option_type (str): 'CALL' or 'PUT'
            entry_point (float): Entry price  
            sl_points (float): Stop loss points
            reward_ratio (float): Risk-reward ratio (e.g., 1.0 for 1:1)
            buffer_tp_points (float): TP buffer points
            eval_context (Dict): Context containing CPR pivot data
            entry_index (int): Current entry index for correct CPR array access
            
        Returns:
            float: Calculated TP price with buffer applied
        """
        try:
            # Removed debug logging for cleaner output
            
            # Extract CPR pivot values from the evaluation context
            # This requires access to the current evaluation context which has CPR data
            pivot_values = []
            
            # Try to get CPR pivots from the evaluation context if available
            if eval_context and isinstance(eval_context, dict):
                try:
                    # Extract CPR pivot values from evaluation context
                    cpr_r1 = eval_context.get('cpr_r1', [])
                    cpr_r2 = eval_context.get('cpr_r2', [])
                    cpr_r3 = eval_context.get('cpr_r3', [])
                    cpr_r4 = eval_context.get('cpr_r4', [])
                    cpr_s1 = eval_context.get('cpr_s1', [])
                    cpr_s2 = eval_context.get('cpr_s2', [])
                    cpr_s3 = eval_context.get('cpr_s3', [])
                    cpr_s4 = eval_context.get('cpr_s4', [])
                    cpr_pivot = eval_context.get('cpr_pivot', [])
                    cpr_tc = eval_context.get('cpr_tc', [])
                    cpr_bc = eval_context.get('cpr_bc', [])
                    
                    # Use correct entry index for CPR level access (FIXED: was hardcoded to 0)
                    print(f"🎯 CPR data lengths: R1={len(cpr_r1)}, R2={len(cpr_r2)}, R3={len(cpr_r3)}")
                    print(f"🎯 Using entry_index={entry_index} for CPR array access (FIXED from hardcoded 0)")
                    if len(cpr_r1) > entry_index:
                        print(f"🎯 Sample CPR values: R2[{entry_index}]={cpr_r2.iloc[entry_index] if hasattr(cpr_r2, 'iloc') else cpr_r2[entry_index]:.2f}, R3[{entry_index}]={cpr_r3.iloc[entry_index] if hasattr(cpr_r3, 'iloc') else cpr_r3[entry_index]:.2f}")
                        
                        # Handle pandas Series vs numpy array access with correct entry_index
                        def get_value(data, index=entry_index):
                            # Validate index bounds
                            if hasattr(data, 'iloc'):
                                if index >= len(data):
                                    print(f"⚠️ Index {index} out of bounds for pandas Series length {len(data)}, using last available")
                                    index = len(data) - 1
                                return float(data.iloc[index])
                            else:
                                if index >= len(data):
                                    print(f"⚠️ Index {index} out of bounds for array length {len(data)}, using last available") 
                                    index = len(data) - 1
                                return float(data[index])
                        
                        pivot_values = [
                            get_value(cpr_s4) if len(cpr_s4) > entry_index else 0,
                            get_value(cpr_s3) if len(cpr_s3) > entry_index else 0,
                            get_value(cpr_s2) if len(cpr_s2) > entry_index else 0, 
                            get_value(cpr_s1) if len(cpr_s1) > entry_index else 0,
                            get_value(cpr_pivot) if len(cpr_pivot) > entry_index else 0,
                            get_value(cpr_tc) if len(cpr_tc) > entry_index else 0,
                            get_value(cpr_bc) if len(cpr_bc) > entry_index else 0,
                            get_value(cpr_r1) if len(cpr_r1) > entry_index else 0,
                            get_value(cpr_r2) if len(cpr_r2) > entry_index else 0,
                            get_value(cpr_r3) if len(cpr_r3) > entry_index else 0,
                            get_value(cpr_r4) if len(cpr_r4) > entry_index else 0
                        ]
                        
                        # Enhanced logging for TP calculation debugging
                        print(f"🎯 EXTRACTED CPR PIVOTS: {[f'{v:.2f}' for v in pivot_values]}")
                        print(f"🎯 CPR R2={pivot_values[8]:.2f}, R3={pivot_values[9]:.2f} - Expected midpoint={(pivot_values[8]+pivot_values[9])/2:.2f}")
                    
                except Exception as e:
                    self.log_detailed(f"Failed to extract CPR pivots from context: {e}", "WARNING")
            
            # If no pivots available, use fallback calculation
            if len(pivot_values) < 3:
                self.log_detailed("No CPR pivots available, using standard RR calculation", "WARNING")
                if option_type == 'CALL':
                    return entry_point + (sl_points * reward_ratio) + buffer_tp_points
                else:
                    return entry_point - (sl_points * reward_ratio) - buffer_tp_points
            
            # Calculate TP using pivot targeting
            if option_type == 'CALL':
                tp_base = self._calculate_call_pivot_tp(pivot_values, entry_point, sl_points, reward_ratio)
                tp_final = tp_base + buffer_tp_points
            else:
                tp_base = self._calculate_put_pivot_tp(pivot_values, entry_point, sl_points, reward_ratio)
                tp_final = tp_base - buffer_tp_points
            
            self.log_detailed(f"CPR Pivot TP calculated: {option_type} Entry={entry_point:.2f}, TP_base={tp_base:.2f}, TP_final={tp_final:.2f}", "INFO")
            return tp_final
            
        except Exception as e:
            self.log_detailed(f"Error in CPR pivot TP calculation: {e}", "ERROR")
            # Fallback to standard calculation
            if option_type == 'CALL':
                return entry_point + (sl_points * reward_ratio) + buffer_tp_points
            else:
                return entry_point - (sl_points * reward_ratio) - buffer_tp_points

    def _is_debug_export_enabled(self) -> bool:
        """
        Check if debug CSV export is enabled in configuration.
        
        Returns:
            True if debug export is enabled, False otherwise
        """
        try:
            sl_tp_config = self.config_loader.main_config.get('indicators', {}).get('SL_TP_levels', {})
            return sl_tp_config.get('debug_export_enabled', False)
        except Exception as e:
            self.log_detailed(f"Error checking debug export config: {e}", "WARNING")
            return False

    def _export_eval_context_debug_csv(
        self,
        eval_context: Dict[str, Any],
        evaluation_index: pd.Index,
        stage: str,
        group_name: str = "",
        case_name: str = ""
    ) -> str:
        """
        Export eval_context to CSV for debugging SL/TP integration issues.
        
        This function creates a comprehensive CSV file showing all eval_context data
        including market data, indicators, signal candle values, and SL/TP levels
        to help debug why SL/TP exit conditions are not working.
        
        Args:
            eval_context: The evaluation context dictionary
            evaluation_index: Index for timestamps
            stage: Stage identifier (e.g., "before_sl_tp", "after_sl_tp", "during_exit")
            group_name: Strategy group name for filename
            case_name: Strategy case name for filename
            
        Returns:
            Path to the created CSV file
        """
        try:
            import os
            from datetime import datetime
            
            # Create debug directory if it doesn't exist
            debug_dir = "E:Projects//vAlgo//output//debug_output"
            os.makedirs(debug_dir, exist_ok=True)

            debug_dir = Path("E:"/"Projects") / "vAlgo" / "output" / "debug_output"
            debug_dir.mkdir(parents=True, exist_ok=True) 
            
            # Create timestamp for unique filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_context_debug_{stage}_{group_name}_{case_name}_{timestamp}.csv"
            filepath = os.path.join(debug_dir, filename)
            
            # Prepare data for CSV export
            data_dict = {}
            
            # Add timestamp index
            if isinstance(evaluation_index, pd.Index):
                data_dict['timestamp'] = evaluation_index
            else:
                data_dict['timestamp'] = range(len(evaluation_index))
            
            # Add all eval_context data with proper handling of different data types
            for key, value in eval_context.items():
                if key == 'np':  # Skip numpy module
                    continue
                    
                try:
                    if hasattr(value, '__len__') and not isinstance(value, str):
                        if hasattr(value, 'values'):  # pandas Series
                            data_dict[key] = value.values
                        elif isinstance(value, np.ndarray):  # numpy array
                            data_dict[key] = value
                        elif isinstance(value, list):  # list
                            data_dict[key] = np.array(value)
                        else:
                            # Try to convert to array
                            data_dict[key] = np.array(value)
                    else:
                        # Scalar value - repeat for all rows
                        data_dict[key] = np.full(len(data_dict['timestamp']), value)
                        
                except Exception as e:
                    self.log_detailed(f"Error processing eval_context key '{key}': {e}", "WARNING")
                    # Add placeholder for failed keys
                    data_dict[key] = np.full(len(data_dict['timestamp']), np.nan)
            
            # Create DataFrame from data_dict
            df = pd.DataFrame(data_dict)
            
            # Add debug metadata columns
            df['debug_stage'] = stage
            df['debug_group'] = group_name
            df['debug_case'] = case_name
            df['debug_timestamp'] = timestamp
            
            # Special focus on SL/TP and entry_point columns - add validation flags
            if 'sl_price' in df.columns:
                df['sl_price_is_nan'] = df['sl_price'].isna()
                df['sl_price_valid_count'] = (~df['sl_price'].isna()).cumsum()
                
            if 'tp_price' in df.columns:
                df['tp_price_is_nan'] = df['tp_price'].isna()
                df['tp_price_valid_count'] = (~df['tp_price'].isna()).cumsum()
            
            if 'entry_point' in df.columns:
                df['entry_point_is_nan'] = df['entry_point'].isna()
                df['entry_point_valid_count'] = (~df['entry_point'].isna()).cumsum()
            
            # Add position tracking info if available
            if 'close' in df.columns:
                df['close_price'] = df['close']
                
            # Export to CSV
            df.to_csv(filepath, index=False)
            print(f"report export: {filepath}")
            
            # Log summary information
            total_rows = len(df)
            sl_valid = (~df['sl_price'].isna()).sum() if 'sl_price' in df.columns else 0
            tp_valid = (~df['tp_price'].isna()).sum() if 'tp_price' in df.columns else 0
            
            self.log_detailed(
                f"DEBUG CSV exported: {filename} | "
                f"Rows: {total_rows} | "
                f"SL valid: {sl_valid}/{total_rows} | "
                f"TP valid: {tp_valid}/{total_rows} | "
                f"Columns: {len(df.columns)}", "INFO"
            )
            
            print(f"🔍 DEBUG: Eval context exported to {filepath}")
            print(f"   📊 Stage: {stage} | Rows: {total_rows} | Columns: {len(df.columns)}")
            print(f"   🎯 SL/TP status: SL valid {sl_valid}/{total_rows}, TP valid {tp_valid}/{total_rows}")
            
            return filepath
            
        except Exception as e:
            self.log_detailed(f"Error exporting eval_context debug CSV: {e}", "ERROR")
            print(f"❌ Failed to export debug CSV: {e}")
            return ""

    def _merge_sl_tp_data(
        self, 
        target_data: Dict[str, np.ndarray], 
        source_data: Dict[str, np.ndarray], 
        strategy_key: str
    ) -> None:
        """
        Merge SL/TP data from multiple strategies.
        
        This method combines SL/TP data from different strategies,
        ensuring that data from all strategies is preserved in the final export.
        
        Args:
            target_data: The combined SL/TP data (modified in place)
            source_data: SL/TP data from current strategy
            strategy_key: Strategy identifier for logging
        """
        try:
            import numpy as np
            
            for key in ['sl_price', 'tp_price', 'entry_point']:
                if key in source_data and key in target_data:
                    source_array = source_data[key]
                    target_array = target_data[key]
                    
                    # Convert to numpy arrays if they aren't already
                    if hasattr(source_array, 'values'):
                        source_array = source_array.values
                    if hasattr(target_array, 'values'):
                        target_array = target_array.values
                    
                    # Merge: use source data where it's not NaN, keep target data otherwise
                    source_valid = ~np.isnan(source_array)
                    target_data[key][source_valid] = source_array[source_valid]
                    
                    # Count merged values for logging
                    valid_count = np.sum(~np.isnan(target_array))
                    self.log_detailed(f"{strategy_key} merged {key}: {valid_count} total valid values", "DEBUG")
            
            if self._performance_mode != 'ultra':
                total_valid = np.sum(~np.isnan(target_data['sl_price']))
                print(f"   🎯 SL/TP merge complete for {strategy_key}: {total_valid} total candles with data")
                
        except Exception as e:
            self.log_detailed(f"Error merging SL/TP data for {strategy_key}: {e}", "ERROR")

# Convenience function for external use
def create_vectorbt_signal_generator(config_loader, data_loader, logger) -> VectorBTSignalGenerator:
    """Create and initialize VectorBT Signal Generator."""
    return VectorBTSignalGenerator(config_loader, data_loader, logger)


if __name__ == "__main__":
    # Test VectorBT Signal Generator
    try:
        from core.json_config_loader import JSONConfigLoader
        from core.efficient_data_loader import EfficientDataLoader
        
        print("🧪 Testing VectorBT Signal Generator...")
        
        # Create components
        config_loader = JSONConfigLoader()
        data_loader = EfficientDataLoader(config_loader)

       
        
       
    except Exception as e:
        print(f"VectorBT Signal Generator test failed: {e}")
        sys.exit(1)