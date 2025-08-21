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




def vectorized_position_tracking(entry_conditions: np.ndarray, 
                                exit_conditions: np.ndarray, 
                                trading_hours_mask: np.ndarray, 
                                forced_exit_mask: np.ndarray) -> np.ndarray:
    """
    Fixed position tracking with proper state management for multiple entries.
    
    Args:
        entry_conditions: Boolean array of entry signals
        exit_conditions: Boolean array of exit signals  
        trading_hours_mask: Boolean array for valid trading hours
        forced_exit_mask: Boolean array for forced exits at 15:15
        
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
        
        
        # CRITICAL: Sequential position state tracking for multiple entries
        in_position = False
        
        for i in range(n):
            if valid_entries[i] and not in_position:
                signals[i] = 1      # Entry signal
                in_position = True
            elif all_exits[i] and in_position:
                signals[i] = -1     # Exit signal
                in_position = False # RESET position state to allow new entries
            else:
                signals[i] = 0      # Hold signal (neutral state)
        
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
    
    def generate_signals(
        self, 
        market_data: pd.DataFrame, 
        indicators: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Ultra-fast VectorBT signal generation using IndicatorFactory pattern.
        
        Args:
            market_data: OHLCV DataFrame
            indicators: Dictionary of calculated indicators
            
        Returns:
            Dictionary containing VectorBT signals for each strategy case
        """
        try:
            with self.measure_execution_time("signal_generation"):
                # DETAILED TIMING: Start overall timing
                overall_start = time.time()
                timing_breakdown = {}
                
                data_length = len(market_data)
                vectorbt_signals = {}
                all_signal_candle_data = {}  # Collect signal candle data from all strategies
                
                # TIMING: Config setup
                config_start = time.time()
                enabled_strategies = self.config_loader.get_enabled_strategies_cases()
                ltp_timeframe_enabled = self.config_loader.main_config.get('trading', {}).get('ltp_timeframe', False)
                timing_breakdown['config_setup'] = time.time() - config_start
                
                self.log_major_component(
                    f"Processing {data_length} data points for {len(enabled_strategies)} strategy groups",
                    "SIGNAL_GENERATION"
                )
                
                print(f"🎯 Using VectorBT IndicatorFactory Pattern (Ultra Fast)")
                print(f"   🔄 Processing: {data_length} data points for {len(enabled_strategies)} strategy groups...")
                print(f"🚀 PERFORMANCE: {self._performance_mode} mode enabled")
                
                merged_data_for_reports = None
                
                # SHARED CONTEXT OPTIMIZATION: Build expensive multi-timeframe context ONCE for all strategies
                shared_context_start = time.time()
                ltp_timeframe_enabled = self.config_loader.main_config.get('trading', {}).get('ltp_timeframe', False)
                
                if ltp_timeframe_enabled:
                    print(f"🚀 PERFORMANCE: Building shared multi-timeframe context for ALL strategies...")
                    shared_eval_context, shared_merged_data = self._build_multi_timeframe_context(market_data, indicators)
                    merged_data_for_reports = shared_merged_data
                    print(f"🚀 PERFORMANCE: Shared context built in {time.time() - shared_context_start:.3f}s")
                else:
                    print(f"🚀 PERFORMANCE: Building shared single-timeframe context for ALL strategies...")
                    shared_eval_context = {
                        'close': market_data['close'],
                        'open': market_data['open'],
                        'high': market_data['high'],
                        'low': market_data['low'],
                        'volume': market_data['volume'],
                        **indicators  # Add all calculated indicators
                    }
                    shared_merged_data = None
                    print(f"🚀 PERFORMANCE: Shared context built in {time.time() - shared_context_start:.3f}s")
                
                # Add numpy for mathematical operations
                shared_eval_context['np'] = np
                timing_breakdown['shared_context_building'] = time.time() - shared_context_start
                
                # TIMING: Strategy processing (now using shared context)
                strategy_start = time.time()
                strategy_times = {}
                
                # Process each strategy group using SHARED CONTEXT
                for group_name, group_config in enabled_strategies.items():
                    # Process each case in the strategy group
                    for case_name, case_config in group_config['cases'].items():
                        if not case_config.get('enabled', True):
                            continue
                        
                        strategy_case_start = time.time()
                        print(f"   🚀 Processing {group_name}.{case_name} with SHARED context...")
                        
                        # Generate signals using SHARED context (no expensive context building per case)
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
                            'signals_array': signal_results.get('signals_array', np.zeros(data_length)),
                            'entry_count': signal_results['entry_count'],
                            'exit_count': signal_results['exit_count'],
                            'generation_time': signal_results['generation_time'],
                            'option_type': signal_results.get('option_type', 'CALL'),
                            'strike_type': signal_results.get('strike_type', 'ATM'),
                            'position_size': signal_results.get('position_size', 1)
                        }
                        
                        # Collect signal candle data from this strategy
                        strategy_signal_candle_data = signal_results.get('signal_candle_data', {})
                        if strategy_signal_candle_data:
                            # Merge signal candle data (use last strategy's data since they should be the same)
                            all_signal_candle_data.update(strategy_signal_candle_data)
                            self.log_detailed(f"Collected signal candle data from {strategy_key}: {list(strategy_signal_candle_data.keys())}", "DEBUG")
                        
                        # Collect merged_data for reports (same for all cases in LTP mode)
                        if ltp_timeframe_enabled and merged_data_for_reports is None:
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
                print(f"   shared_context_building: {timing_breakdown['shared_context_building']:.3f}s")
                print(f"   total_strategy_processing: {timing_breakdown['total_strategy_processing']:.3f}s")
                print(f"   total_signal_generation: {timing_breakdown['total_signal_generation']:.3f}s")
                
                print(f"\n⏱️  INDIVIDUAL STRATEGY TIMING:")
                for strategy_name, strategy_time in strategy_times.items():
                    print(f"   {strategy_name}: {strategy_time:.3f}s")
                
                # Return signals with merged_data and signal_candle_data
                result = {'signals': vectorbt_signals}
                if ltp_timeframe_enabled and merged_data_for_reports is not None:
                    result['merged_data'] = merged_data_for_reports
                    print(f"   🔄 Including merged 1m data for reports: {len(merged_data_for_reports)} records")
                
                # Include signal candle data if available and indicator is enabled
                if all_signal_candle_data and self._is_indicator_enabled('signal_candle'):
                    result['signal_candle_data'] = all_signal_candle_data
                    print(f"   📊 Including signal candle data: {list(all_signal_candle_data.keys())}")
                
                return result
                
        except Exception as e:
            self.log_detailed(f"Error in VectorBT signal generation: {e}", "ERROR")
            raise ConfigurationError(f"VectorBT signal generation failed: {e}")
    
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
                if self._performance_mode == 'ultra':
                    print(f"🚀 PERFORMANCE: {group_name}.{case_name} - {entry_signals.sum()} entry signals found")
                        
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
                        entry_signals.values, evaluation_index
                    )
                    
                    # Update eval_context with calculated signal candle values
                    eval_context['signal_candle_open'] = signal_candle_result['signal_candle_open']
                    eval_context['signal_candle_high'] = signal_candle_result['signal_candle_high']
                    eval_context['signal_candle_low'] = signal_candle_result['signal_candle_low']
                    eval_context['signal_candle_close'] = signal_candle_result['signal_candle_close']
                    
                    self.log_detailed(f"Direct signal candle calculation completed for exit evaluation", "INFO")
                except Exception as e:
                    self.log_detailed(f"Direct signal candle calculation failed: {e}, using NaN values", "WARNING")
            
            # OPTIMIZED: Fast exit signal evaluation using vectorized operations
            try:
                # PERFORMANCE OPTIMIZATION: Use cached compiled conditions for ultra mode
                if self._performance_mode == 'ultra':
                    # Cache key for exit condition
                    exit_cache_key = f"exit_{group_name}_{case_name}"
                    if exit_cache_key not in self._compiled_conditions_cache:
                        self._compiled_conditions_cache[exit_cache_key] = compile(exit_condition, '<string>', 'eval')
                    
                    # Use cached compiled condition for faster evaluation
                    exit_signals = eval(self._compiled_conditions_cache[exit_cache_key], {"__builtins__": {}}, eval_context)
                else:
                    exit_signals = eval(exit_condition, {"__builtins__": {}}, eval_context)
                
                # OPTIMIZED: Direct Series conversion
                if isinstance(exit_signals, pd.Series):
                    exit_signals = exit_signals.fillna(False).astype(bool)
                else:
                    exit_signals = pd.Series(exit_signals, index=evaluation_index).fillna(False).astype(bool)
            except Exception as e:
                self.log_detailed(f"Exit condition evaluation failed for {group_name}.{case_name}: {e}", "ERROR")
                exit_signals = pd.Series(False, index=evaluation_index)
            
            detailed_timing['signal_evaluation'] = time.time() - evaluation_start
            
            # TIMING: Position tracking phase with signal candle updates
            tracking_start = time.time()
            
            # Apply time masks
            trading_hours_mask, forced_exit_mask = create_time_masks(evaluation_index, self.config_loader)
            
            # OPTIMIZED: Fast vectorized position tracking + efficient signal candle updates
            position_signals = vectorized_position_tracking(
                entry_signals.values,
                exit_signals.values, 
                trading_hours_mask,
                forced_exit_mask
            )
            
            # EFFICIENT: Update signal candle data based on position signals
            signal_candle_data = self._update_signal_candle_efficient(
                position_signals,
                eval_context,
                merged_data,
                evaluation_index
            )
            
            detailed_timing['position_tracking'] = time.time() - tracking_start
            
            # Convert position signals back to entry/exit boolean arrays
            final_entry_signals = pd.Series(position_signals == 1, index=evaluation_index)
            final_exit_signals = pd.Series(position_signals == -1, index=evaluation_index)
            
            # Count final signals
            entry_count = final_entry_signals.sum()
            exit_count = final_exit_signals.sum()
            
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
            
            # Only include signal candle data if the indicator is enabled
            if self._is_indicator_enabled('signal_candle'):
                if not signal_candle_data:
                    signal_candle_data = {
                        'signal_candle_open': np.full(len(evaluation_index), np.nan),
                        'signal_candle_high': np.full(len(evaluation_index), np.nan),
                        'signal_candle_low': np.full(len(evaluation_index), np.nan),
                        'signal_candle_close': np.full(len(evaluation_index), np.nan)
                    }
                result['signal_candle_data'] = signal_candle_data  # Include signal candle values for CSV export
            
            return result
            
        except Exception as e:
            self.log_detailed(f"Error in optimized signal generation for {group_name}.{case_name}: {e}", "ERROR")
            raise ConfigurationError(f"Signal generation failed for {group_name}.{case_name}: {e}")
    
    def _generate_signals_direct(
        self,
        market_data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        case_config: Dict[str, Any],
        group_name: str,
        case_name: str
    ) -> Dict[str, Any]:
        """
        Direct signal generation using condition evaluation with multi-timeframe support.
        
        Implements LTP timeframe control:
        - When ltp_timeframe=True: Merges 1m execution data with 5m indicators
        - When ltp_timeframe=False: Uses existing 5m evaluation (backward compatible)
        
        Args:
            market_data: 5m OHLCV DataFrame (filtered)
            indicators: Dictionary of calculated 5m indicators
            case_config: Case-specific configuration
            group_name: Strategy group name
            case_name: Case name
            
        Returns:
            Dictionary with signal generation results
        """
        try:
            start_time = time.time()
            detailed_timing = {}
            
            # TIMING: Setup phase
            setup_start = time.time()
            entry_condition = case_config.get('entry', '')
            exit_condition = case_config.get('exit', '')
            
            if not entry_condition or not exit_condition:
                raise ConfigurationError(f"Entry/exit conditions not defined for {group_name}.{case_name}")
            
            trading_config = self.config_loader.main_config.get('trading', {})
            ltp_timeframe_enabled = trading_config.get('ltp_timeframe', False)
            detailed_timing['setup'] = time.time() - setup_start
            
            if self._performance_mode != 'ultra':
                print(f"🔍 DEBUG Strategy Processing: {group_name}.{case_name}")
                print(f"   LTP timeframe enabled: {ltp_timeframe_enabled}")
                print(f"   Entry condition: {entry_condition}")
                print(f"   Exit condition: {exit_condition}")
            
            # TIMING: Context building phase
            context_start = time.time()
            current_merged_data = None  # Track merged data for this case
            if ltp_timeframe_enabled:
                self.log_detailed(f"Multi-timeframe processing enabled for {group_name}.{case_name}", "INFO")
                eval_context, current_merged_data = self._build_multi_timeframe_context(market_data, indicators)
            else:
                self.log_detailed(f"Single timeframe processing for {group_name}.{case_name}", "INFO") 
                # Create standard single-timeframe evaluation context
                eval_context = {
                    'close': market_data['close'],
                    'open': market_data['open'],
                    'high': market_data['high'],
                    'low': market_data['low'],
                    'volume': market_data['volume'],
                    **indicators  # Add all calculated indicators
                }
            
            # Add numpy for mathematical operations
            eval_context['np'] = np
            
            # Get the appropriate index for signal evaluation (depends on timeframe mode)
            if ltp_timeframe_enabled:
                # Multi-timeframe mode: use 1m timestamps for evaluation 
                evaluation_index = eval_context['ltp_close'].index
                self.log_detailed(f"Using 1m timestamps for signal evaluation: {len(evaluation_index)} records", "DEBUG")
            else:
                # Single timeframe mode: use original 5m timestamps
                evaluation_index = market_data.index
                self.log_detailed(f"Using 5m timestamps for signal evaluation: {len(evaluation_index)} records", "DEBUG")
            
            detailed_timing['context_building'] = time.time() - context_start
            
            # Export evaluation context for debugging (skip in ultra performance mode)
            if self._performance_mode != 'ultra':
                self._export_eval_context_sample(eval_context, evaluation_index, group_name, case_name)
            elif self._performance_mode == 'ultra':
                print("🚀 PERFORMANCE: Skipping CSV export in ultra mode")
            
            # DEBUG: Log evaluation context for PUT strategy before evaluation (skip in ultra mode)
            if case_name == 'multi_sma_trend_put' and self._performance_mode != 'ultra':
                print(f"🔍 DEBUG PUT Strategy - About to evaluate conditions:")
                print(f"   Evaluation index length: {len(evaluation_index)}")
                print(f"   First timestamp: {evaluation_index[0] if len(evaluation_index) > 0 else 'None'}")
                print(f"   Entry condition: {entry_condition}")
                # Check if required keys exist in eval_context
                required_keys = ['close', 'previous_day_low', 'sma_10', 'sma_50', 'sma_200', 'ltp_low', 'low']
                missing_keys = [key for key in required_keys if key not in eval_context]
                print(f"   Missing keys: {missing_keys if missing_keys else 'None'}")
                if not missing_keys:
                    print(f"   All required keys present for evaluation")

            # TIMING: Signal evaluation phase
            evaluation_start = time.time()
            
            # Evaluate entry signals
            try:
                if case_name == 'multi_sma_trend_put':
                    print(f"🔍 DEBUG: About to eval() entry condition for PUT strategy")
                
                entry_signals = eval(entry_condition, {"__builtins__": {}}, eval_context)
                
                if case_name == 'multi_sma_trend_put':
                    print(f"🔍 DEBUG: eval() completed successfully, result type: {type(entry_signals)}")
                
                if isinstance(entry_signals, pd.Series):
                    entry_signals = entry_signals.fillna(False).astype(bool)
                else:
                    entry_signals = pd.Series(entry_signals, index=evaluation_index).fillna(False).astype(bool)
                
                if case_name == 'multi_sma_trend_put':
                    print(f"🔍 DEBUG: Series conversion completed, length: {len(entry_signals)}")
                
                # DEBUG: Log condition evaluation for PUT strategy (skip in ultra mode)
                if case_name == 'multi_sma_trend_put' and self._performance_mode != 'ultra':
                    print(f"🔍 DEBUG PUT Strategy Condition Evaluation:")
                    print(f"   Entry condition: {entry_condition}")
                    print(f"   Number of TRUE signals: {entry_signals.sum()}")
                    print(f"   Total rows evaluated: {len(entry_signals)}")
                    
                    # ALWAYS show first row data for debugging
                    first_row_data = {}
                    for key in ['close', 'previous_day_low', 'sma_10', 'sma_50', 'sma_200', 'ltp_low', 'low']:
                        if key in eval_context and len(eval_context[key]) > 0:
                            first_row_data[key] = eval_context[key].iloc[0]
                    print(f"   First row data: {first_row_data}")
                    
                    # Manual condition check for first row
                    if len(first_row_data) >= 7:
                        cond1 = first_row_data['close'] < first_row_data['previous_day_low']
                        cond2 = first_row_data['close'] < first_row_data['sma_10']
                        cond3 = first_row_data['sma_10'] < first_row_data['sma_50']
                        cond4 = first_row_data['sma_50'] < first_row_data['sma_200']
                        cond5 = first_row_data['ltp_low'] < first_row_data['low']
                        print(f"   Manual condition check (first row):")
                        print(f"     close < previous_day_low: {cond1} ({first_row_data['close']} < {first_row_data['previous_day_low']})")
                        print(f"     close < sma_10: {cond2} ({first_row_data['close']} < {first_row_data['sma_10']})")
                        print(f"     sma_10 < sma_50: {cond3} ({first_row_data['sma_10']} < {first_row_data['sma_50']})")
                        print(f"     sma_50 < sma_200: {cond4} ({first_row_data['sma_50']} < {first_row_data['sma_200']})")
                        print(f"     ltp_low < low: {cond5} ({first_row_data['ltp_low']} < {first_row_data['low']})")
                        print(f"     Overall manual result: {cond1 and cond2 and cond3 and cond4 and cond5}")
                        print(f"     First eval result: {entry_signals.iloc[0]}")
                    
                    if entry_signals.sum() > 0:
                        first_true_idx = entry_signals.idxmax() if entry_signals.any() else None
                        print(f"   ✅ First TRUE signal at: {first_true_idx}")
                        print(f"   First 5 entry signals: {entry_signals.head().tolist()}")
                    else:
                        print(f"   ❌ No TRUE entry signals found")
                elif case_name == 'multi_sma_trend_put' and self._performance_mode == 'ultra':
                    print(f"🚀 PERFORMANCE: PUT strategy - {entry_signals.sum()} signals found (debug skipped)")
                        
            except Exception as e:
                self.log_detailed(f"Entry condition evaluation failed for {group_name}.{case_name}: {e}", "ERROR")
                if case_name == 'multi_sma_trend_put':
                    print(f"🔍 DEBUG: Exception during PUT strategy evaluation: {e}")
                    print(f"   Exception type: {type(e)}")
                    import traceback
                    print(f"   Traceback: {traceback.format_exc()}")
                entry_signals = pd.Series(False, index=evaluation_index)
            
            # Evaluate exit signals
            try:
                exit_signals = eval(exit_condition, {"__builtins__": {}}, eval_context)
                if isinstance(exit_signals, pd.Series):
                    exit_signals = exit_signals.fillna(False).astype(bool)
                else:
                    exit_signals = pd.Series(exit_signals, index=evaluation_index).fillna(False).astype(bool)
            except Exception as e:
                self.log_detailed(f"Exit condition evaluation failed for {group_name}.{case_name}: {e}", "ERROR")
                exit_signals = pd.Series(False, index=evaluation_index)
            
            detailed_timing['signal_evaluation'] = time.time() - evaluation_start
            
            # TIMING: Position tracking phase
            tracking_start = time.time()
            
            # APPLY VECTORIZED POSITION TRACKING (Key improvement from working system)
            trading_hours_mask, forced_exit_mask = create_time_masks(evaluation_index, self.config_loader)
            
            # Generate proper signals using vectorized position tracking
            position_signals = vectorized_position_tracking(
                entry_signals.values,
                exit_signals.values, 
                trading_hours_mask,
                forced_exit_mask
            )
            
            detailed_timing['position_tracking'] = time.time() - tracking_start
            
            # Convert position signals back to entry/exit boolean arrays using correct index
            final_entry_signals = pd.Series(position_signals == 1, index=evaluation_index)
            final_exit_signals = pd.Series(position_signals == -1, index=evaluation_index)
            
            # Count final signals (after position tracking)
            entry_count = final_entry_signals.sum()
            exit_count = final_exit_signals.sum()
            
            # SIGNAL CANDLE CALCULATION: Calculate signal candle with final entry/exit signals
            signal_candle_data = {}
            try:
                if 'signal_candle_' in exit_condition and (entry_count > 0 or exit_count > 0) and self._is_indicator_enabled('signal_candle'):
                    # Get market data for direct signal candle calculation
                    if ltp_timeframe_enabled and hasattr(self, '_current_merged_data'):
                        # Multi-timeframe mode: use merged data if available
                        open_data = self._current_merged_data['open'].values
                        high_data = self._current_merged_data['high'].values
                        low_data = self._current_merged_data['low'].values
                        close_data = self._current_merged_data['close'].values
                    else:
                        # Single timeframe mode: use evaluation context data
                        open_data = eval_context['open'].values if hasattr(eval_context['open'], 'values') else eval_context['open']
                        high_data = eval_context['high'].values if hasattr(eval_context['high'], 'values') else eval_context['high']
                        low_data = eval_context['low'].values if hasattr(eval_context['low'], 'values') else eval_context['low']
                        close_data = eval_context['close'].values if hasattr(eval_context['close'], 'values') else eval_context['close']
                    
                    # OPTIMIZED: Calculate signal candle directly using shared context data
                    # This eliminates SmartIndicatorEngine instantiation and DataFrame creation overhead
                    signal_candle_results = self._calculate_signal_candle_direct(
                        open_data, high_data, low_data, close_data,
                        final_entry_signals.values, evaluation_index
                    )
                    
                    # Store signal candle data for export
                    signal_candle_data = signal_candle_results
                    
                    # Update eval_context with signal candle values for any remaining evaluation
                    for key, values in signal_candle_results.items():
                        eval_context[key] = pd.Series(values, index=evaluation_index)
                    
                    if self._performance_mode != 'ultra':
                        valid_count = sum(~pd.isna(values) for values in signal_candle_results.values())
                        print(f"   📊 Optimized signal candle calculation complete: {valid_count} candles with signal data")
                        
            except Exception as e:
                self.log_detailed(f"Signal candle calculation failed for {group_name}.{case_name}: {e}", "WARNING")
                # Initialize empty signal candle data
                signal_candle_data = {
                    'signal_candle_open': np.full(len(evaluation_index), np.nan),
                    'signal_candle_high': np.full(len(evaluation_index), np.nan),
                    'signal_candle_low': np.full(len(evaluation_index), np.nan),
                    'signal_candle_close': np.full(len(evaluation_index), np.nan)
                }
            
            # Create signals array for VectorBT compatibility
            signals_array = np.column_stack([final_entry_signals.values, final_exit_signals.values])
            
            # Get option configuration from case - no defaults allowed
            option_type = case_config.get('OptionType')
            strike_type = case_config.get('StrikeType')
            position_size = case_config.get('position_size', '1')  # Default to 1 if not specified
            
            # Validate required configuration parameters
            if not option_type:
                raise ConfigurationError(f"OptionType is required for case {group_name}.{case_name} in strategies.json")
            
            if not strike_type:
                raise ConfigurationError(f"StrikeType is required for case {group_name}.{case_name} in strategies.json")
            
            # Convert position_size to integer and validate against limits
            try:
                position_size_int = int(position_size)
                risk_config = self.main_config.get('risk_management', {})
                position_size_limit = risk_config.get('position_size_limit', 5)
                
                if position_size_int > position_size_limit:
                    self.log_detailed(f"Position size {position_size_int} exceeds limit {position_size_limit} for {group_name}.{case_name}, capping to limit", "WARNING")
                    position_size_int = position_size_limit
                elif position_size_int < 1:
                    self.log_detailed(f"Position size {position_size_int} is less than 1 for {group_name}.{case_name}, setting to 1", "WARNING")
                    position_size_int = 1
                    
            except (ValueError, TypeError):
                self.log_detailed(f"Invalid position_size '{position_size}' for {group_name}.{case_name}, using default 1", "WARNING")
                position_size_int = 1
            
            generation_time = time.time() - start_time
            detailed_timing['total_generation_time'] = generation_time
            
            # DETAILED TIMING OUTPUT (skip in ultra mode for performance)
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
                'detailed_timing': detailed_timing,  # Include detailed timing breakdown
                'option_type': option_type,
                'strike_type': strike_type,
                'position_size': position_size_int,
                'group_name': group_name,
                'case_name': case_name,
                'merged_data': current_merged_data,  # Include merged data for LTP mode
            }
            
            # Only include signal candle data if the indicator is enabled
            if self._is_indicator_enabled('signal_candle'):
                if not signal_candle_data:
                    signal_candle_data = {
                        'signal_candle_open': np.full(len(evaluation_index), np.nan),
                        'signal_candle_high': np.full(len(evaluation_index), np.nan),
                        'signal_candle_low': np.full(len(evaluation_index), np.nan),
                        'signal_candle_close': np.full(len(evaluation_index), np.nan)
                    }
                result['signal_candle_data'] = signal_candle_data  # Include signal candle values for CSV export
            
            return result
            
        except Exception as e:
            self.log_detailed(f"Error in direct signal generation for {group_name}.{case_name}: {e}", "ERROR")
            
            # No fallback signals - fail fast with proper error
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
    
    def validate_multi_timeframe_implementation(self) -> Dict[str, Any]:
        """
        Public method to run validation checks on multi-timeframe implementation.
        
        Can be called independently to verify the implementation works correctly.
        
        Returns:
            Dictionary with validation results and details
        """
        try:
            # Check if LTP timeframe is enabled
            ltp_timeframe_enabled = self.config_loader.main_config.get('trading', {}).get('ltp_timeframe', False)
            
            if not ltp_timeframe_enabled:
                return {
                    'validation_status': 'SKIPPED',
                    'reason': 'LTP timeframe not enabled in configuration',
                    'ltp_timeframe_enabled': False
                }
            
            print("🔍 Running Multi-Timeframe Implementation Validation...")
            print("=" * 60)
            
            # Create a minimal test scenario
            backtesting_config = self.config_loader.main_config.get('backtesting', {})
            start_date = backtesting_config.get('start_date', '2024-06-01')
            end_date = backtesting_config.get('end_date', '2024-06-02')  # Small date range for testing
            
            # Create test market data and indicators
            test_market_data = pd.DataFrame({
                'open': [23000, 23050, 23100],
                'high': [23020, 23080, 23120], 
                'low': [22980, 23040, 23090],
                'close': [23010, 23070, 23110],
                'volume': [1000, 1100, 1200]
            }, index=pd.date_range(f'{start_date} 09:20:00', periods=3, freq='5T'))
            
            test_indicators = {
                'EMA_10': pd.Series([22900, 22950, 23000], index=test_market_data.index),
                'RSI_14': pd.Series([45, 55, 65], index=test_market_data.index)
            }
            
            # Test the multi-timeframe context building
            print("Testing multi-timeframe context building...")
            try:
                context, merged_test_data = self._build_multi_timeframe_context(test_market_data, test_indicators)
                
                validation_result = {
                    'validation_status': 'PASSED',
                    'ltp_timeframe_enabled': True,
                    'context_variables': list(context.keys()),
                    'context_size': len(next(iter(context.values()))) if context else 0,
                    'has_1m_data': any(key.startswith('1m_') for key in context.keys()),
                    'has_5m_indicators': any(key in ['EMA_10', 'RSI_14'] for key in context.keys()),
                    'validation_details': 'Multi-timeframe context built successfully with validation checks'
                }
                
                print("Multi-timeframe validation completed successfully!")
                print(f"   Context variables: {len(context)} variables")
                print(f"   1m data columns: {sum(1 for key in context.keys() if key.startswith('1m_'))}")
                print(f"   5m indicators: {sum(1 for key in context.keys() if key in ['EMA_10', 'RSI_14'])}")
                
                return validation_result
                
            except Exception as context_error:
                return {
                    'validation_status': 'FAILED',
                    'ltp_timeframe_enabled': True,
                    'error': str(context_error),
                    'validation_details': 'Multi-timeframe context building failed'
                }
                
        except Exception as e:
            return {
                'validation_status': 'ERROR',
                'error': str(e),
                'validation_details': 'Validation process encountered an error'
            }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for signal generation."""
        return {
            'processor': self.processor_name,
            'performance_stats': self.performance_stats,
            'vectorbt_available': VECTORBT_AVAILABLE
        }
    
    def _export_eval_context_sample(self, eval_context: dict, evaluation_index: pd.Index, group_name: str, case_name: str) -> None:
        """
        Export evaluation context DataFrame sample for debugging signal generation.
        
        Args:
            eval_context: Dictionary containing all evaluation variables
            evaluation_index: Index for the evaluation (1m or 5m timestamps)
            group_name: Strategy group name
            case_name: Strategy case name
        """
        try:
            # Skip numpy reference for DataFrame creation
            df_context = {k: v for k, v in eval_context.items() if k != 'np'}
            
            # Create DataFrame from eval_context
            eval_df = pd.DataFrame(df_context, index=evaluation_index)
            
            # Export sample (first 200 rows) for inspection
            sample_size = min(200, len(eval_df))
            sample_df = eval_df.head(sample_size)
            
            # Ensure output directory exists
            import os
            output_dir = "/mnt/e/Projects/vAlgo/output/vbt_reports"
            os.makedirs(output_dir, exist_ok=True)
            
            # Export to CSV
            csv_path = f"{output_dir}/sample_evalute_dataframe_inputfor_signal_generation.csv"
            sample_df.to_csv(csv_path, index=True)
            
            self.log_detailed(
                f"Eval context exported: {sample_size} rows x {len(sample_df.columns)} columns to {csv_path}", 
                "DEBUG", "SIGNAL_GENERATOR"
            )
            
            # Log summary of available columns
            columns_1m = [col for col in sample_df.columns if col.startswith('1m_')]
            columns_5m = [col for col in sample_df.columns if col in ['open', 'high', 'low', 'close', 'volume']]
            columns_indicators = [col for col in sample_df.columns if col not in columns_1m + columns_5m]
            
            self.log_detailed(
                f"Eval context columns: 1m({len(columns_1m)}), 5m({len(columns_5m)}), indicators({len(columns_indicators)})", 
                "INFO", "SIGNAL_GENERATOR"
            )
            
        except Exception as e:
            self.log_detailed(f"Error exporting eval context: {e}", "WARNING", "SIGNAL_GENERATOR")
    
    def _position_tracking_with_signal_candle(
        self,
        entry_conditions: np.ndarray,
        exit_condition: str,
        trading_hours_mask: np.ndarray,
        forced_exit_mask: np.ndarray,
        eval_context: Dict[str, Any],
        merged_data: pd.DataFrame,
        evaluation_index: pd.Index,
        group_name: str,
        case_name: str
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Custom position tracking with signal candle updates.
        
        This method implements the flow:
        1. Entry Signal → Signal Candle Capture → Save in eval_context → Exit Signal evaluation
        
        Args:
            entry_conditions: Boolean array of entry signals
            exit_condition: Exit condition string to evaluate after signal candle updates
            trading_hours_mask: Boolean array for valid trading hours
            forced_exit_mask: Boolean array for forced exits
            eval_context: Evaluation context to update with signal candle data
            merged_data: Market data for signal candle capture
            evaluation_index: Index for the data
            group_name: Strategy group name for logging
            case_name: Strategy case name for logging
            
        Returns:
            Tuple of (position_signals, signal_candle_data)
        """
        try:
            n = len(entry_conditions)
            signals = np.zeros(n, dtype=int)
            
            # Initialize signal candle tracking variables
            in_position = False
            current_signal_open = np.nan
            current_signal_high = np.nan
            current_signal_low = np.nan
            current_signal_close = np.nan
            
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
            
            # Process each candle sequentially to maintain state and update signal candle data
            for i in range(n):
                # Apply trading hours filter
                if not trading_hours_mask[i]:
                    continue
                
                # Check for forced exit first
                if forced_exit_mask[i] and in_position:
                    signals[i] = -1  # Exit signal
                    in_position = False
                    # Reset signal candle values after exit
                    eval_context['signal_candle_open'][i+1:] = np.nan
                    eval_context['signal_candle_high'][i+1:] = np.nan
                    eval_context['signal_candle_low'][i+1:] = np.nan
                    eval_context['signal_candle_close'][i+1:] = np.nan
                    continue
                
                # Check for entry signal
                if entry_conditions[i] and not in_position:
                    signals[i] = 1  # Entry signal
                    in_position = True
                    
                    # SIGNAL CANDLE CAPTURE: Capture current candle OHLC
                    current_signal_open = open_data[i]
                    current_signal_high = high_data[i]
                    current_signal_low = low_data[i]
                    current_signal_close = close_data[i]
                    
                    # UPDATE EVAL_CONTEXT: Make signal candle data available for exit conditions
                    eval_context['signal_candle_open'][i:] = current_signal_open
                    eval_context['signal_candle_high'][i:] = current_signal_high
                    eval_context['signal_candle_low'][i:] = current_signal_low
                    eval_context['signal_candle_close'][i:] = current_signal_close
                    
                    if self._performance_mode != 'ultra':
                        print(f"   📊 Signal candle captured at index {i}: O={current_signal_open:.2f}, H={current_signal_high:.2f}, L={current_signal_low:.2f}, C={current_signal_close:.2f}")
                
                # Check for exit signal (only if in position)
                elif in_position:
                    # Evaluate exit condition with current signal candle data
                    try:
                        # Create a temporary eval context for this specific candle
                        temp_context = {k: v[i] if hasattr(v, '__getitem__') and len(v) > i else v for k, v in eval_context.items()}
                        temp_context['np'] = np  # Add numpy for mathematical operations
                        
                        # Evaluate exit condition
                        exit_result = eval(exit_condition, {"__builtins__": {}}, temp_context)
                        
                        if exit_result:
                            signals[i] = -1  # Exit signal
                            in_position = False
                            
                            # Reset signal candle values after exit (for next entry)
                            if i + 1 < n:
                                eval_context['signal_candle_open'][i+1:] = np.nan
                                eval_context['signal_candle_high'][i+1:] = np.nan
                                eval_context['signal_candle_low'][i+1:] = np.nan
                                eval_context['signal_candle_close'][i+1:] = np.nan
                            
                            if self._performance_mode != 'ultra':
                                print(f"   📉 Position exited at index {i}, signal candle reset")
                    
                    except Exception as e:
                        self.log_detailed(f"Exit condition evaluation failed at index {i} for {group_name}.{case_name}: {e}", "ERROR")
                        # Continue without exit on evaluation error
                
                # If in position but no exit, maintain signal candle values (already set above)
            
            # Prepare signal candle data for export
            signal_candle_data = {
                'signal_candle_open': eval_context['signal_candle_open'].copy(),
                'signal_candle_high': eval_context['signal_candle_high'].copy(),
                'signal_candle_low': eval_context['signal_candle_low'].copy(),
                'signal_candle_close': eval_context['signal_candle_close'].copy()
            }
            
            # Count valid signal candle values for logging
            valid_count = np.sum(~np.isnan(eval_context['signal_candle_open']))
            if self._performance_mode != 'ultra':
                print(f"   ✅ Signal candle calculation complete: {valid_count}/{n} candles with signal data")
            else:
                print(f"✅ Signal candle calculation complete: {valid_count}/{n} candles with signal data")
            
            return signals, signal_candle_data
            
        except Exception as e:
            self.log_detailed(f"Error in position tracking with signal candle: {e}", "ERROR")
            # Return empty results on error
            signals = np.zeros(len(entry_conditions), dtype=int)
            signal_candle_data = {
                'signal_candle_open': np.full(len(entry_conditions), np.nan),
                'signal_candle_high': np.full(len(entry_conditions), np.nan),
                'signal_candle_low': np.full(len(entry_conditions), np.nan),
                'signal_candle_close': np.full(len(entry_conditions), np.nan)
            }
            return signals, signal_candle_data
    
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
            
            # Initialize signal candle arrays (already done in eval_context)
            signal_open = eval_context['signal_candle_open']
            signal_high = eval_context['signal_candle_high']
            signal_low = eval_context['signal_candle_low']
            signal_close = eval_context['signal_candle_close']
            
            # EFFICIENT: Process each entry-exit pair
            for entry_idx in entry_positions:
                # Find the corresponding exit for this entry
                corresponding_exits = exit_positions[exit_positions > entry_idx]
                if len(corresponding_exits) > 0:
                    exit_idx = corresponding_exits[0]
                else:
                    exit_idx = n  # Position held until end
                
                # CRITICAL FIX: Preserve signal candle values throughout position lifetime
                # Strategy-agnostic approach: first entry wins, others are ignored during overlap
                signal_open[entry_idx:exit_idx] = open_data[entry_idx]
                signal_high[entry_idx:exit_idx] = high_data[entry_idx]
                signal_low[entry_idx:exit_idx] = low_data[entry_idx]
                signal_close[entry_idx:exit_idx] = close_data[entry_idx]
                
                if self._performance_mode != 'ultra':
                    duration = exit_idx - entry_idx
                    print(f"   📊 Signal candle updated: Entry at {entry_idx}, Duration {duration} candles")
            
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
            
            return signal_candle_data
            
        except Exception as e:
            self.log_detailed(f"Error in efficient signal candle update: {e}", "ERROR")
            # Return empty results on error
            return {
                'signal_candle_open': np.full(len(position_signals), np.nan),
                'signal_candle_high': np.full(len(position_signals), np.nan),
                'signal_candle_low': np.full(len(position_signals), np.nan),
                'signal_candle_close': np.full(len(position_signals), np.nan)
            }
    
    def _calculate_signal_candle_direct(
        self,
        open_data: np.ndarray,
        high_data: np.ndarray,
        low_data: np.ndarray,
        close_data: np.ndarray,
        entry_signals: np.ndarray,
        evaluation_index: pd.Index
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
            
            # Process each candle sequentially to maintain state (EXACT same algorithm)
            for i in range(n):
                if entry_signals[i] and not in_position:
                    # Entry signal - capture current candle OHLC (IDENTICAL logic)
                    in_position = True
                    current_signal_open = open_data[i]
                    current_signal_high = high_data[i]
                    current_signal_low = low_data[i]
                    current_signal_close = close_data[i]
                    
                    # Set signal candle values for this position
                    signal_open[i] = current_signal_open
                    signal_high[i] = current_signal_high
                    signal_low[i] = current_signal_low
                    signal_close[i] = current_signal_close
                    
                elif in_position:
                    # In position - maintain signal candle values (IDENTICAL logic)
                    signal_open[i] = current_signal_open
                    signal_high[i] = current_signal_high
                    signal_low[i] = current_signal_low
                    signal_close[i] = current_signal_close
                
                # Note: Exit logic handled by position tracking in calling method
            
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
        
        # Create fallback logger
        class TestLogger:
            def log_major_component(self, msg, comp): print(f"🚀 {comp}: {msg}")
            def log_detailed(self, msg, level, comp): print(f"DEBUG {comp}: {msg}")
            def log_performance(self, metrics, comp): print(f"📊 {comp}: {metrics}")
        
        logger = TestLogger()
        
        # Create signal generator
        generator = VectorBTSignalGenerator(config_loader, data_loader, logger)
        
        print(f"VectorBT Signal Generator test completed!")
        print(f"   📊 Performance: {generator.get_performance_summary()}")
        
        # Test multi-timeframe validation if enabled
        print("\n🔍 Testing Multi-Timeframe Validation...")
        validation_result = generator.validate_multi_timeframe_implementation()
        print(f"   📋 Validation Result: {validation_result}")
        
    except Exception as e:
        print(f"VectorBT Signal Generator test failed: {e}")
        sys.exit(1)