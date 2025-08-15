"""
Signal Extractor - Phase 4 of Ultimate Efficiency Engine
========================================================

Entry/exit signal extraction and trade pair generation.
Extracts Phase 4 functionality from ultimate_efficiency_engine.py for modular architecture.

Features:
- Strategy-level signal grouping and extraction
- Ultra-fast trade pair generation using vectorized operations
- Portfolio creation with VectorBT integration
- Signal analysis and statistics
- Proper entry/exit matching with group exclusivity

Author: vAlgo Development Team
Created: July 29, 2025
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
from collections import defaultdict

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

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
    print("Warning: VectorBT not available - some portfolio features may be limited")


class SignalExtractor(ProcessorBase):
    """
    Entry/exit signal extraction and trade pair generation.
    
    Implements Phase 4 of the Ultimate Efficiency Engine:
    - Extracts entry/exit signals using strategy-level grouping
    - Groups cases under their parent strategy instead of treating each case as separate
    - Generates trade pairs with proper entry/exit matching
    - Creates portfolios for performance analysis
    - Provides signal statistics and analysis
    """
    
    def __init__(self, config_loader, data_loader, logger):
        """Initialize Signal Extractor."""
        super().__init__(config_loader, data_loader, logger)
        
        self.log_major_component("Signal Extractor initialized", "SIGNAL_EXTRACTOR")
    
    def extract_signals(
        self, 
        vectorbt_signals: Dict[str, Any], 
        market_data: pd.DataFrame,
        timeframe: str = "5m"
    ) -> Dict[str, Any]:
        """
        Extract entry/exit signals using strategy-level grouping with multi-timeframe support.
        
        Args:
            vectorbt_signals: VectorBT signals from IndicatorFactory (case-level)
            market_data: OHLCV DataFrame (may be 5m data when using multi-timeframe)
            timeframe: Effective timeframe string (e.g., '1m' for LTP mode, '5m' for standard)
            
        Returns:
            Dictionary with strategy-level extracted signals and trade pairs
        """
        try:
            with self.measure_execution_time("signal_extraction"):
                self.log_major_component("Extracting entry/exit signals (strategy-level grouping)", "SIGNAL_EXTRACTION")
                extraction_start = time.time()
                
                trade_signals = {
                    'entries': [],
                    'exits': [],
                    'active_trades': [],
                    'strategy_performance': {},
                    'trade_pairs': []
                }
                
                # Multi-timeframe detection and data alignment
                ltp_timeframe_enabled = self.config_loader.main_config.get('trading', {}).get('ltp_timeframe', False)
                
                if ltp_timeframe_enabled and timeframe == "1m":
                    # Multi-timeframe mode: signals are on 1m, need to get 1m reference data
                    self.log_detailed("Multi-timeframe mode detected: fetching 1m reference data for signal extraction", "INFO")
                    reference_data = self._get_ltp_reference_data(market_data)
                else:
                    # Standard mode: use provided market data
                    reference_data = market_data
                    self.log_detailed(f"Standard mode: using {timeframe} market data for signal extraction", "INFO")
                
                # RISK MANAGEMENT: Initialize daily entry counter per strategy
                # Format: daily_entry_counts[date][strategy_group] = count
                daily_entry_counts = defaultdict(lambda: defaultdict(int))
                
                # Get max daily entries from risk management config
                max_daily_entries = self.config_loader.main_config.get(
                    'risk_management', {}
                ).get('max_daily_entries', 3)
                
                self.log_detailed(f"Risk Management: Max daily entries per strategy = {max_daily_entries}", "INFO", "RISK_MANAGEMENT")
                print(f"🛡️ Risk Management: Max {max_daily_entries} entries per strategy per day")
                
                # Group cases by strategy for proper portfolio creation
                strategy_groups = {}
                
                # Determine correct array size for signal combination
                ltp_timeframe_enabled = self.config_loader.main_config.get('trading', {}).get('ltp_timeframe', False)
                if ltp_timeframe_enabled and timeframe == "1m":
                    # Multi-timeframe mode: use reference data length (1m)
                    signal_array_length = len(reference_data)
                    self.log_detailed(f"Multi-timeframe mode: Using reference data length {signal_array_length} for signal arrays", "INFO", "SIGNAL_EXTRACTOR")
                else:
                    # Standard mode: use market data length (5m)
                    signal_array_length = len(market_data)
                    self.log_detailed(f"Standard mode: Using market data length {signal_array_length} for signal arrays", "INFO", "SIGNAL_EXTRACTOR")
                
                for strategy_key, signal_data in vectorbt_signals.items():
                    group_name, case_name = strategy_key.split('.', 1)
                    
                    if group_name not in strategy_groups:
                        strategy_groups[group_name] = {
                            'cases': {},
                            'combined_entries': np.zeros(signal_array_length, dtype=bool),
                            'combined_exits': np.zeros(signal_array_length, dtype=bool),
                            'case_count': 0,
                            'total_entry_signals': 0,
                            'total_exit_signals': 0
                        }
                    
                    # Store case data
                    strategy_groups[group_name]['cases'][case_name] = signal_data
                    strategy_groups[group_name]['case_count'] += 1
                
                # Process each STRATEGY (not case) to create ONE portfolio per strategy
                for group_name, group_data in strategy_groups.items():
                    self.log_major_component(f"Processing Strategy Group: {group_name} ({group_data['case_count']} cases)", "STRATEGY_PROCESSING")
                    print(f"Processing Strategy Group: {group_name} ({group_data['case_count']} cases)")
                    
                    # Process each case within the strategy
                    for case_name, case_signals in group_data['cases'].items():
                        # Extract entry/exit arrays from VectorBT results
                        vectorbt_result = case_signals.get('signals', {})
                        
                        if vectorbt_result:
                            entries = vectorbt_result.get('entries', pd.Series(False, index=market_data.index))
                            exits = vectorbt_result.get('exits', pd.Series(False, index=market_data.index))
                            
                            # Convert to boolean arrays if needed
                            if isinstance(entries, pd.Series):
                                entries = entries.fillna(False).astype(bool)
                            if isinstance(exits, pd.Series):
                                exits = exits.fillna(False).astype(bool)
                            
                            # Multi-timeframe alignment check
                            if len(entries) != len(market_data):
                                self.log_detailed(f"Multi-timeframe detected: {len(entries)} signals vs {len(market_data)} market data points", "INFO", "SIGNAL_EXTRACTOR")
                                
                                # In multi-timeframe mode, signals are already on the correct timeframe (1m)
                                # DO NOT resample - this causes signal loss!
                                ltp_timeframe_enabled = self.config_loader.main_config.get('trading', {}).get('ltp_timeframe', False)
                                
                                if ltp_timeframe_enabled:
                                    # Multi-timeframe mode: use signals as-is (1m frequency)
                                    self.log_detailed(f"Multi-timeframe mode: Using 1m signals directly (NO resampling)", "INFO", "SIGNAL_EXTRACTOR")
                                    # signals stay on 1m frequency - no resampling needed
                                else:
                                    # Standard mode: resample signals to match market data frequency
                                    self.log_detailed(f"Standard mode: Resampling {len(entries)} signals to {len(market_data)} market data points", "INFO", "SIGNAL_EXTRACTOR")
                                    entries, exits = self._resample_signals_to_market_timeframe(entries, exits, market_data.index)
                                    self.log_detailed(f"Signal resampling complete: {len(entries)} resampled signals", "INFO", "SIGNAL_EXTRACTOR")
                            
                            entry_count = entries.sum() if hasattr(entries, 'sum') else np.sum(entries)
                            exit_count = exits.sum() if hasattr(exits, 'sum') else np.sum(exits)
                            
                            print(f"   Case Analysis ({group_name}.{case_name}):")
                            print(f"      Entry signals: {entry_count} ({entry_count/signal_array_length*100:.1f}%)")
                            print(f"      Exit signals: {exit_count} ({exit_count/signal_array_length*100:.1f}%)")
                            
                            # Combine signals at strategy level
                            if hasattr(entries, 'values'):
                                group_data['combined_entries'] |= entries.values
                            else:
                                group_data['combined_entries'] |= entries
                                
                            if hasattr(exits, 'values'):
                                group_data['combined_exits'] |= exits.values
                            else:
                                group_data['combined_exits'] |= exits
                            
                            group_data['total_entry_signals'] += entry_count
                            group_data['total_exit_signals'] += exit_count
                            
                            # Store individual entry/exit signals with timeframe for timestamp adjustment
                            # RISK MANAGEMENT: Pass daily entry counters for strategy-wise limits
                            self._store_individual_signals(
                                trade_signals, entries, exits, reference_data, 
                                group_name, case_name, case_signals, timeframe,
                                daily_entry_counts, max_daily_entries
                            )
                    
                    # Create strategy-level combined signals
                    combined_entry_count = np.sum(group_data['combined_entries'])
                    combined_exit_count = np.sum(group_data['combined_exits'])
                    
                    print(f"   Strategy Combined Signals ({group_name}):")
                    print(f"      Combined Entry signals: {combined_entry_count} ({combined_entry_count/signal_array_length*100:.1f}%)")
                    print(f"      Combined Exit signals: {combined_exit_count} ({combined_exit_count/signal_array_length*100:.1f}%)")
                    print(f"   Signal extraction complete for {group_name} (using options premium for P&L)")
                    
                    # Store strategy performance
                    trade_signals['strategy_performance'][group_name] = {
                        'combined_entries': combined_entry_count,
                        'combined_exits': combined_exit_count,
                        'case_count': group_data['case_count'],
                        'extraction_time': time.time() - extraction_start
                    }
                
                # Generate trade pairs from extracted signals
                trade_pairs = self._generate_trade_pairs_from_signals(trade_signals, reference_data)
                trade_signals['trade_pairs'] = trade_pairs
                
                extraction_time = time.time() - extraction_start
                
                self.log_major_component(
                    f"Strategy-Level Entry/Exit extraction complete: {len(strategy_groups)} groups, {len(trade_signals['entries'])} entries, {len(trade_signals['exits'])} exits",
                    "SIGNAL_EXTRACTION"
                )
                
                # RISK MANAGEMENT: Simple summary
                total_dates = len(daily_entry_counts)
                print(f"🛡️ Risk Management: Daily limits enforced across {total_dates} trading days")
                
                print(f"Strategy-Level Entry/Exit extraction complete:")
                print(f"   Strategy Groups Processed: {len(strategy_groups)}")
                print(f"   Total entry signals: {len(trade_signals['entries'])}")
                print(f"   Total exit signals: {len(trade_signals['exits'])}")
                print(f"   Trade pairs: {len(trade_pairs)}")
                print(f"   Options P&L Analysis: Ready for premium database lookup")
                print(f"   Extraction time: {extraction_time:.4f}s")
                
                return trade_signals
                
        except Exception as e:
            self.log_detailed(f"Error in signal extraction: {e}", "ERROR")
            raise ConfigurationError(f"Signal extraction failed: {e}")
    
    def _get_ltp_reference_data(self, market_data_5m: pd.DataFrame) -> pd.DataFrame:
        """
        Get 1m reference data for multi-timeframe signal extraction.
        
        When in LTP timeframe mode, signals are generated on 1m data but we need
        corresponding 1m OHLCV data for signal extraction and timestamp alignment.
        
        Args:
            market_data_5m: Original 5m market data passed to signal extraction
            
        Returns:
            1m OHLCV DataFrame for signal extraction reference
        """
        try:
            # Get configuration for 1m data fetch
            backtesting_config = self.config_loader.main_config.get('backtesting', {})
            trading_config = self.config_loader.main_config.get('trading', {})
            
            symbol = trading_config.get('symbol', 'NIFTY')
            exchange = trading_config.get('exchange', 'NSE_INDEX')
            start_date = backtesting_config.get('start_date')
            end_date = backtesting_config.get('end_date')
            
            if not start_date or not end_date:
                raise ConfigurationError("start_date and end_date required for LTP reference data")
            
            # Fetch 1m LTP data using data loader
            ltp_reference_data = self.data_loader.get_ltp_timeframe_data(
                symbol=symbol,
                exchange=exchange,
                start_date=start_date,
                end_date=end_date
            )
            
            if ltp_reference_data.empty:
                self.log_detailed("No 1m reference data found, falling back to 5m data", "WARNING")
                return market_data_5m
            
            # Apply same filtering as in signal generator for consistency (using config)
            from datetime import time
            # Get trading start time from config instead of hardcoding
            trading_start_str = self.config_loader.main_config.get('risk_management', {}).get('trading_start_time', '09:20')
            hour, minute = map(int, trading_start_str.split(':'))
            trading_start_time = time(hour, minute)  # From config: when first 5m candle completes
            original_count = len(ltp_reference_data)
            ltp_reference_data = ltp_reference_data[ltp_reference_data.index.time >= trading_start_time]
            filtered_count = len(ltp_reference_data)
            
            if ltp_reference_data.empty:
                self.log_detailed(f"No 1m reference data after {trading_start_str} filtering, falling back to 5m data", "WARNING")
                return market_data_5m
            
            self.log_detailed(
                f"LTP reference data filtered to trading start: {original_count} -> {filtered_count} records "
                f"(starting from {ltp_reference_data.index[0]})", "INFO"
            )
            return ltp_reference_data
            
        except Exception as e:
            self.log_detailed(f"Error getting LTP reference data: {e}", "WARNING")
            # Fallback to original market data
            return market_data_5m
    
    def _resample_signals_to_market_timeframe(
        self, 
        entries: pd.Series, 
        exits: pd.Series, 
        target_index: pd.Index
    ) -> tuple[pd.Series, pd.Series]:
        """
        Resample high-frequency signals (e.g., 1m) to lower frequency (e.g., 5m) for pipeline compatibility.
        
        Args:
            entries: High-frequency entry signals (e.g., 1m timestamps)
            exits: High-frequency exit signals (e.g., 1m timestamps)  
            target_index: Target timeframe index (e.g., 5m timestamps)
            
        Returns:
            Tuple of (resampled_entries, resampled_exits) aligned to target_index
        """
        try:
            # Convert to DataFrame for easier resampling
            signal_df = pd.DataFrame({
                'entries': entries,
                'exits': exits
            }, index=entries.index)
            
            # Determine resampling frequency from target index
            if len(target_index) > 1:
                # Infer frequency from target index
                freq = pd.infer_freq(target_index)
                if freq is None:
                    # Fallback: calculate approximate frequency
                    time_diff = target_index[1] - target_index[0]
                    if time_diff == pd.Timedelta(minutes=5):
                        freq = '5T'  # 5 minutes
                    else:
                        freq = '5T'  # Default to 5 minutes
            else:
                freq = '5T'  # Default frequency
            
            # Resample using 'any' aggregation (if any signal in period, trigger)
            resampled = signal_df.resample(freq).any()
            
            # Align with target index using forward fill + backward fill
            resampled = resampled.reindex(target_index, method='ffill').fillna(False)
            
            # Extract resampled series
            resampled_entries = resampled['entries'].astype(bool)
            resampled_exits = resampled['exits'].astype(bool)
            
            self.log_detailed(
                f"Signal resampling: {len(entries)} -> {len(resampled_entries)} records "
                f"(freq: {freq}, entries: {entries.sum()} -> {resampled_entries.sum()}, "
                f"exits: {exits.sum()} -> {resampled_exits.sum()})", 
                "DEBUG", "SIGNAL_EXTRACTOR"
            )
            
            return resampled_entries, resampled_exits
            
        except Exception as e:
            self.log_detailed(f"Error in signal resampling: {e}", "ERROR", "SIGNAL_EXTRACTOR")
            # Fallback: create empty signals with target index
            return (pd.Series(False, index=target_index), pd.Series(False, index=target_index))
    
    def _store_individual_signals(
        self,
        trade_signals: Dict[str, Any],
        entries: Union[pd.Series, np.ndarray],
        exits: Union[pd.Series, np.ndarray], 
        market_data: pd.DataFrame,
        group_name: str,
        case_name: str,
        case_signals: Dict[str, Any],
        timeframe: str,
        daily_entry_counts: defaultdict,
        max_daily_entries: int
    ) -> None:
        """OPTIMIZATION: Store individual entry/exit signals with PRE-CALCULATED STRIKES."""
        try:
            timestamps = market_data.index
            prices = market_data['close']
            option_type = case_signals.get('option_type', '')
            position_size = case_signals.get('position_size', 1)
            
            
            # PERFORMANCE OPTIMIZATION: Get strategy configuration for strike calculation
            strategy_config = self._get_strategy_config_for_case(group_name, case_name)
            strike_type = strategy_config.get('StrikeType', 'ATM')
            
            print(f"STRIKE PRE-CALCULATION: {group_name}.{case_name} using {strike_type} strikes")
            
            # Process entry signals with PRE-CALCULATED STRIKES
            if hasattr(entries, 'values'):
                entry_indices = np.where(entries.values)[0]
            else:
                entry_indices = np.where(entries)[0]
                
            # RISK MANAGEMENT: Track entries processed and filtered
            entries_processed = 0
            entries_filtered = 0
            
            for idx in entry_indices:
                entries_processed += 1
                underlying_price = prices.iloc[idx]
                
                # CRITICAL: Adjust timestamp to reflect when signal is actually available
                signal_timestamp = self._adjust_signal_timestamp(timestamps[idx], timeframe)
                
                # RISK MANAGEMENT: Check daily entry limit per strategy
                signal_date = signal_timestamp.date()  # Extract date from adjusted timestamp
                current_daily_count = daily_entry_counts[signal_date][group_name]
                
                if current_daily_count >= max_daily_entries:
                    entries_filtered += 1
                    # Log first few filtered signals for debugging
                    if entries_filtered <= 3:
                        self.log_detailed(
                            f"Daily limit reached: {group_name} has {current_daily_count}/{max_daily_entries} entries on {signal_date}. "
                            f"Skipping signal at {signal_timestamp}",
                            "INFO", "RISK_MANAGEMENT"
                        )
                    continue  # Skip this signal - daily limit reached for this strategy
                
                # RISK MANAGEMENT: Increment daily counter for this strategy FIRST
                daily_entry_counts[signal_date][group_name] += 1
                
                # TRADE TRACKING: Capture daily sequence number for Trade1/Trade2/Trade3 tracking
                daily_trade_sequence = daily_entry_counts[signal_date][group_name]
                
                # KEY OPTIMIZATION: Calculate strike during signal extraction (not P&L phase)
                entry_strike = self._calculate_strike_for_strategy(underlying_price, strike_type, option_type)
                
                # Store the entry signal (passed daily limit check)
                trade_signals['entries'].append({
                    'index': idx,
                    'timestamp': signal_timestamp,  # ADJUSTED: Next candle start time
                    'price': underlying_price,
                    'group': group_name,
                    'case': case_name,
                    'option_type': option_type,
                    'position_size': position_size,
                    'strike': entry_strike,  # PRE-CALCULATED STRIKE
                    'strike_type': strike_type,
                    'daily_trade_sequence': daily_trade_sequence  # TRADE TRACKING: 1, 2, or 3
                })
            
            # Process exit signals with SAME STRIKES (for matching)
            if hasattr(exits, 'values'):
                exit_indices = np.where(exits.values)[0]
            else:
                exit_indices = np.where(exits)[0]
                
            for idx in exit_indices:
                underlying_price = prices.iloc[idx]
                
                # CRITICAL: For exits, we'll match with entry strikes during trade pairing
                # Store underlying price for now, actual strike matching happens in trade pairing
                exit_strike = self._calculate_strike_for_strategy(underlying_price, strike_type, option_type)
                
                # CRITICAL: Adjust timestamp to reflect when signal is actually available
                signal_timestamp = self._adjust_signal_timestamp(timestamps[idx], timeframe)
                
                trade_signals['exits'].append({
                    'index': idx,
                    'timestamp': signal_timestamp,  # ADJUSTED: Next candle start time
                    'price': underlying_price,
                    'group': group_name,
                    'case': case_name,
                    'option_type': option_type,
                    'position_size': position_size,
                    'strike': exit_strike,  # Will be overridden during trade pairing
                    'strike_type': strike_type
                })
            
            # RISK MANAGEMENT: Log filtering results for this case
            entries_accepted = entries_processed - entries_filtered
            if entries_filtered > 0:
                print(f"     🛡️ Risk Management ({group_name}.{case_name}): {entries_processed} signals -> {entries_accepted} accepted, {entries_filtered} filtered (daily limit)")
                self.log_detailed(
                    f"Daily limit filtering: {group_name}.{case_name} - {entries_processed} signals processed, "
                    f"{entries_accepted} accepted, {entries_filtered} filtered by daily limit",
                    "INFO", "RISK_MANAGEMENT"
                )
                
        except Exception as e:
            self.log_detailed(f"Error storing individual signals: {e}", "ERROR")
    
    def _parse_timeframe_to_minutes(self, timeframe: str) -> int:
        """
        Parse timeframe string to minutes with strict validation.
        
        Args:
            timeframe: Timeframe string (e.g., '5m', '1m', '3m', '15m', '1h')
            
        Returns:
            Number of minutes as integer
            
        Raises:
            ConfigurationError: If timeframe format is invalid
        """
        if not timeframe or not isinstance(timeframe, str):
            raise ConfigurationError(f"Invalid timeframe type: {timeframe}. Expected string like '5m', '3m', '1m'")
        
        timeframe = timeframe.lower().strip()
        
        if timeframe.endswith('m'):
            try:
                minutes = int(timeframe[:-1])
                if minutes <= 0:
                    raise ConfigurationError(f"Invalid timeframe minutes: {minutes}. Must be positive integer")
                return minutes
            except ValueError:
                raise ConfigurationError(f"Invalid timeframe format: '{timeframe}'. Expected format: '5m', '3m', etc.")
        
        elif timeframe.endswith('h'):
            try:
                hours = int(timeframe[:-1])
                if hours <= 0:
                    raise ConfigurationError(f"Invalid timeframe hours: {hours}. Must be positive integer")
                return hours * 60  # Convert hours to minutes
            except ValueError:
                raise ConfigurationError(f"Invalid timeframe format: '{timeframe}'. Expected format: '1h', '2h', etc.")
        
        else:
            raise ConfigurationError(f"Unsupported timeframe format: '{timeframe}'. Supported: '1m', '5m', '15m', '1h', etc.")
    
    def _adjust_signal_timestamp(self, candle_timestamp, timeframe: str):
        """
        Convert candle close timestamp to actual signal timestamp (next candle start).
        
        When a signal is generated at a candle close (e.g., 9:30), the actual signal
        becomes available only when the next candle starts (e.g., 9:35 for 5m timeframe).
        
        OPTIMIZATION: For LTP-enabled mode with 1m timeframe, no adjustment is needed
        since we're using perfectly merged 1m+5m data with correct timestamps.
        
        Args:
            candle_timestamp: Original candle close timestamp
            timeframe: Timeframe string (e.g., '5m', '3m', '1m')
            
        Returns:
            Adjusted timestamp representing when signal is actually available
            
        Raises:
            ConfigurationError: If timeframe is invalid or timestamp calculation fails
        """
        try:
            # Check if LTP timeframe is enabled with 1m data (perfectly merged data)
            ltp_timeframe_enabled = self.config_loader.main_config.get('trading', {}).get('ltp_timeframe', False)
            
            if ltp_timeframe_enabled and timeframe == "1m":
                # LTP mode with 1m: Use original timestamp (no adjustment needed)
                # Merged 1m+5m data already has the correct execution timestamps
                self.log_detailed(
                    f"LTP mode: Using original timestamp {candle_timestamp} (no adjustment needed)", 
                    "DEBUG"
                )
                return candle_timestamp
            
            # Standard mode: Apply timeframe adjustment
            timeframe_minutes = self._parse_timeframe_to_minutes(timeframe)  # Will raise if invalid
            signal_timestamp = candle_timestamp + pd.Timedelta(minutes=timeframe_minutes)
            
            self.log_detailed(
                f"Standard mode timestamp adjustment: {candle_timestamp} -> {signal_timestamp} (+{timeframe_minutes}m)", 
                "DEBUG"
            )
            
            return signal_timestamp
            
        except ConfigurationError:
            raise  # Re-raise timeframe validation errors
        except Exception as e:
            raise ConfigurationError(f"Error adjusting signal timestamp: {e}")
    
    def _get_strategy_config_for_case(self, group_name: str, case_name: str) -> Dict[str, Any]:
        """Get strategy configuration for strike calculation."""
        try:
            if hasattr(self, 'strategies_config') and self.strategies_config:
                strategy_config = self.strategies_config.get(group_name, {})
                case_config = strategy_config.get('cases', {}).get(case_name, {})
                return case_config
            return {}
        except Exception as e:
            self.log_detailed(f"Error getting strategy config: {e}", "WARNING")
            return {}
    
    def _calculate_strike_for_strategy(self, underlying_price: float, strike_type: str, option_type: str) -> int:
        """🚀 ULTRA-FAST strike calculation using NIFTY-specific logic."""
        try:
            # NIFTY-specific strike step
            strike_step = 50
            
            # Base ATM calculation (round to nearest 50)
            atm_strike = int(round(underlying_price / strike_step) * strike_step)
            
            if strike_type.upper() == 'ATM':
                return atm_strike
            
            # Handle ITM/OTM strikes
            if strike_type.upper().startswith('ITM') or strike_type.upper().startswith('OTM'):
                # Extract offset (ITM1 -> 1, OTM2 -> 2, etc.)
                try:
                    offset = int(strike_type[3:]) if len(strike_type) > 3 else 1
                except ValueError:
                    offset = 1
                
                # Calculate strike based on option type and moneyness
                if (strike_type.upper().startswith('ITM') and option_type.upper() == 'CALL') or \
                   (strike_type.upper().startswith('OTM') and option_type.upper() == 'PUT'):
                    # CALL ITM or PUT OTM: strikes below underlying
                    return atm_strike - (offset * strike_step)
                else:
                    # CALL OTM or PUT ITM: strikes above underlying
                    return atm_strike + (offset * strike_step)
            
            # Default to ATM
            return atm_strike
            
        except Exception as e:
            self.log_detailed(f"Error calculating strike: {e}", "WARNING")
            # Fallback to simple ATM
            return int(round(underlying_price / 50) * 50)
    
    def _generate_trade_pairs_from_signals(
        self, 
        trade_signals: Dict[str, Any], 
        market_data: pd.DataFrame
    ) -> List[Dict[str, Any]]:
        """Generate trade pairs using ultra-fast vectorized operations."""
        try:
            with self.measure_execution_time("trade_pair_generation"):
                entries = trade_signals['entries']
                exits = trade_signals['exits']
                
                if not entries or not exits:
                    self.log_detailed("No entries or exits available for trade pair generation", "WARNING")
                    return []
                
                return self._ultra_fast_trade_pair_generation(entries, exits)
                
        except Exception as e:
            self.log_detailed(f"Error generating trade pairs: {e}", "ERROR")
            return []
    
    def _ultra_fast_trade_pair_generation(self, entries: List[Dict], exits: List[Dict]) -> List[Dict]:
        """Ultra-fast trade pair generation using vectorized group-based matching."""
        try:
            if not entries or not exits:
                return []
            
            # Group entries and exits by strategy group for proper matching
            entry_groups = {}
            exit_groups = {}
            
            for entry in entries:
                group_key = entry['group']
                if group_key not in entry_groups:
                    entry_groups[group_key] = []
                entry_groups[group_key].append(entry)
            
            for exit in exits:
                group_key = exit['group']
                if group_key not in exit_groups:
                    exit_groups[group_key] = []
                exit_groups[group_key].append(exit)
            
            trade_pairs = []
            
            # Match entries and exits within each group
            for group_key in entry_groups:
                if group_key in exit_groups:
                    group_entries = sorted(entry_groups[group_key], key=lambda x: x['index'])
                    group_exits = sorted(exit_groups[group_key], key=lambda x: x['index'])
                    
                    # Simple matching: each entry gets matched with the next available exit
                    entry_idx = 0
                    exit_idx = 0
                    
                    while entry_idx < len(group_entries) and exit_idx < len(group_exits):
                        entry = group_entries[entry_idx]
                        exit = group_exits[exit_idx]
                        
                        # Exit must come after entry
                        if exit['index'] > entry['index']:
                            # 🚀 CRITICAL OPTIMIZATION: Use SAME STRIKE for entry and exit
                            entry_strike = entry.get('strike', int(round(entry['price'] / 50) * 50))
                            
                            seq_num = entry.get('daily_trade_sequence', 0)
                            trade_pairs.append({
                                'entry': entry,
                                'exit': exit,
                                'group': entry['group'],
                                'case': entry['case'],
                                'duration_candles': exit['index'] - entry['index'],
                                'duration_time': exit['timestamp'] - entry['timestamp'],
                                'gross_pnl': (exit['price'] - entry['price']) * self.main_config['trading']['lot_size'],
                                'option_type': entry['option_type'],
                                # Enhanced fields for options P&L calculation with PRE-CALCULATED STRIKES
                                'entry_timestamp': entry['timestamp'],
                                'entry_price': entry['price'],
                                'exit_timestamp': exit['timestamp'],
                                'exit_price': exit['price'],
                                'entry_strike': entry_strike,     # 🚀 PRE-CALCULATED STRIKE
                                'exit_strike': entry_strike,      # 🚀 SAME STRIKE FOR EXIT
                                'strike_type': entry.get('strike_type', 'ATM'),
                                'position_size': entry.get('position_size', 1),
                                'daily_trade_sequence': seq_num  # TRADE TRACKING
                            })
                            entry_idx += 1
                            exit_idx += 1
                        else:
                            exit_idx += 1
            
            self.log_detailed(f"Generated {len(trade_pairs)} trade pairs from {len(entries)} entries and {len(exits)} exits", "INFO")
            return trade_pairs
            
        except Exception as e:
            self.log_detailed(f"Error in ultra-fast trade pair generation: {e}", "ERROR")
            return []
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for signal extraction."""
        return {
            'processor': self.processor_name,
            'performance_stats': self.performance_stats,
            'vectorbt_available': VECTORBT_AVAILABLE
        }


# Convenience function for external use
def create_signal_extractor(config_loader, data_loader, logger) -> SignalExtractor:
    """Create and initialize Signal Extractor."""
    return SignalExtractor(config_loader, data_loader, logger)


if __name__ == "__main__":
    # Test Signal Extractor
    try:
        from core.json_config_loader import JSONConfigLoader
        from core.efficient_data_loader import EfficientDataLoader
        
        print("Testing Signal Extractor...")
        
        # Create components
        config_loader = JSONConfigLoader()
        data_loader = EfficientDataLoader(config_loader)
        
        # Create fallback logger
        class TestLogger:
            def log_major_component(self, msg, comp): print(f"SIGNAL_GENERATOR: {msg}")
            def log_detailed(self, msg, level, comp): print(f"DETAIL: {msg}")
            def log_performance(self, metrics, comp): print(f"PERFORMANCE: {metrics}")
        
        logger = TestLogger()
        
        # Create signal extractor
        extractor = SignalExtractor(config_loader, data_loader, logger)
        
        print(f"Signal Extractor test completed!")
        print(f"   Performance: {extractor.get_performance_summary()}")
        
    except Exception as e:
        print(f"Signal Extractor test failed: {e}")
        sys.exit(1)