"""
Options P&L Calculator - Phase 5 of Ultimate Efficiency Engine
==============================================================

Real options P&L calculation with premium lookup and vectorized operations.
Extracts Phase 5 functionality from ultimate_efficiency_engine.py for modular architecture.

Features:
- Ultra-fast vectorized options P&L calculation using NumPy + batch SQL
- Batch premium lookup with single SQL query for maximum performance
- Case-specific option type handling (CALL/PUT per strategy case)
- Strike selection and premium data integration
- Vectorized P&L calculations with proper position sizing

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

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.processor_base import ProcessorBase
from core.json_config_loader import ConfigurationError

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class OptionsPnLCalculator(ProcessorBase):
    """
    Real options P&L calculation with premium lookup and vectorized operations.
    
    Implements Phase 5 of the Ultimate Efficiency Engine:
    - Ultra-fast vectorized options P&L calculation using NumPy + batch SQL
    - Batch premium lookup with case-specific option types
    - Strike selection and premium data integration
    - Vectorized P&L calculations with proper position sizing
    - Real money integration with accurate pricing data
    """
    
    def __init__(self, config_loader, data_loader, logger):
        """Initialize Options P&L Calculator."""
        super().__init__(config_loader, data_loader, logger)
        
        self.log_major_component("Options P&L Calculator initialized", "PNL_CALCULATOR")
        
        # Load special trading days from config for dynamic handling
        self.special_trading_days = self._load_special_trading_days()
    
    def calculate_pnl_with_signals(
        self, 
        close_prices: pd.Series, 
        trade_signals: Dict[str, Any], 
        strategy_name: str = None
    ) -> Dict[str, Any]:
        """
        Ultra-fast vectorized options P&L calculation using trade signals with position sizing.
        
        Args:
            close_prices: Underlying close prices
            trade_signals: Trade signals with position size metadata (contains adjusted timestamps)
            strategy_name: Strategy name for option type determination
            
        Returns:
            Dictionary with options P&L calculations
        """
        try:
            with self.measure_execution_time("pnl_calculation"):
                self.validate_options_enabled()
                
                self.log_major_component("PERFORMANCE OPTIMIZATION: Calculating real options P&L with detailed timing", "PNL_CALCULATION")
                total_start_time = time.time()
                
                # 🔍 DETAILED TIMING METRICS FOR BOTTLENECK IDENTIFICATION
                timing_metrics = {}
                
                # Phase 1: Extract pre-generated trade pairs (with adjusted timestamps)
                phase1_start = time.time()
                trade_pairs = trade_signals.get('trade_pairs', [])
                timing_metrics['trade_pair_extraction'] = time.time() - phase1_start
                
                if not trade_pairs:
                    print("No trade pairs found - returning zero-trades P&L result")
                    # Create timestamps from close_prices index for empty result
                    return self._create_empty_pnl_result(close_prices, close_prices.index)
                
                print(f"🚀 USING PRE-GENERATED TRADE PAIRS: Processing {len(trade_pairs)} trades with ADJUSTED TIMESTAMPS")
                print(f"⏱️  Phase 1 - Trade pair extraction: {timing_metrics['trade_pair_extraction']*1000:.1f}ms")
                
                # Convert trade pairs to trades_data format for compatibility
                phase2_start = time.time()
                trades_data = self._convert_trade_pairs_to_trades_data(trade_pairs)
                timing_metrics['trade_pair_conversion'] = time.time() - phase2_start
                print(f"⏱️  Phase 2 - Trade pair conversion: {timing_metrics['trade_pair_conversion']*1000:.1f}ms")
                
                # FAIL-FAST: Validate trade data conversion
                if not trades_data:
                    raise ConfigurationError(f"FAIL-FAST: Trade pair conversion failed - {len(trade_pairs)} trade pairs resulted in 0 trades_data. Check _convert_trade_pairs_to_trades_data method.")
                
                # Phase 3: Premium lookup and P&L calculation
                phase3_start = time.time()
                result = self._process_trades_with_position_sizing(trades_data, close_prices, strategy_name)
                timing_metrics['premium_lookup_and_pnl'] = time.time() - phase3_start
                print(f"⏱️  Phase 3 - Premium lookup + P&L: {timing_metrics['premium_lookup_and_pnl']*1000:.1f}ms")
                
                # Total timing
                timing_metrics['total_pnl_calculation'] = time.time() - total_start_time
                
                # 📊 PERFORMANCE ANALYSIS WITH FAIL-FAST VALIDATION
                print(f"\n🎯 PERFORMANCE BREAKDOWN:")
                
                # FAIL-FAST: Validate timing_metrics structure
                required_keys = ['trade_pair_extraction', 'trade_pair_conversion', 'premium_lookup_and_pnl', 'total_pnl_calculation']
                for key in required_keys:
                    if key not in timing_metrics:
                        raise ConfigurationError(f"FAIL-FAST: Missing timing metric '{key}' in performance analysis")
                
                # Display performance with corrected keys
                total_time = timing_metrics['total_pnl_calculation']
                print(f"   Trade pair extraction: {timing_metrics['trade_pair_extraction']*1000:.1f}ms ({timing_metrics['trade_pair_extraction']/total_time*100:.1f}%)")
                print(f"   Trade pair conversion: {timing_metrics['trade_pair_conversion']*1000:.1f}ms ({timing_metrics['trade_pair_conversion']/total_time*100:.1f}%)")
                print(f"   Premium lookup + P&L: {timing_metrics['premium_lookup_and_pnl']*1000:.1f}ms ({timing_metrics['premium_lookup_and_pnl']/total_time*100:.1f}%)")
                print(f"   🏁 TOTAL: {total_time*1000:.1f}ms")
                
                # Add timing metrics to result
                result['detailed_timing_metrics'] = timing_metrics
                
                # FAIL-FAST: Validate final result structure
                required_result_keys = ['total_trades', 'trades_data', 'total_pnl']
                for key in required_result_keys:
                    if key not in result:
                        raise ConfigurationError(f"FAIL-FAST: Missing required key '{key}' in P&L result")
                
                # FAIL-FAST: Validate trade count consistency  
                result_trades = result.get('total_trades', 0)
                trades_data_count = len(result.get('trades_data', []))
                if result_trades != trades_data_count:
                    raise ConfigurationError(f"FAIL-FAST: Trade count mismatch - total_trades: {result_trades}, trades_data count: {trades_data_count}")
                
                if result_trades == 0:
                    raise ConfigurationError(f"FAIL-FAST: P&L calculation resulted in 0 trades despite {len(trade_pairs)} input trade pairs. Check premium lookup and validation logic.")
                
                print(f"✅ P&L VALIDATION PASSED: {result_trades} trades with total P&L: ₹{result.get('total_pnl', 0):,.2f}")
                return result
                
        except ConfigurationError:
            # Re-raise configuration errors immediately (fail-fast)
            raise
        except Exception as e:
            # Convert unexpected errors to configuration errors for fail-fast behavior
            raise ConfigurationError(f"FAIL-FAST: Unexpected error in P&L calculation: {e}")
    
    def calculate_pnl(
        self, 
        close_prices: pd.Series, 
        entries: pd.Series, 
        exits: pd.Series, 
        timestamps: pd.Index, 
        strategy_name: str = None
    ) -> Dict[str, Any]:
        """
        Ultra-fast vectorized options P&L calculation using NumPy + batch SQL.
        
        Args:
            close_prices: Underlying close prices
            entries: Entry signals (boolean Series)
            exits: Exit signals (boolean Series)
            timestamps: Timestamps for each data point
            strategy_name: Strategy name for option type determination
            
        Returns:
            Dictionary with options P&L calculations
        """
        try:
            with self.measure_execution_time("pnl_calculation"):
                self.validate_options_enabled()
                
                self.log_major_component("Calculating real options P&L (Ultra-Fast Vectorized)", "PNL_CALCULATION")
                start_time = time.time()
                
                # Step 1: Vectorized signal extraction (no loops)
                entry_mask = entries.values.astype(bool)
                exit_mask = exits.values.astype(bool)
                
                entry_indices = np.where(entry_mask)[0]
                exit_indices = np.where(exit_mask)[0]
                
                if len(entry_indices) == 0:
                    print("No entry signals found - returning zero-trades P&L result")
                    return self._create_empty_pnl_result(close_prices, timestamps)
                
                print(f"Ultra-fast P&L: Processing {len(entry_indices)} entries, {len(exit_indices)} exits")
                
                # Step 2: Vectorized entry/exit matching using NumPy
                trades_data = self._match_entries_exits_vectorized(
                    entry_indices, exit_indices, timestamps, close_prices
                )
                
                if not trades_data:
                    raise ConfigurationError("No valid trades found - entry/exit matching failed")
                
                match_time = time.time()
                print(f"   Entry/Exit matching: {(match_time - start_time)*1000:.1f}ms")
                
                # Step 3: Batch premium lookup with single SQL query
                premium_data = self._batch_get_all_premiums_vectorized(trades_data, strategy_name)
                
                premium_time = time.time()
                print(f"   Batch premium lookup: {(premium_time - match_time)*1000:.1f}ms")
                
                # Step 4: Vectorized P&L calculations (pure NumPy)
                pnl_results = self._calculate_vectorized_pnl(premium_data)
                
                pnl_time = time.time()
                print(f"   Vectorized P&L calculation: {(pnl_time - premium_time)*1000:.1f}ms")
                
                total_time = pnl_time - start_time
                print(f"Total options P&L calculation: {total_time*1000:.1f}ms")
                print(f"   Processed {len(premium_data)} trades with real premium data")
                print(f"   Net P&L: Rs{pnl_results.get('net_pnl', 0):,.2f}")
                print(f"   Win Rate: {pnl_results.get('win_rate', 0)*100:.1f}%")
                
                pnl_results.update({
                    'total_calculation_time': total_time,
                    'trades_processed': len(premium_data),
                    'entry_signals': len(entry_indices),
                    'exit_signals': len(exit_indices)
                })
                
                return pnl_results
                
        except Exception as e:
            self.log_detailed(f"Error in ultra-fast options P&L calculation: {e}", "ERROR")
            raise ConfigurationError(f"Ultra-fast options P&L calculation failed: {e}")
    
    def _create_empty_pnl_result(self, close_prices: pd.Series, timestamps: pd.Index) -> Dict[str, Any]:
        """Create empty P&L result when no trades are generated."""
        return {
            'total_pnl': 0.0,
            'total_return_pct': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'max_drawdown_pct': 0.0,
            'sharpe_ratio': 0.0,
            'initial_capital': self.main_config['trading']['initial_capital'],
            'final_portfolio_value': self.main_config['trading']['initial_capital'],
            'trades_data': [],
            'entry_signals': 0,
            'exit_signals': 0,
            'start_date': timestamps[0],
            'end_date': timestamps[-1],
            'total_days': len(timestamps),
            'avg_daily_return': 0.0
        }
    
    def _match_entries_exits_vectorized(
        self, 
        entry_indices: np.ndarray, 
        exit_indices: np.ndarray, 
        timestamps: pd.Index, 
        close_prices: pd.Series
    ) -> List[Dict[str, Any]]:
        """Vectorized entry/exit matching using NumPy operations."""
        try:
            trades_data = []
            
            # Simple first-in-first-out matching
            entry_idx = 0
            exit_idx = 0
            
            while entry_idx < len(entry_indices) and exit_idx < len(exit_indices):
                entry_pos = entry_indices[entry_idx]
                exit_pos = exit_indices[exit_idx]
                
                # Exit must come after entry
                if exit_pos > entry_pos:
                    trades_data.append({
                        'entry_timestamp': timestamps[entry_pos],
                        'entry_price': close_prices.iloc[entry_pos],
                        'exit_timestamp': timestamps[exit_pos],
                        'exit_price': close_prices.iloc[exit_pos],
                        'duration_candles': exit_pos - entry_pos,
                        'gross_pnl': (close_prices.iloc[exit_pos] - close_prices.iloc[entry_pos]) * self.main_config['trading']['lot_size']
                    })
                    entry_idx += 1
                    exit_idx += 1
                else:
                    exit_idx += 1
            
            return trades_data
            
        except Exception as e:
            self.log_detailed(f"Error matching entries and exits: {e}", "ERROR")
            return []
    
    def _match_entries_exits_with_position_sizing(
        self, 
        entries_data: List[Dict], 
        exits_data: List[Dict], 
        timestamps: pd.Index, 
        close_prices: pd.Series
    ) -> List[Dict[str, Any]]:
        """Enhanced entry/exit matching with position sizing information."""
        try:
            trades_data = []
            
            # Convert to arrays for easier processing
            entry_indices = np.array([entry['index'] for entry in entries_data])
            exit_indices = np.array([exit['index'] for exit in exits_data])
            
            # Simple first-in-first-out matching with position sizing
            entry_idx = 0
            exit_idx = 0
            
            while entry_idx < len(entry_indices) and exit_idx < len(exit_indices):
                entry_pos = entry_indices[entry_idx]
                exit_pos = exit_indices[exit_idx]
                
                if exit_pos > entry_pos:
                    # Valid trade pair found
                    entry_signal = entries_data[entry_idx]
                    exit_signal = exits_data[exit_idx]
                    
                    
                    # Calculate entry strike once and preserve it for exit
                    entry_price = close_prices.iloc[entry_pos]
                    # Use enhanced ATM calculation with proper rounding
                    entry_strike = int(round(entry_price / 50) * 50)
                    
                    trade = {
                        'entry_timestamp': timestamps[entry_pos],
                        'exit_timestamp': timestamps[exit_pos],
                        'entry_price': entry_price,
                        'exit_price': close_prices.iloc[exit_pos],
                        'entry_index': entry_pos,
                        'exit_index': exit_pos,
                        'case': entry_signal.get('case', 'default'),
                        'group': entry_signal.get('group', 'default'),
                        'option_type': entry_signal.get('option_type', 'CALL'),
                        'position_size': entry_signal.get('position_size', 1),
                        'entry_strike': entry_strike,  # CRITICAL: Store entry strike for exit reuse
                        'exit_strike': entry_strike    # CRITICAL: Use same strike for exit
                    }
                    
                    trades_data.append(trade)
                    entry_idx += 1
                    exit_idx += 1
                else:
                    exit_idx += 1
            
            #print(f"DEBUG: Enhanced matching created {len(trades_data)} trades with preserved strikes")
            return trades_data
            
        except Exception as e:
            self.log_detailed(f"Error matching entries and exits with position sizing: {e}", "ERROR")
            return []
    
    def _convert_trade_pairs_to_trades_data(self, trade_pairs: List[Dict]) -> List[Dict]:
        """
        Convert pre-generated trade pairs to trades_data format.
        
        Args:
            trade_pairs: List of trade pairs from signal_extractor with adjusted timestamps
            
        Returns:
            List of trades in expected format for premium lookup
        """
        try:
            trades_data = []
            
            for trade_pair in trade_pairs:
                trade = {
                    'entry_timestamp': trade_pair.get('entry_timestamp'),  # Already adjusted timestamp
                    'exit_timestamp': trade_pair.get('exit_timestamp'),    # Already adjusted timestamp
                    'entry_price': trade_pair.get('entry_price'),
                    'exit_price': trade_pair.get('exit_price'),
                    'entry_strike': trade_pair.get('entry_strike'),
                    'exit_strike': trade_pair.get('exit_strike'),
                    'option_type': trade_pair.get('option_type'),
                    'position_size': trade_pair.get('position_size', 1),
                    'strike_type': trade_pair.get('strike_type', 'ATM'),
                    'group': trade_pair.get('group'),
                    'case': trade_pair.get('case')
                }
                trades_data.append(trade)
            
            print(f"📊 Converted {len(trade_pairs)} trade pairs to trades_data with ADJUSTED TIMESTAMPS")
            return trades_data
            
        except Exception as e:
            self.log_detailed(f"Error converting trade pairs to trades data: {e}", "ERROR")
            return []
    
    def _process_trades_with_position_sizing(
        self, 
        trades_data: List[Dict], 
        close_prices: pd.Series, 
        strategy_name: str
    ) -> Dict[str, Any]:
        """Process trades with position sizing for P&L calculation."""
        try:
            print(f"🔍 BOTTLENECK ANALYSIS: Starting premium lookup for {len(trades_data)} trades")
            
            # Phase 3.1: 🚀 ULTRA-FAST BATCH PREMIUM LOOKUP using pre-calculated strikes
            premium_start = time.time()
            all_enriched_trades = self._ultra_fast_batch_premium_lookup(trades_data, strategy_name)
            premium_time = time.time() - premium_start
            print(f"⏱️  Phase 3.1 - OPTIMIZED Premium lookup: {premium_time*1000:.1f}ms ({premium_time:.3f}s) 🚀")
            
            if not all_enriched_trades:
                print("CRITICAL: No enriched trades after premium lookup - this causes 'NO TRADES GENERATED'")
                self.log_detailed("No enriched trades after premium lookup", "WARNING")
                return self._create_empty_pnl_result(close_prices, close_prices.index)
            
            # Phase 3.2: Vectorized P&L calculation
            pnl_start = time.time()
            pnl_result = self._calculate_vectorized_pnl(all_enriched_trades)
            pnl_time = time.time() - pnl_start
            print(f"⏱️  Phase 3.2 - P&L calculation: {pnl_time*1000:.1f}ms")
            
            # Add detailed timing breakdown
            pnl_result['premium_lookup_time'] = premium_time
            pnl_result['pnl_calculation_time'] = pnl_time
            
            # Performance analysis
            total_phase3_time = premium_time + pnl_time
            print(f"📊 PHASE 3 BREAKDOWN:")
            print(f"   Premium lookup: {premium_time*1000:.1f}ms ({premium_time/total_phase3_time*100:.1f}%)")
            print(f"   P&L calculation: {pnl_time*1000:.1f}ms ({pnl_time/total_phase3_time*100:.1f}%)")
            
            # Add metadata - extract timestamps from close_prices index
            timestamps_index = close_prices.index
            pnl_result.update({
                'trades_data': all_enriched_trades,
                'start_date': timestamps_index[0],
                'end_date': timestamps_index[-1],
                'total_days': len(timestamps_index),
                'entry_signals': len([t for t in all_enriched_trades if 'entry_premium' in t]),
                'exit_signals': len([t for t in all_enriched_trades if 'exit_premium' in t])
            })
            
            return pnl_result
            
        except Exception as e:
            self.log_detailed(f"Error processing trades with position sizing: {e}", "ERROR")
            return self._create_empty_pnl_result(close_prices, close_prices.index)
    
    def _ultra_fast_batch_premium_lookup(self, trades_data: List[Dict], strategy_name: str = None) -> List[Dict]:
        """🚀 ULTRA-FAST batch premium lookup with trade-specific option types - 100x faster than individual queries."""
        try:
            if not trades_data:
                raise ConfigurationError("No trades data provided for premium lookup")
            
            print(f"🚀 ULTRA-FAST PREMIUM LOOKUP: {len(trades_data)} trades with pre-calculated strikes")
            
            # Phase 1: Group trades by option type
            batch_start = time.time()
            trades_by_option_type = {'CALL': [], 'PUT': []}
            
            for i, trade in enumerate(trades_data):
                trade_option_type = trade.get('option_type', 'CALL')
                trades_by_option_type[trade_option_type].append((i, trade))
            
            print(f"🔧 OPTION TYPE GROUPING: {len(trades_by_option_type['CALL'])} CALL trades, {len(trades_by_option_type['PUT'])} PUT trades")
            
            # Phase 2: Process each option type separately
            all_enriched_trades = []
            total_query_time = 0
            
            for option_type, type_trades in trades_by_option_type.items():
                if not type_trades:
                    continue
                    
                print(f"🔄 Processing {option_type} trades: {len(type_trades)} trades")
                
                # Collect timestamp/strike pairs for this option type
                timestamp_strike_pairs = []
                trade_indices = []
                
                for trade_idx, trade in type_trades:
                    entry_timestamp = trade.get('entry_timestamp')
                    exit_timestamp = trade.get('exit_timestamp')
                    entry_strike = trade.get('entry_strike', int(round(trade.get('entry_price', 0) / 50) * 50))
                    
                    # Add to batch lookup (ensuring same strike for entry/exit)
                    entry_key = (entry_timestamp, entry_strike)
                    exit_key = (exit_timestamp, entry_strike)  # SAME STRIKE
                    
                    timestamp_strike_pairs.extend([entry_key, exit_key])
                    trade_indices.append(trade_idx)
                
                # Batch query for this option type
                query_start = time.time()
                batch_premiums = self.data_loader.batch_get_premiums_by_timestamp_strike_pairs(
                    timestamp_strike_pairs, option_type
                )
                query_time = time.time() - query_start
                total_query_time += query_time
                
                print(f"   🚀 {option_type} query: {query_time*1000:.1f}ms for {len(batch_premiums)} results")
                
                # Enrich trades with batch results
                for trade_idx, trade in type_trades:
                    try:
                        entry_timestamp = trade.get('entry_timestamp')
                        exit_timestamp = trade.get('exit_timestamp')
                        entry_strike = trade.get('entry_strike', int(round(trade.get('entry_price', 0) / 50) * 50))
                        
                        entry_key = (entry_timestamp, entry_strike)
                        exit_key = (exit_timestamp, entry_strike)  # SAME STRIKE
                        
                        entry_premium_data = batch_premiums.get(entry_key)
                        exit_premium_data = batch_premiums.get(exit_key)
                        
                        if entry_premium_data and exit_premium_data:
                            # Create enriched trade with real premium data and correct option type
                            enriched_trade = trade.copy()
                            enriched_trade.update({
                                'entry_strike': entry_strike,
                                'exit_strike': entry_strike,  # SAME STRIKE
                                'entry_premium': float(entry_premium_data['premium']),
                                'exit_premium': float(exit_premium_data['premium']),
                                'entry_iv': entry_premium_data.get('iv', 0),
                                'exit_iv': exit_premium_data.get('iv', 0),
                                'option_type': option_type  # Use actual trade option type
                            })
                            all_enriched_trades.append(enriched_trade)
                        else:
                            # Log missing data but don't fail completely
                            missing_type = "entry" if not entry_premium_data else "exit"
                            self.log_detailed(f"Missing {missing_type} premium for {option_type} trade {trade_idx}", "WARNING")
                            
                    except Exception as e:
                        self.log_detailed(f"Error enriching {option_type} trade {trade_idx}: {e}", "WARNING")
                        continue
            
            prep_time = time.time() - batch_start - total_query_time
            total_time = time.time() - batch_start
            
            # Performance metrics
            total_pairs = sum(len(trades_by_option_type[ot]) * 2 for ot in ['CALL', 'PUT'])
            individual_time_estimate = total_pairs * 0.01  # 10ms per individual query
            speedup = individual_time_estimate / total_time if total_time > 0 else 1000
            
            print(f"📊 ULTRA-FAST BATCH LOOKUP PERFORMANCE:")
            print(f"   Preparation: {prep_time*1000:.1f}ms")
            print(f"   Database queries: {total_query_time*1000:.1f}ms")
            print(f"   🏁 TOTAL: {total_time*1000:.1f}ms")
            print(f"   🚀 SPEEDUP: {speedup:.0f}x faster than individual queries")
            print(f"   ✅ SUCCESS: {len(all_enriched_trades)}/{len(trades_data)} trades enriched")
            
            return all_enriched_trades
            
        except Exception as e:
            self.log_detailed(f"Error in ultra-fast batch premium lookup: {e}", "ERROR")
            # Fallback to original method
            print("⚠️  Falling back to original batch method")
            return self._batch_get_all_premiums_vectorized(trades_data, strategy_name)
    
    
    def _batch_get_all_premiums_vectorized(self, trades_data: List[Dict], strategy_name: str = None) -> List[Dict]:
        """Batch premium lookup using case-specific option types."""
        try:
            batch_start = time.time()
            print(f"🔍 DETAILED PREMIUM LOOKUP: {len(trades_data)} trades")
            
            if not trades_data:
                raise ConfigurationError("No trades data provided for premium lookup")
            
            # Timing: Validation
            val_start = time.time()
            self.validate_options_enabled()
            val_time = time.time() - val_start
            print(f"⏱️  Validation time: {val_time*1000:.1f}ms")
            
            # Group trades by case to handle different option types
            #print(f"DEBUG: Grouping {len(trades_data)} trades by case...")
            trades_by_case = {}
            for trade in trades_data:
                case_name = trade.get('case', 'default')
                if case_name not in trades_by_case:
                    trades_by_case[case_name] = []
                trades_by_case[case_name].append(trade)
            
            #print(f"DEBUG: Found cases: {list(trades_by_case.keys())}")
            #print(f"DEBUG: Trade counts per case: {[(case, len(trades)) for case, trades in trades_by_case.items()]}")
            
            all_enriched_trades = []
            
            # Process each case separately to use correct option type
            for case_name, case_trades in trades_by_case.items():
                self.log_detailed(f"Processing case: {case_name} with {len(case_trades)} trades", "INFO")
                
                # Get case configuration for option type
                case_config = None
                if strategy_name and hasattr(self, 'strategies_config'):
                    strategy_config = self.strategies_config.get(strategy_name, {})
                    case_config = strategy_config.get('cases', {}).get(case_name, {})
                    self.log_detailed(f"Found case config for {case_name}: {case_config}", "INFO")
                else:
                    self.log_detailed(f"No strategies_config available, using default for {case_name}", "WARNING")
                
                # Extract timestamps with predetermined strikes for this case
                timestamps_strikes = []
                for trade in case_trades:
                    # Use predetermined strikes from trade data
                    entry_strike = trade.get('entry_strike', int(round(trade['entry_price'] / 50) * 50))
                    exit_strike = trade.get('exit_strike', entry_strike)  # CRITICAL: Use same strike for exit
                    
                    timestamps_strikes.append((trade['entry_timestamp'], trade['entry_price'], 'entry', entry_strike))
                    timestamps_strikes.append((trade['exit_timestamp'], trade['exit_price'], 'exit', exit_strike))
                
                # Get premium data in batch for this case with predetermined strikes
                self.log_detailed(f"Executing premium query for {len(timestamps_strikes)} timestamp/strike pairs", "INFO")
                premium_lookup = self._execute_batch_premium_query_with_strikes(timestamps_strikes, strategy_name, case_config)
                self.log_detailed(f"Premium query returned {len(premium_lookup)} results", "INFO")
                
                # Attach premium data to trades for this case
                for trade in case_trades:
                    entry_strike = trade.get('entry_strike', int(round(trade['entry_price'] / 50) * 50))
                    exit_strike = trade.get('exit_strike', entry_strike)
                    
                    entry_key = (trade['entry_timestamp'], entry_strike)
                    exit_key = (trade['exit_timestamp'], exit_strike)
                                       
                    entry_premium = premium_lookup.get(entry_key, {})
                    exit_premium = premium_lookup.get(exit_key, {})
                    
                    if not entry_premium:
                        # Check if entry timestamp is on a special trading day
                        entry_date = trade['entry_timestamp'].date()
                        if self._is_special_trading_day(entry_date):
                            self.log_detailed(f"Entry on special trading day {entry_date} - skipping trade", "WARNING")
                            continue  # Skip this trade
                        else:
                            self.log_detailed(f"PREMIUM LOOKUP FAILED: No entry premium for {trade['entry_timestamp']} at price {trade['entry_price']}", "ERROR")
                            raise ConfigurationError(f"No entry premium data found for timestamp {trade['entry_timestamp']} at price {trade['entry_price']}")
                    
                    if not exit_premium:
                        # Check if exit timestamp is on a special trading day
                        exit_date = trade['exit_timestamp'].date()
                        if self._is_special_trading_day(exit_date):
                            self.log_detailed(f"Exit on special trading day {exit_date} - skipping trade", "WARNING")
                            continue  # Skip this trade
                        else:
                            self.log_detailed(f"PREMIUM LOOKUP FAILED: No exit premium for {trade['exit_timestamp']} at price {trade['exit_price']}", "ERROR")
                            raise ConfigurationError(f"No exit premium data found for timestamp {trade['exit_timestamp']} at price {trade['exit_price']}")
                    
                    # Use real premium data only - preserve strikes from trade
                    entry_prem_val = entry_premium['premium']
                    exit_prem_val = exit_premium['premium']
                    
                    #print(f"DEBUG: Trade enrichment - Entry Strike: {entry_strike}, Exit Strike: {exit_strike}")
                    #print(f"DEBUG: Trade enrichment - Entry Premium: {entry_prem_val}, Exit Premium: {exit_prem_val}")
                    
                    trade.update({
                        'entry_strike': entry_strike,  # Use preserved strike from trade
                        'exit_strike': exit_strike,    # Use same strike for exit  
                        'entry_premium': entry_prem_val,  # Must have real premium
                        'exit_premium': exit_prem_val,    # Must have real premium
                        'entry_iv': entry_premium.get('iv', 0),
                        'exit_iv': exit_premium.get('iv', 0)
                    })
                    
                    all_enriched_trades.append(trade)
            
            return all_enriched_trades
            
        except Exception as e:
            print(f"CRITICAL: Batch premium lookup failed with error: {e}")
            print(f"Error type: {type(e).__name__}")
            print(f"Failed on {len(trades_data) if trades_data else 'unknown'} trades for strategy: {strategy_name}")
            self.log_detailed(f"CRITICAL: Batch premium lookup failed completely: {e}", "ERROR")
            self.log_detailed(f"Failed on {len(trades_data)} trades for strategy: {strategy_name}", "ERROR")
            # No fallbacks - raise error for exact premium matching requirement
            raise ConfigurationError(f"Premium lookup failed: {e}")
    
    def _execute_batch_premium_query_with_strikes(self, timestamps_strikes: List[Tuple], strategy_name: str = None, case_config: Dict = None) -> Dict:
        """Execute batch premium query using predetermined strikes to ensure same strike for entry/exit."""
        try:
            if not timestamps_strikes:
                raise ConfigurationError("No timestamp/strike data provided for premium query")
            
            self.validate_options_enabled()
            
            premium_lookup = {}
            
            # Determine option type from case_config or strategy or use default
            option_type = ''  # Default string value
            if case_config and 'OptionType' in case_config:
                option_type = case_config['OptionType']
            elif strategy_name and hasattr(self, 'strategies_config'):
                strategy_config = self.strategies_config.get(strategy_name, {})
                for case_name, case_cfg in strategy_config.get('cases', {}).items():
                    option_type = case_cfg.get('OptionType', '')
                    break  # Use first case's option type
            
            # Group timestamps by date for efficient querying
            timestamps_by_date = {}
            for timestamp, price, signal_type, strike in timestamps_strikes:
                date_key = timestamp.date()
                if date_key not in timestamps_by_date:
                    timestamps_by_date[date_key] = []
                timestamps_by_date[date_key].append((timestamp, price, signal_type, strike))
            
            # Query premiums for each date using EfficientDataLoader
            for date_key, date_timestamps in timestamps_by_date.items():
                try:
                    # Process each timestamp/strike pair with predetermined strike
                    for timestamp, underlying_price, signal_type, predetermined_strike in date_timestamps:
                        try:
                            # Use predetermined strike instead of recalculating
                            selected_strike = predetermined_strike
                            
                            # Get premium for predetermined strike and timestamp using data loader
                            premium_data = self.data_loader.get_option_premium_at_time(
                                timestamp=timestamp,
                                strike=selected_strike,
                                option_type=option_type
                            )
                            
                            if not premium_data:
                                # Check if this is a special trading day
                                if self._is_special_trading_day(date_key):
                                    self.log_detailed(f"Special trading day detected: {date_key} - skipping options trading", "WARNING")
                                    continue  # Skip this timestamp gracefully
                                else:
                                    raise ConfigurationError(f"No premium data found for strike {selected_strike} at {timestamp}")
                            
                            # Store premium data with timestamp+strike as key (not price)
                            key = (timestamp, selected_strike)
                            premium_lookup[key] = {
                                'strike': selected_strike,
                                'premium': premium_data.get('premium', 0),
                                'iv': premium_data.get('iv', 0),
                                'option_type': option_type
                            }
                                    
                        except ConfigurationError:
                            # Re-raise configuration errors (no fallback)
                            raise
                        except Exception as e:
                            # Log other errors but continue processing
                            self.log_detailed(f"Error processing timestamp {timestamp} with strike {predetermined_strike}: {e}", "ERROR")
                            continue
                            
                except ConfigurationError:
                    # Re-raise configuration errors (no fallback)
                    raise
                except Exception as e:
                    # Log other errors but continue processing
                    self.log_detailed(f"Error processing date {date_key}: {e}", "ERROR")
                    continue
            
            #print(f"DEBUG: Premium lookup with predetermined strikes returned {len(premium_lookup)} results")
            return premium_lookup
            
        except Exception as e:
            self.log_detailed(f"Error in batch premium query with strikes: {e}", "ERROR")
            raise ConfigurationError(f"Batch premium query with strikes failed: {e}")
    
    def _execute_batch_premium_query(self, timestamps_prices: List[Tuple], strategy_name: str = None, case_config: Dict = None) -> Dict:
        """Execute single SQL query to get all premiums at once using EfficientDataLoader."""
        try:
            if not timestamps_prices:
                raise ConfigurationError("No timestamp/price data provided for premium query")
            
            self.validate_options_enabled()
            
            premium_lookup = {}
            
            # Determine option type from case_config or strategy or use default
            option_type = ''  # Default string value
            if case_config and 'OptionType' in case_config:
                option_type = case_config['OptionType']
            elif strategy_name and hasattr(self, 'strategies_config'):
                strategy_config = self.strategies_config.get(strategy_name, {})
                for case_name, case_cfg in strategy_config.get('cases', {}).items():
                    option_type = case_cfg.get('OptionType', '')
                    break  # Use first case's option type
            
            # Group timestamps by date for efficient querying
            timestamps_by_date = {}
            for timestamp, price, signal_type in timestamps_prices:
                date_key = timestamp.date()
                if date_key not in timestamps_by_date:
                    timestamps_by_date[date_key] = []
                timestamps_by_date[date_key].append((timestamp, price, signal_type))
            
            # Query premiums for each date using EfficientDataLoader
            for date_key, date_timestamps in timestamps_by_date.items():
                try:
                    # Get available strikes for this date using data loader
                    available_strikes = self.data_loader.get_available_strikes_for_date(date_key)
                    
                    if not available_strikes:
                        # Check if this is a special trading day (equity trades, options don't)
                        if self._is_special_trading_day(date_key):
                            self.log_detailed(f"Special trading day detected: {date_key} - skipping options trading", "WARNING")
                            continue  # Skip this date gracefully
                        else:
                            raise ConfigurationError(f"No option strikes available for date {date_key} - real options data required")
                    
                    # Process each timestamp/price pair
                    for timestamp, underlying_price, signal_type in date_timestamps:
                        try:
                            # Get strike type from case config (default to ATM)
                            strike_type = 'ATM'
                            if case_config and 'StrikeType' in case_config:
                                strike_type = case_config['StrikeType']
                            
                            # Select appropriate strike using enhanced strike selection logic
                            if strike_type.upper() == 'ATM':
                                # Use improved ATM logic with NIFTY-specific tolerance
                                selected_strike = self.data_loader.select_best_strike(
                                    underlying_price, available_strikes
                                )
                            else:
                                # Use advanced strike selection for ITM/OTM
                                selected_strike = self.data_loader.select_strike_by_type(
                                    underlying_price, available_strikes, strike_type, option_type
                                )
                            
                            if not selected_strike:
                                raise ConfigurationError(f"No suitable {strike_type} strike found for underlying price {underlying_price} on {date_key}")
                            
                            # Get premium for selected strike and timestamp using data loader
                            premium_data = self.data_loader.get_option_premium_at_time(
                                timestamp=timestamp,
                                strike=selected_strike,
                                option_type=option_type
                            )
                            
                            if not premium_data:
                                raise ConfigurationError(f"No premium data found for strike {selected_strike} at {timestamp}")
                            
                            # Store premium data
                            key = (timestamp, underlying_price)
                            premium_lookup[key] = {
                                'strike': selected_strike,
                                'premium': premium_data.get('premium', 0),
                                'iv': premium_data.get('iv', 0),
                                'option_type': option_type
                            }
                                    
                        except ConfigurationError:
                            # Re-raise configuration errors (no fallback)
                            raise
                        except Exception as e:
                            # Log other errors but continue processing
                            self.log_detailed(f"Error processing timestamp {timestamp}: {e}", "ERROR")
                            continue
                            
                except ConfigurationError:
                    # Re-raise configuration errors (no fallback)
                    raise
                except Exception as e:
                    # Log other errors but continue processing
                    self.log_detailed(f"Error processing date {date_key}: {e}", "ERROR")
                    continue
            
            return premium_lookup
            
        except Exception as e:
            self.log_detailed(f"Error in batch premium query: {e}", "ERROR")
            raise ConfigurationError(f"Batch premium query failed: {e}")
    
    def _calculate_vectorized_pnl(self, premium_data: List[Dict]) -> Dict[str, Any]:
        """Calculate comprehensive vectorized P&L with advanced risk metrics using premium data."""
        try:
            if not premium_data:
                return self._get_empty_comprehensive_result()
            
            # Extract premium data for vectorized calculations
            entry_premiums = np.array([trade['entry_premium'] for trade in premium_data])
            exit_premiums = np.array([trade['exit_premium'] for trade in premium_data])
            position_sizes = np.array([trade.get('position_size', 1) for trade in premium_data])
            option_types = np.array([trade.get('option_type', 'CALL') for trade in premium_data])
            
            # Configuration
            lot_size = self.main_config.get('trading', {}).get('lot_size', 75)
            commission = self.main_config.get('trading', {}).get('commission_per_trade', 20)
            initial_capital = self.main_config.get('trading', {}).get('initial_capital', 100000)
            slippage_percentage = self.main_config.get('trading', {}).get('slippage_percentage', 0.005)
            
            # Vectorized P&L calculation with option-type-specific logic
            # For options trading: We assume BUY strategy (buy low, sell high)
            # CALL: P&L = (Exit Premium - Entry Premium) * Lot Size * Position Size
            # PUT:  P&L = (Exit Premium - Entry Premium) * Lot Size * Position Size
            # Both CALL and PUT use same formula for BUY strategy
            gross_pnls = (exit_premiums - entry_premiums) * lot_size * position_sizes
            commission_array = np.full(len(premium_data), 2 * commission)  # Commission per trade, not per lot
            slippage_array = (entry_premiums + exit_premiums) * lot_size * position_sizes * slippage_percentage
            
            # Net P&L after costs
            net_pnls = gross_pnls - commission_array - slippage_array
            
            # Basic statistics
            total_trades = len(net_pnls)
            winning_trades = np.sum(net_pnls > 0)
            losing_trades = np.sum(net_pnls < 0)
            
            total_pnl = np.sum(net_pnls)
            total_return_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
            final_portfolio_value = initial_capital + total_pnl
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            
            winning_pnls = net_pnls[net_pnls > 0]
            losing_pnls = net_pnls[net_pnls < 0]
            
            avg_winning_trade = np.mean(winning_pnls) if len(winning_pnls) > 0 else 0
            avg_losing_trade = np.mean(losing_pnls) if len(losing_pnls) > 0 else 0
            
            best_trade = np.max(net_pnls) if len(net_pnls) > 0 else 0
            worst_trade = np.min(net_pnls) if len(net_pnls) > 0 else 0
            
            # COMPREHENSIVE PORTFOLIO ANALYTICS CALCULATIONS
            
            # 1. PROFIT FACTOR CALCULATION
            total_wins = float(np.sum(winning_pnls)) if len(winning_pnls) > 0 else 0.0
            total_losses = float(abs(np.sum(losing_pnls))) if len(losing_pnls) > 0 else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
            
            # 2. WIN/LOSS RATIO
            win_loss_ratio = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else float('inf') if avg_winning_trade > 0 else 0.0
            
            # 3. STREAK ANALYSIS
            trade_results = (net_pnls > 0).astype(int)  # 1 for win, 0 for loss
            
            # Calculate win streaks
            win_streaks = []
            current_win_streak = 0
            for result in trade_results:
                if result == 1:
                    current_win_streak += 1
                else:
                    if current_win_streak > 0:
                        win_streaks.append(current_win_streak)
                    current_win_streak = 0
            if current_win_streak > 0:
                win_streaks.append(current_win_streak)
            
            # Calculate loss streaks
            loss_streaks = []
            current_loss_streak = 0
            for result in trade_results:
                if result == 0:
                    current_loss_streak += 1
                else:
                    if current_loss_streak > 0:
                        loss_streaks.append(current_loss_streak)
                    current_loss_streak = 0
            if current_loss_streak > 0:
                loss_streaks.append(current_loss_streak)
            
            max_win_streak = int(max(win_streaks)) if win_streaks else 0
            max_loss_streak = int(max(loss_streaks)) if loss_streaks else 0
            
            # 4. DRAWDOWN ANALYSIS
            cumulative_pnl = np.cumsum(net_pnls)
            portfolio_value = initial_capital + cumulative_pnl
            
            # Calculate running maximum portfolio value
            running_max = np.maximum.accumulate(portfolio_value)
            
            # Calculate drawdown at each point
            drawdown_array = (portfolio_value - running_max) / running_max * 100
            
            # Find maximum drawdown
            max_drawdown_pct = float(abs(np.min(drawdown_array))) if len(drawdown_array) > 0 else 0.0
            max_drawdown_idx = np.argmin(drawdown_array) if len(drawdown_array) > 0 else 0
            max_drawdown_amount = float(abs(portfolio_value[max_drawdown_idx] - running_max[max_drawdown_idx])) if len(drawdown_array) > 0 else 0.0
            
            # 5. RISK-ADJUSTED METRICS
            if len(net_pnls) > 1:
                # Calculate trade returns for risk metrics
                trade_returns = net_pnls / initial_capital  # Returns as fraction of capital
                avg_return = np.mean(trade_returns)
                return_std = np.std(trade_returns, ddof=1)
                
                # Sharpe Ratio (assuming risk-free rate = 0 for simplicity)
                sharpe_ratio = float(avg_return / return_std) if return_std > 0 else 0.0
                
                # Sortino Ratio (downside deviation)
                negative_returns = trade_returns[trade_returns < 0]
                downside_std = np.std(negative_returns, ddof=1) if len(negative_returns) > 1 else 0.0
                sortino_ratio = float(avg_return / downside_std) if downside_std > 0 else 0.0
                
                # Recovery Factor
                recovery_factor = float(total_return_pct / max_drawdown_pct) if max_drawdown_pct > 0 else 0.0
                
                # Calmar Ratio (same as recovery factor for this timeframe)
                calmar_ratio = recovery_factor
                
            else:
                sharpe_ratio = 0.0
                sortino_ratio = 0.0
                recovery_factor = 0.0
                calmar_ratio = 0.0
            
            # 6. POSITION SIZING ANALYTICS (Strategy-based position sizing)
            position_sizes = []
            for trade in premium_data:
                # Extract position size from trade metadata if available
                position_size = trade.get('position_size', 1)
                position_sizes.append(position_size)
            
            if position_sizes:
                avg_position_size = np.mean(position_sizes)
                max_position_size = np.max(position_sizes)
            else:
                avg_position_size = 1.0
                max_position_size = 1.0
            
            # Capital efficiency calculation
            avg_capital_per_trade = np.mean(entry_premiums) * lot_size if len(entry_premiums) > 0 else 0.0
            capital_efficiency = float((avg_capital_per_trade / initial_capital) * 100) if initial_capital > 0 else 0.0
            
            return {
                # Basic metrics
                'total_pnl': float(total_pnl),
                'net_pnl': float(total_pnl),  # Alias for compatibility
                'gross_pnl': float(np.sum(gross_pnls)),
                'total_return_pct': float(total_return_pct),
                'total_trades': int(total_trades),
                'winning_trades': int(winning_trades),
                'losing_trades': int(losing_trades),
                'win_rate': float(win_rate),
                'avg_pnl_per_trade': float(total_pnl / total_trades) if total_trades > 0 else 0,
                
                # Capital metrics
                'initial_capital': float(initial_capital),
                'final_portfolio_value': float(final_portfolio_value),
                
                # Trade extremes
                'best_trade': float(best_trade),
                'worst_trade': float(worst_trade),
                'avg_winning_trade': float(avg_winning_trade),
                'avg_losing_trade': float(avg_losing_trade),
                'max_profit': float(best_trade),  # Alias for compatibility
                'max_loss': float(worst_trade),   # Alias for compatibility
                'avg_win': float(avg_winning_trade),  # Alias for compatibility
                'avg_loss': float(avg_losing_trade),  # Alias for compatibility
                
                # Risk metrics
                'max_drawdown_pct': max_drawdown_pct,
                'max_drawdown_amount': max_drawdown_amount,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'recovery_factor': recovery_factor,
                'calmar_ratio': calmar_ratio,
                
                # Trade statistics
                'profit_factor': profit_factor,
                'win_loss_ratio': win_loss_ratio,
                'max_win_streak': max_win_streak,
                'max_loss_streak': max_loss_streak,
                
                # Position sizing
                'avg_position_size': avg_position_size,
                'max_position_size': max_position_size,
                'capital_efficiency': capital_efficiency,
                'avg_capital_used': float(avg_capital_per_trade),
                
                # Trading costs breakdown
                'total_commission': float(np.sum(commission_array)),
                'total_slippage': float(np.sum(slippage_array)),
                'slippage_percentage': slippage_percentage,
                'avg_slippage_per_trade': float(np.mean(slippage_array)) if len(slippage_array) > 0 else 0.0,
                
                # System info
                'trades_data': premium_data,
                'calculation_method': 'comprehensive_vectorized',
                
                # Case-wise analytics for dynamic strategy analysis
                'case_wise_performance': self._calculate_case_wise_statistics(premium_data, net_pnls, gross_pnls, commission_array, slippage_array)
            }
            
        except Exception as e:
            self.log_detailed(f"Error in comprehensive vectorized P&L calculation: {e}", "ERROR")
            raise ConfigurationError(f"Comprehensive vectorized P&L calculation failed: {e}")
    
    def _calculate_case_wise_statistics(
        self, 
        premium_data: List[Dict], 
        net_pnls: np.ndarray, 
        gross_pnls: np.ndarray,
        commission_array: np.ndarray,
        slippage_array: np.ndarray
    ) -> Dict[str, Any]:
        """Calculate case-wise performance statistics for dynamic strategy analysis."""
        try:
            case_wise_stats = {}
            
            # Group trades by case
            case_trades = {}
            for i, trade in enumerate(premium_data):
                case_name = trade.get('case', 'unknown')
                if case_name not in case_trades:
                    case_trades[case_name] = []
                case_trades[case_name].append({
                    'index': i,
                    'trade': trade,
                    'net_pnl': net_pnls[i],
                    'gross_pnl': gross_pnls[i],
                    'commission': commission_array[i] if i < len(commission_array) else 0,
                    'slippage': slippage_array[i] if i < len(slippage_array) else 0
                })
            
            # Calculate statistics for each case
            for case_name, trades in case_trades.items():
                if not trades:
                    continue
                    
                # Extract P&L arrays for this case
                case_net_pnls = np.array([t['net_pnl'] for t in trades])
                case_gross_pnls = np.array([t['gross_pnl'] for t in trades])
                
                # Basic statistics
                total_trades = len(case_net_pnls)
                winning_trades = np.sum(case_net_pnls > 0)
                losing_trades = np.sum(case_net_pnls < 0)
                
                total_pnl = float(np.sum(case_net_pnls))
                total_gross_pnl = float(np.sum(case_gross_pnls))
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                # Trade extremes
                best_trade = float(np.max(case_net_pnls)) if len(case_net_pnls) > 0 else 0
                worst_trade = float(np.min(case_net_pnls)) if len(case_net_pnls) > 0 else 0
                avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
                
                # Win/Loss analysis
                winning_pnls = case_net_pnls[case_net_pnls > 0]
                losing_pnls = case_net_pnls[case_net_pnls < 0]
                
                avg_winning_trade = float(np.mean(winning_pnls)) if len(winning_pnls) > 0 else 0
                avg_losing_trade = float(np.mean(losing_pnls)) if len(losing_pnls) > 0 else 0
                
                # Profit factor and win/loss ratio
                total_wins = float(np.sum(winning_pnls)) if len(winning_pnls) > 0 else 0.0
                total_losses = float(abs(np.sum(losing_pnls))) if len(losing_pnls) > 0 else 0.0
                profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
                win_loss_ratio = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else float('inf') if avg_winning_trade > 0 else 0.0
                
                # Option type for this case - extract from case name for reliability
                if case_name.endswith('_call'):
                    option_type = 'CALL'
                elif case_name.endswith('_put'):
                    option_type = 'PUT'
                else:
                    # Fallback to trade data
                    option_type = trades[0]['trade'].get('option_type', 'UNKNOWN') if trades else 'UNKNOWN'
                
                # Total costs for this case
                total_commission = float(np.sum([t['commission'] for t in trades]))
                total_slippage = float(np.sum([t['slippage'] for t in trades]))
                
                case_wise_stats[case_name] = {
                    'case_name': case_name,
                    'option_type': option_type,
                    'total_trades': int(total_trades),
                    'winning_trades': int(winning_trades),
                    'losing_trades': int(losing_trades),
                    'win_rate': float(win_rate),
                    'total_pnl': total_pnl,
                    'total_gross_pnl': total_gross_pnl,
                    'avg_pnl_per_trade': float(avg_pnl_per_trade),
                    'best_trade': best_trade,
                    'worst_trade': worst_trade,
                    'avg_winning_trade': avg_winning_trade,
                    'avg_losing_trade': avg_losing_trade,
                    'profit_factor': profit_factor,
                    'win_loss_ratio': win_loss_ratio,
                    'total_commission': total_commission,
                    'total_slippage': total_slippage
                }
            
            self.log_detailed(f"Calculated case-wise statistics for {len(case_wise_stats)} cases", "INFO")
            return case_wise_stats
            
        except Exception as e:
            self.log_detailed(f"Error calculating case-wise statistics: {e}", "ERROR")
            return {}

    def _get_empty_comprehensive_result(self) -> Dict[str, Any]:
        """Return empty comprehensive P&L result structure."""
        initial_capital = self.main_config.get('trading', {}).get('initial_capital', 1000000)
        return {
            # Basic metrics
            'total_pnl': 0.0,
            'net_pnl': 0.0,
            'gross_pnl': 0.0,
            'total_return_pct': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_per_trade': 0.0,
            
            # Capital metrics
            'initial_capital': float(initial_capital),
            'final_portfolio_value': float(initial_capital),
            
            # Trade extremes
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_winning_trade': 0.0,
            'avg_losing_trade': 0.0,
            'max_profit': 0.0,
            'max_loss': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            
            # Risk metrics
            'max_drawdown_pct': 0.0,
            'max_drawdown_amount': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'recovery_factor': 0.0,
            'calmar_ratio': 0.0,
            
            # Trade statistics
            'profit_factor': 0.0,
            'win_loss_ratio': 0.0,
            'max_win_streak': 0,
            'max_loss_streak': 0,
            
            # Position sizing
            'avg_position_size': 1.0,
            'max_position_size': 1.0,
            'capital_efficiency': 0.0,
            'avg_capital_used': 0.0,
            
            # Trading costs breakdown
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'slippage_percentage': 0.001,
            'avg_slippage_per_trade': 0.0,
            
            # System info
            'trades_data': [],
            'calculation_method': 'comprehensive_vectorized'
        }
    
    def _load_special_trading_days(self) -> List[str]:
        """Load special trading days from config.json dynamically."""
        try:
            special_days_config = self.main_config.get('special_trading_days', [])
            special_days = []
            
            for day_config in special_days_config:
                if isinstance(day_config, dict) and 'date' in day_config:
                    date_str = day_config['date']
                    reason = day_config.get('reason', 'No reason provided')
                    special_days.append(date_str)
                    self.log_detailed(f"Loaded special trading day: {date_str} - {reason}", "INFO")
                elif isinstance(day_config, str):
                    # Support simple string format as well
                    special_days.append(day_config)
                    self.log_detailed(f"Loaded special trading day: {day_config}", "INFO")
            
            self.log_detailed(f"Total special trading days loaded: {len(special_days)}", "INFO")
            return special_days
            
        except Exception as e:
            self.log_detailed(f"Error loading special trading days from config: {e}", "WARNING")
            return []
    
    def _is_special_trading_day(self, date_key) -> bool:
        """Check if the given date is a special trading day (dynamic check)."""
        try:
            # Convert date_key to string format for comparison
            if hasattr(date_key, 'strftime'):
                date_str = date_key.strftime('%Y-%m-%d')
            else:
                date_str = str(date_key)
            
            # Check if this date exists in our special trading days list
            is_special = date_str in self.special_trading_days
            
            if is_special:
                self.log_detailed(f"Date {date_str} confirmed as special trading day", "INFO")
            
            return is_special
            
        except Exception as e:
            self.log_detailed(f"Error checking special trading day for {date_key}: {e}", "ERROR")
            return False
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for P&L calculation."""
        return {
            'processor': self.processor_name,
            'performance_stats': self.performance_stats,
            'options_enabled': self.options_enabled
        }


# Convenience function for external use
def create_options_pnl_calculator(config_loader, data_loader, logger) -> OptionsPnLCalculator:
    """Create and initialize Options P&L Calculator."""
    return OptionsPnLCalculator(config_loader, data_loader, logger)


if __name__ == "__main__":
    # Test Options P&L Calculator
    try:
        from core.json_config_loader import JSONConfigLoader
        from core.efficient_data_loader import EfficientDataLoader
        
        print("Testing Options P&L Calculator...")
        
        # Create components
        config_loader = JSONConfigLoader()
        data_loader = EfficientDataLoader(config_loader)
        
        # Create fallback logger
        class TestLogger:
            def log_major_component(self, msg, comp): print(f"PNL_CALCULATOR: {msg}")
            def log_detailed(self, msg, level, comp): print(f"DETAIL: {msg}")
            def log_performance(self, metrics, comp): print(f"PERFORMANCE: {metrics}")
        
        logger = TestLogger()
        
        # Create P&L calculator
        calculator = OptionsPnLCalculator(config_loader, data_loader, logger)
        
        print(f"Options P&L Calculator test completed!")
        print(f"   Performance: {calculator.get_performance_summary()}")
        
    except Exception as e:
        print(f"Options P&L Calculator test failed: {e}")
        sys.exit(1)