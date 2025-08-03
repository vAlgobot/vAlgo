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
import warnings

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
    raise ConfigurationError("VectorBT not available. Install with: pip install vectorbt[full]")


def create_time_masks(timestamps: pd.Index) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create time masks for trading hours validation and forced exits.
    
    Args:
        timestamps: DataFrame index with datetime timestamps
        
    Returns:
        Tuple of (trading_hours_mask, forced_exit_mask)
    """
    try:
        # Define trading hours (IST)
        trading_start = dt_time(9, 20)  # 09:20 AM
        trading_end = dt_time(15, 0)    # 03:00 PM
        forced_exit_time = dt_time(15, 15)  # 03:15 PM
        
        # Extract time from timestamps
        times = pd.to_datetime(timestamps).time
        
        # Trading hours mask (09:20 - 15:00)
        trading_hours_mask = (times >= trading_start) & (times <= trading_end)
        
        # Forced exit mask (15:15)
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
        
        # Filter entries by trading hours (09:20-15:00)
        valid_entries = entry_conditions & trading_hours_mask
        
        # Combine natural exits with forced exits at 15:15
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
        print(f"❌ Error in position tracking: {e}")
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
        
        self.log_major_component("VectorBT Signal Generator initialized", "SIGNAL_GENERATOR")
    
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
                data_length = len(market_data)
                vectorbt_signals = {}
                
                # Get enabled strategies for processing
                enabled_strategies = self.config_loader.get_enabled_strategies_cases()
                
                self.log_major_component(
                    f"Processing {data_length} data points for {len(enabled_strategies)} strategy groups",
                    "SIGNAL_GENERATION"
                )
                
                print(f"🎯 Using VectorBT IndicatorFactory Pattern (Ultra Fast)")
                print(f"   🔄 Processing: {data_length} data points for {len(enabled_strategies)} strategy groups...")
                
                # Process each strategy group using VectorBT IndicatorFactory
                for group_name, group_config in enabled_strategies.items():
                    # Process each case in the strategy group
                    for case_name, case_config in group_config['cases'].items():
                        if not case_config.get('enabled', True):
                            continue
                        
                        print(f"   🚀 Processing {group_name}.{case_name} with VectorBT IndicatorFactory...")
                        
                        # Generate signals using direct VectorBT IndicatorFactory pattern
                        signal_results = self._generate_signals_direct(
                            market_data=market_data,
                            indicators=indicators,
                            case_config=case_config,
                            group_name=group_name,
                            case_name=case_name
                        )
                        
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
                        
                        # Log case results
                        print(f"     ✅ Generated {signal_results['entry_count']} entries, {signal_results['exit_count']} exits in {signal_results['generation_time']:.3f}s")
                        print(f"     🎯 Strike: {signal_results.get('strike_type', 'ATM')}, Option: {signal_results.get('option_type', 'CALL')}")
                
                self.log_major_component(
                    f"VectorBT signal generation complete: {len(vectorbt_signals)} strategy signals generated",
                    "SIGNAL_GENERATION"
                )
                
                print(f"✅ VectorBT signal generation complete:")
                print(f"   📊 Strategy signals generated: {len(vectorbt_signals)}")
                print(f"   ⚡ Using proven IndicatorFactory pattern")
                
                return vectorbt_signals
                
        except Exception as e:
            self.log_detailed(f"Error in VectorBT signal generation: {e}", "ERROR")
            raise ConfigurationError(f"VectorBT signal generation failed: {e}")
    
    def _generate_signals_direct(
        self,
        market_data: pd.DataFrame,
        indicators: Dict[str, pd.Series],
        case_config: Dict[str, Any],
        group_name: str,
        case_name: str
    ) -> Dict[str, Any]:
        """
        Direct signal generation using condition evaluation.
        
        Args:
            market_data: OHLCV DataFrame
            indicators: Dictionary of calculated indicators
            case_config: Case-specific configuration
            group_name: Strategy group name
            case_name: Case name
            
        Returns:
            Dictionary with signal generation results
        """
        try:
            start_time = time.time()
            
            # Get entry and exit conditions
            entry_condition = case_config.get('entry', '')
            exit_condition = case_config.get('exit', '')
            
            if not entry_condition or not exit_condition:
                raise ConfigurationError(f"Entry/exit conditions not defined for {group_name}.{case_name}")
            
            # Create evaluation context with market data and indicators
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
            
            # Evaluate entry signals
            try:
                entry_signals = eval(entry_condition, {"__builtins__": {}}, eval_context)
                if isinstance(entry_signals, pd.Series):
                    entry_signals = entry_signals.fillna(False).astype(bool)
                else:
                    entry_signals = pd.Series(entry_signals, index=market_data.index).fillna(False).astype(bool)
            except Exception as e:
                self.log_detailed(f"Entry condition evaluation failed for {group_name}.{case_name}: {e}", "ERROR")
                entry_signals = pd.Series(False, index=market_data.index)
            
            # Evaluate exit signals
            try:
                exit_signals = eval(exit_condition, {"__builtins__": {}}, eval_context)
                if isinstance(exit_signals, pd.Series):
                    exit_signals = exit_signals.fillna(False).astype(bool)
                else:
                    exit_signals = pd.Series(exit_signals, index=market_data.index).fillna(False).astype(bool)
            except Exception as e:
                self.log_detailed(f"Exit condition evaluation failed for {group_name}.{case_name}: {e}", "ERROR")
                exit_signals = pd.Series(False, index=market_data.index)
            
            # APPLY VECTORIZED POSITION TRACKING (Key improvement from working system)
            trading_hours_mask, forced_exit_mask = create_time_masks(market_data.index)
            
            # Generate proper signals using vectorized position tracking
            position_signals = vectorized_position_tracking(
                entry_signals.values,
                exit_signals.values, 
                trading_hours_mask,
                forced_exit_mask
            )
            
            # Convert position signals back to entry/exit boolean arrays
            final_entry_signals = pd.Series(position_signals == 1, index=market_data.index)
            final_exit_signals = pd.Series(position_signals == -1, index=market_data.index)
            
            # Count final signals (after position tracking)
            entry_count = final_entry_signals.sum()
            exit_count = final_exit_signals.sum()
            
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
            
            return {
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
                'group_name': group_name,
                'case_name': case_name
            }
            
        except Exception as e:
            self.log_detailed(f"Error in direct signal generation for {group_name}.{case_name}: {e}", "ERROR")
            
            # No fallback signals - fail fast with proper error
            raise ConfigurationError(f"Signal generation failed for {group_name}.{case_name}: {e}")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for signal generation."""
        return {
            'processor': self.processor_name,
            'performance_stats': self.performance_stats,
            'vectorbt_available': VECTORBT_AVAILABLE
        }


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
            def log_detailed(self, msg, level, comp): print(f"🔍 {comp}: {msg}")
            def log_performance(self, metrics, comp): print(f"📊 {comp}: {metrics}")
        
        logger = TestLogger()
        
        # Create signal generator
        generator = VectorBTSignalGenerator(config_loader, data_loader, logger)
        
        print(f"✅ VectorBT Signal Generator test completed!")
        print(f"   📊 Performance: {generator.get_performance_summary()}")
        
    except Exception as e:
        print(f"❌ VectorBT Signal Generator test failed: {e}")
        sys.exit(1)