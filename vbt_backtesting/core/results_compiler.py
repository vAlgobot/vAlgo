"""
Results Compiler - Phase 6 of Ultimate Efficiency Engine
========================================================

Results compilation and performance display for Ultimate Efficiency Engine.
Extracts Phase 6 functionality from ultimate_efficiency_engine.py for modular architecture.

Features:
- Comprehensive results compilation from all phases
- Performance metrics aggregation and analysis
- Performance summary display with detailed timing
- Results structuring for export and analysis
- Component performance tracking

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


class ResultsCompiler(ProcessorBase):
    """
    Results compilation and performance display for Ultimate Efficiency Engine.
    
    Implements Phase 6 of the Ultimate Efficiency Engine:
    - Comprehensive results compilation from all processing phases
    - Performance metrics aggregation and analysis
    - Detailed performance summary display with timing breakdowns
    - Results structuring for export and further analysis
    - Component performance tracking and optimization insights
    """
    
    def __init__(self, config_loader, logger):
        """Initialize Results Compiler."""
        # Results compiler doesn't need data_loader, pass None
        super().__init__(config_loader, None, logger)
        
        self.log_major_component("Results Compiler initialized", "RESULTS_COMPILER")
    
    def compile_results(
        self,
        market_data: pd.DataFrame,
        all_indicators: Dict[str, pd.Series],
        trade_signals: Dict[str, Any],
        performance_results: Dict[str, Any],
        signal_candle_data: Dict[str, pd.Series] = None,
        sl_tp_data: Dict[str, pd.Series] = None
    ) -> Dict[str, Any]:
        """
        Compile comprehensive results from all processing phases.
        
        Args:
            market_data: OHLCV market data
            all_indicators: Calculated indicators dictionary
            trade_signals: Extracted trade signals and pairs
            performance_results: Options P&L calculation results
            signal_candle_data: Signal candle data from signal generation
            sl_tp_data: SL/TP levels data from signal generation
            
        Returns:
            Dictionary with comprehensive analysis results
        """
        try:
            with self.measure_execution_time("results_compilation"):
                self.log_major_component("Compiling comprehensive results", "RESULTS_COMPILATION")
                
                # Compile market data summary
                market_summary = self._compile_market_data_summary(market_data)
                
                # Merge signal candle data with all indicators for CSV export
                combined_indicators = all_indicators.copy()
                
                # Debug: Check signal candle data
                original_signal_candles = {k: type(v) for k, v in combined_indicators.items() if 'signal_candle' in k}
                if original_signal_candles:
                    print(f"🔍 DEBUG: Original signal candle indicators: {original_signal_candles}")
                
                if signal_candle_data:
                    print(f"🔍 DEBUG: Signal candle data to merge: {[(k, type(v), len(v) if hasattr(v, '__len__') else 'no len') for k, v in signal_candle_data.items()]}")
                    
                    # Debug: Check if arrays contain actual values
                    import numpy as np
                    for k, v in signal_candle_data.items():
                        if hasattr(v, '__len__') and len(v) > 0:
                            valid_count = np.sum(~np.isnan(v)) if isinstance(v, np.ndarray) else sum(pd.notna(v))
                            print(f"🔍 DEBUG: {k} has {valid_count} valid (non-NaN) values out of {len(v)}")
                    
                    combined_indicators.update(signal_candle_data)
                    self.log_detailed(f"Added {len(signal_candle_data)} signal candle indicators to results", "INFO")
                    
                    # Debug: Check after merge
                    merged_signal_candles = {k: type(v) for k, v in combined_indicators.items() if 'signal_candle' in k}
                    print(f"🔍 DEBUG: After merge signal candle indicators: {merged_signal_candles}")
                else:
                    print(f"🔍 DEBUG: No signal candle data to merge")
                
                # Merge SL/TP data with combined indicators for CSV export
                if sl_tp_data:
                    print(f"🔍 DEBUG: SL/TP data to merge: {[(k, type(v), len(v) if hasattr(v, '__len__') else 'no len') for k, v in sl_tp_data.items()]}")
                    
                    # Debug: Check if SL/TP arrays contain actual values
                    import numpy as np
                    for k, v in sl_tp_data.items():
                        if hasattr(v, '__len__') and len(v) > 0:
                            valid_count = np.sum(~np.isnan(v)) if isinstance(v, np.ndarray) else sum(pd.notna(v))
                            print(f"🔍 DEBUG: {k} has {valid_count} valid (non-NaN) values out of {len(v)}")
                    
                    combined_indicators.update(sl_tp_data)
                    self.log_detailed(f"Added {len(sl_tp_data)} SL/TP indicators to results", "INFO")
                    print(f"🎯 Added SL/TP data to combined indicators: {list(sl_tp_data.keys())}")
                else:
                    print(f"🔍 DEBUG: No SL/TP data to merge")
                
                # Compile indicators summary
                indicators_summary = self._compile_indicators_summary(combined_indicators)
                
                # Compile signals summary
                signals_summary = self._compile_signals_summary(trade_signals)
                
                # Add raw entries and exits for CSV export
                signals_summary['entries'] = trade_signals.get('entries', [])
                signals_summary['exits'] = trade_signals.get('exits', [])
                
                # Compile performance summary
                performance_summary = self._compile_performance_summary(performance_results)
                
                # Compile system performance metrics
                system_metrics = self._compile_system_metrics()
                
                # Create comprehensive results dictionary
                results = {
                    'timestamp': datetime.now().isoformat(),
                    'system_info': {
                        'version': self.main_config.get('system', {}).get('version', '2.0'),
                        'architecture': self.main_config.get('system', {}).get('architecture', 'ultimate_efficiency'),
                        'environment': self.main_config.get('system', {}).get('environment', 'production')
                    },
                    'market_data': market_summary,
                    'raw_market_data': market_data,  # Add actual DataFrame for CSV export
                    'indicators': indicators_summary,
                    'raw_indicators': combined_indicators,  # Add actual indicators for CSV export (including signal candle data)
                    'signals': signals_summary,
                    'performance': performance_summary,
                    'system_metrics': system_metrics,
                    'configuration': {
                        'trading': self.main_config.get('trading', {}),
                        'options': self.main_config.get('options', {}),
                        'backtesting': self.main_config.get('backtesting', {})
                    }
                }
                
                self.log_major_component(
                    f"Results compilation complete: {len(results)} main sections compiled",
                    "RESULTS_COMPILATION"
                )
                
                return results
                
        except Exception as e:
            self.log_detailed(f"Error compiling results: {e}", "ERROR")
            raise ConfigurationError(f"Results compilation failed: {e}")
    
    def _compile_market_data_summary(self, market_data: pd.DataFrame) -> Dict[str, Any]:
        """Compile market data summary statistics."""
        try:
            if market_data.empty:
                return {'error': 'No market data available'}
            
            return {
                'total_candles': len(market_data),
                'date_range': {
                    'start': market_data.index[0].isoformat(),
                    'end': market_data.index[-1].isoformat()
                },
                'price_summary': {
                    'open_range': {
                        'min': float(market_data['open'].min()),
                        'max': float(market_data['open'].max()),
                        'avg': float(market_data['open'].mean())
                    },
                    'close_range': {
                        'min': float(market_data['close'].min()),
                        'max': float(market_data['close'].max()),
                        'avg': float(market_data['close'].mean())
                    },
                    'volume_stats': {
                        'total': int(market_data['volume'].sum()),
                        'avg_per_candle': float(market_data['volume'].mean()),
                        'max_volume': int(market_data['volume'].max())
                    }
                },
                'volatility': {
                    'daily_returns_std': float(market_data['close'].pct_change().std()),
                    'price_range_avg': float((market_data['high'] - market_data['low']).mean())
                }
            }
            
        except Exception as e:
            self.log_detailed(f"Error compiling market data summary: {e}", "ERROR")
            return {'error': str(e)}
    
    def _compile_indicators_summary(self, all_indicators: Dict[str, pd.Series]) -> Dict[str, Any]:
        """Compile indicators summary and statistics."""
        try:
            if not all_indicators:
                return {'error': 'No indicators available'}
            
            calculated_indicators = {}
            market_data_keys = ['close', 'open', 'high', 'low', 'volume']
            
            # Separate calculated indicators from market data
            for key, series in all_indicators.items():
                if key not in market_data_keys:
                    if isinstance(series, pd.Series) and not series.empty:
                        calculated_indicators[key] = {
                            'total_values': len(series),
                            'valid_values': series.notna().sum(),
                            'completion_rate': float(series.notna().sum() / len(series)),
                            'value_range': {
                                'min': float(series.min()) if series.notna().any() else None,
                                'max': float(series.max()) if series.notna().any() else None,
                                'mean': float(series.mean()) if series.notna().any() else None
                            }
                        }
            
            return {
                'total_indicators': len(all_indicators),
                'calculated_indicators': calculated_indicators,
                'market_data_fields': len([k for k in all_indicators.keys() if k in market_data_keys]),
                'calculation_summary': {
                    'total_calculated': len(calculated_indicators),
                    'fully_calculated': len([k for k, v in calculated_indicators.items() if v.get('completion_rate', 0) > 0.8]),
                    'partially_calculated': len([k for k, v in calculated_indicators.items() if 0.2 < v.get('completion_rate', 0) <= 0.8])
                }
            }
            
        except Exception as e:
            self.log_detailed(f"Error compiling indicators summary: {e}", "ERROR")
            return {'error': str(e)}
    
    def _compile_signals_summary(self, trade_signals: Dict[str, Any]) -> Dict[str, Any]:
        """Compile signals and trade pairs summary."""
        try:
            if not trade_signals:
                return {'error': 'No trade signals available'}
            
            entries = trade_signals.get('entries', [])
            exits = trade_signals.get('exits', [])
            trade_pairs = trade_signals.get('trade_pairs', [])
            strategy_performance = trade_signals.get('strategy_performance', {})
            
            # Analyze trade pairs
            trade_analysis = {}
            if trade_pairs:
                durations = [trade.get('duration_candles', 0) for trade in trade_pairs]
                gross_pnls = [trade.get('gross_pnl', 0) for trade in trade_pairs]
                
                trade_analysis = {
                    'total_trades': len(trade_pairs),
                    'duration_stats': {
                        'avg_duration_candles': float(np.mean(durations)) if durations else 0,
                        'max_duration': int(max(durations)) if durations else 0,
                        'min_duration': int(min(durations)) if durations else 0
                    },
                    'gross_pnl_stats': {
                        'total_gross_pnl': float(sum(gross_pnls)) if gross_pnls else 0,
                        'avg_gross_pnl': float(np.mean(gross_pnls)) if gross_pnls else 0,
                        'winning_trades': len([pnl for pnl in gross_pnls if pnl > 0]),
                        'losing_trades': len([pnl for pnl in gross_pnls if pnl < 0])
                    }
                }
            
            return {
                'signal_counts': {
                    'total_entries': len(entries),
                    'total_exits': len(exits),
                    'entry_exit_ratio': len(exits) / len(entries) if entries else 0
                },
                'trade_pairs': trade_analysis,
                'strategy_performance': strategy_performance,
                'signal_distribution': self._analyze_signal_distribution(entries, exits)
            }
            
        except Exception as e:
            self.log_detailed(f"Error compiling signals summary: {e}", "ERROR")
            return {'error': str(e)}
    
    def _analyze_signal_distribution(self, entries: List[Dict], exits: List[Dict]) -> Dict[str, Any]:
        """Analyze distribution of signals across strategies and cases."""
        try:
            entry_distribution = {}
            exit_distribution = {}
            
            # Analyze entries by group and case
            for entry in entries:
                group = entry.get('group', 'unknown')
                case = entry.get('case', 'unknown')
                key = f"{group}.{case}"
                
                entry_distribution[key] = entry_distribution.get(key, 0) + 1
            
            # Analyze exits by group and case
            for exit in exits:
                group = exit.get('group', 'unknown')
                case = exit.get('case', 'unknown')
                key = f"{group}.{case}"
                
                exit_distribution[key] = exit_distribution.get(key, 0) + 1
            
            return {
                'entries_by_strategy_case': entry_distribution,
                'exits_by_strategy_case': exit_distribution,
                'most_active_strategy': max(entry_distribution.items(), key=lambda x: x[1]) if entry_distribution else None
            }
            
        except Exception as e:
            self.log_detailed(f"Error analyzing signal distribution: {e}", "ERROR")
            return {'error': str(e)}
    
    def _compile_performance_summary(self, performance_results: Dict[str, Any]) -> Dict[str, Any]:
        """Compile performance results summary."""
        try:
            if not performance_results:
                return {'error': 'No performance results available'}
            
            # Extract key performance metrics and include ALL data for CSV export
            performance_summary = {
                'options_pnl': performance_results,  # Include ALL performance data including trades_data
                'execution_metrics': {
                    'total_calculation_time': performance_results.get('total_calculation_time', 0),
                    'trades_processed': performance_results.get('trades_processed', 0),
                    'entry_signals': performance_results.get('entry_signals', 0),
                    'exit_signals': performance_results.get('exit_signals', 0)
                }
            }
            
            # Calculate derived metrics
            if performance_results.get('total_trades', 0) > 0:
                avg_win = performance_results.get('avg_win', 0)
                avg_loss = performance_results.get('avg_loss', 0)
                performance_summary['derived_metrics'] = {
                    'profit_factor': abs(avg_win / avg_loss) if avg_loss < 0 else 0,
                    'risk_reward_ratio': abs(avg_win / avg_loss) if avg_loss < 0 else 0,
                    'trades_per_second': performance_summary['execution_metrics']['trades_processed'] / performance_summary['execution_metrics']['total_calculation_time'] if performance_summary['execution_metrics']['total_calculation_time'] > 0 else 0
                }
            
            return performance_summary
            
        except Exception as e:
            self.log_detailed(f"Error compiling performance summary: {e}", "ERROR")
            return {'error': str(e)}
    
    def _compile_system_metrics(self) -> Dict[str, Any]:
        """Compile system performance metrics."""
        try:
            return {
                'processor_performance': self.performance_stats,
                'architecture_info': {
                    'processor_name': self.processor_name,
                    'modular_architecture': True,
                    'performance_target': self.main_config.get('performance', {}).get('target_processing_speed', 875000)
                }
            }
            
        except Exception as e:
            self.log_detailed(f"Error compiling system metrics: {e}", "ERROR")
            return {'error': str(e)}
    
    def display_performance_summary(self, performance_stats: Dict[str, Any] = None) -> None:
        """Display comprehensive performance summary."""
        try:
            stats = performance_stats or self.performance_stats
            
            if not stats:
                print("Warning: No performance statistics available for display")
                return
            
            print("\nULTIMATE EFFICIENCY PERFORMANCE SUMMARY")
            print("=" * 60)
            
            # Display timing information
            if 'results_compilation_time' in stats:
                print(f"Results Compilation: {stats['results_compilation_time']:.4f}s")
            
            # Display component timings if available
            timing_components = ['data_loading_time', 'indicator_calculation_time', 
                               'signal_generation_time', 'signal_extraction_time', 
                               'pnl_calculation_time']
            
            total_component_time = 0
            for component in timing_components:
                if component in stats:
                    component_name = component.replace('_', ' ').title().replace('Pnl', 'P&L')
                    print(f"{component_name}: {stats[component]:.4f}s")
                    total_component_time += stats[component]
            
            if total_component_time > 0:
                print(f"Total Processing Time: {total_component_time:.4f}s")
            
            # Calculate and display performance multiplier if possible
            target_time = 1.0  # Assume 1 second baseline
            if total_component_time > 0:
                performance_multiplier = target_time / total_component_time
                print(f"Performance Multiplier: {performance_multiplier:.1f}x")
                
                if performance_multiplier >= 5:
                    print("TARGET ACHIEVED: 5-10x performance improvement")
                else:
                    print("TARGET IN PROGRESS: Working towards 5-10x improvement")
            
            print("=" * 60)
            
            # Log to system
            self.log_performance(stats, "PERFORMANCE_SUMMARY")
            
        except Exception as e:
            self.log_detailed(f"Error displaying performance summary: {e}", "ERROR")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for results compilation."""
        return {
            'processor': self.processor_name,
            'performance_stats': self.performance_stats
        }


# Convenience function for external use
def create_results_compiler(config_loader, logger) -> ResultsCompiler:
    """Create and initialize Results Compiler."""
    return ResultsCompiler(config_loader, logger)


if __name__ == "__main__":
    # Test Results Compiler
    try:
        from core.json_config_loader import JSONConfigLoader
        
        print("Testing Results Compiler...")
        
        # Create components
        config_loader = JSONConfigLoader()
        
        # Create fallback logger
        class TestLogger:
            def log_major_component(self, msg, comp): print(f"RESULTS_COMPILER: {msg}")
            def log_detailed(self, msg, level, comp): print(f"DETAIL: {msg}")
            def log_performance(self, metrics, comp): print(f"PERFORMANCE: {metrics}")
        
        logger = TestLogger()
        
        # Create results compiler
        compiler = ResultsCompiler(config_loader, logger)
        
        # Test performance display
        test_stats = {
            'results_compilation_time': 0.0123,
            'data_loading_time': 0.0456,
            'indicator_calculation_time': 0.0789,
            'signal_generation_time': 0.0234,
            'signal_extraction_time': 0.0567,
            'pnl_calculation_time': 0.0891
        }
        
        compiler.display_performance_summary(test_stats)
        
        print(f"\nResults Compiler test completed!")
        print(f"   Performance: {compiler.get_performance_summary()}")
        
    except Exception as e:
        print(f"Results Compiler test failed: {e}")
        sys.exit(1)