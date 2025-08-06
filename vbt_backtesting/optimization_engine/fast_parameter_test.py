#!/usr/bin/env python3
"""
Fast Parameter Optimization Test - Generate Best Windows CSV Report
===================================================================

Focused parameter optimization test that identifies best RSI, EMA, SMA windows
for 5-minute NIFTY candles and generates comprehensive CSV report.

Features:
- Reduced parameter combinations for faster execution
- Direct database access to real NIFTY data
- Best windows identification and CSV export
- End-to-end validation

Author: vAlgo Development Team
Created: August 5, 2025
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.json_config_loader import JSONConfigLoader
from core.efficient_data_loader import EfficientDataLoader

# VectorBT import for fast indicator calculations
try:
    import vectorbt as vbt
    print("✅ VectorBT loaded for fast parameter optimization")
except ImportError:
    print("❌ VectorBT not available. Install with: pip install vectorbt[full]")
    sys.exit(1)


class FastParameterOptimizer:
    """
    Fast parameter optimizer for identifying best indicator windows.
    Focuses on core indicators: RSI, EMA, SMA with reduced parameter space.
    """
    
    def __init__(self):
        """Initialize fast parameter optimizer."""
        print("🔧 Initializing Fast Parameter Optimizer...")
        
        # Load configuration
        config_dir = str(Path(__file__).parent / "config")
        self.config_loader = JSONConfigLoader(config_dir)
        self.main_config = self.config_loader.main_config
        self.strategies_config = self.config_loader.strategies_config
        
        # Focused parameter ranges for faster testing
        self.test_parameters = {
            'rsi': [10, 12, 14, 16, 18, 20],  # 6 values instead of 11
            'ema_fast': [7, 9, 11, 13, 15],  # 5 values
            'ema_slow': [18, 20, 22, 25, 28], # 5 values  
            'sma_fast': [7, 9, 11, 13, 15],  # 5 values
            'sma_medium': [18, 20, 22, 25, 28], # 5 values
            'sma_slow': [45, 50, 55, 60, 65], # 5 values
            'sma_trend': [175, 200, 225] # 3 values
        }
        
        print(f"   📊 Parameter combinations: {self._calculate_total_combinations():,}")
        print("✅ Fast Parameter Optimizer ready")
    
    def _calculate_total_combinations(self):
        """Calculate total parameter combinations."""
        return (len(self.test_parameters['rsi']) * 
                len(self.test_parameters['ema_fast']) * 
                len(self.test_parameters['ema_slow']) *
                len(self.test_parameters['sma_fast']) *
                len(self.test_parameters['sma_medium']) *
                len(self.test_parameters['sma_slow']) *
                len(self.test_parameters['sma_trend']))
    
    def optimize_parameters(self, market_data, symbol="NIFTY", timeframe="5m"):
        """
        Run focused parameter optimization on real market data.
        
        Args:
            market_data: OHLCV data from database
            symbol: Trading symbol  
            timeframe: Data timeframe
            
        Returns:
            Dictionary with optimization results and best parameters
        """
        try:
            print(f"\n🚀 Starting Fast Parameter Optimization for {symbol} {timeframe}")
            start_time = time.time()
            
            close_prices = market_data['close']
            print(f"   📊 Data points: {len(close_prices):,}")
            print(f"   📅 Date range: {market_data.index[0]} to {market_data.index[-1]}")
            
            # Phase 1: Calculate all indicator combinations using VectorBT
            print("⚡ Phase 1: Calculating indicators with VectorBT...")
            indicator_start = time.time()
            
            # Calculate RSI for all periods simultaneously
            rsi_periods = self.test_parameters['rsi']
            rsi_data = vbt.RSI.run(close_prices, window=rsi_periods)
            
            # Calculate EMA for all periods simultaneously  
            all_ema_periods = sorted(set(self.test_parameters['ema_fast'] + self.test_parameters['ema_slow']))
            ema_data = vbt.EMA.run(close_prices, window=all_ema_periods)
            
            # Calculate SMA for all periods simultaneously
            all_sma_periods = sorted(set(
                self.test_parameters['sma_fast'] + self.test_parameters['sma_medium'] + 
                self.test_parameters['sma_slow'] + self.test_parameters['sma_trend']
            ))
            sma_data = vbt.SMA.run(close_prices, window=all_sma_periods)
            
            indicator_time = time.time() - indicator_start
            print(f"   ✅ All indicators calculated in {indicator_time:.2f}s")
            
            # Phase 2: Test strategy conditions for parameter combinations
            print("🎯 Phase 2: Testing strategy conditions...")
            test_start = time.time()
            
            results = []
            active_strategies = self._get_active_strategies()
            total_combinations = self._calculate_total_combinations()
            
            tested_combinations = 0
            for strategy_group, strategy_cases in active_strategies.items():
                for case_name, case_config in strategy_cases.items():
                    print(f"   📈 Testing strategy: {strategy_group}.{case_name}")
                    
                    case_results = self._test_strategy_parameters(
                        case_name, case_config, market_data,
                        rsi_data, ema_data, sma_data,
                        rsi_periods, all_ema_periods, all_sma_periods
                    )
                    
                    results.extend(case_results)
                    tested_combinations += len(case_results)
            
            test_time = time.time() - test_start
            print(f"   ✅ Tested {tested_combinations:,} combinations in {test_time:.2f}s")
            
            # Phase 3: Rank and compile results
            print("🏆 Phase 3: Ranking results...")
            ranking_start = time.time()
            
            # Sort by total return percentage
            results.sort(key=lambda x: x['total_return_pct'], reverse=True)
            
            # Get best results
            top_results = results[:20]  # Top 20
            
            # Find best parameters by strategy
            best_by_strategy = {}
            for result in results:
                strategy_key = f"{result['strategy_group']}.{result['strategy_case']}"
                if strategy_key not in best_by_strategy:
                    best_by_strategy[strategy_key] = result
                elif result['total_return_pct'] > best_by_strategy[strategy_key]['total_return_pct']:
                    best_by_strategy[strategy_key] = result
            
            ranking_time = time.time() - ranking_start
            total_time = time.time() - start_time
            
            # Compile final results
            optimization_results = {
                'summary': {
                    'symbol': symbol,
                    'timeframe': timeframe,
                    'total_combinations_tested': tested_combinations,
                    'optimization_time_seconds': total_time,
                    'combinations_per_second': tested_combinations / total_time,
                    'data_points': len(close_prices),
                    'date_range': f"{market_data.index[0]} to {market_data.index[-1]}"
                },
                'top_results': top_results,
                'best_by_strategy': best_by_strategy,
                'all_results': results,
                'timing': {
                    'indicator_calculation': indicator_time,
                    'strategy_testing': test_time,
                    'ranking': ranking_time,
                    'total': total_time
                }
            }
            
            print(f"   ✅ Results compiled in {ranking_time:.2f}s")
            print(f"🎉 Total optimization time: {total_time:.2f} seconds")
            
            return optimization_results
            
        except Exception as e:
            print(f"❌ Fast parameter optimization failed: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _get_active_strategies(self):
        """Get active strategies from configuration."""
        active_strategies = {}
        
        for group_name, group_config in self.strategies_config.items():
            if group_config.get('status', 'InActive') == 'Active':
                active_cases = {}
                for case_name, case_config in group_config.get('cases', {}).items():
                    if case_config.get('status', 'InActive') == 'Active':
                        active_cases[case_name] = case_config
                
                if active_cases:
                    active_strategies[group_name] = active_cases
        
        return active_strategies
    
    def _test_strategy_parameters(self, case_name, case_config, market_data,
                                 rsi_data, ema_data, sma_data,
                                 rsi_periods, ema_periods, sma_periods):
        """Test all parameter combinations for a single strategy case."""
        results = []
        
        entry_condition = case_config['entry']
        exit_condition = case_config['exit']
        option_type = case_config['OptionType']
        
        # Test subset of combinations for speed
        sample_size = min(100, self._calculate_total_combinations())  # Limit to 100 combinations per strategy
        
        tested = 0
        for rsi_period in self.test_parameters['rsi']:
            if tested >= sample_size:
                break
                
            for ema_fast in self.test_parameters['ema_fast']:
                if tested >= sample_size:
                    break
                    
                for ema_slow in self.test_parameters['ema_slow']:
                    if tested >= sample_size:
                        break
                        
                    if ema_fast >= ema_slow:  # Skip invalid combinations
                        continue
                    
                    for sma_fast in self.test_parameters['sma_fast']:
                        if tested >= sample_size:
                            break
                            
                        for sma_medium in self.test_parameters['sma_medium']:
                            if tested >= sample_size:
                                break
                                
                            if sma_fast >= sma_medium:  # Skip invalid combinations
                                continue
                            
                            for sma_slow in self.test_parameters['sma_slow']:
                                if tested >= sample_size:
                                    break
                                    
                                if sma_medium >= sma_slow:  # Skip invalid combinations
                                    continue
                                
                                for sma_trend in self.test_parameters['sma_trend']:
                                    if tested >= sample_size:
                                        break
                                        
                                    if sma_slow >= sma_trend:  # Skip invalid combinations
                                        continue
                                    
                                    # Create parameter combination
                                    params = {
                                        'rsi_14': rsi_period,
                                        'ema_9': ema_fast,
                                        'ema_20': ema_slow,
                                        'sma_9': sma_fast,
                                        'sma_20': sma_medium,
                                        'sma_50': sma_slow,
                                        'sma_200': sma_trend
                                    }
                                    
                                    # Test this parameter combination
                                    performance = self._test_single_combination(
                                        params, entry_condition, exit_condition, option_type,
                                        market_data, rsi_data, ema_data, sma_data,
                                        rsi_periods, ema_periods, sma_periods
                                    )
                                    
                                    if performance:
                                        result = {
                                            'strategy_group': self._get_strategy_group_for_case(case_name),
                                            'strategy_case': case_name,
                                            'option_type': option_type,
                                            'parameters': params,
                                            **performance
                                        }
                                        results.append(result)
                                    
                                    tested += 1
        
        print(f"     ✅ Tested {tested} parameter combinations for {case_name}")
        return results
    
    def _test_single_combination(self, params, entry_condition, exit_condition, option_type,
                                market_data, rsi_data, ema_data, sma_data,
                                rsi_periods, ema_periods, sma_periods):
        """Test a single parameter combination."""
        try:
            # Extract indicators for this parameter combination
            indicators = self._extract_indicators(
                params, market_data, rsi_data, ema_data, sma_data,
                rsi_periods, ema_periods, sma_periods
            )
            
            # Evaluate conditions
            namespace = indicators.copy()
            
            try:
                entries = eval(entry_condition, {"__builtins__": {}}, namespace)
                exits = eval(exit_condition, {"__builtins__": {}}, namespace)
                
                if not isinstance(entries, pd.Series):
                    entries = pd.Series(entries, index=market_data.index)
                if not isinstance(exits, pd.Series):
                    exits = pd.Series(exits, index=market_data.index)
                    
            except Exception:
                return None
            
            # Calculate performance
            return self._calculate_performance(entries, exits, market_data, option_type)
            
        except Exception:
            return None
    
    def _extract_indicators(self, params, market_data, rsi_data, ema_data, sma_data,
                           rsi_periods, ema_periods, sma_periods):
        """Extract indicators for specific parameter combination."""
        indicators = {
            'close': market_data['close'],
            'high': market_data['high'],
            'low': market_data['low'],
            'volume': market_data['volume']
        }
        
        # Extract RSI
        if params['rsi_14'] in rsi_periods:
            rsi_idx = rsi_periods.index(params['rsi_14'])
            indicators['rsi_14'] = rsi_data.rsi.iloc[:, rsi_idx]
        
        # Extract EMAs
        if params['ema_9'] in ema_periods:
            ema_idx = ema_periods.index(params['ema_9'])
            indicators['ema_9'] = ema_data.ema.iloc[:, ema_idx]
            
        if params['ema_20'] in ema_periods:
            ema_idx = ema_periods.index(params['ema_20'])
            indicators['ema_20'] = ema_data.ema.iloc[:, ema_idx]
        
        # Extract SMAs
        sma_mapping = ['sma_9', 'sma_20', 'sma_50', 'sma_200']
        for sma_key in sma_mapping:
            if params[sma_key] in sma_periods:
                sma_idx = sma_periods.index(params[sma_key])
                indicators[sma_key] = sma_data.sma.iloc[:, sma_idx]
        
        return indicators
    
    def _calculate_performance(self, entries, exits, market_data, option_type):
        """Calculate performance metrics."""
        total_entries = entries.sum()
        total_exits = exits.sum()
        total_trades = min(total_entries, total_exits)
        
        if total_trades == 0:
            return {
                'total_trades': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_return_per_trade': 0
            }
        
        # Simple performance calculation
        close_prices = market_data['close']
        entry_signals = entries[entries].index[:total_trades]
        exit_signals = exits[exits].index[:total_trades]
        
        if len(entry_signals) == 0 or len(exit_signals) == 0:
            return {
                'total_trades': 0,
                'total_return_pct': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'avg_return_per_trade': 0
            }
        
        entry_prices = close_prices[entry_signals]
        exit_prices = close_prices[exit_signals]
        
        if option_type == "CALL":
            returns = (exit_prices.values - entry_prices.values) / entry_prices.values
        else:  # PUT
            returns = (entry_prices.values - exit_prices.values) / entry_prices.values
        
        total_return_pct = np.sum(returns) * 100
        win_rate = (returns > 0).mean() * 100
        avg_return_per_trade = np.mean(returns) * 100
        
        positive_returns = returns[returns > 0]
        negative_returns = returns[returns < 0]
        
        profit_factor = (np.sum(positive_returns) / abs(np.sum(negative_returns))) if len(negative_returns) > 0 else 999
        
        return {
            'total_trades': total_trades,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_return_per_trade': avg_return_per_trade
        }
    
    def _get_strategy_group_for_case(self, case_name):
        """Find strategy group for case name."""
        for group_name, group_config in self.strategies_config.items():
            if case_name in group_config.get('cases', {}):
                return group_name
        return "Unknown"
    
    def export_best_windows_csv(self, optimization_results, filename=None):
        """Export best windows to CSV report."""
        try:
            if not filename:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"best_windows_NIFTY_5m_{timestamp}.csv"
            
            # Prepare data for CSV
            csv_data = []
            
            # Add summary row
            summary = optimization_results['summary']
            csv_data.append({
                'Type': 'SUMMARY',
                'Strategy_Group': 'ALL',
                'Strategy_Case': 'OPTIMIZATION_SUMMARY',
                'Total_Combinations_Tested': summary['total_combinations_tested'],
                'Optimization_Time_Seconds': summary['optimization_time_seconds'],
                'Combinations_Per_Second': f"{summary['combinations_per_second']:.0f}",
                'Data_Points': summary['data_points'],
                'Date_Range': summary['date_range'],
                'RSI_Period': 'N/A',
                'EMA_Fast': 'N/A',
                'EMA_Slow': 'N/A',
                'SMA_Fast': 'N/A',
                'SMA_Medium': 'N/A',
                'SMA_Slow': 'N/A',
                'SMA_Trend': 'N/A',
                'Total_Return_Pct': 'N/A',
                'Win_Rate': 'N/A',
                'Total_Trades': 'N/A',
                'Profit_Factor': 'N/A',
                'Avg_Return_Per_Trade': 'N/A'
            })
            
            # Add best results by strategy
            for strategy_key, best_result in optimization_results['best_by_strategy'].items():
                params = best_result['parameters']
                csv_data.append({
                    'Type': 'BEST_BY_STRATEGY',
                    'Strategy_Group': best_result['strategy_group'],
                    'Strategy_Case': best_result['strategy_case'],
                    'Total_Combinations_Tested': '',
                    'Optimization_Time_Seconds': '',
                    'Combinations_Per_Second': '',
                    'Data_Points': '',
                    'Date_Range': '',
                    'RSI_Period': params['rsi_14'],
                    'EMA_Fast': params['ema_9'],
                    'EMA_Slow': params['ema_20'],
                    'SMA_Fast': params['sma_9'],
                    'SMA_Medium': params['sma_20'],
                    'SMA_Slow': params['sma_50'],
                    'SMA_Trend': params['sma_200'],
                    'Total_Return_Pct': f"{best_result['total_return_pct']:.2f}",
                    'Win_Rate': f"{best_result['win_rate']:.1f}",
                    'Total_Trades': best_result['total_trades'],
                    'Profit_Factor': f"{best_result['profit_factor']:.2f}",
                    'Avg_Return_Per_Trade': f"{best_result['avg_return_per_trade']:.2f}"
                })
            
            # Add top 10 overall results
            for i, result in enumerate(optimization_results['top_results'][:10], 1):
                params = result['parameters']
                csv_data.append({
                    'Type': f'TOP_{i}',
                    'Strategy_Group': result['strategy_group'],
                    'Strategy_Case': result['strategy_case'],
                    'Total_Combinations_Tested': '',
                    'Optimization_Time_Seconds': '',
                    'Combinations_Per_Second': '',
                    'Data_Points': '',
                    'Date_Range': '',
                    'RSI_Period': params['rsi_14'],
                    'EMA_Fast': params['ema_9'],
                    'EMA_Slow': params['ema_20'],
                    'SMA_Fast': params['sma_9'],
                    'SMA_Medium': params['sma_20'],
                    'SMA_Slow': params['sma_50'],
                    'SMA_Trend': params['sma_200'],
                    'Total_Return_Pct': f"{result['total_return_pct']:.2f}",
                    'Win_Rate': f"{result['win_rate']:.1f}",
                    'Total_Trades': result['total_trades'],
                    'Profit_Factor': f"{result['profit_factor']:.2f}",
                    'Avg_Return_Per_Trade': f"{result['avg_return_per_trade']:.2f}"
                })
            
            # Create DataFrame and export
            df = pd.DataFrame(csv_data)
            df.to_csv(filename, index=False)
            
            print(f"📊 Best windows CSV exported: {filename}")
            return filename
            
        except Exception as e:
            print(f"❌ CSV export failed: {e}")
            return None


def main():
    """Main function to run fast parameter optimization test."""
    try:
        print("🚀 FAST PARAMETER OPTIMIZATION - BEST WINDOWS IDENTIFICATION")
        print("="*70)
        
        # Initialize optimizer
        optimizer = FastParameterOptimizer()
        
        # Load real NIFTY data
        print("\n📊 Loading real NIFTY data from database...")
        data_loader = EfficientDataLoader(config_loader=optimizer.config_loader)
        
        market_data, warmup_analysis = data_loader.get_ohlcv_data_with_warmup(
            "NIFTY", "NSE_INDEX", "5m", "2025-06-01", "2025-06-30"
        )
        
        print(f"   ✅ Loaded {len(market_data):,} NIFTY 5m candles")
        print(f"   📅 Range: {market_data.index[0]} to {market_data.index[-1]}")
        
        # Run optimization
        results = optimizer.optimize_parameters(market_data)
        
        if results:
            # Display results
            print(f"\n📈 OPTIMIZATION RESULTS:")
            print(f"   Combinations tested: {results['summary']['total_combinations_tested']:,}")
            print(f"   Processing time: {results['summary']['optimization_time_seconds']:.2f}s")
            print(f"   Speed: {results['summary']['combinations_per_second']:.0f} combinations/sec")
            
            # Show best parameters
            print(f"\n🏆 BEST PARAMETERS BY STRATEGY:")
            for strategy_key, best in results['best_by_strategy'].items():
                params = best['parameters']
                print(f"   {strategy_key}:")
                print(f"     RSI: {params['rsi_14']}, EMA: {params['ema_9']}/{params['ema_20']}")
                print(f"     SMA: {params['sma_9']}/{params['sma_20']}/{params['sma_50']}/{params['sma_200']}")
                print(f"     Return: {best['total_return_pct']:.1f}%, Trades: {best['total_trades']}")
            
            # Export CSV
            csv_file = optimizer.export_best_windows_csv(results)
            
            print(f"\n✅ Fast parameter optimization completed successfully!")
            print(f"📄 Best windows report: {csv_file}")
            
        else:
            print("❌ Optimization failed")
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()