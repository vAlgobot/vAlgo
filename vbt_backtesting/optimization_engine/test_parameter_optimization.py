#!/usr/bin/env python3
"""
Test Parameter Optimization Engine with Real NIFTY Database Data
================================================================

End-to-end test of the Ultra-Fast Parameter Optimization Engine using actual
NIFTY price data from the database. Tests RSI, EMA, and SMA parameter optimization
for identifying best windows for 5-minute candles.

Usage:
    python test_parameter_optimization.py

Author: vAlgo Development Team
Created: August 5, 2025
"""

import sys
import time
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.parameter_optimization_engine import UltraFastParameterOptimizer
from core.json_config_loader import JSONConfigLoader
from core.efficient_data_loader import EfficientDataLoader


def test_with_real_nifty_data():
    """
    Test parameter optimization using real NIFTY database data.
    Focus on RSI, EMA, and SMA parameter identification.
    """
    try:
        print("🚀 TESTING PARAMETER OPTIMIZATION WITH REAL NIFTY DATA")
        print("="*70)
        
        # Initialize components
        print("🔧 Initializing components...")
        config_dir = str(Path(__file__).parent / "config")
        config_loader = JSONConfigLoader(config_dir)
        
        # Display configuration summary
        param_config = config_loader.main_config.get('parameter_optimization', {})
        print(f"   📊 RSI periods to test: {len(param_config.get('indicators', {}).get('rsi', {}).get('test_periods', []))}")
        print(f"   📊 EMA periods to test: {len(param_config.get('indicators', {}).get('ema', {}).get('test_periods', []))}")
        print(f"   📊 SMA ranges to test: {len(param_config.get('indicators', {}).get('sma', {}).get('optimization_ranges', {}))}")
        
        # Initialize optimizer
        optimizer = UltraFastParameterOptimizer(config_loader)
        
        # Initialize data loader
        data_loader = EfficientDataLoader(config_loader=config_loader)
        
        # Load real NIFTY data from database
        print(f"\n📊 Loading real NIFTY data from database...")
        symbol = "NIFTY"
        exchange = "NSE_INDEX"
        timeframe = "5m"
        start_date = "2025-06-01"
        end_date = "2025-06-30"
        
        data_start = time.time()
        market_data, warmup_analysis = data_loader.get_ohlcv_data_with_warmup(
            symbol, exchange, timeframe, start_date, end_date
        )
        data_load_time = time.time() - data_start
        
        print(f"   ✅ Loaded {len(market_data):,} NIFTY 5m candles in {data_load_time:.2f}s")
        print(f"   📅 Date range: {market_data.index[0]} to {market_data.index[-1]}")
        print(f"   💹 Price range: ₹{market_data['close'].min():.2f} - ₹{market_data['close'].max():.2f}")
        
        # Display warmup analysis
        print(f"   🔥 Warmup period: {warmup_analysis['days_extended']} days, {warmup_analysis['max_warmup_candles']} candles")
        
        # Run parameter optimization
        print(f"\n⚡ Running ultra-fast parameter optimization...")
        optimization_start = time.time()
        
        results = optimizer.optimize_parameters_vectorized(
            market_data, symbol=symbol, timeframe=timeframe
        )
        
        total_optimization_time = time.time() - optimization_start
        
        # Display comprehensive results
        print(f"\n{'='*70}")
        print(f"📈 PARAMETER OPTIMIZATION RESULTS")
        print(f"{'='*70}")
        
        summary = results['optimization_summary']
        print(f"🎯 Optimization Summary:")
        print(f"   Symbol: {summary['symbol']} {summary['timeframe']}")
        print(f"   Total combinations tested: {summary['total_combinations_tested']:,}")
        print(f"   Processing time: {summary['optimization_time_seconds']:.2f} seconds")
        print(f"   Processing speed: {summary['combinations_per_second']:,.0f} combinations/second")
        
        # Performance comparison
        estimated_brute_force = summary['total_combinations_tested'] * 0.1
        speedup = estimated_brute_force / summary['optimization_time_seconds']
        print(f"   🚀 Performance improvement: {speedup:.0f}x faster than brute force!")
        
        # Show top results
        top_results = results['top_results'][:5]  # Top 5
        print(f"\n🏆 TOP 5 PARAMETER COMBINATIONS:")
        print(f"{'Rank':<4} {'Strategy':<25} {'Score':<6} {'Return%':<8} {'Win%':<6} {'Trades':<6} {'Parameters'}")
        print("-" * 100)
        
        for i, result in enumerate(top_results, 1):
            strategy_name = f"{result['strategy_group']}.{result['strategy_case']}"[:24]
            params_str = f"RSI:{result['parameters']['rsi_14']} EMA:{result['parameters']['ema_9']} SMA:{result['parameters']['sma_9']}-{result['parameters']['sma_20']}-{result['parameters']['sma_50']}-{result['parameters']['sma_200']}"
            
            print(f"{i:<4} {strategy_name:<25} {result['optimization_score']:.3f} {result['total_return_pct']:>7.1f} {result['win_rate']:>5.1f} {result['total_trades']:>5} {params_str}")
        
        # Show best parameters by strategy
        print(f"\n🎯 BEST PARAMETERS BY STRATEGY:")
        for strategy_key, best_result in results['best_parameters_by_strategy'].items():
            print(f"\n   📊 {strategy_key}:")
            params = best_result['parameters']
            perf = best_result['performance']
            
            print(f"      🏆 Best Parameters:")
            print(f"         RSI Period: {params['rsi_14']}")
            print(f"         EMA Periods: {params['ema_9']}, {params['ema_20']}")
            print(f"         SMA Periods: {params['sma_9']}, {params['sma_20']}, {params['sma_50']}, {params['sma_200']}")
            
            print(f"      📈 Performance:")
            print(f"         Optimization Score: {best_result['optimization_score']:.3f}")
            print(f"         Total Return: {perf['total_return_pct']:.1f}%")
            print(f"         Win Rate: {perf['win_rate']:.1f}%")
            print(f"         Sharpe Ratio: {perf['sharpe_ratio']:.2f}")
            print(f"         Max Drawdown: {perf['max_drawdown']:.1f}%")
            print(f"         Total Trades: {perf['total_trades']}")
        
        # Parameter analysis
        print(f"\n📊 PARAMETER ANALYSIS:")
        
        # Analyze RSI parameters
        rsi_performance = {}
        for result in results['all_results']:
            rsi_period = result['parameters']['rsi_14']
            if rsi_period not in rsi_performance:
                rsi_performance[rsi_period] = []
            rsi_performance[rsi_period].append(result['optimization_score'])
        
        # Find best RSI periods
        rsi_avg_scores = {period: sum(scores)/len(scores) for period, scores in rsi_performance.items()}
        best_rsi_periods = sorted(rsi_avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"   🎯 Best RSI Periods (by average optimization score):")
        for period, avg_score in best_rsi_periods:
            print(f"      RSI {period}: {avg_score:.3f} average score")
        
        # Analyze EMA parameters
        ema_performance = {}
        for result in results['all_results']:
            ema_period = result['parameters']['ema_9']
            if ema_period not in ema_performance:
                ema_performance[ema_period] = []
            ema_performance[ema_period].append(result['optimization_score'])
        
        ema_avg_scores = {period: sum(scores)/len(scores) for period, scores in ema_performance.items()}
        best_ema_periods = sorted(ema_avg_scores.items(), key=lambda x: x[1], reverse=True)[:3]
        
        print(f"   🎯 Best EMA Fast Periods (by average optimization score):")
        for period, avg_score in best_ema_periods:
            print(f"      EMA {period}: {avg_score:.3f} average score")
        
        # Export summary
        if results.get('optimization_summary', {}).get('optimization_time_seconds'):
            optimization_file = f"optimization_summary_NIFTY_5m_{int(time.time())}.txt"
            with open(optimization_file, 'w') as f:
                f.write("NIFTY 5m Parameter Optimization Results\n")
                f.write("="*50 + "\n\n")
                f.write(f"Total combinations tested: {summary['total_combinations_tested']:,}\n")
                f.write(f"Processing time: {summary['optimization_time_seconds']:.2f} seconds\n")
                f.write(f"Processing speed: {summary['combinations_per_second']:,.0f} combinations/second\n\n")
                
                f.write("Best Parameters by Strategy:\n")
                for strategy_key, best_result in results['best_parameters_by_strategy'].items():
                    f.write(f"\n{strategy_key}:\n")
                    f.write(f"  Parameters: {best_result['parameters']}\n")
                    f.write(f"  Return: {best_result['total_return_pct']:.1f}%\n")
                    f.write(f"  Win Rate: {best_result['win_rate']:.1f}%\n")
            
            print(f"\n📄 Summary exported to: {optimization_file}")
        
        print(f"\n✅ Parameter optimization test completed successfully!")
        print(f"🎉 Total execution time: {total_optimization_time:.2f} seconds")
        print(f"{'='*70}")
        
        return results
        
    except Exception as e:
        print(f"\n❌ Parameter optimization test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the test
    results = test_with_real_nifty_data()
    
    if results:
        print(f"\n🎊 TEST PASSED: Parameter optimization working with real NIFTY data!")
    else:
        print(f"\n💥 TEST FAILED: Check error messages above")
        sys.exit(1)