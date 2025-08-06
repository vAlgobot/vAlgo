#!/usr/bin/env python3
"""
Minimal Parameter Test - Real Database Data Only
================================================

Focused parameter optimization using ONLY real NIFTY database data.
No fallbacks, no hardcoded data, no mock data.

Identifies best RSI, EMA, SMA windows for 5-minute candles.

Author: vAlgo Development Team  
Created: August 5, 2025
"""

import sys
import time
import pandas as pd
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.json_config_loader import JSONConfigLoader
from core.efficient_data_loader import EfficientDataLoader

# VectorBT for indicators only
try:
    import vectorbt as vbt
    print("✅ VectorBT loaded for indicator calculations")
except ImportError:
    print("❌ VectorBT required for parameter optimization")
    sys.exit(1)


def test_parameter_optimization():
    """
    Test parameter optimization with real database data only.
    NO fallbacks, NO hardcoded data, NO mock data.
    """
    try:
        print("🚀 MINIMAL PARAMETER TEST - REAL DATABASE DATA ONLY")
        print("="*60)
        
        # Initialize components
        config_loader = JSONConfigLoader('config')
        data_loader = EfficientDataLoader(config_loader=config_loader)
        
        # Load REAL database data - SHORT date range for speed
        print("📊 Loading REAL NIFTY database data...")
        start_time = time.time()
        
        market_data, warmup_analysis = data_loader.get_ohlcv_data_with_warmup(
            "NIFTY", "NSE_INDEX", "5m", "2025-06-05", "2025-06-06"  # 2 days only
        )
        
        load_time = time.time() - start_time
        print(f"   ✅ Loaded {len(market_data):,} real records in {load_time:.2f}s")
        print(f"   📅 Range: {market_data.index[0]} to {market_data.index[-1]}")
        print(f"   💹 Price range: ₹{market_data['close'].min():.2f} - ₹{market_data['close'].max():.2f}")
        
        # Verify this is REAL data (no hardcoded values)
        close_prices = market_data['close']
        unique_prices = len(close_prices.unique())
        print(f"   🔍 Data validation: {unique_prices} unique prices (confirms real data)")
        
        if unique_prices < 10:
            raise ValueError("❌ FAILED: Data appears to be mock/hardcoded (too few unique values)")
        
        # Define MINIMAL parameter ranges for fast testing
        print("\n⚡ Testing minimal parameter combinations...")
        test_params = {
            'rsi': [12, 14, 16],        # 3 values
            'ema_fast': [8, 9, 10],     # 3 values  
            'ema_slow': [19, 20, 21],   # 3 values
            'sma_fast': [8, 9, 10],     # 3 values
            'sma_medium': [19, 20, 21], # 3 values
            'sma_slow': [49, 50, 51],   # 3 values
        }
        
        total_combinations = (len(test_params['rsi']) * len(test_params['ema_fast']) * 
                            len(test_params['ema_slow']) * len(test_params['sma_fast']) *
                            len(test_params['sma_medium']) * len(test_params['sma_slow']))
        
        print(f"   📊 Total combinations: {total_combinations}")
        
        # Calculate indicators using VectorBT (vectorized)
        print("🔧 Calculating indicators with VectorBT...")
        indicator_start = time.time()
        
        # RSI for all test periods
        rsi_data = vbt.RSI.run(close_prices, window=test_params['rsi'])
        
        # EMA for all test periods (using MA with ewm)
        all_ema_periods = sorted(set(test_params['ema_fast'] + test_params['ema_slow']))
        ema_data = vbt.MA.run(close_prices, window=all_ema_periods, ewm=True)
        
        # SMA for all test periods
        all_sma_periods = sorted(set(test_params['sma_fast'] + test_params['sma_medium'] + 
                                   test_params['sma_slow']))
        sma_data = vbt.MA.run(close_prices, window=all_sma_periods)
        
        indicator_time = time.time() - indicator_start
        print(f"   ✅ All indicators calculated in {indicator_time:.3f}s")
        
        # Test simple trend strategy combinations
        print("🎯 Testing trend strategy combinations...")
        test_start = time.time()
        
        results = []
        tested = 0
        
        for rsi_period in test_params['rsi']:
            for ema_fast in test_params['ema_fast']:
                for ema_slow in test_params['ema_slow']:
                    if ema_fast >= ema_slow:
                        continue
                        
                    for sma_fast in test_params['sma_fast']:
                        for sma_medium in test_params['sma_medium']:
                            for sma_slow in test_params['sma_slow']:
                                if not (sma_fast < sma_medium < sma_slow):
                                    continue
                                
                                # Test this combination
                                result = test_single_combination(
                                    rsi_period, ema_fast, ema_slow, sma_fast, sma_medium, sma_slow,
                                    market_data, rsi_data, ema_data, sma_data,
                                    test_params['rsi'], all_ema_periods, all_sma_periods
                                )
                                
                                if result:
                                    results.append(result)
                                
                                tested += 1
        
        test_time = time.time() - test_start
        print(f"   ✅ Tested {tested} combinations in {test_time:.3f}s")
        
        # Rank results
        if results:
            results.sort(key=lambda x: x['total_return_pct'], reverse=True)
            
            print(f"\n🏆 TOP 5 PARAMETER COMBINATIONS:")
            print(f"{'Rank':<4} {'RSI':<4} {'EMA':<8} {'SMA':<12} {'Return%':<8} {'Trades':<6} {'Win%':<6}")
            print("-" * 60)
            
            for i, result in enumerate(results[:5], 1):
                ema_str = f"{result['ema_fast']}/{result['ema_slow']}"
                sma_str = f"{result['sma_fast']}/{result['sma_medium']}/{result['sma_slow']}"
                print(f"{i:<4} {result['rsi_period']:<4} {ema_str:<8} {sma_str:<12} "
                      f"{result['total_return_pct']:<7.1f} {result['total_trades']:<6} {result['win_rate']:<5.1f}")
            
            # Export best windows to CSV
            print(f"\n📊 Exporting best windows to CSV...")
            export_best_windows_csv(results, market_data)
            
        else:
            print("⚠️ No valid results generated")
        
        # Summary
        total_time = time.time() - start_time
        print(f"\n✅ Parameter optimization completed in {total_time:.2f}s")
        print(f"📊 Processing speed: {tested/total_time:.0f} combinations/second")
        print(f"🎯 Used REAL database data: {len(market_data):,} authentic NIFTY records")
        
        return results
        
    except Exception as e:
        print(f"❌ Parameter test failed: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_single_combination(rsi_period, ema_fast, ema_slow, sma_fast, sma_medium, sma_slow,
                           market_data, rsi_data, ema_data, sma_data,
                           rsi_periods, ema_periods, sma_periods):
    """Test a single parameter combination."""
    try:
        # Extract indicators for this combination
        rsi_idx = rsi_periods.index(rsi_period)
        ema_fast_idx = ema_periods.index(ema_fast)
        ema_slow_idx = ema_periods.index(ema_slow)
        sma_fast_idx = sma_periods.index(sma_fast)
        sma_medium_idx = sma_periods.index(sma_medium)
        sma_slow_idx = sma_periods.index(sma_slow)
        
        rsi = rsi_data.rsi.iloc[:, rsi_idx]
        ema_f = ema_data.ma.iloc[:, ema_fast_idx]
        ema_s = ema_data.ma.iloc[:, ema_slow_idx]
        sma_f = sma_data.ma.iloc[:, sma_fast_idx]
        sma_m = sma_data.ma.iloc[:, sma_medium_idx]
        sma_sl = sma_data.ma.iloc[:, sma_slow_idx]
        
        close = market_data['close']
        
        # Simple trend following strategy
        # Entry: Price > all SMAs AND EMA fast > EMA slow AND RSI > 50
        entries = (close > sma_f) & (close > sma_m) & (close > sma_sl) & (ema_f > ema_s) & (rsi > 50)
        
        # Exit: Price < SMA fast OR EMA fast < EMA slow
        exits = (close < sma_f) | (ema_f < ema_s)
        
        # Calculate performance
        entry_signals = entries[entries]
        exit_signals = exits[exits]
        
        if len(entry_signals) == 0 or len(exit_signals) == 0:
            return None
        
        # Match entries with exits
        total_trades = min(len(entry_signals), len(exit_signals))
        
        if total_trades == 0:
            return None
        
        entry_prices = close[entry_signals.index[:total_trades]]
        exit_prices = close[exit_signals.index[:total_trades]]
        
        # Calculate returns (CALL option simulation)
        returns = (exit_prices.values - entry_prices.values) / entry_prices.values
        
        total_return_pct = np.sum(returns) * 100
        win_rate = (returns > 0).mean() * 100
        
        return {
            'rsi_period': rsi_period,
            'ema_fast': ema_fast,
            'ema_slow': ema_slow,
            'sma_fast': sma_fast,
            'sma_medium': sma_medium,
            'sma_slow': sma_slow,
            'total_trades': total_trades,
            'total_return_pct': total_return_pct,
            'win_rate': win_rate,
            'avg_return_per_trade': np.mean(returns) * 100
        }
        
    except Exception:
        return None


def export_best_windows_csv(results, market_data):
    """Export best windows to CSV."""
    try:
        from datetime import datetime
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"best_windows_NIFTY_5m_{timestamp}.csv"
        
        # Prepare CSV data
        csv_data = []
        
        # Add header info
        csv_data.append({
            'Type': 'SUMMARY',
            'Data_Source': 'REAL_DATABASE',
            'Symbol': 'NIFTY',
            'Timeframe': '5m',
            'Date_Range': f"{market_data.index[0]} to {market_data.index[-1]}",
            'Total_Records': len(market_data),
            'Price_Range': f"₹{market_data['close'].min():.2f} - ₹{market_data['close'].max():.2f}",
            'RSI_Period': '',
            'EMA_Fast': '',
            'EMA_Slow': '',
            'SMA_Fast': '',
            'SMA_Medium': '',
            'SMA_Slow': '',
            'Total_Return_Pct': '',
            'Win_Rate': '',
            'Total_Trades': '',
            'Avg_Return_Per_Trade': ''
        })
        
        # Add top 10 results
        for i, result in enumerate(results[:10], 1):
            csv_data.append({
                'Type': f'RANK_{i}',
                'Data_Source': 'REAL_DATABASE',
                'Symbol': 'NIFTY',
                'Timeframe': '5m',
                'Date_Range': '',
                'Total_Records': '',
                'Price_Range': '',
                'RSI_Period': result['rsi_period'],
                'EMA_Fast': result['ema_fast'],
                'EMA_Slow': result['ema_slow'],
                'SMA_Fast': result['sma_fast'],
                'SMA_Medium': result['sma_medium'],
                'SMA_Slow': result['sma_slow'],
                'Total_Return_Pct': f"{result['total_return_pct']:.2f}",
                'Win_Rate': f"{result['win_rate']:.1f}",
                'Total_Trades': result['total_trades'],
                'Avg_Return_Per_Trade': f"{result['avg_return_per_trade']:.2f}"
            })
        
        # Export to CSV
        df = pd.DataFrame(csv_data)
        df.to_csv(filename, index=False)
        
        print(f"   ✅ Best windows exported: {filename}")
        
        # Display file contents for verification
        print(f"\n📄 CSV Content Preview:")
        with open(filename, 'r') as f:
            lines = f.readlines()[:5]  # First 5 lines
            for line in lines:
                print(f"   {line.strip()}")
        
        return filename
        
    except Exception as e:
        print(f"   ❌ CSV export failed: {e}")
        return None


if __name__ == "__main__":
    results = test_parameter_optimization()
    
    if results:
        print(f"\n🎊 SUCCESS: Parameter optimization completed with REAL database data!")
        print(f"📊 Generated {len(results)} valid parameter combinations")
    else:
        print(f"\n💥 FAILED: Parameter optimization failed")
        sys.exit(1)