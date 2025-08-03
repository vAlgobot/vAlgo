"""
Ultra-Fast Vectorized P&L Calculator for VectorBT System
=======================================================

Ports the proven ultra-fast vectorized P&L calculation methods from
vectorbt_indicator_signal_check.py to achieve maximum performance.

Key Features:
- Pure NumPy vectorized operations for all calculations
- Comprehensive portfolio analytics with institutional metrics
- Ultra-fast slippage and commission calculations
- Vectorized drawdown analysis and risk metrics
- Zero-loop processing architecture

Author: vAlgo Development Team
Created: July 29, 2025
Based on proven patterns from vectorbt_indicator_signal_check.py (Lines 1568-1796)
"""

import sys
import time
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime
import warnings

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.json_config_loader import JSONConfigLoader, ConfigurationError

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# vAlgo selective logger - required for production
from utils.selective_logger import get_selective_logger

# Default values - will be overridden by JSON configuration
DEFAULT_COMMISSION = 20.0  # Per trade commission
DEFAULT_SLIPPAGE = 0.1     # Slippage percentage
DEFAULT_LOT_SIZE = 75      # NIFTY options lot size


class UltraFastPnLCalculator:
    """
    Ultra-fast vectorized P&L calculator using proven NumPy operations.
    
    Features:
    - Pure NumPy vectorized calculations for maximum performance
    - Comprehensive institutional-grade portfolio analytics
    - Ultra-fast risk metrics (Sharpe, Sortino, Calmar ratios)
    - Vectorized drawdown analysis
    - Advanced streak analysis
    - Zero-loop processing architecture
    """
    
    def __init__(self, config_loader: Optional[JSONConfigLoader] = None):
        """
        Initialize Ultra-Fast P&L Calculator.
        
        Args:
            config_loader: Optional pre-initialized config loader
        """
        self.config_loader = config_loader or JSONConfigLoader()
        self.logger = get_selective_logger("ultra_fast_pnl_calculator")
        
        # Load configurations
        self.main_config, self.strategies_config = self.config_loader.load_all_configs()
        
        # Get trading configuration
        trading_config = self.main_config.get('trading', {})
        options_config = self.main_config.get('options', {})
        
        # Use consolidated lot_size from trading config
        self.lot_size = trading_config.get('lot_size', DEFAULT_LOT_SIZE)
        self.commission_per_trade = trading_config.get('commission_per_trade', DEFAULT_COMMISSION)
        self.slippage_percentage = trading_config.get('slippage_percentage', DEFAULT_SLIPPAGE) * 100
        self.initial_capital = trading_config.get('initial_capital', 100000)
        # Position sizes will come from individual trade data
        
        # Performance tracking
        self.performance_stats = {
            'total_calculations': 0,
            'vectorized_operations_time': 0.0,
            'risk_metrics_time': 0.0,
            'drawdown_analysis_time': 0.0,
            'total_pnl_calculation_time': 0.0
        }
        
        print(f"🚀 Ultra-Fast P&L Calculator initialized")
        print(f"   💰 Lot Size: {self.lot_size}")
        print(f"   💸 Commission: ₹{self.commission_per_trade}/trade")
        print(f"   📊 Slippage: {self.slippage_percentage}%")
        print(f"   🏦 Initial Capital: ₹{self.initial_capital:,}")
        print(f"   ⚡ Vectorized Operations: Enabled")
    
    def calculate_vectorized_pnl(self, enriched_trades: List[Dict]) -> Dict[str, Any]:
        """
        Ultra-fast vectorized P&L calculations using pure NumPy operations.
        Based on proven pattern from vectorbt_indicator_signal_check.py (Lines 1568-1796)
        
        Args:
            enriched_trades: List of trades with premium data
            
        Returns:
            Dictionary with comprehensive performance metrics
        """
        try:
            pnl_start_time = time.time()
            
            if not enriched_trades:
                return self._get_empty_pnl_result()
            
            print(f"🔥 Ultra-fast vectorized P&L calculation for {len(enriched_trades)} trades...")
            
            # Convert to NumPy arrays for vectorized operations
            vectorization_start = time.time()
            
            entry_premiums = np.array([trade['entry_premium'] for trade in enriched_trades])
            exit_premiums = np.array([trade['exit_premium'] for trade in enriched_trades])
            position_sizes = np.array([trade.get('position_size', 1) for trade in enriched_trades])
            
            # Ultra-fast slippage calculations (vectorized)
            # Entry slippage: increase premium (worse for buyer)
            entry_premiums_with_slippage = entry_premiums * (1 + self.slippage_percentage / 100)
            # Exit slippage: decrease premium (worse for seller)
            exit_premiums_with_slippage = exit_premiums * (1 - self.slippage_percentage / 100)
            
            # Calculate slippage amounts for tracking (include position size)
            entry_slippage_array = entry_premiums * (self.slippage_percentage / 100)
            exit_slippage_array = exit_premiums * (self.slippage_percentage / 100)
            total_slippage_array = (entry_slippage_array + exit_slippage_array) * self.lot_size * position_sizes
            
            # Vectorized P&L calculations with slippage and position sizing
            gross_pnl_array = (exit_premiums_with_slippage - entry_premiums_with_slippage) * self.lot_size * position_sizes
            commission_array = np.full_like(gross_pnl_array, self.commission_per_trade)
            net_pnl_array = gross_pnl_array - commission_array
            
            self.performance_stats['vectorized_operations_time'] += time.time() - vectorization_start
            
            # Basic vectorized statistics
            total_pnl = np.sum(net_pnl_array)
            winning_trades = np.sum(net_pnl_array > 0)
            losing_trades = np.sum(net_pnl_array < 0)
            total_trades = len(enriched_trades)
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            
            # Calculate return percentage using configured capital
            total_return_pct = (total_pnl / self.initial_capital * 100) if self.initial_capital > 0 else 0
            
            # =========================================================================
            # COMPREHENSIVE PORTFOLIO ANALYTICS CALCULATIONS (VECTORIZED)
            # =========================================================================
            
            # 1. TRADE EXTREMES ANALYSIS
            winning_pnl = net_pnl_array[net_pnl_array > 0]
            losing_pnl = net_pnl_array[net_pnl_array < 0]
            
            best_trade = float(np.max(net_pnl_array)) if len(net_pnl_array) > 0 else 0.0
            worst_trade = float(np.min(net_pnl_array)) if len(net_pnl_array) > 0 else 0.0
            avg_winning_trade = float(np.mean(winning_pnl)) if len(winning_pnl) > 0 else 0.0
            avg_losing_trade = float(np.mean(losing_pnl)) if len(losing_pnl) > 0 else 0.0
            
            # 2. PROFIT FACTOR CALCULATION
            total_wins = float(np.sum(winning_pnl)) if len(winning_pnl) > 0 else 0.0
            total_losses = float(abs(np.sum(losing_pnl))) if len(losing_pnl) > 0 else 0.0
            profit_factor = total_wins / total_losses if total_losses > 0 else float('inf') if total_wins > 0 else 0.0
            
            # 3. WIN/LOSS RATIO
            win_loss_ratio = abs(avg_winning_trade / avg_losing_trade) if avg_losing_trade != 0 else float('inf') if avg_winning_trade > 0 else 0.0
            
            # 4. ULTRA-FAST STREAK ANALYSIS (VECTORIZED)
            streak_start = time.time()
            
            trade_results = (net_pnl_array > 0).astype(int)  # 1 for win, 0 for loss
            
            # Vectorized streak calculation using numpy
            win_streaks = self._calculate_streaks_vectorized(trade_results, 1)
            loss_streaks = self._calculate_streaks_vectorized(1 - trade_results, 1)  # Invert for losses
            
            max_win_streak = int(max(win_streaks)) if win_streaks else 0
            max_loss_streak = int(max(loss_streaks)) if loss_streaks else 0
            
            # 5. ULTRA-FAST DRAWDOWN ANALYSIS (VECTORIZED)
            drawdown_start = time.time()
            
            cumulative_pnl = np.cumsum(net_pnl_array)
            portfolio_value = self.initial_capital + cumulative_pnl
            
            # Calculate running maximum portfolio value (vectorized)
            running_max = np.maximum.accumulate(portfolio_value)
            
            # Calculate drawdown at each point (vectorized)
            drawdown_array = (portfolio_value - running_max) / running_max * 100
            
            # Find maximum drawdown
            max_drawdown_pct = float(abs(np.min(drawdown_array))) if len(drawdown_array) > 0 else 0.0
            max_drawdown_idx = np.argmin(drawdown_array) if len(drawdown_array) > 0 else 0
            max_drawdown_amount = float(abs(portfolio_value[max_drawdown_idx] - running_max[max_drawdown_idx])) if len(drawdown_array) > 0 else 0.0
            
            # Final portfolio value
            final_portfolio_value = float(self.initial_capital + total_pnl)
            
            self.performance_stats['drawdown_analysis_time'] += time.time() - drawdown_start
            
            # 6. ULTRA-FAST RISK-ADJUSTED METRICS (VECTORIZED)
            risk_metrics_start = time.time()
            
            if len(net_pnl_array) > 1:
                # Calculate trade returns for risk metrics
                trade_returns = net_pnl_array / self.initial_capital  # Returns as fraction of capital
                avg_return = np.mean(trade_returns)
                return_std = np.std(trade_returns, ddof=1)
                
                # Sharpe Ratio (assuming risk-free rate = 0)
                sharpe_ratio = float(avg_return / return_std) if return_std > 0 else 0.0
                
                # Sortino Ratio (downside deviation) - vectorized
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
            
            self.performance_stats['risk_metrics_time'] += time.time() - risk_metrics_start
            
            # 7. POSITION SIZING ANALYTICS (Strategy-based position sizing)
            position_sizes = np.array([trade.get('position_size', 1) for trade in enriched_trades])
            avg_position_size = float(np.mean(position_sizes)) if len(position_sizes) > 0 else 1.0
            max_position_size = float(np.max(position_sizes)) if len(position_sizes) > 0 else 1.0
            
            # Capital efficiency calculation
            avg_capital_per_trade = avg_position_size * self.lot_size * np.mean(entry_premiums) if len(entry_premiums) > 0 else 0.0
            capital_efficiency = float((avg_capital_per_trade / self.initial_capital) * 100) if self.initial_capital > 0 else 0.0
            
            # Build detailed trades list with slippage information
            detailed_trades = []
            for i, trade in enumerate(enriched_trades):
                detailed_trades.append({
                    'entry_time': trade['entry_timestamp'],
                    'exit_time': trade['exit_timestamp'],
                    'entry_price': trade['entry_price'],
                    'exit_price': trade['exit_price'],
                    'strike': trade['entry_strike'],
                    'entry_premium': trade['entry_premium'],
                    'exit_premium': trade['exit_premium'],
                    'entry_premium_with_slippage': entry_premiums_with_slippage[i],
                    'exit_premium_with_slippage': exit_premiums_with_slippage[i],
                    'slippage_cost': total_slippage_array[i],
                    'gross_pnl': gross_pnl_array[i],
                    'commission': commission_array[i],
                    'net_pnl': net_pnl_array[i],
                    'position_size': trade.get('position_size', 1)
                })
            
            # Update performance stats
            total_pnl_time = time.time() - pnl_start_time
            self.performance_stats['total_pnl_calculation_time'] += total_pnl_time
            self.performance_stats['total_calculations'] += 1
            
            print(f"✅ Ultra-fast vectorized P&L calculation completed:")
            print(f"   💰 Total P&L: ₹{total_pnl:,.2f}")
            print(f"   📊 Win Rate: {win_rate:.1f}%")
            print(f"   📈 Profit Factor: {profit_factor:.2f}")
            print(f"   ⚡ Calculation Time: {total_pnl_time:.4f}s")
            
            return {
                # Basic metrics
                'total_pnl': float(total_pnl),
                'total_return_pct': float(total_return_pct),
                'total_trades': int(total_trades),
                'winning_trades': int(winning_trades),
                'losing_trades': int(losing_trades),
                'win_rate': float(win_rate),
                'avg_pnl_per_trade': float(avg_pnl_per_trade),
                
                # Capital metrics
                'initial_capital': float(self.initial_capital),
                'final_portfolio_value': final_portfolio_value,
                
                # Trade extremes
                'best_trade': best_trade,
                'worst_trade': worst_trade,
                'avg_winning_trade': avg_winning_trade,
                'avg_losing_trade': avg_losing_trade,
                
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
                
                # System info
                'trades': detailed_trades,
                'commission_per_trade': self.commission_per_trade,
                'lot_size': self.lot_size,
                'calculation_method': 'ultra_fast_vectorized',
                
                # Trading costs breakdown
                'total_commission': float(np.sum(commission_array)),
                'total_slippage': float(np.sum(total_slippage_array)),
                'slippage_percentage': self.slippage_percentage,
                'avg_slippage_per_trade': float(np.mean(total_slippage_array)) if len(total_slippage_array) > 0 else 0.0,
                
                # Performance metrics
                'calculation_time': total_pnl_time,
                'vectorization_enabled': True
            }
            
        except Exception as e:
            self.logger.log_detailed(f"Error in ultra-fast vectorized P&L calculations: {e}", "ERROR", "PNL_CALCULATOR")
            return self._get_empty_pnl_result()
    
    def _calculate_streaks_vectorized(self, binary_array: np.ndarray, target_value: int) -> List[int]:
        """
        Ultra-fast vectorized streak calculation using NumPy operations.
        
        Args:
            binary_array: Array of 0s and 1s
            target_value: Value to count streaks for (0 or 1)
            
        Returns:
            List of streak lengths
        """
        try:
            # Find where the array equals target value
            mask = (binary_array == target_value)
            
            if not np.any(mask):
                return []
            
            # Find transitions using numpy diff
            transitions = np.diff(np.concatenate(([False], mask, [False])).astype(int))
            
            # Find start and end indices of streaks
            starts = np.where(transitions == 1)[0]
            ends = np.where(transitions == -1)[0]
            
            # Calculate streak lengths
            streaks = (ends - starts).tolist()
            
            return streaks
            
        except Exception as e:
            self.logger.log_detailed(f"Error in vectorized streak calculation: {e}", "ERROR", "PNL_CALCULATOR")
            return []
    
    def _get_empty_pnl_result(self) -> Dict[str, Any]:
        """Return empty P&L result structure."""
        return {
            # Basic metrics
            'total_pnl': 0.0,
            'total_return_pct': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_per_trade': 0.0,
            
            # Capital metrics
            'initial_capital': float(self.initial_capital),
            'final_portfolio_value': float(self.initial_capital),
            
            # Trade extremes
            'best_trade': 0.0,
            'worst_trade': 0.0,
            'avg_winning_trade': 0.0,
            'avg_losing_trade': 0.0,
            
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
            
            # Position sizing (empty case)
            'avg_position_size': 1.0,
            'max_position_size': 1.0,
            'capital_efficiency': 0.0,
            'avg_capital_used': 0.0,
            
            # System info
            'trades': [],
            'commission_per_trade': self.commission_per_trade,
            'lot_size': self.lot_size,
            'calculation_method': 'ultra_fast_vectorized',
            
            # Trading costs breakdown
            'total_commission': 0.0,
            'total_slippage': 0.0,
            'slippage_percentage': self.slippage_percentage,
            'avg_slippage_per_trade': 0.0,
            
            # Performance metrics
            'calculation_time': 0.0,
            'vectorization_enabled': True
        }
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get comprehensive performance summary of P&L calculations."""
        total_time = (
            self.performance_stats['vectorized_operations_time'] +
            self.performance_stats['risk_metrics_time'] +
            self.performance_stats['drawdown_analysis_time']
        )
        
        calculations_per_second = (
            self.performance_stats['total_calculations'] / total_time
            if total_time > 0 else 0
        )
        
        return {
            'total_vectorized_operations_time': round(self.performance_stats['vectorized_operations_time'], 4),
            'risk_metrics_calculation_time': round(self.performance_stats['risk_metrics_time'], 4),
            'drawdown_analysis_time': round(self.performance_stats['drawdown_analysis_time'], 4),
            'total_pnl_calculation_time': round(self.performance_stats['total_pnl_calculation_time'], 4),
            'total_calculations': self.performance_stats['total_calculations'],
            'calculations_per_second': round(calculations_per_second, 0),
            'numpy_vectorization_enabled': True,
            'comprehensive_analytics_enabled': True
        }


# Convenience function for external use
def create_ultra_fast_pnl_calculator(config_dir: Optional[str] = None) -> UltraFastPnLCalculator:
    """
    Create and initialize Ultra-Fast P&L Calculator.
    
    Args:
        config_dir: Optional configuration directory path
        
    Returns:
        Initialized UltraFastPnLCalculator instance
    """
    config_loader = JSONConfigLoader(config_dir) if config_dir else None
    return UltraFastPnLCalculator(config_loader)


if __name__ == "__main__":
    # Test Ultra-Fast P&L Calculator
    try:
        print("🧪 Testing Ultra-Fast Vectorized P&L Calculator...")
        
        # Create P&L calculator
        pnl_calculator = UltraFastPnLCalculator()
        
        # Create sample enriched trades for testing
        sample_trades = []
        np.random.seed(42)  # For reproducible results
        
        for i in range(100):  # Test with 100 trades
            entry_premium = np.random.uniform(50, 200)
            exit_premium = entry_premium + np.random.normal(0, 50)  # Random P&L
            
            trade = {
                'entry_timestamp': pd.Timestamp('2024-06-01') + pd.Timedelta(hours=i),
                'exit_timestamp': pd.Timestamp('2024-06-01') + pd.Timedelta(hours=i+1),
                'entry_price': 23000 + np.random.normal(0, 100),
                'exit_price': 23000 + np.random.normal(0, 100),
                'entry_strike': 23000,
                'entry_premium': entry_premium,
                'exit_premium': exit_premium
            }
            sample_trades.append(trade)
        
        # Test ultra-fast P&L calculation
        print(f"\n🚀 Testing ultra-fast vectorized P&L calculation...")
        results = pnl_calculator.calculate_vectorized_pnl(sample_trades)
        
        print(f"\n📊 P&L Calculation Results:")
        print(f"   💰 Total P&L: ₹{results['total_pnl']:,.2f}")
        print(f"   📈 Win Rate: {results['win_rate']:.1f}%")
        print(f"   🔄 Total Trades: {results['total_trades']}")
        print(f"   📊 Profit Factor: {results['profit_factor']:.2f}")
        print(f"   📉 Max Drawdown: {results['max_drawdown_pct']:.2f}%")
        print(f"   ⚡ Calculation Time: {results['calculation_time']:.4f}s")
        
        # Display performance summary
        performance = pnl_calculator.get_performance_summary()
        print(f"\n📋 Performance Summary:")
        for key, value in performance.items():
            print(f"   {key}: {value}")
        
        print("🎉 Ultra-Fast P&L Calculator test completed!")
        
    except Exception as e:
        print(f"❌ P&L calculator test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)