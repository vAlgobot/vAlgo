#!/usr/bin/env python3
"""
VectorBT Indicator Signal Checking System - TAlib Enhanced Version
================================================================

Demonstrates VectorBT IndicatorFactory with industry-standard TAlib implementations:
- TAlibRSI (14 & 21 periods) using TAlib's optimized RSI calculation
- TAlibSMA (9 & 20 periods) using TAlib's optimized SMA calculation
- Superior performance, reliability, and consistency with trading platforms
- Comprehensive CSV output for analysis

Key Improvements:
- Replaced 145+ lines of custom code with 6 lines of TAlib calls
- Better performance through optimized C implementations
- Industry-standard calculations used by professional traders
- Eliminated edge case bugs and calculation errors

Requirements:
- TAlib: pip install TA-Lib (requires system dependencies)
- VectorBT: pip install vectorbt[full]
- See: https://github.com/mrjbq7/ta-lib for TAlib installation guide

Author: vAlgo Development Team
Created: January 25, 2025
Enhanced: January 26, 2025 (TAlib Integration)
"""

import sys
import warnings
import time
from pathlib import Path
from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from datetime import datetime, time as dt_time

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# VectorBT imports
try:
    import vectorbt as vbt
    VECTORBT_AVAILABLE = True
    print(f"✅ VectorBT {vbt.__version__} loaded successfully")
except ImportError:
    VECTORBT_AVAILABLE = False
    print("❌ VectorBT not available. Install with: pip install vectorbt[full]")
    sys.exit(1)

# TAlib imports for technical indicators
try:
    import talib
    TALIB_AVAILABLE = True
    print(f"✅ TAlib {talib.__version__} loaded successfully")
except ImportError:
    TALIB_AVAILABLE = False
    print("❌ TAlib not available. Using fallback NumPy implementations")
    print("💡 Note: For production use, install TAlib with: pip install TA-Lib")

# vAlgo system imports
try:
    from data_manager.database import DatabaseManager
    from utils.logger import get_logger
    VALGO_IMPORTS_AVAILABLE = True
    print("✅ vAlgo system imports loaded successfully")
except ImportError as e:
    VALGO_IMPORTS_AVAILABLE = False
    print(f"❌ vAlgo imports failed: {e}")
    sys.exit(1)

# Options trading imports
try:
    from data_manager.options_database import OptionsDatabase
    from utils.advanced_strike_selector import AdvancedStrikeSelector, StrikeType, OptionType
    from backtest_engine.options_pnl_calculator import OptionsPnLCalculator
    OPTIONS_IMPORTS_AVAILABLE = True
    print("✅ Options trading components loaded successfully")
except ImportError as e:
    OPTIONS_IMPORTS_AVAILABLE = False
    print(f"❌ Options imports failed: {e}")
    print("💡 Note: Options components are optional. System will run in equity mode only.")
    # Don't exit - allow system to run without options

# =============================================================================
# SIGNAL STRATEGY CONFIGURATIONS - Choose your strategy combination
# =============================================================================

SIGNAL_STRATEGIES = {
    'sma_basic': {
        'entry': 'close > ma_20 & close > ma_9',
        'exit': 'close < ma_20',
        'OptionType': 'CALL',
        'description': 'Basic SMA20 crossover',
        'status': 'InActive'
    },
    'multi_sma_trend': {
        'entry': '(close > ma_9) & (ma_9 > ma_20) & (ma_20 > ma_50) & (ma_50 > ma_200)',
        'exit': 'close < ma_20',
        'OptionType': 'CALL',
        'status': 'Active',
        'description': 'Multi-SMA trend entry (9>20>50>200), exit on close < SMA20'
    },
     'multi_sma_trend_put': {
        'entry': '(close < ma_9) & (ma_9 < ma_20) & (ma_20 < ma_50) & (ma_50 < ma_200)',
        'exit': 'close > ma_20',
        'OptionType': 'PUT',
        'status': 'Active',
        'description': 'Multi-SMA trend PUT entry (9<20<50<200), exit on close > SMA20'
    }
}

def get_active_strategies() -> List[str]:
    """
    Get list of strategies with 'Active' status.
    
    Returns:
        List of active strategy names
    """
    active_strategies = []
    for strategy_name, strategy_config in SIGNAL_STRATEGIES.items():
        if strategy_config.get('status', '').lower() == 'active':
            active_strategies.append(strategy_name)
    
    if not active_strategies:
        # Fallback to ACTIVE_STRATEGY if no strategies are marked as active
        active_strategies = [ACTIVE_STRATEGY]
        print(f"⚠️  No strategies marked as 'Active', using fallback: {ACTIVE_STRATEGY}")
    else:
        print(f"✅ Found {len(active_strategies)} active strategies: {', '.join(active_strategies)}")
    
    return active_strategies

# Choose which strategy to use - Change this to test different strategies
ACTIVE_STRATEGY = 'multi_sma_trend_put'  # Will be updated to use DEFAULT_STRATEGY after it's defined

# Portfolio Engine Configuration
# Using VectorBT IndicatorFactory for optimal performance

# Options Portfolio Configuration
USE_OPTIONS_PORTFOLIO = True if OPTIONS_IMPORTS_AVAILABLE else False  # True = Real options P&L, False = Equity P&L
USE_ULTRA_FAST_PNL = True  # Always use ultra-fast vectorized method
DEFAULT_OPTION_TYPE = OptionType.PUT if OPTIONS_IMPORTS_AVAILABLE else None
DEFAULT_STRIKE_TYPE = StrikeType.ATM if OPTIONS_IMPORTS_AVAILABLE else None

# Portfolio Capital and Position Sizing Configuration
INITIAL_CAPITAL = 200000  # ₹2 Lakh capital for portfolio analytics
ENABLE_DYNAMIC_POSITION_SIZING = False  # True = Dynamic sizing, False = Fixed 1 lot
FIXED_POSITION_SIZE = 1  # Fixed position size when dynamic sizing is disabled

# =============================================================================
# GLOBAL SYSTEM CONFIGURATION - Centralized Settings for Easy Modification
# =============================================================================

# Analysis Period Configuration
ANALYSIS_START_DATE = "2024-06-01"      # Start date for backtesting analysis
ANALYSIS_END_DATE = "2025-06-30"        # End date for backtesting analysis
DATA_TIMEFRAME = "5m"                   # Data timeframe (5-minute candles)
VECTORBT_FREQUENCY = "5T"               # VectorBT frequency format

# Trading Instrument Configuration  
TRADING_SYMBOL = "NIFTY"                # Primary trading symbol
TRADING_EXCHANGE = "NSE_INDEX"          # Exchange for the symbol
DEFAULT_STRATEGY = "multi_sma_trend_put"    # Default trading strategy

# Financial Configuration
VECTORBT_INIT_CASH = 1000000           # 10 Lakh initial cash for VectorBT
OPTIONS_COMMISSION = 50.0               # Commission per options trade in ₹
OPTIONS_LOT_SIZE = 50                  # NIFTY options lot size
OPTIONS_SLIPPAGE = 0.05                 # Slippage percentage for options

# Trading Hours Configuration
TRADE_START_TIME = "09:20:00"           # No entries before this time
TRADE_END_TIME = "15:00:00"             # No new entries after this time
FORCED_EXIT_TIME = "15:15:00"           # Force exit any open positions

# Dynamic Technical Indicator Configuration
INDICATOR_CONFIG = {
    'RSI': {
        'enabled': True,
        'periods': [14, 21],  # Generate rsi_14, rsi_21, rsi_20, rsi_30
    },
    'SMA': {
        'enabled': True, 
        'periods': [9, 20, 50, 200],  # Generate ma_5, ma_9, ma_20, etc.
    },
    'EMA': {
        'enabled': False,  # Can be toggled on/off
        'periods': [12, 26],
    }
}

# RSI Signal Thresholds
RSI_OVERSOLD_THRESHOLD = 30            # RSI oversold level
RSI_OVERBOUGHT_THRESHOLD = 70          # RSI overbought level

# Performance Grading Thresholds (Return %)
GRADE_A_PLUS_EXCEPTIONAL = 1000       # A+ Exceptional performance
GRADE_A_PLUS_EXCELLENT = 500          # A+ Excellent performance  
GRADE_A_VERY_GOOD = 100               # A Very Good performance
GRADE_B_PLUS_GOOD = 50                # B+ Good performance
GRADE_B_ABOVE_AVERAGE = 25            # B Above Average performance
GRADE_C_AVERAGE = 10                  # C Average performance

# Output Configuration
OUTPUT_DIRECTORY = "output/vectorbt"   # Output directory for results

# Update ACTIVE_STRATEGY to use centralized config
ACTIVE_STRATEGY = DEFAULT_STRATEGY

print(f"🎯 Active Strategy: {ACTIVE_STRATEGY} - {SIGNAL_STRATEGIES[ACTIVE_STRATEGY]['description']}")
print(f"   📈 Entry: {SIGNAL_STRATEGIES[ACTIVE_STRATEGY]['entry']}")
print(f"   📉 Exit: {SIGNAL_STRATEGIES[ACTIVE_STRATEGY]['exit']}")
print(f"⚡ Portfolio Engine: VectorBT IndicatorFactory (Optimized Performance)")
print(f"💰 Portfolio Mode: {'Real Options P&L' if USE_OPTIONS_PORTFOLIO else 'Equity P&L'}")
if USE_OPTIONS_PORTFOLIO:
    print(f"   📊 Option Type: {DEFAULT_OPTION_TYPE.value if DEFAULT_OPTION_TYPE else 'N/A'}")
    print(f"   🎯 Strike Type: {DEFAULT_STRIKE_TYPE.value if DEFAULT_STRIKE_TYPE else 'N/A'}")
    print(f"   📦 Lot Size: {OPTIONS_LOT_SIZE}")
    print(f"   🔧 P&L Method: {'Ultra-Fast Vectorized' if USE_ULTRA_FAST_PNL else 'VectorBT from_orders()'}")

# Display dynamic indicator configuration
print(f"🔧 Dynamic Indicator Configuration:")
for indicator_type, config in INDICATOR_CONFIG.items():
    if config['enabled']:
        periods_str = ', '.join(map(str, config['periods']))
        print(f"   ✅ {indicator_type}: {periods_str}")
    else:
        print(f"   ❌ {indicator_type}: Disabled")
print("-" * 80)

# Fallback NumPy implementations for when TAlib is not available
def _calculate_rsi_numpy(close: np.ndarray, period: int) -> np.ndarray:
    """Fallback RSI calculation using NumPy."""
    try:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        
        # Calculate exponential moving averages
        avg_gain = np.empty_like(close)
        avg_loss = np.empty_like(close)
        avg_gain[:] = np.nan
        avg_loss[:] = np.nan
        
        # Initial SMA
        if len(gain) >= period:
            avg_gain[period] = np.mean(gain[:period])
            avg_loss[period] = np.mean(loss[:period])
            
            # Subsequent EMA
            for i in range(period + 1, len(close)):
                avg_gain[i] = (avg_gain[i-1] * (period - 1) + gain[i-1]) / period
                avg_loss[i] = (avg_loss[i-1] * (period - 1) + loss[i-1]) / period
        
        # Calculate RSI
        rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    except Exception:
        return np.full_like(close, 50.0)  # Default to neutral RSI

def _calculate_sma_numpy(close: np.ndarray, period: int) -> np.ndarray:
    """Fallback SMA calculation using NumPy."""
    try:
        sma = np.empty_like(close)
        sma[:] = np.nan
        
        for i in range(period - 1, len(close)):
            sma[i] = np.mean(close[i - period + 1:i + 1])
        
        return sma
    except Exception:
        return np.full_like(close, np.nan)

# TAlib-based VectorBT Indicator Functions for superior performance and reliability
def talib_rsi_apply_func(close, period, **kwargs):
    """
    TAlib-based RSI calculation for VectorBT IndicatorFactory.
    Uses industry-standard TAlib implementation for reliability and performance.
    
    Advantages over custom implementation:
    - Battle-tested and widely used in trading systems
    - Optimized C code for better performance  
    - Consistent with other trading platforms
    - No edge case bugs or calculation errors
    - Much simpler and more maintainable code
    
    Args:
        close: Price series (numpy array or pandas Series)
        period: RSI period (typically 14 or 21)
        **kwargs: Additional parameters (ignored)
    
    Returns:
        numpy array with RSI values (0-100 range)
    """
    # Convert to numpy array with proper dtype for TAlib
    close = np.asarray(close, dtype=np.float64).flatten()
    period = int(period)
    
    print(f"🔍 RSI calculation - close length: {len(close)}, period: {period}")
    
    # Use TAlib RSI function if available, otherwise fallback to numpy implementation
    if TALIB_AVAILABLE:
        rsi_values = talib.RSI(close, timeperiod=period)
    else:
        # Fallback NumPy RSI calculation
        rsi_values = _calculate_rsi_numpy(close, period)
    
    # Validation and debug output
    valid_count = np.sum(~np.isnan(rsi_values))
    print(f"✅ TAlib RSI complete - {valid_count} valid values")
    
    if valid_count > 0:
        valid_rsi = rsi_values[~np.isnan(rsi_values)]
        rsi_min, rsi_max = valid_rsi.min(), valid_rsi.max()
        print(f"📊 RSI range: {rsi_min:.2f} - {rsi_max:.2f}")
        
        # Show sample values
        sample_start = np.where(~np.isnan(rsi_values))[0][0] if valid_count > 0 else period
        sample_end = min(sample_start + 5, len(rsi_values))
        sample_values = rsi_values[sample_start:sample_end]
        print(f"📈 Sample RSI: {sample_values}")
    
    return rsi_values

def talib_sma_apply_func(close, period, **kwargs):
    """
    TAlib-based SMA calculation for VectorBT IndicatorFactory.
    Uses industry-standard TAlib implementation for reliability and performance.
    
    Advantages over custom implementation:
    - Optimized and battle-tested implementation
    - Consistent with other trading platforms
    - Better performance for large datasets
    - Much simpler code (3 lines vs 15+ lines)
    
    Args:
        close: Price series (numpy array or pandas Series)
        period: SMA period (typically 9 or 20)
        **kwargs: Additional parameters (ignored)
        
    Returns:
        numpy array with SMA values
    """
    # Convert to numpy array with proper dtype for TAlib
    close = np.asarray(close, dtype=np.float64).flatten()
    period = int(period)
    
    print(f"🔍 SMA calculation - close length: {len(close)}, period: {period}")
    
    # Use TAlib SMA function if available, otherwise fallback to numpy implementation
    if TALIB_AVAILABLE:
        sma_values = talib.SMA(close, timeperiod=period)
    else:
        # Fallback NumPy SMA calculation
        sma_values = _calculate_sma_numpy(close, period)
    
    # Validation and debug output
    valid_count = np.sum(~np.isnan(sma_values))
    print(f"✅ TAlib SMA complete - {valid_count} valid values")
    
    if valid_count > 0:
        valid_sma = sma_values[~np.isnan(sma_values)]
        sma_min, sma_max = valid_sma.min(), valid_sma.max()
        print(f"📊 SMA range: {sma_min:.2f} - {sma_max:.2f}")
        
        # Show sample values
        sample_start = np.where(~np.isnan(sma_values))[0][0] if valid_count > 0 else period - 1
        sample_end = min(sample_start + 3, len(sma_values))
        sample_values = sma_values[sample_start:sample_end]
        print(f"📈 Sample SMA: {sample_values}")
    
    return sma_values

def combined_indicators_apply_func_v2(close, periods, indicator_types):
    """
    Calculate indicators based on flattened period and type arrays.
    
    This approach uses single-length parameter arrays to avoid VectorBT broadcasting issues.
    Temporary fallback for testing - in production, require TAlib installation.
    
    Args:
        close: Close price series
        periods: Array of periods for each indicator [14, 21, 20, 30, 5, 9, 20, ...]
        indicator_types: Array of indicator types ['RSI', 'RSI', 'RSI', 'RSI', 'SMA', 'SMA', ...]
        
    Returns:
        Tuple of all calculated indicator values in order
    """
    results = []
    
    # Process each indicator based on type and period
    for period, ind_type in zip(periods, indicator_types):
        if ind_type == 'RSI':
            if TALIB_AVAILABLE:
                rsi_values = talib.RSI(close, timeperiod=period)
            else:
                # Temporary fallback for testing
                rsi_values = _calculate_rsi_numpy(close.values, period)
                rsi_values = pd.Series(rsi_values, index=close.index)
            results.append(rsi_values)
            
        elif ind_type == 'SMA':
            if TALIB_AVAILABLE:
                sma_values = talib.SMA(close, timeperiod=period)
            else:
                # Temporary fallback for testing
                sma_values = _calculate_sma_numpy(close.values, period)
                sma_values = pd.Series(sma_values, index=close.index)
            results.append(sma_values)
            
        elif ind_type == 'EMA':
            if TALIB_AVAILABLE:
                ema_values = talib.EMA(close, timeperiod=period)
            else:
                # Temporary fallback for testing
                ema_values = close.ewm(span=period).mean()
            results.append(ema_values)
        
        else:
            raise ValueError(f"Unsupported indicator type: {ind_type}")
    
    return tuple(results)

# Create VectorBT Custom Indicators using IndicatorFactory
print("Creating combined VectorBT IndicatorFactory for all indicators...")

# Create individual indicator factories for each period to avoid broadcasting issues
def create_individual_indicators():
    """Create individual IndicatorFactory for each indicator period"""
    indicators = {}
    
    # Create RSI indicators individually
    if INDICATOR_CONFIG['RSI']['enabled']:
        for period in INDICATOR_CONFIG['RSI']['periods']:
            def make_rsi_func(p):
                def rsi_func(close):
                    if TALIB_AVAILABLE:
                        # FIXED: Proper array conversion and validation
                        if hasattr(close, 'values'):
                            close_array = close.values.flatten()
                        else:
                            close_array = np.asarray(close, dtype=np.float64).flatten()
                        
                        rsi_values = talib.RSI(close_array, timeperiod=p)
                        
                        # Return as pandas Series with proper index
                        if hasattr(close, 'index'):
                            return pd.Series(rsi_values, index=close.index)
                        else:
                            return rsi_values
                    else:
                        rsi_values = _calculate_rsi_numpy(close.values, p)
                        if rsi_values.ndim > 1:
                            rsi_values = rsi_values.flatten()
                        return pd.Series(rsi_values, index=close.index)
                return rsi_func
            
            indicator_name = f'rsi_{period}'
            indicators[indicator_name] = vbt.IndicatorFactory(
                class_name=f"RSI_{period}",
                short_name=f"rsi{period}",
                input_names=["close"],
                param_names=[],
                output_names=[indicator_name]
            ).from_apply_func(
                make_rsi_func(period),
                keep_pd=True
            )
            print(f"✅ Created individual RSI_{period} factory")
    
    # Create SMA indicators individually
    if INDICATOR_CONFIG['SMA']['enabled']:
        for period in INDICATOR_CONFIG['SMA']['periods']:
            def make_sma_func(p):
                def sma_func(close):
                    if TALIB_AVAILABLE:
                        # FIXED: Proper array conversion and validation
                        if hasattr(close, 'values'):
                            close_array = close.values.flatten()
                        else:
                            close_array = np.asarray(close, dtype=np.float64).flatten()
                        
                        sma_values = talib.SMA(close_array, timeperiod=p)
                        
                        # Return as pandas Series with proper index
                        if hasattr(close, 'index'):
                            return pd.Series(sma_values, index=close.index)
                        else:
                            return sma_values
                    else:
                        sma_values = _calculate_sma_numpy(close.values, p)
                        if sma_values.ndim > 1:
                            sma_values = sma_values.flatten()
                        return pd.Series(sma_values, index=close.index)
                return sma_func
            
            indicator_name = f'ma_{period}'
            indicators[indicator_name] = vbt.IndicatorFactory(
                class_name=f"SMA_{period}",
                short_name=f"sma{period}",
                input_names=["close"],
                param_names=[],
                output_names=[indicator_name]
            ).from_apply_func(
                make_sma_func(period),
                keep_pd=True
            )
            print(f"✅ Created individual SMA_{period} factory")
    
    return indicators

# Create individual indicator factories (no broadcasting issues)
INDIVIDUAL_INDICATORS = create_individual_indicators()

print("✅ VectorBT CombinedIndicators IndicatorFactory created with TAlib implementations")

def evaluate_condition_string(condition_str: str, data_dict: dict) -> np.ndarray:
    """
    Safely evaluate a condition string using numpy arrays from data_dict.
    
    Args:
        condition_str: Condition string like '(close > ma_9) & (ma_9 > ma_20)'
        data_dict: Dictionary with numpy arrays (close, ma_9, ma_20, etc.)
    
    Returns:
        Boolean numpy array representing the condition
    """
    try:
        # Create a safe evaluation environment with only the data arrays
        safe_dict = {
            '__builtins__': {},
            'np': np,
            **data_dict
        }
        
        # Evaluate the condition string
        result = eval(condition_str, safe_dict)
        
        # Ensure result is a boolean numpy array
        if not isinstance(result, np.ndarray):
            result = np.array(result, dtype=bool)
        
        return result.astype(bool)
        
    except Exception as e:
        print(f"⚠️  Error evaluating condition '{condition_str}': {e}")
        # Return all False as fallback
        return np.zeros(len(data_dict['close']), dtype=bool)

def strategy_signals_func(close, *indicators, strategy_name='multi_sma_trend_put'):
    """Enhanced strategy function with dynamic indicator support"""
    
    # Convert to numpy for faster processing - FIXED: Flatten to 1D array
    close_np = close.to_numpy().flatten()
    signals = np.zeros_like(close_np, dtype=int)
    
    # Create indicator dictionary for easy access
    indicator_dict = {}
    indicator_names = []
    
    # Dynamically map indicators based on configuration
    if INDICATOR_CONFIG['RSI']['enabled']:
        indicator_names.extend([f'rsi_{p}' for p in INDICATOR_CONFIG['RSI']['periods']])
    if INDICATOR_CONFIG['SMA']['enabled']:
        indicator_names.extend([f'ma_{p}' for p in INDICATOR_CONFIG['SMA']['periods']])
    if INDICATOR_CONFIG['EMA']['enabled']:
        indicator_names.extend([f'ema_{p}' for p in INDICATOR_CONFIG['EMA']['periods']])
    
    # Map indicators to dictionary for strategy logic - FIXED: Flatten all arrays to 1D
    for i, name in enumerate(indicator_names):
        if i < len(indicators):
            # Handle both pandas Series and numpy arrays - ENSURE 1D ARRAYS
            if hasattr(indicators[i], 'to_numpy'):
                indicator_dict[name] = indicators[i].to_numpy().flatten()
            elif hasattr(indicators[i], 'values'):
                values = indicators[i].values
                indicator_dict[name] = values.flatten() if values.ndim > 1 else values
            else:
                array_val = np.asarray(indicators[i])
                indicator_dict[name] = array_val.flatten() if array_val.ndim > 1 else array_val
    
    # Helper function to safely get indicator or return default
    def get_indicator(name, default=None):
        if name in indicator_dict:
            return indicator_dict[name]
        if default is not None:
            return np.full_like(close_np, default)
        return close_np  # Fallback to close prices
    
    # Get strategy configuration from SIGNAL_STRATEGIES
    if strategy_name not in SIGNAL_STRATEGIES:
        print(f"⚠️  Strategy '{strategy_name}' not found, using default sma_basic")
        strategy_name = 'sma_basic'
    
    strategy_config = SIGNAL_STRATEGIES[strategy_name]
    
    # Create data dictionary for condition evaluation
    data_dict = {
        'close': close_np,
        **indicator_dict  # Include all indicators
    }
    
    # Dynamically evaluate entry and exit conditions from strategy configuration
    try:
        entry_condition_str = strategy_config['entry']
        exit_condition_str = strategy_config['exit']
        
        # Evaluate conditions using the data dictionary
        entry_condition = evaluate_condition_string(entry_condition_str, data_dict)
        exit_condition = evaluate_condition_string(exit_condition_str, data_dict)
        
    except Exception as e:
        print(f"⚠️  Error evaluating strategy conditions for '{strategy_name}': {e}")
        # Fallback to basic SMA strategy
        ma20_np = get_indicator('ma_20')
        entry_condition = close_np > ma20_np
        exit_condition = close_np < ma20_np
    
    
    # VECTORIZED POSITION TRACKING WITH FORCED EXITS AT 15:15
    # Create time masks for trading hours validation and forced exits
    trading_hours_mask, forced_exit_mask = create_time_masks(close.index)
    
    # Apply vectorized position tracking with time validation
    signals = vectorized_position_tracking(
        entry_condition, 
        exit_condition, 
        trading_hours_mask, 
        forced_exit_mask
    )
    
    return signals

def create_dynamic_strategy_indicator():
    """Create StrategyIndicator with dynamic input names based on available indicators"""
    
    # Get all available indicators dynamically
    input_names = ["close"]  # Always include close
    
    # Add RSI indicators if enabled
    if INDICATOR_CONFIG['RSI']['enabled']:
        input_names.extend([f'rsi_{p}' for p in INDICATOR_CONFIG['RSI']['periods']])
    
    # Add SMA indicators if enabled  
    if INDICATOR_CONFIG['SMA']['enabled']:
        input_names.extend([f'ma_{p}' for p in INDICATOR_CONFIG['SMA']['periods']])
        
    # Add EMA indicators if enabled
    if INDICATOR_CONFIG['EMA']['enabled']:
        input_names.extend([f'ema_{p}' for p in INDICATOR_CONFIG['EMA']['periods']])
    
    print(f"🔧 Creating dynamic StrategyIndicator with inputs: {input_names}")
    
    return vbt.IndicatorFactory(
        class_name="StrategySignals",
        short_name="strategy", 
        input_names=input_names,
        param_names=["strategy_name"],
        output_names=["signals"]
    ).from_apply_func(
        strategy_signals_func,
        strategy_name=ACTIVE_STRATEGY,
        keep_pd=True
    )

# Create dynamic indicator at module level
StrategyIndicator = create_dynamic_strategy_indicator()

print("✅ VectorBT StrategyIndicator IndicatorFactory created with dynamic indicator support")

def evaluate_signal_condition(condition_str: str, data_dict: dict) -> pd.Series:
    """
    Safely evaluate signal condition string using available data.
    
    Args:
        condition_str: String condition like 'close > ma_20'
        data_dict: Dictionary with data series (close, rsi_14, ma_20, etc.)
        
    Returns:
        pandas Series with boolean values
    """
    try:
        # Create local namespace with available data
        local_vars = data_dict.copy()
        
        # Evaluate the condition string
        result = eval(condition_str, {"__builtins__": {}}, local_vars)
        
        # Ensure result is a pandas Series
        if isinstance(result, pd.Series):
            return result
        else:
            # Convert to Series if it's a scalar or array
            return pd.Series(result, index=data_dict['close'].index)
            
    except Exception as e:
        print(f"❌ Error evaluating condition '{condition_str}': {e}")
        # Return False series as fallback
        return pd.Series(False, index=data_dict['close'].index)

def filter_consecutive_signals(signal_series):
    """Remove consecutive duplicate signals using pure NumPy vectorization (no for loops)"""
    signal_np = signal_series.values.astype(bool)
    
    # Use numpy.diff to find signal changes efficiently
    # diff() finds where signal changes from False to True
    signal_changes = np.diff(signal_np.astype(int), prepend=0)
    
    # Keep only rising edges (False to True transitions)
    filtered = signal_changes == 1
    
    return pd.Series(filtered, index=signal_series.index)

def filter_consecutive_signals_v2(signal_series):
    """Ultra-fast signal filtering using numpy roll operations"""
    signal_np = signal_series.values.astype(bool)
    
    # Create shifted array for comparison
    prev_signal = np.roll(signal_np, 1)
    prev_signal[0] = False  # Set first element
    
    # Keep signals where current=True AND previous=False
    filtered = signal_np & ~prev_signal
    
    return pd.Series(filtered, index=signal_series.index)

def compare_filtering_methods(signal_series):
    """Compare performance of different filtering methods"""
    import time
    
    # Method 1: Pure NumPy diff
    start = time.time()
    result1 = filter_consecutive_signals(signal_series)
    time1 = time.time() - start
    
    # Method 2: NumPy roll & boolean ops
    start = time.time() 
    result2 = filter_consecutive_signals_v2(signal_series)
    time2 = time.time() - start
    
    print(f"🚀 NumPy diff method: {time1:.6f}s")
    print(f"🚀 NumPy roll method: {time2:.6f}s")
    
    # Use the faster method
    if time2 < time1:
        print("✅ Using NumPy roll method (faster)")
        return result2
    else:
        print("✅ Using NumPy diff method (faster)")
        return result1

def create_time_masks(timestamps):
    """Create vectorized time validation masks using NumPy operations"""
    try:
        # Convert trading hour strings to time objects
        trade_start = dt_time(*map(int, TRADE_START_TIME.split(':')))
        trade_end = dt_time(*map(int, TRADE_END_TIME.split(':'))) 
        forced_exit = dt_time(*map(int, FORCED_EXIT_TIME.split(':')))
        
        # Extract time component from timestamps vectorized
        times = np.array([ts.time() for ts in timestamps])
        
        # Create boolean masks for time validation
        trading_hours_mask = (times >= trade_start) & (times <= trade_end)
        forced_exit_mask = times >= forced_exit
        
        return trading_hours_mask, forced_exit_mask
        
    except Exception as e:
        print(f"❌ Error creating time masks: {e}")
        # Fallback to all True masks
        n = len(timestamps)
        return np.ones(n, dtype=bool), np.zeros(n, dtype=bool)

def vectorized_position_tracking(entry_conditions, exit_conditions, trading_hours_mask, forced_exit_mask):
    """Fixed position tracking with proper state management for multiple entries"""
    try:
        n = len(entry_conditions)
        signals = np.zeros(n, dtype=int)
        
        # Filter entries by trading hours (09:20-15:00)
        valid_entries = entry_conditions & trading_hours_mask
        
        # Combine natural exits with forced exits at 15:15
        all_exits = exit_conditions | forced_exit_mask
        
        # CRITICAL FIX: Sequential position state tracking for multiple entries
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


class VectorBTIndicatorSignalChecker:
    """
    VectorBT-powered indicator signal checking system with TAlib integration.
    
    Leverages VectorBT's IndicatorFactory with industry-standard TAlib RSI and SMA implementations
    for superior performance, reliability, and consistency with professional trading platforms.
    """
    
    def __init__(self):
        """Initialize the signal checker with options trading capabilities."""
        self.logger = get_logger(__name__)
        self.database = DatabaseManager()
        
        # Initialize options components if available
        self.options_enabled = OPTIONS_IMPORTS_AVAILABLE and USE_OPTIONS_PORTFOLIO
        if self.options_enabled:
            try:
                self.options_db = OptionsDatabase()
                self.strike_selector = AdvancedStrikeSelector(self.options_db)
                self.options_calculator = OptionsPnLCalculator()
                self.logger.info("Options trading components initialized successfully")
            except Exception as e:
                self.logger.warning(f"Failed to initialize options components: {e}")
                self.options_enabled = False
        else:
            self.options_db = None
            self.strike_selector = None
            self.options_calculator = None
        
        # VectorBT settings for optimal performance
        self._configure_vectorbt()
        
        # Data storage
        self.ohlc_data = None
        self.indicators = {}
        
        mode = "Options" if self.options_enabled else "Equity"
        self.logger.info(f"VectorBT Indicator Signal Checker initialized in {mode} mode")
    
    def _configure_vectorbt(self) -> None:
        """Configure VectorBT settings for optimal performance."""
        try:
            # Set frequency for time series operations
            vbt.settings.array_wrapper['freq'] = VECTORBT_FREQUENCY  # 5-minute frequency
            
            # Portfolio settings for signal evaluation
            vbt.settings.portfolio['init_cash'] = VECTORBT_INIT_CASH  # 10 Lakh initial cash
            vbt.settings.portfolio['fees'] = 0.0003  # 0.03% brokerage
            vbt.settings.portfolio['slippage'] = 0.0001  # 0.01% slippage
            
            self.logger.info(f"VectorBT configured for {DATA_TIMEFRAME} {TRADING_SYMBOL} analysis")
            
        except Exception as e:
            self.logger.error(f"Error configuring VectorBT: {e}")
            raise
    
    def load_nifty_data(
        self, 
        start_date: str = "2024-06-01", 
        end_date: str = "2024-06-30"
    ) -> bool:
        """
        Load OHLC data from database.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            bool: True if data loaded successfully
        """
        try:
            self.logger.info(f"Loading {TRADING_SYMBOL} {DATA_TIMEFRAME} data from {start_date} to {end_date}")
            
            # Load data from database
            raw_data = self.database.get_ohlcv_data(
                symbol=TRADING_SYMBOL,
                exchange=TRADING_EXCHANGE,
                timeframe=DATA_TIMEFRAME,
                start_date=start_date,
                end_date=end_date
            )
            
            if raw_data.empty:
                self.logger.error(f"No {TRADING_SYMBOL} data found in database for specified date range")
                return False
            
            # Prepare data for VectorBT (ensure proper datetime index)
            self.ohlc_data = raw_data.copy()
            
            # Validate OHLC data
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in self.ohlc_data.columns]
            
            if missing_columns:
                self.logger.error(f"Missing required columns: {missing_columns}")
                return False
            
            self.logger.info(f"Loaded {len(self.ohlc_data)} {TRADING_SYMBOL} {DATA_TIMEFRAME} candles")
            self.logger.info(f"Date range: {self.ohlc_data.index[0]} to {self.ohlc_data.index[-1]}")
            self.logger.info(f"Price range: {self.ohlc_data['close'].min():.2f} - {self.ohlc_data['close'].max():.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading {TRADING_SYMBOL} data: {e}")
            return False
    
    def calculate_all_indicators(self) -> Dict[str, Any]:
        """
        Calculate all enabled indicators in one efficient call using CombinedIndicators factory.
        
        Uses single VectorBT IndicatorFactory call to calculate all RSI, SMA, and EMA indicators
        based on INDICATOR_CONFIG settings. Much faster than separate calculations.
        
        Returns:
            Dictionary with all indicator calculation statistics
        """
        try:
            if self.ohlc_data is None:
                raise ValueError("OHLC data not loaded. Call load_nifty_data() first.")
            
            # Check if any indicators are enabled
            any_enabled = (INDICATOR_CONFIG['RSI']['enabled'] or 
                          INDICATOR_CONFIG['SMA']['enabled'] or 
                          INDICATOR_CONFIG['EMA']['enabled'])
            
            if not any_enabled:
                self.logger.info("All indicators disabled in configuration")
                return {}
                
            self.logger.info("Calculating all indicators using combined VectorBT IndicatorFactory...")
            
            # Calculate indicators using individual factories
            stats = {}
            total_calculated = 0
            
            # Calculate all individual indicators
            for indicator_name, factory in INDIVIDUAL_INDICATORS.items():
                try:
                    result = factory.run(self.ohlc_data['close'])
                    indicator_values = getattr(result, indicator_name)
                    self.indicators[indicator_name] = indicator_values
                    
                    # Track statistics
                    valid_count = (~indicator_values.isna()).sum()
                    stats[f'{indicator_name}_count'] = valid_count
                    total_calculated += 1
                    
                    self.logger.info(f"✅ {indicator_name}: {valid_count} valid values calculated")
                except Exception as e:
                    self.logger.error(f"❌ Failed to calculate {indicator_name}: {e}")
            
            # Calculate total expected indicators
            total_expected = (len(INDICATOR_CONFIG['RSI']['periods']) if INDICATOR_CONFIG['RSI']['enabled'] else 0) + \
                           (len(INDICATOR_CONFIG['SMA']['periods']) if INDICATOR_CONFIG['SMA']['enabled'] else 0) + \
                           (len(INDICATOR_CONFIG['EMA']['periods']) if INDICATOR_CONFIG['EMA']['enabled'] else 0)
            
            self.logger.info(f"All indicators calculated successfully: {total_calculated}/{total_expected} indicators")
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Error calculating all indicators: {e}")
            raise
    
    def calculate_portfolio_backtest(self) -> Dict[str, Any]:
        """
        Calculate VectorBT portfolio backtest from configurable signals.
        
        Uses global ACTIVE_STRATEGY configuration for entry/exit conditions.
        
        Returns:
            Dictionary with portfolio performance metrics
        """
        try:
            if self.ohlc_data is None:
                raise ValueError("OHLC data not available. Run load_nifty_data() first.")
            
            # Get active strategy configuration
            strategy = SIGNAL_STRATEGIES[ACTIVE_STRATEGY]
            
            self.logger.info(f"Calculating VectorBT portfolio backtest using strategy: {ACTIVE_STRATEGY}")
            print(f"\n🎯 Using Strategy: {strategy['description']}")
            print(f"   📈 Entry Condition: {strategy['entry']}")
            print(f"   📉 Exit Condition: {strategy['exit']}")
            
            # Prepare data dictionary for signal evaluation
            data_dict = {
                'close': self.ohlc_data['close'],
                'open': self.ohlc_data['open'],
                'high': self.ohlc_data['high'],
                'low': self.ohlc_data['low'],
                'volume': self.ohlc_data['volume']
            }
            
            # Add indicator data if available
            for key, value in self.indicators.items():
                if key not in ['portfolio', 'entries', 'exits']:  # Skip portfolio objects
                    data_dict[key] = value
            
            # Generate signals using VectorBT IndicatorFactory pattern (FAST APPROACH)
            import time
            signal_start = time.time()
            
            print(f"🎯 Using VectorBT IndicatorFactory Pattern (Ultra Fast)")
            
            # Prepare dynamic indicator inputs
            indicator_inputs = [data_dict['close']]
            
            # Add indicators based on configuration
            if INDICATOR_CONFIG['RSI']['enabled']:
                for period in INDICATOR_CONFIG['RSI']['periods']:
                    indicator_name = f'rsi_{period}'
                    if indicator_name in data_dict:
                        indicator_inputs.append(data_dict[indicator_name])
                    else:
                        self.logger.warning(f"Missing indicator: {indicator_name}")
                        
            if INDICATOR_CONFIG['SMA']['enabled']:
                for period in INDICATOR_CONFIG['SMA']['periods']:
                    indicator_name = f'ma_{period}'
                    if indicator_name in data_dict:
                        indicator_inputs.append(data_dict[indicator_name])
                    else:
                        self.logger.warning(f"Missing indicator: {indicator_name}")
                        
            if INDICATOR_CONFIG['EMA']['enabled']:
                for period in INDICATOR_CONFIG['EMA']['periods']:
                    indicator_name = f'ema_{period}'
                    if indicator_name in data_dict:
                        indicator_inputs.append(data_dict[indicator_name])
                    else:
                        self.logger.warning(f"Missing indicator: {indicator_name}")
            
            print(f"🔧 Running strategy with {len(indicator_inputs)} dynamic inputs")
            
            # Run the strategy indicator with dynamic inputs
            result = StrategyIndicator.run(*indicator_inputs, strategy_name=ACTIVE_STRATEGY)
            
            # Extract signals (exact same pattern as your example)
            entries = result.signals == 1.0
            exits = result.signals == -1.0
            
            signal_time = time.time() - signal_start
            
            print(f"🔍 Signal Analysis (IndicatorFactory):")
            print(f"   📈 Entry signals: {entries.sum()} ({entries.mean()*100:.1f}%)")
            print(f"   📉 Exit signals: {exits.sum()} ({exits.mean()*100:.1f}%)")
            
            # Create portfolio using VectorBT (same pattern as your example)
            portfolio_start = time.time()
            
            print("⚡ Creating VectorBT Portfolio (IndicatorFactory Pattern)")
            portfolio = vbt.Portfolio.from_signals(
                data_dict['close'], 
                entries, 
                exits,
                init_cash=VECTORBT_INIT_CASH,
                fees=0.0003,
                slippage=0.0001
            )
            
            portfolio_time = time.time() - portfolio_start

            # Calculate performance metrics using VectorBT (same pattern as your example)
            stats_start = time.time()
            
            try:
                total_return = portfolio.total_return()
                sharpe_ratio = portfolio.sharpe_ratio()
                max_drawdown = portfolio.max_drawdown()
                final_value = portfolio.value().iloc[-1]
                
                stats = {
                    'Total Return [%]': total_return * 100,
                    'Sharpe Ratio': sharpe_ratio,
                    'Max Drawdown [%]': max_drawdown * 100,
                    'End Value': final_value,
                    'Total Trades': len(portfolio.orders.records)
                }
            except Exception as e:
                print(f"⚠️ VectorBT stats calculation failed: {e}")
                stats = {
                    'Total Return [%]': 'N/A',
                    'Sharpe Ratio': 'N/A', 
                    'Max Drawdown [%]': 'N/A',
                    'End Value': 'N/A',
                    'Total Trades': 'N/A'
                }
            
            stats_time = time.time() - stats_start
            
            total_time = signal_time + portfolio_time + stats_time
            
            # Performance timing summary (VectorBT IndicatorFactory Pattern)
            print(f"\n⏱️  Performance Timing (VectorBT IndicatorFactory Pattern):")
            print(f"   📊 Signal Generation (IndicatorFactory): {signal_time:.4f}s")
            print(f"   💼 Portfolio Creation: {portfolio_time:.4f}s")
            print(f"   📈 Stats Calculation: {stats_time:.4f}s")
            print(f"   🎯 Total Time: {total_time:.4f}s")
            
            # Debug: Print calculated stats
            print(f"\n📋 VectorBT Stats calculated: {list(stats.keys())}")
            
            # Store portfolio results (VectorBT IndicatorFactory pattern)
            self.indicators['portfolio'] = portfolio  # VectorBT portfolio object
            self.indicators['entries'] = entries
            self.indicators['exits'] = exits
            self.indicators['strategy_signals'] = result.signals  # Keep full signal array
            
            # Create performance summary with safe key access
            performance_summary = {}
            
            # Safely extract essential metrics
            essential_metrics = [
                ('total_return', 'Total Return [%]'),
                ('sharpe_ratio', 'Sharpe Ratio'),
                ('max_drawdown', 'Max Drawdown [%]'),
                ('final_value', 'End Value'),
                ('total_trades', 'Total Trades')
            ]
            
            for key, stat_key in essential_metrics:
                if stat_key in stats:
                    performance_summary[key] = stats[stat_key]
                else:
                    performance_summary[key] = 'N/A'
            
            # Add calculated values that might not be in stats
            if 'win_rate' not in performance_summary:
                performance_summary['win_rate'] = 'N/A'  # Would need more complex calculation
            if 'profit_factor' not in performance_summary:
                performance_summary['profit_factor'] = 'N/A'  # Would need more complex calculation
            
            print(f"\n📊 Portfolio Performance Summary:")
            
            # Safe printing with N/A handling
            def safe_print_metric(label, value, format_str=None):
                if value == 'N/A':
                    print(f"   {label}: {value}")
                else:
                    if format_str:
                        print(f"   {label}: {format_str.format(value)}")
                    else:
                        print(f"   {label}: {value}")
            
            safe_print_metric("💰 Total Return", performance_summary['total_return'], "{:.2f}%")
            safe_print_metric("📈 Sharpe Ratio", performance_summary['sharpe_ratio'], "{:.2f}")
            safe_print_metric("📉 Max Drawdown", performance_summary['max_drawdown'], "{:.2f}%")
            safe_print_metric("🔄 Total Trades", performance_summary['total_trades'])
            safe_print_metric("🎯 Win Rate", performance_summary['win_rate'], "{:.1f}%")
            safe_print_metric("⚖️  Profit Factor", performance_summary['profit_factor'], "{:.2f}")
            safe_print_metric("💵 Final Value", performance_summary['final_value'], "₹{:,.0f}")
            
            self.logger.info("Portfolio backtest calculated successfully")
            return performance_summary
            
        except Exception as e:
            self.logger.error(f"Error calculating portfolio backtest: {e}")
            raise

    def calculate_options_portfolio_backtest(self, strategy_name: str = None) -> Dict[str, Any]:
        """
        Calculate real options portfolio P&L using actual option chain data.
        
        Returns:
            Dictionary with options portfolio performance metrics
        """
        try:
            if not self.options_enabled:
                self.logger.warning("Options mode not enabled, falling back to equity mode")
                return self.calculate_portfolio_backtest()
            
            if self.ohlc_data is None:
                raise ValueError("OHLC data not available. Run load_nifty_data() first.")
            
            # Use provided strategy or fall back to ACTIVE_STRATEGY
            if strategy_name is None:
                strategy_name = ACTIVE_STRATEGY
            
            # Get strategy configuration
            strategy = SIGNAL_STRATEGIES[strategy_name]
            
            # Get strategy-specific option type
            strategy_option_type = strategy.get('OptionType', 'CALL')
            
            self.logger.info(f"Calculating Real Options Portfolio using strategy: {strategy_name}")
            print(f"\n💰 Real Options Portfolio Calculation")
            print(f"🎯 Strategy: {strategy['description']}")
            print(f"📊 Option Type: {strategy_option_type}")
            print(f"🎯 Strike Type: {DEFAULT_STRIKE_TYPE.value}")
            print(f"📦 Lot Size: {OPTIONS_LOT_SIZE}")
            
            # Prepare data dictionary for signal evaluation
            data_dict = {
                'close': self.ohlc_data['close'],
                'open': self.ohlc_data['open'],
                'high': self.ohlc_data['high'],
                'low': self.ohlc_data['low'],
                'volume': self.ohlc_data['volume']
            }
            
            # Add indicator data if available
            for key, value in self.indicators.items():
                if key not in ['portfolio', 'entries', 'exits', 'options_portfolio']:
                    data_dict[key] = value
            
            # Generate signals using VectorBT IndicatorFactory pattern
            import time
            signal_start = time.time()
            
            print(f"🎯 Generating signals using VectorBT IndicatorFactory...")
            
            # Prepare dynamic indicator inputs for options trading
            indicator_inputs = [data_dict['close']]
            
            # Add indicators based on configuration
            if INDICATOR_CONFIG['RSI']['enabled']:
                for period in INDICATOR_CONFIG['RSI']['periods']:
                    indicator_name = f'rsi_{period}'
                    if indicator_name in data_dict:
                        indicator_inputs.append(data_dict[indicator_name])
                        
            if INDICATOR_CONFIG['SMA']['enabled']:
                for period in INDICATOR_CONFIG['SMA']['periods']:
                    indicator_name = f'ma_{period}'
                    if indicator_name in data_dict:
                        indicator_inputs.append(data_dict[indicator_name])
                        
            if INDICATOR_CONFIG['EMA']['enabled']:
                for period in INDICATOR_CONFIG['EMA']['periods']:
                    indicator_name = f'ema_{period}'
                    if indicator_name in data_dict:
                        indicator_inputs.append(data_dict[indicator_name])
            
            # Run the strategy indicator with dynamic inputs
            result = StrategyIndicator.run(*indicator_inputs, strategy_name=strategy_name)
            
            # Extract entry/exit signals
            entries = result.signals == 1.0
            exits = result.signals == -1.0
            
            signal_time = time.time() - signal_start
            
            print(f"✅ Signals generated in {signal_time:.4f}s")
            print(f"   📈 Entry signals: {entries.sum()}")
            print(f"   📉 Exit signals: {exits.sum()}")
            
            # Calculate options P&L using selected method
            options_start = time.time()
            
            if USE_ULTRA_FAST_PNL:
                print(f"💰 Calculating real options P&L (Ultra-Fast Vectorized)...")
                options_pnl = self._calculate_real_options_pnl_ultra_fast(
                data_dict['close'], entries, exits, data_dict['close'].index, strategy_name
                )
            else:
                print(f"🚀 Calculating real options P&L (VectorBT from_orders)...")
                options_pnl = self._calculate_real_options_pnl_vectorbt_orders(
                data_dict['close'], entries, exits, data_dict['close'].index, strategy_name
                )

            # Use ultra-fast vectorized method
            # print(f"💰 Calculating real options P&L (Ultra-Fast Vectorized)...")
            # options_pnl = self._calculate_real_options_pnl_ultra_fast(
            #     data_dict['close'], entries, exits, data_dict['close'].index
            # )
            options_time = time.time() - options_start
            
            print(f"✅ Options P&L calculated in {options_time:.4f}s")
            
            # Store results
            self.indicators['entries'] = entries
            self.indicators['exits'] = exits
            self.indicators['strategy_signals'] = result.signals
            self.indicators['options_portfolio'] = options_pnl
            
            total_time = signal_time + options_time
            
            print(f"\n⏱️  Performance Timing (Options Portfolio):")
            print(f"   📊 Signal Generation: {signal_time:.4f}s")
            print(f"   💰 Options P&L Calculation: {options_time:.4f}s")
            print(f"   🎯 Total Time: {total_time:.4f}s")
            
            # Display comprehensive portfolio analytics summary
            self.display_portfolio_summary(options_pnl)
            
            # Print compact options performance summary
            print(f"\n📊 QUICK PERFORMANCE SUMMARY:")
            print(f"   💰 Total P&L: ₹{options_pnl['total_pnl']:,.2f}")
            print(f"   📈 Total Return: {options_pnl['total_return_pct']:.2f}%")
            print(f"   🔄 Total Trades: {options_pnl['total_trades']}")
            print(f"   🎯 Win Rate: {options_pnl['win_rate']:.1f}%")
            print(f"   💵 Average P&L per Trade: ₹{options_pnl['avg_pnl_per_trade']:,.2f}")
            
            self.logger.info("Options portfolio backtest calculated successfully")
            return options_pnl
            
        except Exception as e:
            self.logger.error(f"Error calculating options portfolio backtest: {e}")
            raise

    def _calculate_real_options_pnl_ultra_fast(self, close_prices: pd.Series, entries: pd.Series, 
                                             exits: pd.Series, timestamps: pd.Index, strategy_name: str = None) -> Dict[str, Any]:
        """
        Ultra-fast vectorized options P&L calculation using NumPy + batch SQL.
        
        Performance target: <0.1s for 20k+ records (24x faster than loop method)
        
        Args:
            close_prices: Underlying close prices
            entries: Entry signals
            exits: Exit signals  
            timestamps: Timestamps for each data point
            
        Returns:
            Dictionary with options P&L calculations
        """
        try:
            import time
            start_time = time.time()
            
            # Step 1: Vectorized signal extraction (no loops)
            entry_mask = entries.values.astype(bool)
            exit_mask = exits.values.astype(bool)
            
            entry_indices = np.where(entry_mask)[0]
            exit_indices = np.where(exit_mask)[0]
            
            if len(entry_indices) == 0:
                return self._get_empty_pnl_result()
            
            print(f"🚀 Ultra-fast P&L: Processing {len(entry_indices)} entries, {len(exit_indices)} exits")
            
            # Step 2: Vectorized entry/exit matching using NumPy
            trades_data = self._match_entries_exits_vectorized(
                entry_indices, exit_indices, timestamps, close_prices
            )
            
            if not trades_data:
                return self._get_empty_pnl_result()
            
            match_time = time.time()
            print(f"   ⚡ Entry/Exit matching: {(match_time - start_time)*1000:.1f}ms")
            
            # Step 3: Batch premium lookup with single SQL query
            premium_data = self._batch_get_all_premiums_vectorized(trades_data, strategy_name)
            
            premium_time = time.time()
            print(f"   ⚡ Batch premium lookup: {(premium_time - match_time)*1000:.1f}ms")
            
            # Step 4: Vectorized P&L calculations (pure NumPy)
            pnl_results = self._calculate_vectorized_pnl(premium_data)
            
            calc_time = time.time()
            print(f"   ⚡ Vectorized P&L calc: {(calc_time - premium_time)*1000:.1f}ms")
            
            total_time = calc_time - start_time
            print(f"🎯 Ultra-fast P&L completed in {total_time*1000:.1f}ms")
            
            return pnl_results
            
        except Exception as e:
            self.logger.error(f"Error in ultra-fast options P&L calculation: {e}")
            # Fallback to original method if ultra-fast fails
            return self._calculate_real_options_pnl(close_prices, entries, exits, timestamps)

    def _match_entries_exits_vectorized(self, entry_indices: np.ndarray, exit_indices: np.ndarray,
                                       timestamps: pd.Index, close_prices: pd.Series) -> List[Dict]:
        """
        Vectorized entry/exit matching using NumPy operations.
        
        Returns:
            List of trade dictionaries with entry/exit pairs
        """
        try:
            trades = []
            
            # Use searchsorted for fast matching
            in_position = False
            current_entry_idx = None
            
            # Combine and sort all signal indices
            all_signals = []
            for idx in entry_indices:
                all_signals.append((idx, 'entry'))
            for idx in exit_indices:
                all_signals.append((idx, 'exit'))
            
            # Sort by timestamp index
            all_signals.sort(key=lambda x: x[0])
            
            for signal_idx, signal_type in all_signals:
                if signal_type == 'entry' and not in_position:
                    current_entry_idx = signal_idx
                    in_position = True
                    
                elif signal_type == 'exit' and in_position and current_entry_idx is not None:
                    # Create trade pair
                    entry_timestamp = timestamps[current_entry_idx]
                    exit_timestamp = timestamps[signal_idx]
                    entry_price = close_prices.iloc[current_entry_idx]
                    exit_price = close_prices.iloc[signal_idx]
                    
                    trades.append({
                        'entry_idx': current_entry_idx,
                        'exit_idx': signal_idx,
                        'entry_timestamp': entry_timestamp,
                        'exit_timestamp': exit_timestamp,
                        'entry_price': entry_price,
                        'exit_price': exit_price
                    })
                    
                    # Reset position
                    in_position = False
                    current_entry_idx = None
            
            return trades
            
        except Exception as e:
            self.logger.error(f"Error in vectorized entry/exit matching: {e}")
            return []

    def _batch_get_all_premiums_vectorized(self, trades_data: List[Dict], strategy_name: str = None) -> List[Dict]:
        """
        Batch premium lookup using single SQL JOIN query for maximum performance.
        
        Args:
            trades_data: List of trade dictionaries
            
        Returns:
            List of trades with premium data added
        """
        try:
            if not trades_data:
                return []
            
            # Extract all unique timestamps and prices for batch lookup
            timestamps_prices = []
            for trade in trades_data:
                timestamps_prices.append((trade['entry_timestamp'], trade['entry_price'], 'entry'))
                timestamps_prices.append((trade['exit_timestamp'], trade['exit_price'], 'exit'))
            
            # Get premium data in batch
            premium_lookup = self._execute_batch_premium_query(timestamps_prices, strategy_name)
            
            # Attach premium data to trades
            enriched_trades = []
            for trade in trades_data:
                entry_key = (trade['entry_timestamp'], trade['entry_price'])
                exit_key = (trade['exit_timestamp'], trade['exit_price'])
                
                entry_premium = premium_lookup.get(entry_key, {})
                exit_premium = premium_lookup.get(exit_key, {})
                
                if entry_premium and exit_premium:
                    trade.update({
                        'entry_strike': entry_premium.get('strike', 0),
                        'entry_premium': entry_premium.get('premium', 0),
                        'exit_premium': exit_premium.get('premium', 0),
                        'entry_iv': entry_premium.get('iv', 0),
                        'exit_iv': exit_premium.get('iv', 0)
                    })
                    enriched_trades.append(trade)
            
            return enriched_trades
            
        except Exception as e:
            self.logger.error(f"Error in batch premium lookup: {e}")
            return trades_data

    def _execute_batch_premium_query(self, timestamps_prices: List[Tuple], strategy_name: str = None) -> Dict:
        """
        Execute single SQL query to get all premiums at once.
        
        Args:
            timestamps_prices: List of (timestamp, price, type) tuples
            
        Returns:
            Dictionary mapping (timestamp, price) -> premium data
        """
        try:
            if not timestamps_prices:
                return {}
            
            # Determine option type columns based on strategy-specific OptionType
            option_type_prefix = "call"  # Default fallback
            
            if strategy_name and strategy_name in SIGNAL_STRATEGIES:
                # Use strategy-specific option type
                strategy_option_type = SIGNAL_STRATEGIES[strategy_name].get('OptionType', 'CALL')
                option_type_prefix = strategy_option_type.lower()
            elif DEFAULT_OPTION_TYPE:
                # Fallback to global default
                option_type_prefix = DEFAULT_OPTION_TYPE.value.lower()
            
            print(f"🎯 Using {option_type_prefix.upper()} option type for strategy: {strategy_name or 'default'}")
            
            # Create temporary table with signal data
            signal_data = []
            for ts, price, trade_type in timestamps_prices:
                # Calculate ATM strike (round to nearest 50)
                atm_strike = round(price / 50) * 50
                signal_data.append({
                    'timestamp': ts,
                    'underlying_price': price,
                    'atm_strike': atm_strike,
                    'trade_type': trade_type
                })
            
            # Convert to DataFrame for SQL operations
            signals_df = pd.DataFrame(signal_data)
            
            if signals_df.empty:
                return {}
            
            # Use DuckDB's efficient JOIN with dynamic option type columns
            query = f"""
            SELECT 
                s.timestamp,
                s.underlying_price,
                s.atm_strike,
                o.{option_type_prefix}_ltp as premium,
                o.{option_type_prefix}_iv as iv,
                o.{option_type_prefix}_delta as delta
            FROM signals_df s
            LEFT JOIN {self.options_db.options_table} o 
                ON s.atm_strike = o.strike 
                AND s.timestamp = o.timestamp
            WHERE o.{option_type_prefix}_ltp IS NOT NULL AND o.{option_type_prefix}_ltp > 0
            """
            
            # Register DataFrame and execute query
            self.options_db.connection.register('signals_df', signals_df)
            result_df = self.options_db.connection.execute(query).fetchdf()
            
            # Build lookup dictionary
            premium_lookup = {}
            for _, row in result_df.iterrows():
                key = (row['timestamp'], row['underlying_price'])
                premium_lookup[key] = {
                    'strike': row['atm_strike'],
                    'premium': row['premium'],
                    'iv': row.get('iv', 0),
                    'delta': row.get('delta', 0)
                }
            
            return premium_lookup
            
        except Exception as e:
            self.logger.error(f"Error in batch premium SQL query: {e}")
            return {}

    def _calculate_vectorized_pnl(self, enriched_trades: List[Dict]) -> Dict[str, Any]:
        """
        Vectorized P&L calculations using pure NumPy operations.
        
        Args:
            enriched_trades: List of trades with premium data
            
        Returns:
            Dictionary with performance metrics
        """
        try:
            if not enriched_trades:
                return self._get_empty_pnl_result()
            
            # Convert to NumPy arrays for vectorized operations
            entry_premiums = np.array([trade['entry_premium'] for trade in enriched_trades])
            exit_premiums = np.array([trade['exit_premium'] for trade in enriched_trades])
            
            # Apply slippage to premiums (vectorized)
            # Entry slippage: increase premium (worse for buyer)
            entry_premiums_with_slippage = entry_premiums * (1 + OPTIONS_SLIPPAGE / 100)
            # Exit slippage: decrease premium (worse for seller)
            exit_premiums_with_slippage = exit_premiums * (1 - OPTIONS_SLIPPAGE / 100)
            
            # Calculate slippage amounts for tracking
            entry_slippage_array = entry_premiums * (OPTIONS_SLIPPAGE / 100)
            exit_slippage_array = exit_premiums * (OPTIONS_SLIPPAGE / 100)
            total_slippage_array = (entry_slippage_array + exit_slippage_array) * OPTIONS_LOT_SIZE
            
            # Vectorized P&L calculations with slippage
            gross_pnl_array = (exit_premiums_with_slippage - entry_premiums_with_slippage) * OPTIONS_LOT_SIZE
            commission_array = np.full_like(gross_pnl_array, OPTIONS_COMMISSION)
            net_pnl_array = gross_pnl_array - commission_array
            
            # Vectorized statistics
            total_pnl = np.sum(net_pnl_array)
            winning_trades = np.sum(net_pnl_array > 0)
            losing_trades = np.sum(net_pnl_array < 0)
            total_trades = len(enriched_trades)
            
            win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
            avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0
            
            # Calculate return percentage using configured capital
            initial_capital = INITIAL_CAPITAL  # Use configured capital (₹2 Lakh)
            total_return_pct = (total_pnl / initial_capital * 100) if initial_capital > 0 else 0
            
            # =============================================================================
            # COMPREHENSIVE PORTFOLIO ANALYTICS CALCULATIONS
            # =============================================================================
            
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
            
            # 4. STREAK ANALYSIS
            trade_results = (net_pnl_array > 0).astype(int)  # 1 for win, 0 for loss
            
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
            
            # 5. DRAWDOWN ANALYSIS
            cumulative_pnl = np.cumsum(net_pnl_array)
            portfolio_value = initial_capital + cumulative_pnl
            
            # Calculate running maximum portfolio value
            running_max = np.maximum.accumulate(portfolio_value)
            
            # Calculate drawdown at each point
            drawdown_array = (portfolio_value - running_max) / running_max * 100
            
            # Find maximum drawdown
            max_drawdown_pct = float(abs(np.min(drawdown_array))) if len(drawdown_array) > 0 else 0.0
            max_drawdown_idx = np.argmin(drawdown_array) if len(drawdown_array) > 0 else 0
            max_drawdown_amount = float(abs(portfolio_value[max_drawdown_idx] - running_max[max_drawdown_idx])) if len(drawdown_array) > 0 else 0.0
            
            # Final portfolio value
            final_portfolio_value = float(initial_capital + total_pnl)
            
            # 6. RISK-ADJUSTED METRICS
            if len(net_pnl_array) > 1:
                # Calculate daily returns for risk metrics
                trade_returns = net_pnl_array / initial_capital  # Returns as fraction of capital
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
            
            # 7. POSITION SIZING ANALYTICS (Fixed position sizing)
            avg_position_size = float(FIXED_POSITION_SIZE)  # Always 1 lot
            max_position_size = float(FIXED_POSITION_SIZE)  # Always 1 lot
            
            # Capital efficiency calculation
            avg_capital_per_trade = avg_position_size * OPTIONS_LOT_SIZE * np.mean(entry_premiums) if len(entry_premiums) > 0 else 0.0
            capital_efficiency = float((avg_capital_per_trade / initial_capital) * 100) if initial_capital > 0 else 0.0
            
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
                    'position_size': FIXED_POSITION_SIZE
                })
            
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
                'initial_capital': float(initial_capital),
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
                'commission_per_trade': OPTIONS_COMMISSION,
                'lot_size': OPTIONS_LOT_SIZE,
                'calculation_method': 'comprehensive_vectorized',
                
                # Trading costs breakdown
                'total_commission': float(np.sum(commission_array)),
                'total_slippage': float(np.sum(total_slippage_array)),
                'slippage_percentage': OPTIONS_SLIPPAGE,
                'avg_slippage_per_trade': float(np.mean(total_slippage_array)) if len(total_slippage_array) > 0 else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error in vectorized P&L calculations: {e}")
            return self._get_empty_pnl_result()

    def _get_empty_pnl_result(self) -> Dict[str, Any]:
        """Return empty P&L result structure."""
        return {
            'total_pnl': 0.0,
            'total_return_pct': 0.0,
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'win_rate': 0.0,
            'avg_pnl_per_trade': 0.0,
            'trades': [],
            'commission_per_trade': OPTIONS_COMMISSION,
            'lot_size': OPTIONS_LOT_SIZE,
            'calculation_method': 'ultra_fast_vectorized'
        }

    def _calculate_real_options_pnl_vectorbt_orders(self, close_prices: pd.Series, entries: pd.Series, 
                                                   exits: pd.Series, timestamps: pd.Index, strategy_name: str = None) -> Dict[str, Any]:
        """
        VectorBT from_orders() method for maximum performance using native VectorBT engine.
        
        This method leverages VectorBT's optimized C++ portfolio engine with option premiums.
        
        Args:
            close_prices: Underlying close prices
            entries: Entry signals
            exits: Exit signals  
            timestamps: Timestamps for each data point
            
        Returns:
            Dictionary with options P&L calculations
        """
        try:
            import time
            start_time = time.time()
            
            print(f"🚀 VectorBT from_orders() method: Processing signals...")
            
            # Step 1: Get all option premiums for the entire dataset
            premium_prices = self._get_full_premium_series_vectorized(close_prices, timestamps, strategy_name)
            
            # Step 2: Create position sizes for VectorBT from entry/exit signals
            # Positive for entries (buy), negative for exits (sell)
            position_sizes = entries.astype(float) * OPTIONS_LOT_SIZE - exits.astype(float) * OPTIONS_LOT_SIZE
            print(f"   📊 Position sizes created: {(position_sizes != 0).sum()} non-zero positions")
            
            # Step 3: Use VectorBT's optimized portfolio engine
            portfolio = vbt.Portfolio.from_orders(
                close=premium_prices,  # Use option premiums as price feed
                size=position_sizes,  # Position sizes (positive for buy, negative for sell)
                fees=OPTIONS_COMMISSION,  # Commission per trade
                freq='5T',  # 5-minute frequency
                init_cash=VECTORBT_INIT_CASH  # Initial cash amount
            )
            
            print(portfolio.stats().to_string())
            # Step 4: Extract performance metrics using VectorBT's native methods
            try:
                total_return = portfolio.total_return()
                sharpe_ratio = portfolio.sharpe_ratio()
                max_drawdown = portfolio.max_drawdown()
                final_value = portfolio.value().iloc[-1]
                total_trades = len(portfolio.orders.records)
                
                # Calculate additional metrics
                returns = portfolio.returns()
                winning_trades = (returns > 0).sum()
                losing_trades = (returns < 0).sum()
                win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
                
                pnl_time = time.time() - start_time
                print(f"🎯 VectorBT from_orders() completed in {pnl_time*1000:.1f}ms")
                
                return {
                    'total_pnl': float(final_value - INITIAL_CAPITAL),  # Assuming 1L initial capital
                    'total_return_pct': float(total_return * 100),
                    'total_trades': int(total_trades),
                    'winning_trades': int(winning_trades),
                    'losing_trades': int(losing_trades),
                    'win_rate': float(win_rate),
                    'avg_pnl_per_trade': float((final_value - INITIAL_CAPITAL) / total_trades) if total_trades > 0 else 0,
                    'sharpe_ratio': float(sharpe_ratio),
                    'max_drawdown': float(max_drawdown * 100),
                    'trades': [],  # VectorBT handles trade details internally
                    'commission_per_trade': OPTIONS_COMMISSION,
                    'lot_size': OPTIONS_LOT_SIZE,
                    'calculation_method': 'vectorbt_from_orders'
                }
                
            except Exception as e:
                self.logger.warning(f"VectorBT stats calculation failed: {e}")
                # Fallback to basic calculations
                return self._get_empty_pnl_result()
            
        except Exception as e:
            self.logger.error(f"Error in VectorBT from_orders() method: {e}")
            # Fallback to ultra-fast method
            return self._calculate_real_options_pnl_ultra_fast(close_prices, entries, exits, timestamps)

    def _get_full_premium_series_vectorized(self, close_prices: pd.Series, timestamps: pd.Index, strategy_name: str = None) -> pd.Series:
        """
        Get full option premium series aligned with close prices for VectorBT.
        
        Args:
            close_prices: Underlying close prices
            timestamps: Timestamps for each data point
            
        Returns:
            Series with option premiums aligned to timestamps
        """
        try:
            # Create DataFrame for batch lookup
            data_for_lookup = pd.DataFrame({
                'timestamp': timestamps,
                'underlying_price': close_prices,
                'atm_strike': (close_prices / 50).round() * 50  # Calculate ATM strikes
            })
            
            # Determine option type columns based on strategy-specific OptionType
            option_type_prefix = "call"  # Default fallback
            
            if strategy_name and strategy_name in SIGNAL_STRATEGIES:
                # Use strategy-specific option type
                strategy_option_type = SIGNAL_STRATEGIES[strategy_name].get('OptionType', 'CALL')
                option_type_prefix = strategy_option_type.lower()
            elif DEFAULT_OPTION_TYPE:
                # Fallback to global default
                option_type_prefix = DEFAULT_OPTION_TYPE.value.lower()
            
            # Use batch SQL query to get all premiums
            query = f"""
            SELECT 
                d.timestamp,
                COALESCE(o.{option_type_prefix}_ltp, 100.0) as premium
            FROM data_for_lookup d
            LEFT JOIN {self.options_db.options_table} o 
                ON d.atm_strike = o.strike 
                AND d.timestamp = o.timestamp
            ORDER BY d.timestamp
            """
            
            # Register DataFrame and execute query
            self.options_db.connection.register('data_for_lookup', data_for_lookup)
            result_df = self.options_db.connection.execute(query).fetchdf()
            
            # Create premium series aligned with original index
            premium_series = pd.Series(
                data=result_df['premium'].values,
                index=timestamps,
                name='option_premium'
            )
            
            return premium_series
            
        except Exception as e:
            self.logger.error(f"Error getting full premium series: {e}")
            # Fallback: use close prices as approximation
            return close_prices / 100  # Simple approximation

    def save_data_to_csv(self, strategy_name: str = None) -> bool:
        """Save all calculated data to ONE comprehensive CSV file in output/vectorbt directory."""
        try:
            import os
            
            # Ensure output directory exists
            output_dir = OUTPUT_DIRECTORY
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create ONE comprehensive DataFrame with all data
            if self.ohlc_data is not None and self.indicators:
                # Start with OHLC data
                comprehensive_df = self.ohlc_data.copy()
                
                # Add RSI indicators dynamically
                if INDICATOR_CONFIG['RSI']['enabled']:
                    for period in INDICATOR_CONFIG['RSI']['periods']:
                        indicator_key = f'rsi_{period}'
                        if indicator_key in self.indicators:
                            comprehensive_df[f'RSI_{period}'] = self.indicators[indicator_key]
                
                # Add SMA indicators dynamically
                if INDICATOR_CONFIG['SMA']['enabled']:
                    for period in INDICATOR_CONFIG['SMA']['periods']:
                        indicator_key = f'ma_{period}'
                        if indicator_key in self.indicators:
                            comprehensive_df[f'MA_{period}'] = self.indicators[indicator_key]
                            
                # Add EMA indicators dynamically
                if INDICATOR_CONFIG['EMA']['enabled']:
                    for period in INDICATOR_CONFIG['EMA']['periods']:
                        indicator_key = f'ema_{period}'
                        if indicator_key in self.indicators:
                            comprehensive_df[f'EMA_{period}'] = self.indicators[indicator_key]
                
                # Add basic analysis columns (boolean values for better readability)
                ##comprehensive_df['RSI_14_Overbought'] = (comprehensive_df['RSI_14'] > 70)
                ##comprehensive_df['RSI_14_Oversold'] = (comprehensive_df['RSI_14'] < 30)
                ##comprehensive_df['Price_Above_MA_9'] = (comprehensive_df['close'] > comprehensive_df['MA_9'])
                ##comprehensive_df['Price_Above_MA_20'] = (comprehensive_df['close'] > comprehensive_df['MA_20'])
                
                # Add portfolio signals (VectorBT IndicatorFactory pattern)
                if 'entries' in self.indicators:
                    comprehensive_df['Entry_Signal'] = self.indicators['entries']
                if 'exits' in self.indicators:
                    comprehensive_df['Exit_Signal'] = self.indicators['exits']
                
                # Add strategy signals array for analysis
                if 'strategy_signals' in self.indicators:
                    comprehensive_df['Strategy_Signals'] = self.indicators['strategy_signals']
                    
                    # Add human-readable version of strategy signals
                    comprehensive_df['Strategy_Signal_Text'] = comprehensive_df['Strategy_Signals'].map({
                        1: 'Entry',
                        -1: 'Exit',
                        0: 'Hold'
                    })
                
                # Add options portfolio data if available
                if self.options_enabled and 'options_portfolio' in self.indicators:
                    options_data = self.indicators['options_portfolio']
                    
                    # Initialize options columns with default values
                    comprehensive_df['Strike_Price'] = 0
                    comprehensive_df['Entry_Premium'] = 0.0
                    comprehensive_df['Exit_Premium'] = 0.0
                    comprehensive_df['Options_PnL'] = 0.0
                    comprehensive_df['Option_Type'] = DEFAULT_OPTION_TYPE.value if DEFAULT_OPTION_TYPE else 'CALL'
                    comprehensive_df['Commission'] = 0.0
                    comprehensive_df['Net_PnL'] = 0.0
                    comprehensive_df['Trade_Status'] = 'Hold'
                    
                    # Fill in actual trade data
                    if 'trades' in options_data:
                        for trade in options_data['trades']:
                            entry_time = trade['entry_time']
                            exit_time = trade['exit_time']
                            
                            # Mark entry point
                            if entry_time in comprehensive_df.index:
                                comprehensive_df.loc[entry_time, 'Strike_Price'] = trade['strike']
                                comprehensive_df.loc[entry_time, 'Entry_Premium'] = trade['entry_premium']
                                comprehensive_df.loc[entry_time, 'Commission'] = trade['commission']
                                comprehensive_df.loc[entry_time, 'Trade_Status'] = 'Entry'
                            
                            # Mark exit point
                            if exit_time in comprehensive_df.index:
                                comprehensive_df.loc[exit_time, 'Strike_Price'] = trade['strike']
                                comprehensive_df.loc[exit_time, 'Exit_Premium'] = trade['exit_premium']
                                comprehensive_df.loc[exit_time, 'Options_PnL'] = trade['gross_pnl']
                                comprehensive_df.loc[exit_time, 'Commission'] = trade['commission']
                                comprehensive_df.loc[exit_time, 'Net_PnL'] = trade['net_pnl']
                                comprehensive_df.loc[exit_time, 'Trade_Status'] = 'Exit'
                    
                    print(f"✅ Options portfolio data added to CSV")
                    print(f"   💰 Total Options P&L: ₹{options_data.get('total_pnl', 0):,.2f}")
                    print(f"   🔄 Total Options Trades: {options_data.get('total_trades', 0)}")
                
                # Save the comprehensive CSV file with strategy name
                strategy_suffix = f"_{strategy_name}" if strategy_name else ""
                comprehensive_file = f"{output_dir}/nifty_5m_complete_analysis{strategy_suffix}_{timestamp}.csv"
                comprehensive_df.to_csv(comprehensive_file)
                
                print(f"✅ Complete analysis data saved to: {comprehensive_file}")
                print(f"📊 Columns included: {len(comprehensive_df.columns)} total columns")
                print(f"📈 Data points: {len(comprehensive_df)} rows")
                
                return True
            
            else:
                print("❌ No data available to save")
                return False
                
        except Exception as e:
            self.logger.error(f"Error saving data to CSV: {e}")
            print(f"❌ Error saving data to CSV: {e}")
            return False
    
    def display_portfolio_summary(self, options_pnl: Dict[str, Any]) -> None:
        """
        Display comprehensive portfolio analytics summary matching institutional format.
        
        Args:
            options_pnl: Dictionary containing comprehensive portfolio performance metrics
        """
        try:
            print(f"\n{'=' * 80}")
            print(f"📈 PORTFOLIO ANALYTICS SUMMARY - CAPITAL MANAGEMENT")
            print(f"{'=' * 80}")
            
            # Extract comprehensive metrics
            initial_capital = options_pnl.get('initial_capital', INITIAL_CAPITAL)
            final_portfolio = options_pnl.get('final_portfolio_value', initial_capital)
            total_pnl = options_pnl.get('total_pnl', 0)
            total_return_pct = options_pnl.get('total_return_pct', 0)
            
            # Extract trading costs
            total_commission = options_pnl.get('total_commission', 0)
            total_slippage = options_pnl.get('total_slippage', 0)
            slippage_percentage = options_pnl.get('slippage_percentage', OPTIONS_SLIPPAGE)
            
            # Risk metrics
            max_drawdown_pct = options_pnl.get('max_drawdown_pct', 0)
            max_drawdown_amount = options_pnl.get('max_drawdown_amount', 0)
            sharpe_ratio = options_pnl.get('sharpe_ratio', 0)
            sortino_ratio = options_pnl.get('sortino_ratio', 0)
            recovery_factor = options_pnl.get('recovery_factor', 0)
            calmar_ratio = options_pnl.get('calmar_ratio', 0)
            
            # Trade statistics
            total_trades = options_pnl.get('total_trades', 0)
            winning_trades = options_pnl.get('winning_trades', 0)
            losing_trades = options_pnl.get('losing_trades', 0)
            win_rate = options_pnl.get('win_rate', 0)
            profit_factor = options_pnl.get('profit_factor', 0)
            win_loss_ratio = options_pnl.get('win_loss_ratio', 0)
            
            # Trade extremes
            best_trade = options_pnl.get('best_trade', 0)
            worst_trade = options_pnl.get('worst_trade', 0)
            avg_winning_trade = options_pnl.get('avg_winning_trade', 0)
            avg_losing_trade = options_pnl.get('avg_losing_trade', 0)
            max_win_streak = options_pnl.get('max_win_streak', 0)
            max_loss_streak = options_pnl.get('max_loss_streak', 0)
            
            # Position sizing analytics
            avg_position_size = options_pnl.get('avg_position_size', 1)
            max_position_size = options_pnl.get('max_position_size', 1)
            capital_efficiency = options_pnl.get('capital_efficiency', 0)
            avg_capital_used = options_pnl.get('avg_capital_used', 0)
            
            # Performance grading
            if total_return_pct >= GRADE_A_PLUS_EXCEPTIONAL:
                grade = "A+ (Exceptional)"
            elif total_return_pct >= GRADE_A_PLUS_EXCELLENT:
                grade = "A+ (Excellent)"
            elif total_return_pct >= GRADE_A_VERY_GOOD:
                grade = "A (Very Good)"
            elif total_return_pct >= GRADE_B_PLUS_GOOD:
                grade = "B+ (Good)"
            elif total_return_pct >= GRADE_B_ABOVE_AVERAGE:
                grade = "B (Above Average)"
            elif total_return_pct >= GRADE_C_AVERAGE:
                grade = "C (Average)"
            elif total_return_pct >= 0:
                grade = "D (Below Average)"
            else:
                grade = "F (Poor)"
            print(f"📅 Backetesting Period: {ANALYSIS_START_DATE} to {ANALYSIS_END_DATE}")
            print()
            # Display Capital Performance
            print(f"💰 CAPITAL PERFORMANCE:")
            print(f"   Initial Capital        : ₹{initial_capital:>12,.2f}")
            print(f"   Final Portfolio        : ₹{final_portfolio:>12,.2f}")
            print(f"   Total P&L              : ₹{total_pnl:>12,.2f} ({total_return_pct:+.2f}%)")
            print(f"   Broker Commission      : ₹{total_commission:>12,.2f}")
            print(f"   Slippage               : ₹{total_slippage:>12,.2f} ({slippage_percentage:.2f}%)")
            print(f"   Return on Capital      :      {total_return_pct:>8.2f}%")
            print()
            
            # Display Risk Metrics
            print(f"📉 RISK METRICS:")
            print(f"   Max Drawdown       :        {max_drawdown_pct:>6.2f}% (₹{max_drawdown_amount:>10,.2f})")
            print(f"   Sharpe Ratio       :         {sharpe_ratio:>5.2f}")
            print(f"   Sortino Ratio      :        {sortino_ratio:>6.2f}")
            print(f"   Recovery Factor    :       {recovery_factor:>7.2f}")
            print(f"   Calmar Ratio       :       {calmar_ratio:>7.2f}")
            print()
            
            # Display Trade Statistics
            print(f"🎯 TRADE STATISTICS:")
            print(f"   Total Trades       :          {total_trades:>3}")
            print(f"   Winning Trades     :          {winning_trades:>3} ({win_rate:.1f}%)")
            print(f"   Losing Trades      :          {losing_trades:>3} ({100-win_rate:.1f}%)")
            print(f"   Profit Factor      :         {profit_factor:>5.2f}")
            print(f"   Win/Loss Ratio     :         {win_loss_ratio:>5.2f}")
            print()
            
            # Display Trade Extremes
            print(f"🏆 TRADE EXTREMES:")
            print(f"   Best Trade         : ₹{best_trade:>12,.2f}")
            print(f"   Worst Trade        : ₹{worst_trade:>12,.2f}")
            print(f"   Avg Winning Trade  : ₹{avg_winning_trade:>12,.2f}")
            print(f"   Avg Losing Trade   : ₹{avg_losing_trade:>12,.2f}")
            print(f"   Max Win Streak     :           {max_win_streak:>2}")
            print(f"   Max Loss Streak    :            {max_loss_streak:>1}")
            print()
            
            # Display Position Sizing Analytics
            print(f"⚡ POSITION SIZING ANALYTICS:")
            print(f"   Avg Position Size  :          {avg_position_size:>3.1f} lots")
            print(f"   Max Position Size  :           {max_position_size:>2} lots")
            print(f"   Capital Efficiency :         {capital_efficiency:>5.1f}%")
            print(f"   Avg Capital Used   : ₹{avg_capital_used:>12,.2f}")
            print()
            
            # Overall Performance Grade
            print(f"🎆 OVERALL PERFORMANCE GRADE: {grade}")
            print(f"{'=' * 80}")
            
        except Exception as e:
            self.logger.error(f"Error displaying comprehensive portfolio summary: {e}")
            print(f"❌ Error displaying portfolio summary: {e}")
    
    def run_complete_analysis(
        self, 
        start_date: str = ANALYSIS_START_DATE, 
        end_date: str = ANALYSIS_END_DATE,
        strategies: List[str] = None
    ) -> bool:
        """
        Run complete VectorBT indicator analysis with portfolio backtesting.
        
        Args:
            start_date: Analysis start date
            end_date: Analysis end date
            strategies: List of strategy names to run (default: [ACTIVE_STRATEGY])
            
        Returns:
            bool: True if analysis completed successfully
        """
        try:
            # ⏱️ Start method execution timing
            method_start_time = time.time()
            data_loading_start = time.time()
            
            # Default to single strategy if none provided
            if strategies is None:
                strategies = [ACTIVE_STRATEGY]
            
            print(f"🚀 Starting VectorBT Indicator Signal Analysis...")
            print(f"📅 Period: {start_date} to {end_date}")
            print(f"🎯 Running {len(strategies)} strateg{'y' if len(strategies) == 1 else 'ies'}: {', '.join(strategies)}")
            
            # Step 1: Load market data (once for all strategies)
            if not self.load_nifty_data(start_date, end_date):
                return False
            
            data_loading_time = time.time() - data_loading_start
            
            # Step 2: Calculate ALL indicators in one efficient call (shared across strategies)
            indicators_start_time = time.time()
            indicator_stats = self.calculate_all_indicators()
            indicators_time = time.time() - indicators_start_time
            print(f"✅ All indicators calculated: {indicator_stats}")
            
            # Step 3: Calculate Portfolio Backtest for each strategy
            portfolio_start_time = time.time()
            all_portfolio_stats = {}
            
            for strategy_name in strategies:
                print(f"\n📊 Processing strategy: {strategy_name}")
                strategy_config = SIGNAL_STRATEGIES[strategy_name]
                print(f"   📈 Entry: {strategy_config['entry']}")
                print(f"   📉 Exit: {strategy_config['exit']}")
                print(f"   🎯 Option Type: {strategy_config.get('OptionType', 'N/A')}")
                
                if self.options_enabled:
                    portfolio_stats = self.calculate_options_portfolio_backtest(strategy_name)
                    print(f"✅ Options portfolio backtest calculated for {strategy_name}")
                else:
                    portfolio_stats = self.calculate_portfolio_backtest(strategy_name)
                    print(f"✅ Equity portfolio backtest calculated for {strategy_name}")
                
                all_portfolio_stats[strategy_name] = portfolio_stats
            
            portfolio_time = time.time() - portfolio_start_time
            
            # Step 4: Save all data to CSV files (strategy-specific)
            csv_start_time = time.time()
            print(f"\n💾 Saving data to CSV files...")
            for strategy_name in strategies:
                if self.save_data_to_csv(strategy_name):
                    print(f"✅ Data saved for strategy: {strategy_name}")
                else:
                    print(f"⚠️  Warning: Data may not have been saved properly for {strategy_name}")
            csv_time = time.time() - csv_start_time
            
            # ⏱️ Calculate total method execution time
            total_method_time = time.time() - method_start_time
            
            # 📊 Display detailed execution timing breakdown
            print(f"\n{'=' * 80}")
            print(f"⏱️  DETAILED EXECUTION TIMING BREAKDOWN")
            print(f"{'=' * 80}")
            print(f"📈 Data Loading     : {data_loading_time:.3f}s")
            print(f"🔧 All Indicators   : {indicators_time:.3f}s")
            print(f"💰 Portfolio Calc   : {portfolio_time:.3f}s ({len(strategies)} strateg{'y' if len(strategies) == 1 else 'ies'})")
            print(f"💾 CSV Export       : {csv_time:.3f}s")
            print(f"{'─' * 80}")
            print(f"🎯 Total Method Time: {total_method_time:.3f}s")
            
            # Calculate data processing rate if we have data
            if hasattr(self, 'ohlc_data') and self.ohlc_data is not None:
                data_points = len(self.ohlc_data)
                processing_rate = data_points / total_method_time if total_method_time > 0 else 0
                print(f"📊 Data Points      : {data_points:,} records")
                print(f"⚡ Processing Speed : {processing_rate:,.0f} records/second")
            
            print(f"{'=' * 80}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error in complete analysis: {e}")
            print(f"❌ Analysis failed: {e}")
            return False


def main():
    """Main execution function."""
    # ⏱️ Start overall bot execution timing
    bot_start_time = time.time()
    bot_start_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    
    try:
        import argparse
        
        # Set up command line argument parsing
        parser = argparse.ArgumentParser(
            description='VectorBT Indicator Signal Checking System',
            formatter_class=argparse.RawDescriptionHelpFormatter,
            epilog='''
Examples:
  python vectorbt_indicator_signal_check.py
  python vectorbt_indicator_signal_check.py --start-date 2024-06-01 --end-date 2024-06-30
  python vectorbt_indicator_signal_check.py --start-date 2024-01-01
            '''
        )
        
        parser.add_argument(
            '--start-date', 
            type=str, 
            default=ANALYSIS_START_DATE,
            help=f'Start date for analysis (YYYY-MM-DD format, default: {ANALYSIS_START_DATE})'
        )
        
        parser.add_argument(
            '--end-date', 
            type=str, 
            default=ANALYSIS_END_DATE,
            help=f'End date for analysis (YYYY-MM-DD format, default: {ANALYSIS_END_DATE})'
        )
        
        parser.add_argument(
            '--strategy',
            type=str,
            default=ACTIVE_STRATEGY,
            choices=list(SIGNAL_STRATEGIES.keys()),
            help=f'Single strategy to use (default: {ACTIVE_STRATEGY})'
        )
        
        parser.add_argument(
            '--strategies',
            type=str,
            nargs='+',
            choices=list(SIGNAL_STRATEGIES.keys()),
            help='Multiple strategies to run simultaneously (e.g., --strategies multi_sma_trend multi_sma_trend_put)'
        )
        
        # Parse command line arguments
        args = parser.parse_args()
        
        # Determine strategy selection
        if args.strategies:
            # Multi-strategy mode (explicit selection)
            selected_strategies = args.strategies
            print(f"🎯 Multi-Strategy Mode (Explicit): Running {len(selected_strategies)} strategies")
        elif args.strategy != ACTIVE_STRATEGY:
            # Single strategy mode (different from default)
            selected_strategies = [args.strategy]
            print(f"🎯 Single Strategy Mode: Running {args.strategy}")
        else:
            # Default mode - use all active strategies
            selected_strategies = get_active_strategies()
            if len(selected_strategies) > 1:
                print(f"🎯 Multi-Strategy Mode (Auto): Running {len(selected_strategies)} active strategies")
            else:
                print(f"🎯 Single Strategy Mode (Auto): Running {selected_strategies[0]}")
        
        print("🚀 VectorBT Indicator Signal Checking System")
        print("=" * 50)
        print(f"📅 Analysis Period: {args.start_date} to {args.end_date}")
        
        # Initialize the signal checker
        checker = VectorBTIndicatorSignalChecker()
        
        # Display selected strategies
        for i, strategy in enumerate(selected_strategies, 1):
            strategy_config = SIGNAL_STRATEGIES[strategy]
            print(f"📊 Strategy {i}: {strategy}")
            print(f"   📈 Entry: {strategy_config['entry']}")
            print(f"   📉 Exit: {strategy_config['exit']}")
            print(f"   🎯 Option Type: {strategy_config.get('OptionType', 'N/A')}")
        
        # Run complete analysis with provided date range and strategies
        success = checker.run_complete_analysis(
            start_date=args.start_date,
            end_date=args.end_date,
            strategies=selected_strategies
        )
        
        # ⏱️ Calculate overall bot execution timing
        bot_end_time = time.time()
        bot_end_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        total_execution_time = bot_end_time - bot_start_time
        
        if success:
            print("\n✅ VectorBT analysis completed successfully!")
            
            # 📊 Display comprehensive execution timing summary
            print(f"\n{'=' * 80}")
            print(f"⏱️  COMPLETE BOT EXECUTION TIMING")
            print(f"{'=' * 80}")
            print(f"🚀 Bot Start    : {bot_start_timestamp}")
            print(f"🏁 Bot End      : {bot_end_timestamp}")
            print(f"⚡ Total Time   : {total_execution_time:.3f}s")
            print(f"{'=' * 80}")
            
        else:
            print("\n❌ VectorBT analysis failed!")
            print(f"\n⏱️ Execution Time: {total_execution_time:.3f}s (Failed)")
            return 1
        
        return 0
        
    except KeyboardInterrupt:
        print("\n⏹️  Analysis interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)