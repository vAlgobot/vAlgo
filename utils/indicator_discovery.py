"""
Indicator Discovery Engine for vAlgo Trading System
Auto-discovers available indicators from various sources and updates Rule_Types sheet
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Set
import pandas as pd
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from utils.config_loader import ConfigLoader


class IndicatorDiscovery:
    """
    Auto-discovery engine for indicators from various data sources
    """
    
    def __init__(self, config_loader: ConfigLoader):
        """
        Initialize indicator discovery
        
        Args:
            config_loader: ConfigLoader instance
        """
        self.config_loader = config_loader
        self.logger = get_logger(__name__)
        
        # Define indicator categories for better organization
        self.indicator_categories = {
            'price_data': ['open', 'high', 'low', 'close', 'volume'],
            'ltp_data': ['LTP_Open', 'LTP_High', 'LTP_Low', 'LTP_Close'],
            'moving_averages': ['ema_', 'sma_', 'wma_'],
            'oscillators': ['rsi_', 'stoch_', 'cci_'],
            'momentum': ['macd', 'roc', 'momentum'],
            'volatility': ['bb_', 'atr_', 'vix_'],
            'volume': ['volume_', 'obv_', 'vwap_'],
            'support_resistance': ['cpr_', 'pivot_', 'support_', 'resistance_'],
            'candle_patterns': ['currentcandle_', 'previouscandle_', 'currentday_', 'previousday_'],
            'sl_tp_levels': ['TP', 'SL', 'LTP_High', 'LTP_Low', 'CurrentCandle_Close']
        }
    
    def discover_available_indicators(self) -> List[Dict[str, Any]]:
        """
        Discover all available indicators from various sources
        
        Returns:
            List of discovered indicator dictionaries
        """
        discovered_indicators = []
        
        try:
            # 1. Discover from UnifiedIndicatorEngine output
            engine_indicators = self._discover_from_indicator_engine()
            discovered_indicators.extend(engine_indicators)
            
            # 2. Discover LTP parameters for 1-minute exit conditions
            ltp_indicators = self._discover_ltp_indicators()
            discovered_indicators.extend(ltp_indicators)
            
            # 3. Discover SL/TP levels for exit conditions
            sl_tp_indicators = self._discover_sl_tp_indicators()
            discovered_indicators.extend(sl_tp_indicators)
            
            # 4. Discover basic OHLCV data
            ohlcv_indicators = self._discover_ohlcv_indicators()
            discovered_indicators.extend(ohlcv_indicators)
            
            # 4. Discover candle reference indicators
            candle_indicators = self._discover_candle_indicators()
            discovered_indicators.extend(candle_indicators)
            
            # Remove duplicates
            discovered_indicators = self._remove_duplicates(discovered_indicators)
            
            self.logger.info(f"Discovered {len(discovered_indicators)} total indicators")
            return discovered_indicators
            
        except Exception as e:
            self.logger.error(f"Error during indicator discovery: {e}")
            return []
    
    def _discover_from_indicator_engine(self) -> List[Dict[str, Any]]:
        """Discover indicators from UnifiedIndicatorEngine output"""
        indicators = []
        
        try:
            # Try to get a sample of indicator data
            from indicators.unified_indicator_engine import UnifiedIndicatorEngine
            
            # Initialize engine with config
            engine = UnifiedIndicatorEngine(self.config_loader.config_file)
            
            # Get sample data to discover available indicators
            sample_data = self._get_sample_data_for_discovery(engine)
            
            if sample_data is not None and not sample_data.empty:
                # Extract column names that look like indicators
                for column in sample_data.columns:
                    if self._is_indicator_column(column):
                        indicator_info = {
                            'name': column,
                            'description': self._generate_description(column),
                            'category': self._categorize_indicator(column),
                            'source': 'UnifiedIndicatorEngine'
                        }
                        indicators.append(indicator_info)
            
            self.logger.info(f"Discovered {len(indicators)} indicators from UnifiedIndicatorEngine")
            
        except Exception as e:
            self.logger.warning(f"Could not discover from UnifiedIndicatorEngine: {e}")
        
        return indicators
    
    def _discover_ltp_indicators(self) -> List[Dict[str, Any]]:
        """Discover LTP (Live Trading Price) indicators for 1-minute exit conditions"""
        ltp_indicators = [
            {
                'name': 'LTP_Open',
                'description': 'Live 1-minute candle OPEN price for exit conditions',
                'category': 'ltp_data',
                'source': 'ExecutionData'
            },
            {
                'name': 'LTP_High',
                'description': 'Live 1-minute candle HIGH price for exit conditions',
                'category': 'ltp_data',
                'source': 'ExecutionData'
            },
            {
                'name': 'LTP_Low',
                'description': 'Live 1-minute candle LOW price for exit conditions',
                'category': 'ltp_data',
                'source': 'ExecutionData'
            },
            {
                'name': 'LTP_Close',
                'description': 'Live 1-minute candle CLOSE price for exit conditions',
                'category': 'ltp_data',
                'source': 'ExecutionData'
            }
        ]
        
        self.logger.info(f"Discovered {len(ltp_indicators)} LTP indicators")
        return ltp_indicators
    
    def _discover_ohlcv_indicators(self) -> List[Dict[str, Any]]:
        """Discover basic OHLCV market data indicators"""
        ohlcv_indicators = [
            {
                'name': 'open',
                'description': 'Opening price of the candle',
                'category': 'price_data',
                'source': 'MarketData'
            },
            {
                'name': 'high',
                'description': 'Highest price of the candle',
                'category': 'price_data',
                'source': 'MarketData'
            },
            {
                'name': 'low',
                'description': 'Lowest price of the candle',
                'category': 'price_data',
                'source': 'MarketData'
            },
            {
                'name': 'close',
                'description': 'Closing price of the candle',
                'category': 'price_data',
                'source': 'MarketData'
            },
            {
                'name': 'volume',
                'description': 'Trading volume of the candle',
                'category': 'price_data',
                'source': 'MarketData'
            }
        ]
        
        self.logger.info(f"Discovered {len(ohlcv_indicators)} OHLCV indicators")
        return ohlcv_indicators
    
    def _discover_sl_tp_indicators(self) -> List[Dict[str, Any]]:
        """Discover SL/TP level indicators for exit conditions"""
        sl_tp_indicators = [
            {
                'name': 'TP',
                'description': 'Take Profit price level calculated by SL/TP engine',
                'category': 'sl_tp_levels',
                'source': 'SLTPEngine'
            },
            {
                'name': 'SL',
                'description': 'Stop Loss price level calculated by SL/TP engine',
                'category': 'sl_tp_levels',
                'source': 'SLTPEngine'
            },
            {
                'name': 'LTP_High',
                'description': 'Live 1-minute candle HIGH price from execution data',
                'category': 'sl_tp_levels',
                'source': 'ExecutionData'
            },
            {
                'name': 'LTP_Low',
                'description': 'Live 1-minute candle LOW price from execution data',
                'category': 'sl_tp_levels',
                'source': 'ExecutionData'
            },
            {
                'name': 'CurrentCandle_Close',
                'description': 'Current candle closing price for exit conditions',
                'category': 'sl_tp_levels',
                'source': 'MarketData'
            }
        ]
        
        self.logger.info(f"Discovered {len(sl_tp_indicators)} SL/TP level indicators")
        return sl_tp_indicators
    
    def _discover_candle_indicators(self) -> List[Dict[str, Any]]:
        """Discover candle reference indicators (CurrentCandle, PreviousCandle, etc.)"""
        candle_indicators = []
        
        # Define candle reference types
        candle_types = [
            ('CurrentCandle', 'Current candle'),
            ('PreviousCandle', 'Previous candle'),
            ('CurrentDayCandle', 'Current day candle'),
            ('PreviousDayCandle', 'Previous day candle')
        ]
        
        price_types = ['Open', 'High', 'Low', 'Close']
        
        for candle_type, description in candle_types:
            for price_type in price_types:
                indicator_name = f"{candle_type}_{price_type}"
                candle_indicators.append({
                    'name': indicator_name,
                    'description': f'{description} {price_type.lower()} price',
                    'category': 'candle_patterns',
                    'source': 'CandleReference'
                })
        
        self.logger.info(f"Discovered {len(candle_indicators)} candle reference indicators")
        return candle_indicators
    
    def _get_sample_data_for_discovery(self, engine) -> Optional[pd.DataFrame]:
        """Get sample data from UnifiedIndicatorEngine for discovery"""
        try:
            # Try to get sample data for a known symbol
            instruments = self.config_loader.get_active_instruments()
            
            if not instruments:
                self.logger.warning("No active instruments found for discovery")
                return None
            
            # Use first active instrument for discovery
            sample_symbol = instruments[0].get('symbol', 'NIFTY')
            
            # Get a small sample of data (last 100 records)
            sample_data = engine.calculate_indicators_for_symbol(
                symbol=sample_symbol,
                start_date='2024-01-01',
                end_date='2024-01-31'
            )
            
            if sample_data is not None and not sample_data.empty:
                # Return just the first few rows for column discovery
                return sample_data.head(10)
            
        except Exception as e:
            self.logger.warning(f"Could not get sample data for discovery: {e}")
        
        return None
    
    def _is_indicator_column(self, column_name: str) -> bool:
        """Check if a column name represents an indicator"""
        column_lower = column_name.lower()
        
        # Skip basic market data columns
        basic_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'symbol']
        if column_lower in basic_columns:
            return False
        
        # Look for indicator patterns
        indicator_patterns = [
            'ema_', 'sma_', 'wma_', 'rsi_', 'macd', 'bb_', 'atr_', 'stoch_',
            'cci_', 'roc_', 'momentum', 'obv_', 'vwap_', 'cpr_', 'pivot_',
            'support_', 'resistance_', 'candle_', 'signal_'
        ]
        
        return any(pattern in column_lower for pattern in indicator_patterns)
    
    def _categorize_indicator(self, indicator_name: str) -> str:
        """Categorize an indicator based on its name"""
        name_lower = indicator_name.lower()
        
        for category, patterns in self.indicator_categories.items():
            if any(pattern in name_lower for pattern in patterns):
                return category
        
        return 'technical_indicator'  # Default category
    
    def _generate_description(self, indicator_name: str) -> str:
        """Generate description for an indicator"""
        name_lower = indicator_name.lower()
        
        # Common indicator descriptions
        descriptions = {
            'ema_': 'Exponential Moving Average',
            'sma_': 'Simple Moving Average',
            'wma_': 'Weighted Moving Average',
            'rsi_': 'Relative Strength Index',
            'macd': 'Moving Average Convergence Divergence',
            'bb_': 'Bollinger Bands',
            'atr_': 'Average True Range',
            'stoch_': 'Stochastic Oscillator',
            'cci_': 'Commodity Channel Index',
            'roc_': 'Rate of Change',
            'obv_': 'On-Balance Volume',
            'vwap_': 'Volume Weighted Average Price',
            'cpr_': 'Central Pivot Range',
            'pivot_': 'Pivot Point'
        }
        
        for pattern, description in descriptions.items():
            if pattern in name_lower:
                return f"{description}: {indicator_name}"
        
        return f"Technical Indicator: {indicator_name}"
    
    def _remove_duplicates(self, indicators: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate indicators based on name"""
        seen_names = set()
        unique_indicators = []
        
        for indicator in indicators:
            name = indicator.get('name', '')
            if name and name not in seen_names:
                seen_names.add(name)
                unique_indicators.append(indicator)
        
        removed_count = len(indicators) - len(unique_indicators)
        if removed_count > 0:
            self.logger.info(f"Removed {removed_count} duplicate indicators")
        
        return unique_indicators
    
    def generate_discovery_report(self, indicators: List[Dict[str, Any]]) -> str:
        """Generate a summary report of discovered indicators"""
        if not indicators:
            return "No indicators discovered"
        
        # Group by category
        categories = {}
        for indicator in indicators:
            category = indicator.get('category', 'unknown')
            if category not in categories:
                categories[category] = []
            categories[category].append(indicator['name'])
        
        # Build report
        report_lines = [
            f"Indicator Discovery Report",
            f"=" * 50,
            f"Total Indicators: {len(indicators)}",
            f"",
            f"By Category:",
            f"-" * 30
        ]
        
        for category, names in sorted(categories.items()):
            report_lines.append(f"{category}: {len(names)} indicators")
            for name in sorted(names):
                report_lines.append(f"  - {name}")
            report_lines.append("")
        
        return "\n".join(report_lines)


def discover_indicators(config_file: str = None) -> List[Dict[str, Any]]:
    """
    Convenience function to discover indicators
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        List of discovered indicators
    """
    try:
        config_loader = ConfigLoader(config_file)
        if not config_loader.load_config():
            raise ValueError("Failed to load configuration")
        
        discovery = IndicatorDiscovery(config_loader)
        return discovery.discover_available_indicators()
        
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error in indicator discovery: {e}")
        return []


def update_rule_types_with_discovery(config_file: str = None) -> bool:
    """
    Discover indicators and update Rule_Types sheet
    
    Args:
        config_file: Path to configuration file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        config_loader = ConfigLoader(config_file)
        if not config_loader.load_config():
            raise ValueError("Failed to load configuration")
        
        # Discover indicators
        discovery = IndicatorDiscovery(config_loader)
        discovered_indicators = discovery.discover_available_indicators()
        
        if not discovered_indicators:
            print("❌ No indicators discovered - this indicates a system error")
            return False
        
        # Generate and print discovery report (always show what was discovered)
        report = discovery.generate_discovery_report(discovered_indicators)
        print(report)
        
        # Update Rule_Types sheet
        success = config_loader.update_rule_types_sheet(discovered_indicators)
        
        if success:
            print(f"\n✅ Successfully updated Rule_Types sheet with {len(discovered_indicators)} indicators")
            print("📋 Rule_Types sheet is now up-to-date with all available indicators")
            return True
        else:
            print("❌ Failed to update Rule_Types sheet")
            return False
            
    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Error updating Rule_Types with discovery: {e}")
        print(f"Error: {e}")
        return False


if __name__ == "__main__":
    # Test the discovery system
    print("Running Indicator Discovery Test...")
    success = update_rule_types_with_discovery()
    
    if success:
        print("\nIndicator discovery completed successfully!")
    else:
        print("\nIndicator discovery failed!")