"""
Results Exporter - Export Utility for Ultimate Efficiency Engine
================================================================

Results export functionality for Ultimate Efficiency Engine.
Extracts export functionality from ultimate_efficiency_engine.py for modular architecture.

Features:
- Export results to multiple formats (JSON, CSV, Excel)
- Structured file naming with timestamps
- Trade data export with comprehensive details
- Performance metrics export
- Configurable export paths and formats

Author: vAlgo Development Team
Created: July 29, 2025
"""

import sys
import json
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


class ResultsExporter(ProcessorBase):
    """
    Results export functionality for Ultimate Efficiency Engine.
    
    Provides comprehensive export capabilities:
    - Export results to multiple formats (JSON, CSV, Excel)
    - Structured file naming with timestamps and metadata
    - Trade data export with comprehensive trade details
    - Performance metrics export for analysis
    - Configurable export paths and formats
    """
    
    def __init__(self, config_loader, logger):
        """Initialize Results Exporter."""
        # Results exporter doesn't need data_loader, pass None
        super().__init__(config_loader, None, logger)
        
        # Get export configuration
        self.export_config = self.main_config.get('reports', {})
        self.export_path = self.export_config.get('export_path', 'output/vbt_reports/')
        self.enabled_formats = self.export_config.get('enabled_formats', ['json', 'csv'])
        
        self.log_major_component("Results Exporter initialized", "RESULTS_EXPORTER")
    
    def export_results(
        self,
        results: Dict[str, Any],
        symbol: str,
        timeframe: str,
        start_date: str,
        end_date: str,
        additional_suffix: str = None
    ) -> Dict[str, str]:
        """
        Export analysis results to configured formats.
        
        Args:
            results: Complete analysis results dictionary
            symbol: Trading symbol
            timeframe: Timeframe used
            start_date: Analysis start date
            end_date: Analysis end date
            additional_suffix: Optional additional suffix for filename
            
        Returns:
            Dictionary with exported file paths
        """
        try:
            with self.measure_execution_time("results_export"):
                self.log_major_component("Exporting analysis results", "RESULTS_EXPORT")
                
                # Create output directory
                output_dir = Path(self.export_path)
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Generate base filename
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_components = [
                    "ultimate_efficiency",
                    symbol.upper(),
                    timeframe,
                    start_date.replace('-', ''),
                    end_date.replace('-', ''),
                    timestamp
                ]
                
                if additional_suffix:
                    base_components.append(additional_suffix)
                
                filename_base = "_".join(base_components)
                
                exported_files = {}
                
                # Export in configured formats
                if 'json' in self.enabled_formats:
                    json_path = self._export_json(results, output_dir, filename_base)
                    if json_path:
                        exported_files['json'] = str(json_path)
                
                if 'csv' in self.enabled_formats:
                    csv_paths = self._export_csv(results, output_dir, filename_base)
                    exported_files.update(csv_paths)
                
                # Export trade comparison CSV if available
                self.log_major_component("[CHECKING] CHECKING TRADE COMPARISON EXPORT", "RESULTS_EXPORT")
                self.log_detailed(f"Available keys in results: {list(results.keys())}", "INFO")
                self.log_detailed(f"Looking for 'trade_comparison' key in results", "INFO")
                
                if 'trade_comparison' in results:
                    self.log_major_component("[SUCCESS] TRADE COMPARISON DATA FOUND - STARTING EXPORT", "RESULTS_EXPORT")
                    trade_comparison_data = results['trade_comparison']
                    self.log_detailed(f"Trade comparison data type: {type(trade_comparison_data)}", "DEBUG")
                    self.log_detailed(f"Trade comparison data keys: {list(trade_comparison_data.keys()) if isinstance(trade_comparison_data, dict) else 'Not a dict'}", "DEBUG")
                    
                    comparison_csv_path = self._export_trade_comparison_csv(
                        trade_comparison_data, output_dir, filename_base
                    )
                    if comparison_csv_path:
                        exported_files['trade_comparison_csv'] = str(comparison_csv_path)
                        self.log_major_component(f"[SUCCESS] TRADE COMPARISON CSV EXPORT SUCCESS: {comparison_csv_path.name}", "RESULTS_EXPORT")
                    else:
                        self.log_major_component("[ERROR] TRADE COMPARISON CSV EXPORT FAILED", "RESULTS_EXPORT")
                else:
                    self.log_major_component("[ERROR] NO TRADE COMPARISON DATA IN RESULTS", "RESULTS_EXPORT")
                    self.log_detailed("This means trade comparison was not added to results in Phase 6.5", "WARNING")
                
                if 'excel' in self.enabled_formats:
                    excel_path = self._export_excel(results, output_dir, filename_base)
                    if excel_path:
                        exported_files['excel'] = str(excel_path)
                
                # Display export summary
                self._display_export_summary(exported_files, output_dir)
                
                self.log_major_component(
                    f"Results export complete: {len(exported_files)} files exported",
                    "RESULTS_EXPORT"
                )
                
                return exported_files
                
        except Exception as e:
            self.log_detailed(f"Error exporting results: {e}", "ERROR")
            return {'error': str(e)}
    
    def _export_json(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Optional[Path]:
        """Export results to JSON format."""
        try:
            json_path = output_dir / f"{filename_base}.json"
            
            # Convert numpy types to native Python types for JSON serialization
            json_results = self._convert_for_json(results)
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(json_results, f, indent=2, default=str, ensure_ascii=False)
            
            self.log_detailed(f"JSON export completed: {json_path.name}", "INFO")
            return json_path
            
        except Exception as e:
            self.log_detailed(f"Error exporting JSON: {e}", "ERROR")
            return None
    
    def _export_csv(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Dict[str, str]:
        """Export results to CSV format(s) using two-CSV architecture."""
        exported_csvs = {}
        
        try:
            # Two-CSV Architecture: Separate signal data and trade data to avoid mapping conflicts
            
            # Export 1: Signals CSV - Pure signal data with market data and indicators
            signals_path = self._export_signals_csv(results, output_dir, filename_base)
            if signals_path:
                exported_csvs['signals_csv'] = str(signals_path)
            
            # Export 2: Trades CSV - Pure trade data with premiums and P&L
            trades_path = self._export_trades_csv(results, output_dir, filename_base)
            if trades_path:
                exported_csvs['trades_csv'] = str(trades_path)
            
            return exported_csvs
            
        except Exception as e:
            self.log_detailed(f"Error exporting CSV: {e}", "ERROR")
            return {'csv_error': str(e)}
    
    def _export_signals_csv(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Optional[Path]:
        """
        Export pure signal data CSV with market data and indicators.
        
        Creates clean signals CSV with:
        - OHLC market data (timestamp, open, high, low, close, volume)
        - Technical indicators (RSI_14, RSI_21, MA_9, MA_20, MA_50, MA_200, etc.)
        - Signal data (Entry_Signal, Exit_Signal, Strategy_Signals, Strategy_Signal_Text)
        - Signal_Adjusted_Timestamp for timestamp tracking
        
        Args:
            results: Complete analysis results dictionary
            output_dir: Output directory path
            filename_base: Base filename
            
        Returns:
            Path to exported signals CSV file or None if failed
        """
        try:
            # Extract ACTUAL market data DataFrame from results
            market_data_df = results.get('raw_market_data')
            if market_data_df is None or market_data_df.empty:
                self.log_detailed("No raw market data available for signals CSV export", "WARNING")
                return None
            
            # Create signals DataFrame from real market data
            signals_df = market_data_df.copy()
            
            # Ensure proper datetime index
            if not isinstance(signals_df.index, pd.DatetimeIndex):
                signals_df.index = pd.to_datetime(signals_df.index)
            
            self.log_detailed(f"Creating signals CSV with {len(signals_df)} rows", "INFO")
            
            # Add REAL technical indicators from analysis
            raw_indicators = results.get('raw_indicators', {})
            self._add_real_indicators_to_dataframe(signals_df, raw_indicators)
            
            # Add signal columns with adjusted timestamps
            signals_data = results.get('signals', {})
            entries_list = signals_data.get('entries', [])
            exits_list = signals_data.get('exits', [])
            self._add_signals_to_dataframe(signals_df, entries_list, exits_list)
            
            # Export signals CSV
            csv_path = output_dir / f"{filename_base}_signals.csv"
            signals_df.to_csv(csv_path, index=True)
            
            self.log_detailed(f"Signals CSV export completed: {csv_path.name}", "INFO")
            return csv_path
            
        except Exception as e:
            self.log_detailed(f"Error exporting signals CSV: {e}", "ERROR")
            return None
    
    def _create_enriched_trades_data(self, raw_trades_data: List[Dict]) -> List[Dict]:
        """
        Create enriched trades data with P&L calculations for trade comparison.
        
        Args:
            raw_trades_data: Raw trades data from P&L calculator
            
        Returns:
            List of enriched trade dictionaries with P&L calculations
        """
        try:
            # Get configuration data for calculations
            config_data = self.main_config.get('options', {})
            trading_config = self.main_config.get('trading', {})
            default_option_type = config_data.get('default_option_type', 'CALL')
            commission_per_trade = trading_config.get('commission_per_trade', 20.0)
            lot_size = trading_config.get('lot_size', 75)
            
            enriched_trades = []
            
            for i, trade_data in enumerate(raw_trades_data):
                try:
                    # Extract trade information
                    entry_timestamp = trade_data.get('entry_timestamp')
                    exit_timestamp = trade_data.get('exit_timestamp')
                    entry_premium = float(trade_data.get('entry_premium', 0.0))
                    exit_premium = float(trade_data.get('exit_premium', 0.0))
                    strike_price = int(trade_data.get('entry_strike', trade_data.get('strike', 0)))
                    option_type = str(trade_data.get('option_type', default_option_type))
                    strategy_name = str(trade_data.get('group', ''))
                    case_name = str(trade_data.get('case', ''))
                    position_size = int(trade_data.get('position_size', 1))
                    
                    # Calculate P&L with correct options logic
                    options_pnl_value = (exit_premium - entry_premium) * lot_size * position_size
                    net_pnl = options_pnl_value - commission_per_trade
                    
                    # Extract daily trade sequence for Trade1/Trade2/Trade3 tracking
                    daily_trade_sequence = trade_data.get('daily_trade_sequence', 0)
                    
                    # Create enriched trade record (combining original data with P&L calculations)
                    enriched_trade = dict(trade_data)  # Start with original data
                    enriched_trade.update({
                        'Trade_ID': i + 1,
                        'Daily_Trade_Number': daily_trade_sequence,  # TRADE TRACKING: 1, 2, or 3
                        'Strategy_Name': strategy_name,
                        'Case_Name': case_name,
                        'Position_Size': position_size,
                        'Entry_Timestamp': entry_timestamp,
                        'Entry_Strike': strike_price,
                        'Entry_Premium': entry_premium,
                        'Exit_Timestamp': exit_timestamp,
                        'Exit_Strike': strike_price,
                        'Exit_Premium': exit_premium,
                        'Option_Type': option_type,
                        'Options_PnL': options_pnl_value,
                        'Commission': commission_per_trade,
                        'Net_PnL': net_pnl,
                        'Lot_Size': lot_size
                    })
                    
                    enriched_trades.append(enriched_trade)
                    
                except Exception as e:
                    self.log_detailed(f"Error enriching trade {i}: {e}", "WARNING")
                    continue
                    
            self.log_detailed(f"Created {len(enriched_trades)} enriched trades with P&L calculations", "DEBUG")
            return enriched_trades
            
        except Exception as e:
            self.log_detailed(f"Error creating enriched trades data: {e}", "ERROR")
            return []
    
    def _export_trades_csv(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Optional[Path]:
        """
        Export pure trade data CSV with premiums and P&L calculations.
        
        Creates clean trades CSV with:
        - Trade metadata (Trade_ID, Strategy_Name, Case_Name, Position_Size)
        - Entry data (Entry_Timestamp, Entry_Strike, Entry_Premium, Option_Type)
        - Exit data (Exit_Timestamp, Exit_Strike, Exit_Premium)
        - P&L data (Options_PnL, Commission, Net_PnL)
        
        Args:
            results: Complete analysis results dictionary
            output_dir: Output directory path
            filename_base: Base filename
            
        Returns:
            Path to exported trades CSV file or None if failed
        """
        try:
            # Get REAL trade data from P&L calculation results
            performance_data = results.get('performance', {})
            options_pnl = performance_data.get('options_pnl', {})
            total_trades = options_pnl.get('total_trades', 0)
            trades_data = options_pnl.get('trades_data', [])  # Real trade data from P&L calculator
            
            if total_trades == 0 or not trades_data:
                self.log_detailed("No trade data available for trades CSV export", "WARNING")
                return None
            
            self.log_detailed(f"Creating trades CSV with {len(trades_data)} trades", "INFO")
            
            # Get configuration data for defaults
            config_data = self.main_config.get('options', {})
            trading_config = self.main_config.get('trading', {})
            default_option_type = config_data.get('default_option_type', 'CALL')
            commission_per_trade = trading_config.get('commission_per_trade', 20.0)
            lot_size = trading_config.get('lot_size', 75)
            
            # Create clean trades list
            trades_list = []
            
            for i, trade_data in enumerate(trades_data):
                try:
                    # Extract trade information
                    entry_timestamp = trade_data.get('entry_timestamp')
                    exit_timestamp = trade_data.get('exit_timestamp')
                    entry_premium = float(trade_data.get('entry_premium', 0.0))
                    exit_premium = float(trade_data.get('exit_premium', 0.0))
                    strike_price = int(trade_data.get('entry_strike', trade_data.get('strike', 0)))
                    option_type = str(trade_data.get('option_type', default_option_type))
                    strategy_name = str(trade_data.get('group', ''))
                    case_name = str(trade_data.get('case', ''))
                    position_size = int(trade_data.get('position_size', 1))
                    
                    # Calculate P&L with correct options logic
                    # For BUY strategy: P&L = (Exit Premium - Entry Premium) * Lot Size * Position Size
                    options_pnl_value = (exit_premium - entry_premium) * lot_size * position_size
                    net_pnl = options_pnl_value - commission_per_trade
                    
                    # Extract daily trade sequence for Trade1/Trade2/Trade3 tracking
                    daily_trade_sequence = trade_data.get('daily_trade_sequence', 0)
                    
                    # Create clean trade record
                    trade_record = {
                        'Trade_ID': i + 1,
                        'Daily_Trade_Number': daily_trade_sequence,  # TRADE TRACKING: 1, 2, or 3
                        'Strategy_Name': strategy_name,
                        'Case_Name': case_name,
                        'Position_Size': position_size,
                        'Entry_Timestamp': entry_timestamp,
                        'Entry_Strike': strike_price,
                        'Entry_Premium': entry_premium,
                        'Exit_Timestamp': exit_timestamp,
                        'Exit_Strike': strike_price,  # Same strike for options
                        'Exit_Premium': exit_premium,
                        'Option_Type': option_type,
                        'Options_PnL': options_pnl_value,
                        'Commission': commission_per_trade,
                        'Net_PnL': net_pnl,
                        'Lot_Size': lot_size
                    }
                    
                    trades_list.append(trade_record)
                    
                except Exception as trade_error:
                    self.log_detailed(f"Error processing trade {i}: {trade_error}", "ERROR")
                    continue
            
            if not trades_list:
                self.log_detailed("No valid trades processed for trades CSV", "WARNING")
                return None
            
            # Create trades DataFrame
            trades_df = pd.DataFrame(trades_list)
            
            # Export trades CSV
            csv_path = output_dir / f"{filename_base}_trades.csv"
            trades_df.to_csv(csv_path, index=False)
            
            self.log_detailed(f"Trades CSV export completed: {csv_path.name} with {len(trades_df)} trades", "INFO")
            return csv_path
            
        except Exception as e:
            self.log_detailed(f"Error exporting trades CSV: {e}", "ERROR")
            return None
    
    def _export_comprehensive_trade_csv(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Optional[Path]:
        """
        Export comprehensive trade-level CSV matching original system format.
        
        Creates detailed CSV with:
        - OHLC market data (timestamp, open, high, low, close, volume)
        - Technical indicators (RSI_14, RSI_21, MA_9, MA_20, MA_50, MA_200, etc.)
        - Signal data (Entry_Signal, Exit_Signal, Strategy_Signals, Strategy_Signal_Text)
        - Options trading data (Strike_Price, Entry_Premium, Exit_Premium, Options_PnL, Option_Type, Commission, Net_PnL)
        
        Args:
            results: Complete analysis results dictionary
            output_dir: Output directory path
            filename_base: Base filename
            
        Returns:
            Path to exported CSV file or None if failed
        """
        try:
            # Extract ACTUAL market data DataFrame from results (not summary)
            market_data_df = results.get('raw_market_data')
            if market_data_df is None or market_data_df.empty:
                self.log_detailed("No raw market data available for comprehensive CSV export", "WARNING")
                return None
            
            # Get indicators data
            indicators_data = results.get('indicators', {}).get('calculated_indicators', {})
            
            # Get signals data
            signals_data = results.get('signals', {})
            entries_list = signals_data.get('entries', [])
            exits_list = signals_data.get('exits', [])
            
            # Get performance/options data
            performance_data = results.get('performance', {})
            options_pnl = performance_data.get('options_pnl', {})
            
            # Use ACTUAL market data DataFrame instead of creating synthetic data
            comprehensive_df = market_data_df.copy()
            
            # Ensure proper datetime index
            if not isinstance(comprehensive_df.index, pd.DatetimeIndex):
                comprehensive_df.index = pd.to_datetime(comprehensive_df.index)
            
            self.log_detailed(f"Using real market data with {len(comprehensive_df)} rows", "INFO")
            
            # Add REAL technical indicators from analysis
            raw_indicators = results.get('raw_indicators', {})
            self._add_real_indicators_to_dataframe(comprehensive_df, raw_indicators)
            
            # Add signal columns
            self._add_signals_to_dataframe(comprehensive_df, entries_list, exits_list)
            
            # Add options trading data
            self._add_options_data_to_dataframe(comprehensive_df, performance_data)
            
            # Export to CSV
            csv_path = output_dir / f"{filename_base}_complete_analysis.csv"
            comprehensive_df.to_csv(csv_path, index=True)
            
            self.log_detailed(f"Comprehensive CSV export completed: {csv_path.name}", "INFO")
            return csv_path
            
        except Exception as e:
            self.log_detailed(f"Error exporting comprehensive CSV: {e}", "ERROR")
            return None
    
    def _create_ohlc_base_dataframe(self, results: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Create base OHLC DataFrame from results data."""
        try:
            # Extract market data summary
            market_data = results.get('market_data', {})
            
            # Get configuration for date range and candle count
            config_data = results.get('configuration', {})
            trading_config = config_data.get('trading', {})
            backtesting_config = config_data.get('backtesting', {})
            
            start_date = backtesting_config.get('start_date', '2025-06-06')
            end_date = backtesting_config.get('end_date', '2025-06-06')
            symbol = trading_config.get('symbol', 'NIFTY')
            timeframe = trading_config.get('timeframe', '5m')
            
            # Get total candles from market data
            total_candles = market_data.get('total_candles', 75)
            
            # Create datetime index for trading session (9:15 AM to 3:30 PM IST)
            from datetime import datetime, timedelta
            
            # Parse start date
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            session_start = start_dt.replace(hour=9, minute=15, second=0, microsecond=0)
            
            # Generate timestamps based on timeframe
            if timeframe == '5m':
                freq_minutes = 5
            elif timeframe == '1m':
                freq_minutes = 1
            else:
                freq_minutes = 5  # Default to 5m
            
            # Create timestamp index
            timestamps = []
            current_time = session_start
            for i in range(total_candles):
                timestamps.append(current_time)
                current_time += timedelta(minutes=freq_minutes)
            
            # Get price summary for realistic OHLC data
            price_summary = market_data.get('price_summary', {})
            open_range = price_summary.get('open_range', {})
            close_range = price_summary.get('close_range', {})
            
            min_open = open_range.get('min', 24687.0)
            max_open = open_range.get('max', 25026.85)
            avg_open = open_range.get('avg', 24929.51)
            
            min_close = close_range.get('min', 24687.35)
            max_close = close_range.get('max', 25025.8)
            avg_close = close_range.get('avg', 24932.79)
            
            # Generate realistic OHLC data
            np.random.seed(42)  # For reproducible results
            
            opens = np.random.uniform(min_open, max_open, total_candles)
            closes = np.random.uniform(min_close, max_close, total_candles)
            
            # Generate highs and lows based on opens and closes
            highs = np.maximum(opens, closes) + np.random.uniform(0, 20, total_candles)
            lows = np.minimum(opens, closes) - np.random.uniform(0, 15, total_candles)
            
            # Volume (use 0 as in original data)
            volumes = np.zeros(total_candles)
            
            # Create DataFrame
            ohlc_df = pd.DataFrame({
                'timestamp': timestamps,
                'open': opens,
                'high': highs,
                'low': lows,
                'close': closes,
                'volume': volumes
            })
            
            # Set timestamp as index
            ohlc_df.set_index('timestamp', inplace=True)
            
            self.log_detailed(f"Created OHLC base DataFrame with {len(ohlc_df)} rows", "INFO")
            return ohlc_df
            
        except Exception as e:
            self.log_detailed(f"Error creating OHLC base DataFrame: {e}", "ERROR")
            return None
    
    def _add_real_indicators_to_dataframe(self, df: pd.DataFrame, raw_indicators: Dict[str, pd.Series]) -> None:
        """Add REAL technical indicators calculated during analysis to the comprehensive DataFrame."""
        try:
            # Dynamically create mapping for all available indicators
            indicator_mapping = self._create_dynamic_indicator_mapping(raw_indicators)
            
            # Initialize all indicator columns with NaN first
            for csv_column in indicator_mapping.values():
                df[csv_column] = np.nan
            
            # Add real indicator values from analysis
            for indicator_key, csv_column in indicator_mapping.items():
                if indicator_key in raw_indicators:
                    indicator_series = raw_indicators[indicator_key]
                    
                    # Ensure the indicator series has the same index as DataFrame
                    if isinstance(indicator_series, pd.Series):
                        # Align the indicator data with DataFrame index
                        aligned_data = indicator_series.reindex(df.index)
                        df[csv_column] = aligned_data.values
                        
                        valid_count = aligned_data.notna().sum()
                        self.log_detailed(f"Added {indicator_key} -> {csv_column}: {valid_count} valid values", "INFO")
            
            self.log_detailed("Added real technical indicators to DataFrame", "INFO")
            
        except Exception as e:
            self.log_detailed(f"Error adding real indicators to DataFrame: {e}", "ERROR")
    
    def _create_dynamic_indicator_mapping(self, raw_indicators: Dict[str, pd.Series]) -> Dict[str, str]:
        """
        Create dynamic mapping from indicator keys to CSV column names.
        
        This method automatically detects all available indicators and creates 
        appropriate CSV column names instead of using hardcoded mappings.
        
        Args:
            raw_indicators: Dictionary of calculated indicators
            
        Returns:
            Dictionary mapping indicator keys to CSV column names
        """
        try:
            indicator_mapping = {}
            
            for indicator_key in raw_indicators.keys():
                # Skip non-indicator keys (market data)
                if indicator_key in ['close', 'open', 'high', 'low', 'volume']:
                    continue
                
                # Create CSV column name based on indicator type and period
                if '_' in indicator_key:
                    indicator_type, period = indicator_key.split('_', 1)
                    
                    # Map indicator types to CSV column prefixes
                    if indicator_type == 'rsi':
                        csv_column = f"RSI_{period}"
                    elif indicator_type == 'sma':
                        csv_column = f"MA_{period}"  # Keep existing MA naming for SMA
                    elif indicator_type == 'ema':
                        csv_column = f"EMA_{period}"
                    elif indicator_type == 'vwap':
                        csv_column = f"VWAP_{period}"
                    elif indicator_type == 'bb':
                        csv_column = f"BB_{period}"
                    else:
                        # Generic naming for unknown indicators
                        csv_column = f"{indicator_type.upper()}_{period}"
                else:
                    # Single indicators without periods
                    csv_column = indicator_key.upper()
                
                indicator_mapping[indicator_key] = csv_column
            
            self.log_detailed(f"Created dynamic indicator mapping for {len(indicator_mapping)} indicators", "INFO")
            return indicator_mapping
            
        except Exception as e:
            self.log_detailed(f"Error creating dynamic indicator mapping: {e}", "ERROR")
            # Return empty mapping on error
            return {}
    
    def _add_signals_to_dataframe(self, df: pd.DataFrame, entries_list: List[Dict], exits_list: List[Dict]) -> None:
        """Add signal columns to the comprehensive DataFrame."""
        try:
            # Initialize signal columns
            df['Entry_Signal'] = False
            df['Exit_Signal'] = False
            df['Strategy_Signals'] = 0
            df['Strategy_Signal_Text'] = 'Hold'
            
            # Initialize Signal_Adjusted_Timestamp column (for tracking 5m adjustments)
            df['Signal_Adjusted_Timestamp'] = pd.NaT  # NaT for no signal timestamps
            
            # Mark entry signals with adjusted timestamps
            for entry in entries_list:
                entry_index = entry.get('index', -1)
                if 0 <= entry_index < len(df):
                    df.iloc[entry_index, df.columns.get_loc('Entry_Signal')] = True
                    df.iloc[entry_index, df.columns.get_loc('Strategy_Signals')] = 1
                    df.iloc[entry_index, df.columns.get_loc('Strategy_Signal_Text')] = 'Entry'
                    
                    # Add signal adjusted timestamp for tracking
                    adjusted_timestamp = entry.get('timestamp')  # This is the adjusted timestamp from signal_extractor
                    if adjusted_timestamp:
                        df.iloc[entry_index, df.columns.get_loc('Signal_Adjusted_Timestamp')] = adjusted_timestamp
            
            # Mark exit signals with adjusted timestamps
            for exit in exits_list:
                exit_index = exit.get('index', -1)
                if 0 <= exit_index < len(df):
                    df.iloc[exit_index, df.columns.get_loc('Exit_Signal')] = True
                    df.iloc[exit_index, df.columns.get_loc('Strategy_Signals')] = -1
                    df.iloc[exit_index, df.columns.get_loc('Strategy_Signal_Text')] = 'Exit'
                    
                    # Add signal adjusted timestamp for tracking
                    adjusted_timestamp = exit.get('timestamp')  # This is the adjusted timestamp from signal_extractor
                    if adjusted_timestamp:
                        df.iloc[exit_index, df.columns.get_loc('Signal_Adjusted_Timestamp')] = adjusted_timestamp
            
            entry_count = df['Entry_Signal'].sum()
            exit_count = df['Exit_Signal'].sum()
            self.log_detailed(f"Added signals to DataFrame: {entry_count} entries, {exit_count} exits", "INFO")
            
        except Exception as e:
            self.log_detailed(f"Error adding signals to DataFrame: {e}", "ERROR")
    
    def _add_options_data_to_dataframe(self, df: pd.DataFrame, performance_data: Dict[str, Any]) -> None:
        """Add REAL options trading data to the comprehensive DataFrame using actual trade results."""
        try:
            # Initialize options columns with default values
            df['Strategy_Name'] = ''
            df['Case_Name'] = ''
            df['Position_Size'] = 1
            df['Strike_Price'] = 0
            df['Entry_Premium'] = 0.0
            df['Exit_Premium'] = 0.0
            df['Options_PnL'] = 0.0
            df['Option_Type'] = 'CALL'  # Default option type
            df['Commission'] = 0.0
            df['Net_PnL'] = 0.0
            
            # Get REAL trade data from P&L calculation results
            options_pnl = performance_data.get('options_pnl', {})
            total_trades = options_pnl.get('total_trades', 0)
            trades_data = options_pnl.get('trades_data', [])  # Real trade data from P&L calculator
            
            if total_trades > 0 and trades_data:
                # Use configuration data
                config_data = self.main_config.get('options', {})
                default_option_type = config_data.get('default_option_type', 'CALL')
                commission_per_trade = self.main_config.get('trading', {}).get('commission_per_trade', 20.0)
                lot_size = self.main_config.get('trading', {}).get('lot_size', 75)
                
                self.log_detailed(f"Processing REAL trade data: {len(trades_data)} trades from P&L calculator", "INFO")
                
                # Process each trade using exact timestamp matching for proper indexing
                for i, trade_data in enumerate(trades_data):
                    try:
                        # Extract timestamps and find exact matches in DataFrame
                        entry_timestamp = trade_data.get('entry_timestamp')
                        exit_timestamp = trade_data.get('exit_timestamp')
                        
                        if not entry_timestamp or not exit_timestamp:
                            self.log_detailed(f"Trade {i}: Missing timestamps - skipping", "WARNING")
                            continue
                        
                        # Find exact timestamp matches in DataFrame index
                        entry_matches = df.index == entry_timestamp
                        exit_matches = df.index == exit_timestamp
                        
                        if not entry_matches.any():
                            self.log_detailed(f"Trade {i}: Entry timestamp {entry_timestamp} not found in DataFrame", "WARNING")
                            continue
                            
                        if not exit_matches.any():
                            self.log_detailed(f"Trade {i}: Exit timestamp {exit_timestamp} not found in DataFrame", "WARNING")  
                            continue
                        
                        entry_idx = df.index[entry_matches][0]
                        exit_idx = df.index[exit_matches][0]
                        
                        # Extract REAL trade data with validation
                        entry_premium = float(trade_data.get('entry_premium', 0.0))
                        exit_premium = float(trade_data.get('exit_premium', 0.0))
                        strike_price = int(trade_data.get('entry_strike', trade_data.get('strike', 0)))
                        option_type = str(trade_data.get('option_type', default_option_type))
                        
                        # Extract strategy and case information
                        strategy_name = str(trade_data.get('group', ''))
                        case_name = str(trade_data.get('case', ''))
                        position_size = int(trade_data.get('position_size', 1))
                        
                        # Log premium data for debugging but don't skip trades
                        if entry_premium <= 0 or exit_premium <= 0:
                            self.log_detailed(f"Trade {i}: WARNING - Invalid premium data - Entry: {entry_premium}, Exit: {exit_premium} - Will still export for debugging", "WARNING")
                            # Don't skip - export anyway for debugging
                        
                        # Calculate REAL P&L per trade with correct options logic
                        # For BUY strategy: P&L = (Exit Premium - Entry Premium) * Lot Size * Position Size
                        options_pnl_value = (exit_premium - entry_premium) * lot_size * position_size
                        net_pnl = options_pnl_value - commission_per_trade
                        
                        # Update entry row with REAL data
                        df.loc[entry_idx, 'Strategy_Name'] = strategy_name
                        df.loc[entry_idx, 'Case_Name'] = case_name
                        df.loc[entry_idx, 'Position_Size'] = position_size
                        df.loc[entry_idx, 'Strike_Price'] = strike_price
                        df.loc[entry_idx, 'Entry_Premium'] = entry_premium
                        df.loc[entry_idx, 'Option_Type'] = option_type
                        df.loc[entry_idx, 'Commission'] = commission_per_trade
                        
                        # Update exit row with REAL data
                        df.loc[exit_idx, 'Strategy_Name'] = strategy_name
                        df.loc[exit_idx, 'Case_Name'] = case_name
                        df.loc[exit_idx, 'Position_Size'] = position_size
                        df.loc[exit_idx, 'Strike_Price'] = strike_price
                        df.loc[exit_idx, 'Exit_Premium'] = exit_premium
                        df.loc[exit_idx, 'Options_PnL'] = options_pnl_value  # Real per-trade P&L
                        df.loc[exit_idx, 'Option_Type'] = option_type
                        df.loc[exit_idx, 'Commission'] = commission_per_trade
                        df.loc[exit_idx, 'Net_PnL'] = net_pnl  # Real net P&L after commission
                        
                        self.log_detailed(f"Trade {i}: Updated DataFrame with Entry Premium: {entry_premium}, Exit Premium: {exit_premium}, Strike: {strike_price}", "INFO")
                        
                    except Exception as trade_error:
                        self.log_detailed(f"Error processing trade {i}: {trade_error}", "ERROR")
                        continue
                    
                self.log_detailed(f"Added REAL options data to DataFrame: {len(trades_data)} trades processed", "INFO")
            else:
                self.log_detailed("No real trade data available, using default values", "WARNING")
            
        except Exception as e:
            self.log_detailed(f"Error adding real options data to DataFrame: {e}", "ERROR")
    
    def _export_excel(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Optional[Path]:
        """Export results to Excel format with multiple sheets."""
        try:
            excel_path = output_dir / f"{filename_base}.xlsx"
            
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Export summary sheet
                summary_data = self._create_summary_sheet(results)
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                
                # Export trade pairs
                trade_pairs = results.get('signals', {}).get('trade_pairs', [])
                if trade_pairs:
                    trades_df = pd.DataFrame(trade_pairs)
                    trades_df = self._flatten_dataframe(trades_df)
                    trades_df.to_excel(writer, sheet_name='Trades', index=False)
                
                # Export performance metrics
                performance_data = results.get('performance', {})
                if performance_data:
                    perf_records = self._flatten_dict_to_records(performance_data)
                    if perf_records:
                        perf_df = pd.DataFrame(perf_records)
                        perf_df.to_excel(writer, sheet_name='Performance', index=False)
                
                # Export market data summary
                market_data = results.get('market_data', {})
                if market_data:
                    market_records = self._flatten_dict_to_records(market_data)
                    if market_records:
                        market_df = pd.DataFrame(market_records)
                        market_df.to_excel(writer, sheet_name='Market_Data', index=False)
                
                # Export indicators summary
                indicators_data = results.get('indicators', {})
                if indicators_data:
                    indicators_records = self._flatten_dict_to_records(indicators_data)
                    if indicators_records:
                        indicators_df = pd.DataFrame(indicators_records)
                        indicators_df.to_excel(writer, sheet_name='Indicators', index=False)
            
            self.log_detailed(f"Excel export completed: {excel_path.name}", "INFO")
            return excel_path
            
        except Exception as e:
            self.log_detailed(f"Error exporting Excel: {e}", "ERROR")
            return None
    
    def _create_summary_sheet(self, results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create summary sheet data for Excel export."""
        try:
            summary_data = []
            
            # Add system info
            system_info = results.get('system_info', {})
            for key, value in system_info.items():
                summary_data.append({
                    'Category': 'System',
                    'Metric': key.replace('_', ' ').title(),
                    'Value': str(value)
                })
            
            # Add key performance metrics
            performance = results.get('performance', {})
            options_pnl = performance.get('options_pnl', {})
            
            for key, value in options_pnl.items():
                if isinstance(value, (int, float)):
                    summary_data.append({
                        'Category': 'Performance',
                        'Metric': key.replace('_', ' ').title(),
                        'Value': f"{value:,.2f}" if isinstance(value, float) else str(value)
                    })
            
            # Add market data summary
            market_data = results.get('market_data', {})
            if 'total_candles' in market_data:
                summary_data.append({
                    'Category': 'Market Data',
                    'Metric': 'Total Candles',
                    'Value': str(market_data['total_candles'])
                })
            
            # Add signals summary
            signals = results.get('signals', {})
            signal_counts = signals.get('signal_counts', {})
            for key, value in signal_counts.items():
                summary_data.append({
                    'Category': 'Signals',
                    'Metric': key.replace('_', ' ').title(),
                    'Value': str(value)
                })
            
            return summary_data
            
        except Exception as e:
            self.log_detailed(f"Error creating summary sheet: {e}", "ERROR")
            return []
    
    def _convert_for_json(self, obj: Any) -> Any:
        """Convert numpy types and other non-serializable types for JSON."""
        if isinstance(obj, dict):
            return {key: self._convert_for_json(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [self._convert_for_json(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        elif isinstance(obj, (np.bool_, bool)):
            return bool(obj)
        elif isinstance(obj, pd.Timestamp):
            return obj.isoformat()
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, timedelta):
            return str(obj)
        else:
            return obj
    
    def _flatten_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Flatten nested dictionaries in DataFrame columns."""
        try:
            flattened_df = df.copy()
            
            for column in df.columns:
                if df[column].dtype == 'object':
                    # Check if column contains dictionaries
                    sample_value = df[column].iloc[0] if len(df) > 0 else None
                    if isinstance(sample_value, dict):
                        # Expand dictionary into separate columns
                        expanded = pd.json_normalize(df[column])
                        expanded.columns = [f"{column}_{col}" for col in expanded.columns]
                        
                        # Drop original column and add expanded columns
                        flattened_df = flattened_df.drop(columns=[column])
                        flattened_df = pd.concat([flattened_df, expanded], axis=1)
            
            return flattened_df
            
        except Exception as e:
            self.log_detailed(f"Error flattening DataFrame: {e}", "ERROR")
            return df
    
    def _flatten_dict_to_records(self, data: Dict[str, Any], prefix: str = "") -> List[Dict[str, Any]]:
        """Flatten nested dictionary to list of records."""
        try:
            records = []
            
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                
                if isinstance(value, dict):
                    # Recursively flatten nested dictionaries
                    nested_records = self._flatten_dict_to_records(value, full_key)
                    records.extend(nested_records)
                else:
                    records.append({
                        'Metric': full_key.replace('_', ' ').title(),
                        'Value': str(value) if not isinstance(value, (int, float)) else value,
                        'Type': type(value).__name__
                    })
            
            return records
            
        except Exception as e:
            self.log_detailed(f"Error flattening dictionary: {e}", "ERROR")
            return []
    
    def _display_export_summary(self, exported_files: Dict[str, str], output_dir: Path) -> None:
        """Display export summary to console."""
        try:
            if not exported_files:
                print("Warning: No files were exported")
                return
            
            print(f"\nResults exported to: {output_dir}")
            
            for file_type, file_path in exported_files.items():
                file_name = Path(file_path).name
                file_size = Path(file_path).stat().st_size if Path(file_path).exists() else 0
                
                if 'json' in file_type:
                    print(f"   JSON: {file_name} ({file_size:,} bytes)")
                elif 'csv' in file_type:
                    print(f"   CSV: {file_name} ({file_size:,} bytes)")
                elif 'excel' in file_type:
                    print(f"   Excel: {file_name} ({file_size:,} bytes)")
                else:
                    print(f"   {file_type.upper()}: {file_name} ({file_size:,} bytes)")
            
        except Exception as e:
            self.log_detailed(f"Error displaying export summary: {e}", "ERROR")
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for results export."""
        return {
            'processor': self.processor_name,
            'performance_stats': self.performance_stats,
            'export_config': {
                'export_path': self.export_path,
                'enabled_formats': self.enabled_formats
            }
        }
    
    def _export_trade_comparison_csv(self, trade_comparison_data: Dict[str, Any], 
                                   output_dir: Path, filename_base: str) -> Optional[Path]:
        """
        Export trade comparison results to CSV.
        
        Args:
            trade_comparison_data: Dictionary containing comparison results and statistics
            output_dir: Output directory path
            filename_base: Base filename for export
            
        Returns:
            Path to exported comparison CSV file or None if failed
        """
        try:
            self.log_detailed("[INIT] Starting _export_trade_comparison_csv method", "INFO")
            
            # Extract comparison results
            comparison_results = trade_comparison_data.get('comparison_results', [])
            self.log_detailed(f"Extracted comparison_results: {len(comparison_results) if comparison_results else 0} records", "INFO")
            
            if not comparison_results:
                self.log_detailed("[ERROR] No trade comparison results to export", "WARNING")
                self.log_detailed(f"Available keys in trade_comparison_data: {list(trade_comparison_data.keys()) if isinstance(trade_comparison_data, dict) else 'Not a dict'}", "WARNING")
                return None
                
            # Create comparison DataFrame
            comparison_df = pd.DataFrame(comparison_results)
            
            # Define ordered columns for logical grouping and better readability
            ordered_columns = [
                # 1. TIMESTAMPS & TIME COMPARISONS (All time-related data first)
                'entry_time_vbt', 'entry_time_quantman', 
                'exit_time_vbt', 'exit_time_quantman',
                'entry_time_match', 'exit_time_match', 
                'entry_time_diff_minutes', 'exit_time_diff_minutes',
                
                # 2. STRIKE COMPARISONS  
                'strike_vbt', 'strike_quantman', 'strike_match',
                
                # 3. PREMIUM COMPARISONS
                'entry_premium_vbt', 'entry_premium_quantman', 'entry_premium_diff', 'entry_premium_within_tolerance',
                'exit_premium_vbt', 'exit_premium_quantman', 'exit_premium_diff', 'exit_premium_within_tolerance',
                
                # 4. P&L COMPARISONS
                'pnl_vbt', 'pnl_quantman', 'pnl_diff', 'pnl_percentage_diff', 'pnl_within_tolerance',
                
                # 5. OPTION TYPE COMPARISON
                'option_type_vbt', 'option_type_quantman', 'option_type_match',
                
                # 6. TRADE IDENTIFICATION (Last)
                'trade_status', 'vbt_trade_id', 'quantman_transaction'
            ]
            
            # Reorder columns if they exist
            available_columns = [col for col in ordered_columns if col in comparison_df.columns]
            if available_columns:
                comparison_df = comparison_df[available_columns]
            
            # Create comparison CSV filename
            csv_filename = f"{filename_base}_trade_comparison.csv"
            csv_path = output_dir / csv_filename
            
            # Log exact export path for debugging
            self.log_detailed(f"Trade comparison CSV absolute path: {csv_path.absolute()}", "INFO")
            self.log_detailed(f"Output directory exists: {output_dir.exists()}", "INFO")
            self.log_detailed(f"Output directory absolute path: {output_dir.absolute()}", "INFO")
            
            # Export to CSV
            comparison_df.to_csv(csv_path, index=False)
            
            # Verify file was created
            if csv_path.exists():
                file_size = csv_path.stat().st_size
                self.log_detailed(f"Trade comparison CSV successfully created: {csv_path.name} ({file_size} bytes)", "INFO")
            else:
                self.log_detailed(f"Trade comparison CSV creation failed: {csv_path}", "ERROR")
            
            # Log comparison statistics
            comparison_stats = trade_comparison_data.get('comparison_stats', {})
            self.log_detailed(f"Trade comparison exported: {len(comparison_results)} comparisons", "INFO")
            self.log_detailed(f"Match rate: {comparison_stats.get('match_rate', 0):.2%}", "INFO")
            self.log_detailed(f"Missing in VBT: {comparison_stats.get('missing_in_vbt', 0)}", "INFO")
            
            self.log_major_component(
                f"Trade comparison CSV exported: {csv_filename}",
                "RESULTS_EXPORT"
            )
            
            return csv_path
            
        except Exception as e:
            self.log_detailed(f"Error exporting trade comparison CSV: {e}", "ERROR")
            return None


# Convenience function for external use
def create_results_exporter(config_loader, logger) -> ResultsExporter:
    """Create and initialize Results Exporter."""
    return ResultsExporter(config_loader, logger)


if __name__ == "__main__":
    # Test Results Exporter
    try:
        from core.json_config_loader import JSONConfigLoader
        
        print("Testing Results Exporter...")
        
        # Create components
        config_loader = JSONConfigLoader()
        
        # Create fallback logger
        class TestLogger:
            def log_major_component(self, msg, comp): print(f"RESULTS_EXPORTER: {msg}")
            def log_detailed(self, msg, level, comp): print(f"DETAIL: {msg}")
            def log_performance(self, metrics, comp): print(f"PERFORMANCE: {metrics}")
        
        logger = TestLogger()
        
        # Create results exporter
        exporter = ResultsExporter(config_loader, logger)
        
        # Test with sample data
        sample_results = {
            'system_info': {'version': '2.0', 'architecture': 'ultimate_efficiency'},
            'performance': {'options_pnl': {'net_pnl': 15000.50, 'total_trades': 10}},
            'signals': {'signal_counts': {'total_entries': 5, 'total_exits': 5}}
        }
        
        # Test export (will create test files)
        exported = exporter.export_results(
            sample_results, 
            "NIFTY", 
            "5m", 
            "2024-06-01", 
            "2024-06-03",
            "test"
        )
        
        print(f"\nResults Exporter test completed!")
        print(f"   Performance: {exporter.get_performance_summary()}")
        print(f"   Exported Files: {list(exported.keys())}")
        
    except Exception as e:
        print(f"Results Exporter test failed: {e}")
        sys.exit(1)