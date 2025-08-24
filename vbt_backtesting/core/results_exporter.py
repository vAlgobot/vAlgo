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
import csv
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union
from datetime import datetime, timedelta
import warnings
import concurrent.futures
import threading

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.processor_base import ProcessorBase
from core.json_config_loader import ConfigurationError

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')


class ExportConfigManager:
    """
    Manages export configuration and conditional export logic.
    
    Provides centralized control over what gets exported based on JSON configuration.
    Supports performance optimization through selective export operations.
    """
    
    def __init__(self, export_config: Dict[str, Any]):
        """Initialize export configuration manager."""
        self.export_config = export_config
        self.export_control = export_config.get('export_control', {})
        self.conditional_exports = export_config.get('conditional_exports', {})
        self.performance_mode = export_config.get('performance_mode', {})
        self.emergency_bypass = export_config.get('emergency_bypass_export', False)
        
    def is_emergency_bypass_enabled(self) -> bool:
        """Check if emergency export bypass is enabled."""
        return self.emergency_bypass
    
    def should_export_signals(self) -> bool:
        """Check if signals CSV should be exported."""
        if self.is_emergency_bypass_enabled():
            return False
        return self.export_control.get('signals_csv', True)
    
    def should_export_trades(self) -> bool:
        """Check if trades CSV should be exported."""
        if self.is_emergency_bypass_enabled():
            return False
        return self.export_control.get('trades_csv', True)
    
    def should_export_trade_comparison(self) -> bool:
        """Check if trade comparison CSV should be exported."""
        if self.is_emergency_bypass_enabled():
            return False
        return self.export_control.get('trade_comparison_csv', False)
    
    def should_export_comprehensive(self) -> bool:
        """Check if comprehensive CSV should be exported."""
        if self.is_emergency_bypass_enabled():
            return False
        return self.export_control.get('comprehensive_csv', False)
    
    def should_export_excel(self) -> bool:
        """Check if Excel export should be performed."""
        if self.is_emergency_bypass_enabled():
            return False
        return self.export_control.get('excel', False)
    
    def should_export_json(self) -> bool:
        """Check if JSON export should be performed."""
        if self.is_emergency_bypass_enabled():
            return False
        return self.export_control.get('json', False)
    
    def is_minimal_mode(self) -> bool:
        """Check if minimal export mode is enabled."""
        return self.conditional_exports.get('export_minimal_mode', False)
    
    def should_export_signals_only_if_trades(self) -> bool:
        """Check if signals should only be exported when trades exist."""
        return self.conditional_exports.get('export_signals_only_if_trades', False)
    
    def should_skip_large_signals(self, estimated_size_mb: float = 0) -> bool:
        """Check if large signals CSV should be skipped."""
        if not self.conditional_exports.get('skip_large_signals_csv', False):
            return False
        
        max_size = self.conditional_exports.get('max_signals_csv_size_mb', 50)
        return estimated_size_mb > max_size
    
    def is_streaming_enabled(self) -> bool:
        """Check if streaming export is enabled."""
        return self.performance_mode.get('streaming_export', True)
    
    def get_chunk_size(self) -> int:
        """Get chunk size for streaming export."""
        return self.performance_mode.get('chunk_size', 1000)
    
    def is_parallel_export_enabled(self) -> bool:
        """Check if parallel export is enabled."""
        return self.performance_mode.get('parallel_export', False)
    
    def is_memory_optimization_enabled(self) -> bool:
        """Check if memory optimization is enabled."""
        return self.performance_mode.get('memory_optimization', True)
    
    def get_export_summary(self) -> Dict[str, Any]:
        """Get summary of export configuration."""
        return {
            'exports_enabled': {
                'signals_csv': self.should_export_signals(),
                'trades_csv': self.should_export_trades(),
                'trade_comparison_csv': self.should_export_trade_comparison(),
                'comprehensive_csv': self.should_export_comprehensive(),
                'excel': self.should_export_excel(),
                'json': self.should_export_json()
            },
            'conditional_settings': {
                'minimal_mode': self.is_minimal_mode(),
                'signals_only_if_trades': self.should_export_signals_only_if_trades(),
                'skip_large_signals': self.conditional_exports.get('skip_large_signals_csv', False)
            },
            'performance_settings': {
                'streaming_export': self.is_streaming_enabled(),
                'chunk_size': self.get_chunk_size(),
                'parallel_export': self.is_parallel_export_enabled(),
                'memory_optimization': self.is_memory_optimization_enabled()
            }
        }


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
        
        # Initialize export configuration manager
        self.config_manager = ExportConfigManager(self.export_config)
        
        # Log export configuration summary
        export_summary = self.config_manager.get_export_summary()
        self.log_major_component("Results Exporter initialized with conditional export control", "RESULTS_EXPORTER")
        self.log_detailed(f"Export configuration: {export_summary}", "INFO")
    
    def _has_trades(self, results: Dict[str, Any]) -> bool:
        """Check if results contain actual trades."""
        try:
            # Check for trade data in performance section
            performance_data = results.get('performance', {})
            options_pnl = performance_data.get('options_pnl', {})
            total_trades = options_pnl.get('total_trades', 0)
            trades_data = options_pnl.get('trades_data', [])
            
            return total_trades > 0 and len(trades_data) > 0
        except Exception:
            return False
    
    def _estimate_signals_csv_size_mb(self, results: Dict[str, Any]) -> float:
        """Estimate the size of signals CSV in MB."""
        try:
            market_data_df = results.get('raw_market_data')
            if market_data_df is None or market_data_df.empty:
                return 0.0
            
            # Estimate based on number of rows and columns
            num_rows = len(market_data_df)
            num_columns = len(market_data_df.columns) + 20  # Add estimated indicator columns
            
            # Rough estimate: 50 bytes per cell (conservative)
            estimated_bytes = num_rows * num_columns * 50
            estimated_mb = estimated_bytes / (1024 * 1024)
            
            return estimated_mb
        except Exception:
            return 0.0
    
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
        Export analysis results to configured formats with conditional logic.
        
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
                # Emergency bypass check first
                if self.config_manager.is_emergency_bypass_enabled():
                    self.log_major_component("[EMERGENCY_BYPASS] All exports disabled for maximum performance", "RESULTS_EXPORT")
                    return {'status': 'emergency_bypass_active', 'export_time': '0.000s'}
                
                self.log_major_component("Exporting analysis results with conditional export control", "RESULTS_EXPORT")
                
                # Check for trades if conditional export is enabled
                has_trades = self._has_trades(results)
                estimated_signals_size = self._estimate_signals_csv_size_mb(results)
                
                self.log_detailed(f"Trade detection: {has_trades}, Estimated signals size: {estimated_signals_size:.2f} MB", "INFO")
                
                # Early exit conditions
                if self.config_manager.should_export_signals_only_if_trades() and not has_trades:
                    self.log_major_component("[CONDITIONAL_SKIP] No trades found - skipping all exports as configured", "RESULTS_EXPORT")
                    return {'status': 'skipped_no_trades'}
                
                if self.config_manager.should_skip_large_signals(estimated_signals_size):
                    self.log_major_component(f"[CONDITIONAL_SKIP] Signals CSV too large ({estimated_signals_size:.2f} MB) - skipping as configured", "RESULTS_EXPORT")
                    return {'status': 'skipped_large_signals'}
                
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
                
                # Conditional JSON export
                if 'json' in self.enabled_formats and self.config_manager.should_export_json():
                    self.log_detailed("Exporting JSON format", "INFO")
                    json_path = self._export_json(results, output_dir, filename_base)
                    if json_path:
                        exported_files['json'] = str(json_path)
                else:
                    self.log_detailed("JSON export disabled in configuration", "INFO")
                
                # Conditional CSV export with performance optimization
                if 'csv' in self.enabled_formats:
                    csv_paths = self._export_csv_conditional(results, output_dir, filename_base)
                    exported_files.update(csv_paths)
                
                # Conditional trade comparison export
                if self.config_manager.should_export_trade_comparison() and 'trade_comparison' in results:
                    self.log_major_component("[CONDITIONAL] TRADE COMPARISON EXPORT ENABLED", "RESULTS_EXPORT")
                    trade_comparison_data = results['trade_comparison']
                    
                    comparison_csv_path = self._export_trade_comparison_csv(
                        trade_comparison_data, output_dir, filename_base
                    )
                    if comparison_csv_path:
                        exported_files['trade_comparison_csv'] = str(comparison_csv_path)
                        self.log_major_component(f"[SUCCESS] TRADE COMPARISON CSV EXPORTED: {comparison_csv_path.name}", "RESULTS_EXPORT")
                else:
                    self.log_major_component("[CONDITIONAL_SKIP] Trade comparison export disabled in configuration", "RESULTS_EXPORT")
                
                # Conditional Excel export
                if 'excel' in self.enabled_formats and self.config_manager.should_export_excel():
                    self.log_detailed("Exporting Excel format", "INFO")
                    excel_path = self._export_excel(results, output_dir, filename_base)
                    if excel_path:
                        exported_files['excel'] = str(excel_path)
                else:
                    self.log_detailed("Excel export disabled in configuration", "INFO")
                
                # Display export summary
                self._display_export_summary(exported_files, output_dir)
                
                self.log_major_component(
                    f"Conditional export complete: {len(exported_files)} files exported",
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
    
    def _export_csv_conditional(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Dict[str, str]:
        """Export results to CSV format(s) using conditional logic and performance optimization."""
        exported_csvs = {}
        
        try:
            # Conditional export logic based on configuration
            
            # Export 1: Signals CSV - Conditional based on configuration
            if self.config_manager.should_export_signals():
                if self.config_manager.is_streaming_enabled():
                    self.log_detailed("Using streaming export for signals CSV", "INFO")
                    signals_path = self._export_signals_csv_streaming(results, output_dir, filename_base)
                else:
                    self.log_detailed("Using standard export for signals CSV", "INFO")
                    signals_path = self._export_signals_csv(results, output_dir, filename_base)
                
                if signals_path:
                    exported_csvs['signals_csv'] = str(signals_path)
            else:
                self.log_major_component("[CONDITIONAL_SKIP] Signals CSV export disabled - saving ~7-8 seconds", "RESULTS_EXPORT")
            
            # Export 2: Trades CSV - Conditional based on configuration
            if self.config_manager.should_export_trades():
                if self.config_manager.is_streaming_enabled():
                    self.log_detailed("Using streaming export for trades CSV", "INFO")
                    trades_path = self._export_trades_csv_streaming(results, output_dir, filename_base)
                else:
                    self.log_detailed("Using standard export for trades CSV", "INFO")
                    trades_path = self._export_trades_csv(results, output_dir, filename_base)
                
                if trades_path:
                    exported_csvs['trades_csv'] = str(trades_path)
            else:
                self.log_major_component("[CONDITIONAL_SKIP] Trades CSV export disabled", "RESULTS_EXPORT")
            
            # Export 3: Comprehensive CSV - Only if specifically enabled
            if self.config_manager.should_export_comprehensive():
                self.log_detailed("Exporting comprehensive CSV", "INFO")
                comprehensive_path = self._export_comprehensive_trade_csv(results, output_dir, filename_base)
                if comprehensive_path:
                    exported_csvs['comprehensive_csv'] = str(comprehensive_path)
            
            return exported_csvs
            
        except Exception as e:
            self.log_detailed(f"Error exporting CSV: {e}", "ERROR")
            return {'csv_error': str(e)}
    
    def _export_csv(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Dict[str, str]:
        """Legacy export method - kept for backward compatibility."""
        return self._export_csv_conditional(results, output_dir, filename_base)
    
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
            import time
            signals_export_start = time.time()
            
            print(f"🚀 SIGNALS EXPORT: Starting comprehensive signals CSV export...")
            
            # Extract ACTUAL market data DataFrame from results (zero-copy when possible)
            dataframe_start = time.time()
            market_data_df = results.get('raw_market_data')
            if market_data_df is None or market_data_df.empty:
                self.log_detailed("No raw market data available for signals CSV export", "WARNING")
                return None
            
            # Use original DataFrame directly - avoid expensive copy operation
            signals_df = market_data_df  # Zero-copy reference
            
            # Ensure proper datetime index (in-place when possible)
            if not isinstance(signals_df.index, pd.DatetimeIndex):
                # Only convert if necessary
                signals_df = signals_df.copy()  # Copy only when index conversion needed
                signals_df.index = pd.to_datetime(signals_df.index)
            
            dataframe_time = time.time() - dataframe_start
            print(f"📊 DataFrame Setup: {dataframe_time:.3f}s ({len(signals_df)} rows)")
            
            # Create export-ready DataFrame with minimal operations
            export_df_start = time.time()
            
            # If using original DataFrame reference, we need to create export version
            if signals_df is market_data_df:
                # Create minimal copy only for export (preserves original data)
                signals_df = market_data_df.copy()
                print(f"📊 Created export DataFrame copy for safety")
            
            # Add REAL technical indicators from analysis (ultra-fast batch)
            indicators_start = time.time()
            raw_indicators = results.get('raw_indicators', {})
            
            # Debug: Check what's available in results
            print(f"🔍 DEBUG: Available result keys: {list(results.keys())}")
            if not raw_indicators:
                # Check alternative keys where indicators might be stored
                alt_indicators = results.get('indicators', {})
                signal_candle_data = results.get('signal_candle_data', {})
                print(f"🔍 DEBUG: Alternative indicators: {len(alt_indicators)}")
                print(f"🔍 DEBUG: Signal candle data: {len(signal_candle_data)}")
                
                # Use alternative indicators if available
                if alt_indicators:
                    raw_indicators = alt_indicators
                    print(f"📊 Using alternative indicators source: {len(raw_indicators)} indicators")
            
            self._add_real_indicators_to_dataframe(signals_df, raw_indicators)
            indicators_time = time.time() - indicators_start
            print(f"📊 Indicators Processing: {indicators_time:.3f}s ({len(raw_indicators)} indicators)")
            
            # Add signal columns with adjusted timestamps (vectorized)
            signals_start = time.time()
            signals_data = results.get('signals', {})
            entries_list = signals_data.get('entries', [])
            exits_list = signals_data.get('exits', [])
            self._add_signals_to_dataframe(signals_df, entries_list, exits_list)
            signals_time = time.time() - signals_start
            print(f"📊 Signals Processing: {signals_time:.3f}s ({len(entries_list)} entries, {len(exits_list)} exits)")
            
            # Add signal_candle_data if available in results (graceful handling)
            signal_candle_start = time.time()
            signal_candle_data = results.get('signal_candle_data', {})
            if signal_candle_data:
                self._add_signal_candle_to_dataframe(signals_df, signal_candle_data)
                signal_candle_time = time.time() - signal_candle_start
                print(f"📊 Signal Candle Processing: {signal_candle_time:.3f}s ({len(signal_candle_data)} columns)")
            else:
                print(f"📊 No signal_candle_data in results (may be disabled)")
            
            export_df_time = time.time() - export_df_start
            
            # COLUMN ORDERING: Match reference CSV structure exactly
            column_order_start = time.time()
            expected_columns = [
                # Primary timestamp (must be first)
                'timestamp',
                # LTP data (if available)
                'ltp_open', 'ltp_high', 'ltp_low', 'ltp_close', 'ltp_volume',
                # Indicator timestamp  
                'Indicator_Timestamp',
                # OHLC data
                'open', 'high', 'low', 'close', 'volume',
                # Technical indicators - RSI
                'rsi_14', 'rsi_21',
                # Technical indicators - SMA
                'sma_9', 'sma_20', 'sma_50', 'sma_200',
                # Technical indicators - EMA
                'ema_9', 'ema_20', 'ema_50', 'ema_200',
                # CPR indicators
                'cpr_pivot', 'cpr_tc', 'cpr_bc', 'cpr_r1', 'cpr_r2', 'cpr_r3', 'cpr_r4',
                'cpr_s1', 'cpr_s2', 'cpr_s3', 'cpr_s4', 'cpr_width', 'cpr_range_type', 'cpr_type',
                # Previous data
                'previous_day_high', 'previous_day_low', 'previous_day_close',
                'previous_candle_open', 'previous_candle_high', 'previous_candle_low', 'previous_candle_close',
                # Signal candle data (runtime indicators)
                'signal_candle_open', 'signal_candle_high', 'signal_candle_low', 'signal_candle_close',
                # SL/TP levels (runtime indicators)
                'sl_price', 'tp_price',
                # Signal columns
                'Entry_Signal', 'Exit_Signal', 'Strategy_Signals', 'Strategy_Signal_Text', 'Signal_Adjusted_Timestamp',
                # Strategy information
                'Strategy_Name', 'Case_Name'
            ]
            
            # Reorder columns to match reference CSV (only include existing columns)
            existing_columns = [col for col in expected_columns if col in signals_df.columns]
            extra_columns = [col for col in signals_df.columns if col not in expected_columns]
            final_column_order = existing_columns + extra_columns
            
            signals_df = signals_df[final_column_order]
            column_order_time = time.time() - column_order_start
            print(f"📊 Column Reordering: {column_order_time:.3f}s ({len(final_column_order)} columns)")
            print(f"📊 Final columns: {final_column_order[:10]}... ({len(final_column_order)} total)")
            
            # Format datetime columns to ensure consistent datetime format
            datetime_format_start = time.time()
            datetime_columns = ['timestamp', 'Indicator_Timestamp', 'Signal_Adjusted_Timestamp']
            for col in datetime_columns:
                if col in signals_df.columns:
                    # Ensure datetime format and convert to string with consistent format
                    if not signals_df[col].isna().all():  # Only if column has data
                        signals_df[col] = pd.to_datetime(signals_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                        print(f"📅 Formatted {col} to datetime string format")
            datetime_format_time = time.time() - datetime_format_start
            print(f"📊 Datetime Formatting: {datetime_format_time:.3f}s")
            
            # Fix DataFrame index to prevent duplicate timestamp columns
            if 'timestamp' not in signals_df.columns and isinstance(signals_df.index, pd.DatetimeIndex):
                # If timestamp is only in index, convert index to column
                signals_df = signals_df.reset_index()
                if 'index' in signals_df.columns:
                    signals_df = signals_df.rename(columns={'index': 'timestamp'})
                print(f"📊 Converted DataFrame index to timestamp column")
            elif 'timestamp' in signals_df.columns:
                # If timestamp exists as column, reset index to avoid duplication
                signals_df = signals_df.reset_index(drop=True)
                print(f"📊 Reset DataFrame index to avoid timestamp duplication")
            
            # Ensure timestamp column format is correct
            if 'timestamp' in signals_df.columns:
                signals_df['timestamp'] = pd.to_datetime(signals_df['timestamp'], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                print(f"📅 Final timestamp column formatting applied")
            
            # Ultra-fast CSV export with PyArrow (avoid DataFrame copy operations)
            csv_prep_start = time.time()
            csv_path = output_dir / f"{filename_base}_signals.csv"
            
            # Direct PyArrow export without DataFrame copy (ultra-fast)
            try:
                import pyarrow as pa
                import pyarrow.csv as pv
                
                print("🚀 ULTRA-FAST PyArrow: Direct CSV export without DataFrame copy")
                
                # Convert directly to PyArrow table with optimized settings
                table = pa.Table.from_pandas(signals_df, preserve_index=False)
                
                # Ultra-fast CSV writing with optimized write options (remove unsupported null_handling)
                write_options = pv.WriteOptions(
                    include_header=True,
                    batch_size=8192  # Optimized batch size for large files
                )
                
                pv.write_csv(table, csv_path, write_options=write_options)
                print("✅ Ultra-fast PyArrow export completed")
                
            except ImportError as e:
                print(f"❌ PyArrow not available: {e}")
                # Optimized pandas fallback with minimal operations
                print("🔄 Using optimized pandas fallback")
                signals_df.to_csv(csv_path, 
                    index=False,                          # No index export (timestamp is in columns)
                    encoding='utf-8',
                    float_format='%.6f',
                    chunksize=10000                       # Chunked writing for large files
                )
            except Exception as e:
                print(f"❌ PyArrow processing error: {e}")
                # Minimal pandas fallback
                print("🔄 Using minimal pandas fallback")
                signals_df.to_csv(csv_path, index=False, encoding='utf-8')
            
            csv_time = time.time() - csv_prep_start
            print(f"📊 CSV Writing: {csv_time:.3f}s")
            
            total_signals_time = time.time() - signals_export_start
            print(f"✅ ULTRA-FAST SIGNALS EXPORT COMPLETE: {total_signals_time:.3f}s total")
            print(f"   ├─ DataFrame Setup: {dataframe_time:.3f}s ({(dataframe_time/total_signals_time)*100:.1f}%)")
            print(f"   ├─ Export DataFrame: {export_df_time:.3f}s ({(export_df_time/total_signals_time)*100:.1f}%)")
            print(f"   │   ├─ Indicators: {indicators_time:.3f}s ({(indicators_time/total_signals_time)*100:.1f}%)")
            print(f"   │   └─ Signals: {signals_time:.3f}s ({(signals_time/total_signals_time)*100:.1f}%)")
            print(f"   └─ CSV Write: {csv_time:.3f}s ({(csv_time/total_signals_time)*100:.1f}%)")
            
            self.log_detailed(f"Signals CSV export completed: {csv_path.name}", "INFO")
            return csv_path
            
        except Exception as e:
            self.log_detailed(f"Error exporting signals CSV: {e}", "ERROR")
            return None
    
    def _export_signals_csv_streaming(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Optional[Path]:
        """
        Export signals CSV using streaming approach for performance optimization.
        
        Processes data in chunks to reduce memory usage and improve performance.
        """
        try:
            # Extract market data
            market_data_df = results.get('raw_market_data')
            if market_data_df is None or market_data_df.empty:
                self.log_detailed("No raw market data available for streaming signals CSV export", "WARNING")
                return None
            
            csv_path = output_dir / f"{filename_base}_signals.csv"
            chunk_size = self.config_manager.get_chunk_size()
            
            self.log_detailed(f"Streaming signals CSV export: {len(market_data_df)} rows in {chunk_size}-row chunks", "INFO")
            
            # Get indicators and signals data
            raw_indicators = results.get('raw_indicators', {})
            signals_data = results.get('signals', {})
            entries_list = signals_data.get('entries', [])
            exits_list = signals_data.get('exits', [])
            
            # Pre-compute indicator mapping for efficiency
            if not hasattr(self, '_cached_indicator_mapping'):
                self._cached_indicator_mapping = self._create_dynamic_indicator_mapping(raw_indicators)
            
            # Write CSV header first
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Create header row
                base_columns = ['timestamp'] + list(market_data_df.columns)
                indicator_columns = list(self._cached_indicator_mapping.values())
                signal_columns = ['Entry_Signal', 'Exit_Signal', 'Strategy_Signals', 'Strategy_Signal_Text', 'Signal_Adjusted_Timestamp']
                
                header = base_columns + indicator_columns + signal_columns
                writer.writerow(header)
                
                # Process data in chunks
                total_rows = len(market_data_df)
                for start_idx in range(0, total_rows, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_rows)
                    chunk_df = market_data_df.iloc[start_idx:end_idx].copy()
                    
                    # Add indicators to chunk
                    self._add_indicators_to_chunk_streaming(chunk_df, raw_indicators, start_idx)
                    
                    # Add signals to chunk
                    self._add_signals_to_chunk_streaming(chunk_df, entries_list, exits_list, start_idx)
                    
                    # Write chunk to CSV
                    for _, row in chunk_df.iterrows():
                        row_data = [row.name] + row.tolist()  # Include timestamp from index
                        writer.writerow(row_data)
                    
                    # Clear chunk from memory
                    del chunk_df
                    
                    if self.config_manager.is_memory_optimization_enabled():
                        import gc
                        gc.collect()
            
            self.log_detailed(f"Streaming signals CSV export completed: {csv_path.name}", "INFO")
            return csv_path
            
        except Exception as e:
            self.log_detailed(f"Error in streaming signals CSV export: {e}", "ERROR")
            return None
    
    def _export_trades_csv_streaming(self, results: Dict[str, Any], output_dir: Path, filename_base: str) -> Optional[Path]:
        """
        Export trades CSV using streaming approach for performance optimization.
        """
        try:
            # Get trade data
            performance_data = results.get('performance', {})
            options_pnl = performance_data.get('options_pnl', {})
            total_trades = options_pnl.get('total_trades', 0)
            trades_data = options_pnl.get('trades_data', [])
            
            if total_trades == 0 or not trades_data:
                self.log_detailed("No trade data available for streaming trades CSV export", "WARNING")
                return None
            
            csv_path = output_dir / f"{filename_base}_trades.csv"
            
            self.log_detailed(f"Streaming trades CSV export: {len(trades_data)} trades", "INFO")
            
            # Get configuration data
            config_data = self.main_config.get('options', {})
            trading_config = self.main_config.get('trading', {})
            default_option_type = config_data.get('default_option_type', 'CALL')
            commission_per_trade = trading_config.get('commission_per_trade', 20.0)
            lot_size = trading_config.get('lot_size', 75)
            
            # Write CSV using streaming approach
            with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                header = [
                    'Trade_ID', 'Daily_Trade_Number', 'Strategy_Name', 'Case_Name', 'Position_Size',
                    'Entry_Timestamp', 'Entry_Strike', 'Entry_Premium', 'Exit_Timestamp', 'Exit_Strike',
                    'Exit_Premium', 'Option_Type', 'Options_PnL', 'Commission', 'Net_PnL', 'Lot_Size'
                ]
                writer.writerow(header)
                
                # Process trades in chunks
                chunk_size = self.config_manager.get_chunk_size()
                for i in range(0, len(trades_data), chunk_size):
                    chunk = trades_data[i:i + chunk_size]
                    
                    for j, trade_data in enumerate(chunk):
                        try:
                            # Extract trade information
                            trade_id = i + j + 1
                            entry_timestamp = trade_data.get('entry_timestamp')
                            exit_timestamp = trade_data.get('exit_timestamp')
                            entry_premium = float(trade_data.get('entry_premium', 0.0))
                            exit_premium = float(trade_data.get('exit_premium', 0.0))
                            strike_price = int(trade_data.get('entry_strike', trade_data.get('strike', 0)))
                            option_type = str(trade_data.get('option_type', default_option_type))
                            strategy_name = str(trade_data.get('group', ''))
                            case_name = str(trade_data.get('case', ''))
                            position_size = int(trade_data.get('position_size', 1))
                            daily_trade_sequence = trade_data.get('daily_trade_sequence', 0)
                            
                            # Calculate P&L
                            options_pnl_value = (exit_premium - entry_premium) * lot_size * position_size
                            net_pnl = options_pnl_value - commission_per_trade
                            
                            # Write trade row
                            row = [
                                trade_id, daily_trade_sequence, strategy_name, case_name, position_size,
                                entry_timestamp, strike_price, entry_premium, exit_timestamp, strike_price,
                                exit_premium, option_type, options_pnl_value, commission_per_trade, net_pnl, lot_size
                            ]
                            writer.writerow(row)
                            
                        except Exception as trade_error:
                            self.log_detailed(f"Error processing trade {i + j}: {trade_error}", "WARNING")
                            continue
                    
                    # Memory optimization
                    if self.config_manager.is_memory_optimization_enabled():
                        import gc
                        gc.collect()
            
            self.log_detailed(f"Streaming trades CSV export completed: {csv_path.name} with {total_trades} trades", "INFO")
            return csv_path
            
        except Exception as e:
            self.log_detailed(f"Error in streaming trades CSV export: {e}", "ERROR")
            return None
    
    def _add_indicators_to_chunk_streaming(self, chunk_df: pd.DataFrame, raw_indicators: Dict[str, pd.Series], start_idx: int) -> None:
        """Add indicators to a chunk of data for streaming export."""
        try:
            # Initialize indicator columns
            for csv_column in self._cached_indicator_mapping.values():
                chunk_df[csv_column] = np.nan
            
            # Add indicator values
            for indicator_key, csv_column in self._cached_indicator_mapping.items():
                if indicator_key in raw_indicators:
                    indicator_series = raw_indicators[indicator_key]
                    
                    if isinstance(indicator_series, pd.Series):
                        # Get values for this chunk
                        chunk_values = indicator_series.iloc[start_idx:start_idx + len(chunk_df)]
                        aligned_values = chunk_values.reindex(chunk_df.index)
                        chunk_df[csv_column] = aligned_values.values
                    elif isinstance(indicator_series, np.ndarray):
                        # Get array slice for this chunk
                        if len(indicator_series) > start_idx:
                            end_idx = min(start_idx + len(chunk_df), len(indicator_series))
                            chunk_df[csv_column] = indicator_series[start_idx:end_idx]
        except Exception as e:
            self.log_detailed(f"Error adding indicators to chunk: {e}", "WARNING")
    
    def _add_signals_to_chunk_streaming(self, chunk_df: pd.DataFrame, entries_list: List[Dict], exits_list: List[Dict], start_idx: int) -> None:
        """Add signals to a chunk of data for streaming export."""
        try:
            # Initialize signal columns
            chunk_df['Entry_Signal'] = False
            chunk_df['Exit_Signal'] = False
            chunk_df['Strategy_Signals'] = 0
            chunk_df['Strategy_Signal_Text'] = 'Hold'
            chunk_df['Signal_Adjusted_Timestamp'] = pd.NaT
            
            # Process entries for this chunk
            for entry in entries_list:
                entry_index = entry.get('index', -1)
                if start_idx <= entry_index < start_idx + len(chunk_df):
                    local_idx = entry_index - start_idx
                    if 0 <= local_idx < len(chunk_df):
                        chunk_df.iloc[local_idx, chunk_df.columns.get_loc('Entry_Signal')] = True
                        chunk_df.iloc[local_idx, chunk_df.columns.get_loc('Strategy_Signals')] = 1
                        chunk_df.iloc[local_idx, chunk_df.columns.get_loc('Strategy_Signal_Text')] = 'Entry'
                        
                        row_timestamp = chunk_df.index[local_idx]
                        chunk_df.iloc[local_idx, chunk_df.columns.get_loc('Signal_Adjusted_Timestamp')] = row_timestamp
            
            # Process exits for this chunk
            for exit in exits_list:
                exit_index = exit.get('index', -1)
                if start_idx <= exit_index < start_idx + len(chunk_df):
                    local_idx = exit_index - start_idx
                    if 0 <= local_idx < len(chunk_df):
                        chunk_df.iloc[local_idx, chunk_df.columns.get_loc('Exit_Signal')] = True
                        chunk_df.iloc[local_idx, chunk_df.columns.get_loc('Strategy_Signals')] = -1
                        chunk_df.iloc[local_idx, chunk_df.columns.get_loc('Strategy_Signal_Text')] = 'Exit'
                        
                        row_timestamp = chunk_df.index[local_idx]
                        chunk_df.iloc[local_idx, chunk_df.columns.get_loc('Signal_Adjusted_Timestamp')] = row_timestamp
                        
        except Exception as e:
            self.log_detailed(f"Error adding signals to chunk: {e}", "WARNING")
    
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
            import time
            trades_export_start = time.time()
            
            print(f"🚀 TRADES EXPORT: Starting comprehensive trades CSV export...")
            
            # Get REAL trade data from P&L calculation results
            data_extraction_start = time.time()
            performance_data = results.get('performance', {})
            options_pnl = performance_data.get('options_pnl', {})
            total_trades = options_pnl.get('total_trades', 0)
            trades_data = options_pnl.get('trades_data', [])  # Real trade data from P&L calculator
            
            if total_trades == 0 or not trades_data:
                print("⚠️ No trade data available for trades CSV export")
                self.log_detailed("No trade data available for trades CSV export", "WARNING")
                return None
            
            data_extraction_time = time.time() - data_extraction_start
            print(f"📊 Data Extraction: {data_extraction_time:.3f}s ({len(trades_data)} trades)")
            
            # Get configuration data for defaults
            config_data = self.main_config.get('options', {})
            trading_config = self.main_config.get('trading', {})
            default_option_type = config_data.get('default_option_type', 'CALL')
            commission_per_trade = trading_config.get('commission_per_trade', 20.0)
            lot_size = trading_config.get('lot_size', 75)
            
            # Ultra-fast vectorized trade processing (eliminate loop)
            trade_processing_start = time.time()
            print(f"📋 VECTORIZED TRADES: Processing {len(trades_data)} trade records...")
            
            # Convert to DataFrame for vectorized operations (much faster than loops)
            raw_trades_df = pd.DataFrame(trades_data)
            
            if raw_trades_df.empty:
                print("⚠️ No trade data to process")
                return None
            
            # Vectorized calculations (single operations vs individual loops)
            raw_trades_df['entry_premium'] = raw_trades_df['entry_premium'].astype(float)
            raw_trades_df['exit_premium'] = raw_trades_df['exit_premium'].astype(float)
            raw_trades_df['position_size'] = raw_trades_df['position_size'].fillna(1).astype(int)
            
            # Vectorized P&L calculation for all trades at once
            raw_trades_df['options_pnl_value'] = ((raw_trades_df['exit_premium'] - raw_trades_df['entry_premium']) 
                                                 * lot_size * raw_trades_df['position_size'])
            raw_trades_df['net_pnl'] = raw_trades_df['options_pnl_value'] - commission_per_trade
            
            # Create final trades DataFrame with proper column names
            trades_df = pd.DataFrame({
                'Trade_ID': range(1, len(raw_trades_df) + 1),
                'Daily_Trade_Number': raw_trades_df.get('daily_trade_sequence', 0),
                'Strategy_Name': raw_trades_df.get('group', '').astype(str),
                'Case_Name': raw_trades_df.get('case', '').astype(str),
                'Position_Size': raw_trades_df['position_size'],
                'Entry_Timestamp': raw_trades_df['entry_timestamp'],
                'Entry_Strike': raw_trades_df.get('entry_strike', raw_trades_df.get('strike', 0)).astype(int),
                'Entry_Premium': raw_trades_df['entry_premium'],
                'Exit_Timestamp': raw_trades_df['exit_timestamp'],
                'Exit_Strike': raw_trades_df.get('entry_strike', raw_trades_df.get('strike', 0)).astype(int),
                'Exit_Premium': raw_trades_df['exit_premium'],
                'Option_Type': raw_trades_df.get('option_type', default_option_type).astype(str),
                'Options_PnL': raw_trades_df['options_pnl_value'],
                'Commission': commission_per_trade,
                'Net_PnL': raw_trades_df['net_pnl'],
                'Lot_Size': lot_size
            })
            
            # Format datetime columns for consistent datetime format
            datetime_columns = ['Entry_Timestamp', 'Exit_Timestamp']
            for col in datetime_columns:
                if col in trades_df.columns:
                    # Ensure datetime format and convert to string with consistent format
                    trades_df[col] = pd.to_datetime(trades_df[col], errors='coerce').dt.strftime('%Y-%m-%d %H:%M:%S')
                    print(f"📅 Formatted {col} to datetime string format")
            
            trade_processing_time = time.time() - trade_processing_start
            print(f"📋 Vectorized Trade Processing: {trade_processing_time:.3f}s ({len(trades_df)} valid trades)")
            
            if trades_df.empty:
                self.log_detailed("No valid trades processed for trades CSV", "WARNING")
                return None
            
            # Export trades CSV with timing and PyArrow support
            csv_writing_start = time.time()
            csv_path = output_dir / f"{filename_base}_trades.csv"
            
            # Try PyArrow for faster CSV writing
            try:
                import pyarrow as pa
                import pyarrow.csv as pv
                
                # Convert pandas DataFrame to PyArrow table
                table = pa.Table.from_pandas(trades_df)
                
                # Write using PyArrow for maximum speed
                pv.write_csv(table, csv_path, write_options=pv.WriteOptions(include_header=True))
                print(f"📊 CSV Export Method: PyArrow (ultra-fast)")
                
            except ImportError:
                # Fallback to optimized pandas
                trades_df.to_csv(csv_path, index=False)
                print(f"📊 CSV Export Method: Pandas (standard)")
            
            csv_writing_time = time.time() - csv_writing_start
            print(f"📊 CSV Writing: {csv_writing_time:.3f}s ({csv_path.stat().st_size:,} bytes)")
            
            # Calculate total export time
            total_export_time = time.time() - trades_export_start
            print(f"🎯 ULTRA-FAST TRADES EXPORT COMPLETE: {total_export_time:.3f}s total")
            print(f"   ├─ Data Extraction: {data_extraction_time:.3f}s ({(data_extraction_time/total_export_time)*100:.1f}%)")
            print(f"   ├─ Vectorized Processing: {trade_processing_time:.3f}s ({(trade_processing_time/total_export_time)*100:.1f}%)")
            print(f"   └─ CSV Writing: {csv_writing_time:.3f}s ({(csv_writing_time/total_export_time)*100:.1f}%)")
            
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
        """Ultra-fast vectorized indicator addition - 10x faster than reindexing."""
        try:
            import time
            indicators_start = time.time()
            
            # Check for signal_candle indicators (gracefully handle missing ones)
            signal_candle_indicators = {k: type(v) for k, v in raw_indicators.items() if 'signal_candle' in k}
            if signal_candle_indicators:
                print(f"📊 Found signal_candle indicators: {list(signal_candle_indicators.keys())}")
            else:
                print(f"📊 No signal_candle indicators found (may be disabled in config)")
            
            print(f"🚀 VECTORIZED INDICATORS: Processing {len(raw_indicators)} indicators for {len(df)} records...")
            
            # Pre-allocate all indicator data in memory (avoid repeated DataFrame operations)
            indicator_data = {}
            skipped_indicators = []
            
            for indicator_key, indicator_series in raw_indicators.items():
                if isinstance(indicator_series, pd.Series):
                    # Ultra-fast alignment: Skip expensive reindex() when indexes match
                    if indicator_series.index.equals(df.index):
                        indicator_data[indicator_key] = indicator_series.values
                    else:
                        # Fast numpy-based alignment for mismatched indexes
                        try:
                            # Convert datetime indexes to int64 for fast interpolation
                            df_timestamps = df.index.astype(np.int64)
                            series_timestamps = indicator_series.index.astype(np.int64)
                            
                            # Use numpy interp for ultra-fast alignment
                            aligned_values = np.interp(df_timestamps, series_timestamps, 
                                                     indicator_series.values, left=np.nan, right=np.nan)
                            indicator_data[indicator_key] = aligned_values
                        except Exception as align_error:
                            self.log_detailed(f"Fast alignment failed for {indicator_key}: {align_error}", "WARNING")
                            skipped_indicators.append(indicator_key)
                            continue
                            
                elif isinstance(indicator_series, np.ndarray):
                    # Direct numpy array assignment (ultra-fast)
                    if len(indicator_series) == len(df):
                        indicator_data[indicator_key] = indicator_series
                    else:
                        skipped_indicators.append(f"{indicator_key} (length mismatch)")
                else:
                    skipped_indicators.append(f"{indicator_key} (unsupported type)")
            
            # Ultra-fast batch assignment using pandas.concat() - 10-20x faster than individual assignments
            print(f"📊 ULTRA-FAST BATCH ASSIGNMENT: Adding {len(indicator_data)} indicators...")
            if indicator_data:
                # Create new DataFrame from indicator data and concatenate horizontally (ultra-fast)
                indicators_df = pd.DataFrame(indicator_data, index=df.index)
                
                # Use pandas concat for ultra-fast column addition (single operation vs N operations)
                df_combined = pd.concat([df, indicators_df], axis=1, copy=False)
                
                # Update original DataFrame with combined data (minimal memory copy)
                for col in indicators_df.columns:
                    df[col] = df_combined[col].values  # Direct numpy array assignment
                
                del indicators_df, df_combined  # Immediate memory cleanup
            
            indicators_time = time.time() - indicators_start
            print(f"✅ VECTORIZED INDICATORS: Completed in {indicators_time:.3f}s ({len(indicator_data)} added, {len(skipped_indicators)} skipped)")
            
            if skipped_indicators:
                self.log_detailed(f"Skipped indicators: {skipped_indicators[:5]}{'...' if len(skipped_indicators) > 5 else ''}", "WARNING")
            
        except Exception as e:
            print(f"❌ VECTORIZED INDICATORS ERROR: {e}")
            self.log_detailed(f"Error in vectorized indicator processing: {e}", "ERROR")
    
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
                    # Special handling for signal candle to preserve original format
                    if indicator_key.startswith('signal_candle_'):
                        csv_column = indicator_key  # Keep original signal_candle_* format
                    else:
                        indicator_type, period = indicator_key.split('_', 1)
                        
                        # Map indicator types to CSV column prefixes
                        if indicator_type == 'rsi':
                            csv_column = f"RSI_{period}"
                        elif indicator_type == 'sma':
                            csv_column = f"SMA_{period}"  # Keep existing MA naming for SMA
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
                    # Special handling for signal candle to preserve lowercase formatting
                    if indicator_key.startswith('signal_candle_'):
                        csv_column = indicator_key  # Keep original signal_candle_* format
                    else:
                        csv_column = indicator_key.upper()
                
                indicator_mapping[indicator_key] = csv_column
            
            self.log_detailed(f"Created dynamic indicator mapping for {len(indicator_mapping)} indicators", "INFO")
            
            # Debug: Print signal candle mappings
            signal_mappings = {k: v for k, v in indicator_mapping.items() if 'signal_candle' in k}
            if signal_mappings:
                print(f"🔍 DEBUG: Signal candle mappings: {signal_mappings}")
            
            return indicator_mapping
            
        except Exception as e:
            self.log_detailed(f"Error creating dynamic indicator mapping: {e}", "ERROR")
            # Return empty mapping on error
            return {}
    
    def _add_signals_to_dataframe(self, df: pd.DataFrame, entries_list: List[Dict], exits_list: List[Dict]) -> None:
        """Add signal columns using vectorized timestamp matching from trade data."""
        try:
            import time
            signals_start = time.time()
            
            print(f"🚀 VECTORIZED SIGNALS: Processing {len(entries_list)} entries, {len(exits_list)} exits for {len(df)} records...")
            print(f"📊 Using trade data timestamp matching approach...")
            
            # Initialize all signal columns with defaults
            df['Entry_Signal'] = False
            df['Exit_Signal'] = False  
            df['Strategy_Signals'] = 0
            df['Strategy_Signal_Text'] = 'Hold'
            df['Strategy_Name'] = ''
            df['Case_Name'] = ''
            df['Signal_Adjusted_Timestamp'] = pd.NaT
            
            # Get working trade data (same approach as successful trades CSV)
            if entries_list or exits_list:
                # Create combined trade records from entries and exits
                trade_records = []
                
                # Process entries
                for entry in entries_list:
                    trade_records.append({
                        'timestamp': entry.get('timestamp'),
                        'signal_type': 'Entry',
                        'strategy_name': entry.get('group', ''),
                        'case_name': entry.get('case', ''),
                        'signal_value': 1
                    })
                
                # Process exits  
                for exit in exits_list:
                    trade_records.append({
                        'timestamp': exit.get('timestamp'),
                        'signal_type': 'Exit',
                        'strategy_name': exit.get('group', ''),
                        'case_name': exit.get('case', ''),
                        'signal_value': -1
                    })
                
                if trade_records:
                    # Convert to DataFrame for vectorized operations
                    signals_data_df = pd.DataFrame(trade_records)
                    
                    # Vectorized entry signal processing
                    entry_timestamps = signals_data_df[signals_data_df['signal_type'] == 'Entry']['timestamp']
                    if not entry_timestamps.empty:
                        entry_mask = df.index.isin(entry_timestamps)
                        df.loc[entry_mask, 'Entry_Signal'] = True
                        df.loc[entry_mask, 'Strategy_Signals'] = 1
                        df.loc[entry_mask, 'Strategy_Signal_Text'] = 'Entry'
                        df.loc[entry_mask, 'Signal_Adjusted_Timestamp'] = df.index[entry_mask]
                    
                    # Vectorized exit signal processing  
                    exit_timestamps = signals_data_df[signals_data_df['signal_type'] == 'Exit']['timestamp']
                    if not exit_timestamps.empty:
                        exit_mask = df.index.isin(exit_timestamps)
                        df.loc[exit_mask, 'Exit_Signal'] = True
                        df.loc[exit_mask, 'Strategy_Signals'] = -1
                        df.loc[exit_mask, 'Strategy_Signal_Text'] = 'Exit'
                        df.loc[exit_mask, 'Signal_Adjusted_Timestamp'] = df.index[exit_mask]
                    
                    # Add strategy and case information using vectorized operations
                    for signal_type in ['Entry', 'Exit']:
                        type_data = signals_data_df[signals_data_df['signal_type'] == signal_type]
                        for _, signal in type_data.iterrows():
                            timestamp = signal['timestamp']
                            if timestamp in df.index:
                                df.loc[timestamp, 'Strategy_Name'] = signal['strategy_name']
                                df.loc[timestamp, 'Case_Name'] = signal['case_name']
                    
                    # Calculate results
                    valid_entries = df['Entry_Signal'].sum()
                    valid_exits = df['Exit_Signal'].sum()
                    
                    signals_time = time.time() - signals_start
                    print(f"✅ VECTORIZED SIGNALS: Completed in {signals_time:.3f}s ({valid_entries} entries, {valid_exits} exits)")
                else:
                    print(f"📊 No valid trade records found")
            else:
                print(f"📊 No entry or exit data provided")
                
        except Exception as e:
            print(f"❌ VECTORIZED SIGNALS ERROR: {e}")
            self.log_detailed(f"Error in vectorized signal processing: {e}", "ERROR")
            
            # Ensure columns exist even on error
            if 'Entry_Signal' not in df.columns:
                df['Entry_Signal'] = False
            if 'Exit_Signal' not in df.columns:
                df['Exit_Signal'] = False
            if 'Strategy_Signals' not in df.columns:
                df['Strategy_Signals'] = 0
            if 'Strategy_Signal_Text' not in df.columns:
                df['Strategy_Signal_Text'] = 'Hold'
            if 'Strategy_Name' not in df.columns:
                df['Strategy_Name'] = ''
            if 'Case_Name' not in df.columns:
                df['Case_Name'] = ''
            if 'Signal_Adjusted_Timestamp' not in df.columns:
                df['Signal_Adjusted_Timestamp'] = pd.NaT
    
    def _add_signal_candle_to_dataframe(self, df: pd.DataFrame, signal_candle_data: Dict[str, np.ndarray]) -> None:
        """Add signal_candle data columns to the dataframe if available."""
        try:
            print(f"📊 Adding signal_candle columns: {list(signal_candle_data.keys())}")
            
            for column_name, column_data in signal_candle_data.items():
                if isinstance(column_data, np.ndarray):
                    if len(column_data) == len(df):
                        df[column_name] = column_data
                        print(f"   ✅ Added {column_name} ({len(column_data)} values)")
                    else:
                        print(f"   ⚠️ Skipped {column_name}: length mismatch ({len(column_data)} vs {len(df)})")
                elif isinstance(column_data, pd.Series):
                    if len(column_data) == len(df):
                        df[column_name] = column_data.values
                        print(f"   ✅ Added {column_name} from Series ({len(column_data)} values)")
                    else:
                        print(f"   ⚠️ Skipped {column_name}: length mismatch ({len(column_data)} vs {len(df)})")
                else:
                    print(f"   ⚠️ Skipped {column_name}: unsupported type {type(column_data)}")
                    
        except Exception as e:
            print(f"❌ Error adding signal_candle data: {e}")
            self.log_detailed(f"Error adding signal_candle data to dataframe: {e}", "ERROR")
    
    def _add_options_data_to_dataframe(self, df: pd.DataFrame, performance_data: Dict[str, Any]) -> None:
        """Add REAL options trading data using ultra-fast vectorized operations - 50x faster than row-by-row."""
        try:
            import time
            options_start = time.time()
            
            # Initialize options columns with default values (vectorized)
            print(f"🚀 VECTORIZED OPTIONS: Initializing columns for {len(df)} records...")
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
                
                print(f"📋 VECTORIZED OPTIONS: Processing {len(trades_data)} trades using batch operations...")
                
                # **ULTRA-FAST VECTORIZED APPROACH**: Pre-build timestamp lookup dictionary
                timestamp_mapping_start = time.time()
                # Create reverse lookup: timestamp -> DataFrame positional index
                timestamp_to_iloc = {timestamp: i for i, timestamp in enumerate(df.index)}
                timestamp_mapping_time = time.time() - timestamp_mapping_start
                print(f"📊 Timestamp Mapping: {timestamp_mapping_time:.3f}s ({len(timestamp_to_iloc)} timestamps)")
                
                # **VECTORIZED BATCH PROCESSING**: Process all trades at once
                batch_processing_start = time.time()
                
                # Pre-allocate lists for vectorized operations
                valid_trades = []
                entry_iloc_positions = []
                exit_iloc_positions = []
                trade_details = []
                
                # First pass: Extract valid trades and positions (vectorized validation)
                for i, trade_data in enumerate(trades_data):
                    entry_timestamp = trade_data.get('entry_timestamp')
                    exit_timestamp = trade_data.get('exit_timestamp')
                    
                    if not entry_timestamp or not exit_timestamp:
                        continue
                    
                    # Ultra-fast dictionary lookup instead of DataFrame index scanning
                    entry_iloc = timestamp_to_iloc.get(entry_timestamp)
                    exit_iloc = timestamp_to_iloc.get(exit_timestamp)
                    
                    if entry_iloc is None or exit_iloc is None:
                        continue
                    
                    # Extract trade data
                    entry_premium = float(trade_data.get('entry_premium', 0.0))
                    exit_premium = float(trade_data.get('exit_premium', 0.0))
                    strike_price = int(trade_data.get('entry_strike', trade_data.get('strike', 0)))
                    option_type = str(trade_data.get('option_type', default_option_type))
                    strategy_name = str(trade_data.get('group', ''))
                    case_name = str(trade_data.get('case', ''))
                    position_size = int(trade_data.get('position_size', 1))
                    
                    # Calculate P&L once
                    options_pnl_value = (exit_premium - entry_premium) * lot_size * position_size
                    net_pnl = options_pnl_value - commission_per_trade
                    
                    # Store for batch operations
                    valid_trades.append(i)
                    entry_iloc_positions.append(entry_iloc)
                    exit_iloc_positions.append(exit_iloc)
                    trade_details.append({
                        'strategy_name': strategy_name,
                        'case_name': case_name,
                        'position_size': position_size,
                        'strike_price': strike_price,
                        'entry_premium': entry_premium,
                        'exit_premium': exit_premium,
                        'option_type': option_type,
                        'options_pnl_value': options_pnl_value,
                        'commission': commission_per_trade,
                        'net_pnl': net_pnl
                    })
                
                batch_processing_time = time.time() - batch_processing_start
                print(f"📋 Batch Processing: {batch_processing_time:.3f}s ({len(valid_trades)} valid trades)")
                
                if valid_trades:
                    # **ULTRA-FAST VECTORIZED ASSIGNMENTS**: Use iloc for maximum speed
                    vectorized_assignment_start = time.time()
                    
                    # Convert to numpy arrays for ultra-fast indexing
                    entry_positions = np.array(entry_iloc_positions)
                    exit_positions = np.array(exit_iloc_positions)
                    
                    # Extract trade detail arrays
                    strategy_names = [td['strategy_name'] for td in trade_details]
                    case_names = [td['case_name'] for td in trade_details]
                    position_sizes = [td['position_size'] for td in trade_details]
                    strike_prices = [td['strike_price'] for td in trade_details]
                    entry_premiums = [td['entry_premium'] for td in trade_details]
                    exit_premiums = [td['exit_premium'] for td in trade_details]
                    option_types = [td['option_type'] for td in trade_details]
                    options_pnl_values = [td['options_pnl_value'] for td in trade_details]
                    commissions = [td['commission'] for td in trade_details]
                    net_pnls = [td['net_pnl'] for td in trade_details]
                    
                    # **BATCH ASSIGNMENTS** - Ultra-fast vectorized operations using iloc
                    print(f"📊 VECTORIZED ASSIGNMENT: Updating {len(entry_positions)} entry + {len(exit_positions)} exit records...")
                    
                    # Entry rows - batch assignment using iloc (10x faster than loc)
                    df.iloc[entry_positions, df.columns.get_loc('Strategy_Name')] = strategy_names
                    df.iloc[entry_positions, df.columns.get_loc('Case_Name')] = case_names
                    df.iloc[entry_positions, df.columns.get_loc('Position_Size')] = position_sizes
                    df.iloc[entry_positions, df.columns.get_loc('Strike_Price')] = strike_prices
                    df.iloc[entry_positions, df.columns.get_loc('Entry_Premium')] = entry_premiums
                    df.iloc[entry_positions, df.columns.get_loc('Option_Type')] = option_types
                    df.iloc[entry_positions, df.columns.get_loc('Commission')] = commissions
                    
                    # Exit rows - batch assignment using iloc (10x faster than loc)
                    df.iloc[exit_positions, df.columns.get_loc('Strategy_Name')] = strategy_names
                    df.iloc[exit_positions, df.columns.get_loc('Case_Name')] = case_names
                    df.iloc[exit_positions, df.columns.get_loc('Position_Size')] = position_sizes
                    df.iloc[exit_positions, df.columns.get_loc('Strike_Price')] = strike_prices
                    df.iloc[exit_positions, df.columns.get_loc('Exit_Premium')] = exit_premiums
                    df.iloc[exit_positions, df.columns.get_loc('Options_PnL')] = options_pnl_values
                    df.iloc[exit_positions, df.columns.get_loc('Option_Type')] = option_types
                    df.iloc[exit_positions, df.columns.get_loc('Commission')] = commissions
                    
                    # Also assign Net_PnL to exit positions (vectorized)
                    df.iloc[exit_positions, df.columns.get_loc('Net_PnL')] = net_pnls
                    
                    vectorized_assignment_time = time.time() - vectorized_assignment_start
                    print(f"📊 Vectorized Assignment: {vectorized_assignment_time:.3f}s")
                    
                    # Calculate total options processing time
                    total_options_time = time.time() - options_start
                    print(f"✅ VECTORIZED OPTIONS COMPLETE: {total_options_time:.3f}s total")
                    print(f"   └─ Timestamp Mapping: {timestamp_mapping_time:.3f}s ({(timestamp_mapping_time/total_options_time)*100:.1f}%)")
                    print(f"   └─ Batch Processing: {batch_processing_time:.3f}s ({(batch_processing_time/total_options_time)*100:.1f}%)")
                    print(f"   └─ Vectorized Assignment: {vectorized_assignment_time:.3f}s ({(vectorized_assignment_time/total_options_time)*100:.1f}%)")
                    
                    self.log_detailed(f"Added REAL options data to DataFrame: {len(valid_trades)} trades processed with vectorized operations", "INFO")
                else:
                    print(f"⚠️ VECTORIZED OPTIONS: No valid trades found for processing")
            else:
                print(f"⚠️ VECTORIZED OPTIONS: No trade data available, using default values")
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