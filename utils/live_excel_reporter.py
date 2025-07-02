#!/usr/bin/env python3
"""
Live Excel Reporter
===================

Real-time Excel report generation for live trading sessions.
Matches backtesting Excel format with comprehensive indicator validation.

Features:
- Live session Excel export with 47+ indicators
- Real-time validation reports
- Performance metrics tracking
- Market session statistics

Author: vAlgo Development Team
Created: July 2, 2025
Version: 1.0.0 (Production)
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union
import pandas as pd
import logging
from collections import defaultdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from utils.logger import get_logger
    from utils.config_loader import ConfigLoader
    UTILS_AVAILABLE = True
except ImportError:
    UTILS_AVAILABLE = False
    def get_logger(name):
        logger = logging.getLogger(name)
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
        return logger


class LiveExcelReporter:
    """
    Live Excel Reporter for comprehensive indicator validation reports.
    Generates Excel reports matching backtesting format during live trading.
    """
    
    def __init__(self, output_dir: str = "outputs/live_indicators"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            self.logger = get_logger(__name__)
        except:
            self.logger = logging.getLogger(__name__)
            if not self.logger.handlers:
                handler = logging.StreamHandler()
                formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
                handler.setFormatter(formatter)
                self.logger.addHandler(handler)
                self.logger.setLevel(logging.INFO)
        
        # Live session data storage
        self.session_data = {}
        self.performance_metrics = defaultdict(list)
        self.indicator_columns = []
        self.session_start_time = datetime.now()
        
    def initialize_session(self, symbols: List[str], indicator_columns: List[str]) -> None:
        """Initialize live session for Excel reporting."""
        try:
            self.indicator_columns = indicator_columns
            for symbol in symbols:
                self.session_data[symbol] = []
            
            self.session_start_time = datetime.now()
            self.logger.info(f"[EXCEL] Live session initialized for {len(symbols)} symbols with {len(indicator_columns)} indicators")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize live session: {e}")
    
    def add_live_update(self, symbol: str, timestamp: datetime, ohlcv_data: Dict[str, float], 
                       indicator_data: Dict[str, Any], processing_time: float = 0.0) -> None:
        """Add live update to session data for Excel export."""
        try:
            # Create comprehensive row data
            row_data = {
                'timestamp': timestamp,
                'open': ohlcv_data.get('open', 0.0),
                'high': ohlcv_data.get('high', 0.0),
                'low': ohlcv_data.get('low', 0.0),
                'close': ohlcv_data.get('close', 0.0),
                'volume': ohlcv_data.get('volume', 0)
            }
            
            # Add all indicator values
            for indicator_key in self.indicator_columns:
                if indicator_key in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    continue
                    
                # Extract indicator value from nested dict structure
                indicator_value = self._extract_indicator_value(indicator_data, indicator_key)
                row_data[indicator_key] = indicator_value
            
            # Add to session data
            if symbol not in self.session_data:
                self.session_data[symbol] = []
            self.session_data[symbol].append(row_data)
            
            # Track performance metrics
            self.performance_metrics[symbol].append({
                'timestamp': timestamp,
                'processing_time': processing_time,
                'indicator_count': len(self.indicator_columns)
            })
            
        except Exception as e:
            self.logger.error(f"Failed to add live update for {symbol}: {e}")
    
    def _extract_indicator_value(self, indicator_data: Dict[str, Any], indicator_key: str) -> float:
        """Extract indicator value from nested dictionary structure."""
        try:
            # Handle different indicator data structures
            if indicator_key in indicator_data:
                value = indicator_data[indicator_key]
                return float(value) if isinstance(value, (int, float)) else 0.0
            
            # Handle nested structures like EMA['9'], RSI['14'], etc.
            for main_key, sub_data in indicator_data.items():
                if isinstance(sub_data, dict) and indicator_key in sub_data:
                    value = sub_data[indicator_key]
                    return float(value) if isinstance(value, (int, float)) else 0.0
                
                # Handle formatted keys like 'ema_9', 'rsi_14'
                if indicator_key.startswith(main_key.lower()):
                    period = indicator_key.split('_')[-1]
                    if isinstance(sub_data, dict) and period in sub_data:
                        value = sub_data[period]
                        return float(value) if isinstance(value, (int, float)) else 0.0
            
            return 0.0
            
        except Exception as e:
            self.logger.warning(f"Failed to extract indicator value for {indicator_key}: {e}")
            return 0.0
    
    def generate_live_excel_report(self, symbol: str, report_type: str = "session") -> str:
        """
        Generate comprehensive Excel report for live trading session.
        
        Args:
            symbol: Trading symbol
            report_type: 'session' for full session, 'interval' for periodic updates
            
        Returns:
            Path to generated Excel file
        """
        try:
            if symbol not in self.session_data or not self.session_data[symbol]:
                self.logger.warning(f"No session data available for {symbol}")
                return ""
            
            # Create DataFrame from session data
            df = pd.DataFrame(self.session_data[symbol])
            
            # Generate Excel filename
            timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
            excel_filename = f"{symbol}_live_indicators_{report_type}_{timestamp_str}.xlsx"
            excel_path = self.output_dir / excel_filename
            
            # Create Excel writer
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Main indicator data sheet
                df.to_excel(writer, sheet_name='Indicators', index=False)
                
                # Performance metrics sheet
                if symbol in self.performance_metrics:
                    perf_df = pd.DataFrame(self.performance_metrics[symbol])
                    perf_df.to_excel(writer, sheet_name='Performance', index=False)
                
                # Session statistics sheet
                stats_data = self._generate_session_statistics(symbol, df)
                stats_df = pd.DataFrame([stats_data])
                stats_df.to_excel(writer, sheet_name='Statistics', index=False)
                
                # Indicator summary sheet
                summary_data = self._generate_indicator_summary(df)
                if summary_data:
                    summary_df = pd.DataFrame(summary_data)
                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            self.logger.info(f"[EXCEL] Generated live Excel report: {excel_path}")
            return str(excel_path)
            
        except Exception as e:
            self.logger.error(f"Failed to generate Excel report for {symbol}: {e}")
            return ""
    
    def _generate_session_statistics(self, symbol: str, df: pd.DataFrame) -> Dict[str, Any]:
        """Generate comprehensive session statistics."""
        try:
            current_time = datetime.now()
            session_duration = (current_time - self.session_start_time).total_seconds()
            
            stats = {
                'Symbol': symbol,
                'Session_Start': self.session_start_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Report_Time': current_time.strftime('%Y-%m-%d %H:%M:%S'),
                'Session_Duration_Minutes': round(session_duration / 60, 2),
                'Total_Updates': len(df),
                'Indicator_Count': len(self.indicator_columns) - 5,  # Subtract OHLCV columns
                'Update_Frequency_Minutes': round(session_duration / 60 / len(df), 2) if len(df) > 0 else 0,
                'First_Update': df['timestamp'].min() if not df.empty else None,
                'Last_Update': df['timestamp'].max() if not df.empty else None,
                'OHLC_Range_High': df['high'].max() if not df.empty else 0,
                'OHLC_Range_Low': df['low'].min() if not df.empty else 0,
                'Total_Volume': df['volume'].sum() if not df.empty else 0,
                'Avg_Processing_Time': self._calculate_avg_processing_time(symbol)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Failed to generate session statistics: {e}")
            return {}
    
    def _generate_indicator_summary(self, df: pd.DataFrame) -> List[Dict[str, Any]]:
        """Generate indicator summary statistics."""
        try:
            summary_data = []
            
            for col in df.columns:
                if col in ['timestamp', 'open', 'high', 'low', 'close', 'volume']:
                    continue
                
                if df[col].dtype in ['float64', 'int64']:
                    summary_data.append({
                        'Indicator': col,
                        'Count': len(df[col].dropna()),
                        'Min': df[col].min(),
                        'Max': df[col].max(),
                        'Mean': df[col].mean(),
                        'Std': df[col].std(),
                        'Last_Value': df[col].iloc[-1] if not df.empty else None
                    })
            
            return summary_data
            
        except Exception as e:
            self.logger.error(f"Failed to generate indicator summary: {e}")
            return []
    
    def _calculate_avg_processing_time(self, symbol: str) -> float:
        """Calculate average processing time for symbol."""
        try:
            if symbol in self.performance_metrics:
                times = [m['processing_time'] for m in self.performance_metrics[symbol]]
                return sum(times) / len(times) if times else 0.0
            return 0.0
        except:
            return 0.0
    
    def export_final_session_report(self, symbols: List[str]) -> Dict[str, str]:
        """Export final session reports for all symbols at market close."""
        try:
            exported_files = {}
            
            for symbol in symbols:
                if symbol in self.session_data and self.session_data[symbol]:
                    excel_path = self.generate_live_excel_report(symbol, "final_session")
                    if excel_path:
                        exported_files[symbol] = excel_path
                        self.logger.info(f"[EXPORT] Final session report exported for {symbol}: {excel_path}")
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Failed to export final session reports: {e}")
            return {}
    
    def generate_periodic_report(self, symbol: str, interval_minutes: int = 5) -> str:
        """Generate periodic Excel report for performance monitoring."""
        try:
            current_time = datetime.now()
            cutoff_time = current_time - timedelta(minutes=interval_minutes)
            
            # Filter recent data
            if symbol in self.session_data:
                recent_data = [
                    record for record in self.session_data[symbol]
                    if record['timestamp'] >= cutoff_time
                ]
                
                if recent_data:
                    # Temporarily store recent data
                    original_data = self.session_data[symbol]
                    self.session_data[symbol] = recent_data
                    
                    # Generate report
                    excel_path = self.generate_live_excel_report(symbol, f"{interval_minutes}min")
                    
                    # Restore original data
                    self.session_data[symbol] = original_data
                    
                    return excel_path
            
            return ""
            
        except Exception as e:
            self.logger.error(f"Failed to generate periodic report: {e}")
            return ""
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary statistics."""
        try:
            current_time = datetime.now()
            session_duration = (current_time - self.session_start_time).total_seconds()
            
            total_updates = sum(len(data) for data in self.session_data.values())
            
            return {
                'session_start': self.session_start_time,
                'current_time': current_time,
                'session_duration_minutes': round(session_duration / 60, 2),
                'symbols_tracked': len(self.session_data),
                'total_updates': total_updates,
                'indicator_count': len(self.indicator_columns) - 5,
                'update_rate_per_minute': round(total_updates / (session_duration / 60), 2) if session_duration > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get session summary: {e}")
            return {}