#!/usr/bin/env python3
"""
Parallel Chart Tracker for vAlgo Trading System
==============================================

Advanced parallel chart tracking system for monitoring both equity OHLC and 
options strike OHLC simultaneously with real-time visualization capabilities.

Features:
- Real-time equity OHLC tracking
- Options strike OHLC tracking with Greeks
- Synchronized chart updates
- Multi-timeframe support (5m signal, 1m execution)
- SL/TP level visualization
- Performance metrics overlay
- Export capabilities for analysis
- Interactive chart generation

Author: vAlgo Development Team
Created: July 11, 2025
Version: 1.0.0 (Production)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import json
from collections import defaultdict, deque
import logging

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from data_manager.options_database import OptionsDatabase, create_options_database
from utils.advanced_strike_selector import AdvancedStrikeSelector, StrikeType, OptionType


class ChartDataPoint:
    """Single chart data point with timestamp and values."""
    
    def __init__(self, timestamp: datetime, **kwargs):
        self.timestamp = timestamp
        self.data = kwargs
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            'timestamp': self.timestamp.isoformat(),
            **self.data
        }


class ParallelChartTracker:
    """
    Advanced parallel chart tracking system for equity and options data.
    """
    
    def __init__(self, max_data_points: int = 10000, options_db: Optional[OptionsDatabase] = None):
        """
        Initialize Parallel Chart Tracker.
        
        Args:
            max_data_points: Maximum number of data points to store per chart
            options_db: Optional OptionsDatabase instance for real data
        """
        self.logger = get_logger(__name__)
        self.max_data_points = max_data_points
        
        # Real options data integration
        self.options_db = options_db or create_options_database()
        self.real_data_mode = options_db is not None
        
        # Data storage
        self.equity_charts = defaultdict(lambda: deque(maxlen=max_data_points))
        self.options_charts = defaultdict(lambda: deque(maxlen=max_data_points))
        
        # Metadata storage
        self.chart_metadata = {}
        self.active_trades = set()
        
        # Real-time tracking
        self.last_update_time = {}
        self.update_counts = defaultdict(int)
        
        # Performance tracking
        self.performance_stats = {
            'total_updates': 0,
            'average_update_time': 0.0,
            'max_update_time': 0.0,
            'data_points_stored': 0,
            'memory_usage_mb': 0.0,
            'real_data_queries': 0,
            'real_data_cache_hits': 0,
            'real_data_errors': 0
        }
        
        # Real data caching for performance
        self.options_data_cache = {}
        self.cache_expiry_seconds = 5  # Cache real option data for 5 seconds
        
        self.logger.info("ParallelChartTracker initialized successfully")
    
    def start_tracking(self, trade_id: str, trade_info: Dict[str, Any]) -> bool:
        """
        Start tracking for a new trade.
        
        Args:
            trade_id: Unique trade identifier
            trade_info: Trade information including symbol, strike, etc.
            
        Returns:
            True if tracking started successfully
        """
        try:
            # Initialize metadata
            self.chart_metadata[trade_id] = {
                'symbol': trade_info.get('symbol', 'UNKNOWN'),
                'strike': trade_info.get('strike', 0),
                'option_type': trade_info.get('option_type', 'CALL'),
                'expiry_date': trade_info.get('expiry_date'),
                'entry_time': trade_info.get('entry_time', datetime.now()),
                'entry_premium': trade_info.get('entry_premium', 0),
                'entry_underlying_price': trade_info.get('underlying_price', 0),
                'sl_tp_levels': trade_info.get('sl_tp_levels', {}),
                'strategy_name': trade_info.get('strategy_name', 'default'),
                'position_size': trade_info.get('position_size', 1),
                'status': 'active',
                'tracking_start_time': datetime.now()
            }
            
            # Add to active trades
            self.active_trades.add(trade_id)
            
            # Initialize tracking counters
            self.update_counts[trade_id] = 0
            self.last_update_time[trade_id] = datetime.now()
            
            self.logger.info(f"Started tracking for trade {trade_id} - "
                           f"{trade_info.get('symbol')} {trade_info.get('option_type')} "
                           f"{trade_info.get('strike')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting tracking for {trade_id}: {e}")
            return False
    
    def update_equity_data(self, trade_id: str, equity_data: Dict[str, Any]) -> bool:
        """
        Update equity OHLC data for a trade.
        
        Args:
            trade_id: Trade identifier
            equity_data: Equity OHLC data
            
        Returns:
            True if update successful
        """
        try:
            if trade_id not in self.active_trades:
                self.logger.warning(f"Trade {trade_id} not found for equity update")
                return False
            
            timestamp = equity_data.get('timestamp', datetime.now())
            
            # Create equity data point
            equity_point = ChartDataPoint(
                timestamp=timestamp,
                open=float(equity_data.get('open', 0)),
                high=float(equity_data.get('high', 0)),
                low=float(equity_data.get('low', 0)),
                close=float(equity_data.get('close', 0)),
                volume=int(equity_data.get('volume', 0)),
                # Add SL/TP levels if available
                sl_level=self._get_equity_sl_level(trade_id, equity_data.get('close', 0)),
                tp_level=self._get_equity_tp_level(trade_id, equity_data.get('close', 0)),
                # Add technical indicators if available
                rsi=equity_data.get('rsi'),
                ema_9=equity_data.get('ema_9'),
                ema_21=equity_data.get('ema_21'),
                bb_upper=equity_data.get('bb_upper'),
                bb_lower=equity_data.get('bb_lower'),
                atr=equity_data.get('atr'),
                # Add trend indicators
                trend_direction=equity_data.get('trend_direction'),
                support_level=equity_data.get('support_level'),
                resistance_level=equity_data.get('resistance_level')
            )
            
            # Add to equity chart
            self.equity_charts[trade_id].append(equity_point)
            
            # Update tracking stats
            self.update_counts[trade_id] += 1
            self.last_update_time[trade_id] = timestamp
            self.performance_stats['total_updates'] += 1
            
            self.logger.debug(f"Updated equity data for {trade_id}: "
                            f"Close {equity_data.get('close', 0):.2f}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating equity data for {trade_id}: {e}")
            return False
    
    def update_options_data(self, trade_id: str, options_data: Dict[str, Any]) -> bool:
        """
        Update options strike data for a trade with real database integration.
        
        Args:
            trade_id: Trade identifier
            options_data: Options strike data (can be enriched with real DB data)
            
        Returns:
            True if update successful
        """
        try:
            if trade_id not in self.active_trades:
                self.logger.warning(f"Trade {trade_id} not found for options update")
                return False
            
            timestamp = options_data.get('timestamp', datetime.now())
            
            # Enrich with real options data if available
            enriched_data = self._enrich_options_data(trade_id, options_data, timestamp)
            
            # Create options data point with enriched real data
            options_point = ChartDataPoint(
                timestamp=timestamp,
                premium=float(enriched_data.get('premium', 0)),
                bid=float(enriched_data.get('bid', 0)),
                ask=float(enriched_data.get('ask', 0)),
                open_interest=int(enriched_data.get('open_interest', 0)),
                volume=int(enriched_data.get('volume', 0)),
                # Greeks (real or calculated)
                delta=float(enriched_data.get('delta', 0)),
                gamma=float(enriched_data.get('gamma', 0)),
                theta=float(enriched_data.get('theta', 0)),
                vega=float(enriched_data.get('vega', 0)),
                rho=float(enriched_data.get('rho', 0)),
                # IV and time decay
                iv=float(enriched_data.get('iv', 0)),
                intrinsic_value=float(enriched_data.get('intrinsic_value', 0)),
                time_value=float(enriched_data.get('time_value', 0)),
                # SL/TP levels
                sl_level=self._get_options_sl_level(trade_id, enriched_data.get('premium', 0)),
                tp_level=self._get_options_tp_level(trade_id, enriched_data.get('premium', 0)),
                # Additional metrics
                moneyness=enriched_data.get('moneyness', 'UNKNOWN'),
                dte=int(enriched_data.get('dte', 0)),
                underlying_price=float(enriched_data.get('underlying_price', 0)),
                # Real data indicators
                data_source=enriched_data.get('data_source', 'provided'),
                cache_hit=enriched_data.get('cache_hit', False),
                real_ltp=float(enriched_data.get('real_ltp', 0)),
                real_call_ltp=float(enriched_data.get('real_call_ltp', 0)),
                real_put_ltp=float(enriched_data.get('real_put_ltp', 0))
            )
            
            # Add to options chart
            self.options_charts[trade_id].append(options_point)
            
            # Update tracking stats
            self.update_counts[trade_id] += 1
            self.last_update_time[trade_id] = timestamp
            self.performance_stats['total_updates'] += 1
            
            self.logger.debug(f"Updated options data for {trade_id}: "
                            f"Premium {enriched_data.get('premium', 0):.2f}, "
                            f"IV {enriched_data.get('iv', 0):.1f}%, "
                            f"Source: {enriched_data.get('data_source', 'provided')}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating options data for {trade_id}: {e}")
            return False
    
    def _enrich_options_data(self, trade_id: str, options_data: Dict[str, Any], 
                           timestamp: datetime) -> Dict[str, Any]:
        """
        Enrich options data with real database information.
        
        Args:
            trade_id: Trade identifier
            options_data: Original options data
            timestamp: Current timestamp
            
        Returns:
            Enriched options data with real database values
        """
        try:
            # Start with provided data
            enriched_data = options_data.copy()
            enriched_data['data_source'] = 'provided'
            enriched_data['cache_hit'] = False
            
            if not self.real_data_mode or not self.options_db:
                return enriched_data
            
            # Get trade metadata for strike info
            if trade_id not in self.chart_metadata:
                return enriched_data
            
            metadata = self.chart_metadata[trade_id]
            strike = metadata.get('strike')
            option_type = metadata.get('option_type', 'CALL')
            symbol = metadata.get('symbol', 'NIFTY')
            
            if not strike:
                return enriched_data
            
            # Check cache first
            cache_key = f"{strike}_{option_type}_{timestamp.strftime('%Y%m%d_%H%M')}"
            current_time = datetime.now()
            
            if (cache_key in self.options_data_cache and 
                (current_time - self.options_data_cache[cache_key]['timestamp']).seconds < self.cache_expiry_seconds):
                
                cached_data = self.options_data_cache[cache_key]['data']
                enriched_data.update(cached_data)
                enriched_data['cache_hit'] = True
                enriched_data['data_source'] = 'database_cached'
                self.performance_stats['real_data_cache_hits'] += 1
                return enriched_data
            
            # Query real options database
            try:
                # Use the correct method name from options_database
                real_data = self.options_db.get_strike_ohlc_data(
                    strike=strike,
                    expiry_date=metadata.get('expiry_date', '2024-07-25'),
                    start_time=timestamp - timedelta(minutes=5),
                    end_time=timestamp + timedelta(minutes=5),
                    option_type=option_type.lower()
                )
                
                if real_data is not None and not real_data.empty:
                    # Extract real values
                    latest_row = real_data.iloc[-1]
                    
                    if option_type == 'CALL':
                        real_premium = latest_row.get('call_ltp', enriched_data.get('premium', 0))
                        real_iv = latest_row.get('call_iv', enriched_data.get('iv', 0))
                        real_delta = latest_row.get('call_delta', enriched_data.get('delta', 0))
                    else:  # PUT
                        real_premium = latest_row.get('put_ltp', enriched_data.get('premium', 0))
                        real_iv = latest_row.get('put_iv', enriched_data.get('iv', 0))
                        real_delta = latest_row.get('put_delta', enriched_data.get('delta', 0))
                    
                    # Update enriched data with real values
                    if real_premium > 0:
                        enriched_data['premium'] = real_premium
                        enriched_data['real_ltp'] = real_premium
                    
                    if real_iv > 0:
                        enriched_data['iv'] = real_iv
                    
                    if abs(real_delta) > 0:
                        enriched_data['delta'] = real_delta
                    
                    # Add both call and put LTPs for reference
                    enriched_data['real_call_ltp'] = latest_row.get('call_ltp', 0)
                    enriched_data['real_put_ltp'] = latest_row.get('put_ltp', 0)
                    
                    # Calculate intrinsic and time value
                    underlying_price = enriched_data.get('underlying_price', 0)
                    if underlying_price > 0:
                        if option_type == 'CALL':
                            intrinsic = max(0, underlying_price - strike)
                        else:  # PUT
                            intrinsic = max(0, strike - underlying_price)
                        
                        enriched_data['intrinsic_value'] = intrinsic
                        enriched_data['time_value'] = max(0, real_premium - intrinsic)
                    
                    enriched_data['data_source'] = 'database_real'
                    
                    # Cache the result
                    self.options_data_cache[cache_key] = {
                        'timestamp': current_time,
                        'data': {
                            'premium': real_premium,
                            'iv': real_iv,
                            'delta': real_delta,
                            'real_call_ltp': latest_row.get('call_ltp', 0),
                            'real_put_ltp': latest_row.get('put_ltp', 0),
                            'intrinsic_value': enriched_data.get('intrinsic_value', 0),
                            'time_value': enriched_data.get('time_value', 0)
                        }
                    }
                    
                    self.performance_stats['real_data_queries'] += 1
                    
                else:
                    self.logger.debug(f"No real options data found for {strike} {option_type} at {timestamp}")
                    
            except Exception as e:
                self.logger.warning(f"Error querying real options data: {e}")
                self.performance_stats['real_data_errors'] += 1
                enriched_data['data_source'] = 'provided_fallback'
            
            return enriched_data
            
        except Exception as e:
            self.logger.error(f"Error enriching options data: {e}")
            return options_data
    
    def update_parallel_data(self, trade_id: str, equity_data: Dict[str, Any],
                           options_data: Dict[str, Any]) -> bool:
        """
        Update both equity and options data simultaneously.
        
        Args:
            trade_id: Trade identifier
            equity_data: Equity OHLC data
            options_data: Options strike data
            
        Returns:
            True if both updates successful
        """
        try:
            equity_success = self.update_equity_data(trade_id, equity_data)
            options_success = self.update_options_data(trade_id, options_data)
            
            return equity_success and options_success
            
        except Exception as e:
            self.logger.error(f"Error updating parallel data for {trade_id}: {e}")
            return False
    
    def _get_equity_sl_level(self, trade_id: str, current_price: float) -> float:
        """Get equity SL level for the trade."""
        try:
            if trade_id not in self.chart_metadata:
                return 0.0
            
            sl_tp_levels = self.chart_metadata[trade_id].get('sl_tp_levels', {})
            return float(sl_tp_levels.get('equity_sl', 0))
            
        except Exception:
            return 0.0
    
    def _get_equity_tp_level(self, trade_id: str, current_price: float) -> float:
        """Get equity TP level for the trade."""
        try:
            if trade_id not in self.chart_metadata:
                return 0.0
            
            sl_tp_levels = self.chart_metadata[trade_id].get('sl_tp_levels', {})
            return float(sl_tp_levels.get('equity_tp', 0))
            
        except Exception:
            return 0.0
    
    def _get_options_sl_level(self, trade_id: str, current_premium: float) -> float:
        """Get options SL level for the trade."""
        try:
            if trade_id not in self.chart_metadata:
                return 0.0
            
            sl_tp_levels = self.chart_metadata[trade_id].get('sl_tp_levels', {})
            return float(sl_tp_levels.get('options_sl', 0))
            
        except Exception:
            return 0.0
    
    def _get_options_tp_level(self, trade_id: str, current_premium: float) -> float:
        """Get options TP level for the trade."""
        try:
            if trade_id not in self.chart_metadata:
                return 0.0
            
            sl_tp_levels = self.chart_metadata[trade_id].get('sl_tp_levels', {})
            return float(sl_tp_levels.get('options_tp', 0))
            
        except Exception:
            return 0.0
    
    def stop_tracking(self, trade_id: str, exit_reason: str = "COMPLETED") -> bool:
        """
        Stop tracking for a trade.
        
        Args:
            trade_id: Trade identifier
            exit_reason: Reason for stopping tracking
            
        Returns:
            True if tracking stopped successfully
        """
        try:
            if trade_id not in self.active_trades:
                self.logger.warning(f"Trade {trade_id} not found for stop tracking")
                return False
            
            # Update metadata
            if trade_id in self.chart_metadata:
                self.chart_metadata[trade_id]['status'] = 'completed'
                self.chart_metadata[trade_id]['exit_reason'] = exit_reason
                self.chart_metadata[trade_id]['tracking_end_time'] = datetime.now()
                self.chart_metadata[trade_id]['total_updates'] = self.update_counts[trade_id]
            
            # Remove from active trades
            self.active_trades.remove(trade_id)
            
            self.logger.info(f"Stopped tracking for trade {trade_id} - {exit_reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping tracking for {trade_id}: {e}")
            return False
    
    def get_equity_chart_data(self, trade_id: str, last_n_points: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get equity chart data for a trade.
        
        Args:
            trade_id: Trade identifier
            last_n_points: Number of last points to return (None for all)
            
        Returns:
            List of equity data points
        """
        try:
            if trade_id not in self.equity_charts:
                return []
            
            chart_data = list(self.equity_charts[trade_id])
            
            if last_n_points:
                chart_data = chart_data[-last_n_points:]
            
            return [point.to_dict() for point in chart_data]
            
        except Exception as e:
            self.logger.error(f"Error getting equity chart data for {trade_id}: {e}")
            return []
    
    def get_options_chart_data(self, trade_id: str, last_n_points: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get options chart data for a trade.
        
        Args:
            trade_id: Trade identifier
            last_n_points: Number of last points to return (None for all)
            
        Returns:
            List of options data points
        """
        try:
            if trade_id not in self.options_charts:
                return []
            
            chart_data = list(self.options_charts[trade_id])
            
            if last_n_points:
                chart_data = chart_data[-last_n_points:]
            
            return [point.to_dict() for point in chart_data]
            
        except Exception as e:
            self.logger.error(f"Error getting options chart data for {trade_id}: {e}")
            return []
    
    def get_parallel_chart_data(self, trade_id: str, last_n_points: Optional[int] = None) -> Dict[str, Any]:
        """
        Get both equity and options chart data for a trade.
        
        Args:
            trade_id: Trade identifier
            last_n_points: Number of last points to return (None for all)
            
        Returns:
            Dictionary containing both equity and options data
        """
        try:
            return {
                'trade_id': trade_id,
                'metadata': self.chart_metadata.get(trade_id, {}),
                'equity_data': self.get_equity_chart_data(trade_id, last_n_points),
                'options_data': self.get_options_chart_data(trade_id, last_n_points),
                'last_update': self.last_update_time.get(trade_id),
                'total_updates': self.update_counts.get(trade_id, 0)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting parallel chart data for {trade_id}: {e}")
            return {}
    
    def get_all_active_trades(self) -> List[str]:
        """Get list of all active trade IDs."""
        return list(self.active_trades)
    
    def get_tracking_summary(self) -> Dict[str, Any]:
        """Get summary of tracking status."""
        try:
            active_trades = len(self.active_trades)
            completed_trades = len([t for t in self.chart_metadata.values() 
                                 if t.get('status') == 'completed'])
            
            return {
                'active_trades': active_trades,
                'completed_trades': completed_trades,
                'total_trades': active_trades + completed_trades,
                'total_updates': self.performance_stats['total_updates'],
                'memory_usage_estimate_mb': self._estimate_memory_usage(),
                'oldest_active_trade': self._get_oldest_active_trade(),
                'newest_active_trade': self._get_newest_active_trade(),
                'average_updates_per_trade': self._calculate_average_updates_per_trade()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting tracking summary: {e}")
            return {}
    
    def _estimate_memory_usage(self) -> float:
        """Estimate memory usage in MB."""
        try:
            # Rough estimate: each data point ~1KB
            total_points = sum(len(chart) for chart in self.equity_charts.values())
            total_points += sum(len(chart) for chart in self.options_charts.values())
            
            return total_points * 1024 / (1024 * 1024)  # Convert to MB
            
        except Exception:
            return 0.0
    
    def _get_oldest_active_trade(self) -> Optional[str]:
        """Get oldest active trade ID."""
        try:
            if not self.active_trades:
                return None
            
            oldest_trade = None
            oldest_time = datetime.now()
            
            for trade_id in self.active_trades:
                if trade_id in self.chart_metadata:
                    start_time = self.chart_metadata[trade_id].get('tracking_start_time')
                    if start_time and start_time < oldest_time:
                        oldest_time = start_time
                        oldest_trade = trade_id
            
            return oldest_trade
            
        except Exception:
            return None
    
    def _get_newest_active_trade(self) -> Optional[str]:
        """Get newest active trade ID."""
        try:
            if not self.active_trades:
                return None
            
            newest_trade = None
            newest_time = datetime(1970, 1, 1)
            
            for trade_id in self.active_trades:
                if trade_id in self.chart_metadata:
                    start_time = self.chart_metadata[trade_id].get('tracking_start_time')
                    if start_time and start_time > newest_time:
                        newest_time = start_time
                        newest_trade = trade_id
            
            return newest_trade
            
        except Exception:
            return None
    
    def _calculate_average_updates_per_trade(self) -> float:
        """Calculate average updates per trade."""
        try:
            if not self.update_counts:
                return 0.0
            
            total_updates = sum(self.update_counts.values())
            total_trades = len(self.update_counts)
            
            return total_updates / total_trades if total_trades > 0 else 0.0
            
        except Exception:
            return 0.0
    
    def export_trade_data(self, trade_id: str, output_dir: str = "outputs/charts") -> Dict[str, str]:
        """
        Export trade data to files.
        
        Args:
            trade_id: Trade identifier
            output_dir: Output directory
            
        Returns:
            Dictionary with file paths
        """
        try:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Get chart data
            chart_data = self.get_parallel_chart_data(trade_id)
            
            if not chart_data:
                return {'error': f'No data found for trade {trade_id}'}
            
            # Export to JSON
            json_file = output_path / f"{trade_id}_parallel_charts.json"
            with open(json_file, 'w') as f:
                json.dump(chart_data, f, indent=2, default=str)
            
            # Export equity data to CSV
            equity_csv = output_path / f"{trade_id}_equity_chart.csv"
            if chart_data['equity_data']:
                equity_df = pd.DataFrame(chart_data['equity_data'])
                equity_df.to_csv(equity_csv, index=False)
            
            # Export options data to CSV
            options_csv = output_path / f"{trade_id}_options_chart.csv"
            if chart_data['options_data']:
                options_df = pd.DataFrame(chart_data['options_data'])
                options_df.to_csv(options_csv, index=False)
            
            # Export metadata
            metadata_file = output_path / f"{trade_id}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(chart_data['metadata'], f, indent=2, default=str)
            
            exported_files = {
                'json': str(json_file),
                'equity_csv': str(equity_csv),
                'options_csv': str(options_csv),
                'metadata': str(metadata_file)
            }
            
            self.logger.info(f"Exported trade data for {trade_id} to {output_dir}")
            
            return exported_files
            
        except Exception as e:
            self.logger.error(f"Error exporting trade data for {trade_id}: {e}")
            return {'error': str(e)}
    
    def cleanup_old_trades(self, days_old: int = 30) -> int:
        """
        Clean up old completed trades.
        
        Args:
            days_old: Number of days old to consider for cleanup
            
        Returns:
            Number of trades cleaned up
        """
        try:
            cutoff_time = datetime.now() - timedelta(days=days_old)
            trades_to_remove = []
            
            for trade_id, metadata in self.chart_metadata.items():
                if (metadata.get('status') == 'completed' and 
                    metadata.get('tracking_end_time') and
                    metadata['tracking_end_time'] < cutoff_time):
                    trades_to_remove.append(trade_id)
            
            # Remove old trades
            for trade_id in trades_to_remove:
                # Remove chart data
                if trade_id in self.equity_charts:
                    del self.equity_charts[trade_id]
                if trade_id in self.options_charts:
                    del self.options_charts[trade_id]
                
                # Remove metadata
                if trade_id in self.chart_metadata:
                    del self.chart_metadata[trade_id]
                
                # Remove tracking info
                if trade_id in self.update_counts:
                    del self.update_counts[trade_id]
                if trade_id in self.last_update_time:
                    del self.last_update_time[trade_id]
            
            self.logger.info(f"Cleaned up {len(trades_to_remove)} old trades")
            
            return len(trades_to_remove)
            
        except Exception as e:
            self.logger.error(f"Error cleaning up old trades: {e}")
            return 0
    
    def generate_performance_report(self) -> Dict[str, Any]:
        """Generate performance report for the tracker."""
        try:
            summary = self.get_tracking_summary()
            
            report = {
                'tracking_summary': summary,
                'performance_stats': self.performance_stats,
                'data_quality': {
                    'equity_charts_count': len(self.equity_charts),
                    'options_charts_count': len(self.options_charts),
                    'metadata_records': len(self.chart_metadata),
                    'total_data_points': sum(len(chart) for chart in self.equity_charts.values()) +
                                       sum(len(chart) for chart in self.options_charts.values())
                },
                'system_health': {
                    'memory_usage_mb': self._estimate_memory_usage(),
                    'max_data_points_per_chart': self.max_data_points,
                    'active_tracking_sessions': len(self.active_trades),
                    'last_system_update': datetime.now().isoformat()
                }
            }
            
            return report
            
        except Exception as e:
            self.logger.error(f"Error generating performance report: {e}")
            return {'error': str(e)}


# Convenience functions
def create_parallel_tracker(max_data_points: int = 10000, 
                          options_db: Optional[OptionsDatabase] = None) -> ParallelChartTracker:
    """
    Create parallel chart tracker with real options database integration.
    
    Args:
        max_data_points: Maximum data points per chart
        options_db: Optional OptionsDatabase instance for real data enrichment
        
    Returns:
        ParallelChartTracker instance
    """
    return ParallelChartTracker(max_data_points, options_db)


# Example usage
if __name__ == "__main__":
    # Example usage of the Parallel Chart Tracker
    try:
        # Create tracker
        tracker = create_parallel_tracker()
        
        # Start tracking a trade
        trade_info = {
            'symbol': 'NIFTY',
            'strike': 21650,
            'option_type': 'CALL',
            'expiry_date': '2024-01-25',
            'entry_time': datetime.now(),
            'entry_premium': 100.0,
            'underlying_price': 21645.0,
            'sl_tp_levels': {
                'equity_sl': 21600,
                'equity_tp': 21700,
                'options_sl': 80,
                'options_tp': 150
            },
            'strategy_name': 'momentum_strategy',
            'position_size': 2
        }
        
        trade_id = "TEST_TRADE_001"
        
        print(f"Starting tracking for trade {trade_id}")
        success = tracker.start_tracking(trade_id, trade_info)
        
        if success:
            print("Tracking started successfully")
            
            # Simulate some data updates
            for i in range(5):
                equity_data = {
                    'timestamp': datetime.now(),
                    'open': 21645.0 + i,
                    'high': 21650.0 + i,
                    'low': 21640.0 + i,
                    'close': 21648.0 + i,
                    'volume': 1000000 + i * 10000,
                    'rsi': 60 + i,
                    'ema_9': 21640 + i,
                    'ema_21': 21635 + i
                }
                
                options_data = {
                    'timestamp': datetime.now(),
                    'premium': 100.0 + i * 5,
                    'bid': 98.0 + i * 5,
                    'ask': 102.0 + i * 5,
                    'volume': 500 + i * 10,
                    'delta': 0.5 + i * 0.05,
                    'gamma': 0.01,
                    'theta': -0.5,
                    'vega': 0.2,
                    'iv': 18.0 + i * 0.5,
                    'intrinsic_value': 0 + i * 2,
                    'time_value': 100 + i * 3,
                    'dte': 10 - i,
                    'underlying_price': 21648.0 + i
                }
                
                print(f"Update {i + 1}: Equity close {equity_data['close']:.2f}, "
                      f"Options premium {options_data['premium']:.2f}")
                
                tracker.update_parallel_data(trade_id, equity_data, options_data)
            
            # Get chart data
            chart_data = tracker.get_parallel_chart_data(trade_id)
            print(f"\nChart data points: Equity {len(chart_data['equity_data'])}, "
                  f"Options {len(chart_data['options_data'])}")
            
            # Export data
            exported = tracker.export_trade_data(trade_id)
            print(f"Exported files: {list(exported.keys())}")
            
            # Get tracking summary
            summary = tracker.get_tracking_summary()
            print(f"\nTracking summary:")
            print(f"Active trades: {summary['active_trades']}")
            print(f"Total updates: {summary['total_updates']}")
            print(f"Memory usage: {summary['memory_usage_estimate_mb']:.2f} MB")
            
            # Stop tracking
            tracker.stop_tracking(trade_id, "TEST_COMPLETED")
            print(f"\nStopped tracking for trade {trade_id}")
            
        else:
            print("Failed to start tracking")
            
    except Exception as e:
        print(f"Error in example usage: {e}")