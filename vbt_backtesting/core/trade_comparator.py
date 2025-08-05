"""
Trade Comparator - Trade Comparison Engine for Ultimate Efficiency Engine
=========================================================================

Trade comparison functionality for Ultimate Efficiency VBT backtesting system.
Compares VBT generated trades with QuantMan backtesting software results for validation.

Features:
- Date range filtering to match backtesting period
- Trade matching by timestamp proximity, option type, and strike
- Premium comparison with configurable tolerance
- Missing trade detection and analysis
- Comprehensive comparison reporting

Author: vAlgo Development Team
Created: August 4, 2025
"""

import sys
import re
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


class TradeComparator(ProcessorBase):
    """
    Trade comparison functionality for Ultimate Efficiency Engine.
    
    Provides comprehensive trade comparison capabilities:
    - Compare VBT trades with QuantMan reference data
    - Date range filtering for relevant trade comparison
    - Trade matching with configurable tolerances
    - Premium accuracy validation
    - Missing trade detection and reporting
    """
    
    def __init__(self, config_loader, logger):
        """
        Initialize Trade Comparator.
        
        Args:
            config_loader: Configuration loader instance
            logger: Selective logger instance
        """
        # Initialize without calling super().__init__ as we don't need data_loader
        self.config_loader = config_loader
        self.logger = logger
        
        self.log_detailed("=== TRADE COMPARATOR INITIALIZATION DEBUG ===", "INFO")
        
        # Load trade comparison configuration
        self.log_detailed("Step 1: Loading trade_comparison config section", "INFO")
        self.comparison_config = self.config_loader.main_config.get('trade_comparison', {})
        self.log_detailed(f"Config found: {bool(self.comparison_config)}", "INFO")
        self.log_detailed(f"Config contents: {self.comparison_config}", "INFO")
        
        self.enabled = self.comparison_config.get('enabled', True)  
        self.log_detailed(f"Step 2: Initial enabled status: {self.enabled}", "INFO")
        
        if not self.enabled:
            self.log_detailed("INITIALIZATION STOPPED: trade_comparison.enabled = False in config", "WARNING")
        elif not self.comparison_config:
            self.log_detailed("INITIALIZATION STOPPED: No trade_comparison config section found", "WARNING")
            self.enabled = False
        else:
            self.log_detailed("Step 3: Loading QuantMan CSV path configuration", "INFO")
            self.quantman_csv_path = self.comparison_config.get('quantman_csv_path', '')
            self.tolerance_premium = self.comparison_config.get('tolerance_premium', 0.05)
            self.tolerance_time_minutes = self.comparison_config.get('tolerance_time_minutes', 15)
            self.tolerance_pnl_percentage = self.comparison_config.get('tolerance_pnl_percentage', 5.0)
            self.export_comparison_csv = self.comparison_config.get('export_comparison_csv', True)
            
            self.log_detailed(f"QuantMan CSV path from config: '{self.quantman_csv_path}'", "INFO")
            self.log_detailed(f"Premium tolerance: {self.tolerance_premium}", "INFO")
            self.log_detailed(f"Time tolerance: {self.tolerance_time_minutes} minutes", "INFO")
            self.log_detailed(f"P&L tolerance: {self.tolerance_pnl_percentage}%", "INFO")
            
            if not self.quantman_csv_path:
                self.log_detailed("INITIALIZATION FAILED: quantman_csv_path is empty or missing", "ERROR")
                self.enabled = False
            else:
                self.log_detailed("Step 4: Validating QuantMan CSV file path", "INFO")
                
                # Convert to Path and resolve relative paths from project root
                quantman_path = Path(self.quantman_csv_path)
                self.log_detailed(f"Original path: {self.quantman_csv_path}", "INFO")
                self.log_detailed(f"Path object: {quantman_path}", "INFO")
                self.log_detailed(f"Is absolute: {quantman_path.is_absolute()}", "INFO")
                
                # If relative path, resolve from project root directory
                if not quantman_path.is_absolute():
                    # Find project root (vAlgo directory)
                    current_path = Path(__file__).parent
                    project_root = None
                    
                    # Walk up the directory tree to find vAlgo folder
                    for parent in [current_path] + list(current_path.parents):
                        if parent.name == 'vAlgo':
                            project_root = parent
                            break
                    
                    if project_root:
                        quantman_path = project_root / quantman_path
                        self.log_detailed(f"Resolved relative path to: {quantman_path}", "INFO")
                    else:
                        self.log_detailed("Could not find vAlgo project root, using current directory", "WARNING")
                        quantman_path = Path.cwd() / quantman_path
                
                self.log_detailed(f"Final absolute path: {quantman_path.absolute()}", "INFO")
                self.log_detailed(f"Path exists: {quantman_path.exists()}", "INFO")
                
                if quantman_path.exists():
                    try:
                        file_size = quantman_path.stat().st_size
                        self.log_detailed(f"File size: {file_size} bytes", "INFO")
                        self.log_detailed("Step 5: QuantMan CSV validation PASSED", "INFO")
                        self.log_detailed(f"SUCCESS: QuantMan CSV found at: {quantman_path.absolute()}", "INFO")
                    except Exception as e:
                        self.log_detailed(f"Error reading file stats: {e}", "ERROR")
                        self.enabled = False
                else:
                    self.log_detailed("Step 5: QuantMan CSV validation FAILED", "ERROR")
                    self.log_detailed(f"FAILURE: QuantMan CSV not found at: {quantman_path.absolute()}", "ERROR")
                    
                    # Try to find potential alternative paths
                    potential_paths = [
                        Path("output/quantman") / quantman_path.name if quantman_path.name else None,
                        Path("../output/quantman") / quantman_path.name if quantman_path.name else None,
                        Path.cwd() / "output/quantman" / quantman_path.name if quantman_path.name else None
                    ]
                    
                    self.log_detailed("Checking potential alternative paths:", "INFO")
                    for i, alt_path in enumerate(potential_paths):
                        if alt_path:
                            self.log_detailed(f"  Alt path {i+1}: {alt_path.absolute()} - Exists: {alt_path.exists()}", "INFO")
                    
                    self.enabled = False
        
        final_status = "ENABLED" if self.enabled else "DISABLED"
        self.log_detailed(f"=== FINAL RESULT: TradeComparator {final_status} ===", "INFO")
        
        self.log_major_component(
            f"Trade Comparator initialized - Enabled: {self.enabled}",
            "TRADE_COMPARATOR"
        )
    
    def log_major_component(self, message: str, component: str):
        """Log major component message."""
        if hasattr(self.logger, 'log_major_component'):
            self.logger.log_major_component(message, component)
        else:
            print(f"{component}: {message}")
    
    def log_detailed(self, message: str, level: str):
        """Log detailed message."""
        if hasattr(self.logger, 'log_detailed'):
            self.logger.log_detailed(message, level)
        else:
            print(f"{level}: {message}")
    
    def compare_trades(self, vbt_trades_data: List[Dict], backtesting_config: Dict) -> Optional[Dict[str, Any]]:
        """
        Compare VBT trades with QuantMan reference data.
        
        Args:
            vbt_trades_data: List of VBT trade dictionaries
            backtesting_config: Backtesting configuration with date range
            
        Returns:
            Dictionary containing comparison results or None if disabled/failed
        """
        if not self.enabled:
            self.log_detailed("Trade comparison disabled - skipping", "INFO")
            return None
            
        try:
            self.log_major_component("Starting trade comparison analysis", "TRADE_COMPARATOR")
            self.log_detailed(f"VBT trades data received: {len(vbt_trades_data) if vbt_trades_data else 0} trades", "DEBUG")
            self.log_detailed(f"Backtesting config: {backtesting_config}", "DEBUG")
            
            # Debug: Log first few VBT trades structure
            if vbt_trades_data:
                self.log_detailed(f"Sample VBT trade structure: {vbt_trades_data[0]}", "DEBUG")
            else:
                self.log_detailed("No VBT trades data provided - cannot perform comparison", "ERROR")
                return None
            
            # Load and filter QuantMan data
            quantman_trades = self._load_and_filter_quantman_data(backtesting_config)
            if not quantman_trades:
                self.log_detailed("No QuantMan trades found for comparison period", "WARNING")
                return None
                
            self.log_detailed(f"QuantMan trades loaded: {len(quantman_trades)} trades", "DEBUG")
            
            # Prepare VBT trades for comparison
            vbt_trades_df = self._prepare_vbt_trades(vbt_trades_data)
            quantman_trades_df = self._prepare_quantman_trades(quantman_trades)
            
            # Perform trade matching and comparison
            comparison_results = self._perform_trade_comparison(vbt_trades_df, quantman_trades_df)
            
            # Generate comparison statistics
            comparison_stats = self._generate_comparison_stats(comparison_results)
            
            self.log_major_component(
                f"Trade comparison completed - {len(comparison_results)} comparisons analyzed",
                "TRADE_COMPARATOR"
            )
            
            return {
                'comparison_results': comparison_results,
                'comparison_stats': comparison_stats,
                'vbt_trades_count': len(vbt_trades_df),
                'quantman_trades_count': len(quantman_trades_df)
            }
            
        except Exception as e:
            self.log_detailed(f"Error in trade comparison: {e}", "ERROR")
            return None
    
    def _load_and_filter_quantman_data(self, backtesting_config: Dict) -> List[Dict]:
        """
        Load QuantMan CSV and filter by backtesting date range.
        
        Args:
            backtesting_config: Configuration containing start_date and end_date
            
        Returns:
            List of filtered QuantMan trade dictionaries
        """
        try:
            # Load QuantMan CSV
            quantman_df = pd.read_csv(self.quantman_csv_path)
            
            # Parse date range from backtesting config
            start_date = pd.to_datetime(backtesting_config.get('start_date'))
            end_date = pd.to_datetime(backtesting_config.get('end_date'))
            
            self.log_detailed(
                f"Filtering QuantMan data for period: {start_date.date()} to {end_date.date()}",
                "INFO"
            )
            
            # Parse entry time and filter by date range
            quantman_df['Entry_Time_Parsed'] = pd.to_datetime(quantman_df['Entry Time'])
            quantman_df['Exit_Time_Parsed'] = pd.to_datetime(quantman_df['Exit Time'])
            
            # Filter trades within backtesting period
            filtered_df = quantman_df[
                (quantman_df['Entry_Time_Parsed'].dt.date >= start_date.date()) &
                (quantman_df['Entry_Time_Parsed'].dt.date <= end_date.date())
            ]
            
            self.log_detailed(
                f"Filtered QuantMan trades: {len(filtered_df)} out of {len(quantman_df)} total trades",
                "INFO"
            )
            
            return filtered_df.to_dict('records')
            
        except Exception as e:
            self.log_detailed(f"Error loading QuantMan data: {e}", "ERROR")
            return []
    
    def _prepare_vbt_trades(self, vbt_trades_data: List[Dict]) -> pd.DataFrame:
        """
        Prepare VBT trades data for comparison.
        
        Args:
            vbt_trades_data: List of VBT trade dictionaries from P&L calculator
            
        Returns:
            Prepared DataFrame with standardized columns
        """
        try:
            if not vbt_trades_data:
                self.log_detailed("No VBT trades data to prepare", "WARNING")
                return pd.DataFrame()
                
            df = pd.DataFrame(vbt_trades_data)
            self.log_detailed(f"VBT trades DataFrame columns: {list(df.columns)}", "DEBUG")
            
            # Handle different possible column names from P&L calculator
            # The P&L calculator returns: entry_timestamp, exit_timestamp, entry_strike, etc.
            timestamp_mapping = {
                'entry_timestamp': 'Entry_Timestamp_Parsed',
                'exit_timestamp': 'Exit_Timestamp_Parsed',
                'Entry_Timestamp': 'Entry_Timestamp_Parsed',  # Alternative format
                'Exit_Timestamp': 'Exit_Timestamp_Parsed'     # Alternative format
            }
            
            # Parse timestamps with flexible column names
            for original_col, new_col in timestamp_mapping.items():
                if original_col in df.columns:
                    df[new_col] = pd.to_datetime(df[original_col])
                    self.log_detailed(f"Parsed timestamps from column: {original_col}", "DEBUG")
                    break
            
            # Also handle exit timestamps separately
            exit_timestamp_mapping = {
                'exit_timestamp': 'Exit_Timestamp_Parsed',
                'Exit_Timestamp': 'Exit_Timestamp_Parsed'
            }
            
            for original_col, new_col in exit_timestamp_mapping.items():
                if original_col in df.columns:
                    df[new_col] = pd.to_datetime(df[original_col])
                    self.log_detailed(f"Parsed exit timestamps from column: {original_col}", "DEBUG")
                    break
            
            # Handle strike price columns
            strike_mapping = {
                'entry_strike': 'Strike_Price',
                'Entry_Strike': 'Strike_Price',
                'strike': 'Strike_Price'
            }
            
            for original_col, new_col in strike_mapping.items():
                if original_col in df.columns:
                    df[new_col] = pd.to_numeric(df[original_col], errors='coerce')
                    self.log_detailed(f"Extracted strike prices from column: {original_col}", "DEBUG")
                    break
            
            # Handle option type columns
            option_type_mapping = {
                'option_type': 'Option_Type',
                'Option_Type': 'Option_Type',
                'type': 'Option_Type'
            }
            
            for original_col, new_col in option_type_mapping.items():
                if original_col in df.columns:
                    df[new_col] = df[original_col]
                    self.log_detailed(f"Extracted option types from column: {original_col}", "DEBUG")
                    break
            
            # Handle premium columns
            premium_mapping = {
                'entry_premium': 'Entry_Premium',
                'exit_premium': 'Exit_Premium',
                'Entry_Premium': 'Entry_Premium',
                'Exit_Premium': 'Exit_Premium'
            }
            
            for original_col, new_col in premium_mapping.items():
                if original_col in df.columns:
                    df[new_col] = pd.to_numeric(df[original_col], errors='coerce')
                    self.log_detailed(f"Extracted premiums from column: {original_col}", "DEBUG")
            
            # Handle P&L columns (Options_PnL is the primary P&L field from VBT)
            pnl_mapping = {
                'Options_PnL': 'VBT_PnL',
                'options_pnl': 'VBT_PnL',
                'Net_PnL': 'VBT_PnL',
                'net_pnl': 'VBT_PnL',
                'pnl': 'VBT_PnL'
            }
            
            for original_col, new_col in pnl_mapping.items():
                if original_col in df.columns:
                    df[new_col] = pd.to_numeric(df[original_col], errors='coerce')
                    self.log_detailed(f"Extracted P&L from column: {original_col}", "DEBUG")
                    break
            
            # If no P&L column found, log warning
            if 'VBT_PnL' not in df.columns:
                self.log_detailed("Warning: No P&L column found in VBT trades data", "WARNING")
                self.log_detailed(f"Available columns: {list(df.columns)}", "DEBUG")
            
            # Verify required columns exist
            required_columns = ['Entry_Timestamp_Parsed', 'Exit_Timestamp_Parsed', 'Strike_Price', 'Option_Type']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.log_detailed(f"Missing required columns in VBT trades: {missing_columns}", "ERROR")
                self.log_detailed(f"Available columns: {list(df.columns)}", "DEBUG")
                return pd.DataFrame()
            
            self.log_detailed(f"Successfully prepared {len(df)} VBT trades for comparison", "DEBUG")
            return df
            
        except Exception as e:
            self.log_detailed(f"Error preparing VBT trades: {e}", "ERROR")
            import traceback
            self.log_detailed(f"Traceback: {traceback.format_exc()}", "DEBUG")
            return pd.DataFrame()
    
    def _prepare_quantman_trades(self, quantman_trades: List[Dict]) -> pd.DataFrame:
        """
        Prepare QuantMan trades data for comparison.
        
        Args:
            quantman_trades: List of filtered QuantMan trade dictionaries
            
        Returns:
            Prepared DataFrame with standardized columns
        """
        try:
            df = pd.DataFrame(quantman_trades)
            
            # Parse instrument to extract strike and option type
            df['Strike_Price'] = df['Instrument'].apply(self._extract_strike_from_instrument)
            df['Option_Type'] = df['Instrument'].apply(self._extract_option_type_from_instrument)
            
            # Standardize premium columns
            df['Entry_Premium'] = pd.to_numeric(df['Entry Price'], errors='coerce')
            df['Exit_Premium'] = pd.to_numeric(df['Exit Price'], errors='coerce')
            
            # Extract P&L data (QuantMan uses 'Profit' column)
            df['QuantMan_PnL'] = pd.to_numeric(df['Profit'], errors='coerce')
            
            return df
            
        except Exception as e:
            self.log_detailed(f"Error preparing QuantMan trades: {e}", "ERROR")
            return pd.DataFrame()
    
    def _extract_strike_from_instrument(self, instrument: str) -> Optional[float]:
        """
        Extract strike price from QuantMan instrument name.
        
        Args:
            instrument: Instrument string like "NIFTY04JAN2421750CE"
            
        Returns:
            Strike price as float or None if not found
        """
        try:
            # Pattern to match strike price in instrument name
            # Example: NIFTY04JAN2421750CE -> 21750
            match = re.search(r'(\d{5})(?:CE|PE)$', instrument)
            if match:
                return float(match.group(1))
            return None
        except Exception:
            return None
    
    def _extract_option_type_from_instrument(self, instrument: str) -> Optional[str]:
        """
        Extract option type from QuantMan instrument name.
        
        Args:
            instrument: Instrument string like "NIFTY04JAN2421750CE"
            
        Returns:
            Option type as "CALL" or "PUT" or None if not found
        """
        try:
            if instrument.endswith('CE'):
                return 'CALL'
            elif instrument.endswith('PE'):
                return 'PUT'
            return None
        except Exception:
            return None
    
    def _perform_trade_comparison(self, vbt_df: pd.DataFrame, quantman_df: pd.DataFrame) -> List[Dict]:
        """
        Perform detailed trade comparison between VBT and QuantMan data.
        
        Args:
            vbt_df: Prepared VBT trades DataFrame
            quantman_df: Prepared QuantMan trades DataFrame
            
        Returns:
            List of comparison result dictionaries
        """
        comparison_results = []
        
        try:
            self.log_detailed(f"Starting comparison: {len(vbt_df)} VBT trades vs {len(quantman_df)} QuantMan trades", "DEBUG")
            
            # Debug: Show sample data from both DataFrames
            if len(vbt_df) > 0:
                sample_vbt = vbt_df.iloc[0]
                self.log_detailed(f"Sample VBT trade: Strike={sample_vbt.get('Strike_Price')}, Type={sample_vbt.get('Option_Type')}, Entry={sample_vbt.get('Entry_Timestamp_Parsed')}", "DEBUG")
            
            if len(quantman_df) > 0:
                sample_qm = quantman_df.iloc[0]
                self.log_detailed(f"Sample QuantMan trade: Strike={sample_qm.get('Strike_Price')}, Type={sample_qm.get('Option_Type')}, Entry={sample_qm.get('Entry_Time_Parsed')}", "DEBUG")
            
            # For each VBT trade, find best matching QuantMan trade
            for idx, vbt_trade in vbt_df.iterrows():
                self.log_detailed(f"Processing VBT trade {idx+1}: Strike={vbt_trade.get('Strike_Price')}, Type={vbt_trade.get('Option_Type')}, Entry={vbt_trade.get('Entry_Timestamp_Parsed')}", "DEBUG")
                match_result = self._find_best_match(vbt_trade, quantman_df)
                comparison_results.append(match_result)
                self.log_detailed(f"Match result for VBT trade {idx+1}: {match_result['trade_status']}", "DEBUG")
            
            # Find QuantMan trades that have no VBT match (missing trades)
            matched_quantman_indices = [r['quantman_match_index'] for r in comparison_results if r['quantman_match_index'] is not None]
            unmatched_quantman = quantman_df[~quantman_df.index.isin(matched_quantman_indices)]
            
            # Add missing trades to results
            for idx, missing_trade in unmatched_quantman.iterrows():
                comparison_results.append({
                    'trade_status': 'MISSING_IN_VBT',
                    'vbt_trade_id': None,
                    'quantman_transaction': missing_trade.get('Transaction', ''),
                    'quantman_match_index': idx,
                    'entry_time_match': False,
                    'exit_time_match': False,
                    'entry_premium_vbt': None,
                    'entry_premium_quantman': missing_trade.get('Entry_Premium'),
                    'entry_premium_diff': None,
                    'entry_premium_within_tolerance': None,
                    'exit_premium_vbt': None,
                    'exit_premium_quantman': missing_trade.get('Exit_Premium'),
                    'exit_premium_diff': None,
                    'exit_premium_within_tolerance': None,
                    'pnl_vbt': None,
                    'pnl_quantman': missing_trade.get('QuantMan_PnL'),
                    'pnl_diff': None,
                    'pnl_percentage_diff': None,
                    'pnl_within_tolerance': None,
                    'strike_vbt': None,
                    'strike_quantman': missing_trade.get('Strike_Price'),
                    'strike_match': False,
                    'option_type_vbt': None,
                    'option_type_quantman': missing_trade.get('Option_Type'),
                    'option_type_match': False,
                    'entry_time_vbt': None,
                    'entry_time_quantman': missing_trade.get('Entry_Time_Parsed'),
                    'exit_time_vbt': None,
                    'exit_time_quantman': missing_trade.get('Exit_Time_Parsed')
                })
            
            return comparison_results
            
        except Exception as e:
            self.log_detailed(f"Error performing trade comparison: {e}", "ERROR")
            return []
    
    def _find_best_match(self, vbt_trade: pd.Series, quantman_df: pd.DataFrame) -> Dict:
        """
        Find the best matching QuantMan trade for a VBT trade.
        
        Args:
            vbt_trade: Single VBT trade row
            quantman_df: QuantMan trades DataFrame
            
        Returns:
            Dictionary with comparison results
        """
        try:
            # Filter potential matches by option type and strike
            potential_matches = quantman_df[
                (quantman_df['Option_Type'] == vbt_trade['Option_Type']) &
                (quantman_df['Strike_Price'] == vbt_trade['Strike_Price'])
            ]
            
            self.log_detailed(f"VBT Trade matching: Strike={vbt_trade.get('Strike_Price')}, Type={vbt_trade.get('Option_Type')}, found {len(potential_matches)} potential matches", "DEBUG")
            
            if potential_matches.empty:
                self.log_detailed(f"No potential matches found for VBT trade", "DEBUG")
                return self._create_unmatched_result(vbt_trade, None, 'NO_MATCH_FOUND')
            
            # Find closest match by entry time
            time_diffs = abs((potential_matches['Entry_Time_Parsed'] - vbt_trade['Entry_Timestamp_Parsed']).dt.total_seconds() / 60)
            best_match_idx = time_diffs.idxmin()
            best_time_diff = time_diffs.loc[best_match_idx]
            
            self.log_detailed(f"Best time match: {best_time_diff:.1f} minutes difference (tolerance: {self.tolerance_time_minutes})", "DEBUG")
            
            # Check if time difference is within tolerance
            if best_time_diff > self.tolerance_time_minutes:
                self.log_detailed(f"Time tolerance exceeded: {best_time_diff:.1f} > {self.tolerance_time_minutes}", "DEBUG")
                return self._create_unmatched_result(vbt_trade, None, 'TIME_TOLERANCE_EXCEEDED')
            
            best_match = quantman_df.loc[best_match_idx]
            self.log_detailed(f"Match found! QuantMan trade at {best_match.get('Entry_Time_Parsed')}", "DEBUG")
            return self._create_matched_result(vbt_trade, best_match, best_match_idx)
            
        except Exception as e:
            self.log_detailed(f"Error finding best match: {e}", "ERROR")
            return self._create_unmatched_result(vbt_trade, None, 'MATCHING_ERROR')
    
    def _create_matched_result(self, vbt_trade: pd.Series, quantman_trade: pd.Series, quantman_idx: int) -> Dict:
        """Create comparison result for matched trades."""
        try:
            # Calculate premium differences
            entry_premium_diff = abs(vbt_trade['Entry_Premium'] - quantman_trade['Entry_Premium']) if pd.notna(vbt_trade['Entry_Premium']) and pd.notna(quantman_trade['Entry_Premium']) else None
            exit_premium_diff = abs(vbt_trade['Exit_Premium'] - quantman_trade['Exit_Premium']) if pd.notna(vbt_trade['Exit_Premium']) and pd.notna(quantman_trade['Exit_Premium']) else None
            
            # Calculate P&L differences
            pnl_vbt = vbt_trade.get('VBT_PnL')
            pnl_quantman = quantman_trade.get('QuantMan_PnL')
            
            pnl_diff = None
            pnl_percentage_diff = None
            pnl_within_tolerance = None
            
            if pd.notna(pnl_vbt) and pd.notna(pnl_quantman):
                pnl_diff = abs(pnl_vbt - pnl_quantman)
                # Calculate percentage difference relative to the absolute value of QuantMan P&L
                if abs(pnl_quantman) > 0:
                    pnl_percentage_diff = (pnl_diff / abs(pnl_quantman)) * 100
                    pnl_within_tolerance = pnl_percentage_diff <= self.tolerance_pnl_percentage
                else:
                    # If QuantMan P&L is zero, check if VBT P&L is also close to zero
                    pnl_percentage_diff = abs(pnl_vbt) if abs(pnl_vbt) > 0 else 0
                    pnl_within_tolerance = abs(pnl_vbt) <= 10  # Allow small absolute difference for near-zero P&L
            
            # Time matching validation
            entry_time_diff = abs((vbt_trade['Entry_Timestamp_Parsed'] - quantman_trade['Entry_Time_Parsed']).total_seconds() / 60)
            exit_time_diff = abs((vbt_trade['Exit_Timestamp_Parsed'] - quantman_trade['Exit_Time_Parsed']).total_seconds() / 60)
            
            # Generate VBT trade ID if not available (use DataFrame index)
            vbt_trade_id = vbt_trade.get('Trade_ID') or vbt_trade.get('trade_id') or vbt_trade.name + 1
            
            return {
                'trade_status': 'MATCHED',
                'vbt_trade_id': vbt_trade_id,
                'quantman_transaction': quantman_trade.get('Transaction', ''),
                'quantman_match_index': quantman_idx,
                'entry_time_match': entry_time_diff <= self.tolerance_time_minutes,
                'exit_time_match': exit_time_diff <= self.tolerance_time_minutes,
                'entry_premium_vbt': vbt_trade.get('Entry_Premium'),
                'entry_premium_quantman': quantman_trade.get('Entry_Premium'),
                'entry_premium_diff': entry_premium_diff,
                'entry_premium_within_tolerance': entry_premium_diff <= self.tolerance_premium if entry_premium_diff is not None else None,
                'exit_premium_vbt': vbt_trade.get('Exit_Premium'),
                'exit_premium_quantman': quantman_trade.get('Exit_Premium'),
                'exit_premium_diff': exit_premium_diff,
                'exit_premium_within_tolerance': exit_premium_diff <= self.tolerance_premium if exit_premium_diff is not None else None,
                'pnl_vbt': pnl_vbt,
                'pnl_quantman': pnl_quantman,
                'pnl_diff': pnl_diff,
                'pnl_percentage_diff': pnl_percentage_diff,
                'pnl_within_tolerance': pnl_within_tolerance,
                'strike_vbt': vbt_trade.get('Strike_Price'),
                'strike_quantman': quantman_trade.get('Strike_Price'),
                'strike_match': vbt_trade.get('Strike_Price') == quantman_trade.get('Strike_Price'),
                'option_type_vbt': vbt_trade.get('Option_Type'),
                'option_type_quantman': quantman_trade.get('Option_Type'),
                'option_type_match': vbt_trade.get('Option_Type') == quantman_trade.get('Option_Type'),
                'entry_time_vbt': vbt_trade['Entry_Timestamp_Parsed'],
                'entry_time_quantman': quantman_trade['Entry_Time_Parsed'],
                'entry_time_diff_minutes': entry_time_diff,
                'exit_time_vbt': vbt_trade['Exit_Timestamp_Parsed'],
                'exit_time_quantman': quantman_trade['Exit_Time_Parsed'],
                'exit_time_diff_minutes': exit_time_diff
            }
        except Exception as e:
            self.log_detailed(f"Error creating matched result: {e}", "ERROR")
            import traceback
            self.log_detailed(f"Matched result traceback: {traceback.format_exc()}", "DEBUG")
            return self._create_unmatched_result(vbt_trade, None, 'RESULT_CREATION_ERROR')
    
    def _create_unmatched_result(self, vbt_trade: pd.Series, quantman_trade: Optional[pd.Series], reason: str) -> Dict:
        """Create comparison result for unmatched trades."""
        
        # Generate VBT trade ID if not available (use DataFrame index)
        vbt_trade_id = vbt_trade.get('Trade_ID') or vbt_trade.get('trade_id') or vbt_trade.name + 1
        
        return {
            'trade_status': f'UNMATCHED_{reason}',
            'vbt_trade_id': vbt_trade_id,
            'quantman_transaction': quantman_trade.get('Transaction', '') if quantman_trade is not None else '',
            'quantman_match_index': None,
            'entry_time_match': False,
            'exit_time_match': False,
            'entry_premium_vbt': vbt_trade.get('Entry_Premium'),
            'entry_premium_quantman': quantman_trade.get('Entry_Premium') if quantman_trade is not None else None,
            'entry_premium_diff': None,
            'entry_premium_within_tolerance': None,
            'exit_premium_vbt': vbt_trade.get('Exit_Premium'),
            'exit_premium_quantman': quantman_trade.get('Exit_Premium') if quantman_trade is not None else None,
            'exit_premium_diff': None,
            'exit_premium_within_tolerance': None,
            'pnl_vbt': vbt_trade.get('VBT_PnL'),
            'pnl_quantman': quantman_trade.get('QuantMan_PnL') if quantman_trade is not None else None,
            'pnl_diff': None,
            'pnl_percentage_diff': None,
            'pnl_within_tolerance': None,
            'strike_vbt': vbt_trade.get('Strike_Price'),
            'strike_quantman': quantman_trade.get('Strike_Price') if quantman_trade is not None else None,
            'strike_match': False,
            'option_type_vbt': vbt_trade.get('Option_Type'),
            'option_type_quantman': quantman_trade.get('Option_Type') if quantman_trade is not None else None,
            'option_type_match': False,
            'entry_time_vbt': vbt_trade.get('Entry_Timestamp_Parsed'),
            'entry_time_quantman': quantman_trade.get('Entry_Time_Parsed') if quantman_trade is not None else None,
            'entry_time_diff_minutes': None,
            'exit_time_vbt': vbt_trade.get('Exit_Timestamp_Parsed'),
            'exit_time_quantman': quantman_trade.get('Exit_Time_Parsed') if quantman_trade is not None else None,
            'exit_time_diff_minutes': None
        }
    
    def _generate_comparison_stats(self, comparison_results: List[Dict]) -> Dict:
        """
        Generate comprehensive statistical summary of trade comparison results.
        
        Args:
            comparison_results: List of comparison result dictionaries
            
        Returns:
            Dictionary with detailed comparison statistics
        """
        try:
            total_comparisons = len(comparison_results)
            matched_trades = [r for r in comparison_results if r['trade_status'] == 'MATCHED']
            missing_in_vbt = [r for r in comparison_results if r['trade_status'] == 'MISSING_IN_VBT']
            unmatched_trades = [r for r in comparison_results if 'UNMATCHED' in r['trade_status']]
            
            # Basic accuracy counts
            entry_premium_accurate = len([r for r in matched_trades if r.get('entry_premium_within_tolerance') == True])
            exit_premium_accurate = len([r for r in matched_trades if r.get('exit_premium_within_tolerance') == True])
            entry_time_accurate = len([r for r in matched_trades if r.get('entry_time_match') == True])
            exit_time_accurate = len([r for r in matched_trades if r.get('exit_time_match') == True])
            pnl_accurate = len([r for r in matched_trades if r.get('pnl_within_tolerance') == True])
            pnl_comparisons = len([r for r in matched_trades if r.get('pnl_diff') is not None])
            
            # Combined time matching (both entry AND exit match)
            both_time_matches = len([r for r in matched_trades if r.get('entry_time_match') == True and r.get('exit_time_match') == True])
            
            # Premium difference statistics
            entry_premium_diffs = [r.get('entry_premium_diff') for r in matched_trades if r.get('entry_premium_diff') is not None]
            exit_premium_diffs = [r.get('exit_premium_diff') for r in matched_trades if r.get('exit_premium_diff') is not None]
            
            # P&L difference statistics  
            pnl_diffs = [r.get('pnl_diff') for r in matched_trades if r.get('pnl_diff') is not None]
            pnl_percentage_diffs = [r.get('pnl_percentage_diff') for r in matched_trades if r.get('pnl_percentage_diff') is not None]
            
            # Time difference statistics
            entry_time_diffs = [r.get('entry_time_diff_minutes') for r in matched_trades if r.get('entry_time_diff_minutes') is not None]
            exit_time_diffs = [r.get('exit_time_diff_minutes') for r in matched_trades if r.get('exit_time_diff_minutes') is not None]
            
            # Calculate statistics helper function
            def calculate_stats(data_list):
                if not data_list:
                    return {'max': 0, 'min': 0, 'avg': 0, 'count': 0}
                return {
                    'max': max(data_list),
                    'min': min(data_list), 
                    'avg': sum(data_list) / len(data_list),
                    'count': len(data_list)
                }
            
            # Calculate comprehensive statistics
            stats = {
                # Basic counts
                'total_comparisons': total_comparisons,
                'vbt_trades_count': len([r for r in comparison_results if r.get('vbt_trade_id') is not None]),
                'quantman_trades_count': len(missing_in_vbt) + len(matched_trades),
                'matched_trades': len(matched_trades),
                'missing_in_vbt': len(missing_in_vbt),
                'unmatched_trades': len(unmatched_trades),
                
                # Accuracy rates
                'match_rate': len(matched_trades) / total_comparisons if total_comparisons > 0 else 0,
                'entry_premium_accuracy': entry_premium_accurate / len(matched_trades) if matched_trades else 0,
                'exit_premium_accuracy': exit_premium_accurate / len(matched_trades) if matched_trades else 0,
                'entry_time_accuracy': entry_time_accurate / len(matched_trades) if matched_trades else 0,
                'exit_time_accuracy': exit_time_accurate / len(matched_trades) if matched_trades else 0,
                'both_time_matches': both_time_matches,
                'both_time_accuracy': both_time_matches / len(matched_trades) if matched_trades else 0,
                'pnl_accuracy': pnl_accurate / pnl_comparisons if pnl_comparisons > 0 else 0,
                'pnl_comparisons_available': pnl_comparisons,
                
                # Premium difference statistics
                'entry_premium_stats': calculate_stats(entry_premium_diffs),
                'exit_premium_stats': calculate_stats(exit_premium_diffs),
                
                # P&L difference statistics
                'pnl_diff_stats': calculate_stats(pnl_diffs),
                'pnl_percentage_stats': calculate_stats(pnl_percentage_diffs),
                
                # Time difference statistics
                'entry_time_diff_stats': calculate_stats(entry_time_diffs),
                'exit_time_diff_stats': calculate_stats(exit_time_diffs),
                
                # Quality grades
                'overall_grade': self._calculate_comparison_grade(len(matched_trades), total_comparisons, 
                                                               entry_premium_accurate + exit_premium_accurate,
                                                               len(matched_trades) * 2, pnl_accurate, pnl_comparisons),
                
                # Additional insights
                'perfect_matches': len([r for r in matched_trades if 
                                     r.get('entry_time_match') == True and r.get('exit_time_match') == True and
                                     r.get('entry_premium_within_tolerance') == True and r.get('exit_premium_within_tolerance') == True and
                                     r.get('pnl_within_tolerance') == True]),
                'tolerance_settings': {
                    'premium_tolerance': self.tolerance_premium,
                    'time_tolerance_minutes': self.tolerance_time_minutes,
                    'pnl_tolerance_percentage': self.tolerance_pnl_percentage
                }
            }
            
            self.log_detailed(f"Comprehensive Comparison Stats Generated: {len(stats)} metrics", "INFO")
            return stats
            
        except Exception as e:
            self.log_detailed(f"Error generating comparison stats: {e}", "ERROR")
            return {}
    
    def _calculate_comparison_grade(self, matched_count: int, total_count: int, 
                                  premium_accurate: int, premium_total: int,
                                  pnl_accurate: int, pnl_total: int) -> str:
        """Calculate overall comparison grade based on various accuracy metrics."""
        try:
            if total_count == 0:
                return "N/A"
                
            match_rate = matched_count / total_count
            premium_accuracy = premium_accurate / premium_total if premium_total > 0 else 0
            pnl_accuracy = pnl_accurate / pnl_total if pnl_total > 0 else 0
            
            # Weighted scoring: 40% match rate, 30% premium accuracy, 30% P&L accuracy
            overall_score = (match_rate * 0.4) + (premium_accuracy * 0.3) + (pnl_accuracy * 0.3)
            
            if overall_score >= 0.95:
                return "A+ (Outstanding)"
            elif overall_score >= 0.90:
                return "A (Excellent)"
            elif overall_score >= 0.85:
                return "A- (Very Good)"
            elif overall_score >= 0.80:
                return "B+ (Good)"
            elif overall_score >= 0.75:
                return "B (Above Average)"
            elif overall_score >= 0.70:
                return "B- (Average)"
            elif overall_score >= 0.60:
                return "C (Below Average)"
            elif overall_score >= 0.50:
                return "D (Poor)"
            else:
                return "F (Very Poor)"
                
        except Exception:
            return "N/A"
    
    def export_comparison_csv(self, comparison_data: Dict, filename_base: str) -> Optional[str]:
        """
        Export trade comparison results to CSV.
        
        Args:
            comparison_data: Dictionary containing comparison results
            filename_base: Base filename for export
            
        Returns:
            Path to exported CSV file or None if failed
        """
        if not self.export_comparison_csv or not comparison_data:
            return None
            
        try:
            # Create output directory if needed
            output_dir = Path(self.comparison_output_path)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Prepare comparison DataFrame
            comparison_df = pd.DataFrame(comparison_data['comparison_results'])
            
            # Create comparison CSV filename
            csv_filename = f"{filename_base}_trade_comparison.csv"
            csv_path = output_dir / csv_filename
            
            # Export to CSV
            comparison_df.to_csv(csv_path, index=False)
            
            self.log_major_component(
                f"Trade comparison CSV exported: {csv_filename}",
                "TRADE_COMPARATOR"
            )
            
            return str(csv_path)
            
        except Exception as e:
            self.log_detailed(f"Error exporting comparison CSV: {e}", "ERROR")
            return None