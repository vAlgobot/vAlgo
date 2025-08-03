"""
Signal Data Processor for vAlgo Rule Engine

Processes historical signal data using configured strategies following the workflow:
Strategy_Config → Entry_Conditions → Exit_Conditions

Generates timestamped signal analysis with rule match details.
"""

import logging
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from pathlib import Path

from .models import RuleResult, IndicatorData
import sys
from pathlib import Path

# Add project root to path for utils import
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.initialize_config_loader import InitializeConfigLoader


class SignalAnalysisResult:
    """Results from signal data analysis"""
    
    def __init__(self, timestamp: datetime, strategy_name: str):
        self.timestamp = timestamp
        self.strategy_name = strategy_name
        self.entry_signal_triggered = False
        self.exit_signal_triggered = False
        self.entry_rule_matches = []
        self.exit_rule_matches = []
        self.entry_rule_details = {}
        self.exit_rule_details = {}
        self.indicator_values = {}
        self.signal_status = "no_signal"  # no_signal, entry, exit, hold
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert result to dictionary for output"""
        return {
            'timestamp': self.timestamp,
            'date': self.timestamp.strftime('%Y-%m-%d'),
            'time': self.timestamp.strftime('%H:%M:%S'),
            'strategy_name': self.strategy_name,
            'signal_status': self.signal_status,
            'entry_signal_triggered': self.entry_signal_triggered,
            'exit_signal_triggered': self.exit_signal_triggered,
            'entry_rule_matches': ', '.join(self.entry_rule_matches),
            'exit_rule_matches': ', '.join(self.exit_rule_matches),
            'entry_rule_details': str(self.entry_rule_details),
            'exit_rule_details': str(self.exit_rule_details),
            'indicator_values_count': len(self.indicator_values)
        }


class SignalDataProcessor:
    """
    Processes signal data using configured strategies and rules
    
    Workflow:
    1. Load configuration from Initialize sheet (dates, signal data path)
    2. Load Strategy_Config to get active strategies and their linked rules
    3. Load Entry_Conditions and Exit_Conditions for actual rule logic
    4. Process signal data applying strategy rules to each data point
    5. Generate comprehensive output with timestamps and signal status
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize signal data processor
        
        Args:
            config_path: Path to Excel config file (optional)
        """
        self.logger = logging.getLogger(__name__)
        self.config_path = config_path
        
        # Initialize config loaders
        self.init_config = InitializeConfigLoader(config_path)
        
        # Load initialization parameters
        self.init_params = {}
        self.signal_data_path = ""
        self.start_date = ""
        self.end_date = ""
        
        # Rule engine will be injected
        self.rule_engine = None
        
        # Results storage
        self.analysis_results: List[SignalAnalysisResult] = []
    
    def load_configuration(self) -> bool:
        """
        Load configuration from Initialize sheet
        
        Returns:
            True if successful, False otherwise
        """
        try:
            if not self.init_config.load_config():
                self.logger.error("Failed to load initialize configuration")
                return False
            
            self.init_params = self.init_config.get_initialize_parameters()
            self.start_date, self.end_date = self.init_config.get_date_range()
            self.signal_data_path = self.init_config.get_signal_data_path()
            
            # Validate configuration
            issues = self.init_config.validate_parameters()
            if issues:
                self.logger.warning(f"Configuration validation issues: {issues}")
            
            if not self.signal_data_path:
                self.logger.error("signal_data_path not configured in Initialize sheet")
                return False
            
            if not self.start_date or not self.end_date:
                self.logger.warning("start_date or end_date not configured, will process all data")
            
            self.logger.info(f"Configuration loaded: signal_data_path={self.signal_data_path}, "
                           f"date_range={self.start_date} to {self.end_date}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error loading configuration: {e}")
            return False
    
    def set_rule_engine(self, rule_engine) -> None:
        """
        Set the rule engine instance to use for evaluations
        
        Args:
            rule_engine: RuleEngine instance
        """
        self.rule_engine = rule_engine
    
    def load_signal_data(self) -> Optional[pd.DataFrame]:
        """
        Load signal data from configured CSV file
        
        Returns:
            DataFrame with signal data or None if error
        """
        try:
            if not self.signal_data_path or not Path(self.signal_data_path).exists():
                self.logger.error(f"Signal data file not found: {self.signal_data_path}")
                return None
            
            # Load CSV data
            df = pd.read_csv(self.signal_data_path)
            
            if df.empty:
                self.logger.error("Signal data file is empty")
                return None
            
            # Parse timestamp column (assuming it exists)
            timestamp_columns = ['timestamp', 'datetime', 'date_time', 'time']
            timestamp_col = None
            
            for col in timestamp_columns:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if timestamp_col:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df = df.sort_values(timestamp_col)
            else:
                self.logger.warning("No timestamp column found in signal data")
            
            # Filter by date range if configured
            if self.start_date and self.end_date and timestamp_col:
                start_dt = datetime.strptime(self.start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(self.end_date, '%Y-%m-%d')
                
                mask = (df[timestamp_col].dt.date >= start_dt.date()) & \
                       (df[timestamp_col].dt.date <= end_dt.date())
                df = df[mask]
                
                self.logger.info(f"Filtered signal data to date range: {self.start_date} to {self.end_date}")
            
            self.logger.info(f"Loaded signal data: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading signal data: {e}")
            return None
    
    def get_active_strategies(self) -> Dict[str, Dict[str, Any]]:
        """
        Get active strategies from Strategy_Config sheet
        
        Returns:
            Dictionary of active strategies with their configurations
        """
        if not self.rule_engine:
            self.logger.error("Rule engine not set")
            return {}
        
        try:
            strategies = self.rule_engine.get_available_strategies()
            active_strategies = {}
            
            for strategy_name in strategies:
                strategy_config = self.rule_engine.get_strategy_config(strategy_name)
                if strategy_config and strategy_config.get('status', '').lower() == 'active':
                    active_strategies[strategy_name] = strategy_config
            
            self.logger.info(f"Found {len(active_strategies)} active strategies")
            return active_strategies
            
        except Exception as e:
            self.logger.error(f"Error getting active strategies: {e}")
            return {}
    
    def process_signal_data(self) -> bool:
        """
        Process signal data using configured strategies
        
        Returns:
            True if successful, False otherwise
        """
        try:
            # Load configuration
            if not self.load_configuration():
                return False
            
            # Load signal data
            signal_df = self.load_signal_data()
            if signal_df is None:
                return False
            
            # Get active strategies
            active_strategies = self.get_active_strategies()
            if not active_strategies:
                self.logger.error("No active strategies found")
                return False
            
            # Clear previous results
            self.analysis_results = []
            
            # Process each row of signal data
            timestamp_col = self._find_timestamp_column(signal_df)
            
            for idx, row in signal_df.iterrows():
                # Get timestamp for this row
                if timestamp_col:
                    timestamp = row[timestamp_col]
                    if not isinstance(timestamp, datetime):
                        timestamp = pd.to_datetime(timestamp)
                else:
                    timestamp = datetime.now()
                
                # Convert row to IndicatorData
                indicator_data = IndicatorData.from_dict(row.to_dict())
                
                # Process each active strategy
                for strategy_name, strategy_config in active_strategies.items():
                    result = self._process_strategy_for_signal(
                        strategy_name, strategy_config, indicator_data, timestamp
                    )
                    if result:
                        self.analysis_results.append(result)
            
            self.logger.info(f"Processed {len(signal_df)} signal data rows for {len(active_strategies)} strategies")
            self.logger.info(f"Generated {len(self.analysis_results)} analysis results")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing signal data: {e}")
            return False
    
    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """Find timestamp column in DataFrame"""
        timestamp_columns = ['timestamp', 'datetime', 'date_time', 'time']
        for col in timestamp_columns:
            if col in df.columns:
                return col
        return None
    
    def _process_strategy_for_signal(self, strategy_name: str, strategy_config: Dict[str, Any], 
                                   indicator_data: IndicatorData, timestamp: datetime) -> Optional[SignalAnalysisResult]:
        """
        Process a single strategy against signal data
        
        Args:
            strategy_name: Name of the strategy
            strategy_config: Strategy configuration
            indicator_data: Indicator data for this signal
            timestamp: Timestamp of the signal
            
        Returns:
            SignalAnalysisResult or None if error
        """
        try:
            result = SignalAnalysisResult(timestamp, strategy_name)
            result.indicator_values = indicator_data.data.copy()
            
            # Get entry and exit rules for this strategy
            entry_rules = strategy_config.get('entry_rules', [])
            exit_rules = strategy_config.get('exit_rules', [])
            
            # Process entry rules
            entry_triggered = False
            for entry_rule in entry_rules:
                entry_result = self.rule_engine.evaluate_entry_rule(entry_rule, indicator_data)
                if entry_result.triggered:
                    entry_triggered = True
                    result.entry_rule_matches.append(entry_rule)
                    result.entry_rule_details[entry_rule] = {
                        'matched_conditions': entry_result.matched_conditions,
                        'evaluation_details': entry_result.evaluation_details
                    }
            
            result.entry_signal_triggered = entry_triggered
            
            # Process exit rules
            exit_triggered = False
            for exit_rule in exit_rules:
                exit_result = self.rule_engine.evaluate_exit_rule(exit_rule, indicator_data)
                if exit_result.triggered:
                    exit_triggered = True
                    result.exit_rule_matches.append(exit_rule)
                    result.exit_rule_details[exit_rule] = {
                        'matched_conditions': exit_result.matched_conditions,
                        'evaluation_details': exit_result.evaluation_details
                    }
            
            result.exit_signal_triggered = exit_triggered
            
            # Determine signal status
            if entry_triggered and exit_triggered:
                result.signal_status = "both_signals"
            elif entry_triggered:
                result.signal_status = "entry"
            elif exit_triggered:
                result.signal_status = "exit"
            else:
                result.signal_status = "no_signal"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing strategy {strategy_name}: {e}")
            return None
    
    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get analysis results as DataFrame
        
        Returns:
            DataFrame with analysis results
        """
        if not self.analysis_results:
            return pd.DataFrame()
        
        # Convert results to list of dictionaries
        results_data = [result.to_dict() for result in self.analysis_results]
        
        return pd.DataFrame(results_data)
    
    def export_results(self, output_path: str, format: str = 'csv') -> bool:
        """
        Export analysis results to file
        
        Args:
            output_path: Path for output file
            format: Output format ('csv' or 'excel')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            results_df = self.get_results_dataframe()
            
            if results_df.empty:
                self.logger.warning("No results to export")
                return False
            
            if format.lower() == 'csv':
                results_df.to_csv(output_path, index=False)
            elif format.lower() in ['excel', 'xlsx']:
                results_df.to_excel(output_path, index=False)
            else:
                self.logger.error(f"Unsupported export format: {format}")
                return False
            
            self.logger.info(f"Exported {len(results_df)} results to {output_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error exporting results: {e}")
            return False
    
    def get_summary_statistics(self) -> Dict[str, Any]:
        """
        Get summary statistics of analysis results
        
        Returns:
            Dictionary with summary statistics
        """
        if not self.analysis_results:
            return {}
        
        total_signals = len(self.analysis_results)
        entry_signals = sum(1 for r in self.analysis_results if r.entry_signal_triggered)
        exit_signals = sum(1 for r in self.analysis_results if r.exit_signal_triggered)
        no_signals = sum(1 for r in self.analysis_results if r.signal_status == "no_signal")
        
        # Group by strategy
        strategy_stats = {}
        for result in self.analysis_results:
            strategy = result.strategy_name
            if strategy not in strategy_stats:
                strategy_stats[strategy] = {
                    'total': 0, 'entry': 0, 'exit': 0, 'no_signal': 0
                }
            
            strategy_stats[strategy]['total'] += 1
            if result.entry_signal_triggered:
                strategy_stats[strategy]['entry'] += 1
            if result.exit_signal_triggered:
                strategy_stats[strategy]['exit'] += 1
            if result.signal_status == "no_signal":
                strategy_stats[strategy]['no_signal'] += 1
        
        return {
            'total_signals_processed': total_signals,
            'entry_signals_triggered': entry_signals,
            'exit_signals_triggered': exit_signals,
            'no_signals': no_signals,
            'entry_signal_rate': entry_signals / total_signals if total_signals > 0 else 0,
            'exit_signal_rate': exit_signals / total_signals if total_signals > 0 else 0,
            'strategy_breakdown': strategy_stats,
            'date_range': f"{self.start_date} to {self.end_date}" if self.start_date and self.end_date else "All data"
        }


# Convenience function
def process_signal_data_with_config(config_path: Optional[str] = None, rule_engine=None) -> SignalDataProcessor:
    """
    Create and run signal data processor with configuration
    
    Args:
        config_path: Path to config file (optional)
        rule_engine: RuleEngine instance
        
    Returns:
        Configured SignalDataProcessor
    """
    processor = SignalDataProcessor(config_path)
    if rule_engine:
        processor.set_rule_engine(rule_engine)
    return processor