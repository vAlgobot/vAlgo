"""
Initialize Configuration Loader for vAlgo Trading System

Reusable utility to load initialization parameters from Excel config file's Initialize sheet.
Provides centralized access to start_date, end_date, signal_data_path and other initialization settings.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

try:
    import pandas as pd
except ImportError:
    pd = None


class InitializeConfigLoader:
    """
    Utility class to load initialization parameters from Excel config
    
    Provides reusable access to initialization settings like:
    - start_date, end_date for date ranges
    - signal_data_path for CSV signal data location
    - mode, capital, and other system parameters
    """
    
    def __init__(self, config_file: Optional[str] = None):
        """
        Initialize the config loader
        
        Args:
            config_file: Path to Excel config file (optional)
        """
        self.logger = logging.getLogger(__name__)
        
        # Default config file path
        if config_file is None:
            config_file = "config/config.xlsx"
        
        # Make config file path absolute
        if not os.path.isabs(config_file):
            project_root = self._get_project_root()
            config_file = str(project_root / config_file)
            
        self.config_file = config_file
        self.config_data: Dict[str, Any] = {}
        self._loaded = False
    
    def _get_project_root(self) -> Path:
        """Find project root directory"""
        current_path = Path(__file__).parent
        
        for _ in range(5):
            if any((current_path / indicator).exists() for indicator in 
                   ['README.md', '.git', 'main_backtest.py', 'requirements.txt']):
                return current_path
            current_path = current_path.parent
            if current_path == current_path.parent:
                break
        
        return Path.cwd()
    
    def load_config(self, force_reload: bool = False) -> bool:
        """
        Load configuration from Excel file
        
        Args:
            force_reload: If True, reload config even if already loaded
        
        Returns:
            True if successful, False otherwise
        """
        global pd
        
        if self._loaded and not force_reload:
            return True
        
        try:
            if not os.path.exists(self.config_file):
                self.logger.error(f"Config file not found: {self.config_file}")
                return False
            
            # Check if pandas is available
            if pd is None:
                try:
                    import pandas as pd_import
                    pd = pd_import
                except ImportError as e:
                    self.logger.error(f"pandas not available: {e}")
                    return False
            
            # Load Initialize sheet from Excel file
            try:
                df = pd.read_excel(self.config_file, sheet_name='Initialize')
                if df.empty:
                    self.logger.warning("Initialize sheet is empty")
                    return False
                
                self.config_data = df
                self._loaded = True
                self.logger.info(f"Initialize config loaded from {self.config_file}")
                return True
                
            except Exception as e:
                self.logger.error(f"Cannot read Initialize sheet from {self.config_file}: {e}")
                return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error loading config: {e}")
            return False
    
    def _ensure_loaded(self) -> bool:
        """Ensure config is loaded before use"""
        if not self._loaded:
            return self.load_config()
        return True
    
    def get_initialize_parameters(self) -> Dict[str, Any]:
        """
        Get all initialization parameters from Initialize sheet
        
        Returns:
            Dictionary with all initialization parameters
        """
        if not self._ensure_loaded():
            self.logger.warning("Config not loaded, returning empty parameters")
            return {}
        
        try:
            df = self.config_data
            config = {}
            
            # Convert DataFrame to dictionary (Parameter -> Value mapping)
            for _, row in df.iterrows():
                try:
                    param = str(row.get('Parameter', '')).lower().strip()
                    value = row.get('Value', '')
                    
                    if param:
                        # Special handling for date parameters
                        if param in ['start_date', 'end_date']:
                            config[param] = self._parse_user_friendly_date(value)
                        elif param in ['mode', 'trading_mode']:
                            config['mode'] = str(value).lower().strip()
                        elif param in ['capital']:
                            config[param] = float(value) if pd.notna(value) else 100000.0
                        elif param in ['daily_entry_limit']:
                            config[param] = int(value) if pd.notna(value) else 1
                        elif param in ['signal_data_path', 'data_path']:
                            # Handle signal data path - make absolute if relative
                            if pd.notna(value) and str(value).strip():
                                data_path = str(value).strip()
                                if not os.path.isabs(data_path):
                                    project_root = self._get_project_root()
                                    data_path = str(project_root / data_path)
                                config['signal_data_path'] = data_path
                        else:
                            config[param] = value
                            
                except Exception as e:
                    self.logger.warning(f"Error processing initialize row: {e}")
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error processing initialize parameters: {e}")
            return {}
    
    def get_parameter(self, parameter_name: str, default_value: Any = None) -> Any:
        """
        Get specific parameter value
        
        Args:
            parameter_name: Name of parameter to retrieve
            default_value: Default value if parameter not found
            
        Returns:
            Parameter value or default
        """
        params = self.get_initialize_parameters()
        param_key = parameter_name.lower().strip()
        return params.get(param_key, default_value)
    
    def get_date_range(self) -> tuple:
        """
        Get start_date and end_date as tuple
        
        Returns:
            Tuple of (start_date, end_date) as strings in YYYY-MM-DD format
        """
        params = self.get_initialize_parameters()
        start_date = params.get('start_date', '')
        end_date = params.get('end_date', '')
        
        return start_date, end_date
    
    def get_signal_data_path(self) -> str:
        """
        Get signal data path from configuration
        
        Returns:
            Absolute path to signal data CSV file
        """
        return self.get_parameter('signal_data_path', '')
    
    def get_trading_mode(self) -> str:
        """
        Get trading mode from configuration
        
        Returns:
            Trading mode (backtest, paper, live, etc.)
        """
        return self.get_parameter('mode', 'backtest')
    
    def get_capital(self) -> float:
        """
        Get initial capital from configuration
        
        Returns:
            Initial capital amount
        """
        return self.get_parameter('capital', 100000.0)
    
    def validate_parameters(self) -> list:
        """
        Validate initialization parameters
        
        Returns:
            List of validation issues
        """
        issues = []
        params = self.get_initialize_parameters()
        
        # Check required parameters
        required_params = ['start_date', 'end_date', 'mode']
        for param in required_params:
            if not params.get(param):
                issues.append(f"Missing required parameter: {param}")
        
        # Validate date format
        start_date = params.get('start_date', '')
        end_date = params.get('end_date', '')
        
        if start_date and not self._is_valid_date_format(start_date):
            issues.append(f"Invalid start_date format: {start_date}. Use YYYY-MM-DD")
        
        if end_date and not self._is_valid_date_format(end_date):
            issues.append(f"Invalid end_date format: {end_date}. Use YYYY-MM-DD")
        
        # Validate date range logic
        if start_date and end_date and self._is_valid_date_format(start_date) and self._is_valid_date_format(end_date):
            try:
                start_dt = datetime.strptime(start_date, '%Y-%m-%d')
                end_dt = datetime.strptime(end_date, '%Y-%m-%d')
                if end_dt < start_dt:
                    issues.append(f"end_date ({end_date}) cannot be before start_date ({start_date})")
            except ValueError as e:
                issues.append(f"Date validation error: {e}")
        
        # Validate signal data path if provided
        signal_path = params.get('signal_data_path', '')
        if signal_path:
            if not os.path.exists(signal_path):
                issues.append(f"Signal data file not found: {signal_path}")
            else:
                # Check signal data date range
                date_range_issues = self._validate_signal_data_date_range(signal_path, start_date, end_date)
                issues.extend(date_range_issues)
        
        # Validate mode
        valid_modes = ['backtest', 'paper', 'live', 'dataload']
        mode = params.get('mode', '')
        if mode and mode not in valid_modes:
            issues.append(f"Invalid mode: {mode}. Valid options: {valid_modes}")
        
        return issues
    
    def _parse_user_friendly_date(self, date_value: Any) -> str:
        """
        Parse date from user-friendly dd-MM-yyyy format to system yyyy-MM-dd format
        
        Args:
            date_value: Date value from Excel (can be datetime, string, etc.)
            
        Returns:
            Date string in yyyy-MM-dd format
        """
        if pd.isna(date_value):
            return ""
        
        # If it's already a datetime object (from Excel), extract date part
        if hasattr(date_value, 'strftime'):
            return date_value.strftime('%Y-%m-%d')
        
        # Convert to string for parsing
        date_str = str(date_value).strip()
        if not date_str or date_str.lower() == 'nan':
            return ""
        
        # Try different date formats
        date_formats = [
            '%d-%m-%Y',           # dd-MM-yyyy (user-friendly)
            '%d-%m-%Y %H:%M:%S',  # dd-MM-yyyy HH:MM:SS
            '%Y-%m-%d',           # yyyy-MM-dd (existing format)
            '%Y-%m-%d %H:%M:%S',  # yyyy-MM-dd HH:MM:SS (existing format)
            '%d/%m/%Y',           # dd/MM/yyyy (alternative)
            '%Y/%m/%d',           # yyyy/MM/dd (alternative)
        ]
        
        for date_format in date_formats:
            try:
                parsed_date = datetime.strptime(date_str, date_format)
                return parsed_date.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        # If no format worked, log warning and return as-is
        self.logger.warning(f"Could not parse date format: {date_str}. Please use dd-MM-yyyy or yyyy-MM-dd format.")
        return date_str
    
    def _is_valid_date_format(self, date_str: str) -> bool:
        """Check if date string is in valid YYYY-MM-DD format"""
        try:
            datetime.strptime(date_str, '%Y-%m-%d')
            return True
        except ValueError:
            return False
    
    def _validate_signal_data_date_range(self, signal_path: str, start_date: str, end_date: str) -> list:
        """
        Validate signal data covers the requested date range
        
        Args:
            signal_path: Path to signal data file
            start_date: Requested start date (YYYY-MM-DD)
            end_date: Requested end date (YYYY-MM-DD)
            
        Returns:
            List of validation issues
        """
        issues = []
        
        try:
            import pandas as pd
            
            # Load signal data
            df = pd.read_csv(signal_path)
            
            if df.empty:
                issues.append("Signal data file is empty")
                return issues
            
            # Find timestamp column
            timestamp_columns = ['timestamp', 'datetime', 'date_time', 'time']
            timestamp_col = None
            
            for col in timestamp_columns:
                if col in df.columns:
                    timestamp_col = col
                    break
            
            if not timestamp_col:
                issues.append("No timestamp column found in signal data. Expected one of: " + ", ".join(timestamp_columns))
                return issues
            
            # Parse timestamps
            df[timestamp_col] = pd.to_datetime(df[timestamp_col])
            
            # Get actual date range in signal data
            actual_start = df[timestamp_col].min().strftime('%Y-%m-%d')
            actual_end = df[timestamp_col].max().strftime('%Y-%m-%d')
            
            # Check coverage
            if start_date and end_date:
                requested_start = datetime.strptime(start_date, '%Y-%m-%d').date()
                requested_end = datetime.strptime(end_date, '%Y-%m-%d').date()
                data_start = df[timestamp_col].min().date()
                data_end = df[timestamp_col].max().date()
                
                # Check if requested range is covered by available data
                if requested_start < data_start:
                    issues.append(f"Requested start_date ({start_date}) is before available data start ({actual_start})")
                
                if requested_end > data_end:
                    issues.append(f"Requested end_date ({end_date}) is after available data end ({actual_end})")
                
                # Filter to requested range to see how much data we'll actually get
                mask = (df[timestamp_col].dt.date >= requested_start) & (df[timestamp_col].dt.date <= requested_end)
                filtered_df = df[mask]
                
                if filtered_df.empty:
                    issues.append(f"No signal data found for requested date range {start_date} to {end_date}. "
                                f"Available data: {actual_start} to {actual_end}")
                elif len(filtered_df) < len(df):
                    filtered_days = (filtered_df[timestamp_col].max().date() - filtered_df[timestamp_col].min().date()).days + 1
                    total_days = (df[timestamp_col].max().date() - df[timestamp_col].min().date()).days + 1
                    issues.append(f"INFO: Date filtering will use {len(filtered_df)}/{len(df)} rows "
                                f"({filtered_days}/{total_days} days) from signal data")
                else:
                    issues.append(f"INFO: All signal data ({len(df)} rows) is within requested date range")
            
            # Add info about actual data range
            issues.append(f"INFO: Signal data contains {len(df)} rows from {actual_start} to {actual_end}")
            
        except Exception as e:
            issues.append(f"Error validating signal data date range: {e}")
        
        return issues


# Convenience function
def load_initialize_config(config_file: Optional[str] = None) -> InitializeConfigLoader:
    """
    Create and load InitializeConfigLoader
    
    Args:
        config_file: Path to config file (optional)
        
    Returns:
        Loaded InitializeConfigLoader instance
    """
    loader = InitializeConfigLoader(config_file)
    loader.load_config()
    return loader