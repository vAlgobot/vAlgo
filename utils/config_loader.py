"""
Configuration loader for vAlgo Trading System
Loads settings from Excel config file and environment variables
"""

import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Union, TYPE_CHECKING

# Type-only imports to avoid circular dependencies
if TYPE_CHECKING:
    import pandas as pd
else:
    try:
        import pandas as pd
    except ImportError:
        pd = None

# Lazy import to avoid circular dependencies
def _get_logger() -> Any:
    """Lazy logger import to avoid circular dependencies"""
    try:
        from utils.logger import get_logger
        return get_logger("config_loader")
    except ImportError:
        import logging
        return logging.getLogger("config_loader")

def _get_constants():
    """Lazy import of constants"""
    try:
        from utils.constants import DEFAULT_CONFIG_FILE, TradingMode
        return DEFAULT_CONFIG_FILE, TradingMode
    except Exception:
        # Fallback values
        class TradingMode:
            BACKTEST = "backtest"
            PAPER = "paper"
            LIVE = "live"
        return "config/config.xlsx", TradingMode

def _get_env_config() -> Dict[str, Any]:
    """Get environment configuration safely"""
    try:
        from utils.env_loader import get_all_config
        config = get_all_config()
        return dict(config) if config else {}
    except ImportError:
        # Fallback to direct environment access
        return {
            'openalgo_url': os.getenv('OPENALGO_URL', 'http://localhost:5000'),
            'database_url': os.getenv('DATABASE_URL', 'data/vAlgo.db'),
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
            'log_file': os.getenv('LOG_FILE', 'logs/vAlgo.log'),
            'log_format': os.getenv('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
        }

def _get_project_root() -> Path:
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

class ConfigLoader:
    """Configuration loader for vAlgo system"""
    
    def __init__(self, config_file: Optional[str] = None) -> None:
        """
        Initialize config loader
        
        Args:
            config_file: Path to Excel config file
        """
        DEFAULT_CONFIG_FILE, _ = _get_constants()
        
        # Make config file path absolute
        if config_file is None:
            config_file = DEFAULT_CONFIG_FILE
        
        if not os.path.isabs(config_file):
            project_root = _get_project_root()
            config_file = str(project_root / config_file)
            
        self.config_file: str = config_file
        self.config_data: Dict[str, Any] = {}
        self.logger = _get_logger()
        
        # Don't auto-load in init to allow graceful handling
        self._loaded: bool = False
    
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
        
        # Clear existing config data if force reloading
        if force_reload:
            self.config_data.clear()
            self._loaded = False
            
        try:
            if not os.path.exists(self.config_file):
                self.logger.error(f"Config file not found: {self.config_file}")
                return False
            
            # Check if pandas and openpyxl are available
            if pd is None:
                try:
                    import pandas as pd_import
                    pd = pd_import
                except ImportError as e:
                    self.logger.error(f"pandas not available: {e}")
                    return False
            
            try:
                import openpyxl
            except ImportError as e:
                self.logger.error(f"openpyxl not available: {e}")
                return False
            
            # Load all sheets from Excel file
            try:
                excel_file = pd.ExcelFile(self.config_file)
            except Exception as e:
                self.logger.error(f"Cannot read Excel file {self.config_file}: {e}")
                return False
            
            loaded_sheets = 0
            for sheet_name in excel_file.sheet_names:
                try:
                    df = pd.read_excel(excel_file, sheet_name=sheet_name)
                    if not df.empty:
                        self.config_data[sheet_name.lower()] = df
                        self.logger.debug(f"Loaded sheet: {sheet_name}")
                        loaded_sheets += 1
                    else:
                        self.logger.warning(f"Sheet {sheet_name} is empty")
                except Exception as e:
                    self.logger.warning(f"Error loading sheet {sheet_name}: {e}")
            
            if loaded_sheets > 0:
                self.logger.info(f"Config loaded from {self.config_file} ({loaded_sheets} sheets)")
                self._loaded = True
                return True
            else:
                self.logger.error("No valid sheets loaded from config file")
                return False
            
        except Exception as e:
            self.logger.error(f"Unexpected error loading config: {e}")
            return False
    
    def _ensure_loaded(self) -> bool:
        """Ensure config is loaded before use"""
        if not self._loaded:
            return self.load_config()
        return True
    
    def force_reload_config(self) -> bool:
        """
        Force reload configuration from Excel file (clears cache)
        
        Returns:
            True if successful, False otherwise
        """
        self.logger.info(f"Force reloading configuration from {self.config_file}")
        return self.load_config(force_reload=True)
    
    def get_initialize_config(self) -> Dict[str, Any]:
        """
        Get initialization configuration
        
        Returns:
            Dictionary with initialization settings
        """
        if not self._ensure_loaded():
            self.logger.warning("Config not loaded, returning empty initialize config")
            return {}
            
        if 'initialize' not in self.config_data:
            self.logger.error("Initialize sheet not found in config")
            return {}
        
        try:
            df = self.config_data['initialize']
            config = {}
            
            # Convert DataFrame to dictionary
            for _, row in df.iterrows():
                try:
                    param = str(row.get('Parameter', '')).lower().strip()
                    value = row.get('Value', '')
                    
                    if param:
                        config[param] = value
                except Exception as e:
                    self.logger.warning(f"Error processing initialize row: {e}")
            
            # Add environment variables (override Excel values)
            env_config = _get_env_config()
            config['openalgo_url'] = env_config.get('openalgo_url', config.get('openalgo_url', 'http://localhost:5000'))
            config['database_url'] = env_config.get('database_url', config.get('database_url', 'data/vAlgo.db'))
            
            return config
            
        except Exception as e:
            self.logger.error(f"Error processing initialize config: {e}")
            return {}
    
    def get_trading_mode(self) -> Union[str, Any]:
        """Get trading mode"""
        _, TradingMode = _get_constants()
        
        init_config = self.get_initialize_config()
        mode_str = str(init_config.get('mode', 'backtest')).lower().strip()
        
        # Handle different TradingMode implementations
        if hasattr(TradingMode, 'BACKTEST'):
            # Enum-style
            if mode_str == 'backtest':
                return TradingMode.BACKTEST
            elif mode_str == 'paper':
                return TradingMode.PAPER
            elif mode_str == 'live':
                return TradingMode.LIVE
            elif mode_str == 'dataload':
                return TradingMode.DATALOAD
            else:
                self.logger.warning(f"Invalid trading mode: {mode_str}, defaulting to backtest")
                return TradingMode.BACKTEST
        else:
            # String-style fallback
            valid_modes = ['backtest', 'paper', 'live', 'dataload']
            if mode_str in valid_modes:
                return mode_str
            else:
                self.logger.warning(f"Invalid trading mode: {mode_str}, defaulting to backtest")
                return 'backtest'
    
    def get_capital(self) -> float:
        """Get initial capital"""
        init_config = self.get_initialize_config()
        capital = init_config.get('capital', 100000)
        
        try:
            return float(capital)
        except (ValueError, TypeError):
            self.logger.warning(f"Invalid capital value: {capital}, using default 100000")
            return 100000.0
    
    def get_active_brokers(self) -> List[Dict[str, Any]]:
        """
        Get active broker configurations
        
        Returns:
            List of active broker configs
        """
        if 'brokers' not in self.config_data:
            self.logger.warning("Brokers sheet not found in config")
            return []
        
        df = self.config_data['brokers']
        active_brokers = []
        
        for _, row in df.iterrows():
            # Safe string conversion for Status field
            status_value = row.get('Status', '')
            status_str = str(status_value).lower().strip() if pd.notna(status_value) else ''
            
            if status_str == 'active':
                broker_config = {
                    'broker': row.get('Broker', ''),
                    'api_key': row.get('API Key', ''),
                    'secret': row.get('Secret', ''),
                    'user_id': row.get('User ID', ''),
                    'openalgo_api_key': row.get('OpenAlgo_API_Key', ''),
                    'openalgo_host': row.get('OpenAlgo_Host', 'http://127.0.0.1:5000'),
                    'openalgo_ws_url': row.get('OpenAlgo_WS_URL', 'ws://127.0.0.1:8765')
                }
                active_brokers.append(broker_config)
        
        self.logger.info(f"Found {len(active_brokers)} active brokers")
        return active_brokers
    
    def get_active_instruments(self) -> List[Dict[str, Any]]:
        """
        Get active instrument configurations
        
        Returns:
            List of active instrument configs
        """
        if 'instruments' not in self.config_data:
            self.logger.error("Instruments sheet not found in config")
            return []
        
        df = self.config_data['instruments']
        active_instruments = []
        
        for _, row in df.iterrows():
            # Safe string conversion for Status field
            status_value = row.get('Status', '')
            if pd.notna(status_value):
                status_str = str(status_value).lower().strip()
            else:
                status_str = ''
            
            if status_str == 'active':
                # Safe string conversion for all fields
                symbol_value = row.get('Symbol', '')
                symbol_str = str(symbol_value).strip() if pd.notna(symbol_value) else ''
                
                timeframe_value = row.get('Timeframe', '1min')
                timeframe_str = str(timeframe_value).strip() if pd.notna(timeframe_value) else '1min'
                
                exchange_value = row.get('Exchange', 'NSE')
                exchange_str = str(exchange_value).strip() if pd.notna(exchange_value) else 'NSE'
                
                if symbol_str:  # Only add if symbol is not empty
                    instrument_config = {
                        'symbol': symbol_str,
                        'timeframe': timeframe_str,
                        'status': status_str,
                        'exchange': exchange_str
                    }
                    active_instruments.append(instrument_config)
        
        self.logger.info(f"Found {len(active_instruments)} active instruments")
        return active_instruments
    
    def get_primary_broker_openalgo_config(self) -> Dict[str, str]:
        """
        Get OpenAlgo configuration for the primary (first active) broker.
        
        Returns:
            Dict with OpenAlgo configuration for primary broker
        """
        active_brokers = self.get_active_brokers()
        
        if not active_brokers:
            self.logger.warning("No active brokers found")
            return {
                'api_key': '',
                'host': 'http://127.0.0.1:5000',
                'ws_url': 'ws://127.0.0.1:8765'
            }
        
        # Use the first active broker as primary
        primary_broker = active_brokers[0]
        
        openalgo_config = {
            'api_key': primary_broker.get('openalgo_api_key', ''),
            'host': primary_broker.get('openalgo_host', 'http://127.0.0.1:5000'),
            'ws_url': primary_broker.get('openalgo_ws_url', 'ws://127.0.0.1:8765'),
            'broker_name': primary_broker.get('broker', '')
        }
        
        self.logger.info(f"Using primary broker: {openalgo_config['broker_name']}")
        return openalgo_config
    
    def get_indicator_config(self) -> List[Dict[str, Any]]:
        """
        Get indicator configurations
        
        Returns:
            List of indicator configs
        """
        if 'indicators' not in self.config_data:
            self.logger.error("Indicators sheet not found in config")
            return []
        
        df = self.config_data['indicators']
        indicator_configs = []
        
        for _, row in df.iterrows():
            # Safe string conversion for Status field
            status_value = row.get('Status', '')
            status_str = str(status_value).lower().strip() if pd.notna(status_value) else ''
            
            if status_str == 'active':
                # Parse parameters with safe handling of float values
                params_value = row.get('Parameters', '')
                parameters = []
                
                # Safe string conversion for parameters
                if pd.notna(params_value) and params_value != '-':
                    params_str = str(params_value).strip()
                    if params_str and params_str != 'nan':
                        try:
                            # Handle comma-separated values
                            if ',' in params_str:
                                # Try to parse as numbers first
                                try:
                                    parameters = [int(float(x.strip())) for x in params_str.split(',')]
                                except (ValueError, TypeError):
                                    # If numeric parsing fails, treat as string parameters
                                    parameters = [x.strip() for x in params_str.split(',')]
                            else:
                                # Single parameter - try numeric first, then string
                                try:
                                    parameters = [int(float(params_str))]
                                except (ValueError, TypeError):
                                    parameters = [params_str]
                        except Exception:
                            self.logger.warning(f"Could not parse parameters for {row.get('Indicator')}: {params_value}")
                            parameters = []
                
                indicator_config = {
                    'indicator': row.get('Indicator', ''),
                    'status': row.get('Status', ''),
                    'parameters': parameters
                }
                indicator_configs.append(indicator_config)
        
        self.logger.info(f"Found {len(indicator_configs)} active indicators")
        return indicator_configs
    
    def get_trading_rules(self) -> List[Dict[str, Any]]:
        """
        Get trading rule configurations
        
        Returns:
            List of trading rules
        """
        if 'rules' not in self.config_data:
            self.logger.error("Rules sheet not found in config")
            return []
        
        df = self.config_data['rules']
        trading_rules = []
        
        for _, row in df.iterrows():
            rule_config = {
                'rule_id': row.get('Rule ID', ''),
                'type': row.get('Type', ''),
                'condition': row.get('Condition', '')
            }
            trading_rules.append(rule_config)
        
        self.logger.info(f"Found {len(trading_rules)} trading rules")
        return trading_rules
    
    def get_environment_variables(self) -> Dict[str, Union[str, bool]]:
        """
        Get all relevant environment variables
        
        Returns:
            Dictionary of environment variables
        """
        env_vars = {
            'openalgo_url': os.getenv('OPENALGO_URL', 'http://localhost:5000') or 'http://localhost:5000',
            'openalgo_api_key': os.getenv('OPENALGO_API_KEY', '') or '',
            'database_url': os.getenv('DATABASE_URL', 'data/vAlgo.db') or 'data/vAlgo.db',
            'log_level': os.getenv('LOG_LEVEL', 'INFO') or 'INFO',
            'log_file': os.getenv('LOG_FILE', 'logs/vAlgo.log') or 'logs/vAlgo.log',
            'paper_trading': (os.getenv('PAPER_TRADING', 'true') or 'true').lower() == 'true'
        }
        
        return env_vars
    
    def validate_config(self) -> List[str]:
        """
        Validate configuration and return list of issues
        
        Returns:
            List of validation errors/warnings
        """
        issues = []
        
        # Check required sheets
        required_sheets = ['initialize', 'instruments', 'indicators', 'rules']
        for sheet in required_sheets:
            if sheet not in self.config_data:
                issues.append(f"Missing required sheet: {sheet}")
        
        # Validate instruments
        instruments = self.get_active_instruments()
        if not instruments:
            issues.append("No active instruments found")
        
        # Validate indicators
        indicators = self.get_indicator_config()
        if not indicators:
            issues.append("No active indicators found")
        
        # Validate rules
        rules = self.get_trading_rules()
        if not rules:
            issues.append("No trading rules found")
        
        # Check environment variables
        env_vars = self.get_environment_variables()
        if not env_vars['openalgo_url']:
            issues.append("OPENALGO_URL not configured")
        
        return issues
    
    def get_entry_conditions_enhanced(self) -> Dict[str, Any]:
        """
        Get entry conditions with multiple row support and Condition_Order
        
        Returns:
            Dictionary with parsed entry rules
        """
        if not self._ensure_loaded():
            self.logger.warning("Config not loaded, returning empty entry conditions")
            return {}
            
        if 'entry_conditions' not in self.config_data:
            self.logger.error("Entry_Conditions sheet not found in config")
            return {}
        
        try:
            df = self.config_data['entry_conditions']
            
            # Forward fill Rule_Name to handle blank cells
            df = df.copy()
            df['Rule_Name_Filled'] = df['Rule_Name'].ffill()
            df['Status_Filled'] = df['Status'].ffill()
            
            parsed_rules = {}
            
            # Group by filled Rule_Name
            for rule_name, rule_group in df.groupby('Rule_Name_Filled'):
                if pd.isna(rule_name):
                    continue
                    
                # Get rule status from first non-null status
                rule_status = rule_group['Status_Filled'].iloc[0]
                if pd.isna(rule_status):
                    rule_status = 'Inactive'
                
                # Process conditions by Condition_Order
                condition_groups = []
                
                if 'Condition_Order' in rule_group.columns:
                    # Group by Condition_Order for OR logic support
                    for order, order_group in rule_group.groupby('Condition_Order'):
                        if pd.isna(order):
                            continue
                            
                        conditions = []
                        logic_ops = []
                        
                        for _, row in order_group.iterrows():
                            # Extract condition components
                            indicator = row.get('Indicator_1', '')
                            operator = row.get('Operator_1', '')
                            value = row.get('Value_1', '')
                            logic = row.get('Logic', '')
                            
                            if pd.notna(indicator) and pd.notna(operator) and pd.notna(value):
                                conditions.append({
                                    'indicator': str(indicator),
                                    'operator': str(operator),
                                    'value': str(value)
                                })
                                logic_ops.append(str(logic) if pd.notna(logic) else '')
                        
                        if conditions:
                            condition_groups.append({
                                'order': int(order),
                                'conditions': conditions,
                                'logic_to_next': logic_ops[0] if logic_ops else ''
                            })
                else:
                    # Fallback: treat each row as separate condition
                    for i, (_, row) in enumerate(rule_group.iterrows()):
                        indicator = row.get('Indicator_1', '')
                        operator = row.get('Operator_1', '')
                        value = row.get('Value_1', '')
                        logic = row.get('Logic', '')
                        
                        if pd.notna(indicator) and pd.notna(operator) and pd.notna(value):
                            condition_groups.append({
                                'order': i + 1,
                                'conditions': [{
                                    'indicator': str(indicator),
                                    'operator': str(operator),
                                    'value': str(value)
                                }],
                                'logic_to_next': str(logic) if pd.notna(logic) else ''
                            })
                
                # Sort condition groups by order
                condition_groups.sort(key=lambda x: x['order'])
                
                parsed_rules[rule_name] = {
                    'status': rule_status,
                    'condition_groups': condition_groups,
                    'total_conditions': sum(len(group['conditions']) for group in condition_groups)
                }
            
            self.logger.info(f"Parsed {len(parsed_rules)} entry rules with enhanced format")
            return parsed_rules
            
        except Exception as e:
            self.logger.error(f"Error parsing entry conditions: {e}")
            return {}
    
    def get_exit_conditions_enhanced(self) -> Dict[str, Any]:
        """
        Get exit conditions with multiple row support
        
        Returns:
            Dictionary with parsed exit rules
        """
        if not self._ensure_loaded():
            self.logger.warning("Config not loaded, returning empty exit conditions")
            return {}
            
        if 'exit_conditions' not in self.config_data:
            self.logger.error("Exit_Conditions sheet not found in config")
            return {}
        
        try:
            df = self.config_data['exit_conditions']
            
            # Forward fill Rule_Name and other fields
            df = df.copy()
            df['Rule_Name_Filled'] = df['Rule_Name'].ffill()
            df['Exit_Type_Filled'] = df['Exit_Type'].ffill()
            df['Status_Filled'] = df['Status'].ffill()
            
            parsed_exits = {}
            
            for rule_name, rule_group in df.groupby('Rule_Name_Filled'):
                if pd.isna(rule_name):
                    continue
                    
                exit_type = rule_group['Exit_Type_Filled'].iloc[0]
                status = rule_group['Status_Filled'].iloc[0]
                
                conditions = []
                for _, row in rule_group.iterrows():
                    indicator = row.get('Indicator', '')
                    operator = row.get('Operator', '')
                    value = row.get('Value', '')
                    
                    if pd.notna(indicator) and pd.notna(operator) and pd.notna(value):
                        conditions.append({
                            'indicator': str(indicator),
                            'operator': str(operator),
                            'value': str(value)
                        })
                
                parsed_exits[rule_name] = {
                    'exit_type': str(exit_type) if pd.notna(exit_type) else 'Signal',
                    'status': str(status) if pd.notna(status) else 'Inactive',
                    'conditions': conditions
                }
            
            self.logger.info(f"Parsed {len(parsed_exits)} exit rules with enhanced format")
            return parsed_exits
            
        except Exception as e:
            self.logger.error(f"Error parsing exit conditions: {e}")
            return {}
    
    def get_strategy_configs_enhanced(self) -> Dict[str, Any]:
        """
        Get strategy configurations linking entry and exit rules
        
        Returns:
            Dictionary with strategy configurations
        """
        if not self._ensure_loaded():
            self.logger.warning("Config not loaded, returning empty strategy configs")
            return {}
            
        if 'strategy_config' not in self.config_data:
            self.logger.error("Strategy_Config sheet not found in config")
            return {}
        
        try:
            df = self.config_data['strategy_config']
            
            # Forward fill Strategy_Name to handle blank cells in vertical format
            df = df.copy()
            df['Strategy_Name_Filled'] = df['Strategy_Name'].ffill()
            
            strategies = {}
            
            # Group by filled Strategy_Name to handle vertical exit rules
            for strategy_name, strategy_group in df.groupby('Strategy_Name_Filled'):
                if pd.isna(strategy_name) or strategy_name == '':
                    continue
                
                # Get strategy parameters from first row
                first_row = strategy_group.iloc[0]
                entry_rule = str(first_row.get('Entry_Rule', '')) if pd.notna(first_row.get('Entry_Rule')) else ''
                position_size = float(first_row.get('Position_Size', 1000)) if pd.notna(first_row.get('Position_Size')) else 1000
                risk_per_trade = float(first_row.get('Risk_Per_Trade', 0.02)) if pd.notna(first_row.get('Risk_Per_Trade')) else 0.02
                max_positions = int(first_row.get('Max_Positions', 1)) if pd.notna(first_row.get('Max_Positions')) else 1
                status = str(first_row.get('Status', 'Inactive')) if pd.notna(first_row.get('Status')) else 'Inactive'
                
                # Collect exit rules from all rows (both old multi-column and new single-column format)
                exit_rules = []
                
                for _, row in strategy_group.iterrows():
                    # Check new single Exit_Rule column format
                    exit_rule = row.get('Exit_Rule', '')
                    if pd.notna(exit_rule) and exit_rule != '':
                        exit_rules.append(str(exit_rule))
                    
                    # Fallback: Check old multi-column format for backward compatibility
                    if not exit_rules:  # Only check old format if no new format found
                        for i in range(1, 4):
                            old_exit_rule = row.get(f'Exit_Rule_{i}', '')
                            if pd.notna(old_exit_rule) and old_exit_rule != '':
                                exit_rules.append(str(old_exit_rule))
                
                # Remove duplicates while preserving order
                exit_rules = list(dict.fromkeys(exit_rules))
                
                strategies[strategy_name] = {
                    'entry_rule': entry_rule,
                    'exit_rules': exit_rules,
                    'position_size': position_size,
                    'risk_per_trade': risk_per_trade,
                    'max_positions': max_positions,
                    'status': status
                }
            
            self.logger.info(f"Parsed {len(strategies)} strategy configurations")
            return strategies
            
        except Exception as e:
            self.logger.error(f"Error parsing strategy configs: {e}")
            return {}
    
    def validate_enhanced_config(self) -> List[str]:
        """
        Validate the enhanced configuration structure
        
        Returns:
            List of validation issues
        """
        issues = []
        
        # Check required sheets for enhanced format
        required_sheets = ['entry_conditions', 'exit_conditions', 'strategy_config']
        for sheet in required_sheets:
            if sheet not in self.config_data:
                issues.append(f"Missing required sheet: {sheet}")
        
        # Validate entry conditions
        try:
            entry_conditions = self.get_entry_conditions_enhanced()
            if not entry_conditions:
                issues.append("No valid entry conditions found")
            else:
                for rule_name, rule_data in entry_conditions.items():
                    if not rule_data['condition_groups']:
                        issues.append(f"Entry rule '{rule_name}' has no conditions")
        except Exception as e:
            issues.append(f"Error validating entry conditions: {e}")
        
        # Validate exit conditions
        try:
            exit_conditions = self.get_exit_conditions_enhanced()
            if not exit_conditions:
                issues.append("No valid exit conditions found")
            else:
                for rule_name, rule_data in exit_conditions.items():
                    if not rule_data['conditions']:
                        issues.append(f"Exit rule '{rule_name}' has no conditions")
        except Exception as e:
            issues.append(f"Error validating exit conditions: {e}")
        
        # Validate strategy configs
        try:
            strategies = self.get_strategy_configs_enhanced()
            if not strategies:
                issues.append("No valid strategies found")
            else:
                for strategy_name, strategy_data in strategies.items():
                    if not strategy_data['entry_rule']:
                        issues.append(f"Strategy '{strategy_name}' has no entry rule")
                    if not strategy_data['exit_rules']:
                        issues.append(f"Strategy '{strategy_name}' has no exit rules")
        except Exception as e:
            issues.append(f"Error validating strategies: {e}")
        
        return issues
    
    def build_rule_expression(self, condition_groups: List[Dict]) -> str:
        """
        Build a readable expression from condition groups
        
        Args:
            condition_groups: List of condition groups with order and logic
            
        Returns:
            Human-readable rule expression
        """
        if not condition_groups:
            return "No conditions"
        
        expression_parts = []
        
        for group in condition_groups:
            conditions = group['conditions']
            logic_to_next = group.get('logic_to_next', '')
            
            if len(conditions) == 1:
                # Single condition
                cond = conditions[0]
                part = f"{cond['indicator']} {cond['operator']} {cond['value']}"
            else:
                # Multiple conditions in same group (typically OR)
                cond_strs = []
                for cond in conditions:
                    cond_strs.append(f"{cond['indicator']} {cond['operator']} {cond['value']}")
                part = "(" + " OR ".join(cond_strs) + ")"
            
            expression_parts.append(part)
        
        # Join parts based on logic operators
        if len(expression_parts) == 1:
            return expression_parts[0]
        else:
            # Default to AND between groups
            return " AND ".join(expression_parts)

# Convenience function
def load_config(config_file: Optional[str] = None) -> ConfigLoader:
    """
    Load configuration
    
    Args:
        config_file: Path to config file
        
    Returns:
        ConfigLoader instance
    """
    return ConfigLoader(config_file)