"""
JSON Configuration Validator for Ultimate Efficiency System
==========================================================

Provides strict validation for all JSON configuration files with detailed error messages.
Ensures configuration integrity and prevents runtime errors.

Author: vAlgo Development Team
Created: July 29, 2025
"""

import json
import re
from typing import Dict, Any, List, Optional
from pathlib import Path
from utils.selective_logger import get_selective_logger, reset_selective_logger


class ConfigurationError(Exception):
    """Raised when configuration validation fails"""
    pass


class ConfigValidator:
    """
    Strict JSON configuration validator with comprehensive error checking.
    Zero tolerance for missing or invalid configurations.
    """
    
    def __init__(self):
        """Initialize validator with required schemas"""
        self.required_system_keys = {
            'version', 'environment', 'logging_level', 'performance_target', 'architecture', 'logger_config'
        }
        self.required_trading_keys = {
            'symbol', 'exchange', 'timeframe', 'lot_size', 'initial_capital'
        }
        self.required_database_keys = {
            'path', 'type', 'required_tables', 'validate_on_startup'
        }
        self.required_options_keys = {
            'strike_type', 'default_option_type', 'expiry_selection'
        }
        self.required_indicators_keys = {
            'allow_numpy_fallback', 'require_complete_calculation'
        }
        self.required_validation_keys = {
            'strict_mode', 'fail_fast', 'require_all_config_parameters'
        }
        self.required_strategy_case_keys = {
            'status', 'priority', 'description', 'entry', 'exit', 'OptionType', 'StrikeType'
        }
        
        # Valid values for enumerated fields
        self.valid_option_types = {'CALL', 'PUT'}
        self.valid_strike_types = {'ATM', 'ITM', 'OTM'}
        self.valid_statuses = {'Active', 'InActive'}
        self.valid_environments = {'production', 'development', 'testing'}
        
    def validate_main_config(self, config_path: str) -> Dict[str, Any]:
        """
        Validate main configuration file with strict requirements.
        
        Args:
            config_path: Path to config.json file
            
        Returns:
            Validated configuration dictionary
            
        Raises:
            ConfigurationError: If validation fails with detailed message
        """
        try:
            config = self._load_json_file(config_path)
            
            # Validate required top-level sections
            required_sections = {'system', 'trading', 'database', 'options', 'backtesting', 'indicators', 'validation', 'performance'}
            self._validate_required_keys(config, required_sections, "main configuration")
            
            # Validate each section
            self._validate_system_config(config['system'])
            self._validate_trading_config(config['trading'])
            self._validate_database_config(config['database'])
            self._validate_options_config(config['options'])
            self._validate_indicators_config(config['indicators'])
            self._validate_validation_config(config['validation'])
            self._validate_performance_config(config['performance'])
            
            print("✅ Main configuration validation passed")
            return config
            
        except Exception as e:
            raise ConfigurationError(f"Main configuration validation failed: {e}")
    
    def validate_strategies_config(self, strategies_path: str) -> Dict[str, Any]:
        """
        Validate hierarchical strategies configuration.
        
        Args:
            strategies_path: Path to strategies.json file
            
        Returns:
            Validated strategies dictionary
            
        Raises:
            ConfigurationError: If validation fails with detailed message
        """
        try:
            strategies = self._load_json_file(strategies_path)
            
            if not strategies:
                raise ConfigurationError("Strategies configuration is empty")
            
            # Validate each strategy group
            for group_name, group_config in strategies.items():
                self._validate_strategy_group(group_name, group_config)
            
            print(f"✅ Strategies configuration validation passed ({len(strategies)} groups)")
            return strategies
            
        except Exception as e:
            raise ConfigurationError(f"Strategies configuration validation failed: {e}")
    
    def _validate_strategy_group(self, group_name: str, group_config: Dict[str, Any]):
        """Validate individual strategy group configuration"""
        required_group_keys = {'status', 'description', 'max_concurrent_cases', 'cases'}
        self._validate_required_keys(group_config, required_group_keys, f"strategy group '{group_name}'")
        
        # Validate group status
        if group_config['status'] not in self.valid_statuses:
            raise ConfigurationError(
                f"Strategy group '{group_name}' has invalid status '{group_config['status']}'. "
                f"Must be one of: {self.valid_statuses}"
            )
        
        # Validate max_concurrent_cases
        if not isinstance(group_config['max_concurrent_cases'], int) or group_config['max_concurrent_cases'] < 1:
            raise ConfigurationError(
                f"Strategy group '{group_name}' max_concurrent_cases must be positive integer"
            )
        
        # Validate cases
        cases = group_config['cases']
        if not cases:
            raise ConfigurationError(f"Strategy group '{group_name}' has no cases defined")
        
        for case_name, case_config in cases.items():
            self._validate_strategy_case(group_name, case_name, case_config)
    
    def _validate_strategy_case(self, group_name: str, case_name: str, case_config: Dict[str, Any]):
        """Validate individual strategy case configuration"""
        self._validate_required_keys(
            case_config, 
            self.required_strategy_case_keys, 
            f"case '{case_name}' in group '{group_name}'"
        )
        
        # Validate status
        if case_config['status'] not in self.valid_statuses:
            raise ConfigurationError(
                f"Case '{case_name}' in group '{group_name}' has invalid status. "
                f"Must be one of: {self.valid_statuses}"
            )
        
        # Validate priority
        if not isinstance(case_config['priority'], int) or case_config['priority'] < 1:
            raise ConfigurationError(
                f"Case '{case_name}' priority must be positive integer"
            )
        
        # Validate OptionType
        if case_config['OptionType'] not in self.valid_option_types:
            raise ConfigurationError(
                f"Case '{case_name}' has invalid OptionType '{case_config['OptionType']}'. "
                f"Must be one of: {self.valid_option_types}"
            )
        
        # Validate StrikeType
        if case_config['StrikeType'] not in self.valid_strike_types:
            raise ConfigurationError(
                f"Case '{case_name}' has invalid StrikeType '{case_config['StrikeType']}'. "
                f"Must be one of: {self.valid_strike_types}"
            )
        
        # Validate entry and exit expressions
        self._validate_expression(case_config['entry'], f"entry condition for case '{case_name}'")
        self._validate_expression(case_config['exit'], f"exit condition for case '{case_name}'")
    
    def _validate_expression(self, expression: str, context: str):
        """Validate trading condition expressions with smart pattern matching"""
        if not isinstance(expression, str) or not expression.strip():
            raise ConfigurationError(f"Invalid {context}: expression cannot be empty")
        
        # Check for dangerous function calls using regex patterns
        # Look for function calls like 'open(', 'exec(', etc. rather than just substrings
        dangerous_function_patterns = [
            r'\bimport\s+',          # import statements
            r'\bexec\s*\(',          # exec() function calls
            r'\beval\s*\(',          # eval() function calls
            r'__\w+__',              # dunder methods
            r'\bopen\s*\(',          # open() function calls
            r'\bfile\s*\(',          # file() function calls
        ]
        
        # Allowed trading indicators that contain potentially dangerous words
        allowed_trading_patterns = [
            r'\bprevious_candle_open\b',    # OHLC previous candle data
            r'\bprevious_day_open\b',       # Previous day open price
            r'\bltp_open\b',                # Live trading price open
            r'\bopen\b(?!\s*\()',          # 'open' as variable/column name (not function call)
        ]
        
        expression_lower = expression.lower()
        
        # First check if expression contains allowed trading patterns
        for allowed_pattern in allowed_trading_patterns:
            if re.search(allowed_pattern, expression_lower):
                # If found allowed pattern, remove it temporarily for danger checking
                temp_expression = re.sub(allowed_pattern, '', expression_lower)
                # Continue with danger checking on cleaned expression
                for danger_pattern in dangerous_function_patterns:
                    if re.search(danger_pattern, temp_expression):
                        raise ConfigurationError(
                            f"Dangerous pattern '{danger_pattern}' found in {context}: {expression}"
                        )
                # If we reach here, the expression is safe - skip remaining danger checks
                break
        else:
            # No allowed patterns found, check for dangerous patterns directly
            for danger_pattern in dangerous_function_patterns:
                if re.search(danger_pattern, expression_lower):
                    # Extract the matched pattern for clearer error message
                    match = re.search(danger_pattern, expression_lower)
                    matched_text = match.group() if match else danger_pattern
                    raise ConfigurationError(
                        f"Dangerous pattern '{matched_text.strip()}' found in {context}: {expression}"
                    )
        
        # Basic syntax validation (parentheses matching)
        if expression.count('(') != expression.count(')'):
            raise ConfigurationError(f"Unmatched parentheses in {context}: {expression}")
    
    def _validate_system_config(self, system_config: Dict[str, Any]):
        """Validate system configuration section"""
        self._validate_required_keys(system_config, self.required_system_keys, "system configuration")
        
        if system_config['environment'] not in self.valid_environments:
            raise ConfigurationError(
                f"Invalid environment '{system_config['environment']}'. "
                f"Must be one of: {self.valid_environments}"
            )
        
        # Validate logger_config
        logger_config = system_config.get('logger_config', {})
        if not isinstance(logger_config, dict):
            raise ConfigurationError("System logger_config must be a dictionary")
        
        required_logger_keys = {'enabled', 'session_name', 'log_level'}
        if not required_logger_keys.issubset(logger_config.keys()):
            missing_keys = required_logger_keys - set(logger_config.keys())
            raise ConfigurationError(f"Missing required logger_config keys: {missing_keys}")
    
    def _validate_database_config(self, database_config: Dict[str, Any]):
        """Validate database configuration section"""
        self._validate_required_keys(database_config, self.required_database_keys, "database configuration")
        
        # Validate database path
        if not database_config['path']:
            raise ConfigurationError("Database path cannot be empty")
        
        # Validate required tables
        required_tables = database_config.get('required_tables', [])
        if not required_tables:
            raise ConfigurationError("Database must specify required_tables")
        
        if not isinstance(required_tables, list):
            raise ConfigurationError("Database required_tables must be a list")
    
    def _validate_validation_config(self, validation_config: Dict[str, Any]):
        """Validate validation configuration section"""
        self._validate_required_keys(validation_config, self.required_validation_keys, "validation configuration")
        
        # Validate boolean fields
        for key in self.required_validation_keys:
            if not isinstance(validation_config[key], bool):
                raise ConfigurationError(f"Validation {key} must be boolean")
    
    def _validate_trading_config(self, trading_config: Dict[str, Any]):
        """Validate trading configuration section"""
        self._validate_required_keys(trading_config, self.required_trading_keys, "trading configuration")
        
        # Validate numeric fields
        if trading_config['lot_size'] <= 0:
            raise ConfigurationError("Trading lot_size must be positive")
        
        if trading_config['initial_capital'] <= 0:
            raise ConfigurationError("Trading initial_capital must be positive")
    
    def _validate_options_config(self, options_config: Dict[str, Any]):
        """Validate options configuration section"""
        self._validate_required_keys(options_config, self.required_options_keys, "options configuration")
        
        if options_config['default_option_type'] not in self.valid_option_types:
            raise ConfigurationError(
                f"Invalid default_option_type '{options_config['default_option_type']}'. "
                f"Must be one of: {self.valid_option_types}"
            )
    
    def _validate_indicators_config(self, indicators_config: Dict[str, Any]):
        """Validate indicators configuration section"""
        if not indicators_config:
            raise ConfigurationError("Indicators configuration cannot be empty")
        
        # Validate required indicators configuration keys
        self._validate_required_keys(indicators_config, self.required_indicators_keys, "indicators configuration")
        
        # Validate boolean configuration fields
        if not isinstance(indicators_config['allow_numpy_fallback'], bool):
            raise ConfigurationError("Indicators allow_numpy_fallback must be boolean")
        
        if not isinstance(indicators_config['require_complete_calculation'], bool):
            raise ConfigurationError("Indicators require_complete_calculation must be boolean")
        
        # Ensure at least one indicator is enabled
        enabled_indicators = [name for name, config in indicators_config.items() 
                            if isinstance(config, dict) and config.get('enabled', False)]
        
        if not enabled_indicators:
            raise ConfigurationError("At least one indicator must be enabled")
    
    def _validate_performance_config(self, performance_config: Dict[str, Any]):
        """Validate performance configuration section"""
        required_performance_keys = {
            'enable_smart_indicators', 'enable_vectorized_state_management', 
            'enable_memory_optimization', 'target_processing_speed'
        }
        self._validate_required_keys(performance_config, required_performance_keys, "performance configuration")
        
        if performance_config['target_processing_speed'] <= 0:
            raise ConfigurationError("Target processing speed must be positive")
    
    def _validate_required_keys(self, config: Dict[str, Any], required_keys: set, context: str):
        """Validate that all required keys are present"""
        missing_keys = required_keys - set(config.keys())
        if missing_keys:
            raise ConfigurationError(
                f"Missing required keys in {context}: {missing_keys}. "
                f"Required keys: {required_keys}"
            )
    
    def _load_json_file(self, file_path: str) -> Dict[str, Any]:
        """Load and parse JSON file with error handling"""
        try:
            path = Path(file_path)
            if not path.exists():
                raise ConfigurationError(f"Configuration file not found: {file_path}")
            
            with open(path, 'r') as f:
                return json.load(f)
                
        except json.JSONDecodeError as e:
            raise ConfigurationError(f"Invalid JSON in {file_path}: {e}")
        except Exception as e:
            raise ConfigurationError(f"Error loading {file_path}: {e}")


def validate_all_configs(config_dir: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Validate all configuration files in the specified directory.
    
    Args:
        config_dir: Directory containing configuration files
        
    Returns:
        Tuple of (main_config, strategies_config)
        
    Raises:
        ConfigurationError: If any validation fails
    """
    validator = ConfigValidator()
    
    config_path = Path(config_dir) / "config.json"
    strategies_path = Path(config_dir) / "strategies.json"
    
    print("🔍 Validating configuration files...")
    
    main_config = validator.validate_main_config(str(config_path))
    strategies_config = validator.validate_strategies_config(str(strategies_path))
    
    print("✅ All configuration files validated successfully")
    
    return main_config, strategies_config


if __name__ == "__main__":
    # Test validation with current directory
    import sys
    import os
    
    current_dir = os.path.dirname(__file__)
    try:
        main_config, strategies_config = validate_all_configs(current_dir)
        print("🎉 Configuration validation test passed!")
        print(f"   📊 Strategies: {len(strategies_config)} groups")
        print(f"   🔧 Indicators: {len([k for k, v in main_config['indicators'].items() if v.get('enabled')])}")
    except ConfigurationError as e:
        print(f"❌ Configuration validation failed: {e}")
        sys.exit(1)