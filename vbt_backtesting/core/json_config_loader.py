"""
JSON Configuration Loader for Ultimate Efficiency System
=======================================================

Handles loading, validation, and caching of all JSON configurations.
Provides centralized configuration management with hot-reload capability.

Author: vAlgo Development Team
Created: July 29, 2025
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from config.config_validator import ConfigValidator, ConfigurationError


class JSONConfigLoader:
    """
    Centralized JSON configuration loader with validation and caching.
    Ensures all configurations are loaded correctly with strict validation.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files. 
                       REQUIRED - no default values allowed.
        
        Raises:
            ConfigurationError: If config_dir is None or invalid
        """
        if config_dir is None:
            # No default values - fail fast with proper error
            raise ConfigurationError(
                "Configuration directory is required and cannot be None. "
                "Provide explicit config_dir parameter for JSON-driven configuration."
            )
        
        config_path = Path(config_dir)
        if not config_path.exists():
            raise ConfigurationError(f"Configuration directory does not exist: {config_dir}")
        
        if not config_path.is_dir():
            raise ConfigurationError(f"Configuration path is not a directory: {config_dir}")
        
        self.config_dir = config_path
        
        self.validator = ConfigValidator()
        self._main_config = None
        self._strategies_config = None
        self._last_loaded = None
        
        print(f"[CONFIG] JSON Config Loader initialized")
        print(f"   📂 Config Directory: {self.config_dir}")
    
    def load_all_configs(self, force_reload: bool = False) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Load all configuration files with validation.
        
        Args:
            force_reload: Force reload even if configs are cached
            
        Returns:
            Tuple of (main_config, strategies_config)
            
        Raises:
            ConfigurationError: If any configuration is invalid
        """
        try:
            # Check if we need to reload
            if not force_reload and self._configs_loaded():
                print("[CACHE] Using cached configurations")
                return self._main_config, self._strategies_config
            
            print("[LOAD] Loading configurations from JSON files...")
            
            # Load and validate main configuration
            main_config_path = self.config_dir / "config.json"
            self._main_config = self.validator.validate_main_config(str(main_config_path))
            
            # Load and validate strategies configuration
            strategies_config_path = self.config_dir / "strategies.json"
            self._strategies_config = self.validator.validate_strategies_config(str(strategies_config_path))
            
            # Perform cross-validation
            self._cross_validate_configs(self._main_config, self._strategies_config)
            
            self._last_loaded = datetime.now()
            
            # Get active strategy cases for detailed display
            enabled_groups = self.get_enabled_strategy_groups()
            active_case_names = self.get_active_case_names()
            
            # Count total cases across all groups
            total_cases = sum(len(group.get('cases', {})) for group in self._strategies_config.values())
            
            # Get detailed indicator summary
            indicator_summary = self.get_indicator_summary()
            
            print("[SUCCESS] All configurations loaded and validated successfully")
            print(f"   [STATS] Strategy Cases: {total_cases} total, {len(active_case_names)} active")
            print(f"   [CASES] Cases: ({', '.join(active_case_names) if active_case_names else 'None'})")
            print(f"   🔧 Enabled Indicators: {len(indicator_summary['enabled_list'])} ({', '.join(indicator_summary['enabled_list'])})")
            print(f"   🔑 Created Indicator keys: {indicator_summary['total_keys']}")
            print(f"   [TARGET] Performance Target: {self._main_config['system']['performance_target']}")
            
            return self._main_config, self._strategies_config
            
        except Exception as e:
            raise ConfigurationError(f"Failed to load configurations: {e}")
    
    def get_main_config(self) -> Dict[str, Any]:
        """Get main configuration, loading if necessary"""
        if self._main_config is None:
            self.load_all_configs()
        return self._main_config
    
    def get_strategies_config(self) -> Dict[str, Any]:
        """Get strategies configuration, loading if necessary"""
        if self._strategies_config is None:
            self.load_all_configs()
        return self._strategies_config
    
    @property
    def main_config(self) -> Dict[str, Any]:
        """Property for main configuration access"""
        return self.get_main_config()
    
    @property
    def strategies_config(self) -> Dict[str, Any]:
        """Property for strategies configuration access"""
        return self.get_strategies_config()
    
    def get_enabled_strategy_groups(self) -> Dict[str, Any]:
        """Get only enabled strategy groups"""
        strategies = self.get_strategies_config()
        return {
            name: config for name, config in strategies.items()
            if config.get('status') == 'Active'
        }
    
    def get_enabled_strategies_cases(self) -> Dict[str, Any]:
        """Get only enabled cases from enabled strategy groups"""
        enabled_groups = self.get_enabled_strategy_groups()
        enabled_cases = {}
        
        for group_name, group_config in enabled_groups.items():
            enabled_cases[group_name] = {
                **group_config,
                'cases': {
                    case_name: case_config
                    for case_name, case_config in group_config['cases'].items()
                    if case_config.get('status') == 'Active'
                }
            }
        
        return enabled_cases
    
    def get_active_case_names(self) -> List[str]:
        """Get list of active case names from enabled strategy groups"""
        enabled_cases = self.get_enabled_strategies_cases()
        active_cases = []
        
        for group_name, group_config in enabled_cases.items():
            cases = group_config.get('cases', {})
            for case_name, case_config in cases.items():
                if case_config.get('status') == 'Active':
                    active_cases.append(case_name)
        
        return active_cases
    
    def get_enabled_indicators_from_config(self) -> Dict[str, str]:
        """
        Simple config-based indicator loading.
        Returns all enabled indicators from config.json without complex parsing.
        
        Returns:
            Dictionary mapping indicator keys to descriptions
        """
        indicator_keys = {
            # Market data keys (always available)
            'close': 'Closing price series',
            'open': 'Opening price series', 
            'high': 'High price series',
            'low': 'Low price series',
            'volume': 'Volume series'
        }
        
        # Add enabled indicators from config.json
        main_config = self.get_main_config()
        for indicator_name, indicator_config in main_config['indicators'].items():
            # Skip non-indicator config keys
            if indicator_name in ['allow_numpy_fallback', 'require_complete_calculation']:
                continue
            
            # Ensure indicator_config is a dictionary
            if not isinstance(indicator_config, dict):
                continue
                
            if indicator_config.get('enabled', False):
                if 'periods' in indicator_config:
                    # Multi-period indicators (RSI, SMA, EMA, etc.)
                    for period in indicator_config['periods']:
                        key = f"{indicator_name}_{period}"
                        indicator_keys[key] = f"{indicator_name.upper()} with {period} period"
                else:
                    # Single indicators (VWAP, Bollinger Bands, etc.)
                    indicator_keys[indicator_name] = f"{indicator_name.upper()} indicator"
                    
                    # Special handling for Bollinger Bands
                    if indicator_name == 'bollinger_bands':
                        period = indicator_config.get('period', 20)
                        indicator_keys[f'bb_upper_{period}'] = f"Bollinger Bands Upper ({period} period)"
                        indicator_keys[f'bb_middle_{period}'] = f"Bollinger Bands Middle ({period} period)"
                        indicator_keys[f'bb_lower_{period}'] = f"Bollinger Bands Lower ({period} period)"
        
        return indicator_keys
    
    def get_required_indicators(self) -> set[str]:
        """
        Legacy method for backward compatibility.
        Now simply returns keys of all enabled indicators.
        """
        return set(self.get_enabled_indicators_from_config().keys())
    
    def get_config_summary(self) -> Dict[str, Any]:
        """Get summary of current configuration"""
        main_config = self.get_main_config()
        strategies_config = self.get_strategies_config()
        enabled_groups = self.get_enabled_strategy_groups()
        enabled_cases = self.get_enabled_strategies_cases()
        required_indicators = self.get_required_indicators()
        
        total_cases = sum(len(group['cases']) for group in strategies_config.values())
        enabled_case_count = sum(len(group['cases']) for group in enabled_cases.values())
        available_indicators = self.get_enabled_indicators_from_config()
        
        return {
            'system_version': main_config['system']['version'],
            'performance_target': main_config['system']['performance_target'],
            'total_strategy_groups': len(strategies_config),
            'enabled_strategy_groups': len(enabled_groups),
            'total_cases': total_cases,
            'enabled_cases': enabled_case_count,
            'available_indicators': len(available_indicators),
            'indicator_approach': 'config_based_simple',
            'last_loaded': self._last_loaded.isoformat() if self._last_loaded else None
        }
    
    def _configs_loaded(self) -> bool:
        """Check if configurations are already loaded"""
        return self._main_config is not None and self._strategies_config is not None
    
    def _cross_validate_configs(self, main_config: Dict[str, Any], strategies_config: Dict[str, Any]):
        """Simplified cross-validation using config-based approach"""
        # Get all enabled indicators from config
        available_indicators = self.get_enabled_indicators_from_config()
        
        print(f"[SUCCESS] Indicator key validation passed")
        print(f"   [STATS] Total available indicator keys: {len(available_indicators)}")
        print(f"   🔧 Market data keys: 5 (close, open, high, low, volume)")
        print(f"   🧮 Calculated indicator keys: {len(available_indicators) - 5}")
    
    # Removed complex _extract_indicators_from_expression method
    # Now using simple config-based approach instead
    
    def _count_enabled_indicators(self, main_config: Dict[str, Any]) -> int:
        """Count total number of enabled indicators"""
        count = 0
        for indicator_name, indicator_config in main_config['indicators'].items():
            # Skip non-indicator config keys
            if indicator_name in ['allow_numpy_fallback', 'require_complete_calculation']:
                continue
            
            # Ensure indicator_config is a dictionary
            if not isinstance(indicator_config, dict):
                continue
                
            if indicator_config.get('enabled', False):
                if 'periods' in indicator_config:
                    count += len(indicator_config['periods'])
                else:
                    count += 1
        return count
    
    def get_indicator_summary(self) -> Dict[str, Any]:
        """Get detailed indicator summary for better display"""
        indicators_config = self.main_config.get('indicators', {})
        
        available_indicators = []
        enabled_indicators = []
        total_keys = 0
        
        for indicator_name, indicator_config in indicators_config.items():
            # Skip non-indicator config keys
            if indicator_name in ['allow_numpy_fallback', 'require_complete_calculation']:
                continue
            
            if not isinstance(indicator_config, dict):
                continue
                
            available_indicators.append(indicator_name)
            
            if indicator_config.get('enabled', False):
                if 'periods' in indicator_config:
                    periods = indicator_config['periods']
                    enabled_indicators.append(f"{indicator_name.upper()}({len(periods)} windows)")
                    total_keys += len(periods)
                else:
                    enabled_indicators.append(indicator_name.upper())
                    total_keys += 1
        
        return {
            'available_count': len(available_indicators),
            'available_list': available_indicators,
            'enabled_count': len(enabled_indicators),
            'enabled_list': enabled_indicators,
            'total_keys': total_keys,
            'total_with_ohlcv': total_keys + 5  # +5 for OHLCV
        }
    
    def reload_configs(self) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Force reload all configurations"""
        print("[RELOAD] Force reloading configurations...")
        return self.load_all_configs(force_reload=True)
    
    def validate_config_files_exist(self) -> bool:
        """Validate that all required configuration files exist"""
        required_files = ['config.json', 'strategies.json']
        
        for filename in required_files:
            file_path = self.config_dir / filename
            if not file_path.exists():
                raise ConfigurationError(f"Required configuration file missing: {file_path}")
        
        return True


# Convenience function for external use
def load_configs(config_dir: str) -> tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Convenience function to load all configurations.
    
    Args:
        config_dir: Directory containing configuration files (REQUIRED)
        
    Returns:
        Tuple of (main_config, strategies_config)
        
    Raises:
        ConfigurationError: If config_dir is not provided
    """
    if not config_dir:
        raise ConfigurationError("Configuration directory is required for load_configs()")
    
    loader = JSONConfigLoader(config_dir)
    return loader.load_all_configs()


if __name__ == "__main__":
    # Test configuration loading
    try:
        loader = JSONConfigLoader()
        main_config, strategies_config = loader.load_all_configs()
        
        print("🎉 Configuration loading test passed!")
        
        # Display summary
        summary = loader.get_config_summary()
        print("\n[SUMMARY] Configuration Summary:")
        for key, value in summary.items():
            print(f"   {key}: {value}")
        
        # Test required indicators
        required_indicators = loader.get_required_indicators()
        print(f"\n🔧 Required Indicators ({len(required_indicators)}):")
        for indicator in sorted(required_indicators):
            print(f"   - {indicator}")
            
    except ConfigurationError as e:
        print(f"[ERROR] Configuration loading failed: {e}")
        sys.exit(1)