"""
Indicator Key Manager for Ultimate Efficiency System
===================================================

Simple and efficient indicator key management system that auto-generates
indicator_keys.json from config.json. No complex parsing - just pure
config-based approach.

Features:
- Auto-generate indicator_keys.json from config.json 
- Simple key validation before signal generation
- User-friendly indicator key reference
- Automatic updates when config changes

Author: vAlgo Development Team
Created: July 29, 2025
Based on simplified config-based approach
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional, Set
from datetime import datetime
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from core.json_config_loader import JSONConfigLoader, ConfigurationError

try:
    from utils.selective_logger import get_selective_logger
    SELECTIVE_LOGGER_AVAILABLE = True
except ImportError:
    SELECTIVE_LOGGER_AVAILABLE = False
    # Create fallback logger functions
    def get_selective_logger(name):
        class FallbackLogger:
            def log_major_component(self, message, component="SYSTEM"): print(f"🚀 {component}: {message}")
            def log_detailed(self, message, level, component): print(f"🔍 {component}: {message}")
            def log_performance(self, metrics, component="PERFORMANCE"): 
                print(f"📊 {component} METRICS:")
                for key, value in metrics.items():
                    print(f"   {key}: {value}")
            def log_portfolio_analysis(self, analysis, strategy): 
                print(f"💼 PORTFOLIO ANALYSIS - {strategy}:")
                for key, value in analysis.items():
                    print(f"   {key}: {value}")
            def log_bot_performance(self, data): print(f"🤖 Performance: {data}")
        return FallbackLogger()


class IndicatorKeyManager:
    """
    Simple indicator key management system using pure config-based approach.
    
    Features:
    - Auto-generate indicator_keys.json from config.json enabled indicators
    - Simple validation before signal generation  
    - No complex parsing or regex operations
    - User-friendly key reference system
    """
    
    def __init__(self, config_loader: Optional[JSONConfigLoader] = None):
        """
        Initialize Indicator Key Manager.
        
        Args:
            config_loader: Optional pre-initialized config loader
        """
        self.config_loader = config_loader or JSONConfigLoader()
        self.selective_logger = get_selective_logger("indicator_key_manager")
        
        # Paths
        self.config_dir = self.config_loader.config_dir
        self.indicator_keys_file = self.config_dir / "indicator_keys.json"
        
        self.selective_logger.log_major_component("Indicator Key Manager initialized", "INDICATOR_KEYS")
        self.selective_logger.log_detailed(f"Keys file location: {self.indicator_keys_file}", "INFO", "INDICATOR_KEYS")
    
    def generate_indicator_keys_json(self) -> Dict[str, Any]:
        """
        Generate comprehensive indicator_keys.json from config.json.
        Pure config-based approach - no parsing complexity.
        
        Returns:
            Dictionary of all available indicator keys
        """
        try:
            self.selective_logger.log_detailed("Generating indicator keys from config.json", "INFO", "INDICATOR_KEYS")
            
            # Get enabled indicators from simplified config loader
            indicator_keys = self.config_loader.get_enabled_indicators_from_config()
            
            # Create comprehensive indicator keys JSON structure
            indicator_keys_json = {
                "metadata": {
                    "generated_at": datetime.now().isoformat(),
                    "version": "1.0",
                    "approach": "config_based_simple",
                    "total_keys": len(indicator_keys),
                    "source": "config.json enabled indicators"
                },
                "market_data_keys": {
                    key: desc for key, desc in indicator_keys.items() 
                    if key in ['close', 'open', 'high', 'low', 'volume']
                },
                "calculated_indicators": {
                    key: desc for key, desc in indicator_keys.items() 
                    if key not in ['close', 'open', 'high', 'low', 'volume']
                },
                "all_available_keys": indicator_keys,
                "usage_examples": {
                    "simple_condition": "close > ma_20",
                    "complex_condition": "(close > ma_9) & (ma_9 > ma_20) & (rsi_14 < 70)",
                    "trend_following": "(ma_9 > ma_20) & (ma_20 > ma_50) & (ma_50 > ma_200)",
                    "reversal_strategy": "(rsi_14 < 30) & (close > ma_20)"
                },
                "configuration_source": {
                    "enabled_indicators": self._get_enabled_indicator_summary(),
                    "how_to_add_indicators": "Enable indicators in config.json with 'enabled': true",
                    "automatic_update": "This file auto-updates when config.json changes"
                }
            }
            
            self.selective_logger.log_detailed(f"Generated {len(indicator_keys)} indicator keys", "INFO", "INDICATOR_KEYS")
            return indicator_keys_json
            
        except Exception as e:
            self.selective_logger.log_detailed(f"Error generating indicator keys: {e}", "ERROR", "INDICATOR_KEYS")
            raise ConfigurationError(f"Failed to generate indicator keys: {e}")
    
    def save_indicator_keys_json(self, force_update: bool = False) -> bool:
        """
        Save indicator_keys.json to config directory.
        
        Args:
            force_update: Force update even if file exists and is recent
            
        Returns:
            True if file was updated, False if no update needed
        """
        try:
            # Check if update is needed
            if not force_update and self._is_keys_file_current():
                self.selective_logger.log_detailed("Indicator keys file is current, no update needed", "INFO", "INDICATOR_KEYS")
                return False
            
            # Generate fresh indicator keys
            indicator_keys_json = self.generate_indicator_keys_json()
            
            # Save to file
            with open(self.indicator_keys_file, 'w', encoding='utf-8') as f:
                json.dump(indicator_keys_json, f, indent=2, ensure_ascii=False)
            
            self.selective_logger.log_major_component(f"Updated indicator_keys.json with {indicator_keys_json['metadata']['total_keys']} keys", "INDICATOR_KEYS")
            self.selective_logger.log_detailed(f"Saved to: {self.indicator_keys_file}", "INFO", "INDICATOR_KEYS")
            
            return True
            
        except Exception as e:
            self.selective_logger.log_detailed(f"Error saving indicator keys: {e}", "ERROR", "INDICATOR_KEYS")
            raise ConfigurationError(f"Failed to save indicator keys: {e}")
    
    def load_indicator_keys_json(self) -> Dict[str, Any]:
        """
        Load existing indicator_keys.json file.
        
        Returns:
            Dictionary of indicator keys or empty dict if file doesn't exist
        """
        try:
            if not self.indicator_keys_file.exists():
                self.selective_logger.log_detailed("Indicator keys file doesn't exist, will create new one", "INFO", "INDICATOR_KEYS")
                return {}
            
            with open(self.indicator_keys_file, 'r', encoding='utf-8') as f:
                keys_data = json.load(f)
            
            self.selective_logger.log_detailed(f"Loaded indicator keys file with {keys_data.get('metadata', {}).get('total_keys', 0)} keys", "INFO", "INDICATOR_KEYS")
            return keys_data
            
        except Exception as e:
            self.selective_logger.log_detailed(f"Error loading indicator keys: {e}", "WARNING", "INDICATOR_KEYS")
            return {}
    
    def validate_indicator_keys(self, current_indicators: Dict[str, Any]) -> Dict[str, Any]:
        """
        Simple validation of current indicators against available keys.
        
        Args:
            current_indicators: Dictionary of current indicator data
            
        Returns:
            Validation results with any updates needed
        """
        try:
            self.selective_logger.log_detailed("Validating indicator keys", "INFO", "INDICATOR_KEYS")
            
            # Load existing keys file
            existing_keys = self.load_indicator_keys_json()
            
            # Get current available keys from config
            current_available_keys = self.config_loader.get_enabled_indicators_from_config()
            
            # Check if keys have changed
            existing_all_keys = existing_keys.get('all_available_keys', {})
            keys_changed = existing_all_keys != current_available_keys
            
            validation_result = {
                'keys_changed': keys_changed,
                'total_current_keys': len(current_available_keys),
                'total_existing_keys': len(existing_all_keys),
                'new_keys': set(current_available_keys.keys()) - set(existing_all_keys.keys()),
                'removed_keys': set(existing_all_keys.keys()) - set(current_available_keys.keys()),
                'update_needed': keys_changed
            }
            
            if keys_changed:
                self.selective_logger.log_major_component("Indicator keys have changed - updating keys file", "INDICATOR_KEYS")
                self.save_indicator_keys_json(force_update=True)
                validation_result['keys_file_updated'] = True
            else:
                self.selective_logger.log_detailed("Indicator keys are current", "INFO", "INDICATOR_KEYS")
                validation_result['keys_file_updated'] = False
            
            return validation_result
            
        except Exception as e:
            self.selective_logger.log_detailed(f"Error validating indicator keys: {e}", "ERROR", "INDICATOR_KEYS")
            raise ConfigurationError(f"Indicator key validation failed: {e}")
    
    def get_user_friendly_keys_summary(self) -> str:
        """
        Get user-friendly summary of available indicator keys.
        
        Returns:
            Formatted string with key information for users
        """
        try:
            # Ensure keys file is current
            self.save_indicator_keys_json()
            
            # Load keys data
            keys_data = self.load_indicator_keys_json()
            
            if not keys_data:
                return "❌ No indicator keys available. Check config.json for enabled indicators."
            
            summary = f"""
📊 **Available Indicator Keys Summary**

**Market Data Keys** ({len(keys_data.get('market_data_keys', {}))} keys):
{self._format_keys_dict(keys_data.get('market_data_keys', {}))}

**Calculated Indicators** ({len(keys_data.get('calculated_indicators', {}))} keys):
{self._format_keys_dict(keys_data.get('calculated_indicators', {}))}

**Total Available Keys**: {keys_data.get('metadata', {}).get('total_keys', 0)}

**Usage Examples**:
- Simple: `close > ma_20`
- Complex: `(close > ma_9) & (ma_9 > ma_20) & (rsi_14 < 70)`
- Trend: `(ma_9 > ma_20) & (ma_20 > ma_50) & (ma_50 > ma_200)`

**To Add More Indicators**: Enable them in config.json with "enabled": true
**Keys File**: {self.indicator_keys_file.name}
"""
            return summary.strip()
            
        except Exception as e:
            return f"❌ Error generating keys summary: {e}"
    
    def _is_keys_file_current(self) -> bool:
        """Check if indicator keys file is current with config.json"""
        if not self.indicator_keys_file.exists():
            return False
        
        try:
            # Compare modification times
            keys_mtime = self.indicator_keys_file.stat().st_mtime
            config_mtime = (self.config_dir / "config.json").stat().st_mtime
            
            return keys_mtime >= config_mtime
            
        except Exception:
            return False
    
    def _get_enabled_indicator_summary(self) -> Dict[str, Any]:
        """Get summary of enabled indicators from config"""
        main_config = self.config_loader.get_main_config()
        enabled_summary = {}
        
        for indicator_name, indicator_config in main_config['indicators'].items():
            # Skip non-indicator config keys
            if indicator_name in ['allow_numpy_fallback', 'require_complete_calculation']:
                continue
            
            # Handle both boolean and dictionary configurations    
            if isinstance(indicator_config, bool):
                if indicator_config:  # If True, add basic configuration
                    enabled_summary[indicator_name] = {
                        'enabled': True,
                        'periods': [],
                        'talib_function': None,
                        'config': {}
                    }
            elif isinstance(indicator_config, dict):
                if indicator_config.get('enabled', False):
                    enabled_summary[indicator_name] = {
                        'enabled': True,
                        'periods': indicator_config.get('periods', []),
                        'talib_function': indicator_config.get('talib_function'),
                        'config': {k: v for k, v in indicator_config.items() if k not in ['enabled']}
                    }
        
        return enabled_summary
    
    def _format_keys_dict(self, keys_dict: Dict[str, str]) -> str:
        """Format keys dictionary for user display"""
        if not keys_dict:
            return "  None"
        
        formatted = []
        for key, desc in keys_dict.items():
            formatted.append(f"  • {key}: {desc}")
        
        return "\n".join(formatted)


# Convenience functions for external use
def create_indicator_key_manager(config_dir: Optional[str] = None) -> IndicatorKeyManager:
    """
    Create and initialize Indicator Key Manager.
    
    Args:
        config_dir: Optional configuration directory path
        
    Returns:
        Initialized IndicatorKeyManager instance
    """
    config_loader = JSONConfigLoader(config_dir) if config_dir else None
    return IndicatorKeyManager(config_loader)


def update_indicator_keys_file(config_dir: Optional[str] = None, force: bool = False) -> bool:
    """
    Convenience function to update indicator keys file.
    
    Args:
        config_dir: Optional configuration directory path
        force: Force update even if file is current
        
    Returns:
        True if file was updated
    """
    manager = create_indicator_key_manager(config_dir)
    return manager.save_indicator_keys_json(force_update=force)


if __name__ == "__main__":
    # Test Indicator Key Manager
    try:
        print("🧪 Testing Indicator Key Manager...")
        
        # Create manager
        manager = IndicatorKeyManager()
        
        # Generate and save indicator keys
        updated = manager.save_indicator_keys_json(force_update=True)
        print(f"✅ Keys file updated: {updated}")
        
        # Display user-friendly summary
        summary = manager.get_user_friendly_keys_summary()
        print("\n" + summary)
        
        print("\n🎉 Indicator Key Manager test completed!")
        
    except Exception as e:
        print(f"❌ Indicator Key Manager test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)