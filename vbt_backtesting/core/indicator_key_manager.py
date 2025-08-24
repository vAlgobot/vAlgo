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
import re
from pathlib import Path
from typing import Dict, Any, Optional, Set, List, Tuple
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
        
        # Strategy validation tracking
        self.validation_cache = {}
        self.last_validation = None
        
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
                    if key in ['close', 'open', 'high', 'low', 'volume', '1m_open', '1m_high', '1m_low', '1m_close', '1m_volume', 'Indicator_Timestamp']
                },
                "calculated_indicators": {
                    key: desc for key, desc in indicator_keys.items() 
                    if key not in ['close', 'open', 'high', 'low', 'volume', '1m_open', '1m_high', '1m_low', '1m_close', '1m_volume', 'Indicator_Timestamp']
                },
                "all_available_keys": indicator_keys,
                "usage_examples": self._generate_usage_examples(indicator_keys),
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
    
    def extract_indicator_references(self, expression: str) -> Set[str]:
        """
        Extract indicator references from strategy entry/exit expressions.
        
        Args:
            expression: Strategy condition expression
            
        Returns:
            Set of indicator keys referenced in the expression
        """
        try:
            # Pattern to match indicator references:
            # - Standard format: rsi_14, sma_20, ema_9, etc.
            # - Market data: close, open, high, low, volume
            # - Special indicators: previous_day_high, cpr_pivot, signal_candle_open, etc.
            pattern = r'\b(?:rsi_\d+|sma_\d+|ema_\d+|ma_\d+|vwap|close|open|high|low|volume|previous_day_\w+|previous_candle_\w+|cpr_\w+|signal_candle_\w+)\b'
            
            matches = re.findall(pattern, expression, re.IGNORECASE)
            return set(matches)
            
        except Exception as e:
            # Silent error handling to avoid encoding issues
            return set()
    
    def _generate_usage_examples(self, indicator_keys: Dict[str, str]) -> Dict[str, str]:
        """
        Generate usage examples dynamically based on available keys.
        
        Args:
            indicator_keys: Dictionary of available indicator keys
            
        Returns:
            Dictionary of usage examples
        """
        # Base examples (always available)
        examples = {
            "simple_condition": "close > ma_20",
            "complex_condition": "(close > ma_9) & (ma_9 > ma_20) & (rsi_14 < 70)",
            "trend_following": "(ma_9 > ma_20) & (ma_20 > ma_50) & (ma_50 > ma_200)",
            "reversal_strategy": "(rsi_14 < 30) & (close > ma_20)"
        }
        
        # Add 1m examples if LTP keys are available
        has_1m_keys = any(key.startswith('1m_') for key in indicator_keys.keys())
        if has_1m_keys:
            examples.update({
                "breakout_strategy": "(1m_high > cpr_r1) & (close > sma_20)",
                "breakdown_strategy": "(1m_low < cpr_s1) & (close < sma_20)",
                "precision_entry": "(1m_close > 1m_open) & (rsi_14 < 70)",
                "volume_confirmation": "(1m_volume > 1000) & (1m_high > previous_candle_high)",
                "multi_timeframe_trend": "(1m_close > 1m_open) & (close > ema_20) & (cpr_type == 'extreme_narrow')"
            })
        
        # Add signal_candle examples if signal_candle keys are available
        has_signal_candle_keys = any(key.startswith('signal_candle_') for key in indicator_keys.keys())
        if has_signal_candle_keys:
            examples.update({
                "signal_candle_stop_loss": "(close < signal_candle_low)",
                "signal_candle_take_profit": "(close > signal_candle_high * 1.02)",
                "signal_candle_breakout_exit": "(close < signal_candle_close * 0.98)",
                "signal_candle_range_exit": "(close < signal_candle_low) | (close > signal_candle_high)"
            })
        
        # Add SL/TP examples if sl_price and tp_price keys are available
        has_sl_tp_keys = 'sl_price' in indicator_keys and 'tp_price' in indicator_keys
        if has_sl_tp_keys:
            examples.update({
                "sl_tp_exit": "(ltp_close <= sl_price) | (ltp_high >= tp_price)",
                "stop_loss_only": "ltp_close <= sl_price",
                "take_profit_only": "ltp_high >= tp_price"
            })
        
        return examples
    
    def validate_strategies_against_indicators(self, force_validation: bool = False) -> Dict[str, Any]:
        """
        Comprehensive validation of all active strategies against available indicators.
        
        Args:
            force_validation: Force validation even if cached
            
        Returns:
            Comprehensive validation report with all issues and recommendations
        """
        try:
            # Silent validation - no logging to avoid emoji encoding issues
            
            # Get current available indicators
            available_indicators = self.config_loader.get_enabled_indicators_from_config()
            available_keys = set(available_indicators.keys())
            
            # Get active strategies
            active_strategies = self.config_loader.get_enabled_strategies_cases()
            
            validation_report = {
                'validation_passed': True,
                'total_strategies_checked': 0,
                'total_cases_checked': 0,
                'issues_found': [],
                'missing_indicators': set(),
                'strategies_with_issues': [],
                'available_indicators': available_keys,
                'config_changes_detected': False,
                'recommendations': [],
                'summary': {}
            }
            
            # Check each active strategy group and case
            for group_name, group_config in active_strategies.items():
                if group_config.get('status') != 'Active':
                    continue
                    
                validation_report['total_strategies_checked'] += 1
                cases = group_config.get('cases', {})
                
                for case_name, case_config in cases.items():
                    if case_config.get('status') != 'Active':
                        continue
                        
                    validation_report['total_cases_checked'] += 1
                    
                    # Extract entry and exit conditions
                    entry_condition = case_config.get('entry', '')
                    exit_condition = case_config.get('exit', '')
                    
                    # Extract indicator references from conditions
                    entry_indicators = self.extract_indicator_references(entry_condition)
                    exit_indicators = self.extract_indicator_references(exit_condition)
                    all_indicators = entry_indicators.union(exit_indicators)
                    
                    # Check for missing indicators
                    missing_in_case = all_indicators - available_keys
                    
                    if missing_in_case:
                        validation_report['validation_passed'] = False
                        validation_report['missing_indicators'].update(missing_in_case)
                        
                        issue = {
                            'strategy_group': group_name,
                            'case_name': case_name,
                            'missing_indicators': list(missing_in_case),
                            'entry_condition': entry_condition,
                            'exit_condition': exit_condition,
                            'all_referenced_indicators': list(all_indicators),
                            'available_alternatives': self._find_indicator_alternatives(missing_in_case, available_keys)
                        }
                        
                        validation_report['issues_found'].append(issue)
                        
                        if case_name not in validation_report['strategies_with_issues']:
                            validation_report['strategies_with_issues'].append(case_name)
            
            # Check if indicator configuration has changed
            validation_report['config_changes_detected'] = self._check_config_changes()
            
            # Generate recommendations
            validation_report['recommendations'] = self._generate_fix_recommendations(validation_report)
            
            # Create summary
            validation_report['summary'] = self._create_validation_summary(validation_report)
            
            self.validation_cache = validation_report
            self.last_validation = datetime.now()
            
            # Silent completion - no logging to avoid emoji issues
            if not validation_report['validation_passed']:
                # Issues found - will be handled by calling method
                pass
            else:
                # Validation passed - continue silently
                pass
            
            return validation_report
            
        except Exception as e:
            # Silent error handling to avoid logging issues
            raise ConfigurationError(f"Strategy validation failed: {e}")
    
    def _find_indicator_alternatives(self, missing_indicators: Set[str], available_indicators: Set[str]) -> Dict[str, List[str]]:
        """Find alternative indicators for missing ones."""
        alternatives = {}
        
        for missing in missing_indicators:
            alts = []
            
            # Extract base indicator type (e.g., 'rsi' from 'rsi_14')
            if '_' in missing:
                base_type = missing.split('_')[0]
                # Find available indicators of the same type
                for available in available_indicators:
                    if available.startswith(f"{base_type}_"):
                        alts.append(available)
            
            alternatives[missing] = alts
        
        return alternatives
    
    def _check_config_changes(self) -> bool:
        """Check if indicator configuration has changed since last validation."""
        try:
            config_file = self.config_dir / "config.json"
            keys_file = self.indicator_keys_file
            
            if not keys_file.exists():
                return True
            
            config_mtime = config_file.stat().st_mtime
            keys_mtime = keys_file.stat().st_mtime
            
            return config_mtime > keys_mtime
            
        except Exception:
            return True
    
    def _generate_fix_recommendations(self, validation_report: Dict[str, Any]) -> List[str]:
        """Generate actionable fix recommendations."""
        recommendations = []
        
        if not validation_report['validation_passed']:
            recommendations.append("⚠️  CRITICAL: Fix strategy configurations before running backtesting")
            
            # Specific recommendations for each issue
            for issue in validation_report['issues_found']:
                case_name = issue['case_name']
                missing = issue['missing_indicators']
                alternatives = issue['available_alternatives']
                
                recommendations.append(f"\n🔧 Strategy '{case_name}' fixes needed:")
                
                for missing_indicator in missing:
                    alts = alternatives.get(missing_indicator, [])
                    if alts:
                        recommendations.append(f"  • Replace '{missing_indicator}' with one of: {', '.join(alts)}")
                    else:
                        recommendations.append(f"  • '{missing_indicator}' not available - enable in config.json or remove from strategy")
        
        if validation_report['config_changes_detected']:
            recommendations.append("\n📋 Configuration changes detected:")
            recommendations.append("  • indicator_keys.json will be automatically updated")
            recommendations.append("  • Review strategy references after indicator parameter changes")
        
        return recommendations
    
    def _create_validation_summary(self, validation_report: Dict[str, Any]) -> Dict[str, Any]:
        """Create validation summary for display."""
        return {
            'total_issues': len(validation_report['issues_found']),
            'affected_strategies': len(validation_report['strategies_with_issues']),
            'missing_indicators_count': len(validation_report['missing_indicators']),
            'validation_status': 'PASSED' if validation_report['validation_passed'] else 'FAILED',
            'total_available_indicators': len(validation_report['available_indicators'])
        }
    
    def display_validation_report(self, validation_report: Dict[str, Any]) -> None:
        """
        Display comprehensive validation report with clear error messages.
        """
        print(f"\n{'=' * 80}")
        print(f"🔍 INDICATOR-STRATEGY VALIDATION REPORT")
        print(f"{'=' * 80}")
        
        summary = validation_report['summary']
        print(f"📊 VALIDATION SUMMARY:")
        print(f"   Status: {summary['validation_status']}")
        print(f"   Strategies Checked: {validation_report['total_strategies_checked']}")
        print(f"   Cases Checked: {validation_report['total_cases_checked']}")
        print(f"   Available Indicators: {summary['total_available_indicators']}")
        
        if not validation_report['validation_passed']:
            print(f"\n❌ CRITICAL ISSUES FOUND:")
            print(f"   Total Issues: {summary['total_issues']}")
            print(f"   Affected Strategies: {summary['affected_strategies']}")
            print(f"   Missing Indicators: {summary['missing_indicators_count']}")
            
            print(f"\n🚨 DETAILED ISSUES:")
            for i, issue in enumerate(validation_report['issues_found'], 1):
                print(f"\n   {i}. Strategy: '{issue['case_name']}' (Group: {issue['strategy_group']})")
                print(f"      Missing: {', '.join(issue['missing_indicators'])}")
                print(f"      Entry: {issue['entry_condition']}")
                print(f"      Exit: {issue['exit_condition']}")
                
                # Show alternatives if available
                for missing_indicator, alternatives in issue['available_alternatives'].items():
                    if alternatives:
                        print(f"      💡 '{missing_indicator}' → Available: {', '.join(alternatives)}")
            
            print(f"\n🔄 CONFIGURATION CHANGES:")
            if validation_report['config_changes_detected']:
                print(f"   ✅ Configuration changes detected - indicator_keys.json will be updated")
            else:
                print(f"   ℹ️  No configuration changes detected")
            
            print(f"\n💡 RECOMMENDATIONS:")
            for rec in validation_report['recommendations']:
                print(f"   {rec}")
            
            print(f"\n⚠️  SYSTEM WILL STOP: Fix configurations before continuing")
        else:
            print(f"\n✅ ALL VALIDATIONS PASSED")
            print(f"   All strategy references are valid")
            print(f"   System ready for backtesting")
        
        print(f"\n{'=' * 80}")
    
    def validate_and_stop_if_issues(self) -> None:
        """
        Main validation method that:
        1. Updates indicator_keys.json if config changed
        2. Validates all active strategies 
        3. Stops execution immediately if issues found
        4. Provides clear guidance to user
        
        Raises:
            ConfigurationError: If validation fails with simple message
        """
        try:
            print(f"\nVALIDATION: Checking indicator configuration...")
            
            # Step 1: Check if config changes detected and update keys file
            config_changed = self._check_config_changes()
            if config_changed:
                print(f"CONFIG UPDATE: Updating indicator_keys.json...")
                self.save_indicator_keys_json(force_update=True)
                print(f"UPDATED: indicator_keys.json updated successfully")
                
                # Show what changed and newly available keys
                self._display_config_changes_simple()
                self._display_newly_available_keys()
            
            # Step 2: Run comprehensive validation
            validation_report = self.validate_strategies_against_indicators()
            
            # Step 3: If validation passed, continue silently
            if validation_report['validation_passed']:
                if config_changed:
                    print(f"SUCCESS: All configurations validated - system ready")
                return  # Continue execution
            
            # Step 4: If validation failed, show simple clear message and stop
            print(f"\nPROBLEM DETECTED: Strategy configuration issues found")
            
            # Always show available keys for user reference
            if not config_changed:
                print(f"\nCURRENT AVAILABLE INDICATOR KEYS:")
                self._display_available_keys_summary()
            
            # Show simple, actionable information
            self._display_simple_validation_issues(validation_report)
            
            # Create simple error message for ConfigurationError
            error_msg = self._create_simple_error_message(validation_report)
            
            # Stop execution immediately
            raise ConfigurationError(error_msg)
            
        except ConfigurationError:
            # Re-raise configuration errors (our intentional stops)
            raise
        except Exception as e:
            # Handle unexpected errors with simple message
            error_msg = f"Validation error: {e}"
            print(f"ERROR: {error_msg}")
            raise ConfigurationError(error_msg)
    
    def _display_config_changes(self) -> None:
        """Display what indicator parameters changed."""
        try:
            # Get old and new configurations for comparison
            main_config = self.config_loader.get_main_config()
            indicators_config = main_config.get('indicators', {})
            
            print(f"\n🔄 INDICATOR CHANGES DETECTED:")
            
            # Show current config parameters
            for indicator_name, indicator_config in indicators_config.items():
                if isinstance(indicator_config, dict) and indicator_config.get('enabled', False):
                    periods = indicator_config.get('periods', [])
                    if periods:
                        print(f"   • {indicator_name.upper()}: {periods} → creates {', '.join([f'{indicator_name}_{p}' for p in periods])}")
                    else:
                        print(f"   • {indicator_name.upper()}: enabled (no periods)")
            
        except Exception as e:
            print(f"   ⚠️  Could not display config changes: {e}")
    
    def _create_detailed_error_message(self, validation_report: Dict[str, Any]) -> str:
        """Create detailed error message for ConfigurationError."""
        
        error_lines = [
            "\n🚨 STRATEGY-INDICATOR VALIDATION FAILED",
            "=" * 60,
            ""
        ]
        
        # Summary of issues
        summary = validation_report['summary']
        error_lines.extend([
            f"📊 ISSUES FOUND:",
            f"   • {summary['total_issues']} validation errors",
            f"   • {summary['affected_strategies']} strategies affected", 
            f"   • {summary['missing_indicators_count']} indicators missing",
            ""
        ])
        
        # Specific issues
        error_lines.append("🔴 SPECIFIC ISSUES:")
        for issue in validation_report['issues_found']:
            error_lines.extend([
                f"   Strategy: '{issue['case_name']}' (Group: {issue['strategy_group']})",
                f"   Missing: {', '.join(issue['missing_indicators'])}",
                f"   Entry: {issue['entry_condition']}",
                f"   Exit: {issue['exit_condition']}",
                ""
            ])
        
        # Available alternatives  
        error_lines.append("💡 AVAILABLE ALTERNATIVES:")
        all_alternatives = {}
        for issue in validation_report['issues_found']:
            all_alternatives.update(issue['available_alternatives'])
        
        for missing, alternatives in all_alternatives.items():
            if alternatives:
                error_lines.append(f"   • Replace '{missing}' with: {', '.join(alternatives)}")
            else:
                error_lines.append(f"   • '{missing}' - Enable in config.json or remove from strategy")
        
        error_lines.extend([
            "",
            "🛠️  REQUIRED ACTIONS:",
            "   1. Update strategies.json with valid indicator references",
            "   2. OR modify config.json to include missing indicators", 
            "   3. OR disable strategies that use unavailable indicators",
            "",
            "✅ GOOD NEWS: indicator_keys.json has been updated automatically",
            "",
            "⚠️  SYSTEM STOPPED: Fix strategy configurations before continuing",
            "=" * 60
        ])
        
        return "\n".join(error_lines)
    
    def _display_simple_validation_issues(self, validation_report: Dict[str, Any]) -> None:
        """Display simple, clear validation issues without overwhelming detail."""
        try:
            issues = validation_report['issues_found']
            
            print(f"\nISSUES FOUND:")
            for i, issue in enumerate(issues, 1):
                missing_indicators = issue['missing_indicators']
                case_name = issue['case_name']
                alternatives = issue['available_alternatives']
                
                print(f"  {i}. Strategy '{case_name}' uses: {', '.join(missing_indicators)} - NOT AVAILABLE")
                
                # Show unique alternatives for all missing indicators
                all_alternatives = set()
                for missing_indicator in missing_indicators:
                    available_alts = alternatives.get(missing_indicator, [])
                    all_alternatives.update(available_alts)
                
                if all_alternatives:
                    print(f"     Available instead: {', '.join(sorted(all_alternatives))}")
                else:
                    print(f"     No alternatives found")
            
            print(f"\nSOLUTIONS:")
            print(f"  1. Edit strategies.json - Replace missing indicators with available ones")
            print(f"  2. Edit config.json - Add missing indicators to periods list") 
            print(f"  3. Disable strategies that use unavailable indicators")
            
            print(f"\nSYSTEM STOPPED: Fix configuration and try again")
            
        except Exception as e:
            print(f"Error displaying issues: {e}")
    
    def _create_simple_error_message(self, validation_report: Dict[str, Any]) -> str:
        """Create simple error message without emojis or complex formatting."""
        
        issues = validation_report['issues_found']
        missing_indicators = validation_report['missing_indicators']
        
        error_parts = [
            "Strategy configuration validation failed",
            f"Missing indicators: {', '.join(missing_indicators)}",
            f"Affected strategies: {len(issues)}",
            "Fix strategies.json or config.json before continuing"
        ]
        
        return " - ".join(error_parts)
    
    def _display_newly_available_keys(self) -> None:
        """Display newly available indicator keys for user reference."""
        try:
            # Get current available indicators
            available_indicators = self.config_loader.get_enabled_indicators_from_config()
            
            print(f"\nNEWLY AVAILABLE INDICATOR KEYS:")
            
            # Group by indicator type for better display
            indicator_groups = {}
            for key, description in available_indicators.items():
                # Skip market data keys
                if key in ['close', 'open', 'high', 'low', 'volume']:
                    continue
                    
                # Extract base indicator type (e.g., 'rsi' from 'rsi_10')
                if '_' in key:
                    base_type = key.split('_')[0].upper()
                else:
                    base_type = key.upper()
                
                if base_type not in indicator_groups:
                    indicator_groups[base_type] = []
                indicator_groups[base_type].append(key)
            
            # Display grouped indicators
            for indicator_type, keys in sorted(indicator_groups.items()):
                if keys:
                    keys_display = ", ".join(sorted(keys))
                    print(f"  {indicator_type}: {keys_display}")
            
            print(f"  Total available: {len(available_indicators)} keys")
            
        except Exception as e:
            print(f"  Error displaying available keys: {e}")
    
    def _display_config_changes_simple(self) -> None:
        """Display what indicator parameters changed in a simple format."""
        try:
            main_config = self.config_loader.get_main_config()
            indicators_config = main_config.get('indicators', {})
            
            print(f"\nCONFIG CHANGES APPLIED:")
            
            # Show current enabled indicators with their periods
            for indicator_name, indicator_config in indicators_config.items():
                if isinstance(indicator_config, dict) and indicator_config.get('enabled', False):
                    periods = indicator_config.get('periods', [])
                    if periods:
                        indicator_keys = [f"{indicator_name}_{p}" for p in periods]
                        print(f"  {indicator_name.upper()}: {periods} -> creates {', '.join(indicator_keys)}")
                    else:
                        print(f"  {indicator_name.upper()}: enabled")
            
        except Exception as e:
            print(f"  Error displaying config changes: {e}")
    
    def _display_available_keys_summary(self) -> None:
        """Display a summary of currently available keys."""
        try:
            available_indicators = self.config_loader.get_enabled_indicators_from_config()
            
            # Group by indicator type
            indicator_groups = {}
            for key, description in available_indicators.items():
                if key in ['close', 'open', 'high', 'low', 'volume']:
                    continue
                    
                if '_' in key:
                    base_type = key.split('_')[0].upper()
                else:
                    base_type = key.upper()
                
                if base_type not in indicator_groups:
                    indicator_groups[base_type] = []
                indicator_groups[base_type].append(key)
            
            # Display in compact format
            for indicator_type, keys in sorted(indicator_groups.items()):
                if keys:
                    keys_display = ", ".join(sorted(keys))
                    print(f"  {indicator_type}: {keys_display}")
            
        except Exception as e:
            print(f"  Error displaying available keys: {e}")


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