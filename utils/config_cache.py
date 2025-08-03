#!/usr/bin/env python3
"""
Config Cache Singleton
======================

Singleton pattern for config loading to eliminate redundant Excel file reads.
Provides centralized config management with automatic refresh capabilities.

Author: vAlgo Development Team
Created: July 4, 2025
Version: 1.0.0
"""

import threading
from typing import Optional, Dict, Any
from pathlib import Path
import time
from datetime import datetime

try:
    from utils.config_loader import ConfigLoader
    CONFIG_LOADER_AVAILABLE = True
except ImportError:
    CONFIG_LOADER_AVAILABLE = False
    ConfigLoader = None


class ConfigCache:
    """
    Singleton config cache to eliminate redundant config loading.
    Provides thread-safe access to cached configuration data.
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls, config_path: str = "config/config.xlsx"):
        """Ensure singleton pattern with thread safety."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(ConfigCache, cls).__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self, config_path: str = "config/config.xlsx"):
        """Initialize config cache if not already initialized."""
        if self._initialized:
            return
            
        self.config_path = config_path
        self._config_loader = None
        self._last_loaded = None
        self._file_modified_time = None
        self._cache_lock = threading.RLock()
        self._load_count = 0
        self._initialized = True
        
        # Load initial config
        self._load_config()
    
    def _load_config(self) -> bool:
        """Load configuration from Excel file."""
        if not CONFIG_LOADER_AVAILABLE:
            print("[CONFIG_CACHE] ConfigLoader not available")
            return False
            
        try:
            with self._cache_lock:
                print(f"[CONFIG_CACHE] Loading config from {self.config_path} (load #{self._load_count + 1})")
                self._config_loader = ConfigLoader(self.config_path)
                
                if self._config_loader.load_config():
                    self._last_loaded = datetime.now()
                    self._file_modified_time = Path(self.config_path).stat().st_mtime
                    self._load_count += 1
                    print(f"[CONFIG_CACHE] ✅ Config loaded successfully (total loads: {self._load_count})")
                    return True
                else:
                    print(f"[CONFIG_CACHE] ❌ Failed to load config from {self.config_path}")
                    return False
                    
        except Exception as e:
            print(f"[CONFIG_CACHE] ❌ Error loading config: {e}")
            return False
    
    def _needs_refresh(self) -> bool:
        """Check if config needs to be refreshed based on file modification time."""
        try:
            if not Path(self.config_path).exists():
                return False
                
            current_mtime = Path(self.config_path).stat().st_mtime
            return current_mtime != self._file_modified_time
            
        except Exception:
            return False
    
    def get_config_loader(self, force_refresh: bool = False) -> Optional[ConfigLoader]:
        """
        Get cached config loader instance.
        
        Args:
            force_refresh: Force reload from file even if cached
            
        Returns:
            ConfigLoader instance or None if loading failed
        """
        with self._cache_lock:
            # Check if refresh is needed
            if force_refresh or self._needs_refresh() or self._config_loader is None:
                if not self._load_config():
                    return None
            
            return self._config_loader
    
    def get_active_indicators(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get active indicators from cached config."""
        config_loader = self.get_config_loader(force_refresh)
        if config_loader:
            return config_loader.get_active_indicators()
        return {}
    
    def get_active_brokers(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get active brokers from cached config."""
        config_loader = self.get_config_loader(force_refresh)
        if config_loader:
            return config_loader.get_active_brokers()
        return {}
    
    def get_active_instruments(self, force_refresh: bool = False) -> Dict[str, Any]:
        """Get active instruments from cached config."""
        config_loader = self.get_config_loader(force_refresh)
        if config_loader:
            return config_loader.get_active_instruments()
        return {}
    
    def get_trading_mode(self, force_refresh: bool = False) -> str:
        """Get trading mode from cached config."""
        config_loader = self.get_config_loader(force_refresh)
        if config_loader:
            return config_loader.get_trading_mode()
        return "BACKTESTING"
    
    def get_primary_broker_openalgo_config(self, force_refresh: bool = False) -> Dict[str, str]:
        """Get primary broker OpenAlgo config from cached config."""
        config_loader = self.get_config_loader(force_refresh)
        if config_loader:
            return config_loader.get_primary_broker_openalgo_config()
        return {}
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics for monitoring."""
        with self._cache_lock:
            return {
                "config_path": self.config_path,
                "total_loads": self._load_count,
                "last_loaded": self._last_loaded.isoformat() if self._last_loaded else None,
                "config_available": self._config_loader is not None,
                "file_exists": Path(self.config_path).exists(),
                "file_modified_time": self._file_modified_time
            }
    
    def force_refresh(self) -> bool:
        """Force refresh of cached config."""
        return self.get_config_loader(force_refresh=True) is not None
    
    @classmethod
    def reset_instance(cls):
        """Reset singleton instance (primarily for testing)."""
        with cls._lock:
            cls._instance = None


# Convenience function for easy access
def get_cached_config(config_path: str = "config/config.xlsx") -> ConfigCache:
    """Get the singleton config cache instance."""
    return ConfigCache(config_path)


# Global instance for easy import
config_cache = get_cached_config()