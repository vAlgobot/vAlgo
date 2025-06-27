"""
Centralized environment variable loader for vAlgo Trading System
Prevents circular imports and manages environment state
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, Union, ClassVar

class EnvironmentLoader:
    """Centralized environment management"""
    
    _instance: ClassVar[Optional["EnvironmentLoader"]] = None
    _initialized: ClassVar[bool] = False
    
    def __new__(cls) -> "EnvironmentLoader":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if not EnvironmentLoader._initialized:
            self.env_loaded: bool = False
            self.load_environment()
            EnvironmentLoader._initialized = True
    
    def load_environment(self) -> bool:
        """Load environment variables from .env file"""
        if self.env_loaded:
            return True
        
        try:
            # Try to import and use python-dotenv
            try:
                from dotenv import load_dotenv
                
                # Find .env file - check current dir and parent dirs
                env_path = self._find_env_file()
                if env_path:
                    load_dotenv(env_path)
                    self.env_loaded = True
                    return True
                else:
                    # No .env file found, continue with system env vars
                    self.env_loaded = True
                    return True
                    
            except ImportError:
                # python-dotenv not available, use system env vars only
                self.env_loaded = True
                return True
                
        except Exception as e:
            print(f"Warning: Error loading environment: {e}")
            self.env_loaded = True  # Continue anyway
            return False
    
    def _find_env_file(self) -> Optional[Path]:
        """Find .env file in current or parent directories"""
        current_dir = Path.cwd()
        
        # Check current directory and up to 3 parent directories
        for _ in range(4):
            env_file = current_dir / ".env"
            if env_file.exists():
                return env_file
            current_dir = current_dir.parent
            if current_dir == current_dir.parent:  # Reached root
                break
        
        return None
    
    def get(self, key: str, default: str = "") -> str:
        """Get environment variable with default"""
        value = os.getenv(key, default)
        return str(value) if value is not None else default
    
    def get_bool(self, key: str, default: bool = False) -> bool:
        """Get boolean environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ('true', '1', 'yes', 'on')
    
    def get_int(self, key: str, default: int = 0) -> int:
        """Get integer environment variable"""
        try:
            value = os.getenv(key)
            return int(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def get_float(self, key: str, default: float = 0.0) -> float:
        """Get float environment variable"""
        try:
            value = os.getenv(key)
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def get_all_config(self) -> Dict[str, Union[str, bool, int, float]]:
        """Get all vAlgo-related environment variables"""
        return {
            'openalgo_url': self.get('OPENALGO_URL', 'http://localhost:5000'),
            'openalgo_api_key': self.get('OPENALGO_API_KEY', ''),
            'database_url': self.get('DATABASE_URL', 'data/vAlgo.db'),
            'database_type': self.get('DATABASE_TYPE', 'duckdb'),
            'log_level': self.get('LOG_LEVEL', 'INFO'),
            'log_file': self.get('LOG_FILE', 'logs/vAlgo.log'),
            'log_format': self.get('LOG_FORMAT', '%(asctime)s - %(name)s - %(levelname)s - %(message)s'),
            'paper_trading': self.get_bool('PAPER_TRADING', True),
            'default_capital': self.get_float('DEFAULT_CAPITAL', 100000),
            'default_risk_per_trade': self.get_float('DEFAULT_RISK_PER_TRADE', 0.02),
            'max_concurrent_trades': self.get_int('MAX_CONCURRENT_TRADES', 5),
            'debug': self.get_bool('DEBUG', False),
            'testing': self.get_bool('TESTING', False),
            'environment': self.get('ENVIRONMENT', 'development')
        }

# Global instance
env = EnvironmentLoader()

# Convenience functions
def get_env(key: str, default: str = "") -> str:
    """Get environment variable"""
    return env.get(key, default)

def get_env_bool(key: str, default: bool = False) -> bool:
    """Get boolean environment variable"""
    return env.get_bool(key, default)

def get_env_int(key: str, default: int = 0) -> int:
    """Get integer environment variable"""
    return env.get_int(key, default)

def get_env_float(key: str, default: float = 0.0) -> float:
    """Get float environment variable"""
    return env.get_float(key, default)

def get_all_config() -> Dict[str, Union[str, bool, int, float]]:
    """Get all configuration"""
    return env.get_all_config()