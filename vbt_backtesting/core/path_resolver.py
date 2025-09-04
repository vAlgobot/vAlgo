"""
Cross-Platform Path Resolver for vAlgo Backtesting System
========================================================

Intelligent path resolution utility that automatically detects project root
and resolves database paths correctly on Windows, WSL, and Linux.

Features:
- Auto-detects project root by searching for key marker files/folders
- Handles both absolute and relative paths from config
- Cross-platform compatibility (Windows/WSL/Linux)
- Single source of truth for all path resolution

Author: vAlgo Development Team
Created: September 1, 2025
"""

import os
from pathlib import Path
from typing import Optional, Union


class PathResolver:
    """
    Intelligent cross-platform path resolver for vAlgo backtesting system.
    
    Automatically detects project root and resolves paths correctly
    regardless of execution directory or operating system.
    """
    
    _project_root_cache: Optional[Path] = None
    
    @classmethod
    def get_project_root(cls) -> Path:
        """
        Intelligently detect project root by searching for key marker files.
        
        Searches upward from current file location for:
        - config/ directory with config.json
        - database/ directory  
        - main_* python files
        
        Returns:
            Path object pointing to project root
            
        Raises:
            RuntimeError: If project root cannot be detected
        """
        if cls._project_root_cache is not None:
            return cls._project_root_cache
            
        # Start from current file's directory
        current_path = Path(__file__).parent
        
        # Search upward for project markers
        for parent in [current_path] + list(current_path.parents):
            # Look for key project markers
            config_dir = parent / "config"
            database_dir = parent / "database" 
            main_files = list(parent.glob("main_*.py"))
            
            # Project root should have config/ and database/ directories
            if (config_dir.exists() and config_dir.is_dir() and 
                database_dir.exists() and database_dir.is_dir()):
                cls._project_root_cache = parent
                return parent
                
            # Alternative: Look for main files + config directory
            if config_dir.exists() and config_dir.is_dir() and len(main_files) > 0:
                cls._project_root_cache = parent  
                return parent
        
        # If nothing found, raise error
        raise RuntimeError(
            "Could not detect project root. Expected to find 'config/' and 'database/' "
            "directories or 'config/' with main_*.py files in project hierarchy."
        )
    
    @classmethod
    def resolve_database_path(cls, config_path: Union[str, Path]) -> str:
        """
        Intelligently resolve database path from config value.
        
        Handles both absolute and relative paths cross-platform:
        - Absolute paths: Returns as-is
        - Relative paths: Resolves from auto-detected project root
        
        Args:
            config_path: Database path from config.json
            
        Returns:
            Absolute path string that works on current platform
            
        Raises:
            RuntimeError: If path cannot be resolved or doesn't exist
        """
        if not config_path:
            raise ValueError("Database path cannot be empty")
            
        config_path = Path(config_path)
        
        # If already absolute, use as-is
        if config_path.is_absolute():
            if not config_path.exists():
                raise RuntimeError(f"Absolute database path does not exist: {config_path}")
            return str(config_path)
        
        # For relative paths, resolve from project root
        project_root = cls.get_project_root()
        resolved_path = project_root / config_path
        
        if not resolved_path.exists():
            raise RuntimeError(
                f"Database not found at resolved path: {resolved_path}\n"
                f"Project root: {project_root}\n"
                f"Config path: {config_path}\n"
                f"Check that database file exists and config path is correct."
            )
            
        return str(resolved_path)
    
    @classmethod
    def resolve_config_path(cls, config_path: Union[str, Path]) -> str:
        """
        Resolve any config-based path (reports, exports, etc.) intelligently.
        
        Args:
            config_path: Path from any config setting
            
        Returns:
            Absolute path string resolved from project root if relative
        """
        if not config_path:
            return ""
            
        config_path = Path(config_path)
        
        # If already absolute, use as-is
        if config_path.is_absolute():
            return str(config_path)
        
        # For relative paths, resolve from project root
        project_root = cls.get_project_root()
        resolved_path = project_root / config_path
        
        # Create directory if it doesn't exist (for output paths)
        resolved_path.parent.mkdir(parents=True, exist_ok=True)
        
        return str(resolved_path)
    
    @classmethod
    def get_database_path_from_config(cls, main_config: dict) -> str:
        """
        Get database path from main config with intelligent resolution.
        
        Args:
            main_config: Main configuration dictionary
            
        Returns:
            Resolved absolute database path
        """
        db_config = main_config.get('database', {})
        raw_path = db_config.get('path')
        
        if not raw_path:
            raise ValueError(
                "Database path not found in config. "
                "Add 'database.path' to config.json"
            )
        
        return cls.resolve_database_path(raw_path)