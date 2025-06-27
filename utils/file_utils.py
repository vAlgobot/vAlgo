"""
File handling utilities for vAlgo Trading System
"""

import os
import json
import pickle
from pathlib import Path
from typing import Any, Dict, Optional, Union
from datetime import datetime

def _safe_import_pandas():
    """Safely import pandas with fallback"""
    try:
        import pandas as pd
        return pd
    except ImportError:
        return None

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

def ensure_directory_exists(path: str) -> bool:
    """
    Ensure that a directory exists, create it if it doesn't
    
    Args:
        path: Directory path to check/create
        
    Returns:
        True if directory exists or was created, False otherwise
    """
    try:
        # Make path absolute if relative
        if not os.path.isabs(path):
            project_root = _get_project_root()
            path = str(project_root / path)
            
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except (OSError, PermissionError) as e:
        print(f"Error creating directory {path}: {e}")
        return False

def save_json(data: Dict[str, Any], file_path: str) -> bool:
    """
    Save data to JSON file
    
    Args:
        data: Data to save
        file_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        return True
    except Exception as e:
        print(f"Error saving JSON to {file_path}: {e}")
        return False

def load_json(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Load data from JSON file
    
    Args:
        file_path: Path to JSON file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading JSON from {file_path}: {e}")
        return None

def save_pickle(data: Any, file_path: str) -> bool:
    """
    Save data to pickle file
    
    Args:
        data: Data to save
        file_path: Path to save file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        ensure_directory_exists(os.path.dirname(file_path))
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        return True
    except Exception as e:
        print(f"Error saving pickle to {file_path}: {e}")
        return False

def load_pickle(file_path: str) -> Optional[Any]:
    """
    Load data from pickle file
    
    Args:
        file_path: Path to pickle file
        
    Returns:
        Loaded data or None if failed
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        print(f"Error loading pickle from {file_path}: {e}")
        return None

def save_dataframe(df, file_path: str, format: str = 'csv') -> bool:
    """
    Save DataFrame to file
    
    Args:
        df: DataFrame to save
        file_path: Path to save file
        format: File format ('csv', 'excel', 'parquet')
        
    Returns:
        True if successful, False otherwise
    """
    pd = _safe_import_pandas()
    if pd is None:
        print("pandas not available for DataFrame operations")
        return False
    
    try:
        # Make path absolute if relative
        if not os.path.isabs(file_path):
            project_root = _get_project_root()
            file_path = str(project_root / file_path)
            
        if not ensure_directory_exists(os.path.dirname(file_path)):
            return False
        
        if format.lower() == 'csv':
            df.to_csv(file_path, index=False)
        elif format.lower() == 'excel':
            df.to_excel(file_path, index=False)
        elif format.lower() == 'parquet':
            df.to_parquet(file_path, index=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
        return True
    except Exception as e:
        print(f"Error saving DataFrame to {file_path}: {e}")
        return False

def load_dataframe(file_path: str, format: Optional[str] = None) -> Optional[Any]:
    """
    Load DataFrame from file
    
    Args:
        file_path: Path to file
        format: File format (auto-detected if None)
        
    Returns:
        Loaded DataFrame or None if failed
    """
    pd = _safe_import_pandas()
    if pd is None:
        print("pandas not available for DataFrame operations")
        return None
    
    try:
        # Make path absolute if relative
        if not os.path.isabs(file_path):
            project_root = _get_project_root()
            file_path = str(project_root / file_path)
            
        if not os.path.exists(file_path):
            print(f"File not found: {file_path}")
            return None
        
        if format is None:
            # Auto-detect format from extension
            ext = Path(file_path).suffix.lower()
            if ext == '.csv':
                format = 'csv'
            elif ext in ['.xlsx', '.xls']:
                format = 'excel'
            elif ext == '.parquet':
                format = 'parquet'
            else:
                raise ValueError(f"Cannot auto-detect format for {file_path}")
        
        if format.lower() == 'csv':
            return pd.read_csv(file_path)
        elif format.lower() == 'excel':
            return pd.read_excel(file_path)
        elif format.lower() == 'parquet':
            return pd.read_parquet(file_path)
        else:
            raise ValueError(f"Unsupported format: {format}")
            
    except Exception as e:
        print(f"Error loading DataFrame from {file_path}: {e}")
        return None

def get_timestamp_filename(base_name: str, extension: str = '.csv') -> str:
    """
    Generate filename with timestamp
    
    Args:
        base_name: Base filename
        extension: File extension
        
    Returns:
        Filename with timestamp
    """
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{base_name}_{timestamp}{extension}"

def get_latest_file(directory: str, pattern: str = '*') -> Optional[str]:
    """
    Get the most recently modified file in a directory
    
    Args:
        directory: Directory to search
        pattern: File pattern to match
        
    Returns:
        Path to latest file or None if none found
    """
    try:
        files = list(Path(directory).glob(pattern))
        if not files:
            return None
        
        latest_file = max(files, key=os.path.getmtime)
        return str(latest_file)
    except Exception as e:
        print(f"Error finding latest file in {directory}: {e}")
        return None

def cleanup_old_files(directory: str, pattern: str = '*', keep_count: int = 10) -> int:
    """
    Clean up old files, keeping only the most recent ones
    
    Args:
        directory: Directory to clean
        pattern: File pattern to match
        keep_count: Number of files to keep
        
    Returns:
        Number of files deleted
    """
    try:
        files = list(Path(directory).glob(pattern))
        if len(files) <= keep_count:
            return 0
        
        # Sort by modification time (newest first)
        files.sort(key=os.path.getmtime, reverse=True)
        
        # Delete older files
        deleted_count = 0
        for file_path in files[keep_count:]:
            try:
                file_path.unlink()
                deleted_count += 1
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")
        
        return deleted_count
    except Exception as e:
        print(f"Error cleaning up files in {directory}: {e}")
        return 0

def backup_file(file_path: str, backup_dir: str = 'backups') -> Optional[str]:
    """
    Create a backup of a file
    
    Args:
        file_path: Path to file to backup
        backup_dir: Directory to store backup
        
    Returns:
        Path to backup file or None if failed
    """
    try:
        if not os.path.exists(file_path):
            return None
        
        ensure_directory_exists(backup_dir)
        
        # Generate backup filename with timestamp
        original_path = Path(file_path)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_name = f"{original_path.stem}_{timestamp}{original_path.suffix}"
        backup_path = Path(backup_dir) / backup_name
        
        # Copy file to backup location
        import shutil
        shutil.copy2(file_path, backup_path)
        
        return str(backup_path)
    except Exception as e:
        print(f"Error backing up {file_path}: {e}")
        return None