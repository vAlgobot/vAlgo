#!/usr/bin/env python3
"""
Options Chain Data Loader for vAlgo Trading System
=================================================

Professional data loader for options chain CSV files with comprehensive
validation, duplicate prevention, and batch processing capabilities.

Features:
- CSV parsing with automatic format detection
- Data validation and quality checks
- Batch processing for large files
- Progress tracking and detailed logging
- Integration with OptionsDatabase
- Duplicate prevention and incremental loading

Author: vAlgo Development Team
Created: July 10, 2025
Version: 1.0.0 (Production)
"""

import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import logging
import re
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from utils.logger import get_logger
from data_manager.options_database import OptionsDatabase, create_options_database


class OptionsDataLoader:
    """
    Professional options chain data loader with validation and batch processing.
    
    Handles CSV file parsing, data validation, and database loading with
    comprehensive error handling and progress tracking.
    """
    
    def __init__(self, options_db: Optional[OptionsDatabase] = None):
        """
        Initialize Options Data Loader.
        
        Args:
            options_db: Optional OptionsDatabase instance
        """
        self.logger = get_logger(__name__)
        self.options_db = options_db or create_options_database()
        
        # Configuration for data processing
        self.batch_size = 10000  # Records per batch
        self.validation_rules = self._setup_validation_rules()
        
        self.logger.info("OptionsDataLoader initialized successfully")
    
    def _setup_validation_rules(self) -> Dict[str, Any]:
        """Setup data validation rules for options chain data."""
        return {
            'price_range': {'min': 0.01, 'max': 10000},  # Premium range
            'iv_range': {'min': 0.1, 'max': 200},        # IV percentage range
            'delta_range': {'min': -1.0, 'max': 1.0},    # Delta range
            'dte_range': {'min': 0, 'max': 365},         # Days to expiry range
            'strike_step': 50,                           # Expected strike step
            'required_columns': [
                'Timestamp', 'Strike', 'Call LTP', 'Put LTP',
                'IV', 'Delta', 'IV1', 'Delta1', 'Expiry'
            ]
        }
    
    def load_csv_file(self, file_path: str, data_source: Optional[str] = None,
                     chunk_size: Optional[int] = None, validate_data: bool = True) -> Dict[str, Any]:
        """
        Load options chain data from CSV file.
        
        Args:
            file_path: Path to CSV file
            data_source: Source identifier for the data
            chunk_size: Number of records to process at once
            validate_data: Whether to perform data validation
            
        Returns:
            Dict with loading results and statistics
        """
        try:
            self.logger.info(f"Starting to load options data from: {file_path}")
            
            # Validate file exists
            if not Path(file_path).exists():
                raise FileNotFoundError(f"Options CSV file not found: {file_path}")
            
            # Auto-detect data source if not provided
            if data_source is None:
                data_source = f"CSV_{Path(file_path).stem}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            
            # Set batch size
            batch_size = chunk_size or self.batch_size
            
            # Load and process data
            start_time = time.time()
            
            # Read CSV with proper parsing
            raw_data = self._read_csv_file(file_path)
            
            if raw_data.empty:
                return {"status": "no_data", "message": "CSV file is empty"}
            
            self.logger.info(f"Raw CSV data loaded: {len(raw_data)} records")
            
            # Clean and validate data
            cleaned_data = self._clean_and_validate_data(raw_data, validate_data)
            
            if cleaned_data.empty:
                return {"status": "validation_failed", "message": "No valid data after cleaning"}
            
            self.logger.info(f"Data after cleaning: {len(cleaned_data)} records")
            
            # Process in batches
            total_processed = 0
            total_duplicates = 0
            batch_results = []
            
            for batch_start in range(0, len(cleaned_data), batch_size):
                batch_end = min(batch_start + batch_size, len(cleaned_data))
                batch_data = cleaned_data.iloc[batch_start:batch_end].copy()
                
                self.logger.info(f"Processing batch {batch_start//batch_size + 1}: "
                               f"records {batch_start}-{batch_end}")
                
                # Store batch in database
                batch_result = self.options_db.store_options_data(
                    batch_data, data_source=data_source, prevent_duplicates=True
                )
                
                batch_results.append(batch_result)
                total_processed += batch_result.get('records_processed', 0)
                total_duplicates += batch_result.get('duplicates_found', 0)
            
            # Calculate processing time
            processing_time = time.time() - start_time
            
            # Generate comprehensive result
            result = {
                "status": "success",
                "file_path": file_path,
                "data_source": data_source,
                "total_input_records": len(raw_data),
                "total_valid_records": len(cleaned_data),
                "total_processed": total_processed,
                "total_duplicates": total_duplicates,
                "processing_time_seconds": round(processing_time, 2),
                "records_per_second": round(len(cleaned_data) / processing_time, 2),
                "batch_results": batch_results,
                "data_quality_summary": self._generate_quality_summary(cleaned_data)
            }
            
            self.logger.info(f"Options data loading completed: {total_processed} records processed, "
                           f"{total_duplicates} duplicates prevented in {processing_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error loading options CSV file: {e}")
            raise
    
    def _read_csv_file(self, file_path: str) -> pd.DataFrame:
        """Read and parse CSV file with automatic format detection and header handling."""
        try:
            # Try different encoding and separator combinations
            encodings = ['utf-8', 'utf-8-sig', 'latin-1', 'cp1252']
            separators = [',', ';', '\t']
            
            data = None
            headers_detected = False
            
            for encoding in encodings:
                for sep in separators:
                    try:
                        # First try with headers
                        data = pd.read_csv(file_path, encoding=encoding, sep=sep, low_memory=False)
                        
                        # Check if we have the expected columns
                        if len(data.columns) >= 8:  # Minimum expected columns
                            # Check if first row contains expected headers
                            expected_headers = ['Timestamp', 'IV', 'Delta', 'Call LTP', 'Strike', 'Put LTP', 'Delta1', 'IV1', 'Expiry']
                            
                            if any(header in str(data.columns) for header in expected_headers):
                                headers_detected = True
                                self.logger.info(f"Successfully parsed CSV with headers using encoding={encoding}, sep='{sep}'")
                            else:
                                # Try without headers (headerless CSV)
                                data = pd.read_csv(file_path, encoding=encoding, sep=sep, header=None, low_memory=False)
                                
                                if len(data.columns) >= 9:  # Should have 9 columns for options data
                                    # Add standard headers for headerless CSV
                                    data.columns = ['Timestamp', 'IV', 'Delta', 'Call LTP', 'Strike', 'Put LTP', 'Delta1', 'IV1', 'Expiry']
                                    headers_detected = False
                                    self.logger.info(f"Successfully parsed headerless CSV using encoding={encoding}, sep='{sep}' and added headers")
                                else:
                                    continue
                            break
                    except Exception:
                        continue
                
                if data is not None and len(data.columns) >= 8:
                    break
            
            if data is None or len(data.columns) < 8:
                raise ValueError("Could not parse CSV file with any encoding/separator combination")
            
            self.logger.info(f"CSV file parsed successfully. Headers detected: {headers_detected}, Columns: {list(data.columns)}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error reading CSV file: {e}")
            raise
    
    def _clean_and_validate_data(self, data: pd.DataFrame, validate: bool = True) -> pd.DataFrame:
        """Clean and validate options chain data."""
        try:
            # Make a copy for processing
            cleaned_data = data.copy()
            
            # Standardize column names
            cleaned_data = self._standardize_column_names(cleaned_data)
            
            # Parse timestamps
            cleaned_data = self._parse_timestamps(cleaned_data)
            
            # Parse expiry dates and calculate DTE
            cleaned_data = self._parse_expiry_dates(cleaned_data)
            
            # Clean numeric data
            cleaned_data = self._clean_numeric_data(cleaned_data)
            
            # Remove invalid records
            if validate:
                cleaned_data = self._validate_data_quality(cleaned_data)
            
            # Ensure required columns exist
            cleaned_data = self._ensure_required_columns(cleaned_data)
            
            return cleaned_data
            
        except Exception as e:
            self.logger.error(f"Error cleaning and validating data: {e}")
            raise
    
    def _standardize_column_names(self, data: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to match database schema."""
        # Column mapping from CSV to database format
        column_mapping = {
            'Timestamp': 'timestamp',
            'Strike': 'strike',
            'Call LTP': 'call_ltp',
            'Put LTP': 'put_ltp',
            'IV': 'call_iv',
            'Delta': 'call_delta',
            'IV1': 'put_iv',
            'Delta1': 'put_delta',
            'Expiry': 'expiry_info'
        }
        
        # Rename columns
        data = data.rename(columns=column_mapping)
        
        # Log column mapping
        mapped_columns = [col for col in column_mapping.values() if col in data.columns]
        self.logger.debug(f"Standardized columns: {mapped_columns}")
        
        return data
    
    def _parse_timestamps(self, data: pd.DataFrame) -> pd.DataFrame:
        """Parse timestamp column to datetime format."""
        try:
            # Clean timestamp strings first
            data['timestamp'] = data['timestamp'].astype(str).str.strip()
            
            # Try different timestamp formats with both with and without comma
            timestamp_formats = [
                '%m/%d/%Y, %H:%M:%S',   # "1/1/2024, 09:15:00"
                '%m/%d/%Y %H:%M:%S',    # "1/1/2024 09:15:00"
                '%d/%m/%Y, %H:%M:%S',   # "1/1/2024, 09:15:00" (day first)
                '%d/%m/%Y %H:%M:%S',    # "1/1/2024 09:15:00" (day first)
                '%Y-%m-%d %H:%M:%S',    # "2024-01-01 09:15:00"
                '%d-%m-%Y %H:%M:%S',    # "01-01-2024 09:15:00"
                '%Y/%m/%d %H:%M:%S',    # "2024/01/01 09:15:00"
                '%Y/%m/%d, %H:%M:%S',   # "2024/01/01, 09:15:00"
            ]
            
            parsed_timestamp = None
            successful_format = None
            
            # First try with errors='coerce' to handle mixed formats
            for fmt in timestamp_formats:
                try:
                    parsed_timestamp = pd.to_datetime(data['timestamp'], format=fmt, errors='coerce')
                    # Check if most records were parsed successfully
                    success_rate = parsed_timestamp.notna().sum() / len(data)
                    if success_rate > 0.8:  # If 80% or more records parsed successfully
                        successful_format = fmt
                        self.logger.info(f"Successfully parsed {success_rate:.1%} of timestamps with format: {fmt}")
                        break
                except Exception:
                    continue
            
            if parsed_timestamp is None or successful_format is None:
                # Try automatic parsing as fallback with mixed format handling
                try:
                    parsed_timestamp = pd.to_datetime(data['timestamp'], format='mixed', errors='coerce')
                    self.logger.info("Used automatic mixed format timestamp parsing")
                except Exception:
                    # Final fallback - try to infer
                    parsed_timestamp = pd.to_datetime(data['timestamp'], errors='coerce')
                    self.logger.info("Used automatic timestamp parsing with error coercion")
            
            data['timestamp'] = parsed_timestamp
            
            # Log how many records failed to parse
            failed_count = data['timestamp'].isna().sum()
            if failed_count > 0:
                self.logger.warning(f"Failed to parse {failed_count} timestamps out of {len(data)} records")
            
            # Remove records with invalid timestamps
            data = data.dropna(subset=['timestamp'])
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error parsing timestamps: {e}")
            raise
    
    def _parse_expiry_dates(self, data: pd.DataFrame) -> pd.DataFrame:
        """Parse expiry information and calculate DTE."""
        try:
            # Extract expiry date from expiry_info column
            # Format: "04-Jan-24 (3 DTE )"
            expiry_pattern = r'(\d{2}-\w{3}-\d{2})'
            dte_pattern = r'\((\d+)\s+DTE\s*\)'
            
            # Extract expiry date
            expiry_matches = data['expiry_info'].str.extract(expiry_pattern)
            data['expiry_date_str'] = expiry_matches[0]
            
            # Parse expiry date
            data['expiry_date'] = pd.to_datetime(data['expiry_date_str'], format='%d-%b-%y', errors='coerce')
            
            # Extract DTE
            dte_matches = data['expiry_info'].str.extract(dte_pattern)
            data['dte'] = pd.to_numeric(dte_matches[0], errors='coerce')
            
            # Calculate DTE if not provided (backup method)
            mask = data['dte'].isna()
            if mask.any():
                # Convert to datetime if not already
                if not pd.api.types.is_datetime64_any_dtype(data['expiry_date']):
                    data['expiry_date'] = pd.to_datetime(data['expiry_date'])
                
                # Calculate days difference
                dte_calculated = (data['expiry_date'] - data['timestamp']).dt.days
                data.loc[mask, 'dte'] = dte_calculated.loc[mask]
            
            # Remove temporary columns
            data = data.drop(['expiry_info', 'expiry_date_str'], axis=1, errors='ignore')
            
            # Remove records with invalid expiry data
            data = data.dropna(subset=['expiry_date', 'dte'])
            
            self.logger.debug(f"Parsed expiry data for {len(data)} records")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error parsing expiry dates: {e}")
            raise
    
    def _clean_numeric_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """Clean and validate numeric columns."""
        try:
            numeric_columns = ['strike', 'call_ltp', 'put_ltp', 'call_iv', 'call_delta', 'put_iv', 'put_delta', 'dte']
            
            for col in numeric_columns:
                if col in data.columns:
                    # Convert to numeric, replacing errors with NaN
                    data[col] = pd.to_numeric(data[col], errors='coerce')
            
            # Log data types
            self.logger.debug(f"Numeric columns cleaned: {numeric_columns}")
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error cleaning numeric data: {e}")
            raise
    
    def _validate_data_quality(self, data: pd.DataFrame) -> pd.DataFrame:
        """Validate data quality based on business rules."""
        try:
            initial_count = len(data)
            rules = self.validation_rules
            
            # Validate price ranges
            price_mask = (
                (data['call_ltp'] >= rules['price_range']['min']) &
                (data['call_ltp'] <= rules['price_range']['max']) &
                (data['put_ltp'] >= rules['price_range']['min']) &
                (data['put_ltp'] <= rules['price_range']['max'])
            )
            
            # Validate IV ranges
            iv_mask = (
                (data['call_iv'] >= rules['iv_range']['min']) &
                (data['call_iv'] <= rules['iv_range']['max']) &
                (data['put_iv'] >= rules['iv_range']['min']) &
                (data['put_iv'] <= rules['iv_range']['max'])
            )
            
            # Validate Delta ranges
            delta_mask = (
                (data['call_delta'] >= 0) &  # Call delta should be positive
                (data['call_delta'] <= 1) &
                (data['put_delta'] >= -1) &  # Put delta should be negative
                (data['put_delta'] <= 0)
            )
            
            # Validate DTE
            dte_mask = (
                (data['dte'] >= rules['dte_range']['min']) &
                (data['dte'] <= rules['dte_range']['max'])
            )
            
            # Validate strikes (should be multiples of strike step)
            strike_mask = (data['strike'] % rules['strike_step'] == 0)
            
            # Combine all validation masks
            valid_mask = price_mask & iv_mask & delta_mask & dte_mask & strike_mask
            
            # Filter valid data
            valid_data = data[valid_mask].copy()
            
            # Log validation results
            rejected_count = initial_count - len(valid_data)
            if rejected_count > 0:
                rejection_rate = (rejected_count / initial_count) * 100
                self.logger.warning(f"Data validation rejected {rejected_count} records "
                                 f"({rejection_rate:.1f}% of total)")
            
            return valid_data
            
        except Exception as e:
            self.logger.error(f"Error validating data quality: {e}")
            return data  # Return original data if validation fails
    
    def _ensure_required_columns(self, data: pd.DataFrame) -> pd.DataFrame:
        """Ensure all required columns exist with proper data types."""
        try:
            required_columns = {
                'timestamp': 'datetime64[ns]',
                'strike': 'int32',
                'expiry_date': 'datetime64[ns]',
                'call_ltp': 'float64',
                'call_iv': 'float64',
                'call_delta': 'float64',
                'put_ltp': 'float64',
                'put_iv': 'float64',
                'put_delta': 'float64',
                'dte': 'int32'
            }
            
            # Ensure all required columns exist
            for col, dtype in required_columns.items():
                if col not in data.columns:
                    raise ValueError(f"Required column missing: {col}")
                
                # Convert to proper data type
                try:
                    if dtype.startswith('datetime'):
                        data[col] = pd.to_datetime(data[col])
                    else:
                        data[col] = data[col].astype(dtype)
                except Exception as e:
                    self.logger.warning(f"Could not convert {col} to {dtype}: {e}")
            
            # Remove any completely null rows
            data = data.dropna(how='all')
            
            return data
            
        except Exception as e:
            self.logger.error(f"Error ensuring required columns: {e}")
            raise
    
    def _generate_quality_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Generate data quality summary statistics."""
        try:
            summary = {
                "total_records": len(data),
                "date_range": {
                    "start": data['timestamp'].min().isoformat() if not data.empty else None,
                    "end": data['timestamp'].max().isoformat() if not data.empty else None
                },
                "strike_range": {
                    "min": int(data['strike'].min()) if not data.empty else None,
                    "max": int(data['strike'].max()) if not data.empty else None,
                    "unique_strikes": int(data['strike'].nunique()) if not data.empty else 0
                },
                "expiry_dates": data['expiry_date'].nunique() if not data.empty else 0,
                "data_completeness": {
                    "call_ltp": (data['call_ltp'].notna().sum() / len(data) * 100) if not data.empty else 0,
                    "put_ltp": (data['put_ltp'].notna().sum() / len(data) * 100) if not data.empty else 0,
                    "call_iv": (data['call_iv'].notna().sum() / len(data) * 100) if not data.empty else 0,
                    "put_iv": (data['put_iv'].notna().sum() / len(data) * 100) if not data.empty else 0
                }
            }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Error generating quality summary: {e}")
            return {"error": str(e)}
    
    def load_multiple_files(self, file_paths: List[str], data_source_prefix: str = "CSV_BATCH") -> Dict[str, Any]:
        """
        Load multiple CSV files in batch.
        
        Args:
            file_paths: List of CSV file paths
            data_source_prefix: Prefix for data source naming
            
        Returns:
            Dict with batch loading results
        """
        try:
            self.logger.info(f"Starting batch loading of {len(file_paths)} files")
            
            batch_results = []
            total_processed = 0
            total_duplicates = 0
            start_time = time.time()
            
            for i, file_path in enumerate(file_paths):
                self.logger.info(f"Processing file {i+1}/{len(file_paths)}: {Path(file_path).name}")
                
                try:
                    data_source = f"{data_source_prefix}_{i+1:03d}_{Path(file_path).stem}"
                    result = self.load_csv_file(file_path, data_source=data_source)
                    
                    batch_results.append({
                        "file_path": file_path,
                        "result": result
                    })
                    
                    total_processed += result.get('total_processed', 0)
                    total_duplicates += result.get('total_duplicates', 0)
                    
                except Exception as e:
                    self.logger.error(f"Error processing file {file_path}: {e}")
                    batch_results.append({
                        "file_path": file_path,
                        "result": {"status": "error", "error": str(e)}
                    })
            
            processing_time = time.time() - start_time
            
            return {
                "status": "completed",
                "total_files": len(file_paths),
                "total_records_processed": total_processed,
                "total_duplicates_prevented": total_duplicates,
                "processing_time_seconds": round(processing_time, 2),
                "file_results": batch_results
            }
            
        except Exception as e:
            self.logger.error(f"Error in batch file loading: {e}")
            raise


# Convenience functions
def load_options_csv(file_path: str, options_db: Optional[OptionsDatabase] = None,
                    data_source: Optional[str] = None) -> Dict[str, Any]:
    """
    Convenience function to load a single options CSV file.
    
    Args:
        file_path: Path to CSV file
        options_db: Optional OptionsDatabase instance
        data_source: Optional data source identifier
        
    Returns:
        Loading results dictionary
    """
    loader = OptionsDataLoader(options_db)
    return loader.load_csv_file(file_path, data_source)


# Example usage and testing
if __name__ == "__main__":
    import time
    
    # Example usage of the Options Data Loader
    try:
        # Create loader
        loader = OptionsDataLoader()
        
        # Test file path (update with actual path)
        test_file = "outputs/data_exports/Option_Chain_Expired_Data.csv"
        
        if Path(test_file).exists():
            print(f"Loading options data from: {test_file}")
            
            result = loader.load_csv_file(test_file, data_source="TEST_LOAD")
            
            print(f"Loading completed:")
            print(f"  Status: {result['status']}")
            print(f"  Total records processed: {result.get('total_processed', 0)}")
            print(f"  Duplicates prevented: {result.get('total_duplicates', 0)}")
            print(f"  Processing time: {result.get('processing_time_seconds', 0)} seconds")
            print(f"  Records per second: {result.get('records_per_second', 0)}")
            
            # Show quality summary
            quality = result.get('data_quality_summary', {})
            print(f"\nData Quality Summary:")
            print(f"  Date range: {quality.get('date_range', {})}")
            print(f"  Strike range: {quality.get('strike_range', {})}")
            print(f"  Expiry dates: {quality.get('expiry_dates', 0)}")
            
        else:
            print(f"Test file not found: {test_file}")
            
    except Exception as e:
        print(f"Error in example usage: {e}")