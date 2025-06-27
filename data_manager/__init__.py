"""
Data Manager Module for vAlgo Trading System
Handles data fetching, storage, and management
"""

from .openalgo_client import (
    OpenAlgoClient,
    TickData,
    HistoricalData,
    ConnectionStatus,
    create_client,
    test_openalgo_connection
)

from .openalgo_official_client import (
    OpenAlgoOfficialClient,
    create_official_client,
    test_official_connection
)

from .database_manager import (
    DatabaseManager,
    OHLCVRecord,
    MarketSession,
    create_database_manager,
    get_default_db_path
)

from .data_fetcher import (
    DataFetcher,
    FetchJob,
    FetchResult,
    FetchStatus,
    create_data_fetcher,
    fetch_historical_data_simple
)

from .enhanced_data_fetcher import (
    EnhancedDataFetcher,
    ClientType,
    create_enhanced_data_fetcher
)

__all__ = [
    # OpenAlgo Client (Custom)
    'OpenAlgoClient',
    'TickData',
    'HistoricalData',
    'ConnectionStatus',
    'create_client',
    'test_openalgo_connection',
    
    # OpenAlgo Client (Official)
    'OpenAlgoOfficialClient',
    'create_official_client',
    'test_official_connection',
    
    # Database Manager
    'DatabaseManager',
    'OHLCVRecord',
    'MarketSession',
    'create_database_manager',
    'get_default_db_path',
    
    # Data Fetcher
    'DataFetcher',
    'FetchJob',
    'FetchResult',
    'FetchStatus',
    'create_data_fetcher',
    'fetch_historical_data_simple',
    
    # Enhanced Data Fetcher
    'EnhancedDataFetcher',
    'ClientType',
    'create_enhanced_data_fetcher'
]