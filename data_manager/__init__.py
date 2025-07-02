"""
Data Manager Module for vAlgo Trading System

This module provides data management capabilities including:
- OpenAlgo API integration
- Market data fetching and storage
- Real-time data streaming
- Historical data management

Author: vAlgo Development Team
Created: June 27, 2025
"""

from .openalgo import OpenAlgo
from .market_data import MarketDataManager
from .database import DatabaseManager

__all__ = [
    'OpenAlgo',
    'MarketDataManager',
    'DatabaseManager'
]

__version__ = '1.0.0'
__author__ = 'vAlgo Development Team'