"""
Constants and enumerations for vAlgo Trading System
"""

from enum import Enum

# Trading Constants
class OrderSide(Enum):
    BUY = "BUY"
    SELL = "SELL"

class OrderType(Enum):
    MARKET = "MARKET"
    LIMIT = "LIMIT"
    STOP_LOSS = "STOP_LOSS"
    STOP_LOSS_LIMIT = "STOP_LOSS_LIMIT"

class OrderStatus(Enum):
    PENDING = "PENDING"
    FILLED = "FILLED"
    CANCELLED = "CANCELLED"
    REJECTED = "REJECTED"

class TradingMode(Enum):
    BACKTEST = "backtest"
    PAPER = "paper" 
    LIVE = "live"
    DATALOAD = "dataload"

# Timeframes
class Timeframe(Enum):
    ONE_MIN = "1min"
    FIVE_MIN = "5min"
    FIFTEEN_MIN = "15min"
    THIRTY_MIN = "30min"
    ONE_HOUR = "1h"
    FOUR_HOUR = "4h"
    ONE_DAY = "1d"

# Indicator Types
class IndicatorType(Enum):
    EMA = "EMA"
    SMA = "SMA"
    RSI = "RSI"
    VWAP = "VWAP"
    CPR = "CPR"
    SUPERTREND = "SUPERTREND"
    BOLLINGER_BANDS = "BB"
    MACD = "MACD"

# Configuration Constants
DEFAULT_CONFIG_FILE = "config/config.xlsx"
DEFAULT_DATABASE_FILE = "data/vAlgo.db"
DEFAULT_LOG_FILE = "logs/vAlgo.log"

# API Endpoints
OPENALGO_DEFAULT_URL = "http://localhost:5000"
OPENALGO_API_VERSION = "v1"

# Risk Management
DEFAULT_RISK_PER_TRADE = 0.02  # 2%
DEFAULT_MAX_DRAWDOWN = 0.20    # 20%
DEFAULT_CAPITAL = 100000       # 1 Lakh

# Performance Metrics
TRADING_DAYS_PER_YEAR = 252
RISK_FREE_RATE = 0.06  # 6% annual risk-free rate