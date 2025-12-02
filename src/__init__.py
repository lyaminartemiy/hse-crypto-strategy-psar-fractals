import warnings

warnings.filterwarnings("ignore")

from src.config import Config
from src.data_loader import DataLoader
from src.indicators import Indicators
from src.metrics import PerformanceMetrics
from src.optimizer import ParameterOptimizer
from src.strategy import TradingStrategy
from src.visualizer import StrategyVisualizer

__version__ = "1.0.0"
__author__ = "Lyamin Artemiy"
__all__ = [
    "DataLoader",
    "Indicators",
    "TradingStrategy",
    "ParameterOptimizer",
    "PerformanceMetrics",
    "StrategyVisualizer",
    "Config",
]
