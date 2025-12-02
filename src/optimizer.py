import numpy as np
import optuna

from src.strategy import TradingStrategy


class ParameterOptimizer:
    """Оптимизация параметров стратегии"""

    def __init__(self, df_train):
        self.df_train = df_train

    def objective(self, trial):
        """Целевая функция для оптимизации"""
        params = {
            "psar_af": trial.suggest_float("psar_af", 0.01, 0.03),
            "psar_max": trial.suggest_float("psar_max", 0.1, 0.3),
            "atr_period": trial.suggest_int("atr_period", 10, 20),
            "atr_sma_period": trial.suggest_int("atr_sma_period", 15, 30),
        }

        strategy = TradingStrategy(params)
        df_result, _ = strategy.execute_strategy(self.df_train.copy())
        returns = df_result["Strategy_Return"].dropna()

        if len(returns) == 0 or returns.std() == 0:
            return -100

        sharpe = (returns.mean() * 252) / (returns.std() * np.sqrt(252))
        return sharpe

    def optimize(self, n_trials=50, show_progress=True):
        """Запуск оптимизации"""
        print("Оптимизация параметров стратегии...")

        try:
            study = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=42), direction="maximize")
            study.optimize(
                self.objective, n_trials=n_trials, show_progress_bar=show_progress
            )

            best_params = study.best_params
            print(f"Лучшие параметры: {best_params}")
            print(f"Лучший Sharpe Ratio: {study.best_value:.4f}")

            return best_params

        except ImportError:
            from .config import Config

            return Config.DEFAULT_PARAMS
