import numpy as np
import pandas as pd
from tabulate import tabulate


class PerformanceMetrics:
    """Класс для расчета метрик производительности"""

    @staticmethod
    def calculate_all_metrics(returns):
        """Расчет всех метрик"""
        returns = returns.dropna()
        if len(returns) == 0:
            return {
                "total_return": 0,
                "annual_return": 0,
                "sharpe": 0,
                "sortino": 0,
                "max_drawdown": 0,
                "win_rate": 0,
                "volatility": 0,
                "profit_factor": 0,
            }

        # Общая доходность
        total_return = (returns + 1).prod() - 1

        # Годовая доходность
        annual_return = (1 + total_return) ** (252 / len(returns)) - 1

        # Волатильность
        volatility = returns.std() * np.sqrt(252)

        # Коэффициент Шарпа
        sharpe = (
            (returns.mean() * 252) / (returns.std() * np.sqrt(252))
            if returns.std() != 0
            else 0
        )

        # Коэффициент Сортино
        negative_returns = returns[returns < 0]
        sortino = (
            (returns.mean() * 252) / (negative_returns.std() * np.sqrt(252))
            if len(negative_returns) > 0
            else 0
        )

        # Максимальная просадка
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.cummax()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()

        # Win Rate
        win_rate = (returns > 0).mean()

        # Profit Factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss != 0 else np.inf

        return {
            "total_return": total_return * 100,
            "annual_return": annual_return * 100,
            "sharpe": sharpe,
            "sortino": sortino,
            "max_drawdown": max_drawdown * 100,
            "win_rate": win_rate * 100,
            "volatility": volatility * 100,
            "profit_factor": profit_factor,
        }

    @staticmethod
    def analyze_trades(trades_info):
        """Анализ сделок"""
        if trades_info["total_trades"] == 0:
            return None

        profits = []
        durations = []

        for i in range(len(trades_info["buy_dates"])):
            if i < len(trades_info["sell_dates"]):
                buy_price = trades_info["buy_prices"][i]
                sell_price = trades_info["sell_prices"][i]
                buy_date = trades_info["buy_dates"][i]
                sell_date = trades_info["sell_dates"][i]

                profit_pct = ((sell_price - buy_price) / buy_price) * 100
                duration = (sell_date - buy_date).days

                profits.append(profit_pct)
                durations.append(duration)

        if not profits:
            return None

        profits = np.array(profits)
        durations = np.array(durations)

        return {
            "total_trades": len(profits),
            "profitable_trades": np.sum(profits > 0),
            "losing_trades": np.sum(profits <= 0),
            "avg_profit": np.mean(profits),
            "median_profit": np.median(profits),
            "max_profit": np.max(profits),
            "max_loss": np.min(profits),
            "std_profit": np.std(profits),
            "avg_duration": np.mean(durations),
            "max_duration": np.max(durations),
            "min_duration": np.min(durations),
            "win_rate": (np.sum(profits > 0) / len(profits)) * 100,
        }

    @staticmethod
    def create_comparison_table(
        train_metrics, test_metrics, train_bh_metrics, test_bh_metrics
    ):
        """Создание сравнительной таблицы"""
        table = [
            ["Период", "Метрика", "Стратегия", "Buy & Hold", "Преимущество"],
            [
                "2022-2024",
                "Общая доходность %",
                f"{train_metrics['total_return']:.1f}",
                f"{train_bh_metrics['total_return']:.1f}",
                f"{train_metrics['total_return'] - train_bh_metrics['total_return']:+.1f}",
            ],
            [
                "2022-2024",
                "Годовая доходность %",
                f"{train_metrics['annual_return']:.1f}",
                f"{train_bh_metrics['annual_return']:.1f}",
                f"{train_metrics['annual_return'] - train_bh_metrics['annual_return']:+.1f}",
            ],
            [
                "2022-2024",
                "Sharpe Ratio",
                f"{train_metrics['sharpe']:.2f}",
                f"{train_bh_metrics['sharpe']:.2f}",
                f"{train_metrics['sharpe'] - train_bh_metrics['sharpe']:+.2f}",
            ],
            [
                "2022-2024",
                "Макс. просадка %",
                f"{abs(train_metrics['max_drawdown']):.1f}",
                f"{abs(train_bh_metrics['max_drawdown']):.1f}",
                f"{abs(train_bh_metrics['max_drawdown']) - abs(train_metrics['max_drawdown']):+.1f}",
            ],
            [
                "2025",
                "Общая доходность %",
                f"{test_metrics['total_return']:.1f}",
                f"{test_bh_metrics['total_return']:.1f}",
                f"{test_metrics['total_return'] - test_bh_metrics['total_return']:+.1f}",
            ],
            [
                "2025",
                "Sharpe Ratio",
                f"{test_metrics['sharpe']:.2f}",
                f"{test_bh_metrics['sharpe']:.2f}",
                f"{test_metrics['sharpe'] - test_bh_metrics['sharpe']:+.2f}",
            ],
            [
                "2025",
                "Win Rate %",
                f"{test_metrics['win_rate']:.1f}",
                f"{test_bh_metrics['win_rate']:.1f}",
                f"{test_metrics['win_rate'] - test_bh_metrics['win_rate']:+.1f}",
            ],
        ]

        return tabulate(table, headers="firstrow", tablefmt="grid", stralign="center")
