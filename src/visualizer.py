import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

from src.config import Config


class StrategyVisualizer:
    """Класс для визуализации результатов стратегии"""

    def __init__(self, colors=None):
        plt.style.use(Config.PLOT_STYLE)
        plt.rcParams["font.size"] = Config.FONT_SIZE
        self.colors = Config.COLORS if colors is None else colors

    def plot_indicators_and_signals(self, df, title_suffix=""):
        """График 1: Цена и индикаторы"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 12))

        # График 1.1: Цена и PSAR
        ax1.plot(
            df.index,
            df["Close"],
            color="black",
            linewidth=1.2,
            label="Цена BTC",
            alpha=0.8,
        )

        # PSAR точки
        bullish_mask = df["PSAR_Trend"] == 1
        bearish_mask = df["PSAR_Trend"] == -1
        ax1.scatter(
            df.index[bullish_mask],
            df["PSAR"][bullish_mask],
            color=self.colors["bullish"],
            s=10,
            marker="^",
            label="PSAR Бычий",
            alpha=0.6,
        )
        ax1.scatter(
            df.index[bearish_mask],
            df["PSAR"][bearish_mask],
            color=self.colors["bearish"],
            s=10,
            marker="v",
            label="PSAR Медвежий",
            alpha=0.6,
        )

        # Fractals
        fractal_high_dates = df.index[df["Fractal_High"]]
        fractal_low_dates = df.index[df["Fractal_Low"]]
        ax1.scatter(
            fractal_high_dates,
            df.loc[fractal_high_dates, "High"],
            color=self.colors["fractal_high"],
            s=60,
            marker="v",
            label="Фрактал High",
            zorder=5,
        )
        ax1.scatter(
            fractal_low_dates,
            df.loc[fractal_low_dates, "Low"],
            color=self.colors["fractal_low"],
            s=60,
            marker="^",
            label="Фрактал Low",
            zorder=5,
        )

        # Торговые сигналы
        buy_signals = df[df["Buy_Signal"]]
        sell_signals = df[df["Sell_Signal"] & (df["Position"].shift(1) == 1)]

        if not buy_signals.empty:
            ax1.scatter(
                buy_signals.index,
                buy_signals["Close"],
                color=self.colors["buy"],
                s=80,
                marker="^",
                edgecolor="black",
                linewidth=1.5,
                label="ПОКУПКА",
                zorder=10,
            )

        if not sell_signals.empty:
            ax1.scatter(
                sell_signals.index,
                sell_signals["Close"],
                color=self.colors["sell"],
                s=80,
                marker="v",
                edgecolor="black",
                linewidth=1.5,
                label="ПРОДАЖА",
                zorder=10,
            )

        ax1.set_title(
            f"BTC-USD - Цена с индикаторами PSAR и Фракталы {title_suffix}",
            fontsize=12,
            fontweight="bold",
        )
        ax1.set_ylabel("Цена ($)")
        ax1.legend(loc="upper left", fontsize=8)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # График 1.2: ATR и волатильность
        ax2.plot(
            df.index, df["ATR"], color=self.colors["atr"], linewidth=1.5, label="ATR"
        )
        ax2.plot(
            df.index,
            df["ATR_SMA"],
            color=self.colors["atr_sma"],
            linewidth=1.5,
            linestyle="--",
            label="ATR SMA",
        )

        ax2.fill_between(
            df.index,
            df["ATR"],
            df["ATR_SMA"],
            where=df["ATR"] > df["ATR_SMA"],
            color="red",
            alpha=0.2,
            label="Высокая волатильность",
        )
        ax2.fill_between(
            df.index,
            df["ATR"],
            df["ATR_SMA"],
            where=df["ATR"] <= df["ATR_SMA"],
            color="green",
            alpha=0.2,
            label="Низкая волатильность",
        )

        ax2.set_title(
            "Average True Range (ATR) - Индикатор волатильности",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_ylabel("ATR")
        ax2.legend(loc="upper left", fontsize=8)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        # График 1.3: Позиция
        ax3.fill_between(
            df.index,
            0,
            df["Position"],
            where=df["Position"] == 1,
            color="green",
            alpha=0.4,
            label="В позиции (LONG)",
        )
        ax3.set_ylim(-0.1, 1.1)
        ax3.set_yticks([0, 1])
        ax3.set_yticklabels(["OUT", "IN"])
        ax3.set_title("Торговая позиция", fontsize=12, fontweight="bold")
        ax3.set_ylabel("Позиция")
        ax3.grid(True, alpha=0.3)
        ax3.tick_params(axis="x", rotation=45)

        # График 1.4: Ежедневные возвраты стратегии
        strategy_returns = df["Strategy_Return"].dropna() * 100
        if len(strategy_returns) > 0:
            colors = ["green" if x > 0 else "red" for x in strategy_returns]
            ax4.bar(
                strategy_returns.index,
                strategy_returns,
                color=colors,
                alpha=0.6,
                width=0.8,
            )

        ax4.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax4.set_title(
            "Ежедневная доходность стратегии (%)", fontsize=12, fontweight="bold"
        )
        ax4.set_ylabel("Доходность %")
        ax4.grid(True, alpha=0.3)
        ax4.tick_params(axis="x", rotation=45)

        plt.suptitle(
            f"ГРАФИК 1: Анализ индикаторов и торговых сигналов {title_suffix}",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        return fig

    def plot_strategy_comparison(self, df_train, df_test, df_full, test_start):
        """График 2: Сравнение стратегий"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 12))

        # График 2.1: Общее сравнение стратегий
        ax1.plot(
            df_train.index,
            df_train["Strategy_Cum"],
            color=self.colors["strategy"],
            linewidth=2.5,
            label="Стратегия (Train 2022-2024)",
        )
        ax1.plot(
            df_test.index,
            df_test["Strategy_Cum"],
            color=self.colors["strategy_test"],
            linewidth=2.5,
            label="Стратегия (Test 2025)",
        )
        ax1.plot(
            df_full.index,
            df_full["BuyHold_Cum"],
            color=self.colors["buyhold"],
            linewidth=2,
            linestyle="--",
            label="Buy & Hold",
        )

        ax1.axvline(
            x=pd.to_datetime(test_start),
            color="red",
            linestyle="--",
            linewidth=2,
            label="Train/Test Split",
        )

        # Заполнение областей
        ax1.fill_between(
            df_train.index,
            df_train["Strategy_Cum"].min(),
            df_train["Strategy_Cum"],
            color=self.colors["strategy"],
            alpha=0.1,
        )
        ax1.fill_between(
            df_test.index,
            df_test["Strategy_Cum"].min(),
            df_test["Strategy_Cum"],
            color=self.colors["strategy_test"],
            alpha=0.1,
        )

        ax1.set_title(
            "Сравнение роста капитала: Оптимизированная стратегия vs Buy & Hold",
            fontsize=13,
            fontweight="bold",
        )
        ax1.set_ylabel("Рост капитала (база = 100)", fontsize=11)
        ax1.legend(loc="upper left", fontsize=9)
        ax1.grid(True, alpha=0.3)
        ax1.tick_params(axis="x", rotation=45)

        # Добавляем аннотации
        train_end_value = df_train["Strategy_Cum"].iloc[-1]
        test_end_value = df_test["Strategy_Cum"].iloc[-1]
        bh_end_value = df_full["BuyHold_Cum"].iloc[-1]

        ax1.annotate(
            f"Train: {train_end_value:.0f}%",
            xy=(df_train.index[-1], train_end_value),
            xytext=(-80, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=self.colors["strategy"], alpha=0.8
            ),
        )

        ax1.annotate(
            f"Test: {test_end_value:.0f}%",
            xy=(df_test.index[-1], test_end_value),
            xytext=(-60, 10),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=self.colors["strategy_test"],
                alpha=0.8,
            ),
        )

        ax1.annotate(
            f"B&H: {bh_end_value:.0f}%",
            xy=(df_full.index[-1], bh_end_value),
            xytext=(-50, -25),
            textcoords="offset points",
            fontsize=10,
            fontweight="bold",
            bbox=dict(
                boxstyle="round,pad=0.3", facecolor=self.colors["buyhold"], alpha=0.8
            ),
        )

        # График 2.2: Детализация последних сделок
        detail_start = df_test.index[-500] if len(df_test) > 500 else df_test.index[0]
        df_detail = df_test[df_test.index >= detail_start].copy()

        ax2.plot(
            df_detail.index,
            df_detail["Close"],
            color="black",
            linewidth=2,
            alpha=0.8,
            label="Цена",
        )

        # PSAR на детальном графике
        detail_bullish_mask = df_detail["PSAR_Trend"] == 1
        detail_bearish_mask = df_detail["PSAR_Trend"] == -1

        ax2.scatter(
            df_detail.index[detail_bullish_mask],
            df_detail["PSAR"][detail_bullish_mask],
            color="green",
            s=40,
            marker="^",
            alpha=0.5,
            label="PSAR Бычий",
        )
        ax2.scatter(
            df_detail.index[detail_bearish_mask],
            df_detail["PSAR"][detail_bearish_mask],
            color="red",
            s=40,
            marker="v",
            alpha=0.5,
            label="PSAR Медвежий",
        )

        ax2.set_title(
            "Детализация последних данных с PSAR", fontsize=13, fontweight="bold"
        )
        ax2.set_ylabel("Цена ($)", fontsize=11)
        ax2.legend(loc="upper left", fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.tick_params(axis="x", rotation=45)

        plt.suptitle(
            "ГРАФИК 2: Сравнение стратегий и детализация торгов",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        return fig

    def plot_performance_metrics(
        self,
        df_full,
        trades_info,
        train_metrics,
        test_metrics,
        train_bh_metrics,
        test_bh_metrics,
    ):
        """График 3: Метрики и статистика"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        # График 3.1: Sharpe Ratio сравнение
        metrics_labels = ["Train Стратегия", "Test Стратегия", "Train B&H", "Test B&H"]
        sharpe_values = [
            train_metrics["sharpe"],
            test_metrics["sharpe"],
            train_bh_metrics["sharpe"],
            test_bh_metrics["sharpe"],
        ]
        sortino_values = [
            train_metrics["sortino"],
            test_metrics["sortino"],
            train_bh_metrics["sortino"],
            test_bh_metrics["sortino"],
        ]

        x = np.arange(len(metrics_labels))
        width = 0.35

        bars1 = ax1.bar(
            x - width / 2,
            sharpe_values,
            width,
            label="Sharpe Ratio",
            color="steelblue",
            alpha=0.7,
        )
        bars2 = ax1.bar(
            x + width / 2,
            sortino_values,
            width,
            label="Sortino Ratio",
            color="lightcoral",
            alpha=0.7,
        )

        ax1.set_xlabel("Период")
        ax1.set_ylabel("Значение")
        ax1.set_title(
            "Сравнение коэффициентов Sharpe и Sortino", fontsize=12, fontweight="bold"
        )
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics_labels, rotation=15)
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis="y")

        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax1.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    height + 0.02,
                    f"{height:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                )

        # График 3.2: Доходность и просадка
        return_values = [
            train_metrics["annual_return"],
            test_metrics["annual_return"],
            train_bh_metrics["annual_return"],
            test_bh_metrics["annual_return"],
        ]
        drawdown_values = [
            abs(train_metrics["max_drawdown"]),
            abs(test_metrics["max_drawdown"]),
            abs(train_bh_metrics["max_drawdown"]),
            abs(test_bh_metrics["max_drawdown"]),
        ]

        bars3 = ax2.bar(
            x - width / 2,
            return_values,
            width,
            label="Годовая доходность %",
            color="forestgreen",
            alpha=0.7,
        )
        bars4 = ax2.bar(
            x + width / 2,
            drawdown_values,
            width,
            label="Макс. просадка %",
            color="crimson",
            alpha=0.7,
        )

        ax2.set_xlabel("Период")
        ax2.set_ylabel("Проценты (%)")
        ax2.set_title(
            "Годовая доходность vs Максимальная просадка",
            fontsize=12,
            fontweight="bold",
        )
        ax2.set_xticks(x)
        ax2.set_xticklabels(metrics_labels, rotation=15)
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis="y")

        # График 3.3: Распределение доходностей
        strategy_pos_returns = (
            df_full["Strategy_Return"][df_full["Strategy_Return"] > 0].dropna() * 100
        )
        strategy_neg_returns = (
            df_full["Strategy_Return"][df_full["Strategy_Return"] <= 0].dropna() * 100
        )
        bh_pos_returns = (
            df_full["BuyHold_Return"][df_full["BuyHold_Return"] > 0].dropna() * 100
        )
        bh_neg_returns = (
            df_full["BuyHold_Return"][df_full["BuyHold_Return"] <= 0].dropna() * 100
        )

        bins = np.linspace(-15, 15, 31)
        ax3.hist(
            strategy_pos_returns,
            bins=bins,
            alpha=0.5,
            color="green",
            label="Стратегия +",
            density=True,
        )
        ax3.hist(
            strategy_neg_returns,
            bins=bins,
            alpha=0.5,
            color="red",
            label="Стратегия -",
            density=True,
        )
        ax3.hist(
            bh_pos_returns,
            bins=bins,
            alpha=0.3,
            color="blue",
            label="B&H +",
            density=True,
            histtype="step",
            linewidth=2,
        )
        ax3.hist(
            bh_neg_returns,
            bins=bins,
            alpha=0.3,
            color="darkblue",
            label="B&H -",
            density=True,
            histtype="step",
            linewidth=2,
        )

        ax3.set_xlabel("Доходность (%)")
        ax3.set_ylabel("Плотность")
        ax3.set_title(
            "Распределение ежедневных доходностей", fontsize=12, fontweight="bold"
        )
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # График 3.4: Круговая диаграмма сделок
        if trades_info["total_trades"] > 0:
            profits = []
            for i in range(len(trades_info["buy_dates"])):
                if i < len(trades_info["sell_dates"]):
                    buy_price = trades_info["buy_prices"][i]
                    sell_price = trades_info["sell_prices"][i]
                    profit_pct = ((sell_price - buy_price) / buy_price) * 100
                    profits.append(profit_pct)

            if profits:
                profits = np.array(profits)
                profitable_trades = np.sum(profits > 0)
                break_even_trades = np.sum(profits == 0)
                losing_trades = np.sum(profits < 0)

                sizes = [profitable_trades, losing_trades, break_even_trades]
                labels = [
                    f"Прибыльные\n{profitable_trades}",
                    f"Убыточные\n{losing_trades}",
                    f"Безубыточные\n{break_even_trades}",
                ]
                colors = ["lightgreen", "lightcoral", "lightgray"]
                explode = (0.1, 0.05, 0) if profitable_trades > 0 else (0, 0.1, 0)

                ax4.pie(
                    sizes,
                    explode=explode,
                    labels=labels,
                    colors=colors,
                    autopct="%1.1f%%",
                    shadow=True,
                    startangle=90,
                )
                ax4.axis("equal")
                ax4.set_title(
                    f"Распределение сделок\nВсего: {trades_info['total_trades']}",
                    fontsize=12,
                    fontweight="bold",
                )
            else:
                ax4.text(
                    0.5,
                    0.5,
                    "Нет завершенных сделок",
                    ha="center",
                    va="center",
                    fontsize=12,
                    fontweight="bold",
                )
                ax4.axis("off")
        else:
            ax4.text(
                0.5,
                0.5,
                "Сделок не совершено",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
            )
            ax4.axis("off")

        plt.suptitle(
            "ГРАФИК 3: Статистика производительности и метрики качества",
            fontsize=14,
            fontweight="bold",
            y=0.98,
        )
        plt.tight_layout()
        return fig

    def plot_trade_details(self, df_test, trades_info):
        """Детальный график сделок на тестовых данных"""
        if trades_info["total_trades"] == 0:
            return None

        fig, ax = plt.subplots(figsize=(15, 8))

        # Берем последние 60 дней для детализации
        detail_start = df_test.index[-60] if len(df_test) > 60 else df_test.index[0]
        df_detail = df_test[df_test.index >= detail_start].copy()

        # Цена
        ax.plot(
            df_detail.index,
            df_detail["Close"],
            color="black",
            linewidth=2,
            alpha=0.8,
            label="Цена BTC",
        )

        # Отмечаем сделки в этом периоде
        for i in range(len(trades_info["buy_dates"])):
            buy_date = trades_info["buy_dates"][i]
            buy_price = trades_info["buy_prices"][i]

            if buy_date >= detail_start:
                # Проверяем есть ли продажа
                if i < len(trades_info["sell_dates"]):
                    sell_date = trades_info["sell_dates"][i]
                    sell_price = trades_info["sell_prices"][i]

                    if sell_date <= df_detail.index[-1]:
                        # Покупаем
                        ax.scatter(
                            buy_date,
                            buy_price,
                            color="lime",
                            s=150,
                            marker="^",
                            edgecolor="black",
                            linewidth=2,
                            zorder=10,
                        )
                        # Продаем
                        ax.scatter(
                            sell_date,
                            sell_price,
                            color="red",
                            s=150,
                            marker="v",
                            edgecolor="black",
                            linewidth=2,
                            zorder=10,
                        )

                        # Линия сделки
                        line_color = "green" if sell_price > buy_price else "red"
                        ax.plot(
                            [buy_date, sell_date],
                            [buy_price, sell_price],
                            color=line_color,
                            linewidth=3,
                            alpha=0.6,
                            zorder=5,
                        )

                        # Подпись прибыли
                        profit_pct = ((sell_price - buy_price) / buy_price) * 100
                        label_color = "darkgreen" if profit_pct > 0 else "darkred"
                        ax.text(
                            buy_date,
                            buy_price * 0.98,
                            f"{profit_pct:+.1f}%",
                            fontsize=9,
                            fontweight="bold",
                            color=label_color,
                            bbox=dict(
                                boxstyle="round,pad=0.2", facecolor="white", alpha=0.9
                            ),
                        )

        # Добавляем PSAR
        detail_bullish_mask = df_detail["PSAR_Trend"] == 1
        detail_bearish_mask = df_detail["PSAR_Trend"] == -1
        ax.scatter(
            df_detail.index[detail_bullish_mask],
            df_detail["PSAR"][detail_bullish_mask],
            color="green",
            s=40,
            marker="^",
            alpha=0.5,
            label="PSAR Бычий",
        )
        ax.scatter(
            df_detail.index[detail_bearish_mask],
            df_detail["PSAR"][detail_bearish_mask],
            color="red",
            s=40,
            marker="v",
            alpha=0.5,
            label="PSAR Медвежий",
        )

        ax.set_title(
            "Детализация сделок на тестовых данных (последние 60 дней)",
            fontsize=14,
            fontweight="bold",
        )
        ax.set_ylabel("Цена ($)", fontsize=11)
        ax.legend(loc="upper left", fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis="x", rotation=45)

        plt.tight_layout()
        return fig
