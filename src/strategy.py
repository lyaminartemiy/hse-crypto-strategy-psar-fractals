import numpy as np
import pandas as pd

from src.indicators import Indicators


class TradingStrategy:
    """Торговая стратегия на основе PSAR, фракталов и ATR"""

    def __init__(self, params=None):
        if params is None:
            from .config import Config

            params = Config.DEFAULT_PARAMS
        self.params = params
        self.indicators = Indicators()

    def generate_signals(self, df):
        """Генерация торговых сигналов"""
        df = df.copy()

        # Расчет индикаторов
        df["ATR"], df["ATR_SMA"], df["High_Vol"] = self.indicators.calculate_atr(
            df["High"],
            df["Low"],
            df["Close"],
            self.params["atr_period"],
            self.params["atr_sma_period"],
        )

        df["PSAR"] = self.indicators.calculate_psar(
            df["High"].values,
            df["Low"].values,
            self.params["psar_af"],
            self.params["psar_max"],
        )
        df["PSAR_Trend"] = np.where(df["Close"] > df["PSAR"], 1, -1)

        df["Fractal_High"], df["Fractal_Low"] = self.indicators.calculate_fractals(
            df["High"], df["Low"]
        )

        # Break Fractal Low
        prev_fractal_lows = df["Low"].where(df["Fractal_Low"]).shift(1)
        df["Break_Fractal_Low"] = (df["Close"] > prev_fractal_lows) & df[
            "Fractal_Low"
        ].shift(1)

        # Сигналы
        df["Buy_Signal"] = (
            (df["Close"] > df["PSAR"]) & df["High_Vol"] & df["Break_Fractal_Low"]
        )
        df["Sell_Signal"] = (df["Close"] < df["PSAR"]) | df["Fractal_High"]

        return df

    def execute_strategy(self, df):
        """Выполнение стратегии и расчет позиций"""
        df = self.generate_signals(df)

        # Применение стратегии
        position = 0
        positions = []
        trades = {
            "buy_dates": [],
            "buy_prices": [],
            "sell_dates": [],
            "sell_prices": [],
            "positions": [],
        }

        for i in range(len(df)):
            if df["Buy_Signal"].iloc[i] and position == 0:
                position = 1
                trades["buy_dates"].append(df.index[i])
                trades["buy_prices"].append(df["Close"].iloc[i])
            elif df["Sell_Signal"].iloc[i] and position == 1:
                position = 0
                trades["sell_dates"].append(df.index[i])
                trades["sell_prices"].append(df["Close"].iloc[i])
            positions.append(position)

        df["Position"] = positions
        trades["total_trades"] = len(trades["buy_dates"])

        # Расчет доходности
        df["Strategy_Return"] = df["Position"].shift(1) * df["Close"].pct_change()
        df["BuyHold_Return"] = df["Close"].pct_change()
        df["Strategy_Cum"] = (1 + df["Strategy_Return"]).cumprod().fillna(1) * 100
        df["BuyHold_Cum"] = (1 + df["BuyHold_Return"]).cumprod().fillna(1) * 100

        return df, trades

    def run(self, data_dict):
        """Запуск стратегии на всех данных"""
        results = {}
        trades = {}

        for key, df in data_dict.items():
            if len(df) > 0:
                df_result, trades_result = self.execute_strategy(df)
                results[key] = df_result
                trades[key] = trades_result

        return results, trades
