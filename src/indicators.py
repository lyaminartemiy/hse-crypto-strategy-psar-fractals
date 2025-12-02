import numpy as np
import pandas as pd


class Indicators:
    """Класс для расчета технических индикаторов"""

    @staticmethod
    def calculate_atr(high, low, close, period=14, sma_period=20):
        """Расчет Average True Range (ATR)"""
        high_low = high - low
        high_close = np.abs(high - pd.Series(close).shift())
        low_close = np.abs(low - pd.Series(close).shift())
        tr = np.maximum(high_low, np.maximum(high_close, low_close))

        atr = pd.Series(tr).rolling(period).mean()
        atr_sma = atr.rolling(sma_period).mean()
        high_vol = atr > atr_sma

        return atr, atr_sma, high_vol

    @staticmethod
    def calculate_psar(high, low, af_step=0.015, af_max=0.15):
        """Расчет Parabolic SAR"""
        length = len(high)
        sar = np.full(length, np.nan)
        af = np.full(length, af_step)
        trend = np.full(length, True, dtype=bool)
        ep = np.full(length, 0.0)

        if length == 0:
            return sar

        sar[0] = low[0]
        ep[0] = high[0]

        if length > 1:
            sar[1] = low[0]
            ep[1] = max(high[0], high[1])

        for i in range(2, length):
            sar[i] = sar[i - 1] + af[i - 1] * (ep[i - 1] - sar[i - 1])

            if trend[i - 1]:
                sar[i] = min(sar[i], low[i - 1], low[i - 2] if i > 2 else low[0])
                if low[i] <= sar[i]:
                    trend[i] = False
                    sar[i] = ep[i - 1]
                    ep[i] = low[i]
                    af[i] = af_step
                else:
                    trend[i] = True
                    ep[i] = max(ep[i - 1], high[i])
                    af[i] = min(af[i - 1] + 0.015, af_max)
            else:
                sar[i] = max(sar[i], high[i - 1], high[i - 2] if i > 2 else high[0])
                if high[i] >= sar[i]:
                    trend[i] = True
                    sar[i] = ep[i - 1]
                    ep[i] = high[i]
                    af[i] = af_step
                else:
                    trend[i] = False
                    ep[i] = min(ep[i - 1], low[i])
                    af[i] = min(af[i - 1] + 0.015, af_max)

        return sar

    @staticmethod
    def calculate_fractals(high, low):
        """Расчет фракталов"""
        fractal_high = (
            (high > high.shift(2))
            & (high > high.shift(1))
            & (high > high.shift(-1))
            & (high > high.shift(-2))
        ).fillna(False)

        fractal_low = (
            (low < low.shift(2))
            & (low < low.shift(1))
            & (low < low.shift(-1))
            & (low < low.shift(-2))
        ).fillna(False)

        return fractal_high, fractal_low
