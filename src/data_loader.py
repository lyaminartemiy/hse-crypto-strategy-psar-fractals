import numpy as np
import pandas as pd
import yfinance as yf

from src.config import Config


class DataLoader:
    """Класс для загрузки и подготовки данных"""

    def __init__(self, ticker=Config.TICKER):
        self.ticker = ticker

    def download_data(self, period="4y", interval="1d"):
        """Загрузка данных с Yahoo Finance"""
        print(f"Загрузка данных для {self.ticker}...")
        data = yf.download(self.ticker, period=period, interval=interval)
        data.columns = [col[0] for col in data.columns]
        print(
            f"Данные загружены: {len(data)} дней "
            f"({data.index[0].date()} - {data.index[-1].date()})"
        )
        return data

    def prepare_data(
        self,
        data,
        train_start=Config.TRAIN_START,
        train_end=Config.TRAIN_END,
        test_start=Config.TEST_START,
    ):
        """Подготовка и разделение данных"""
        df = data.copy()

        # Разделение на обучающую и тестовую выборки
        df_train = df[(df.index >= train_start) & (df.index <= train_end)].copy()
        df_test = df[df.index >= test_start].copy()

        print("Разделение данных:")
        print(
            f"   Train: {len(df_train)} дней ({df_train.index[0].date()} - {df_train.index[-1].date()})"
        )
        print(
            f"   Test:  {len(df_test)} дней ({df_test.index[0].date()} - {df_test.index[-1].date()})"
        )

        return {"full": df, "train": df_train, "test": df_test}

    def get_data(self, period="4y", interval="1d"):
        """Полный процесс получения данных"""
        data = self.download_data(period, interval)
        return self.prepare_data(data)
