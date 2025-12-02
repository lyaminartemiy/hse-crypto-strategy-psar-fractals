class Config:
    # Параметры стратегии по умолчанию
    DEFAULT_PARAMS = {
        "psar_af": 0.015,
        "psar_max": 0.15,
        "atr_period": 14,
        "atr_sma_period": 20,
    }

    # Настройки данных
    TICKER = "BTC-USD"
    TRAIN_START = "2022-01-01"
    TRAIN_END = "2024-12-31"
    TEST_START = "2025-01-01"

    # Настройки визуализации
    PLOT_STYLE = "seaborn-v0_8-darkgrid"
    FONT_SIZE = 10
    COLORS = {
        "bullish": "green",
        "bearish": "red",
        "buy": "lime",
        "sell": "red",
        "fractal_high": "darkred",
        "fractal_low": "darkgreen",
        "strategy": "darkgreen",
        "strategy_test": "lime",
        "buyhold": "blue",
        "atr": "purple",
        "atr_sma": "orange",
    }
