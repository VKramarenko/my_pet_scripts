import pandas as pd
import numpy as np

from .strategy_base import StrategyBase


class RsiStochStrategy(StrategyBase):
    def __init__(self, 
                 rsi_period: int = 14,
                 stoch_k: int = 14,
                 stoch_d: int = 3,
                 rsi_oversold: float = 30.0,
                 rsi_overbought: float = 70.0,
                 stoch_oversold: float = 20.0,
                 stoch_overbought: float = 80.0,
                 qty: float = 1.0):
        super().__init__()
        self.rsi_period = rsi_period
        self.stoch_k = stoch_k
        self.stoch_d = stoch_d
        self.rsi_oversold = rsi_oversold
        self.rsi_overbought = rsi_overbought
        self.stoch_oversold = stoch_oversold
        self.stoch_overbought = stoch_overbought
        self.qty = qty

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # RSI
        delta = df["close"].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=self.rsi_period, min_periods=1).mean()
        avg_loss = loss.rolling(window=self.rsi_period, min_periods=1).mean()
        rs = avg_gain / avg_loss.replace(0, np.nan)
        df["rsi"] = 100 - (100 / (1 + rs))

        # Stochastic
        low_min = df["low"].rolling(window=self.stoch_k, min_periods=1).min()
        high_max = df["high"].rolling(window=self.stoch_k, min_periods=1).max()
        df["stoch_k"] = 100 * (df["close"] - low_min) / (high_max - low_min)
        df["stoch_d"] = df["stoch_k"].rolling(window=self.stoch_d, min_periods=1).mean()

        return df

    def generate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.calculate_indicators(df)
        df["signal"] = 0

        # Условия входа
        rsi_condition = df["rsi"] < self.rsi_oversold
        stoch_condition = (df["stoch_k"] < self.stoch_oversold) & (df["stoch_k"] > df["stoch_d"])
        df.loc[rsi_condition & stoch_condition, "signal"] = 1

        # Условия выхода
        rsi_exit = df["rsi"] > self.rsi_overbought
        stoch_exit = (df["stoch_k"] > self.stoch_overbought) & (df["stoch_k"] < df["stoch_d"])
        df.loc[rsi_exit & stoch_exit, "signal"] = -1

        return df

    def on_fill(self, timestamp: pd.Timestamp, fill: dict, position: dict) -> list:
        # Обработка событий на заполнение ордеров (можно добавить логику после заполнения)
        return []

    def on_snapshot(self, timestamp: pd.Timestamp, snapshot: dict, position: dict) -> list:
        # Обработка обновлений L2-книги ордеров и торговли
        return []

    def on_tick(self, timestamp: pd.Timestamp, price_data: dict, position: dict) -> list:
        # Подразумеваем, что у нас уже есть полные бары (внутри `run_backtest`)
        # Здесь просто возвращаем None — сигналы вычисляются в `generate_signal`
        return []
