from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


class Utils:
    @staticmethod
    def shuffle(df):
        shuffled = df.copy()
        shuffled_indices = np.random.permutation(len(shuffled))
        return shuffled.iloc[shuffled_indices]

    @staticmethod
    def train_test_split(data: pd.DataFrame, n_split: Tuple[float, float] = (0.95, 0.05)):
        n_train = round(n_split[0] * len(data))
        return data.iloc[:n_train], data.iloc[n_train:]

    @staticmethod
    def scale(train, test) -> Tuple[MinMaxScaler, pd.DataFrame, pd.DataFrame]:
        minmax_scaler = MinMaxScaler()
        minmax_scaler.fit(train)
        return (
            minmax_scaler,
            pd.DataFrame(data=minmax_scaler.transform(train), columns=train.columns),
            pd.DataFrame(data=minmax_scaler.transform(test), columns=test.columns)
        )

    @staticmethod
    def loss_metrics(y_pred, y_true):
        mse_per_sample = np.square(y_pred - y_true).mean(axis=1)
        mse_avg = mse_per_sample.mean()
        mse_std = mse_per_sample.std()
        return mse_per_sample, mse_avg, mse_std
