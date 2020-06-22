from tensorflow import keras
from typing import List, Optional
import numpy as np
import pandas as pd
from typing import Union


class Detector:
    def __init__(self, n_nodes: Optional[List[int]] = None):
        if n_nodes is None:
            n_nodes = [150, 60, 8, 60, 150]
        self.model = keras.Sequential()
        self.model.add(keras.layers.Input(shape=(28,)))
        for layer_nodes in n_nodes:
            self.model.add(keras.layers.Dense(layer_nodes, activation='tanh'))
        self.model.add(keras.layers.Dense(28, activation='tanh'))
        self.model.compile(optimizer="adagrad", loss="mse", metrics=["mse"])
        self.mse_avg_train = None
        self.mse_std_train = None
        print(self.model.summary())

    def fit(self, x: Union[np.ndarray, pd.DataFrame], epochs: int):
        self.model.fit(x=x, y=x, epochs=epochs)
        y = self.model.predict(x)
        mse_per_sample, self.mse_avg_train, self.mse_std_train = self.loss_metrics(y_pred=y, y_true=x)

    def predict(self, x: Union[np.ndarray, pd.DataFrame]):
        y = self.model.predict(x)
        mse_per_sample, mse_avg, mse_std = self.loss_metrics(y_pred=y, y_true=x)
        predictions = np.zeros(mse_per_sample.shape)
        predictions[np.abs(mse_per_sample - self.mse_avg_train) > 1 * self.mse_std_train] = 1
        return predictions, mse_per_sample, mse_avg, mse_std

    @staticmethod
    def loss_metrics(y_pred, y_true):
        mse_per_sample = np.square(y_pred - y_true).mean(axis=1)
        mse_avg = mse_per_sample.mean()
        mse_std = mse_per_sample.std()
        return mse_per_sample, mse_avg, mse_std
