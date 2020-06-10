from typing import Tuple, List
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score
from tensorflow import keras
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def shuffle(df):
    shuffled = df.copy()
    shuffled_indices = np.random.permutation(len(shuffled))
    return shuffled.iloc[shuffled_indices]


def train_test_split(data: pd.DataFrame, n_split: Tuple[float, float] = (0.95, 0.05)):
    n_train = round(n_split[0] * len(data))

    return data.iloc[:n_train], data.iloc[n_train:]


def scale(train, test):
    minmax_scaler = MinMaxScaler()
    # scaler = StandardScaler()
    minmax_scaler.fit(train)
    return (
        minmax_scaler,
        pd.DataFrame(data=minmax_scaler.transform(train), columns=train.columns),
        pd.DataFrame(data=minmax_scaler.transform(test), columns=test.columns)
    )


def loss_metrics(y_pred, y_true):
    mse_per_sample = np.square(y_pred - y_true).mean(axis=1)
    mse_avg = mse_per_sample.mean()
    mse_std = mse_per_sample.std()
    return mse_per_sample, mse_avg, mse_std


def classify(sample_mse, mse_avg, mse_std):
    predictions = np.zeros(sample_mse.shape)
    predictions[np.abs(sample_mse - mse_avg) > 0.8*mse_std] = 1
    # predictions[sample_mse > 2 * mse_avg] = 1
    return predictions


def confusion_matrix_plot(y_true: np.ndarray, y_pred: np.ndarray, title: str, labels: List[str]):
    conf_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    conf_mat = pd.DataFrame(conf_mat, index=labels, columns=labels)
    plt.figure(figsize=(10, 7))
    sns.set(font_scale=1.2)  # for label size
    sns.heatmap(conf_mat, annot=True, annot_kws={"size": 8}, fmt='g')  # font size
    plt.title(title, fontdict={"size": 20})
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


if __name__ == '__main__':
    print("Loading data...")
    original_data = pd.read_csv("./credit-card-data/creditcard.csv")
    print("Shuffling data...")
    shuffled_data = shuffle(df=original_data)
    valid = shuffled_data.loc[shuffled_data["Class"] == 0]
    fraudulent = shuffled_data.loc[shuffled_data["Class"] == 1]
    valid.drop(columns=["Class", "Amount", "Time"], inplace=True)
    fraudulent.drop(columns=["Class", "Amount", "Time"], inplace=True)
    print("Splitting data...")
    train_x, test_x = train_test_split(data=valid)
    # test_x = pd.concat([test_x, fraudulent])

    print("Scaling data...")

    scaler, train_x, test_x = scale(train_x, test_x)
    scaled_fraud = scaler.transform(fraudulent)

    test_data = np.concatenate((test_x, np.zeros((test_x.shape[0], 1))), axis=1)
    scaled_fraud_data = np.concatenate((scaled_fraud, np.ones((scaled_fraud.shape[0], 1))), axis=1)

    combined_test = np.concatenate((test_data, scaled_fraud_data), axis=0)
    np.random.shuffle(combined_test)

    combined_test_x = combined_test[:, 0:-1]
    combined_test_y = combined_test[:, -1]

    print("Creating autoencoder model...")

    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28,)))
    model.add(keras.layers.Dense(150, activation='tanh'))
    model.add(keras.layers.Dense(60, activation='tanh'))
    model.add(keras.layers.Dense(8, activation='tanh'))
    model.add(keras.layers.Dense(60, activation='tanh'))
    model.add(keras.layers.Dense(150, activation='tanh'))
    model.add(keras.layers.Dense(28, activation='tanh'))
    model.compile(optimizer="adagrad", loss="mse", metrics=["mse"])

    print(model.summary())

    my_callbacks = [
        keras.callbacks.ModelCheckpoint(filepath='./checkpoints/model.{epoch:02d}.h5'),
        keras.callbacks.TensorBoard(log_dir='./logs'),
    ]

    model.fit(x=train_x, y=train_x, epochs=10)  # , callbacks=my_callbacks)

    model.save("model_complete")

    # Train Predictions
    train_recon = model.predict(train_x)
    loss_per_sample_train, loss_avg_train, loss_std_train = loss_metrics(y_pred=train_recon, y_true=train_x)
    train_pred = classify(loss_per_sample_train, loss_avg_train, loss_std_train)

    # Valid Test Predictions
    test_recon = model.predict(combined_test_x)
    loss_per_sample_combined, loss_avg_combined, loss_std_combined = loss_metrics(y_pred=test_recon,
                                                                                  y_true=combined_test_x)
    combined_test_pred = classify(loss_per_sample_combined, loss_avg_train, loss_std_train)

    # Combined Test Predictions
    valid_test_recon = model.predict(test_x)
    loss_per_sample_valid, loss_avg_valid, loss_std_valid = loss_metrics(y_pred=valid_test_recon,
                                                                         y_true=test_x)
    valid_test_pred = classify(loss_per_sample_valid, loss_avg_train, loss_std_train)

    # Pure Fraud Predictions
    fraud_recon = model.predict(scaled_fraud)
    loss_per_sample_fraud, loss_avg_fraud, loss_std_fraud = loss_metrics(y_pred=fraud_recon, y_true=scaled_fraud)
    fraud_pred = classify(loss_per_sample_fraud, loss_avg_train, loss_std_train)

    acc_train = accuracy_score(y_true=np.zeros(train_pred.shape[0]), y_pred=train_pred)
    acc_valid_test = accuracy_score(y_true=np.zeros(valid_test_pred.shape[0]), y_pred=valid_test_pred)
    acc_fraud = accuracy_score(y_true=np.ones(fraud_pred.shape), y_pred=fraud_pred)

    acc_test_combined = accuracy_score(y_true=combined_test_y, y_pred=combined_test_pred)
    prec_combined = precision_score(y_true=combined_test_y, y_pred=combined_test_pred)
    rec_combined = recall_score(y_true=combined_test_y, y_pred=combined_test_pred)

    print(f"Loss statistics:")
    print(f"Training ---- Average MSE: {loss_avg_train} - MSE Standard Deviation: {loss_std_train}")
    print(f"Valid Test ---- Average MSE: {loss_avg_valid} - MSE Standard Deviation: {loss_std_valid}")
    print(f"Combined Testing ---- Average MSE: {loss_avg_combined} - MSE Standard Deviation: {loss_std_combined}")
    print(f"Pure Fraud ---- Average MSE: {loss_avg_fraud} - MSE Standard Deviation: {loss_std_fraud}")
    print("\nAccuracies:")
    print(f"Training Accuracy: {acc_train}")
    print(f"Valid Test Accuracy: {acc_valid_test}")
    print(f"Pure Fraud Accuracy: {acc_fraud}")
    print(f"Combined Test Accuracy: {acc_test_combined}")
    print("\nTesting Metrics:")
    print(f"Precision: {prec_combined}")
    print(f"Recall: {rec_combined}")

    confusion_matrix_plot(y_true=combined_test_y, y_pred=combined_test_pred, title="Fraud Predictions", labels=["Valid", "Fraudulent"])

    # print("Valid test")
    # test_recon = model.predict(x=test_x)
    # mse_valid = mean_squared_error(y_pred=test_recon, y_true=test_x)
    # print(mse_valid)
    #
    # print("Fraud test")
    # pred_fraud = model.predict(x=scaled_fraud)
    # mse_fraud = mean_squared_error(y_pred=pred_fraud, y_true=scaled_fraud)
    # print(mse_fraud)
