from typing import Tuple, List
from sklearn.metrics import mean_squared_error, accuracy_score, confusion_matrix, precision_score, recall_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .detector import Detector
from .utils import Utils


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


def save_as_new_data(data, columns, save_name):
    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_name)


if __name__ == '__main__':
    print("Loading data...")
    original_data = pd.read_csv("./data/creditcard.csv")
    print("Shuffling data...")
    shuffled_data = Utils.shuffle(df=original_data)
    valid = shuffled_data.loc[shuffled_data["Class"] == 0]
    fraudulent = shuffled_data.loc[shuffled_data["Class"] == 1]
    valid.drop(columns=["Class", "Amount", "Time"], inplace=True)
    fraudulent.drop(columns=["Class", "Amount", "Time"], inplace=True)

    data_columns = valid.columns

    print("Splitting data...")
    train_x, test_x = Utils.train_test_split(data=valid)

    print("Scaling data...")

    scaler, train_x, test_x = Utils.scale(train_x, test_x)
    scaled_fraud = scaler.transform(fraudulent)

    test_data = np.concatenate((test_x, np.zeros((test_x.shape[0], 1))), axis=1)
    scaled_fraud_data = np.concatenate((scaled_fraud, np.ones((scaled_fraud.shape[0], 1))), axis=1)

    combined_test = np.concatenate((test_data, scaled_fraud_data), axis=0)
    np.random.shuffle(combined_test)

    combined_test_x = combined_test[:, 0:-1]
    combined_test_y = combined_test[:, -1]

    print("Creating autoencoder model...")

    model = Detector()

    model.fit(x=train_x, epochs=10)

    # Training Transaction Predictions
    train_pred, train_sample_loss, train_loss_avg, train_loss_std = model.predict(x=train_x)

    # Combined (Fraudulent and Valid Transactions) Test Predictions
    combined_test_pred, combined_sample_loss, combined_loss_avg, combined_loss_std = model.predict(x=combined_test_x)

    # Valid Sample Transactions Test Predictions
    valid_test_pred, valid_sample_loss, valid_loss_avg, valid_loss_std = model.predict(x=test_x)

    fraud_pred, fraud_sample_loss, fraud_loss_avg, fraud_loss_std = model.predict(scaled_fraud)

    acc_train = accuracy_score(y_true=np.zeros(train_pred.shape[0]), y_pred=train_pred)
    acc_valid_test = accuracy_score(y_true=np.zeros(valid_test_pred.shape[0]), y_pred=valid_test_pred)
    acc_fraud = accuracy_score(y_true=np.ones(fraud_pred.shape), y_pred=fraud_pred)

    acc_test_combined = accuracy_score(y_true=combined_test_y, y_pred=combined_test_pred)
    precision_combined = precision_score(y_true=combined_test_y, y_pred=combined_test_pred)
    recall_combined = recall_score(y_true=combined_test_y, y_pred=combined_test_pred)

    print(f"Loss statistics:")
    print(f"Training ---- Average MSE: {train_loss_avg} - MSE Standard Deviation: {train_loss_std}")
    print(f"Valid Test ---- Average MSE: {valid_loss_avg} - MSE Standard Deviation: {valid_loss_std}")
    print(f"Combined Testing ---- Average MSE: {combined_loss_avg} - MSE Standard Deviation: {combined_loss_std}")
    print(f"Pure Fraud ---- Average MSE: {fraud_loss_avg} - MSE Standard Deviation: {fraud_loss_std}")
    print("\nAccuracies:")
    print(f"Training Accuracy: {acc_train}")
    print(f"Valid Test Accuracy: {acc_valid_test}")
    print(f"Pure Fraud Accuracy: {acc_fraud}")
    print(f"Combined Test Accuracy: {acc_test_combined}")
    print("\nTesting Metrics:")
    print(f"Precision: {precision_combined}")
    print(f"Recall: {recall_combined}")

    confusion_matrix_plot(y_true=combined_test_y, y_pred=combined_test_pred, title="Fraud Predictions", labels=["Valid", "Fraudulent"])

