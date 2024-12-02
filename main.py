



from binance.client import Client
import pandas as pd
import pickle
import numpy as np
# from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score
# from sklearn.metrics import mean_squared_error
import time
import os
import utils
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Conv1D, MaxPooling1D, Flatten, TimeDistributed, Input, Dropout, LayerNormalization, MultiHeadAttention, Add
# from keras.optimizers import Adam
# from sklearn.multioutput import MultiOutputRegressor
# from lightgbm import LGBMRegressor
# import optuna
import json
from datetime import datetime, timezone
# from tcn import TCN
# import tensorflow as tf
# from sklearn.ensemble import RandomForestRegressor
import torch
# import torch.nn as nn
# from ta import add_all_ta_features
# from bisect import bisect_left
from sklearn.preprocessing import MinMaxScaler
import joblib
from threading import Thread, Lock
import train
import test
import test2


feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# fetch_and_store_data(symbol, interval=interval, delay=fetch_delay)
symbol = "BTCUSDT"
iterations_per_fetch = "1s"
fetch_delay = 1  # Continuous fetching every 1 second
training_interval = 1  # Retrain every 20 minutes
target_labels = ['future_bid_price', 'future_ask_price']#,'future_timestamp']
test_size=1000
data_lock = Lock()   
import pandas as pd
import time
from threading import Lock

data_lock = Lock()
all_data = pd.DataFrame()

def get_data(training_interval, delay, test_size=1000):
    """
    Load data incrementally and split into training and test sets.

    Args:
        training_interval (int): Time interval in seconds to check for new data.
        delay (int): Additional delay to simulate real-time data loading.
        test_size (int): Number of rows to reserve for the test set.

    Returns:
        tuple: A tuple of (train_data, test_data)
    """
    global all_data
    print("Fetching data...")
    output_file = "order_book_data_batch2.csv"
    required_rows = test_size + 1000  # Ensure enough rows for both training and testing

    all_data = pd.DataFrame()  # Initialize as an empty DataFrame

    while True:
        # Wait for the next training interval
        # time.sleep(training_interval)

        #with data_lock:  # Safely access shared data
            try:
                # Reload the updated file
                print("Reloading the CSV file...")
                latest_data = pd.read_csv(output_file).dropna()
                print(f"Data size before reload: {len(latest_data)} rows.")
                # Append new rows to `all_data`
                if all_data.empty:
                    all_data = latest_data
                else:
                    new_rows = latest_data.loc[~latest_data.index.isin(all_data.index)]
                    all_data = pd.concat([all_data, new_rows], ignore_index=True)

                # Manage memory by keeping only the last 12,000 rows
                if len(all_data) > 5000:
                    all_data = all_data.iloc[-5000:]
                    print("Dropped older rows to manage memory and train on the latest data.")

                print(f"Data size after reload: {len(all_data)} rows.")

                # Ensure enough rows for training and testing
                if len(all_data) < required_rows:
                    print(f"Not enough rows for training. Need at least {required_rows}, but have {len(all_data)}.")
                    continue

                # Split into training and test sets
                train_data = all_data.iloc[:-test_size]
                test_data = all_data.iloc[-test_size:]

                print(f"Train size: {len(train_data)} rows, Test size: {len(test_data)} rows.")
                return train_data, test_data

            except Exception as e:
                print(f"Error reading the CSV file: {e}")
                continue

def train_and_backtest(training_interval, fetch_delay, target_labels):
    """
    Periodically retrain models and run backtesting.

    Args:
        training_interval (int): Time interval for retraining in seconds.
        fetch_delay (int): Delay for fetching new data.
        target_labels (list): List of target labels.
    """
    print(" Training and backtesting thread started.")
    while True:
        with data_lock:
            try:
                # Get train and test data
                train_data, test_data = get_data(training_interval, fetch_delay, test_size)

               

                # Actual Model Training
                
                print(f"Retraining models with {len(train_data)} rows...")
                # Replace the following with your model training code
                train.retrain_models_periodically(target_labels, train_data, fetch_delay)
                #train.train_model(all_data,fetch_delay , target_labels, feature_scaler, target_scaler)
                # train.finetune_models_periodically(target_labels, train_data, fetch_delay)
                print("Model training complete.")

                # Backtesting after training
                print("Starting backtesting...")
                metrics, df = test.backtest(
                    symbol="BTCUSDT",
                    test_data=test_data,
                    iterations="1s",
                    delay=1,
                    target_labels=target_labels,
                    feature_scaler=feature_scaler,
                    target_scaler=target_scaler
                )
                print(f"Backtesting completed. Metrics: {metrics}")

                # Ensure re-fetching data and retraining happens only after backtesting
                print("Fetching new data for next cycle...")
                time.sleep(300)

            except Exception as e:
                print(f"Error during training or backtesting: {e}")
                continue
if __name__ == "__main__":
    train_thread = Thread(target=train_and_backtest, args=(training_interval, fetch_delay, target_labels))

    print("Starting training and backtesting thread...")
    train_thread.start()
    train_thread.join()
    print("Process completed.")