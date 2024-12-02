
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

API_KEY = 'Tk336ytIFldrN1FAfxaS1OYcyzOFZ4Lie7VpctLyTmhGL4QlQxhB3bo9AnmjsEnD'
API_SECRET = 'fSAofJ8C4o8ErfgLMGocRJMrnZDbDPfQBzuD7iTr1cMHza2PwwUH63bo70LxmdHp'


client = Client(API_KEY, API_SECRET)


  

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
def train_model(data,delay, target_label,feature_scaler,target_scaler):

    
    
    # with open("features.txt", 'r') as file:
    #     feature_data= json.load(file)
    #     print("Features Data Loaded:")
    # features = feature_data['keep_features']
    data = data.copy()
    data=utils.compute_technical_indicators(data,delay)
    data=utils.add_targets(data,delay)
    print(data.shape)
    data = utils.preprocessing(data)
    data = utils.preprocess_datetime(data)
    print("j",data.shape)
    utils.check_null(data)
    data = utils.fill_null(data)
    data2=data.dropna().reset_index(drop=True)
    x, y =utils. x_y_split(data2,target_label)
    x = utils.feature_eng.basic_features(x)
    print("k",x.shape)
    x = utils.correlation_filter.filter(x)
    print(x.shape)
    # x = utils.feature_eng.lag_rolling_features(x)
    print(x.shape)
    x = x.ffill()  # Forward fill
    x = x.bfill()
    #print(x.isnull().sum())
    # print(x.head())
    x, y = utils.feature_eng.remove_na(x, y)
    print("na",x.shape)
    y = y[target_label]

    if os.path.exists("features2.txt"):
        with open("features2.txt", 'r') as file:
            feature_data = json.load(file)
            print("Features Data Loaded:")
            features = feature_data.get('keep_features', [])
            utils.feature.save(features, utils.correlation_filter.remove_cols)
    else:
        features = utils.feature_selection.select(x, y)
        utils.feature.save(features, utils.correlation_filter.remove_cols)
    # features = feature_selection.select(x, y)
    
    # features = feature.load()['keep_features']
    
    print("d",x.shape)
    # print(x.columns)
    # print(features)
    
    # if 'timestamp' in x.columns:
    #    x.drop(columns=['timestamp'], inplace=True)

    x = x[[i for i in features if i in x.columns]]
    # x=x[features]
    x = feature_scaler.fit_transform(x)
    y = target_scaler.fit_transform(y)
    # Prepare LSTM input
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl') 
    print(x.shape)
    timesteps =   50# Number of timesteps
    x_lstm,y_lstm = (utils.model.prepare_lstm_input(x, y,timesteps))
    print("shape of x_lstm",x_lstm.shape)
    print(y_lstm.shape)

    input_shape=(x_lstm.shape[1], x_lstm.shape[2])
    print("cnn_lstm")
    cnn_lstm_model=utils.model.cnn_lstm(x_lstm, y_lstm, input_shape=(x_lstm.shape[1], x_lstm.shape[2]), epochs=50, batch_size=32)
  
    
    print(input_shape)
    print("tcn_model")
    tcn_model= utils.model.tcn(x_lstm, y_lstm, input_shape=(x_lstm.shape[1], x_lstm.shape[2]), epochs=50, batch_size=32)
    print("transformer_model")
    transformer_model = utils.model.transformer(x_lstm, y_lstm, input_shape=(x_lstm.shape[1], x_lstm.shape[2]), epochs=50, batch_size=32)
    print("lstm_model")
    lstm_model = utils.model.lstm(x_lstm, y_lstm, input_shape=(x_lstm.shape[1], x_lstm.shape[2]))

    

        # Scale target values
    print("lightgbm")
    lightgbm = utils.model.lightgbm(x, y)
    print("rf")
    rf = utils.model.random_forest(x, y)
    print("xgboost")
    xg=utils.model.xgboost(x, y)

    torch.save(lstm_model.state_dict(), 'lstm_model_multi_output2.pth')
    torch.save(cnn_lstm_model.state_dict(), 'cnn_lstm_model_multi_output2.pth')
    torch.save(tcn_model.state_dict(), 'tcn_model_multi_output2.pth')
    torch.save(transformer_model.state_dict(), 'transformer_multi_output2.pth')

    joblib.dump(rf, 'rf_multi_output2.joblib')
    # Save the entire MultiOutputRegressor
    with open('xg_multi_output2.pkl', 'wb') as f:
        pickle.dump(xg, f)
    # xg.save_model('xg_multi_output2.model')
    
    joblib.dump(lightgbm, 'lgbm_multi_output2.joblib')
      

def fine_tune_model(data,delay, target_label,feature_scaler,target_scaler):

    
    
    # with open("features.txt", 'r') as file:
    #     feature_data= json.load(file)
    #     print("Features Data Loaded:")
    # features = feature_data['keep_features']
    data = data.copy()
    data=utils.compute_technical_indicators(data,delay)
    data=utils.add_targets(data,delay)
    print(data.shape)
    data = utils.preprocessing(data)
    data = utils.preprocess_datetime(data)
    print("j",data.shape)
    utils.check_null(data)
    data = utils.fill_null(data)
    data2=data.dropna().reset_index(drop=True)
    x, y =utils. x_y_split(data2,target_label)
    x = utils.feature_eng.basic_features(x)
    print("k",x.shape)
    x = utils.correlation_filter.filter(x)
    print(x.shape)
    x = utils.feature_eng.lag_rolling_features(x)
    print(x.shape)
    x = x.ffill()  # Forward fill
    x = x.bfill()
    #print(x.isnull().sum())
    # print(x.head())
    x, y = utils.feature_eng.remove_na(x, y)
    print("na",x.shape)
    y = y[target_label]

    if os.path.exists("features2.txt"):
        with open("features2.txt", 'r') as file:
            feature_data = json.load(file)
            print("Features Data Loaded:")
            features = feature_data.get('keep_features', [])
            utils.feature.save(features, utils.correlation_filter.remove_cols)
    else:
        features = utils.feature_selection.select(x, y)
        utils.feature.save(features, utils.correlation_filter.remove_cols)
    # features = feature_selection.select(x, y)
    
    # features = feature.load()['keep_features']
    
    print("d",x.shape)
    # print(x.columns)
    # print(features)
    
    # if 'timestamp' in x.columns:
    #    x.drop(columns=['timestamp'], inplace=True)

    x = x[[i for i in features if i in x.columns]]
    # x=x[features]
    x = feature_scaler.fit_transform(x)
    y = target_scaler.fit_transform(y)
    # Prepare LSTM input
    print(x.shape)
    timesteps =   50# Number of timesteps
    x_lstm,y_lstm = (utils.model.prepare_lstm_input(x, y,timesteps))
    print("shape of x_lstm",x_lstm.shape)
    print(y_lstm.shape)
    # Train LSTM
     # Train CNN-LSTM
    input_shape=(x_lstm.shape[1], x_lstm.shape[2])
    print("cnn_lstm")

      # Load pre-trained models
    lgbm = joblib.load('lgbm_multi_output2.joblib')
    rf = joblib.load('rf_multi_output2.joblib')
    xg = joblib.load('xg_multi_output2.joblib')

    # Load LSTM model
    lstm_model = utils.LSTMModel(input_size=x_lstm.shape[2], output_size=y_lstm.shape[1])
    lstm_model.load_state_dict(torch.load('lstm_model_multi_output2.pth'))
    lstm_model.eval()

    # Load CNN-LSTM model
    cnn_lstm_model = utils.CNNLSTMModel(input_size=x_lstm.shape[2], output_size=y_lstm.shape[1])
    cnn_lstm_model.load_state_dict(torch.load('cnn_lstm_model_multi_output2.pth'))
    cnn_lstm_model.eval()

    # Load TCN model
    tcn_model = utils.TCNModel(input_size=x_lstm.shape[2], output_size=y_lstm.shape[1], num_channels=[16, 32, 64])
    tcn_model.load_state_dict(torch.load('tcn_model_multi_output2.pth'))
    tcn_model.eval()

    # Load Transformer model
    transformer_model = utils.TransformerModel(input_size=x_lstm.shape[2], output_size=y_lstm.shape[1])
    transformer_model.load_state_dict(torch.load('transformer_multi_output2.pth'))
    transformer_model.eval()

    # Fine-tuning each model with additional data
    print("Fine-tuning CNN-LSTM model")
    cnn_lstm_model = utils.model.fine_tune_cnn_lstm(cnn_lstm_model, x_lstm, y_lstm, epochs=50, batch_size=32)

    print("Fine-tuning TCN model")
    tcn_model = utils.model.fine_tune_tcn(tcn_model, x_lstm, y_lstm, epochs=50, batch_size=32)

    print("Fine-tuning Transformer model")
    transformer_model = utils.model.fine_tune_transformer(transformer_model, x_lstm, y_lstm, epochs=50, batch_size=32)

    print("Fine-tuning LSTM model")
    lstm_model = utils.model.fine_tune_lstm(lstm_model, x_lstm, y_lstm, epochs=50, batch_size=32)

    # Fine-tune LightGBM
    print("Fine-tuning LightGBM")
    lightgbm = utils.model.fine_tune_lightgbm(lgbm, x, y)

    # Fine-tune Random Forest
    print("Fine-tuning Random Forest")
    rf = utils.model.fine_tune_random_forest(rf, x, y)

    # Fine-tune XGBoost
    print("Fine-tuning XGBoost")
    xg = utils.model.fine_tune_xgboost(xg, x, y)

    # Save the updated models
    torch.save(lstm_model.state_dict(), 'lstm_model_multi_output2_fine_tuned.pth')
    torch.save(cnn_lstm_model.state_dict(), 'cnn_lstm_model_multi_output2_fine_tuned.pth')
    torch.save(tcn_model.state_dict(), 'tcn_model_multi_output2_fine_tuned.pth')
    torch.save(transformer_model.state_dict(), 'transformer_multi_output2_fine_tuned.pth')

    joblib.dump(rf, 'rf_multi_output2_fine_tuned.joblib')
    joblib.dump(xg, 'xg_multi_output2_fine_tuned.joblib')
    joblib.dump(lightgbm, 'lgbm_multi_output2_fine_tuned.joblib')

        



# def fetch_data_continuously(symbol, interval, delay):





         

file_path = 'order_book_data_batch2.csv'

# Check if the file exists and load it, or create an empty DataFrame
if os.path.exists(file_path):
    all_data = pd.read_csv(file_path)
    print(f"Loaded data from {file_path}. Shape: {all_data.shape}")
else:
    all_data = pd.DataFrame()
    print(f"File {file_path} not found. Created an empty DataFrame.")    


data_lock = Lock()    





def retrain_models_periodically(target_label, train_data, delay):
        try:
            # Retrain models
            all_data = train_data
            print("Retraining models with updated data...")
            train_model(all_data, delay, target_label, feature_scaler, target_scaler)
            # fine_tune_model(all_data, delay, target_label, feature_scaler, target_scaler)
            print("Model retraining complete.")
        except Exception as e:
            print(f"Error during model retraining: {e}")

def finetune_models_periodically(target_label, train_data, delay):
        try:
            # Retrain models
            all_data = train_data
            print("Retraining models with updated data...")
           # train_model(all_data, delay, target_label, feature_scaler, target_scaler)
            fine_tune_model(all_data, delay, target_label, feature_scaler, target_scaler)
            print("Model retraining complete.")
        except Exception as e:
            print(f"Error during model retraining: {e}")





