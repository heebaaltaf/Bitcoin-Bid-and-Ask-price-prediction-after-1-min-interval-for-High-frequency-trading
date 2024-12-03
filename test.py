# import torch.nn as nn

# from binance.client import Client
# import pandas as pd
# import numpy as np
# from xgboost import XGBClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score
# from sklearn.metrics import mean_squared_error
# import time
# import os
# from collections import defaultdict

# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.metrics import mean_squared_error
# from lightgbm import LGBMRegressor
# import optuna
# import json
# from datetime import datetime, timezone
# import numpy as np
# from tcn import TCN
# import tensorflow as tf

# from sklearn.ensemble import RandomForestRegressor
# import torch
# import torch.nn as nn
# from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error
# from lightgbm import LGBMRegressor
# import optuna
# from ta import add_all_ta_features
# import torch
# import numpy as np
# import joblib

# from bisect import bisect_left
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.model_selection import TimeSeriesSplit
# from sklearn.metrics import mean_squared_error

# from lightgbm import LGBMRegressor
# import optuna

# from sklearn.ensemble import RandomForestRegressor
# from sklearn.multioutput import MultiOutputRegressor

# from sklearn.model_selection import TimeSeriesSplit, cross_val_score
# import utils
# from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# #from evolutionary_search import EvolutionaryAlgorithmSearchCV
# from sklearn.model_selection import GridSearchCV
# import joblib
# from scipy.stats import mode


from collections import defaultdict
from binance.client import Client
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split, TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import time
import os
from sklearn.multioutput import MultiOutputRegressor
from lightgbm import LGBMRegressor
import optuna
import json
from datetime import datetime, timezone
# from tcn import TCN
import tensorflow as tf
from sklearn.ensemble import RandomForestRegressor
import torch
import sklearn
import xgboost
from ta import add_all_ta_features
import joblib
from bisect import bisect_left
from sklearn.preprocessing import MinMaxScaler
from scipy.stats import mode
import utils
import pickle

API_KEY = 'xxxxx'
API_SECRET = 'xxxxx'


client = Client(API_KEY, API_SECRET)

feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()
    
# # During prediction
# feature_scaler = joblib.load('feature_scaler.pkl')
# target_scaler = joblib.load('target_scaler.pkl')

all_data = pd.DataFrame()

def fetch_data_continuously(symbol, iterations_per_fetch, delay_between_fetches):
    global all_data
    output_file = "prediction_order_book_data_batch2.csv"
    while True:
        try:
            # Fetch new data
            new_data=utils.fetch_order_book_and_kline(symbol, interval=iterations_per_fetch, prev_bid_update_time=None, prev_ask_update_time=None)
            # print(new_data.shape)
            # new_data = fetch_order_book_and_trades(symbol=symbol, iterations=iterations_per_fetch, delay=1)
            # print(f"Fetched {len(new_data)} new rows of data.")
            
            # Append new data to all_data
            #with data_lock:  # Ensure thread-safe access
            if all_data.empty:
                all_data = new_data
            else:
                all_data = pd.concat([all_data, new_data], ignore_index=True)

            if not os.path.isfile(output_file):
                    all_data.to_csv(output_file, index=False)
            else:
                    all_data.to_csv(output_file, index=False)
                    # new_data.to_csv(output_file, mode="a", header=False, index=False)        

            # # Optional: Drop old data to manage memory (e.g., keep the last 10,000 rows)
            # if len(all_data) > 10000:
            #     all_data = all_data.iloc[-10000:]
            #     print("Dropped older rows to manage memory.")
            if len(all_data) >= 1000:
                print(f"Data limit reached: {len(all_data)} rows. Stopping fetch.")
                break

            # Wait before the next fetch
            # time.sleep(delay_between_fetches)
            time.sleep(delay_between_fetches)
        except KeyboardInterrupt:
            print("Terminating data fetching. Saving remaining data...")
            if new_data:
                df = pd.DataFrame(new_data)
                # print(df.shape)
                # df = compute_technical_indicators(df)
                print(df.shape)
                if not os.path.isfile(output_file):
                    df.to_csv(output_file, index=False)
                else:
                    df.to_csv(output_file, mode="a", header=False, index=False)
            print("All data saved successfully.")
        except Exception as e:
            print(f"Error in continuous data fetch and save: {e}")

def predict_test(data,delay, target_label, feature_scaler, target_scaler):
    """
    Returns predictions and ground truth for regression models.
    """
    # Load pre-saved features and correlation columns to remove
    if os.path.exists("features2.txt"):
        with open("features2.txt", 'r') as file:
            feature_data = json.load(file)
            print("Features Data Loaded:")
            features = feature_data.get('keep_features', [])
            correlation_remove = feature_data.get('correlation_remove',[])
    # features = feature.load()['keep_features']
    # correlation_remove = feature.load()['correlation_remove']
    print(correlation_remove)
    # Preprocess data
    data = data.copy()
    data=utils.compute_technical_indicators(data,delay)
    data=utils.add_targets(data,delay)
    data = utils.preprocessing(data)  # Placeholder function
    data = utils.preprocess_datetime(data)  # Placeholder function
    data = utils.fill_null(data)  # Placeholder function
    print(data.shape)
   
    data2=data.dropna().reset_index(drop=True)
    x, y = utils.x_y_split(data2,target_label)  # Placeholder function
    x = utils.feature_eng.basic_features(x)  # Placeholder function
    x = x.drop(correlation_remove, axis=1)
    x = utils.feature_eng.lag_rolling_features(x)  # Placeholder function
    x = x.ffill().bfill()
    x, y = utils.feature_eng.remove_na(x, y)  # Placeholder function
    y = y[target_label]
    x = x[[i for i in features if i in x.columns]]
    # x = x[features]
    print(x.shape)
    x = feature_scaler.fit_transform(x)
    y = target_scaler.fit_transform(y)

    # Prepare LSTM-like inputs for sequential models
    timesteps = 50
    x_lstm, y_lstm = utils.model.prepare_lstm_input(x, y, timesteps)
    input_shape=(x_lstm.shape[1], x_lstm.shape[2])
    # Load models
    lgbm = joblib.load('lgbm_multi_output2.joblib')
    rf = joblib.load('rf_multi_output2.joblib')
    # xg = joblib.load('xg_multi_output2.joblib')
    # xg = xgboost.Booster()
    # xg.load_model("xgboost_model.json")
    with open('xg_multi_output2.pkl', 'rb') as f:
       xg = pickle.load(f)
    lstm_model = utils.LSTMModel(input_size=x_lstm.shape[2], output_size=y_lstm.shape[1])
    lstm_model.load_state_dict(torch.load('lstm_model_multi_output2.pth'))
    lstm_model.eval()

    cnn_lstm_model = utils.CNNLSTMModel(input_size=x_lstm.shape[2], output_size=y_lstm.shape[1])
    cnn_lstm_model.load_state_dict(torch.load('cnn_lstm_model_multi_output2.pth'))
    cnn_lstm_model.eval()

    tcn_model = utils.TCNModel(input_size=x_lstm.shape[2], output_size=y_lstm.shape[1], num_channels=[16, 32, 64])
    tcn_model.load_state_dict(torch.load('tcn_model_multi_output2.pth'))
    tcn_model.eval()

    transformer_model = utils.TransformerModel(input_size=x_lstm.shape[2], output_size=y_lstm.shape[1])
    transformer_model.load_state_dict(torch.load('transformer_multi_output2.pth'))
    transformer_model.eval()

    # Make predictions
    lgbm_predict = lgbm.predict(x)
    lgbm_predict = target_scaler.inverse_transform(lgbm_predict)

    rf_predict = rf.predict(x)
    rf_predict = target_scaler.inverse_transform(rf_predict)

    xg_predict = xg.predict(x)
    xg_predict = target_scaler.inverse_transform(xg_predict)

    # PyTorch Model Predictions
    x_lstm_tensor = torch.tensor(x_lstm, dtype=torch.float32)

    with torch.no_grad():
        lstm_predict = lstm_model(x_lstm_tensor).numpy()
        cnn_lstm_predict = cnn_lstm_model(x_lstm_tensor).numpy()
        tcn_predict = tcn_model(x_lstm_tensor).numpy()
        transformer_predict = transformer_model(x_lstm_tensor).numpy()

    lstm_predict = target_scaler.inverse_transform(lstm_predict)
    cnn_lstm_predict = target_scaler.inverse_transform(cnn_lstm_predict)
    tcn_predict = target_scaler.inverse_transform(tcn_predict)
    transformer_predict = target_scaler.inverse_transform(transformer_predict)

    y_lstm = target_scaler.inverse_transform(y_lstm)
    y = target_scaler.inverse_transform(y)

    # Ensemble predictions (custom logic)
    ensemble_predict = (lgbm_predict + rf_predict) / 2

    return ensemble_predict, y, y_lstm, lgbm_predict, rf_predict, lstm_predict, cnn_lstm_predict, tcn_predict, transformer_predict,xg_predict

def process_predictions(
    predictions, ground_truth, ground_truth_lstm,
    lgbm_predict, rf_predict, lstm_predict,
    cnn_lstm_predict, tcn_predict, transformer_predict,xg_predict
):
    def process_array(array):
        """
        Process each prediction array:
        - Convert third element of each tuple to UTC datetime.
        - Round the remaining elements to integers.
        """
        processed = []
        for row in array:
           
            # Ensure it's an ndarray with at least three elements
            if isinstance(row, np.ndarray) and len(row) >= 3:
                # print(type(row))
                # Convert third element to a native float for compatibility
                timestamp = float(row[2])
                
                
                # Format the datetime object to a string (e.g., '2024-11-26 19:04:21')
                formatted_datetime = timestamp
                
                
                # Round the first two elements to integers and include the formatted datetime
                p = (round(row[0], 2), round(row[1], 2), formatted_datetime)

               # p=(int(np.round(row[0])), int(np.round(row[1])), formatted_datetime)
                processed.append(p)
            else:
                # Leave unchanged if it's not a valid ndarray
                # print("be")
                processed.append(row)
      
          
        return  np.array(processed, dtype=object)
       
     
    print(((process_array(predictions)))[0])
    # Process each array
    predictions1 = (process_array(predictions))

    ground_truth1 = (process_array(ground_truth))
    ground_truth_lstm1 =(process_array(ground_truth_lstm))
    lgbm_predict1 = (process_array(lgbm_predict))
    rf_predict1 = (process_array(rf_predict))
    xg_predict1 = (process_array(xg_predict))
    lstm_predict1 = (process_array(lstm_predict))
    cnn_lstm_predict1 = (process_array(cnn_lstm_predict))
    tcn_predict1 = (process_array(tcn_predict))
    transformer_predict1 = (process_array(transformer_predict))
    #print(predictions1[0], ground_truth1[0] ,ground_truth_lstm1[0], lgbm_predict1[0], rf_predict1[0], lstm_predict1[0], cnn_lstm_predict1[0], tcn_predict1[0], transformer_predict1[0])
    return (
        predictions1, ground_truth1, ground_truth_lstm1,
        lgbm_predict1, rf_predict1, lstm_predict1,
        cnn_lstm_predict1, tcn_predict1, transformer_predict1,xg_predict1
    )

def backtest(symbol,test_data, iterations=10, delay=1, target_labels=None, feature_scaler=None, target_scaler=None):
    """
    Perform backtesting by simulating predictions on live data.
    Args:
        symbol (str): Trading symbol (e.g., BTCUSDT).
        iterations (int): Number of iterations to fetch data.
        delay (int): Delay between fetches in seconds.
        target_labels (list): List of target labels to evaluate.
        feature_scaler (MinMaxScaler): Scaler used for input features.
        target_scaler (MinMaxScaler): Scaler used for target labels.
    Returns:
        dict: Backtesting metrics for each target.
    """
    if target_labels is None:
        raise ValueError("Target labels must be provided for regression evaluation.")

    # Fetch live data
    # live_data=fetch_data_continuously(symbol, iterations, delay)
    print("hell00000000000000000")
    #live_data=pd.read_csv("prediction_order_book_data_batch2.csv").dropna() 
    live_data=test_data
     
    print(live_data.shape)
    # Simulate predictions
    predictions1, ground_truth1, ground_truth_lstm1, lgbm_predict1, rf_predict1, lstm_predict1, cnn_lstm_predict1, tcn_predict1, transformer_predict1,xg_predict1 = predict_test(
        live_data,delay, target_labels, feature_scaler, target_scaler)
    print(predictions1[0], ground_truth1[0] ,ground_truth_lstm1[0], lgbm_predict1[0], rf_predict1[0], lstm_predict1[0], cnn_lstm_predict1[0], tcn_predict1[0], transformer_predict1[0],xg_predict1[0])


    predictions, ground_truth, ground_truth_lstm,lgbm_predict, rf_predict, lstm_predict,cnn_lstm_predict, tcn_predict, transformer_predict,xg_predict= process_predictions(
    predictions1, ground_truth1, ground_truth_lstm1,
    lgbm_predict1, rf_predict1, lstm_predict1,
    cnn_lstm_predict1, tcn_predict1, transformer_predict1,xg_predict1)
    print(predictions[0])
        
    
    
    print(predictions[0], ground_truth[0] ,ground_truth_lstm[0], lgbm_predict[0], rf_predict[0], lstm_predict[0], cnn_lstm_predict[0], tcn_predict[0], transformer_predict[0],xg_predict[0])
    # Evaluate Metrics for each target

    # Check array lengths
    arrays = {
        "Predictions_Future_Buy": predictions[:, 0],
        "Predictions_Future_Ask": predictions[:, 1],
       # "Predictions_Future_timestamp": predictions[:, 2],
        "Ground_Truth_Future_Buy": ground_truth[:, 0],
        "Ground_Truth_Future_Ask": ground_truth[:, 1],
       # "Ground_Truth_Future_timestamp": ground_truth[:, 2],
        "Ground_Truth_LSTM_Future_Buy": ground_truth_lstm[:, 0],
        "Ground_Truth_LSTM_Future_Ask": ground_truth_lstm[:, 1],
      #  "Ground_Truth_LSTM_Future_timestamp": ground_truth_lstm[:, 2],
        "LGBM_Predict_Future_Buy": lgbm_predict[:, 0],
        "LGBM_Predict_Future_Ask": lgbm_predict[:, 1],
       # "LGBM_Predict_Future_timestamp": lgbm_predict[:, 2],
        "RF_Predict_Future_Buy": rf_predict[:, 0],
        "RF_Predict_Future_Ask": rf_predict[:, 1],
      #  "RF_Predict_Future_timestamp": rf_predict[:, 2],
        "XG_Predict_Future_Buy": xg_predict[:, 0],
        "XG_Predict_Future_Ask": xg_predict[:, 1],
      #  "XG_Predict_Future_timestamp": xg_predict[:, 2],
        "LSTM_Predict_Future_Buy": lstm_predict[:, 0],
        "LSTM_Predict_Future_Ask": lstm_predict[:, 1],
      #  "LSTM_Predict_Future_timestamp": lstm_predict[:, 2],
        "CNN_LSTM_Predict_Future_Buy": cnn_lstm_predict[:, 0],
        "CNN_LSTM_Predict_Future_Ask": cnn_lstm_predict[:, 1],
      #  "CNN_LSTM_Predict_Future_timestamp": cnn_lstm_predict[:, 2],
        "TCN_Predict_Future_Buy": tcn_predict[:, 0],
        "TCN_Predict_Future_Ask": tcn_predict[:, 1],
      #  "TCN_Predict_Future_timestamp": tcn_predict[:, 2],
        "Transformer_Predict_Future_Buy": transformer_predict[:, 0],
        "Transformer_Predict_Future_Ask": transformer_predict[:, 1],
       # "Transformer_Predict_Future_timestamp": transformer_predict[:, 2],
    }

    # Group arrays by length

    grouped_arrays = defaultdict(list)
    for key, value in arrays.items():
        grouped_arrays[len(value)].append((key, value))

    # Create separate DataFrames for each length
    dataframes = {}
    for length, items in grouped_arrays.items():
        data = {key: value for key, value in items}
        df = pd.DataFrame(data)
        dataframes[length] = df
        print(f"Created DataFrame for length {length} with shape: {df.shape}")
    metrics = {}
    for i, target in enumerate(target_labels):
        ground_truth_col = ground_truth[:, i]
        ground_truth_lstm_col = ground_truth_lstm[:, i]
        predictions_col = predictions[:, i]
        lgbm_predict_col = lgbm_predict[:, i]
        rf_predict_col = rf_predict[:, i]
        xg_predict_col = xg_predict[:, i]
        lstm_predict_col = lstm_predict[:, i]   
        cnn_lstm_predict_col = cnn_lstm_predict[:, i]
        tcn_predict_col = tcn_predict[:, i] 
        transformer_predict_col = transformer_predict[:, i] 

        # Calculate metrics
        mse_lgbm = mean_squared_error(ground_truth_col, lgbm_predict_col)
        mae_lgbm = mean_absolute_error(ground_truth_col, lgbm_predict_col)
        r2_lgbm = r2_score(ground_truth_col, lgbm_predict_col)

        mse_rf = mean_squared_error(ground_truth_col, rf_predict_col)
        mae_rf = mean_absolute_error(ground_truth_col, rf_predict_col)
        r2_rf = r2_score(ground_truth_col, rf_predict_col)

        mse_xg = mean_squared_error(ground_truth_col, xg_predict_col)
        mae_xg = mean_absolute_error(ground_truth_col, xg_predict_col)
        r2_xg = r2_score(ground_truth_col, xg_predict_col)

        mse_lstm = mean_squared_error(ground_truth_lstm_col, lstm_predict_col)
        mae_lstm = mean_absolute_error(ground_truth_lstm_col, lstm_predict_col)
        r2_lstm = r2_score(ground_truth_lstm_col, lstm_predict_col)

        mse_cnn_lstm = mean_squared_error(ground_truth_lstm_col, cnn_lstm_predict_col)
        mae_cnn_lstm = mean_absolute_error(ground_truth_lstm_col, cnn_lstm_predict_col)
        r2_cnn_lstm = r2_score(ground_truth_lstm_col, cnn_lstm_predict_col)

        mse_tcn = mean_squared_error(ground_truth_lstm_col, tcn_predict_col)
        mae_tcn = mean_absolute_error(ground_truth_lstm_col, tcn_predict_col)
        r2_tcn = r2_score(ground_truth_lstm_col, tcn_predict_col)

        mse_transformer = mean_squared_error(ground_truth_lstm_col, transformer_predict_col)
        mae_transformer = mean_absolute_error(ground_truth_lstm_col, transformer_predict_col)
        r2_transformer = r2_score(ground_truth_lstm_col, transformer_predict_col)

        mse = mean_squared_error(ground_truth_col, predictions_col)
        mae = mean_absolute_error(ground_truth_col, predictions_col)
        r2 = r2_score(ground_truth_col, predictions_col)

       

        # Store metrics
        metrics[target] = {
            "mean_squared_error": mse,
            "mean_absolute_error": mae,
            "r2_score": r2
        }

        metrics[f"{target}_lgbm"] = {
            "mean_squared_error": mse_lgbm,
            "mean_absolute_error": mae_lgbm,
            "r2_score": r2_lgbm
        }

        metrics[f"{target}_rf"] = {
            "mean_squared_error": mse_rf,
            "mean_absolute_error": mae_rf,
            "r2_score": r2_rf
        }
        metrics[f"{target}_xg"] = {
            "mean_squared_error": mse_xg,
            "mean_absolute_error": mae_xg,
            "r2_score": r2_xg
        }

        metrics[f"{target}_lstm"] = {
            "mean_squared_error": mse_lstm,
            "mean_absolute_error": mae_lstm,
            "r2_score": r2_lstm
        }


        metrics[f"{target}_cnn_lstm"] = {
            "mean_squared_error": mse_cnn_lstm,
            "mean_absolute_error": mae_cnn_lstm,
            "": r2_cnn_lstm
        }

        metrics[f"{target}_tcn"] = {
            "mean_squared_error": mse_tcn,
            "mean_absolute_error": mae_tcn,
            "r2_score": r2_tcn
        }

        metrics[f"{target}_transformer"] = {
            "mean_squared_error": mse_transformer,
            "mean_absolute_error": mae_transformer,
            "r2_score": r2_transformer  
        }
    # Print Backtesting Results
    print("Backtesting Results:")
    for target, metric_values in metrics.items():
        print(f"Metrics for {target}:")
        for metric, value in metric_values.items():
            print(f"  {metric}: {value:.4f}")


  
    return metrics,dataframes


# Example usage
# target_labels = ['future_bid_price', 'future_ask_price','future_timestamp']
# metrics,df = backtest(
#     symbol="BTCUSDT",
#     test_data=test_data
#     iterations="1s",
#     delay=1,
#     target_labels=target_labels,
#     feature_scaler=feature_scaler,
#     target_scaler=target_scaler
# )

