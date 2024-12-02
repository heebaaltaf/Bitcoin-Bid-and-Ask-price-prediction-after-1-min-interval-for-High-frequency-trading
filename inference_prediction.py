import time
import threading
import pandas as pd
import numpy as np
import xgboost
import time
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime, timezone
import os
import json
import joblib
import torch
import utils
import pickle
def process_predictions(
    predictions, 
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
                # print(row)
                # Convert third element to a native float for compatibility
                timestamp = float(row[2])
                
                
                # Format the datetime object to a string (e.g., '2024-11-26 19:04:21')
                datetime_obj = datetime.fromtimestamp(timestamp)
                formatted_datetime = datetime_obj.strftime('%Y-%m-%d %H:%M:%S')
                
                
                # Round the first two elements to integers and include the formatted datetime
                p = (round(row[0], 2), round(row[1], 2), formatted_datetime)
                # p=(int(np.round(row[0])), int(np.round(row[1])), formatted_datetime)
                processed.append(p)
            else:
                # Leave unchanged if it's not a valid ndarray
                print("be")
                processed.append(row)
        # print((processed))     
        return np.array(processed,dtype=object)
       
       

    # Process each array
    predictions = process_array(predictions)
    xg_predict = process_array(xg_predict)
    lgbm_predict = process_array(lgbm_predict)
    rf_predict = process_array(rf_predict)
    lstm_predict = process_array(lstm_predict)
    cnn_lstm_predict = process_array(cnn_lstm_predict)
    tcn_predict = process_array(tcn_predict)
    transformer_predict = process_array(transformer_predict)

    return (
        predictions, 
        lgbm_predict, rf_predict, lstm_predict,
        cnn_lstm_predict, tcn_predict, transformer_predict,xg_predict
    )

def fetch_order_book_and_kline(symbol, interval="1m", prev_bid_update_time=None, prev_ask_update_time=None):
    """
    Fetch the latest order book and kline data simultaneously and synchronize timestamps.
    Includes computation for bid/ask advance times and trade analysis.
    """
    data = []

    prev_bid_update_time = None
    prev_ask_update_time = None
    iteration = 0
    
    while True:
        try:
            # Fetch order book
            order_book = client.get_order_book(symbol=symbol)
            kline = client.get_klines(symbol=symbol, interval=interval, limit=1)[-1]
            trades = client.get_recent_trades(symbol=symbol)



            # Get the current timestamp for the order book and round it to 2 decimal places (10ms)
            order_book_timestamp = pd.Timestamp.now(tz='UTC').floor("s").tz_localize(None)
            print(order_book_timestamp)

            # Convert kline's open time from milliseconds to Timestamp and round to 2 decimal places (10ms)
            kline_open_time = pd.to_datetime(kline[0], unit='ms').tz_localize(None)

            # # Format the timestamps to desired output
            # order_book_timestamp = order_book_timestamp.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4]  # Remove trailing zeros
            # kline_open_time = kline_open_time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-4] 
          
            
            
          
            bid_price = float(order_book['bids'][0][0])
            bid_qty = float(order_book['bids'][0][1])
            ask_price = float(order_book['asks'][0][0])
            ask_qty = float(order_book['asks'][0][1])

            # Compute bid/ask advance times
            bid_update_time = order_book['lastUpdateId']
            ask_update_time = order_book['lastUpdateId']
            bid_advance_time = (bid_update_time - prev_bid_update_time) / 1e6 if prev_bid_update_time else 0
            ask_advance_time = (ask_update_time - prev_ask_update_time) / 1e6 if prev_ask_update_time else 0
            prev_bid_update_time = bid_update_time
            prev_ask_update_time = ask_update_time

        

            kline_df = pd.DataFrame([{
                'kline_open_time': kline_open_time,
                'open': float(kline[1]),
                'high': float(kline[2]),
                'low': float(kline[3]),
                'close': float(kline[4]),
                'volume': float(kline[5]),
            }])


        



            # Fetch recent trades
           
            trade_prices = [float(trade['price']) for trade in trades]
            trade_volumes = [float(trade['qty']) for trade in trades]
            trade_times = [trade['time'] for trade in trades]

            trade_price = trade_prices[-1] if trade_prices else np.nan
            sum_trade_1s = sum(trade_volumes)

            # Analyze recent trades

            current_time = int(datetime.now(timezone.utc).timestamp() * 1000)  # UTC in milliseconds

            # Assume trade times are already in milliseconds (from the API)
            trade_times = [trade['time'] for trade in trades]  # In milliseconds

            # Convert trade times to datetime (optional for debugging or analysis)
            trade_times_utc = [datetime.fromtimestamp(t / 1000, tz=timezone.utc) for t in trade_times]
            # Compute last trade time
            last_trade_time = pd.to_datetime(trade_times[-1], unit='ms') if trade_times else pd.NaT


            # Perform time-based comparisons
            _1s_trades = [trade for trade in trades if current_time - trade['time'] <= 1000]
            _3s_trades = [trade for trade in trades if current_time - trade['time'] <= 3000]
            _5s_trades= [trade for trade in trades if current_time - trade['time'] <= 5000]
        
            

            def compute_side(trades):
                buy_volume = sum(float(trade['qty']) for trade in trades if not trade['isBuyerMaker'])
                sell_volume = sum(float(trade['qty']) for trade in trades if trade['isBuyerMaker'])
                return 1 if buy_volume > sell_volume else 0

            _1s_side = compute_side(_1s_trades)
            _3s_side = compute_side(_3s_trades)
            _5s_side = compute_side(_5s_trades)

            # Prepare combined data
            orderbook_df = pd.DataFrame([{
                'timestamp': order_book_timestamp,
                'bid_price': bid_price,
                'bid_qty': bid_qty,
                'ask_price': ask_price,
                'ask_qty': ask_qty,
                'trade_price': trade_price,
                'sum_trade_1s': sum_trade_1s,
                'bid_advance_time': bid_advance_time,
                'ask_advance_time': ask_advance_time,
                '_1s_side': _1s_side,
                '_3s_side': _3s_side,
                '_5s_side': _5s_side,
                'last_trade_time': last_trade_time,
            }])


            # orderbook_df['timestamp'] = pd.to_datetime(orderbook_df['timestamp'])  # No format needed if already datetime
            # kline_df['kline_open_time'] = pd.to_datetime(kline_df['kline_open_time'])
            # Perform nearest merge
            combined_data = pd.merge_asof(
                orderbook_df.sort_values('timestamp'),  # Sort by timestamp
                kline_df.sort_values('kline_open_time'),  # Sort by kline_open_time
                left_on="timestamp",
                right_on="kline_open_time",
                direction="nearest",
                tolerance=pd.Timedelta("1s")  # Match within 1 second
            )

            # Append combined data to a list
            
            data.append(combined_data)

            # Convert the list of DataFrames to a single DataFrame
            df = pd.concat(data, ignore_index=True)

            # Sort and drop duplicates
            df = df.sort_values("timestamp").drop_duplicates().reset_index(drop=True)

            # Calculate future bid/ask prices and quantities (10 seconds ahead)
        
        # df = compute_technical_indicators(df)
           
        
            return df
          

        except Exception as e:
            print(f"Error fetching data: {e}")
            continue


def predict_response(data, delay, target_label, feature_scaler, target_scaler):
    """
    Returns predictions for regression models when only features (x) are available.
    """
    try:
        # Load pre-saved features and correlation columns to remove
        if os.path.exists("features2.txt"):
            with open("features2.txt", 'r') as file:
                feature_data = json.load(file)
                print("Features Data Loaded")
                features = feature_data.get('keep_features', [])
                correlation_remove = feature_data.get('correlation_remove', [])
        else:
            raise FileNotFoundError("Features file 'features2.txt' not found.")
        data.dropna(inplace=True)
        print(data.shape)
        # Preprocess data
        
        data = data.copy()
        data = utils.compute_technical_indicators(data, delay)
        data = utils.preprocessing(data)
        data = utils.preprocess_datetime(data)
        data = utils.fill_null(data)
         
        data = data.dropna().reset_index(drop=True)
       
        # Feature engineering
        x = utils.feature_eng.basic_features(data)
        x = x.drop(columns=correlation_remove, errors='ignore')
        x = utils.feature_eng.lag_rolling_features(x)
        x = x.ffill().bfill()
        x = utils.feature_eng.remove_na(x)
        print(x.shape)
        print(len(x.columns))
        print(len(features))
        # Ensure features are available
        # if not set(features).issubset(x.columns):
        #     missing_features = set(features) - set(x.columns)
        #     raise ValueError(f"Missing required features in input data: {missing_features}")
        
        x = x[[i for i in features if i in x.columns]]
        # x = x[features]
        
        feature_scaler.fit(x)  # Fit scaler to ensure it works correctly
        x = feature_scaler.transform(x)

        # Prepare LSTM-like inputs for sequential models
        timesteps = 50
        print("x",x.shape)  
        x_lstm = utils.model.prepare_lstm_input(x, lookback=timesteps)
        input_shape = (x_lstm.shape[1], x_lstm.shape[2])
        
        # Load models
        lgbm = joblib.load('lgbm_multi_output2.joblib')
        rf = joblib.load('rf_multi_output2.joblib')
        # xg=joblib.load('xg_multi_output2.joblib')
        # xg = xgboost.Booster()
        # xg.load_model("xgboost_model.json")
        with open('xg_multi_output2.pkl', 'rb') as f:
           xg = pickle.load(f)
        lstm_model = utils.LSTMModel(input_size=x_lstm.shape[2], output_size=len(target_label))
        lstm_model.load_state_dict(torch.load('lstm_model_multi_output2.pth'))
        lstm_model.eval()

        cnn_lstm_model = utils.CNNLSTMModel(input_size=x_lstm.shape[2], output_size=len(target_label))
        cnn_lstm_model.load_state_dict(torch.load('cnn_lstm_model_multi_output2.pth'))
        cnn_lstm_model.eval()

        tcn_model = utils.TCNModel(input_size=x_lstm.shape[2], output_size=len(target_label), num_channels=[16, 32, 64])
        tcn_model.load_state_dict(torch.load('tcn_model_multi_output2.pth'))
        tcn_model.eval()

        transformer_model = utils.TransformerModel(input_size=x_lstm.shape[2], output_size=len(target_label))
        transformer_model.load_state_dict(torch.load('transformer_multi_output2.pth'))
        transformer_model.eval()

        # Make predictions
        lgbm_predict = lgbm.predict(x)
        rf_predict = rf.predict(x)
        xg_predict=xg.predict(x)
        # PyTorch Model Predictions
        x_lstm_tensor = torch.tensor(x_lstm, dtype=torch.float32)

        with torch.no_grad():
            lstm_predict = lstm_model(x_lstm_tensor).numpy()
            cnn_lstm_predict = cnn_lstm_model(x_lstm_tensor).numpy()
            tcn_predict = tcn_model(x_lstm_tensor).numpy()
            transformer_predict = transformer_model(x_lstm_tensor).numpy()

        # Inverse transform predictions
        lgbm_predict = target_scaler.inverse_transform(lgbm_predict)
        rf_predict = target_scaler.inverse_transform(rf_predict)
        xg_predict = target_scaler.inverse_transform(xg_predict)
        lstm_predict = target_scaler.inverse_transform(lstm_predict)
        cnn_lstm_predict = target_scaler.inverse_transform(cnn_lstm_predict)
        tcn_predict = target_scaler.inverse_transform(tcn_predict)
        transformer_predict = target_scaler.inverse_transform(transformer_predict)

        # Ensemble predictions (custom logic)
        ensemble_predict = (lgbm_predict + rf_predict) / 2
        # return ensemble_predict, lgbm_predict, rf_predict,xg_predict

        return ensemble_predict, lgbm_predict, rf_predict, lstm_predict, cnn_lstm_predict, tcn_predict, transformer_predict,xg_predict

    except Exception as e:
        print(f"Error in predict_response: {e}")
        raise


from threading import Thread, Lock    
# Lock for scaler access
scaler_lock = Lock()

try:
    with scaler_lock:
        feature_scaler = joblib.load('feature_scaler.pkl')
        target_scaler = joblib.load('target_scaler.pkl')
except (EOFError, FileNotFoundError) as e:
    print(f"Scaler loading failed: {e}")
    # Recreate scalers if necessary
    feature_scaler = MinMaxScaler()
    target_scaler = MinMaxScaler()
    joblib.dump(feature_scaler, 'feature_scaler.pkl')
    joblib.dump(target_scaler, 'target_scaler.pkl')

def predict_step_by_step(delay, target_labels, feature_scaler, target_scaler):
    """
    Predict step-by-step and store predictions in a global DataFrame.
    """
    global live_data, prediction_results, stop_threads
    processed_indices = set()

    while not stop_threads:
        print(f"Live data length: {len(live_data)}")
        if len(live_data) >= 80:
            live_data = live_data.sort_values(by='timestamp').reset_index(drop=True)

            # Start predictions from the 80th observation
            for i in range(85, len(live_data)):
                if i in processed_indices:
                    continue

                # Extract current observation details
                original_timestamp = pd.to_datetime(live_data.iloc[i]["timestamp"])
                original_bid_price = live_data.iloc[i]["bid_price"]
                original_ask_price = live_data.iloc[i]["ask_price"]

                # Prepare data for prediction
                data_for_prediction = live_data.iloc[:i].copy()

                # Generate predictions
                predictions1, lgbm_predict1, rf_predict1, lstm_predict1, cnn_lstm_predict1, tcn_predict1, transformer_predict1, xg_predict1 = predict_response(
                    data_for_prediction, delay, target_labels, feature_scaler, target_scaler
                )

                ensemble_predict, _, _, _, _, _, _, _ = process_predictions(
                    predictions1, lgbm_predict1, rf_predict1, lstm_predict1, cnn_lstm_predict1, tcn_predict1, transformer_predict1, xg_predict1
                )

                # Extract predictions for the current observation
                pred_bid_price = ensemble_predict[-1][0]
                pred_ask_price = ensemble_predict[-1][1]

                # Add prediction entry to the DataFrame
                prediction_results = pd.concat([
                    prediction_results,
                    pd.DataFrame([{
                        "original_timestamp": original_timestamp,
                        "original_bid_price": original_bid_price,
                        "original_ask_price": original_ask_price,
                        "predicted_bid_price": pred_bid_price,
                        "predicted_ask_price": pred_ask_price,
                        "bid_hit_time": None,
                        "ask_hit_time": None,
                        "hit_bid_price": None,
                        "hit_ask_price": None
                    }])
                ], ignore_index=True)

                print(f"Prediction added for {original_timestamp}:")
                print(f"  Original Bid Price: {original_bid_price}, Original Ask Price: {original_ask_price}")
                print(f"  Predicted Bid Price: {pred_bid_price}, Predicted Ask Price: {pred_ask_price}")

                # Mark the current observation as processed
                processed_indices.add(i)

            time.sleep(delay)
        else:
            print("Not enough data for predictions. Waiting for more data...")
            time.sleep(1)


def match_predictions():
    """
    Continuously match predictions with future observations.
    Updates the global prediction_results DataFrame immediately upon finding a match.
    """
    global live_data, prediction_results, stop_threads

    while not stop_threads:
        print("Matching predictions...")
        if prediction_results.empty or live_data.empty:
            print("No data to process. Waiting...")
            time.sleep(1)  # Wait if there are no predictions or live data
            continue

        for idx, prediction in prediction_results.iterrows():
            # Skip predictions that are already matched
            if pd.notna(prediction["bid_hit_time"]) and pd.notna(prediction["ask_hit_time"]):
                continue

            original_timestamp = prediction["original_timestamp"]
            pred_bid_price = prediction["predicted_bid_price"]
            pred_ask_price = prediction["predicted_ask_price"]

            # Filter future observations after the original timestamp
            future_data = live_data[live_data["timestamp"] > original_timestamp]

            for _, future_row in future_data.iterrows():
                future_bid_price = future_row["bid_price"]
                future_ask_price = future_row["ask_price"]
                future_timestamp = pd.to_datetime(future_row["timestamp"])

                # Check if the bid price condition is met
                if pd.isna(prediction_results.at[idx, "bid_hit_time"]):
                    if (pred_bid_price < prediction["original_bid_price"] and future_bid_price <= pred_bid_price) or \
                       (pred_bid_price > prediction["original_bid_price"] and future_bid_price >= pred_bid_price):
                        prediction_results.at[idx, "bid_hit_time"] = (future_timestamp - original_timestamp).total_seconds()
                        prediction_results.at[idx, "hit_bid_price"] = future_bid_price
                        print(f"Bid Hit for {original_timestamp}:")
                        print(f"  Predicted Bid Price: {pred_bid_price}, Observed Bid Price: {future_bid_price}")
                        print(f"  Bid Hit Time: {prediction_results.at[idx, 'bid_hit_time']} seconds")

                # Check if the ask price condition is met
                if pd.isna(prediction_results.at[idx, "ask_hit_time"]):
                    if (pred_ask_price < prediction["original_ask_price"] and future_ask_price <= pred_ask_price) or \
                       (pred_ask_price > prediction["original_ask_price"] and future_ask_price >= pred_ask_price):
                        prediction_results.at[idx, "ask_hit_time"] = (future_timestamp - original_timestamp).total_seconds()
                        prediction_results.at[idx, "hit_ask_price"] = future_ask_price
                        print(f"Ask Hit for {original_timestamp}:")
                        print(f"  Predicted Ask Price: {pred_ask_price}, Observed Ask Price: {future_ask_price}")
                        print(f"  Ask Hit Time: {prediction_results.at[idx, 'ask_hit_time']} seconds")

                # Exit early for this prediction if both bid and ask are matched
                if pd.notna(prediction_results.at[idx, "bid_hit_time"]) and pd.notna(prediction_results.at[idx, "ask_hit_time"]):
                    break

        # Print the updated DataFrame
        print("Updated Prediction Results:")
        print(prediction_results)

        # Wait for 1 second before checking again
        time.sleep(1)


def fetch_data_continuously(symbol, iterations_per_fetch, delay_between_fetches):
    global live_data, stop_threads
    while not stop_threads:
        # Simulate data fetching (replace with actual fetch logic)
        new_data = utils.fetch_order_book_and_kline(symbol, interval=iterations_per_fetch)
        
        # Append to live_data
        if live_data.empty:
            live_data = new_data
        else:
            live_data = pd.concat([live_data, new_data], ignore_index=True)
        
        # Drop older rows to manage memory
        if len(live_data) > 1000:
            live_data = live_data.iloc[-1000:]

        if len(live_data) >= 200:
            print("Live data reached 500 rows. Stopping threads.")
            stop_threads = True
            break

        time.sleep(delay_between_fetches)


# Initialize global variables
live_data = pd.DataFrame()
prediction_results = pd.DataFrame()
stop_threads = False

# # Load scalers
# feature_scaler = joblib.load('feature_scaler.pkl')
# target_scaler = joblib.load('target_scaler.pkl')
target_labels = ['future_bid_price', 'future_ask_price']#,'future_timestamp']
# Start threads
fetch_thread = Thread(target=fetch_data_continuously, args=("BTCUSDT", "1s", 1))
predict_thread = Thread(target=predict_step_by_step, args=(1, target_labels, feature_scaler, target_scaler))
match_thread = Thread(target=match_predictions)

fetch_thread.start()
predict_thread.start()
match_thread.start()

fetch_thread.join()
predict_thread.join()
match_thread.join()

# Return final DataFrame
print("Final Prediction Results:")
print(prediction_results)
