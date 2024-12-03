import pandas as pd
import time
import os
from threading import Thread, Lock
from binance.client import Client
import pandas as pd
import numpy as np

import time
import os


from datetime import datetime, timezone
import numpy as np

from sklearn.ensemble import RandomForestRegressor
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import optuna
from ta import add_all_ta_features

from bisect import bisect_left
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
import optuna

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

from sklearn.model_selection import TimeSeriesSplit, cross_val_score


#from evolutionary_search import EvolutionaryAlgorithmSearchCV
from sklearn.model_selection import GridSearchCV
import joblib
from scipy.stats import mode

API_KEY = 'xxxxxxx'
API_SECRET = 'xxxxxxx'


client = Client(API_KEY, API_SECRET)


file_path = 'order_book_data_batch_time2.csv'

# Check if the file exists and load it, or create an empty DataFrame
if os.path.exists(file_path):
    all_data = pd.read_csv(file_path)
    print(f"Loaded data from {file_path}. Shape: {all_data.shape}")
else:
    all_data = pd.DataFrame()
    print(f"File {file_path} not found. Created an empty DataFrame.")    


data_lock = Lock()  
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


def fetch_data_continuously(symbol, iterations_per_fetch, delay_between_fetches):
    global all_data
    output_file = "order_book_data_batch2.csv"
    while True:
        try:
            # Fetch new data
            for i in range(100):
                new_data=fetch_order_book_and_kline(symbol, interval=iterations_per_fetch, prev_bid_update_time=None, prev_ask_update_time=None)
                # print(new_data.shape)
                # new_data = fetch_order_book_and_trades(symbol=symbol, iterations=iterations_per_fetch, delay=1)
                # print(f"Fetched {len(new_data)} new rows of data.")
                
                # Append new data to all_data
                with data_lock:  # Ensure thread-safe access
                    if all_data.empty:
                        all_data = new_data
                    else:
                        all_data = pd.concat([all_data, new_data], ignore_index=True)

                # # Optional: Drop old data to manage memory (e.g., keep the last 10,000 rows)
                # if len(all_data) > 10000:
                #     all_data = all_data.iloc[-10000:]
                #     print("Dropped older rows to manage memory.")        

                if not os.path.isfile(output_file):
                        all_data.to_csv(output_file, index=False)
                else:
                        new_data.to_csv(output_file, mode="a", header=False, index=False)        

               
                
                time.sleep(delay_between_fetches)  # Wait before fetching again
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

symbol = "BTCUSDT"
iterations_per_fetch = "1s"
fetch_delay = 1  # Continuous fetching every 1 second
training_interval = 120  # Retrain every 2 minutes
target_label = ['future_bid_price', 'future_ask_price']

# Start threads for fetching and training
fetch_thread = Thread(target=fetch_data_continuously, args=(symbol, iterations_per_fetch, fetch_delay))
# train_thread = Thread(target=retrain_models_periodically, args=(target_label, feature_scaler, target_scaler, training_interval,fetch_delay))

fetch_thread.start()
# train_thread.start()

fetch_thread.join()
# train_thread.join()       
