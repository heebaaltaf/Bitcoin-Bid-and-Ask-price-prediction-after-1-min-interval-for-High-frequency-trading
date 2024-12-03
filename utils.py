from binance.client import Client
import pandas as pd
import numpy as np


from threading import Thread, Lock
from sklearn.metrics import mean_squared_error


from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import optuna
import json
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


from bisect import bisect_left
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error

from lightgbm import LGBMRegressor
import optuna

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor





from scipy.stats import mode
from sklearn.ensemble import  RandomForestRegressor

from sklearn.model_selection import TimeSeriesSplit

API_KEY = 'xxxxxxxxx'
API_SECRET = 'xxxxxxxxxxx'


client = Client(API_KEY, API_SECRET)


  
def preprocessing(data):
    """
    Preprocess the data by:
    - Validating required columns
    - Parsing and converting datetime columns to UNIX timestamps (float)
    - Converting numeric columns to float
    - Sorting by timestamp
    """
       # Required columns (excluding optional future columns)
    required_columns = [
        'timestamp', 'bid_price', 'bid_qty', 'ask_price', 'ask_qty', 
        'trade_price', 'sum_trade_1s', 'bid_advance_time', 
        'ask_advance_time', '_1s_side', '_3s_side', '_5s_side',
        'last_trade_time', 'kline_open_time', 'open', 'high', 'low', 
        'close', 'volume'
    ]
    
    # Optional columns
    optional_columns = ['future_bid_price', 'future_ask_price']
    
    # Check for missing columns
    missing_columns = set(required_columns) - set(data.columns)
    if missing_columns:
        raise ValueError(f"Missing columns in data: {missing_columns}")
    
    # List of datetime columns to convert to UNIX timestamps (float)
    datetime_columns = ['timestamp', 'last_trade_time', 'kline_open_time','future_timestamp']
    print("metoo",data.shape)
    def parse_datetime_mixed(col):
        """
        Parse a datetime column with mixed formats:
        - Includes seconds and microseconds
        - Includes date-only formats (YYYY-MM-DD)
        - Returns NaT for invalid entries.
        """
        def parse_single(value):
            try:
                # Attempt parsing with microseconds
                return pd.to_datetime(value, format='%Y-%m-%d %H:%M:%S.%f', utc=True)
            except ValueError:
                try:
                    # Fallback to seconds-only
                    return pd.to_datetime(value, format='%Y-%m-%d %H:%M:%S', utc=True)
                except ValueError:
                    try:
                        # Fallback to date-only format
                        return pd.to_datetime(value, format='%Y-%m-%d', utc=True)
                    except ValueError:
                        # Return NaT if all formats fail
                        return pd.NaT
        
        return col.apply(parse_single)

    
    # Convert datetime columns to UNIX timestamps
    for col in datetime_columns:
        if col in data.columns:
            data[col] = parse_datetime_mixed(data[col]).astype('int64') / 1e9  # Convert to float seconds
    
    # # Convert datetime columns to UNIX timestamps
    # for col in datetime_columns:
    #     if col in data.columns:
    #         data[col] = pd.to_datetime(data[col]).astype('int64') / 1e9  # Convert to float seconds
    
    float_list = [
        'bid_price', 'bid_qty', 'ask_price', 'ask_qty', 
        'trade_price', 'sum_trade_1s', 'bid_advance_time', 
        'ask_advance_time', 'open', 'high', 'low', 'close', 'volume'
    ]

    # Add optional columns if present
    for col in optional_columns:
        if col in data.columns:
            float_list.append(col)
    
    # Convert numeric columns to float
    for col in float_list:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')  # Handle invalid values as NaN
    
    # Sort data by timestamp
    data = data.sort_values(by='timestamp', ascending=True).reset_index(drop=True)
    
    return data
def preprocess_datetime(data):
    """
    Convert datetime column to numeric features.
    Args:
        data (pd.DataFrame): Input DataFrame.
    Returns:
        pd.DataFrame: DataFrame with datetime converted to numeric features.
    """
    data = data.copy()
    
    # Ensure 'timestamp' exists and is of datetime type
    if 'timestamp' in data.columns and pd.api.types.is_datetime64_any_dtype(data['timestamp']):
        # Extract useful features from the timestamp
        data['hour'] = data['timestamp'].dt.hour
        data['day'] = data['timestamp'].dt.day
        data['month'] = data['timestamp'].dt.month
        data['weekday'] = data['timestamp'].dt.weekday

        # Optional: Convert to UNIX timestamp (seconds since epoch)
        data['timestamp_unix'] = data['timestamp'].dt.tz_localize(None).astype('int64') // 10**9
        
        # Drop the original timestamp column if it is no longer needed
        #data.drop(columns=['timestamp'], inplace=True)

    return data

# data = preprocessing(data)
# data = preprocess_datetime(data)
# print("j",data.shape)

def check_null(data, detailed=False):
    '''Check null values in DataFrame with an option for detailed stats.'''
    data = data.copy()
    have_null_cols = data.columns[data.isnull().any()]
    # print(f"Columns with null values: {', '.join(have_null_cols)}")
    # print(f"Total null values: {data.isnull().sum().sum()}")

    if detailed:
        for col in have_null_cols:
            print(f"Column '{col}':")
            print(f"  Null Count: {data[col].isnull().sum()}")
            print(f"  Null Percentage: {round(data[col].isnull().sum() / data.shape[0] * 100, 2)}%")


# check_null(data)            



def fill_null(data):
    """
    Fill null values based on predefined logic:
    - `sum_trade_1s`: Fill with 0 where null.
    - `last_trade_time`: Fill incrementally if possible, otherwise NaN.
    """
    data = data.copy()

    # Fill `sum_trade_1s` with 0
    #data['sum_trade_1s'].fillna(0, inplace=True)
    data['sum_trade_1s'] = data['sum_trade_1s'].fillna(0)

    # Fill `last_trade_time` using incremental logic
    prev_last_trade_time = None
    prev_timestamp = None

    def fill_last_trade_time(row):
        nonlocal prev_last_trade_time, prev_timestamp
        last_trade_time = row['last_trade_time']
        timestamp = row['timestamp']

        if pd.isnull(last_trade_time):
            if prev_timestamp is not None:
                time_interval = (timestamp - prev_timestamp).total_seconds()
                if time_interval <= 1:
                    last_trade_time = prev_last_trade_time + time_interval
        prev_last_trade_time = last_trade_time
        prev_timestamp = timestamp
        return last_trade_time

    data['last_trade_time'] = data.apply(fill_last_trade_time, axis=1)

    # Print summary
    null_counts = data.isnull().sum()
    # print(f"Remaining null counts:\n{null_counts[null_counts > 0]}")
    return data

# data = fill_null(data)

def x_y_split(data, target_labels):
    """
    Split the data into features (X) and targets (Y).
    
    Args:
        data (pd.DataFrame): The input dataset.
        target_labels (list): List of target column names.
        
    Returns:
        x (pd.DataFrame): Feature data.
        y (pd.DataFrame): Target data.
    """
    # Ensure all target columns exist in the dataset
    missing_labels = set(target_labels) - set(data.columns)
    if missing_labels:
        raise ValueError(f"Missing target columns in data: {missing_labels}")
    
    # Separate target columns
    y = data[target_labels].copy()
    
    # Remaining columns are features
    feature_cols = list(set(data.columns) - set(target_labels))
    x = data[feature_cols].copy()
    
    return x, y


# x, y = x_y_split(data,target_label)

class feature_eng:
    timestamp = None
    max_lag = 5
    num_window = [5, 10, 20]
    sec_window = [1, 3, 5, 10]
    rolling_sum_cols = []
    rolling_mean_cols = []
    rolling_max_cols = []
    rolling_min_cols = []
    rolling_std_cols = []

    @staticmethod
    def bid_ask_spread(data):
        """Calculate bid-ask spread."""
        if 'ask_price' in data.columns and 'bid_price' in data.columns:
            data['spread'] = data['ask_price'] - data['bid_price']
        else:
            raise ValueError("Missing columns: 'ask_price' and/or 'bid_price'")

    @staticmethod
    def bid_ask_qty_comb(data):
        """Calculate total and differential quantities for bid and ask."""
        if 'ask_qty' in data.columns and 'bid_qty' in data.columns:
            data['bid_ask_qty_total'] = data['ask_qty'] + data['bid_qty']
            data['bid_ask_qty_diff'] = data['ask_qty'] - data['bid_qty']
        else:
            raise ValueError("Missing columns: 'ask_qty' and/or 'bid_qty'")

    @staticmethod
    def trade_price_feature(data):
        """Generate features related to trade price positions."""
        if not {'trade_price', 'bid_price', 'ask_price', 'last_trade_time', 'timestamp'}.issubset(data.columns):
            raise ValueError("Missing required columns for trade price feature.")
        
        data['trade_price_compare'] = 0
        data.loc[data['trade_price'] <= data['bid_price'], 'trade_price_compare'] = -1
        data.loc[data['trade_price'] >= data['ask_price'], 'trade_price_compare'] = 1

        # Determine trade price position relative to bid/ask
        trade_price_pos = []
        for i in range(len(data)):
            trade_price = data['trade_price'].iloc[i]
            bid_price = data['bid_price'].iloc[i]
            ask_price = data['ask_price'].iloc[i]

            if bid_price <= trade_price <= ask_price:
                trade_price_pos.append(0)
            elif trade_price < bid_price:
                trade_price_pos.append(-1)
            elif trade_price > ask_price:
                trade_price_pos.append(1)
            else:
                trade_price_pos.append(np.nan)
        data['trade_price_pos'] = trade_price_pos

    @staticmethod
    def diff_feature(data):
        """Calculate difference features."""
        for col in data.columns.difference(['timestamp']):
            data[f'{col}_diff'] = data[col].diff()

    @staticmethod
    def up_or_down(data):
        """Indicate price movement direction."""
        if 'bid_price_diff' in data.columns and 'ask_price_diff' in data.columns:
            data['up_down'] = 0
            data.loc[data['bid_price_diff'] < 0, 'up_down'] = -1
            data.loc[data['ask_price_diff'] > 0, 'up_down'] = 1

    @staticmethod
    def lag_feature(data, col, lag):
        """Create lag features for a specific column."""
        if col in data.columns:
            data[f'{col}_lag_{lag}'] = data[col].shift(lag)
        else:
            raise ValueError(f"Column '{col}' not found in data.")
    
    @staticmethod
    def lag_feature(data, cols, lags):
        """
        Create lag features for multiple columns at once.
        Args:
            data (pd.DataFrame): Input DataFrame.
            cols (list): List of column names to generate lag features for.
            lags (int): Maximum number of lags.
        Returns:
            pd.DataFrame: DataFrame with lag features added.
        """
        new_columns = {}
        for col in cols:
            for lag in range(1, lags + 1):
                new_columns[f'{col}_lag_{lag}'] = data[col].shift(lag)
        return pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)
    @staticmethod
    def rolling_feature(data, col, window, feature):
        """Create rolling features."""
        if col not in data.columns:
            raise ValueError(f"Column '{col}' not found in data.")
        
        rolling = data[col].rolling(window=window)
        new_col = f'{col}_rolling_{feature}_{window}'

        if feature == 'sum':
            data[new_col] = rolling.sum()
        elif feature == 'mean':
            data[new_col] = rolling.mean()
        elif feature == 'max':
            data[new_col] = rolling.max()
        elif feature == 'min':
            data[new_col] = rolling.min()
        elif feature == 'std':
            data[new_col] = rolling.std()
        elif feature == 'mode':
            data[new_col] = rolling.apply(lambda x: mode(x)[0], raw=False)
        else:
            raise ValueError(f"Unsupported rolling feature: '{feature}'")

        # Fill NaNs after rolling operation (if needed)
        data[new_col].fillna(0, inplace=True)

    @classmethod
    def basic_features(cls, data):
        """Generate basic features."""
        if 'timestamp' not in data.columns:
            raise ValueError("Column 'timestamp' not found in data.")

        data = data.copy()
        cls.timestamp = data['timestamp']

        cls.bid_ask_spread(data)
        cls.bid_ask_qty_comb(data)
        cls.trade_price_feature(data)
        cls.diff_feature(data)
        cls.up_or_down(data)

        return data

    # @classmethod
    # def lag_rolling_features(cls, data):
    #     """
    #     Generate lag and rolling features for numeric columns only.
    #     """
    #     data = data.copy()

    #     # Exclude non-numeric columns
    #     numeric_cols = data.select_dtypes(include=[np.number]).columns
    #     rolling_cols = set(numeric_cols) - {'trade_price_compare', 'trade_price_pos'}

    #     # Identify columns for different rolling operations
    #     cls.rolling_sum_cols = [col for col in rolling_cols if 'diff' in col or 'up_down' in col]
    #     cls.rolling_mean_cols = rolling_cols
    #     cls.rolling_max_cols = [col for col in rolling_cols if 'bid_qty' in col or 'ask_qty' in col]
    #     cls.rolling_min_cols = [col for col in rolling_cols if 'bid_qty' in col or 'ask_qty' in col]
    #     cls.rolling_std_cols = rolling_cols

    #     # Generate lag features
    #     for col in rolling_cols:
    #         for lag in range(1, cls.max_lag + 1):
    #             cls.lag_feature(data, col, lag)

    #     # Generate rolling features
    #     for col in rolling_cols:
    #         for window in cls.num_window:
    #             if col in cls.rolling_sum_cols:
    #                 cls.rolling_feature(data, col, window, 'sum')
    #             if col in cls.rolling_mean_cols:
    #                 cls.rolling_feature(data, col, window, 'mean')
    #             if col in cls.rolling_max_cols:
    #                 cls.rolling_feature(data, col, window, 'max')
    #             if col in cls.rolling_min_cols:
    #                 cls.rolling_feature(data, col, window, 'min')
    #             if col in cls.rolling_std_cols:
    #                 cls.rolling_feature(data, col, window, 'std')

    #     return data
    @classmethod
    def lag_rolling_features(cls, data):
        """
        Generate lag and rolling features for numeric columns only.
        """
        data = data.copy()

        # Exclude non-numeric columns
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        rolling_cols = set(numeric_cols) - {'trade_price_compare', 'trade_price_pos'}

        # Identify columns for different rolling operations
        cls.rolling_sum_cols = [col for col in rolling_cols if 'diff' in col or 'up_down' in col]
        cls.rolling_mean_cols = rolling_cols
        cls.rolling_max_cols = [col for col in rolling_cols if 'bid_qty' in col or 'ask_qty' in col]
        cls.rolling_min_cols = [col for col in rolling_cols if 'bid_qty' in col or 'ask_qty' in col]
        cls.rolling_std_cols = rolling_cols

        # Create a dictionary to store new columns
        new_columns = {}

        # Generate lag features
        for col in rolling_cols:
            for lag in range(1, cls.max_lag + 1):
                new_columns[f'{col}_lag_{lag}'] = data[col].shift(lag)

        # Generate rolling features
        for col in rolling_cols:
            for window in cls.num_window:
                if col in cls.rolling_sum_cols:
                    new_columns[f'{col}_rolling_sum_{window}'] = data[col].rolling(window=window).sum()
                if col in cls.rolling_mean_cols:
                    new_columns[f'{col}_rolling_mean_{window}'] = data[col].rolling(window=window).mean()
                if col in cls.rolling_max_cols:
                    new_columns[f'{col}_rolling_max_{window}'] = data[col].rolling(window=window).max()
                if col in cls.rolling_min_cols:
                    new_columns[f'{col}_rolling_min_{window}'] = data[col].rolling(window=window).min()
                if col in cls.rolling_std_cols:
                    new_columns[f'{col}_rolling_std_{window}'] = data[col].rolling(window=window).std()

        # Add all new columns to the DataFrame at once
        new_data = pd.concat([data, pd.DataFrame(new_columns, index=data.index)], axis=1)

        # Fill NaNs for rolling features
        for col in new_columns.keys():
            if 'rolling' in col:
                # new_data[col].fillna(0, inplace=True)
                new_data[col] = new_data[col].fillna(0)

                

        return new_data



    def remove_na(x, y=None):
        """
        Remove rows with NaN values from x, and align y if provided.
        
        Args:
            x (pd.DataFrame): Feature data.
            y (pd.DataFrame or None, optional): Target data. Defaults to None.
        
        Returns:
            tuple or pd.DataFrame: Cleaned x and y if y is provided, otherwise just x.
        """
        x = x.reset_index(drop=True)  # Reset index for consistent alignment
        x = x.dropna()  # Remove rows with NaN values from x
        
        if y is not None:
            # Align y with the indices of the cleaned x
            y = y.loc[x.index, :].reset_index(drop=True)
            return x, y
        else:
            return x
   
   
# print(x.shape)    
# x = feature_eng.basic_features(x)
# print(x.shape)    

    
class correlation_filter:
    remove_cols = []

    @classmethod
    def filter(cls, x, threshold=0.99, log_file=None):
        """
        Filter out highly correlated features based on a correlation threshold.

        Args:
            x (pd.DataFrame): The input DataFrame with features.
            threshold (float): The correlation threshold for removing features.
            log_file (str): Optional file path to log removed features.

        Returns:
            pd.DataFrame: DataFrame with highly correlated features removed.
        """
        if x.shape[1] < 2:
            raise ValueError("Input DataFrame must have at least two columns to calculate correlations.")
        
        # Calculate the correlation matrix
        corr_matrix = x.corr().abs()

        # Select the upper triangle of the correlation matrix
        upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        # Find features with correlation above the threshold
        to_drop = [
            column for column in upper_triangle.columns
            if any(upper_triangle[column] > threshold)
        ]

        # Log removed columns
        if log_file:
            with open(log_file, 'a') as f:
                f.write(f"Removed features due to correlation (threshold={threshold}): {', '.join(to_drop)}\n")

        # Drop the highly correlated features
        cls.remove_cols.extend(to_drop)
        return x.drop(columns=to_drop)
# x = correlation_filter.filter(x)
# print(x.shape)
# x = feature_eng.lag_rolling_features(x)
# print(x.shape)
# x = x.ffill()  # Forward fill
# x = x.bfill()
# x, y = feature_eng.remove_na(x, y)
# y = y[target_label]


class feature_selection:
    """Feature selection based solely on Random Forest feature importance."""

    @classmethod
    def select(cls, x, y, top_perc=0.05):
        """
        Select top features based on Random Forest feature importance.
        Args:
            x (pd.DataFrame): Feature matrix.
            y (pd.Series or np.array): Target values.
            top_perc (float): Percentage of top features to select.
        Returns:
            list: Selected features.
        """
        if not isinstance(x, pd.DataFrame):
            raise ValueError("Input `x` must be a pandas DataFrame.")
        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input `x` and `y` must not be empty.")

        feature_imp = cls.rf_importance_selection(x, y)
        perc_threshold = np.percentile(feature_imp['avg_importance'], (1 - top_perc) * 100)
        selected_features = feature_imp.loc[feature_imp['avg_importance'] >= perc_threshold, 'feature'].tolist()

        return selected_features

    @staticmethod
    def rf_importance_selection(x, y, iter_time=3):
        """
        Calculate feature importance using Random Forest.
        Args:
            x (pd.DataFrame): Feature matrix.
            y (pd.Series or np.array): Target values.
            iter_time (int): Number of iterations for averaging feature importances.
        Returns:
            pd.DataFrame: DataFrame with feature importances.
        """
        x = x.select_dtypes(include=[np.number])

        if len(x) == 0 or len(y) == 0:
            raise ValueError("Input `x` and `y` must not be empty.")

        feature_imp = pd.DataFrame({'feature': x.columns})

        for i in range(iter_time):
            rf = RandomForestRegressor(n_estimators=100, max_depth=8, random_state=i)
            rf.fit(x, y)
            feature_imp[f'importance_{i+1}'] = rf.feature_importances_

        feature_imp['avg_importance'] = feature_imp.iloc[:, 1:].mean(axis=1)

        return feature_imp
# features = feature_selection.select(x, y)
# print("d",x.shape)
class feature:
    @staticmethod
    def save(features, correlation_remove):
        final = {
            'keep_features': features,
            'correlation_remove': correlation_remove
        }

        with open('features2.txt', 'w') as f:
            f.write(json.dumps(final))

    @staticmethod
    def load():
        with open('features2.txt', 'r') as f:
            features = f.read()
            features = json.loads(features)

        return features
# print("d",x.shape)
# feature.save(features, correlation_filter.remove_cols)
# x.drop(columns=['timestamp'], inplace=True)



# class model:
#     @staticmethod
#     def random_forest(x, y):
#         """
#         Train a Random Forest Regressor.
#         Args:
#             x (pd.DataFrame): Feature matrix.
#             y (pd.Series or np.array): Target values.
#         Returns:
#             RandomForestRegressor: Trained model.
#         """
#         rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
#         rf.fit(x, y)
#         return rf

#     @classmethod
#     def lightgbm(cls, x, y):
#         """
#         Train a LightGBM Regressor using Optuna for hyperparameter tuning.
#         Args:
#             x (pd.DataFrame): Feature matrix.
#             y (pd.Series or np.array): Target values.
#         Returns:
#             LGBMRegressor: Trained model.
#         """
#         def objective(trial):
#             # Define the hyperparameter search space
#             params = {
#                 'learning_rate': trial.suggest_float('learning_rate', 0.0005, 0.01, log=True),
#                 'n_estimators': trial.suggest_int('n_estimators', 800, 2000, step=200),
#                 'max_depth': trial.suggest_int('max_depth', 3, 10),
#                 'colsample_bytree': trial.suggest_float('colsample_bytree', 0.2, 0.8),
#                 'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
#                 'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
#                 'random_state': 42,
#             }

#             # Define the LightGBM model
#             model = LGBMRegressor(**params)
            
#             # Use TimeSeriesSplit for validation
#             cv = TimeSeriesSplit(n_splits=4)
            
#             # Evaluate with cross-validation
#             scores = cross_val_score(model, x, y, cv=cv, scoring="neg_mean_squared_error")
#             return -np.mean(scores)

#         # Create and optimize an Optuna study
#         study = optuna.create_study(direction="minimize")
#         study.optimize(objective, n_trials=50, show_progress_bar=True)

#         # Get the best hyperparameters
#         best_params = study.best_params
#         print("Best Hyperparameters:", best_params)

#         # Train and return the final model
#         best_model = LGBMRegressor(**best_params, random_state=42)
#         best_model.fit(x, y)
#         return best_model
    
  
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor
import optuna
import numpy as np

class LSTMModel(nn.Module):
            def __init__(self, input_size, output_size, hidden_size=50):
                super(LSTMModel, self).__init__()
                self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
                self.fc = nn.Linear(hidden_size, output_size)

            def forward(self, x):
                _, (hidden, _) = self.lstm(x)
                return self.fc(hidden[-1])
            

class CNNLSTMModel(nn.Module):
            def __init__(self, input_size, output_size, cnn_filters=64, lstm_hidden_size=50):
                super(CNNLSTMModel, self).__init__()
                self.conv = nn.Conv1d(input_size, cnn_filters, kernel_size=3, stride=1, padding=1)
                self.pool = nn.MaxPool1d(kernel_size=2)
                self.lstm = nn.LSTM(cnn_filters, lstm_hidden_size, batch_first=True)
                self.fc = nn.Linear(lstm_hidden_size, output_size)

            def forward(self, x):
                x = self.conv(x.transpose(1, 2))
                x = self.pool(x).transpose(1, 2)
                _, (hidden, _) = self.lstm(x)
                return self.fc(hidden[-1])  

class TCNBlock(nn.Module):
            def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.2):
                super(TCNBlock, self).__init__()
                self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, dilation=dilation, padding=dilation)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                return self.dropout(self.relu(self.conv1(x)))

class TCNModel(nn.Module):
            def __init__(self, input_size, output_size, num_channels, kernel_size=2, dropout=0.2):
                super(TCNModel, self).__init__()
                layers = []
                for i in range(len(num_channels)):
                    in_channels = input_size if i == 0 else num_channels[i - 1]
                    layers.append(TCNBlock(in_channels, num_channels[i], kernel_size, dilation=2**i, dropout=dropout))
                self.tcn = nn.Sequential(*layers)
                self.fc = nn.Linear(num_channels[-1], output_size)

            def forward(self, x):
                x = x.transpose(1, 2)
                x = self.tcn(x)
                x = x.mean(dim=-1)
                return self.fc(x)
                      
class TransformerModel(nn.Module):
        def __init__(self, input_size, output_size, d_model=64, nhead=4, num_layers=2, dim_feedforward=128, dropout=0.1):
            super(TransformerModel, self).__init__()
            self.embedding = nn.Linear(input_size, d_model)
            encoder_layer = nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout, batch_first=True)
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
            self.fc = nn.Linear(d_model, output_size)

        def forward(self, x):
            x = self.embedding(x)  # Map input to d_model dimensional space
            x = self.transformer(x)  # Transformer expects input in shape [batch, seq_len, d_model]
            return self.fc(x[:, -1, :])  # Use the output of the last token for regression
            




class model:
    @staticmethod
    def random_forest(x, y):
        """
        Train a Random Forest Regressor.
        """
        rf = MultiOutputRegressor(RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42))
        rf.fit(x, y)
        return rf
    
    @staticmethod
    def xgboost(x, y):
        """
        Train an XGBoost Regressor for multi-output regression.
        """
        from xgboost import XGBRegressor

        # Initialize the MultiOutputRegressor with XGBoost as the base model
        xgb_model = MultiOutputRegressor(
            XGBRegressor(
                objective='reg:squarederror',
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                random_state=42,
                tree_method='auto'
            )
        )

        # Train the model
        xgb_model.fit(x, y)

        return xgb_model

    @staticmethod
    def lstm(x, y, input_shape, epochs=50, batch_size=1):
        """
        Train an LSTM model for multi-output regression using PyTorch.
        """
        

        input_size = input_shape[-1]
        output_size = y.shape[1]
        model = LSTMModel(input_size, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        return model
    
    

    @staticmethod
    def cnn_lstm(x, y, input_shape, epochs=50, batch_size=1):
        """
        Train a CNN-LSTM model for multi-output regression using PyTorch.
        """

        input_size = input_shape[-1]
        output_size = y.shape[1]
        model = CNNLSTMModel(input_size, output_size)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        return model

    @staticmethod
    def tcn(x, y, input_shape, epochs=50, batch_size=1):
        """
        Train a Temporal Convolutional Network (TCN) model using PyTorch.
        """
        

        input_size = input_shape[-1]
        output_size = y.shape[1]
        num_channels = [16, 32, 64]
        model = TCNModel(input_size, output_size, num_channels)
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        return model
    
    @staticmethod
    def prepare_lstm_input(data, labels=None, lookback=50):
        """
        Prepare data for LSTM by creating sequences of length `lookback`.
        Args:
            data (np.array): Feature data, scaled and in numpy array format.
            labels (np.array, optional): Target data, scaled and in numpy array format. Defaults to None.
            lookback (int): Number of previous time steps to include in each sequence.
        Returns:
            tuple: Reshaped feature (X) and target (y) for LSTM models if labels are provided,
                   otherwise just the feature sequences (X).
        """
        if labels is not None:
            X_lstm, y_lstm = model.create_lstm_data(data, labels, lookback)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], X_lstm.shape[2]))
            return X_lstm, y_lstm
        else:
            # print("blag")
            X_lstm = model.create_lstm_data(data, lookback=lookback)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], X_lstm.shape[2]))
            return X_lstm

    @staticmethod
    def create_lstm_data(data, labels=None, lookback=50):
        """
        Create LSTM-ready sequences of length `lookback` for input data and corresponding target labels.
        Args:
            data (np.array): Feature data, scaled and in numpy array format.
            labels (np.array, optional): Target data, scaled and in numpy array format. Defaults to None.
            lookback (int): Number of previous time steps to include in each sequence.
        Returns:
            tuple: LSTM-ready input (X) and corresponding output (y) if labels are provided,
                   otherwise just the input sequences (X).
        """
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            if labels is not None:
                y.append(labels[i])
        
        if labels is not None:
            return np.array(X), np.array(y)
        else:
            # print("flag")
            return np.array(X)

    @staticmethod
    def transformer(x, y, input_shape, epochs=50, batch_size=1):
        """
        Train a Transformer model for multi-output regression using PyTorch.
        """

        
        # Model setup
        input_size = input_shape[-1]
        output_size = y.shape[1]
        model = TransformerModel(input_size, output_size)

        # Define loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        # Prepare input tensors
        x_tensor = torch.tensor(x, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.float32)

        # Ensure data shape is batch-first
        x_tensor = x_tensor.view(-1, x.shape[1], x.shape[2])  # [batch_size, seq_len, input_dim]
        y_tensor = y_tensor.view(-1, y.shape[1])  # [batch_size, output_dim]

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")

        # Switch model to evaluation mode
        model.eval()
        return model

    @staticmethod
    def lightgbm(x, y):
        """
        Train a LightGBM Regressor using Optuna for hyperparameter tuning.
        """
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
            }
            model = MultiOutputRegressor(LGBMRegressor(**params))
            cv = TimeSeriesSplit(n_splits=4)
            scores = []
            for train_idx, test_idx in cv.split(x):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(x_train, y_train)
                preds = model.predict(x_test)
                scores.append(mean_squared_error(y_test, preds))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=10)
        best_params = study.best_params
        final_model = MultiOutputRegressor(LGBMRegressor(**best_params))
        final_model.fit(x, y)
        return final_model
    @staticmethod
    def lightgbm(x, y):
        """
        Train a LightGBM Regressor using Optuna for hyperparameter tuning.
        """
        def objective(trial):
            params = {
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 30),
                "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.2),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            }
            model = MultiOutputRegressor(LGBMRegressor(**params))
            cv = TimeSeriesSplit(n_splits=4)
            scores = []
            for train_idx, test_idx in cv.split(x):
                x_train, x_test = x[train_idx], x[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                model.fit(x_train, y_train)
                preds = model.predict(x_test)
                scores.append(mean_squared_error(y_test, preds))
            return np.mean(scores)

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=50)  # Increased trials
        best_params = study.best_params
        final_model = MultiOutputRegressor(LGBMRegressor(**best_params))
        final_model.fit(x, y)
        return final_model

    

class fine_tune_model:
    @staticmethod
    def fine_tune_random_forest(model, x_new, y_new):
        """
        Fine-tune a pre-trained Random Forest Regressor.
        """
        model.estimators_ = model.estimators_[:len(model.estimators_) // 2]  # Optional: reduce complexity
        model.fit(x_new, y_new)  # Fit on additional data
        return model
    
    
    @staticmethod
    def fine_tune_xgboost(model, x_new, y_new):
        """
        Fine-tune a pre-trained XGBoost Regressor.
        """
        for estimator in model.estimators_:
            estimator.fit(x_new, y_new, xgb_model__verbose=False)  # Update each base model
        return model


    @staticmethod
    def fine_tune_lstm(model, x_new, y_new, epochs=10, batch_size=32):
        """
        Fine-tune a pre-trained LSTM model with additional data.
        """
        x_tensor = torch.tensor(x_new, dtype=torch.float32)
        y_tensor = torch.tensor(y_new, dtype=torch.float32)
        
        model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            print(f"Fine-Tune Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        return model

    

    @staticmethod
    def fine_tune_cnn_lstm(model, x_new, y_new, epochs=10, batch_size=32):
        """
        Fine-tune a pre-trained CNN-LSTM model with additional data.
        """
        return model.fine_tune_lstm(model, x_new, y_new, epochs, batch_size)  # Similar to LSTM


    @staticmethod
    def fine_tune_tcn(model, x_new, y_new, epochs=10, batch_size=32):
        """
        Fine-tune a pre-trained TCN model with additional data.
        """
        x_tensor = torch.tensor(x_new, dtype=torch.float32)
        y_tensor = torch.tensor(y_new, dtype=torch.float32)
        
        model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            print(f"Fine-Tune Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        return model

    
    @staticmethod
    def prepare_lstm_input(data, labels=None, lookback=50):
        """
        Prepare data for LSTM by creating sequences of length `lookback`.
        Args:
            data (np.array): Feature data, scaled and in numpy array format.
            labels (np.array, optional): Target data, scaled and in numpy array format. Defaults to None.
            lookback (int): Number of previous time steps to include in each sequence.
        Returns:
            tuple: Reshaped feature (X) and target (y) for LSTM models if labels are provided,
                   otherwise just the feature sequences (X).
        """
        if labels is not None:
            X_lstm, y_lstm = model.create_lstm_data(data, labels, lookback)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], X_lstm.shape[2]))
            return X_lstm, y_lstm
        else:
            # print("blag")
            X_lstm = model.create_lstm_data(data, lookback=lookback)
            X_lstm = X_lstm.reshape((X_lstm.shape[0], X_lstm.shape[1], X_lstm.shape[2]))
            return X_lstm

    @staticmethod
    def create_lstm_data(data, labels=None, lookback=50):
        """
        Create LSTM-ready sequences of length `lookback` for input data and corresponding target labels.
        Args:
            data (np.array): Feature data, scaled and in numpy array format.
            labels (np.array, optional): Target data, scaled and in numpy array format. Defaults to None.
            lookback (int): Number of previous time steps to include in each sequence.
        Returns:
            tuple: LSTM-ready input (X) and corresponding output (y) if labels are provided,
                   otherwise just the input sequences (X).
        """
        X, y = [], []
        for i in range(lookback, len(data)):
            X.append(data[i - lookback:i])
            if labels is not None:
                y.append(labels[i])
        
        if labels is not None:
            return np.array(X), np.array(y)
        else:
            # print("flag")
            return np.array(X)

    @staticmethod
    def fine_tune_transformer(model, x_new, y_new, epochs=10, batch_size=32):
        """
        Fine-tune a pre-trained Transformer model with additional data.
        """
        x_tensor = torch.tensor(x_new, dtype=torch.float32)
        y_tensor = torch.tensor(y_new, dtype=torch.float32)
        
        model.train()
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = model(x_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            print(f"Fine-Tune Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
        return model


    @staticmethod
    def fine_tune_lightgbm(model, x_new, y_new):
        """
        Fine-tune a pre-trained LightGBM model.
        """
        for estimator in model.estimators_:
            estimator.fit(x_new, y_new, eval_set=[(x_new, y_new)], verbose=0)
        return model

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




def compute_technical_indicators(df,delay):
    """
    Compute technical indicators based on kline data (OHLCV).
    """
    # Ensure there is sufficient data for technical indicators
    if len(df) < 20:  # 20 is a common lookback period
        return df

    try:
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss.replace({0: np.nan})  # Prevent division by zero
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD (Moving Average Convergence Divergence)
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

        # Bollinger Bands
        rolling_mean = df['close'].rolling(window=20).mean()
        rolling_std = df['close'].rolling(window=20).std()
        df['bollinger_upper'] = rolling_mean + (2 * rolling_std)
        df['bollinger_lower'] = rolling_mean - (2 * rolling_std)

        # Stochastic Oscillator
        low_14 = df['low'].rolling(window=14).min()
        high_14 = df['high'].rolling(window=14).max()
        diff = high_14 - low_14
        df['stoch_k'] = 100 * (df['close'] - low_14) / diff.replace({0: np.nan})  # Prevent division by zero
        df['stoch_d'] = df['stoch_k'].rolling(window=3).mean()

        # ATR (Average True Range)
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = true_range.rolling(window=14).mean()

        # SMA and EMA
        df['sma_10'] = df['close'].rolling(window=10).mean()
        df['ema_10'] = df['close'].ewm(span=10, adjust=False).mean()

        # Rate of Change
        df['rate_of_change'] = df['close'].pct_change(periods=10)

        # VWAP (Volume Weighted Average Price)
        cumulative_volume = df['volume'].cumsum()
        df['vwap'] = (df['close'] * df['volume']).cumsum() / cumulative_volume.replace({0: np.nan})  # Prevent division by zero

        # Donchian Channels
        df['donchian_upper'] = df['high'].rolling(window=20).max()
        df['donchian_lower'] = df['low'].rolling(window=20).min()
        df['donchian_mid'] = (df['donchian_upper'] + df['donchian_lower']) / 2

        # Williams %R
        df['williams_r'] = -100 * (high_14 - df['close']) / diff.replace({0: np.nan})  # Prevent division by zero
        # future_shift = int(10 / delay)  # Number of rows to shift for 10 seconds
        # if future_shift < len(df):
        #     df['future_bid_price'] = df['bid_price'].shift(-future_shift)
        #     # df['future_bid_qty'] = df['bid_qty'].shift(-future_shift)
        #     df['future_ask_price'] = df['ask_price'].shift(-future_shift)
        #     # df['future_ask_qty'] = df['ask_qty'].shift(-future_shift)
        return df

    except Exception as e:
        print(f"Error in computing technical indicators: {e}")
        return df
def add_targets(df,delay):
    """
    Add target variables to the dataframe.
    """
    future_shift = int(300 / delay)  # Number of rows to shift for 10 seconds
    if future_shift < len(df):
        df['future_bid_price'] = df['bid_price'].shift(-future_shift)
        # df['future_bid_qty'] = df['bid_qty'].shift(-future_shift)
        df['future_ask_price'] = df['ask_price'].shift(-future_shift)
        # df['future_ask_qty'] = df['ask_qty'].shift(-future_shift)
        df['future_timestamp'] = df['timestamp'].shift(-future_shift)
    return df  

def add_targets(df, delay):
    """
    Add target variables to the dataframe based on future timestamps (299, 300, or 301 seconds ahead).
    """
    # Ensure dataframe is sorted by timestamp
    df = df.sort_values(by="timestamp").reset_index(drop=True)

    # Convert 'timestamp' to datetime for accurate calculations
    def parse_single(value):
            try:
                # Attempt parsing with microseconds
                return pd.to_datetime(value, format='%Y-%m-%d %H:%M:%S.%f', utc=True)
            except ValueError:
                try:
                    # Fallback to seconds-only
                    return pd.to_datetime(value, format='%Y-%m-%d %H:%M:%S', utc=True)
                except ValueError:
                    try:
                        # Fallback to date-only format
                        return pd.to_datetime(value, format='%Y-%m-%d', utc=True)
                    except ValueError:
                        # Return NaT if all formats fail
                        return pd.NaT
    df["timestamp"]=df["timestamp"].apply(parse_single)               
   #  df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Initialize target columns
    df["future_bid_price"] = None
    df["future_ask_price"] = None
    df["future_timestamp"] = None

    # Iterate through each row and find the appropriate future timestamp
    for i, row in df.iterrows():
        original_time = row["timestamp"]
        target_times = [original_time + pd.Timedelta(seconds=60),
                        original_time + pd.Timedelta(seconds=61),
                        original_time + pd.Timedelta(seconds=62)]

        # Find the closest matching future timestamp within the dataframe
        future_rows = df[df["timestamp"].isin(target_times)]

        if not future_rows.empty:
            # Use the first match within the range
            future_row = future_rows.iloc[0]
            df.at[i, "future_bid_price"] = future_row["bid_price"]
            df.at[i, "future_ask_price"] = future_row["ask_price"]
            df.at[i, "future_timestamp"] = future_row["timestamp"]

    # Return the updated dataframe
    return df


