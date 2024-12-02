
## Bitcoin Bid and Ask Price Prediction After 1-Minute Interval for High-Frequency Trading
This project is a machine learning and deep learning-based pipeline for predicting the future bid and ask prices of Bitcoin with a 1-minute interval. It is designed for high-frequency trading scenarios, enabling efficient decision-making and strategy optimization.

### Features
### Real-Time Data Fetching:

Fetches real-time order book, kline, and trade data from Binance API.
Maintains a sliding window of recent data for efficient memory usage.
### Data Preprocessing and Feature Engineering:

Implements technical indicators such as RSI, MACD, Bollinger Bands, and more.
Supports lag and rolling features for numeric data.
Handles missing data and provides robust preprocessing pipelines.
### Model Training and Prediction:

Models supported:
Random Forest
LightGBM
LSTM
CNN-LSTM
Temporal Convolutional Networks (TCN)
Transformer
Supports multi-output regression for predicting bid and ask prices simultaneously.
### Continuous Retraining and Backtesting:

Periodically retrains models with updated data.
Evaluates model performance through backtesting on the latest test set.
### Threaded Execution:

Implements multithreading for continuous data fetching, training, and backtesting.
Prerequisites
System Requirements
Python 3.8 or higher
Stable internet connection for API requests
### Dependencies
### Install dependencies using:

bash
Copy code

pip install -r requirements.txt

Installation

### Clone the repository:

bash
Copy code
git clone https://github.com/your-username/Bitcoin-Bid-and-Ask-price-prediction-after-1-min-interval-for-High-frequency-trading.git

cd Bitcoin-Bid-and-Ask-price-prediction-after-1-min-interval-for-High-frequency-trading

### Install required libraries:

bash
Copy code

pip install -r requirements.txt

### Set up Binance API keys in the main.py file:

python
Copy code

API_KEY = 'your_api_key'

API_SECRET = 'your_api_secret'

### Usage
### Step 1: Data Fetching
### Run the data fetching script to collect real-time market data:

bash
Copy code

python main.py

### Step 2: Training Models
### Automatically trains models with the latest fetched data:

Models are retrained periodically based on the specified interval.
Fine-tuning is supported for pre-trained models.

### Step 3: Backtesting
### Evaluates the performance of trained models:

Provides prediction metrics and error analysis.
Outputs results for strategy validation.
### Configuration
### Symbol: Change the cryptocurrency symbol to match your use case:

python
Copy code

symbol = "BTCUSDT"

### Interval: Modify the fetching interval:

python
Copy code

fetch_delay = 1  # Fetch every 1 second

### Training Interval: Set the retraining frequency:

python
Copy code

training_interval = 300  # Retrain every 5 minutes

### Target Labels: Define prediction targets:

python
Copy code

target_labels = ['future_bid_price', 'future_ask_price']

Outputs
### Trained Models:

Stored in .pth and .joblib formats in the models/ directory.

### Backtesting Results:

Metrics such as mean squared error and R-squared values.

Logs of prediction performance.

File Structure

python

Copy code

Bitcoin-Bid-and-Ask-price-prediction-after-1-min-interval-for-High-frequency-trading/

│
├── data/                     # Raw and processed data
├── models/                   # Trained model files
├── utils/                    # Utility scripts for preprocessing and feature engineering
├── train.py                  # Training script
├── test.py                   # Backtesting script
├── main.py                   # Main execution script
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
