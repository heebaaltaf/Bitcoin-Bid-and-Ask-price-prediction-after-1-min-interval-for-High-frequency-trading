# Bitcoin Bid and Ask Price Prediction After 1-Minute Interval for High-Frequency Trading

This project provides a machine learning and deep learning-based pipeline to predict the future bid and ask prices of Bitcoin with a 1-minute interval. It is designed for high-frequency trading scenarios, enabling efficient decision-making and strategy optimization.

---

## Features

### 1. Real-Time Data Fetching
- Fetches real-time order book, kline, and trade data from the Binance API.
- Maintains a sliding window of recent data for efficient memory usage.

### 2. Data Preprocessing and Feature Engineering
- Implements technical indicators such as RSI, MACD, Bollinger Bands, and more.
- Supports lag and rolling features for numeric data.
- Handles missing data and provides robust preprocessing pipelines.

### 3. Model Training and Prediction
- **Models supported**:
  - Random Forest
  - LightGBM
  - LSTM
  - CNN-LSTM
  - Temporal Convolutional Networks (TCN)
  - Transformer
- Supports multi-output regression for predicting bid and ask prices simultaneously.

### 4. Continuous Retraining and Backtesting
- Periodically retrains models with updated data.
- Evaluates model performance through backtesting on the latest test set.

### 5. Threaded Execution
- Implements multithreading for continuous data fetching, training, and backtesting.

---

## Prerequisites

### System Requirements
- Python 3.8 or higher
- Stable internet connection for API requests

### Dependencies
Install dependencies using:
```bash
pip install -r requirements.txt
