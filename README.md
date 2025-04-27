# ETF Trend vs Oscillation Prediction Project

## Project Overview

This project implements machine learning models to predict whether the SPY ETF will trend (>0.5% daily price movement) or oscillate (≤0.5% daily price movement) using historical market data and technical indicators.

## Current Results

The project currently achieves approximately 4% improvement over the naive predictor (67.3% baseline) for SPY.

## Approach

### Data Collection
- Historical market data (2007-2024) downloaded via yfinance for:
  - Global indices (Nikkei 225, Hang Seng, SSE Composite, ASX 200, DAX, FTSE 100, CAC 40, Euro Stoxx 50, SPY)
  - Volatility indices (VIX, VIX Brazil, DAX Volatility)
  - Currency pairs (US Dollar Index, EUR/USD, JPY/USD, CNY/USD)
  - Commodities (Gold, Crude Oil, Silver, Corn, Copper)
- Economic indicators from FRED API:
  - Federal Funds Rate
  - Treasury yield curves
  - CPI, unemployment rate
  - Financial stress indices
- High-importance economic calendar events from investing.com

### Feature Engineering
- Technical indicators calculated for all assets:
  - RSI (14-day)
  - ATR (14-day)
  - Bollinger Bands
  - MACD
  - Moving averages (SMA/EMA)
  - Momentum indicators
  - Stochastic oscillators
- Different data handling based on market timing:
  - Asian markets: Current day technical indicators
  - European markets: Lag-1 technical indicators + current open
  - SPY: Lag-1 technical indicators + current open + lag-1 raw values
  - Other assets: Lag-1 technical indicators + current open

### Models Implemented

1. **XGBoost Classifier**
   - Initial run with all features
   - Feature selection based on importance (top features covering 98% importance)
   - Hyperparameters: max_depth=6, learning_rate=0.1, n_estimators=100

2. **Logistic Regression**
   - Run with selected features
   - Class weight balancing applied
   - Max iterations: 1000

3. **LSTM Neural Network**
   - Hyperparameter tuning using Keras Tuner
   - Architecture: Two LSTM layers with dropout
   - Optimized learning rate and unit counts
   - Class weight balancing for imbalanced data

### Evaluation
- Train/test split: 2007-2023 for training, 2024 for testing
- Target: SPY oscillation (absolute daily % change ≤ 0.5%)
- Baseline accuracy: 67.3% (naive predictor)
- Current best improvement: ~4% (LSTM model)
- Metrics: Accuracy, precision, recall, F1-score, confusion matrices

## Dependencies
tensorflow
numpy
pandas
yfinance
fredapi
investpy
scikit-learn
xgboost
ta
keras_tuner
matplotlib
seaborn

## Current Performance Summary

| Model                | Accuracy | Improvement over Naive |
|---------------------|----------|------------------------|
| Logistic Regression | 65.87%   | -1.59%                |
| XGBoost             | 64.29%   | -3.17%                |
| LSTM                | 71.43%   | 3.97%                 |

## Usage

1. Install required packages
2. Run the Jupyter notebook on Google Colab with GPU acceleration
3. Ensure FRED API key is available in the Colab environment

## Challenges and Next Steps

- Current performance falls short of the 12% improvement target for full credit
- Potential improvements:
  - Feature engineering refinement
  - Additional hyperparameter tuning
  - Try more advanced architectures (CNNs, attention mechanisms)
  - Explore different thresholds for oscillation definition
  - Implement ensemble methods to combine model predictions

## Author

Jacob Miller and Rachel Tan

## Course

ECON 506 - Research Design
