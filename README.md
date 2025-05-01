# ETF Market Behavior Prediction

This project applies machine learning techniques to predict short-term market behavior of major ETFs (SPY, IWM, QQQ, DIA), distinguishing between trending and oscillating states. The focus is on predicting whether SPY will oscillate (daily price movements within 0.5%) or trend (movements exceeding 0.5%).

## Project Overview

- **Goal**: Predict market oscillation for SPY ETF with at least 12% improvement over a naive classifier
- **Data**: 15-25 years of historical market data with macroeconomic indicators and technical variables
- **Methods**: Multiple ML models compared (LSTM, Random Forest, KNN, Logistic Regression, MLP)
- **Validation**: Rolling window cross-validation with 2024 data used for evaluation

## Results Summary

The LSTM model achieved the best performance:
- **Precision improvement over naive predictor**: 12.34% (exceeding the minimum 12% requirement)
- **Accuracy**: 0.734
- **Precision for oscillation class**: 0.798

Other models' precision improvements:
- Logistic Regression: 4.26%
- KNN: 1.95%
- Random Forest: -0.79%
- MLP Classifier: 4.13%

## Running Locally with Conda

1. Clone the repository:
```cmd
git clone https://github.com/jacobmillerforever/ECON_506.git
```
2. Create and activate the conda environment:
```cmd
conda env create -f environment.yml
conda activate tf_env
```
3. Run Jupyter Notebook:
```cmd
jupyter notebook
```
4. Open the 506_Final_Project.ipynb notebook and run all cells.



## Running on Google Colab

1. Open Google Colab: [https://colab.research.google.com/](https://colab.research.google.com/)

2. Upload the notebook via File → Upload notebook → Choose `506_Final_Project.ipynb`

3. Execute the necessary package installations at the beginning of the notebook:
```python
!pip install fredapi
!pip install investpy
!pip install ta
!pip install keras_tuner
```
Run all cells in the notebook (Runtime → Run all)
Note: To access GPU acceleration on Colab, select Runtime → Change runtime type → Hardware accelerator → GPU

## Data Sources
The project uses multiple data sources:

ETF price data from Yahoo Finance (yfinance)
Economic indicators from FRED (Federal Reserve Economic Data)
Calendar dates for economic events from Investing.com

## Key Features
The prediction model leverages several important feature categories:

Technical indicators (RSI, MACD, Bollinger Bands, etc.)
Market volatility metrics (VIX)
Macroeconomic indicators (interest rates, CPI, unemployment)
Calendar events (Fed meetings, economic announcements)
Temporal features (day of week, month)

## Model Architecture
The best-performing LSTM model uses:

Two LSTM layers (224 and 96 units) with dropout
Dense layer with 128 units
Training with class weights to address imbalance
Multi-seed training strategy to optimize stability
