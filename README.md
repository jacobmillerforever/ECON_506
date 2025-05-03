# ETF Market Behavior Prediction

This project applies machine‑learning techniques to predict short‑term market behavior of the **SPY** ETF, classifying each day as **trending** (> ±0.5 % move) or **oscillating** (±0.5 % or smaller). While SPY is the prediction target, a broad set of global asset‑class signals, macroeconomic indicators, and technical‑analysis (TA) features are engineered to give the model a richer view of market conditions.

---
## Project Overview
- **Goal** – Improve precision for the oscillation class by **≥ 12 pp** versus a naïve baseline.
- **Data horizon** – 15–25 years of daily data (​≈ 2000 – 2024​).
- **Methods compared** – LSTM, Random Forest, k‑NN, Logistic Regression, MLP.
- **Validation** – Expanding‑window CV; 2024 is the final out‑of‑sample test set.

---
## Market Indices and Tickers Used
| Asset class | Index (human‑readable) | Yahoo Finance ticker | Definition |
|-------------|------------------------|----------------------|------------|
| **Equity**  | Nikkei 225 (Japan) | `^N225` | Price‑weighted index of 225 blue‑chip companies listed on the Tokyo Stock Exchange, representing Japan’s large‑cap equity market.|
|             | DAX 40 (Germany) | `^GDAXI` | Free‑float‑adjusted performance index of the 40 largest German companies traded on the Frankfurt Stock Exchange.|
|             | EURO STOXX 50 (Eurozone) | `^STOXX50E` | Capitalisation‑weighted index of 50 leading blue‑chip companies from 11 Eurozone countries.|
|             | **SPDR S&P 500 ETF Trust** | `SPY` | Exchange‑traded fund tracking the S&P 500; used here as the prediction target.|
| **Volatility** | CBOE S&P 500 Volatility Index | `^VIX` | Market‑implied 30‑day volatility derived from S&P 500 option prices (often dubbed the “fear gauge”).|
| **Currency** | ICE US Dollar Index | `DX‑Y.NYB` | Measures the USD’s value versus a basket of six major currencies (EUR, JPY, GBP, CAD, SEK, CHF).|
| **Commodity** | COMEX Gold Futures (front‑month) | `GC=F` | Continuous front‑month futures on gold (troy ounces) traded on the COMEX division of the CME Group.|

---
## Macroeconomic Indicators (FRED Series)
| Series ID | Description | Why it matters |
|-----------|-------------|----------------|
| **DFF** | *Federal Funds Effective Rate* – the overnight rate at which depository institutions lend balances at the Federal Reserve to other institutions. | Proxy for U.S. monetary‑policy stance.|
| **T10Y2Y** | *10‑Year Treasury minus 2‑Year Treasury Constant‑Maturity Spread*. Often cited as the yield‑curve slope. | Negative values have historically preceded recessions; gauges growth expectations.|
| **CPIAUCSL** | *Consumer Price Index for All Urban Consumers (SA)*. | Headline U.S. inflation measure; informs real‑rate calculations.|
| **UNRATE** | *Civilian Unemployment Rate*. | Key labour‑market slack indicator influencing Fed policy.|
| **STLFSI** | *St. Louis Fed Financial Stress Index*. | Composite of spreads & volatilities capturing systemic stress.|
| **M2SL** | *M2 Money Stock* (seasonally adjusted). | Broad money supply; growth signals liquidity conditions.|
| **USSLIND** | *U.S. Leading Index* (Conference Board). | Forward‑looking aggregate of economic indicators.|
| **BAMLH0A0HYM2** | *ICE BofA US High‑Yield Index Option‑Adjusted Spread*. | Risk‑premium on sub‑investment‑grade debt; widens during risk‑off episodes.|
| **GS5** | *5‑Year Treasury Constant Maturity Rate*. | Mid‑term risk‑free rate, used in term‑structure signals.|
| **GS30** | *30‑Year Treasury Constant Maturity Rate*. | Long‑end yield, complements GS5 in curve‑steepness measures.|
| **BAMLC0A0CM** | *ICE BofA US Corporate BBB Option‑Adjusted Spread*. | Investment‑grade credit spread; tightens/widens with risk appetite.|

---
## Technical‑Analysis Features
### Price‑Based Indices (All symbols except VIX)
For each index we compute the following daily features (window lengths in parentheses):
| Feature | Definition | Predictive intuition |
|---------|------------|----------------------|
| **gap1** | `(Open − Prev_Close) / Prev_Close` | Overnight sentiment shift; large gaps often precede intraday continuation or mean reversion.|
| **ret1 / ret5** | 1‑day and 5‑day simple returns. | Short‑ & medium‑term momentum signals.|
| **range1** | `(High − Low) / Close`. | Intraday volatility proxy; wide ranges can foreshadow breakouts.|
| **sma10** | 10‑day simple moving average of close. | Short‑term trend direction.|
| **ema5 / ema20** | 5‑ & 20‑day exponential MAs. | Faster reaction to trend changes; their **ema_spread** (ema5 − ema20) captures trend acceleration.|
| **vol10** | 10‑day rolling mean of volume. | Participation confirmation; rising volume can validate price moves.|
| **atr14** | 14‑day *Average True Range*. | Volatility magnitude; spikes often precede regime shifts.|
| **bb_width** | Width of 20‑day Bollinger Bands as fraction of mid‑band. | Squeezes (narrow width) hint at upcoming volatility expansion.|
| **rsi14** | 14‑day *Relative Strength Index*. | Momentum oscillator; extreme values indicate overbought/oversold.|
| **macd / macd_signal** | *Moving Average Convergence Divergence* (12‑26‑9) and its 9‑period signal line. | Crossovers capture momentum inflection.|
| **vol_ma10** | 10‑day SMA of volume. | Smooths raw volume to filter noise.|
| **obv** | *On‑Balance Volume* cumulative sum. | Combines price direction & volume to gauge buying/selling pressure.|

### VIX‑Specific Features
The VIX behaves fundamentally differently from price indices (it is already a volatility metric, exhibits strong mean‑reversion, and is bounded below by zero). Accordingly, a tailored feature set is engineered:
| Feature | Definition |
|---------|------------|
| **ma20** | 20‑day SMA of VIX close.|
| **std20** | 20‑day rolling standard deviation.|
| **zscore** | `(Close − ma20) / std20` – standardised deviation from recent mean.|
| **rsi14** | Momentum oscillator applied to VIX (captures volatility regime shifts).|
| **bb_width** | Bollinger‑band width of VIX (vol‑of‑vol signal).|
| **acf1** | Rolling 5‑day lag‑1 autocorrelation.|

> **Why a separate VIX feature set?**  
> Price‑trend indicators (moving‑average crossovers, OBV, etc.) have limited meaning on volatility indices, which are not traded like equities and revert to a long‑run mean. Instead, dispersion, z‑scores, and autocorrelation better describe VIX dynamics and their spill‑over into equity markets.

---
## Calendar Event Categories
Below‑the‑surface price moves are often catalyzed by scheduled macro releases and Federal Reserve communications. We tag each trading day with dummies for these categories, plus ±3‑day leads/lags so the model learns anticipatory and post‑announcement behaviour.

| Category | Labels mapped | What happens & why it matters |
|----------|---------------|-------------------------------|
| **CPI** (`cpi`) | “CPI” | Monthly inflation print; surprises shift real‑rate & policy expectations.|
| **Employment** (`employment`) | “Non‑Farm Payrolls”, “Unemployment Rate” | Labour‑market snapshot; drives growth outlook and Fed reaction function.|
| **Fed Meeting** (`fed_meeting`) | “FOMC Rate Decision”, “FOMC Statement”, “FOMC Minutes” | Eight scheduled meetings; set policy rate & tone; minutes give extra colour.|
| **Fed Projections** (`fed_proj`) | “Summary of Economic Projections” | Quarterly dot‑plot & forecasts; re‑prices yield curve on median‑dot shifts.|

---
## Data Sources
* **Yahoo Finance** – Historical OHLCV data for indices via *yfinance*.
* **FRED** – Macroeconomic series listed above via *fredapi*.
* **Investing.com** – Calendar for macro‑event dummies.

---
## Model Pipeline (High Level)
1. **Feature engineering** (TA & macro series synchronised to a daily calendar).
2. **Sample‑label construction** – classify each day’s SPY close as oscillating or trending.
3. **Train/test splits** – expanding windows ending 2023‑12‑31; 2024 is held out.
4. **Imbalance handling** – minority‑class oversampling with **SMOTE** inside each training fold.
5. **Model training** – hyperparameter‑tuned LSTM (see architecture below) & baseline models.
6. **Threshold selection** – precision‑constrained search (< 0.99) to maximise recall subject to ≥ 0.80 precision.

---
## LSTM Model Architecture
```
Input:  (batch, lookback, n_features)  # e.g., (None, 30, 85)
│
├── LSTM  (224 units, return_sequences=True)
│   └─ dropout 0.25
├── LSTM  (96 units)
│   └─ dropout 0.25
├── Dense (128 units, ReLU)
│   └─ dropout 0.3
└── Dense (1 unit, Sigmoid)
Output: probability of oscillation
```
* **Sequential stacking** lets the first LSTM learn low‑level temporal patterns while the second captures higher‑order dependencies.
* **Dropout** layers combat overfitting on correlated market data.
* **Dense‑128 bottleneck** translates temporal features into a compact representation before binary classification.
* **Adam** optimiser with learning‑rate scheduling; **binary‑cross‑entropy** loss; **Precision** tracked as primary metric.

The tuned model delivered a **12.34 pp** precision gain over baseline, with **0.798 precision** and **0.734 accuracy** on the 2024 hold‑out.

---
## Running Locally with Conda
```cmd
# 1) Clone the repository
git clone https://github.com/jacobmillerforever/ECON_506.git

# 2) Create and activate the environment
conda env create -f environment.yml
conda activate tf_env

# 3) Launch Jupyter
jupyter notebook
# Open 506_Final_Project.ipynb and run all cells
```

---
## Running on Google Colab
1. Open **Google Colab** → *File ▸ Upload notebook* → choose `506_Final_Project.ipynb`.
2. Install dependencies:
   ```python
   !pip install fredapi investpy ta keras_tuner
   ```
3. *(Optional)* Enable GPU: *Runtime ▸ Change runtime type ▸ Hardware accelerator ▸ GPU*.
4. *Runtime ▸ Run all* to execute the notebook.

---
## Results Snapshot
| Model | Precision‑lift vs naïve |
|-------|------------------------|
| **LSTM** | **+12.3 pp** |
| Logistic Regression | +4.3 pp |
| k‑NN | +2.0 pp |
| Random Forest | −0.8 pp |
| MLP Classifier | +4.1 pp |
