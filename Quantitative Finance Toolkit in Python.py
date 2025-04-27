# Quantitative Finance Toolkit in Python

# --- Libraries ---
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression

# --- Data Collection ---
data = yf.download("AAPL", start="2020-01-01", end="2024-12-31")
data['Returns'] = data['Adj Close'].pct_change().dropna()

# --- Descriptive Stats ---
print("Mean Return:", data['Returns'].mean())
print("Volatility:", data['Returns'].std())
print("Skewness:", data['Returns'].skew())
print("Kurtosis:", data['Returns'].kurtosis())
sns.histplot(data['Returns'], kde=True)
plt.title("Return Distribution")
plt.show()

# --- CAPM ---
market = yf.download("^GSPC", start="2020-01-01", end="2024-12-31")['Adj Close']
market_returns = market.pct_change().dropna()
stock_returns = data['Returns'].loc[market_returns.index]
X = sm.add_constant(market_returns)
capm_model = sm.OLS(stock_returns, X).fit()
print(capm_model.summary())

# --- Efficient Frontier ---
tickers = ['AAPL', 'MSFT', 'GOOGL', 'TSLA']
prices = yf.download(tickers, start="2021-01-01", end="2024-12-31")['Adj Close']
returns = prices.pct_change().dropna()
mean_returns = returns.mean()
cov_matrix = returns.cov()
weights = np.random.dirichlet(np.ones(len(tickers)), 10000)
port_returns = np.dot(weights, mean_returns)
port_vols = [np.sqrt(np.dot(w.T, np.dot(cov_matrix, w))) for w in weights]
plt.scatter(port_vols, port_returns, alpha=0.3)
plt.xlabel('Volatility')
plt.ylabel('Expected Return')
plt.title('Efficient Frontier')
plt.show()

# --- Value at Risk (VaR) ---
confidence = 0.95
VaR_95 = np.percentile(data['Returns'].dropna(), (1 - confidence) * 100)
print(f"95% Historical VaR: {VaR_95:.4%}")

# --- Monte Carlo Simulation ---
S0 = data['Adj Close'][-1]
mu = data['Returns'].mean()
sigma = data['Returns'].std()
T = 252
N = 1000
simulations = np.zeros((T, N))
for i in range(N):
    daily_returns = np.random.normal(mu, sigma, T)
    simulations[:, i] = S0 * np.cumprod(1 + daily_returns)
plt.plot(simulations)
plt.title("Monte Carlo Simulated Stock Prices")
plt.show()

# --- Moving Average Strategy ---
data['SMA50'] = data['Adj Close'].rolling(50).mean()
data['SMA200'] = data['Adj Close'].rolling(200).mean()
data['Signal'] = 0
data.loc[data['SMA50'] > data['SMA200'], 'Signal'] = 1
data['Position'] = data['Signal'].shift(1)
data['StrategyReturns'] = data['Position'] * data['Returns']
cum_strategy = (1 + data['StrategyReturns'].dropna()).cumprod()
cum_stock = (1 + data['Returns'].dropna()).cumprod()
plt.plot(cum_stock, label='Buy & Hold')
plt.plot(cum_strategy, label='Strategy')
plt.legend()
plt.title("Backtest: SMA Crossover Strategy")
plt.show()

# --- Black-Scholes Option Pricing ---
def black_scholes(S, K, T, r, sigma, option_type='call'):
    from scipy.stats import norm
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    if option_type == 'call':
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

# Example Usage
bs_price = black_scholes(S=100, K=100, T=1, r=0.05, sigma=0.2, option_type='call')
print(f"Black-Scholes Call Price: {bs_price:.2f}")

# --- Factor Models (Fama-French Example Placeholder) ---
# You can download Fama-French factors from Kenneth French’s data library or use a library like 'ffn'.
# Example with dummy factors:
factors = pd.DataFrame({
    'MKT': np.random.normal(0.001, 0.02, len(stock_returns)),
    'SMB': np.random.normal(0.0005, 0.01, len(stock_returns)),
    'HML': np.random.normal(0.0003, 0.01, len(stock_returns))
})
X_ff = sm.add_constant(factors)
ff_model = sm.OLS(stock_returns[:len(factors)], X_ff).fit()
print(ff_model.summary())

# --- Risk Parity Portfolio Example ---
inv_vol_weights = 1 / returns.std()
inv_vol_weights /= inv_vol_weights.sum()
print("Inverse Volatility Portfolio Weights:")
print(inv_vol_weights)

# --- Placeholder for Kalman Filter / ML Forecasting ---
# These require larger setups: use pykalman or sklearn for time-series models like ARIMA, LSTM, etc.
