pip install yfinance numpy scipy matplotlib plotly
import yfinance as yf
import numpy as np

# Choose a stock
ticker = "AAPL"
stock = yf.Ticker(ticker)

# Get live data
current_price = stock.history(period="1d")['Close'].iloc[-1]

# Get historical data for volatility estimate
hist = stock.history(period="6mo")
returns = np.log(hist['Close'] / hist['Close'].shift(1)).dropna()
vol = np.std(returns) * np.sqrt(252)  # Annualized volatility

print(f"Live Price: {current_price:.2f}")
print(f"Estimated Volatility: {vol:.4f}")

from pricing.black_scholes import black_scholes_price
from greeks.bs_greeks import bs_greeks

S = current_price
K = round(S, 0)  # Use ATM strike
T = 30 / 365     # 30 days to expiry
r = 0.05         # Assume 5% risk-free rate
sigma = vol

price = black_scholes_price(S, K, T, r, sigma, option_type='call')
greeks = bs_greeks(S, K, T, r, sigma)

print(f"Option Price: {price:.2f}")
print("Greeks:")
for g, val in greeks.items():
    print(f"{g}: {val:.4f}")

from utils.plot_utils import plot_greek_surface

S_range = np.linspace(S * 0.8, S * 1.2, 30)
sigma_range = np.linspace(0.1, 0.5, 30)
plot_greek_surface(bs_greeks, S_range, sigma_range, K, T, r)
