"""Historical volatility calculation from price data."""

import numpy as np
import pandas as pd
from typing import Union

from optionpricing.utils.constants import TRADING_DAYS_PER_YEAR


def historical_volatility(
    prices: Union[np.ndarray, pd.Series],
    window: int = 30,
    annualize: bool = True,
) -> float:
    """Calculate historical volatility from price data.

    Uses log returns: r_t = ln(S_t / S_{t-1})

    Args:
        prices: Array or Series of historical prices
        window: Number of periods for calculation (default: 30)
        annualize: Whether to annualize the volatility (default: True)

    Returns:
        Historical volatility (annualized if annualize=True)
    """
    if isinstance(prices, pd.Series):
        prices = prices.values

    if len(prices) < window + 1:
        raise ValueError(f"Need at least {window + 1} prices for window of {window}")

    # Use last 'window' returns
    recent_prices = prices[-(window + 1):]

    # Calculate log returns
    log_returns = np.diff(np.log(recent_prices))

    # Sample standard deviation
    volatility = np.std(log_returns, ddof=1)

    # Annualize if requested (assumes daily data)
    if annualize:
        volatility *= np.sqrt(TRADING_DAYS_PER_YEAR)

    return volatility
