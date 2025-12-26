"""Market data fetching using yfinance."""

import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional

from optionpricing.utils.exceptions import DataFetchError


class MarketDataFetcher:
    """Fetches market data for options pricing.

    Uses yfinance to retrieve stock prices, option chains, and risk-free rates.
    """

    def __init__(self, cache_hours: int = 1):
        """Initialize market data fetcher.

        Args:
            cache_hours: Hours to cache data (default: 1)
        """
        self.cache_hours = cache_hours
        self._cache = {}

    def get_spot_price(self, symbol: str) -> float:
        """Get current spot price for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Current spot price

        Raises:
            DataFetchError: If data cannot be fetched
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="1d")

            if hist.empty:
                raise DataFetchError(f"No price data available for {symbol}")

            return float(hist['Close'].iloc[-1])

        except Exception as e:
            raise DataFetchError(f"Failed to fetch spot price for {symbol}: {str(e)}")

    def get_option_chain(self, symbol: str, expiration_date: Optional[str] = None) -> pd.DataFrame:
        """Get option chain for a symbol.

        Args:
            symbol: Stock ticker symbol
            expiration_date: Expiration date (YYYY-MM-DD) or None for nearest

        Returns:
            DataFrame with option chain data

        Raises:
            DataFetchError: If data cannot be fetched
        """
        try:
            ticker = yf.Ticker(symbol)

            # Get available expiration dates
            expirations = ticker.options

            if not expirations:
                raise DataFetchError(f"No options available for {symbol}")

            # Use specified date or nearest
            if expiration_date is None:
                expiration_date = expirations[0]
            elif expiration_date not in expirations:
                raise DataFetchError(
                    f"Expiration {expiration_date} not available. "
                    f"Available: {expirations[:5]}"
                )

            # Get option chain
            chain = ticker.option_chain(expiration_date)

            # Combine calls and puts
            calls = chain.calls.copy()
            calls['option_type'] = 'call'

            puts = chain.puts.copy()
            puts['option_type'] = 'put'

            df = pd.concat([calls, puts], ignore_index=True)
            df['expiration'] = expiration_date

            return df

        except Exception as e:
            raise DataFetchError(f"Failed to fetch option chain for {symbol}: {str(e)}")

    def get_risk_free_rate(self) -> float:
        """Get current risk-free rate (10-year Treasury yield).

        Returns:
            Risk-free rate as decimal

        Raises:
            DataFetchError: If data cannot be fetched
        """
        try:
            # Use 10-year Treasury (^TNX gives percentage, divide by 100)
            tnx = yf.Ticker("^TNX")
            hist = tnx.history(period="5d")

            if hist.empty:
                # Fallback to default if unavailable
                return 0.04  # 4% default

            rate_pct = float(hist['Close'].iloc[-1])
            return rate_pct / 100.0

        except Exception:
            # Return default if fetch fails
            return 0.04

    def get_dividend_yield(self, symbol: str) -> float:
        """Get dividend yield for a symbol.

        Args:
            symbol: Stock ticker symbol

        Returns:
            Annual dividend yield as decimal

        Raises:
            DataFetchError: If data cannot be fetched
        """
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info

            # Try to get dividend yield
            div_yield = info.get('dividendYield', 0.0)

            if div_yield is None:
                div_yield = 0.0

            return float(div_yield)

        except Exception:
            # Return 0 if cannot determine
            return 0.0

    def calculate_time_to_expiry(self, expiration_date: str) -> float:
        """Calculate time to expiry in years.

        Args:
            expiration_date: Expiration date string (YYYY-MM-DD)

        Returns:
            Time to expiry in years
        """
        exp_date = datetime.strptime(expiration_date, "%Y-%m-%d")
        today = datetime.now()

        days_to_expiry = (exp_date - today).days
        years_to_expiry = days_to_expiry / 365.0

        return max(years_to_expiry, 1/365)  # Minimum 1 day
