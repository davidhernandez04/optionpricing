"""Black-Scholes option pricing model.

This module implements the classic Black-Scholes-Merton formula for European
options, providing both exact pricing and analytical Greeks.

The Black-Scholes formula assumes:
- European exercise (no early exercise)
- Constant volatility and risk-free rate
- Log-normal distribution of underlying returns
- No transaction costs or taxes
- Continuous trading
"""

import math
import numpy as np
from scipy.stats import norm
from typing import Tuple

from optionpricing.models.base import (
    Option,
    OptionType,
    ExerciseStyle,
    Greeks,
    OptionPricingModel,
)
from optionpricing.utils.exceptions import InvalidOptionError
from optionpricing.utils.constants import CALENDAR_DAYS_PER_YEAR


class BlackScholesModel(OptionPricingModel):
    """Black-Scholes pricing model for European options.

    This implementation provides analytical solutions for both option prices
    and Greeks using closed-form formulas.

    Attributes:
        option: The option contract to price
    """

    def __init__(self, option: Option):
        """Initialize Black-Scholes model.

        Args:
            option: Option contract to price

        Raises:
            InvalidOptionError: If option is not European style
        """
        super().__init__(option)
        if option.exercise_style != ExerciseStyle.EUROPEAN:
            raise InvalidOptionError(
                "Black-Scholes model only supports European options. "
                "Use Binomial or Monte Carlo models for American options."
            )

    def _calculate_d1_d2(self) -> Tuple[float, float]:
        """Calculate d1 and d2 parameters for Black-Scholes formula.

        The d1 and d2 values are key intermediate calculations:
        - d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        - d2 = d1 - σ√T

        Returns:
            Tuple of (d1, d2) values
        """
        S = self.option.spot_price
        K = self.option.strike_price
        T = self.option.time_to_expiry
        r = self.option.risk_free_rate
        q = self.option.dividend_yield
        sigma = self.option.volatility

        sqrt_T = math.sqrt(T)
        sigma_sqrt_T = sigma * sqrt_T

        # d1 = [ln(S/K) + (r - q + σ²/2)T] / (σ√T)
        d1 = (math.log(S / K) + (r - q + 0.5 * sigma ** 2) * T) / sigma_sqrt_T

        # d2 = d1 - σ√T
        d2 = d1 - sigma_sqrt_T

        return d1, d2

    def price(self) -> float:
        """Calculate option price using Black-Scholes formula.

        For a call option:
        C = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)

        For a put option:
        P = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)

        Where N(x) is the cumulative standard normal distribution function.

        Returns:
            The theoretical option price
        """
        d1, d2 = self._calculate_d1_d2()

        S = self.option.spot_price
        K = self.option.strike_price
        T = self.option.time_to_expiry
        r = self.option.risk_free_rate
        q = self.option.dividend_yield

        # Discount factors
        pv_S = S * math.exp(-q * T)  # Present value of stock
        pv_K = K * math.exp(-r * T)  # Present value of strike

        if self.option.option_type == OptionType.CALL:
            # Call = S·e^(-qT)·N(d1) - K·e^(-rT)·N(d2)
            price = pv_S * norm.cdf(d1) - pv_K * norm.cdf(d2)
        else:  # PUT
            # Put = K·e^(-rT)·N(-d2) - S·e^(-qT)·N(-d1)
            price = pv_K * norm.cdf(-d2) - pv_S * norm.cdf(-d1)

        return price

    def delta(self) -> float:
        """Calculate Delta: ∂V/∂S (sensitivity to underlying price).

        Delta represents the rate of change of option price with respect to
        the underlying asset price. It approximates the hedge ratio.

        For calls: Δ = e^(-qT)·N(d1)
        For puts: Δ = -e^(-qT)·N(-d1) = e^(-qT)·[N(d1) - 1]

        Returns:
            Delta value (between -1 and 1)
        """
        d1, _ = self._calculate_d1_d2()
        q = self.option.dividend_yield
        T = self.option.time_to_expiry

        discount = math.exp(-q * T)

        if self.option.option_type == OptionType.CALL:
            return discount * norm.cdf(d1)
        else:  # PUT
            return -discount * norm.cdf(-d1)

    def gamma(self) -> float:
        """Calculate Gamma: ∂²V/∂S² (sensitivity of Delta to underlying price).

        Gamma measures the rate of change of Delta. Higher gamma means Delta
        is more sensitive to price changes.

        Γ = e^(-qT)·φ(d1) / (S·σ·√T)

        where φ(x) is the standard normal probability density function.

        Returns:
            Gamma value (always positive)
        """
        d1, _ = self._calculate_d1_d2()

        S = self.option.spot_price
        T = self.option.time_to_expiry
        q = self.option.dividend_yield
        sigma = self.option.volatility

        discount = math.exp(-q * T)
        sqrt_T = math.sqrt(T)

        # φ(d1) = (1/√(2π)) * e^(-d1²/2)
        phi_d1 = norm.pdf(d1)

        gamma = (discount * phi_d1) / (S * sigma * sqrt_T)

        return gamma

    def theta(self) -> float:
        """Calculate Theta: ∂V/∂t (sensitivity to time decay).

        Theta measures the rate of option value decay as time passes.
        By convention, theta is typically expressed per day.

        Returns:
            Theta value (per calendar day, typically negative for long positions)
        """
        d1, d2 = self._calculate_d1_d2()

        S = self.option.spot_price
        K = self.option.strike_price
        T = self.option.time_to_expiry
        r = self.option.risk_free_rate
        q = self.option.dividend_yield
        sigma = self.option.volatility

        sqrt_T = math.sqrt(T)
        phi_d1 = norm.pdf(d1)

        # Common term for both call and put
        term1 = -(S * sigma * math.exp(-q * T) * phi_d1) / (2 * sqrt_T)

        if self.option.option_type == OptionType.CALL:
            term2 = q * S * math.exp(-q * T) * norm.cdf(d1)
            term3 = -r * K * math.exp(-r * T) * norm.cdf(d2)
            theta_per_year = term1 + term2 + term3
        else:  # PUT
            term2 = -q * S * math.exp(-q * T) * norm.cdf(-d1)
            term3 = r * K * math.exp(-r * T) * norm.cdf(-d2)
            theta_per_year = term1 + term2 + term3

        # Convert to per-day theta
        theta_per_day = theta_per_year / CALENDAR_DAYS_PER_YEAR

        return theta_per_day

    def vega(self) -> float:
        """Calculate Vega: ∂V/∂σ (sensitivity to volatility).

        Vega measures the sensitivity of the option price to changes in
        implied volatility. Higher vega means the option is more sensitive
        to volatility changes.

        ν = S·e^(-qT)·√T·φ(d1)

        Returns:
            Vega value (expressed per 1% change in volatility)
        """
        d1, _ = self._calculate_d1_d2()

        S = self.option.spot_price
        T = self.option.time_to_expiry
        q = self.option.dividend_yield

        sqrt_T = math.sqrt(T)
        phi_d1 = norm.pdf(d1)
        discount = math.exp(-q * T)

        # Vega per 1% volatility change
        vega = S * discount * sqrt_T * phi_d1 / 100

        return vega

    def rho(self) -> float:
        """Calculate Rho: ∂V/∂r (sensitivity to risk-free rate).

        Rho measures the sensitivity of the option price to changes in the
        risk-free interest rate.

        For calls: ρ = K·T·e^(-rT)·N(d2)
        For puts: ρ = -K·T·e^(-rT)·N(-d2)

        Returns:
            Rho value (expressed per 1% change in interest rate)
        """
        _, d2 = self._calculate_d1_d2()

        K = self.option.strike_price
        T = self.option.time_to_expiry
        r = self.option.risk_free_rate

        pv_K = K * math.exp(-r * T)

        if self.option.option_type == OptionType.CALL:
            rho = pv_K * T * norm.cdf(d2) / 100
        else:  # PUT
            rho = -pv_K * T * norm.cdf(-d2) / 100

        return rho

    def greeks(self) -> Greeks:
        """Calculate all Greeks at once.

        Returns:
            Greeks object containing Delta, Gamma, Theta, Vega, and Rho
        """
        return Greeks(
            delta=self.delta(),
            gamma=self.gamma(),
            theta=self.theta(),
            vega=self.vega(),
            rho=self.rho(),
        )

    def _get_additional_info(self) -> dict[str, float]:
        """Get additional Black-Scholes specific information.

        Returns:
            Dictionary with d1, d2 values
        """
        d1, d2 = self._calculate_d1_d2()
        return {
            "d1": d1,
            "d2": d2,
        }
