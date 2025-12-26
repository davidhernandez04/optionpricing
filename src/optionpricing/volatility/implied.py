"""Implied volatility calculation using numerical methods."""

import math
from typing import Optional

from optionpricing.models.base import Option, OptionType, ExerciseStyle
from optionpricing.models.black_scholes import BlackScholesModel
from optionpricing.utils.exceptions import ImpliedVolatilityError
from optionpricing.utils.constants import (
    IV_SOLVER_TOLERANCE,
    IV_SOLVER_MAX_ITERATIONS,
    IV_INITIAL_GUESS,
    IV_MIN_VALUE,
    IV_MAX_VALUE,
)


def implied_volatility(
    spot_price: float,
    strike_price: float,
    time_to_expiry: float,
    risk_free_rate: float,
    market_price: float,
    option_type: OptionType,
    dividend_yield: float = 0.0,
    initial_guess: float = IV_INITIAL_GUESS,
    tolerance: float = IV_SOLVER_TOLERANCE,
    max_iterations: int = IV_SOLVER_MAX_ITERATIONS,
) -> float:
    """Calculate implied volatility from market price using Newton-Raphson method.

    This function inverts the Black-Scholes formula to find the volatility
    that makes the theoretical price equal to the market price.

    Args:
        spot_price: Current price of underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time until expiration in years
        risk_free_rate: Risk-free interest rate
        market_price: Observed market price of the option
        option_type: Call or put option
        dividend_yield: Dividend yield (default: 0)
        initial_guess: Starting volatility guess (default: 0.3)
        tolerance: Convergence tolerance (default: 1e-5)
        max_iterations: Maximum iterations (default: 50)

    Returns:
        Implied volatility as a decimal (e.g., 0.20 for 20%)

    Raises:
        ImpliedVolatilityError: If solver fails to converge or inputs are invalid
    """
    # Validate inputs
    intrinsic_value = _calculate_intrinsic_value(
        spot_price, strike_price, time_to_expiry, risk_free_rate, option_type, dividend_yield
    )

    if market_price < intrinsic_value:
        raise ImpliedVolatilityError(
            f"Market price {market_price} is below intrinsic value {intrinsic_value:.2f}"
        )

    if market_price <= 0:
        raise ImpliedVolatilityError("Market price must be positive")

    # For deep ITM options with very little time value, use bisection
    if market_price - intrinsic_value < 0.01:
        return _bisection_method(
            spot_price, strike_price, time_to_expiry, risk_free_rate,
            market_price, option_type, dividend_yield, tolerance, max_iterations
        )

    # Newton-Raphson method with fallback to bisection
    try:
        return _newton_raphson(
            spot_price, strike_price, time_to_expiry, risk_free_rate,
            market_price, option_type, dividend_yield, initial_guess,
            tolerance, max_iterations
        )
    except ImpliedVolatilityError:
        # Fallback to bisection if Newton-Raphson fails
        return _bisection_method(
            spot_price, strike_price, time_to_expiry, risk_free_rate,
            market_price, option_type, dividend_yield, tolerance, max_iterations
        )


def _calculate_intrinsic_value(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    option_type: OptionType,
    div_yield: float
) -> float:
    """Calculate intrinsic value (present value of max(S-K, 0))."""
    if option_type == OptionType.CALL:
        return max(spot * math.exp(-div_yield * time) - strike * math.exp(-rate * time), 0)
    else:
        return max(strike * math.exp(-rate * time) - spot * math.exp(-div_yield * time), 0)


def _newton_raphson(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    target_price: float,
    option_type: OptionType,
    div_yield: float,
    initial_vol: float,
    tolerance: float,
    max_iter: int,
) -> float:
    """Newton-Raphson method for implied volatility.

    Uses the formula: σ_new = σ_old - (V(σ) - V_market) / Vega(σ)
    """
    vol = initial_vol

    for i in range(max_iter):
        # Create option with current volatility guess
        option = Option(
            spot_price=spot,
            strike_price=strike,
            time_to_expiry=time,
            volatility=vol,
            risk_free_rate=rate,
            option_type=option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            dividend_yield=div_yield,
        )

        model = BlackScholesModel(option)

        # Calculate price and vega
        price = model.price()
        vega = model.vega()

        # Price difference
        price_diff = price - target_price

        # Check convergence
        if abs(price_diff) < tolerance:
            return vol

        # Vega should not be zero for normal cases
        if abs(vega) < 1e-10:
            raise ImpliedVolatilityError("Vega too small for Newton-Raphson")

        # Newton-Raphson update: σ_new = σ_old - f(σ) / f'(σ)
        # Note: model.vega() returns vega per 1%, so multiply by 100
        vol_update = price_diff / (vega * 100)
        vol = vol - vol_update

        # Keep volatility in reasonable bounds
        vol = max(IV_MIN_VALUE, min(IV_MAX_VALUE, vol))

    raise ImpliedVolatilityError(
        f"Newton-Raphson failed to converge after {max_iter} iterations"
    )


def _bisection_method(
    spot: float,
    strike: float,
    time: float,
    rate: float,
    target_price: float,
    option_type: OptionType,
    div_yield: float,
    tolerance: float,
    max_iter: int,
) -> float:
    """Bisection method for implied volatility (fallback method).

    More robust but slower than Newton-Raphson.
    """
    vol_low = IV_MIN_VALUE
    vol_high = IV_MAX_VALUE

    for i in range(max_iter):
        vol_mid = (vol_low + vol_high) / 2

        # Price at midpoint
        option = Option(
            spot_price=spot,
            strike_price=strike,
            time_to_expiry=time,
            volatility=vol_mid,
            risk_free_rate=rate,
            option_type=option_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            dividend_yield=div_yield,
        )

        price_mid = BlackScholesModel(option).price()

        # Check convergence
        if abs(price_mid - target_price) < tolerance:
            return vol_mid

        # Update bounds
        if price_mid < target_price:
            vol_low = vol_mid
        else:
            vol_high = vol_mid

        # Check if bounds are too narrow
        if vol_high - vol_low < tolerance:
            return vol_mid

    raise ImpliedVolatilityError(
        f"Bisection method failed to converge after {max_iter} iterations"
    )


class ImpliedVolatilityCalculator:
    """Helper class for calculating implied volatility from option data."""

    def __init__(
        self,
        tolerance: float = IV_SOLVER_TOLERANCE,
        max_iterations: int = IV_SOLVER_MAX_ITERATIONS,
    ):
        """Initialize IV calculator.

        Args:
            tolerance: Convergence tolerance
            max_iterations: Maximum iterations
        """
        self.tolerance = tolerance
        self.max_iterations = max_iterations

    def calculate(
        self,
        spot_price: float,
        strike_price: float,
        time_to_expiry: float,
        risk_free_rate: float,
        market_price: float,
        option_type: OptionType,
        dividend_yield: float = 0.0,
    ) -> float:
        """Calculate implied volatility.

        See implied_volatility function for parameter descriptions.
        """
        return implied_volatility(
            spot_price=spot_price,
            strike_price=strike_price,
            time_to_expiry=time_to_expiry,
            risk_free_rate=risk_free_rate,
            market_price=market_price,
            option_type=option_type,
            dividend_yield=dividend_yield,
            tolerance=self.tolerance,
            max_iterations=self.max_iterations,
        )
