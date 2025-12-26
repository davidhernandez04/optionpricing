"""Numerical Greeks calculator using finite difference methods.

This module provides numerical approximation of Greeks for pricing models
that don't have closed-form formulas (e.g., Binomial, Monte Carlo).
"""

from typing import TYPE_CHECKING

from optionpricing.models.base import Option, Greeks
from optionpricing.utils.constants import CALENDAR_DAYS_PER_YEAR

if TYPE_CHECKING:
    from optionpricing.models.base import OptionPricingModel


class NumericalGreeksCalculator:
    """Calculator for numerical Greeks using finite difference methods.

    This class implements the "bump-and-revalue" methodology, where we
    perturb input parameters and observe changes in option price.

    Attributes:
        model: The pricing model to use for calculations
    """

    def __init__(self, model: "OptionPricingModel"):
        """Initialize numerical Greeks calculator.

        Args:
            model: Pricing model that implements price() method
        """
        self.model = model

    def delta(self, h_pct: float = 0.01) -> float:
        """Calculate Delta using central difference.

        Delta = (V(S+h) - V(S-h)) / (2h)

        Args:
            h_pct: Step size as percentage of spot price (default: 0.01 = 1%)

        Returns:
            Delta (sensitivity to spot price)
        """
        original_spot = self.model.option.spot_price
        h = original_spot * h_pct  # Convert percentage to absolute

        # Create options with perturbed spot prices
        option_up = Option(
            **{**self.model.option.model_dump(), "spot_price": original_spot + h}
        )
        option_down = Option(
            **{**self.model.option.model_dump(), "spot_price": original_spot - h}
        )

        # Calculate prices at perturbed values
        model_up = self.model.__class__(option_up, *self._get_model_args())
        model_down = self.model.__class__(option_down, *self._get_model_args())

        price_up = model_up.price()
        price_down = model_down.price()

        # Central difference approximation
        delta = (price_up - price_down) / (2 * h)

        return delta

    def gamma(self, h_pct: float = 0.01) -> float:
        """Calculate Gamma using central difference of Delta.

        Gamma = (V(S+h) - 2V(S) + V(S-h)) / h²

        Args:
            h_pct: Step size as percentage of spot price (default: 0.01 = 1%)

        Returns:
            Gamma (sensitivity of Delta to spot price)
        """
        original_spot = self.model.option.spot_price
        h = original_spot * h_pct  # Convert percentage to absolute

        # Get base price
        price = self.model.price()

        # Create options with perturbed spot prices
        option_up = Option(
            **{**self.model.option.model_dump(), "spot_price": original_spot + h}
        )
        option_down = Option(
            **{**self.model.option.model_dump(), "spot_price": original_spot - h}
        )

        # Calculate prices
        model_up = self.model.__class__(option_up, *self._get_model_args())
        model_down = self.model.__class__(option_down, *self._get_model_args())

        price_up = model_up.price()
        price_down = model_down.price()

        # Second derivative approximation
        gamma = (price_up - 2 * price + price_down) / (h ** 2)

        return gamma

    def theta(self, h: float = 1/365) -> float:
        """Calculate Theta using forward difference.

        Theta = (V(T-h) - V(T)) / h

        Note: Theta is typically expressed per calendar day.

        Args:
            h: Step size in years (default: 1/365 for 1 day)

        Returns:
            Theta (time decay per calendar day)
        """
        original_T = self.model.option.time_to_expiry

        # Can't go forward in time from current option
        # So we measure: (current price - price if time advanced)
        if original_T <= h:
            # Too close to expiry for finite difference
            # Return very negative theta (rapid decay)
            return -self.model.price() / h

        # Create option with less time to expiry
        option_later = Option(
            **{**self.model.option.model_dump(), "time_to_expiry": original_T - h}
        )

        # Calculate current price and future price
        price_now = self.model.price()
        model_later = self.model.__class__(option_later, *self._get_model_args())
        price_later = model_later.price()

        # Theta per step h
        theta = (price_later - price_now) / h

        # Convert to per-day theta if h != 1/365
        theta_per_day = theta * h * CALENDAR_DAYS_PER_YEAR

        return theta_per_day

    def vega(self, h: float = 0.01) -> float:
        """Calculate Vega using central difference.

        Vega = (V(σ+h) - V(σ-h)) / (2h)

        Note: Our convention is vega per 1% volatility change.

        Args:
            h: Step size for volatility (default: 0.01 = 1%)

        Returns:
            Vega (sensitivity to volatility, per 1% change)
        """
        original_vol = self.model.option.volatility

        # Create options with perturbed volatility
        option_up = Option(
            **{**self.model.option.model_dump(), "volatility": original_vol + h}
        )
        option_down = Option(
            **{**self.model.option.model_dump(), "volatility": original_vol - h}
        )

        # Calculate prices
        model_up = self.model.__class__(option_up, *self._get_model_args())
        model_down = self.model.__class__(option_down, *self._get_model_args())

        price_up = model_up.price()
        price_down = model_down.price()

        # Central difference
        vega = (price_up - price_down) / 2  # Already per 1% since h=0.01

        return vega

    def rho(self, h: float = 0.01) -> float:
        """Calculate Rho using central difference.

        Rho = (V(r+h) - V(r-h)) / (2h)

        Note: Our convention is rho per 1% rate change.

        Args:
            h: Step size for risk-free rate (default: 0.01 = 1%)

        Returns:
            Rho (sensitivity to risk-free rate, per 1% change)
        """
        original_rate = self.model.option.risk_free_rate

        # Create options with perturbed risk-free rate
        option_up = Option(
            **{**self.model.option.model_dump(), "risk_free_rate": original_rate + h}
        )
        option_down = Option(
            **{**self.model.option.model_dump(), "risk_free_rate": original_rate - h}
        )

        # Calculate prices
        model_up = self.model.__class__(option_up, *self._get_model_args())
        model_down = self.model.__class__(option_down, *self._get_model_args())

        price_up = model_up.price()
        price_down = model_down.price()

        # Central difference
        rho = (price_up - price_down) / 2  # Already per 1% since h=0.01

        return rho

    def calculate_all_greeks(self) -> Greeks:
        """Calculate all Greeks at once.

        Returns:
            Greeks object containing all sensitivity measures
        """
        return Greeks(
            delta=self.delta(),
            gamma=self.gamma(),
            theta=self.theta(),
            vega=self.vega(),
            rho=self.rho(),
        )

    def _get_model_args(self) -> tuple:
        """Get additional model-specific arguments.

        For binomial models, this includes num_steps.
        For Monte Carlo, this includes num_simulations, etc.

        Returns:
            Tuple of additional arguments to pass to model constructor
        """
        # Check if model has num_steps attribute (binomial)
        if hasattr(self.model, 'num_steps'):
            return (self.model.num_steps,)

        # Check if model has num_simulations attribute (Monte Carlo)
        if hasattr(self.model, 'num_simulations'):
            args = [self.model.num_simulations]
            if hasattr(self.model, 'use_antithetic'):
                args.append(self.model.use_antithetic)
            if hasattr(self.model, 'random_seed'):
                args.append(self.model.random_seed)
            return tuple(args)

        return ()
