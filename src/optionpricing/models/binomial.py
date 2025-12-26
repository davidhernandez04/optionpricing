"""Binomial Tree option pricing model.

This module implements the Cox-Ross-Rubinstein binomial tree model for pricing
both European and American options using a discrete-time lattice approach.

The binomial model is particularly useful for:
- American options (early exercise)
- Options on dividend-paying stocks
- Understanding convergence to continuous-time models
"""

import numpy as np
from typing import Optional

from optionpricing.models.base import (
    Option,
    OptionType,
    ExerciseStyle,
    Greeks,
    OptionPricingModel,
)
from optionpricing.greeks.calculator import NumericalGreeksCalculator
from optionpricing.utils.constants import DEFAULT_NUM_STEPS


class BinomialTreeModel(OptionPricingModel):
    """Binomial tree pricing model (Cox-Ross-Rubinstein).

    This implementation uses the CRR parameterization for tree construction
    and supports both European and American exercise styles.

    Attributes:
        option: The option contract to price
        num_steps: Number of time steps in the binomial tree (default: 100)
    """

    def __init__(self, option: Option, num_steps: int = DEFAULT_NUM_STEPS):
        """Initialize binomial tree model.

        Args:
            option: Option contract to price
            num_steps: Number of time steps in tree (more steps = more accuracy)
        """
        super().__init__(option)
        self.num_steps = num_steps
        self._greeks_calculator = NumericalGreeksCalculator(self)

    def _build_stock_tree(self) -> np.ndarray:
        """Build the stock price tree using Cox-Ross-Rubinstein parameters.

        The CRR model uses:
        - u = e^(σ√Δt) (up factor)
        - d = e^(-σ√Δt) = 1/u (down factor)
        - p = (e^((r-q)Δt) - d) / (u - d) (risk-neutral probability)

        Returns:
            2D array representing stock prices at each node
        """
        S = self.option.spot_price
        sigma = self.option.volatility
        T = self.option.time_to_expiry
        r = self.option.risk_free_rate
        q = self.option.dividend_yield

        dt = T / self.num_steps

        # CRR parameters
        u = np.exp(sigma * np.sqrt(dt))  # Up factor
        d = 1 / u  # Down factor

        # Build stock price tree
        # tree[i, j] = stock price at time step i, after j up moves
        tree = np.zeros((self.num_steps + 1, self.num_steps + 1))

        for i in range(self.num_steps + 1):
            for j in range(i + 1):
                # j up moves, (i-j) down moves
                tree[i, j] = S * (u ** j) * (d ** (i - j))

        return tree

    def _calculate_risk_neutral_probability(self) -> float:
        """Calculate risk-neutral probability for CRR model.

        Returns:
            Risk-neutral probability of up move
        """
        r = self.option.risk_free_rate
        q = self.option.dividend_yield
        T = self.option.time_to_expiry
        sigma = self.option.volatility

        dt = T / self.num_steps
        u = np.exp(sigma * np.sqrt(dt))
        d = 1 / u

        # Risk-neutral probability
        p = (np.exp((r - q) * dt) - d) / (u - d)

        return p

    def _get_payoff(self, stock_price: float) -> float:
        """Calculate option payoff at a given stock price.

        Args:
            stock_price: The underlying stock price

        Returns:
            Option payoff (intrinsic value)
        """
        K = self.option.strike_price

        if self.option.option_type == OptionType.CALL:
            return max(stock_price - K, 0)
        else:  # PUT
            return max(K - stock_price, 0)

    def price(self) -> float:
        """Calculate option price using binomial tree.

        For European options, we use backward induction without early exercise.
        For American options, we check for optimal early exercise at each node.

        Returns:
            The option price
        """
        stock_tree = self._build_stock_tree()
        p = self._calculate_risk_neutral_probability()

        r = self.option.risk_free_rate
        T = self.option.time_to_expiry
        dt = T / self.num_steps
        discount = np.exp(-r * dt)

        # Initialize option values at maturity
        option_tree = np.zeros_like(stock_tree)
        for j in range(self.num_steps + 1):
            option_tree[self.num_steps, j] = self._get_payoff(stock_tree[self.num_steps, j])

        # Backward induction
        for i in range(self.num_steps - 1, -1, -1):
            for j in range(i + 1):
                # Expected value (discounted risk-neutral expectation)
                hold_value = discount * (p * option_tree[i + 1, j + 1] + (1 - p) * option_tree[i + 1, j])

                if self.option.exercise_style == ExerciseStyle.AMERICAN:
                    # For American options, check early exercise
                    exercise_value = self._get_payoff(stock_tree[i, j])
                    option_tree[i, j] = max(hold_value, exercise_value)
                else:
                    # European option - no early exercise
                    option_tree[i, j] = hold_value

        return option_tree[0, 0]

    def greeks(self) -> Greeks:
        """Calculate Greeks using numerical approximation.

        The binomial model uses finite difference methods for Greeks
        since analytical formulas are not available.

        Returns:
            Greeks object with all sensitivity measures
        """
        return self._greeks_calculator.calculate_all_greeks()

    def delta(self) -> float:
        """Calculate Delta using numerical approximation."""
        return self._greeks_calculator.delta()

    def gamma(self) -> float:
        """Calculate Gamma using numerical approximation."""
        return self._greeks_calculator.gamma()

    def theta(self) -> float:
        """Calculate Theta using numerical approximation."""
        return self._greeks_calculator.theta()

    def vega(self) -> float:
        """Calculate Vega using numerical approximation."""
        return self._greeks_calculator.vega()

    def rho(self) -> float:
        """Calculate Rho using numerical approximation."""
        return self._greeks_calculator.rho()

    def _get_additional_info(self) -> dict[str, float]:
        """Get additional binomial tree specific information.

        Returns:
            Dictionary with num_steps and convergence info
        """
        return {
            "num_steps": float(self.num_steps),
            "dt": self.option.time_to_expiry / self.num_steps,
        }
