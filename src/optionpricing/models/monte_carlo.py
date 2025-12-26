"""Monte Carlo simulation for option pricing.

This module implements Monte Carlo simulation for European options with
variance reduction techniques including antithetic variates.
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
from optionpricing.utils.exceptions import InvalidOptionError
from optionpricing.utils.constants import (
    DEFAULT_NUM_SIMULATIONS,
    DEFAULT_RANDOM_SEED,
)


class MonteCarloModel(OptionPricingModel):
    """Monte Carlo simulation pricing model for European options.

    Uses Geometric Brownian Motion to simulate price paths and calculates
    option value as discounted expected payoff.

    Attributes:
        option: The option contract to price
        num_simulations: Number of Monte Carlo paths
        use_antithetic: Whether to use antithetic variates for variance reduction
        random_seed: Random seed for reproducibility
    """

    def __init__(
        self,
        option: Option,
        num_simulations: int = DEFAULT_NUM_SIMULATIONS,
        use_antithetic: bool = True,
        random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    ):
        """Initialize Monte Carlo model.

        Args:
            option: Option contract to price
            num_simulations: Number of simulation paths
            use_antithetic: Use antithetic variates for variance reduction
            random_seed: Random seed (None for no seeding)
        """
        super().__init__(option)

        if option.exercise_style != ExerciseStyle.EUROPEAN:
            raise InvalidOptionError(
                "Monte Carlo model (as implemented) only supports European options. "
                "Use Binomial model for American options."
            )

        self.num_simulations = num_simulations
        self.use_antithetic = use_antithetic
        self.random_seed = random_seed
        self._greeks_calculator = NumericalGreeksCalculator(self)

        # Cache for performance
        self._last_paths: Optional[np.ndarray] = None
        self._standard_error: Optional[float] = None

    def _generate_price_paths(self) -> np.ndarray:
        """Generate stock price paths using Geometric Brownian Motion.

        Uses the formula:
        S(T) = S(0) * exp((r - q - σ²/2)T + σ√T*Z)

        where Z ~ N(0,1)

        Returns:
            Array of terminal stock prices
        """
        if self.random_seed is not None:
            np.random.seed(self.random_seed)

        S = self.option.spot_price
        r = self.option.risk_free_rate
        q = self.option.dividend_yield
        sigma = self.option.volatility
        T = self.option.time_to_expiry

        # Drift and diffusion
        drift = (r - q - 0.5 * sigma ** 2) * T
        diffusion = sigma * np.sqrt(T)

        # Number of paths to generate
        if self.use_antithetic:
            # Generate half, create antithetic pairs
            n_paths = self.num_simulations // 2
            Z = np.random.standard_normal(n_paths)
            # Antithetic variates: use both Z and -Z
            Z_anti = np.concatenate([Z, -Z])
        else:
            Z_anti = np.random.standard_normal(self.num_simulations)

        # Terminal stock prices
        ST = S * np.exp(drift + diffusion * Z_anti)

        self._last_paths = ST
        return ST

    def _calculate_payoffs(self, stock_prices: np.ndarray) -> np.ndarray:
        """Calculate option payoffs at maturity.

        Args:
            stock_prices: Array of terminal stock prices

        Returns:
            Array of option payoffs
        """
        K = self.option.strike_price

        if self.option.option_type == OptionType.CALL:
            payoffs = np.maximum(stock_prices - K, 0)
        else:  # PUT
            payoffs = np.maximum(K - stock_prices, 0)

        return payoffs

    def price(self) -> float:
        """Calculate option price using Monte Carlo simulation.

        Returns:
            The estimated option price
        """
        # Generate price paths
        ST = self._generate_price_paths()

        # Calculate payoffs
        payoffs = self._calculate_payoffs(ST)

        # Discount to present value
        r = self.option.risk_free_rate
        T = self.option.time_to_expiry
        discount_factor = np.exp(-r * T)

        # Monte Carlo estimate: average discounted payoff
        price = discount_factor * np.mean(payoffs)

        # Calculate standard error for confidence intervals
        std_payoffs = np.std(payoffs, ddof=1)
        self._standard_error = discount_factor * std_payoffs / np.sqrt(self.num_simulations)

        return price

    def get_confidence_interval(self, confidence_level: float = 0.95) -> tuple[float, float]:
        """Get confidence interval for the price estimate.

        Args:
            confidence_level: Confidence level (default: 0.95 for 95%)

        Returns:
            Tuple of (lower_bound, upper_bound)
        """
        if self._standard_error is None:
            # Need to run price() first
            price = self.price()
        else:
            price = self.price()

        # Z-score for confidence level (approximate for 95%: 1.96)
        from scipy.stats import norm
        z = norm.ppf((1 + confidence_level) / 2)

        margin = z * self._standard_error
        return (price - margin, price + margin)

    def greeks(self) -> Greeks:
        """Calculate Greeks using numerical approximation.

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
        """Get additional Monte Carlo specific information.

        Returns:
            Dictionary with simulation info
        """
        info = {
            "num_simulations": float(self.num_simulations),
            "use_antithetic": float(self.use_antithetic),
        }

        if self._standard_error is not None:
            info["standard_error"] = self._standard_error
            ci_lower, ci_upper = self.get_confidence_interval()
            info["ci_95_lower"] = ci_lower
            info["ci_95_upper"] = ci_upper

        return info
