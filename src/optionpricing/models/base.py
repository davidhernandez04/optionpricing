"""Core abstractions for option pricing models.

This module defines the fundamental data structures and interfaces used throughout
the options pricing suite.
"""

from abc import ABC, abstractmethod
from enum import Enum
from pydantic import BaseModel, ConfigDict, Field, field_validator


class OptionType(str, Enum):
    """Type of option contract."""

    CALL = "call"
    PUT = "put"


class ExerciseStyle(str, Enum):
    """Exercise style of the option."""

    EUROPEAN = "european"
    AMERICAN = "american"


class Option(BaseModel):
    """Represents an option contract with all necessary parameters.

    Attributes:
        spot_price: Current price of the underlying asset
        strike_price: Strike price of the option
        time_to_expiry: Time until expiration in years
        volatility: Annualized volatility (standard deviation of returns)
        risk_free_rate: Risk-free interest rate (annualized)
        option_type: Call or put option
        exercise_style: European or American exercise
        dividend_yield: Continuous dividend yield (annualized)
    """

    spot_price: float = Field(gt=0, description="Current price of underlying asset")
    strike_price: float = Field(gt=0, description="Strike price of the option")
    time_to_expiry: float = Field(gt=0, description="Time to expiration in years")
    volatility: float = Field(gt=0, description="Annualized volatility")
    risk_free_rate: float = Field(description="Risk-free interest rate")
    option_type: OptionType
    exercise_style: ExerciseStyle = ExerciseStyle.EUROPEAN
    dividend_yield: float = Field(default=0.0, ge=0, description="Dividend yield")

    model_config = ConfigDict(use_enum_values=False)

    @field_validator('time_to_expiry')
    @classmethod
    def validate_time_to_expiry(cls, v: float) -> float:
        """Ensure time to expiry is positive."""
        if v <= 0:
            raise ValueError("Time to expiry must be positive")
        return v


class Greeks(BaseModel):
    """Container for option Greeks (sensitivity measures).

    Attributes:
        delta: Rate of change of option price with respect to underlying price
        gamma: Rate of change of delta with respect to underlying price
        theta: Rate of change of option price with respect to time (per day)
        vega: Rate of change of option price with respect to volatility
        rho: Rate of change of option price with respect to risk-free rate
    """

    delta: float = Field(description="Sensitivity to underlying price")
    gamma: float = Field(description="Sensitivity of delta to underlying price")
    theta: float = Field(description="Time decay (per calendar day)")
    vega: float = Field(description="Sensitivity to volatility")
    rho: float = Field(description="Sensitivity to risk-free rate")

    def __str__(self) -> str:
        """Pretty string representation of Greeks."""
        return (
            f"Greeks:\n"
            f"  Delta: {self.delta:>8.4f}\n"
            f"  Gamma: {self.gamma:>8.4f}\n"
            f"  Theta: {self.theta:>8.4f}\n"
            f"  Vega:  {self.vega:>8.4f}\n"
            f"  Rho:   {self.rho:>8.4f}"
        )


class PricingResult(BaseModel):
    """Result from pricing an option.

    Attributes:
        price: The calculated option price
        greeks: The option Greeks
        method: Name of the pricing method used
        additional_info: Additional method-specific information
    """

    price: float = Field(description="Option price")
    greeks: Greeks = Field(description="Option Greeks")
    method: str = Field(description="Pricing method name")
    additional_info: dict[str, float] = Field(
        default_factory=dict,
        description="Additional method-specific information"
    )


class OptionPricingModel(ABC):
    """Abstract base class for all option pricing models.

    This class defines the interface that all pricing models must implement,
    enabling polymorphism and consistent usage across different pricing methods.
    """

    def __init__(self, option: Option):
        """Initialize the pricing model with an option contract.

        Args:
            option: The option contract to price
        """
        self.option = option

    @abstractmethod
    def price(self) -> float:
        """Calculate the option price.

        Returns:
            The calculated option price
        """
        pass

    @abstractmethod
    def greeks(self) -> Greeks:
        """Calculate the option Greeks.

        Returns:
            Greeks object containing all sensitivity measures
        """
        pass

    def price_with_greeks(self) -> PricingResult:
        """Calculate both price and Greeks.

        Returns:
            PricingResult containing price, Greeks, and method info
        """
        price = self.price()
        greeks = self.greeks()

        return PricingResult(
            price=price,
            greeks=greeks,
            method=self.__class__.__name__,
            additional_info=self._get_additional_info()
        )

    def _get_additional_info(self) -> dict[str, float]:
        """Get additional model-specific information.

        Subclasses can override this to provide extra information.

        Returns:
            Dictionary of additional information
        """
        return {}
