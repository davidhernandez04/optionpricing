"""Options Pricing Suite - Professional Python package for option pricing and Greeks calculation.

This package provides multiple pricing models including:
- Black-Scholes (analytical)
- Binomial Tree (American options)
- Monte Carlo simulation
- Implied volatility solver
"""

__version__ = "0.1.0"

from optionpricing.models.base import (
    Option,
    OptionType,
    ExerciseStyle,
    Greeks,
    PricingResult,
    OptionPricingModel,
)

__all__ = [
    "__version__",
    "Option",
    "OptionType",
    "ExerciseStyle",
    "Greeks",
    "PricingResult",
    "OptionPricingModel",
]
