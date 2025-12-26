"""Custom exceptions for the options pricing package."""


class OptionPricingError(Exception):
    """Base exception for all option pricing errors."""
    pass


class ValidationError(OptionPricingError):
    """Raised when input validation fails."""
    pass


class ConvergenceError(OptionPricingError):
    """Raised when an iterative method fails to converge."""
    pass


class DataFetchError(OptionPricingError):
    """Raised when fetching market data fails."""
    pass


class InvalidOptionError(ValidationError):
    """Raised when option parameters are invalid."""
    pass


class ImpliedVolatilityError(ConvergenceError):
    """Raised when implied volatility solver fails to converge."""
    pass


class PricingMethodError(OptionPricingError):
    """Raised when a pricing method encounters an error."""
    pass
