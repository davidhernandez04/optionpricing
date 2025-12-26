"""Pytest configuration and fixtures for option pricing tests."""

import pytest
from optionpricing.models.base import Option, OptionType, ExerciseStyle


@pytest.fixture
def simple_call_option():
    """Fixture for a simple European call option.

    Parameters based on typical textbook example:
    - S = 100, K = 100 (at-the-money)
    - T = 1 year
    - σ = 20%
    - r = 5%
    - q = 0% (no dividends)
    """
    return Option(
        spot_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        volatility=0.20,
        risk_free_rate=0.05,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        dividend_yield=0.0
    )


@pytest.fixture
def simple_put_option():
    """Fixture for a simple European put option.

    Same parameters as call option for put-call parity testing.
    """
    return Option(
        spot_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        volatility=0.20,
        risk_free_rate=0.05,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN,
        dividend_yield=0.0
    )


@pytest.fixture
def itm_call_option():
    """In-the-money call option (S > K)."""
    return Option(
        spot_price=110.0,
        strike_price=100.0,
        time_to_expiry=0.5,
        volatility=0.25,
        risk_free_rate=0.04,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        dividend_yield=0.0
    )


@pytest.fixture
def otm_put_option():
    """Out-of-the-money put option (S > K)."""
    return Option(
        spot_price=110.0,
        strike_price=100.0,
        time_to_expiry=0.5,
        volatility=0.25,
        risk_free_rate=0.04,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.EUROPEAN,
        dividend_yield=0.0
    )


@pytest.fixture
def american_put_option():
    """American put option for early exercise testing."""
    return Option(
        spot_price=90.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        volatility=0.30,
        risk_free_rate=0.05,
        option_type=OptionType.PUT,
        exercise_style=ExerciseStyle.AMERICAN,
        dividend_yield=0.0
    )


@pytest.fixture
def dividend_paying_option():
    """Option on dividend-paying stock."""
    return Option(
        spot_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        volatility=0.20,
        risk_free_rate=0.05,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
        dividend_yield=0.03
    )
