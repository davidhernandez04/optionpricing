"""Tests for Black-Scholes pricing model."""

import pytest
import math
from optionpricing.models.black_scholes import BlackScholesModel
from optionpricing.models.base import Option, OptionType, ExerciseStyle
from optionpricing.utils.exceptions import InvalidOptionError


class TestBlackScholesBasicPricing:
    """Test basic pricing functionality."""

    def test_atm_call_price(self, simple_call_option):
        """Test at-the-money call option pricing."""
        model = BlackScholesModel(simple_call_option)
        price = model.price()

        # For ATM option with S=K=100, T=1, σ=0.2, r=0.05, q=0
        # Expected price ≈ 10.45 (from standard BS calculators)
        assert 10.4 < price < 10.5, f"Expected price ~10.45, got {price}"

    def test_atm_put_price(self, simple_put_option):
        """Test at-the-money put option pricing."""
        model = BlackScholesModel(simple_put_option)
        price = model.price()

        # ATM put should satisfy put-call parity
        # P = C - S + K·e^(-rT)
        call_model = BlackScholesModel(
            Option(
                **{**simple_put_option.model_dump(), "option_type": OptionType.CALL}
            )
        )
        call_price = call_model.price()

        S = simple_put_option.spot_price
        K = simple_put_option.strike_price
        r = simple_put_option.risk_free_rate
        T = simple_put_option.time_to_expiry

        expected_put = call_price - S + K * math.exp(-r * T)

        assert abs(price - expected_put) < 0.01, "Put-call parity violated"

    def test_itm_call_price(self, itm_call_option):
        """Test in-the-money call option pricing."""
        model = BlackScholesModel(itm_call_option)
        price = model.price()

        # ITM call (S=110, K=100) should have intrinsic value > 10
        intrinsic_value = itm_call_option.spot_price - itm_call_option.strike_price
        assert price > intrinsic_value, "ITM option price below intrinsic value"

    def test_otm_put_price(self, otm_put_option):
        """Test out-of-the-money put option pricing."""
        model = BlackScholesModel(otm_put_option)
        price = model.price()

        # OTM put should have small positive value
        assert 0 < price < 5, f"OTM put price {price} seems unreasonable"

    def test_dividend_effect_on_call(self, dividend_paying_option):
        """Test that dividends reduce call option value."""
        model_with_div = BlackScholesModel(dividend_paying_option)
        price_with_div = model_with_div.price()

        # Same option without dividends
        no_div_option = Option(
            **{**dividend_paying_option.model_dump(), "dividend_yield": 0.0}
        )
        model_no_div = BlackScholesModel(no_div_option)
        price_no_div = model_no_div.price()

        assert price_with_div < price_no_div, "Dividends should reduce call value"

    def test_american_option_raises_error(self, american_put_option):
        """Test that American options raise an error."""
        with pytest.raises(InvalidOptionError, match="European"):
            BlackScholesModel(american_put_option)


class TestBlackScholesGreeks:
    """Test Greeks calculations."""

    def test_call_delta_range(self, simple_call_option):
        """Test that call Delta is between 0 and 1."""
        model = BlackScholesModel(simple_call_option)
        delta = model.delta()

        assert 0 < delta < 1, f"Call Delta {delta} out of range [0, 1]"

    def test_put_delta_range(self, simple_put_option):
        """Test that put Delta is between -1 and 0."""
        model = BlackScholesModel(simple_put_option)
        delta = model.delta()

        assert -1 < delta < 0, f"Put Delta {delta} out of range [-1, 0]"

    def test_gamma_positive(self, simple_call_option):
        """Test that Gamma is always positive."""
        model = BlackScholesModel(simple_call_option)
        gamma = model.gamma()

        assert gamma > 0, f"Gamma {gamma} should be positive"

    def test_gamma_same_for_call_and_put(self, simple_call_option):
        """Test that Gamma is the same for calls and puts."""
        call_model = BlackScholesModel(simple_call_option)
        call_gamma = call_model.gamma()

        put_option = Option(
            **{**simple_call_option.model_dump(), "option_type": OptionType.PUT}
        )
        put_model = BlackScholesModel(put_option)
        put_gamma = put_model.gamma()

        assert abs(call_gamma - put_gamma) < 1e-10, "Gamma should be same for calls and puts"

    def test_vega_positive(self, simple_call_option):
        """Test that Vega is positive (options benefit from higher volatility)."""
        model = BlackScholesModel(simple_call_option)
        vega = model.vega()

        assert vega > 0, f"Vega {vega} should be positive"

    def test_vega_same_for_call_and_put(self, simple_call_option):
        """Test that Vega is the same for calls and puts."""
        call_model = BlackScholesModel(simple_call_option)
        call_vega = call_model.vega()

        put_option = Option(
            **{**simple_call_option.model_dump(), "option_type": OptionType.PUT}
        )
        put_model = BlackScholesModel(put_option)
        put_vega = put_model.vega()

        assert abs(call_vega - put_vega) < 1e-10, "Vega should be same for calls and puts"

    def test_theta_negative_for_long_call(self, simple_call_option):
        """Test that Theta is typically negative for long calls (time decay)."""
        model = BlackScholesModel(simple_call_option)
        theta = model.theta()

        # For most calls, theta is negative (time decay)
        assert theta < 0, f"Theta {theta} should typically be negative for long calls"

    def test_call_rho_positive(self, simple_call_option):
        """Test that Rho is positive for calls (benefit from higher rates)."""
        model = BlackScholesModel(simple_call_option)
        rho = model.rho()

        assert rho > 0, f"Call Rho {rho} should be positive"

    def test_put_rho_negative(self, simple_put_option):
        """Test that Rho is negative for puts (hurt by higher rates)."""
        model = BlackScholesModel(simple_put_option)
        rho = model.rho()

        assert rho < 0, f"Put Rho {rho} should be negative"


class TestBlackScholesNumericalValidation:
    """Test Greeks against numerical approximations."""

    def test_delta_numerical_approximation(self, simple_call_option):
        """Verify Delta using finite difference."""
        model = BlackScholesModel(simple_call_option)
        analytical_delta = model.delta()

        # Numerical delta using finite difference
        h = 0.01  # Small price change
        price_up = BlackScholesModel(
            Option(**{**simple_call_option.model_dump(), "spot_price": simple_call_option.spot_price + h})
        ).price()
        price_down = BlackScholesModel(
            Option(**{**simple_call_option.model_dump(), "spot_price": simple_call_option.spot_price - h})
        ).price()

        numerical_delta = (price_up - price_down) / (2 * h)

        assert abs(analytical_delta - numerical_delta) < 0.01, \
            f"Delta mismatch: analytical={analytical_delta}, numerical={numerical_delta}"

    def test_gamma_numerical_approximation(self, simple_call_option):
        """Verify Gamma using finite difference."""
        model = BlackScholesModel(simple_call_option)
        analytical_gamma = model.gamma()

        # Numerical gamma using second derivative
        h = 0.01
        price = model.price()
        price_up = BlackScholesModel(
            Option(**{**simple_call_option.model_dump(), "spot_price": simple_call_option.spot_price + h})
        ).price()
        price_down = BlackScholesModel(
            Option(**{**simple_call_option.model_dump(), "spot_price": simple_call_option.spot_price - h})
        ).price()

        numerical_gamma = (price_up - 2 * price + price_down) / (h ** 2)

        assert abs(analytical_gamma - numerical_gamma) < 0.01, \
            f"Gamma mismatch: analytical={analytical_gamma}, numerical={numerical_gamma}"

    def test_vega_numerical_approximation(self, simple_call_option):
        """Verify Vega using finite difference."""
        model = BlackScholesModel(simple_call_option)
        analytical_vega = model.vega()

        # Numerical vega (note: our vega is per 1% vol change)
        h = 0.01  # 1% volatility change
        price_up = BlackScholesModel(
            Option(**{**simple_call_option.model_dump(), "volatility": simple_call_option.volatility + h})
        ).price()
        price_down = BlackScholesModel(
            Option(**{**simple_call_option.model_dump(), "volatility": simple_call_option.volatility - h})
        ).price()

        numerical_vega = (price_up - price_down) / 2  # Already per 1% change

        assert abs(analytical_vega - numerical_vega) < 0.01, \
            f"Vega mismatch: analytical={analytical_vega}, numerical={numerical_vega}"


class TestBlackScholesPutCallParity:
    """Test put-call parity relationships."""

    def test_put_call_parity_zero_dividends(self, simple_call_option):
        """Test put-call parity: C - P = S - K·e^(-rT)"""
        call_model = BlackScholesModel(simple_call_option)
        call_price = call_model.price()

        put_option = Option(
            **{**simple_call_option.model_dump(), "option_type": OptionType.PUT}
        )
        put_model = BlackScholesModel(put_option)
        put_price = put_model.price()

        S = simple_call_option.spot_price
        K = simple_call_option.strike_price
        r = simple_call_option.risk_free_rate
        T = simple_call_option.time_to_expiry

        lhs = call_price - put_price
        rhs = S - K * math.exp(-r * T)

        assert abs(lhs - rhs) < 0.01, \
            f"Put-call parity violated: C-P={lhs}, S-Ke^(-rT)={rhs}"

    def test_put_call_parity_with_dividends(self, dividend_paying_option):
        """Test put-call parity with dividends: C - P = S·e^(-qT) - K·e^(-rT)"""
        call_model = BlackScholesModel(dividend_paying_option)
        call_price = call_model.price()

        put_option = Option(
            **{**dividend_paying_option.model_dump(), "option_type": OptionType.PUT}
        )
        put_model = BlackScholesModel(put_option)
        put_price = put_model.price()

        S = dividend_paying_option.spot_price
        K = dividend_paying_option.strike_price
        r = dividend_paying_option.risk_free_rate
        q = dividend_paying_option.dividend_yield
        T = dividend_paying_option.time_to_expiry

        lhs = call_price - put_price
        rhs = S * math.exp(-q * T) - K * math.exp(-r * T)

        assert abs(lhs - rhs) < 0.01, \
            f"Put-call parity with dividends violated: C-P={lhs}, Se^(-qT)-Ke^(-rT)={rhs}"


class TestBlackScholesEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_deep_itm_call_approaches_intrinsic_value(self):
        """Test that deep ITM call approaches S - K·e^(-rT) as volatility → 0."""
        deep_itm_call = Option(
            spot_price=150.0,
            strike_price=100.0,
            time_to_expiry=0.25,
            volatility=0.01,  # Very low volatility
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        model = BlackScholesModel(deep_itm_call)
        price = model.price()

        # Theoretical lower bound
        S = deep_itm_call.spot_price
        K = deep_itm_call.strike_price
        r = deep_itm_call.risk_free_rate
        T = deep_itm_call.time_to_expiry

        intrinsic_pv = S - K * math.exp(-r * T)

        assert price >= intrinsic_pv, "Price below intrinsic PV"
        assert price < intrinsic_pv + 1.0, "Price too far above intrinsic PV for low vol"

    def test_very_short_expiry(self):
        """Test option near expiration."""
        short_expiry = Option(
            spot_price=105.0,
            strike_price=100.0,
            time_to_expiry=1/365,  # 1 day
            volatility=0.20,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
        )

        model = BlackScholesModel(short_expiry)
        price = model.price()

        # Should be close to intrinsic value
        intrinsic = max(short_expiry.spot_price - short_expiry.strike_price, 0)
        assert abs(price - intrinsic) < 1.0, "Short expiry option should be near intrinsic value"


class TestBlackScholesIntegration:
    """Integration tests for complete workflow."""

    def test_price_with_greeks(self, simple_call_option):
        """Test the convenience method for getting price and greeks together."""
        model = BlackScholesModel(simple_call_option)
        result = model.price_with_greeks()

        assert result.price == model.price()
        assert result.greeks.delta == model.delta()
        assert result.greeks.gamma == model.gamma()
        assert result.method == "BlackScholesModel"
        assert "d1" in result.additional_info
        assert "d2" in result.additional_info

    def test_greeks_object_string_representation(self, simple_call_option):
        """Test that Greeks object has nice string representation."""
        model = BlackScholesModel(simple_call_option)
        greeks = model.greeks()

        greeks_str = str(greeks)
        assert "Delta" in greeks_str
        assert "Gamma" in greeks_str
        assert "Theta" in greeks_str
        assert "Vega" in greeks_str
        assert "Rho" in greeks_str
