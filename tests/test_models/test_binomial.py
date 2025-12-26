"""Tests for Binomial Tree pricing model."""

import pytest
from optionpricing.models.binomial import BinomialTreeModel
from optionpricing.models.black_scholes import BlackScholesModel
from optionpricing.models.base import Option, OptionType, ExerciseStyle


class TestBinomialBasicPricing:
    """Test basic binomial pricing functionality."""

    def test_european_call_convergence(self, simple_call_option):
        """Test that European call converges to Black-Scholes."""
        bs_model = BlackScholesModel(simple_call_option)
        bs_price = bs_model.price()

        # Test with increasing steps
        bin_model_50 = BinomialTreeModel(simple_call_option, num_steps=50)
        bin_model_100 = BinomialTreeModel(simple_call_option, num_steps=100)
        bin_model_200 = BinomialTreeModel(simple_call_option, num_steps=200)

        bin_price_50 = bin_model_50.price()
        bin_price_100 = bin_model_100.price()
        bin_price_200 = bin_model_200.price()

        # Check convergence - more steps should be closer to BS
        error_50 = abs(bin_price_50 - bs_price)
        error_100 = abs(bin_price_100 - bs_price)
        error_200 = abs(bin_price_200 - bs_price)

        assert error_200 < error_100, "200 steps should be more accurate than 100"
        assert error_100 < error_50 * 1.5, "100 steps should be reasonably accurate"
        assert error_200 < 0.5, f"200 steps should be close to BS: error={error_200}"

    def test_european_put_convergence(self, simple_put_option):
        """Test that European put converges to Black-Scholes."""
        bs_model = BlackScholesModel(simple_put_option)
        bs_price = bs_model.price()

        bin_model = BinomialTreeModel(simple_put_option, num_steps=150)
        bin_price = bin_model.price()

        assert abs(bin_price - bs_price) < 0.5, \
            f"Binomial put {bin_price} should be close to BS {bs_price}"

    def test_american_put_early_exercise_premium(self, american_put_option):
        """Test that American put is worth more than European put (early exercise premium)."""
        # European version of same option
        european_option = Option(
            **{**american_put_option.model_dump(), "exercise_style": ExerciseStyle.EUROPEAN}
        )

        american_model = BinomialTreeModel(american_put_option, num_steps=100)
        european_model = BinomialTreeModel(european_option, num_steps=100)

        american_price = american_model.price()
        european_price = european_model.price()

        assert american_price >= european_price, \
            "American option should be worth at least as much as European"

        # For deep ITM put, there should be meaningful early exercise premium
        if american_put_option.spot_price < american_put_option.strike_price * 0.9:
            assert american_price > european_price, \
                "Deep ITM American put should have early exercise premium"

    def test_american_call_no_dividend_equals_european(self):
        """Test that American call without dividends equals European (no early exercise)."""
        american_call = Option(
            spot_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            volatility=0.20,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.AMERICAN,
            dividend_yield=0.0  # No dividends
        )

        european_call = Option(
            **{**american_call.model_dump(), "exercise_style": ExerciseStyle.EUROPEAN}
        )

        american_model = BinomialTreeModel(american_call, num_steps=100)
        european_model = BinomialTreeModel(european_call, num_steps=100)

        american_price = american_model.price()
        european_price = european_model.price()

        # Should be essentially equal (within numerical error)
        assert abs(american_price - european_price) < 0.01, \
            "American call without dividends should equal European call"

    def test_increasing_steps_improves_accuracy(self, simple_call_option):
        """Test that increasing tree steps improves convergence."""
        bs_model = BlackScholesModel(simple_call_option)
        bs_price = bs_model.price()

        prices = []
        for steps in [20, 50, 100, 200]:
            model = BinomialTreeModel(simple_call_option, num_steps=steps)
            prices.append(model.price())

        errors = [abs(p - bs_price) for p in prices]

        # Generally, errors should decrease (though may oscillate slightly)
        assert errors[-1] < errors[0], "More steps should reduce error overall"


class TestBinomialGreeks:
    """Test Greeks calculations for binomial model."""

    def test_delta_positive_for_call(self, simple_call_option):
        """Test that Delta is positive for call."""
        model = BinomialTreeModel(simple_call_option, num_steps=100)
        delta = model.delta()

        assert 0 < delta < 1, f"Call delta {delta} should be in (0, 1)"

    def test_delta_negative_for_put(self, simple_put_option):
        """Test that Delta is negative for put."""
        model = BinomialTreeModel(simple_put_option, num_steps=100)
        delta = model.delta()

        assert -1 < delta < 0, f"Put delta {delta} should be in (-1, 0)"

    def test_gamma_positive(self, simple_call_option):
        """Test that Gamma is positive."""
        model = BinomialTreeModel(simple_call_option, num_steps=100)
        gamma = model.gamma()

        assert gamma > 0, f"Gamma {gamma} should be positive"

    def test_vega_positive(self, simple_call_option):
        """Test that Vega is positive."""
        model = BinomialTreeModel(simple_call_option, num_steps=100)
        vega = model.vega()

        assert vega > 0, f"Vega {vega} should be positive"

    def test_greeks_approximate_black_scholes(self, simple_call_option):
        """Test that binomial Greeks are reasonably close to Black-Scholes."""
        bs_model = BlackScholesModel(simple_call_option)
        bs_greeks = bs_model.greeks()

        bin_model = BinomialTreeModel(simple_call_option, num_steps=150)
        bin_greeks = bin_model.greeks()

        # Delta should be within 10%
        assert abs(bin_greeks.delta - bs_greeks.delta) < 0.1, \
            f"Delta: BS={bs_greeks.delta}, Bin={bin_greeks.delta}"

        # Gamma: just verify it's positive and reasonable magnitude
        # (Numerical second derivatives are very sensitive to step size)
        assert bin_greeks.gamma > 0, "Gamma should be positive"
        assert bin_greeks.gamma < 1.0, f"Gamma should be reasonable: {bin_greeks.gamma}"

        # Vega should be within 20%
        assert abs(bin_greeks.vega - bs_greeks.vega) / bs_greeks.vega < 0.2, \
            f"Vega: BS={bs_greeks.vega}, Bin={bin_greeks.vega}"


class TestBinomialEdgeCases:
    """Test edge cases for binomial model."""

    def test_deep_itm_call(self):
        """Test deep in-the-money call."""
        deep_itm = Option(
            spot_price=150.0,
            strike_price=100.0,
            time_to_expiry=0.5,
            volatility=0.20,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )

        model = BinomialTreeModel(deep_itm, num_steps=100)
        price = model.price()

        # Should be worth at least intrinsic value
        intrinsic = deep_itm.spot_price - deep_itm.strike_price
        assert price >= intrinsic, "Price should be at least intrinsic value"

    def test_deep_otm_put(self):
        """Test deep out-of-the-money put."""
        deep_otm = Option(
            spot_price=150.0,
            strike_price=100.0,
            time_to_expiry=0.5,
            volatility=0.20,
            risk_free_rate=0.05,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN,
        )

        model = BinomialTreeModel(deep_otm, num_steps=100)
        price = model.price()

        # Should be close to zero but positive
        assert 0 < price < 1.0, f"Deep OTM put should have small value: {price}"

    def test_very_short_expiry(self):
        """Test option with very short time to expiry."""
        short_expiry = Option(
            spot_price=105.0,
            strike_price=100.0,
            time_to_expiry=1/252,  # 1 trading day
            volatility=0.30,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )

        model = BinomialTreeModel(short_expiry, num_steps=50)
        price = model.price()

        # Should be close to intrinsic value
        intrinsic = max(short_expiry.spot_price - short_expiry.strike_price, 0)
        assert abs(price - intrinsic) < 2.0, \
            "Short expiry option should be near intrinsic value"

    def test_high_volatility(self):
        """Test option with high volatility."""
        high_vol = Option(
            spot_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            volatility=0.80,  # 80% vol
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
        )

        model = BinomialTreeModel(high_vol, num_steps=100)
        price = model.price()

        # High vol option should have significant value
        assert price > 20, f"High vol ATM call should be valuable: {price}"

        # Compare with Black-Scholes
        bs_model = BlackScholesModel(high_vol)
        bs_price = bs_model.price()

        assert abs(price - bs_price) < 2.0, "Should still converge for high vol"


class TestBinomialWithDividends:
    """Test binomial model with dividend-paying stocks."""

    def test_dividend_reduces_call_value(self):
        """Test that dividends reduce call option value."""
        no_div_option = Option(
            spot_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            volatility=0.20,
            risk_free_rate=0.05,
            option_type=OptionType.CALL,
            exercise_style=ExerciseStyle.EUROPEAN,
            dividend_yield=0.0,
        )

        with_div_option = Option(
            **{**no_div_option.model_dump(), "dividend_yield": 0.03}
        )

        model_no_div = BinomialTreeModel(no_div_option, num_steps=100)
        model_with_div = BinomialTreeModel(with_div_option, num_steps=100)

        price_no_div = model_no_div.price()
        price_with_div = model_with_div.price()

        assert price_with_div < price_no_div, \
            "Dividends should reduce call value"

    def test_dividend_increases_put_value(self):
        """Test that dividends increase put option value."""
        no_div_option = Option(
            spot_price=100.0,
            strike_price=100.0,
            time_to_expiry=1.0,
            volatility=0.20,
            risk_free_rate=0.05,
            option_type=OptionType.PUT,
            exercise_style=ExerciseStyle.EUROPEAN,
            dividend_yield=0.0,
        )

        with_div_option = Option(
            **{**no_div_option.model_dump(), "dividend_yield": 0.03}
        )

        model_no_div = BinomialTreeModel(no_div_option, num_steps=100)
        model_with_div = BinomialTreeModel(with_div_option, num_steps=100)

        price_no_div = model_no_div.price()
        price_with_div = model_with_div.price()

        assert price_with_div > price_no_div, \
            "Dividends should increase put value"


class TestBinomialIntegration:
    """Integration tests for binomial model."""

    def test_price_with_greeks(self, simple_call_option):
        """Test complete pricing workflow with Greeks."""
        model = BinomialTreeModel(simple_call_option, num_steps=100)
        result = model.price_with_greeks()

        assert result.price > 0
        assert result.greeks.delta > 0
        assert result.greeks.gamma > 0
        assert result.method == "BinomialTreeModel"
        assert "num_steps" in result.additional_info
        assert result.additional_info["num_steps"] == 100.0

    def test_different_step_counts(self, simple_call_option):
        """Test that model works with various step counts."""
        for steps in [10, 25, 50, 100, 200]:
            model = BinomialTreeModel(simple_call_option, num_steps=steps)
            price = model.price()
            assert price > 0, f"Price should be positive for {steps} steps"
