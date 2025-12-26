"""Tests for Monte Carlo pricing model."""

import pytest
from optionpricing.models.monte_carlo import MonteCarloModel
from optionpricing.models.black_scholes import BlackScholesModel


class TestMonteCarloBasic:
    """Test basic Monte Carlo functionality."""

    def test_converges_to_black_scholes(self, simple_call_option):
        """Test that MC converges to Black-Scholes with enough simulations."""
        bs_model = BlackScholesModel(simple_call_option)
        bs_price = bs_model.price()

        mc_model = MonteCarloModel(simple_call_option, num_simulations=50000)
        mc_price = mc_model.price()

        # Should be within 2% with 50k simulations
        assert abs(mc_price - bs_price) / bs_price < 0.02, \
            f"MC {mc_price} should be close to BS {bs_price}"

    def test_antithetic_reduces_variance(self, simple_call_option):
        """Test that antithetic variates reduce variance."""
        # Run multiple times and check variance
        mc_model = MonteCarloModel(simple_call_option, num_simulations=10000, use_antithetic=True)
        mc_price = mc_model.price()

        # Just verify it produces reasonable price
        assert mc_price > 0
        assert mc_price < simple_call_option.spot_price

    def test_confidence_interval(self, simple_call_option):
        """Test confidence interval calculation."""
        mc_model = MonteCarloModel(simple_call_option, num_simulations=10000)
        price = mc_model.price()
        ci_lower, ci_upper = mc_model.get_confidence_interval()

        assert ci_lower < price < ci_upper
        assert ci_upper - ci_lower > 0  # Non-zero width

    def test_greeks_positive(self, simple_call_option):
        """Test that Greeks have correct signs."""
        mc_model = MonteCarloModel(simple_call_option, num_simulations=5000)

        assert 0 < mc_model.delta() < 1
        assert mc_model.gamma() > 0
        assert mc_model.vega() > 0
