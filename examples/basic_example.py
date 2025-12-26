"""Basic example of using the options pricing suite."""

from optionpricing import Option, OptionType, ExerciseStyle
from optionpricing.models.black_scholes import BlackScholesModel
from optionpricing.models.binomial import BinomialTreeModel
from optionpricing.models.monte_carlo import MonteCarloModel


def main():
    # Create an at-the-money European call option
    option = Option(
        spot_price=100.0,
        strike_price=100.0,
        time_to_expiry=1.0,
        volatility=0.20,
        risk_free_rate=0.05,
        option_type=OptionType.CALL,
        exercise_style=ExerciseStyle.EUROPEAN,
    )

    print("=" * 60)
    print("OPTIONS PRICING SUITE - Basic Example")
    print("=" * 60)
    print(f"\nOption Parameters:")
    print(f"  Spot Price: ${option.spot_price}")
    print(f"  Strike Price: ${option.strike_price}")
    print(f"  Time to Expiry: {option.time_to_expiry} years")
    print(f"  Volatility: {option.volatility:.0%}")
    print(f"  Risk-Free Rate: {option.risk_free_rate:.0%}")
    print(f"  Option Type: {option.option_type.value.upper()}")
    print(f"  Exercise Style: {option.exercise_style.value.upper()}")

    # Black-Scholes pricing
    print("\n" + "-" * 60)
    print("BLACK-SCHOLES MODEL (Analytical)")
    print("-" * 60)
    bs_model = BlackScholesModel(option)
    bs_result = bs_model.price_with_greeks()
    print(f"Price: ${bs_result.price:.4f}")
    print(f"\nGreeks:")
    print(f"  Delta: {bs_result.greeks.delta:>8.4f}")
    print(f"  Gamma: {bs_result.greeks.gamma:>8.4f}")
    print(f"  Theta: {bs_result.greeks.theta:>8.4f}")
    print(f"  Vega:  {bs_result.greeks.vega:>8.4f}")
    print(f"  Rho:   {bs_result.greeks.rho:>8.4f}")

    # Binomial Tree pricing
    print("\n" + "-" * 60)
    print("BINOMIAL TREE MODEL")
    print("-" * 60)
    bin_model = BinomialTreeModel(option, num_steps=150)
    bin_price = bin_model.price()
    print(f"Price: ${bin_price:.4f}")
    print(f"Difference from BS: ${bin_price - bs_result.price:.4f}")

    # Monte Carlo pricing
    print("\n" + "-" * 60)
    print("MONTE CARLO SIMULATION")
    print("-" * 60)
    mc_model = MonteCarloModel(option, num_simulations=50000, use_antithetic=True)
    mc_price = mc_model.price()
    ci_lower, ci_upper = mc_model.get_confidence_interval()
    print(f"Price: ${mc_price:.4f}")
    print(f"95% CI: [${ci_lower:.4f}, ${ci_upper:.4f}]")
    print(f"Difference from BS: ${mc_price - bs_result.price:.4f}")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
