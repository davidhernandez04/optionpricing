# Options Pricing Suite 📈

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://optionspricingsuite.streamlit.app)
[![GitHub](https://img.shields.io/badge/GitHub-optionpricing-blue?logo=github)](https://github.com/davidhernandez04/optionpricing)

A professional Python package for options pricing and Greeks calculation, featuring multiple pricing models, real market data integration, and a beautiful CLI interface.

## 🌐 [**Try the Live Web App!**](https://optionspricingsuite.streamlit.app) 🚀

## Features ✨

### Pricing Models
- **Black-Scholes** - Analytical pricing for European options with closed-form Greeks
- **Binomial Tree** - Cox-Ross-Rubinstein model supporting American options and early exercise
- **Monte Carlo** - Simulation-based pricing with antithetic variates variance reduction
- **Implied Volatility** - Newton-Raphson solver with bisection fallback

### Greeks Calculation
- **Analytical Greeks** for Black-Scholes (Delta, Gamma, Theta, Vega, Rho)
- **Numerical Greeks** for Binomial and Monte Carlo models
- Validated against finite difference methods

### Real Market Data
- Integration with Yahoo Finance via yfinance
- Fetch current spot prices, option chains, and risk-free rates
- Automatic dividend yield retrieval

### Professional CLI
- Beautiful terminal output with Rich formatting
- Multiple pricing commands
- Greeks calculation and visualization
- Method comparison tools

## Installation

\`\`\`bash
# Clone the repository
git clone https://github.com/yourusername/optionpricing.git
cd optionpricing

# Install with uv (recommended)
uv sync

# Or install with pip
pip install -e .
\`\`\`

## Quick Start

### Basic Option Pricing

\`\`\`python
from optionpricing import Option, OptionType, ExerciseStyle
from optionpricing.models.black_scholes import BlackScholesModel

# Create an option
option = Option(
    spot_price=100.0,
    strike_price=100.0,
    time_to_expiry=1.0,
    volatility=0.20,
    risk_free_rate=0.05,
    option_type=OptionType.CALL,
    exercise_style=ExerciseStyle.EUROPEAN,
)

# Price with Black-Scholes
model = BlackScholesModel(option)
price = model.price()
greeks = model.greeks()

print(f"Option Price: \${price:.2f}")
print(greeks)
\`\`\`

### Command-Line Interface

\`\`\`bash
# Price an option
uv run python -m optionpricing price --spot 100 --strike 100 --expiry 1 --vol 0.20

# Calculate Greeks
uv run python -m optionpricing greeks --spot 100 --strike 105 --expiry 0.5 --vol 0.25 --type put

# Compute implied volatility
uv run python -m optionpricing iv --spot 100 --strike 100 --expiry 1 --price 10.45

# Compare pricing methods
uv run python -m optionpricing compare --spot 100 --strike 100 --expiry 1 --vol 0.20
\`\`\`

## Testing

\`\`\`bash
# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=optionpricing
\`\`\`

## Technical Highlights

- **Black-Scholes**: Closed-form solutions with analytical Greeks
- **Binomial Tree**: Supports American options with early exercise
- **Monte Carlo**: Antithetic variates variance reduction
- **Implied Volatility**: Newton-Raphson with bisection fallback
- **Type Safe**: Full type hints with Pydantic validation
- **Well Tested**: Comprehensive test suite with >90% coverage

## Requirements

- Python >= 3.10
- NumPy, SciPy, pandas
- yfinance (market data)
- Typer, Rich (CLI)
- Pydantic (validation)

## License

MIT License

## Author

David Hernandez

---

**Built with modern Python best practices - perfect for showcasing quantitative finance and software engineering skills! 🚀**
