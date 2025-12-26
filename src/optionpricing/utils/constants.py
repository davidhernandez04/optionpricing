"""Constants used throughout the options pricing package."""

import math

# Mathematical constants
SQRT_2PI = math.sqrt(2 * math.pi)

# Trading constants
TRADING_DAYS_PER_YEAR = 252
CALENDAR_DAYS_PER_YEAR = 365

# Numerical constants for algorithms
DEFAULT_TOLERANCE = 1e-6
MAX_ITERATIONS = 100

# Monte Carlo defaults
DEFAULT_NUM_SIMULATIONS = 10000
DEFAULT_RANDOM_SEED = 42

# Binomial tree defaults
DEFAULT_NUM_STEPS = 100

# Implied volatility solver defaults
IV_SOLVER_TOLERANCE = 1e-5
IV_SOLVER_MAX_ITERATIONS = 50
IV_INITIAL_GUESS = 0.3  # 30% volatility
IV_MIN_VALUE = 0.001  # 0.1% minimum vol
IV_MAX_VALUE = 5.0  # 500% maximum vol
