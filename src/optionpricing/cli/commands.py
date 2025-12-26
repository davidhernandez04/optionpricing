"""Command-line interface for options pricing."""

import typer
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from typing import Optional
from datetime import datetime

from optionpricing.models.base import Option, OptionType, ExerciseStyle
from optionpricing.models.black_scholes import BlackScholesModel
from optionpricing.models.binomial import BinomialTreeModel
from optionpricing.models.monte_carlo import MonteCarloModel
from optionpricing.volatility.implied import implied_volatility
from optionpricing.data.fetcher import MarketDataFetcher

app = typer.Typer(
    name="optionpricing",
    help="Professional options pricing suite with Black-Scholes, Binomial Tree, and Monte Carlo models",
    add_completion=False,
)

console = Console()


@app.command()
def price(
    spot: float = typer.Option(..., "--spot", "-S", help="Current spot price"),
    strike: float = typer.Option(..., "--strike", "-K", help="Strike price"),
    expiry: float = typer.Option(..., "--expiry", "-T", help="Time to expiry (years)"),
    volatility: float = typer.Option(..., "--vol", "-v", help="Volatility (e.g., 0.20 for 20%)"),
    rate: float = typer.Option(0.05, "--rate", "-r", help="Risk-free rate"),
    option_type: str = typer.Option("call", "--type", "-t", help="Option type (call/put)"),
    method: str = typer.Option("black-scholes", "--method", "-m", help="Pricing method"),
    dividend: float = typer.Option(0.0, "--dividend", "-q", help="Dividend yield"),
):
    """Price an option using specified method."""
    try:
        opt_type = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT

        option = Option(
            spot_price=spot,
            strike_price=strike,
            time_to_expiry=expiry,
            volatility=volatility,
            risk_free_rate=rate,
            option_type=opt_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            dividend_yield=dividend,
        )

        # Select pricing model
        if method == "black-scholes":
            model = BlackScholesModel(option)
        elif method == "binomial":
            model = BinomialTreeModel(option, num_steps=100)
        elif method == "monte-carlo":
            model = MonteCarloModel(option, num_simulations=10000)
        else:
            rprint(f"[red]Unknown method: {method}[/red]")
            return

        price = model.price()

        rprint(f"\n[bold green]Option Price:[/bold green] ${price:.4f}")
        rprint(f"Method: {method}")

    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")


@app.command()
def greeks(
    spot: float = typer.Option(..., "--spot", "-S", help="Current spot price"),
    strike: float = typer.Option(..., "--strike", "-K", help="Strike price"),
    expiry: float = typer.Option(..., "--expiry", "-T", help="Time to expiry (years)"),
    volatility: float = typer.Option(..., "--vol", "-v", help="Volatility"),
    rate: float = typer.Option(0.05, "--rate", "-r", help="Risk-free rate"),
    option_type: str = typer.Option("call", "--type", "-t", help="Option type"),
    dividend: float = typer.Option(0.0, "--dividend", "-q", help="Dividend yield"),
):
    """Calculate option Greeks."""
    try:
        opt_type = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT

        option = Option(
            spot_price=spot,
            strike_price=strike,
            time_to_expiry=expiry,
            volatility=volatility,
            risk_free_rate=rate,
            option_type=opt_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            dividend_yield=dividend,
        )

        model = BlackScholesModel(option)
        result = model.price_with_greeks()

        # Create beautiful table
        table = Table(title="Option Pricing Results", show_header=True)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="magenta")

        table.add_row("Price", f"${result.price:.4f}")
        table.add_row("Delta", f"{result.greeks.delta:.4f}")
        table.add_row("Gamma", f"{result.greeks.gamma:.4f}")
        table.add_row("Theta", f"{result.greeks.theta:.4f}")
        table.add_row("Vega", f"{result.greeks.vega:.4f}")
        table.add_row("Rho", f"{result.greeks.rho:.4f}")

        console.print(table)

    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")


@app.command()
def iv(
    spot: float = typer.Option(..., "--spot", "-S", help="Current spot price"),
    strike: float = typer.Option(..., "--strike", "-K", help="Strike price"),
    expiry: float = typer.Option(..., "--expiry", "-T", help="Time to expiry (years)"),
    market_price: float = typer.Option(..., "--price", "-p", help="Market option price"),
    rate: float = typer.Option(0.05, "--rate", "-r", help="Risk-free rate"),
    option_type: str = typer.Option("call", "--type", "-t", help="Option type"),
    dividend: float = typer.Option(0.0, "--dividend", "-q", help="Dividend yield"),
):
    """Calculate implied volatility from market price."""
    try:
        opt_type = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT

        iv = implied_volatility(
            spot_price=spot,
            strike_price=strike,
            time_to_expiry=expiry,
            risk_free_rate=rate,
            market_price=market_price,
            option_type=opt_type,
            dividend_yield=dividend,
        )

        rprint(f"\n[bold green]Implied Volatility:[/bold green] {iv:.2%} ({iv:.4f})")

    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")


@app.command()
def compare(
    spot: float = typer.Option(..., "--spot", "-S"),
    strike: float = typer.Option(..., "--strike", "-K"),
    expiry: float = typer.Option(..., "--expiry", "-T"),
    volatility: float = typer.Option(..., "--vol", "-v"),
    rate: float = typer.Option(0.05, "--rate", "-r"),
    option_type: str = typer.Option("call", "--type", "-t"),
    dividend: float = typer.Option(0.0, "--dividend", "-q"),
):
    """Compare prices across all pricing methods."""
    try:
        opt_type = OptionType.CALL if option_type.lower() == "call" else OptionType.PUT

        option = Option(
            spot_price=spot,
            strike_price=strike,
            time_to_expiry=expiry,
            volatility=volatility,
            risk_free_rate=rate,
            option_type=opt_type,
            exercise_style=ExerciseStyle.EUROPEAN,
            dividend_yield=dividend,
        )

        # Price with all methods
        bs_model = BlackScholesModel(option)
        bin_model = BinomialTreeModel(option, num_steps=150)
        mc_model = MonteCarloModel(option, num_simulations=50000)

        bs_price = bs_model.price()
        bin_price = bin_model.price()
        mc_price = mc_model.price()

        # Create comparison table
        table = Table(title="Pricing Method Comparison", show_header=True)
        table.add_column("Method", style="cyan")
        table.add_column("Price", style="magenta")
        table.add_column("Difference from BS", style="yellow")

        table.add_row("Black-Scholes", f"${bs_price:.4f}", "-")
        table.add_row(
            "Binomial (150 steps)",
            f"${bin_price:.4f}",
            f"{(bin_price - bs_price):.4f} ({(bin_price/bs_price - 1):.2%})"
        )
        table.add_row(
            "Monte Carlo (50k sims)",
            f"${mc_price:.4f}",
            f"{(mc_price - bs_price):.4f} ({(mc_price/bs_price - 1):.2%})"
        )

        console.print(table)

    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")


@app.command()
def fetch(
    symbol: str = typer.Argument(..., help="Stock ticker symbol"),
    expiration: Optional[str] = typer.Option(None, "--expiry", "-e", help="Expiration date (YYYY-MM-DD)"),
):
    """Fetch real market data for a symbol."""
    try:
        fetcher = MarketDataFetcher()

        rprint(f"\n[bold]Fetching data for {symbol}...[/bold]\n")

        # Get spot price
        spot = fetcher.get_spot_price(symbol)
        rprint(f"Spot Price: [green]${spot:.2f}[/green]")

        # Get risk-free rate
        rate = fetcher.get_risk_free_rate()
        rprint(f"Risk-Free Rate: [green]{rate:.2%}[/green]")

        # Get dividend yield
        div = fetcher.get_dividend_yield(symbol)
        rprint(f"Dividend Yield: [green]{div:.2%}[/green]")

        # Get option chain
        chain = fetcher.get_option_chain(symbol, expiration)
        rprint(f"\n[bold]Option Chain ({chain['expiration'].iloc[0]}):[/bold]")
        rprint(f"Total Contracts: {len(chain)}")

    except Exception as e:
        rprint(f"[red]Error: {str(e)}[/red]")


if __name__ == "__main__":
    app()
