"""Streamlit Web App for Options Pricing Suite.

A beautiful, interactive web interface for pricing options and calculating Greeks.
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from optionpricing import Option, OptionType, ExerciseStyle
from optionpricing.models.black_scholes import BlackScholesModel
from optionpricing.models.binomial import BinomialTreeModel
from optionpricing.models.monte_carlo import MonteCarloModel
from optionpricing.volatility.implied import implied_volatility, ImpliedVolatilityError
from optionpricing.data.fetcher import MarketDataFetcher

# Page configuration
st.set_page_config(
    page_title="Options Pricing Suite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📈 Options Pricing Suite</h1>', unsafe_allow_html=True)
st.markdown("### Professional options pricing with Black-Scholes, Binomial Tree, and Monte Carlo models")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")

    mode = st.radio(
        "Select Mode:",
        ["Manual Input", "Real Market Data", "Implied Volatility Calculator"],
        help="Choose how to input option parameters"
    )

    st.markdown("---")

    if mode == "Real Market Data":
        st.subheader("📊 Market Data")
        symbol = st.text_input("Stock Symbol", value="AAPL", help="Enter ticker symbol (e.g., AAPL, TSLA, SPY)")

        if st.button("Fetch Data", type="primary"):
            with st.spinner(f"Fetching {symbol} data..."):
                try:
                    fetcher = MarketDataFetcher()
                    spot = fetcher.get_spot_price(symbol)
                    rate = fetcher.get_risk_free_rate()
                    div = fetcher.get_dividend_yield(symbol)

                    st.session_state.spot = spot
                    st.session_state.rate = rate
                    st.session_state.div = div
                    st.success(f"✅ Data fetched for {symbol}!")
                except Exception as e:
                    st.error(f"Error fetching data: {str(e)}")

    st.markdown("---")
    st.caption("Built by David Hernandez")
    st.caption("[GitHub Repository](https://github.com/davidhernandez04/optionpricing)")

# Main content
if mode == "Implied Volatility Calculator":
    st.header("🔍 Implied Volatility Calculator")
    st.markdown("Reverse-engineer the volatility implied by market prices")

    col1, col2 = st.columns(2)

    with col1:
        iv_spot = st.number_input("Spot Price ($)", value=100.0, min_value=0.01)
        iv_strike = st.number_input("Strike Price ($)", value=100.0, min_value=0.01)
        iv_market_price = st.number_input("Market Option Price ($)", value=10.45, min_value=0.01)

    with col2:
        iv_expiry = st.number_input("Time to Expiry (years)", value=1.0, min_value=0.01, max_value=10.0)
        iv_rate = st.number_input("Risk-Free Rate", value=0.05, min_value=0.0, max_value=1.0, format="%.4f")
        iv_type = st.selectbox("Option Type", ["Call", "Put"])

    if st.button("Calculate Implied Volatility", type="primary"):
        try:
            opt_type = OptionType.CALL if iv_type == "Call" else OptionType.PUT

            iv = implied_volatility(
                spot_price=iv_spot,
                strike_price=iv_strike,
                time_to_expiry=iv_expiry,
                risk_free_rate=iv_rate,
                market_price=iv_market_price,
                option_type=opt_type,
            )

            st.success(f"### Implied Volatility: **{iv:.2%}** ({iv:.4f})")

            # Show comparison
            st.markdown("---")
            st.subheader("Verification")
            option = Option(
                spot_price=iv_spot,
                strike_price=iv_strike,
                time_to_expiry=iv_expiry,
                volatility=iv,
                risk_free_rate=iv_rate,
                option_type=opt_type,
                exercise_style=ExerciseStyle.EUROPEAN,
            )

            bs_model = BlackScholesModel(option)
            calculated_price = bs_model.price()

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Market Price", f"${iv_market_price:.4f}")
            with col2:
                st.metric("Calculated Price", f"${calculated_price:.4f}",
                         delta=f"${calculated_price - iv_market_price:.4f}")

        except ImpliedVolatilityError as e:
            st.error(f"Could not calculate implied volatility: {str(e)}")
        except Exception as e:
            st.error(f"Error: {str(e)}")

else:
    # Option Parameters
    st.header("📋 Option Parameters")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        spot = st.number_input(
            "Spot Price ($)",
            value=st.session_state.get('spot', 100.0),
            min_value=0.01,
            help="Current price of the underlying asset"
        )

    with col2:
        strike = st.number_input(
            "Strike Price ($)",
            value=100.0,
            min_value=0.01,
            help="Exercise price of the option"
        )

    with col3:
        expiry = st.number_input(
            "Time to Expiry (years)",
            value=1.0,
            min_value=0.01,
            max_value=10.0,
            help="Time until option expiration"
        )

    with col4:
        volatility = st.slider(
            "Volatility",
            min_value=0.05,
            max_value=1.0,
            value=0.20,
            step=0.01,
            format="%.2f",
            help="Annualized volatility (standard deviation)"
        )

    col5, col6, col7 = st.columns(3)

    with col5:
        rate = st.number_input(
            "Risk-Free Rate",
            value=st.session_state.get('rate', 0.05),
            min_value=0.0,
            max_value=1.0,
            format="%.4f",
            help="Annualized risk-free interest rate"
        )

    with col6:
        dividend = st.number_input(
            "Dividend Yield",
            value=st.session_state.get('div', 0.0),
            min_value=0.0,
            max_value=1.0,
            format="%.4f",
            help="Annualized dividend yield"
        )

    with col7:
        option_type = st.selectbox("Option Type", ["Call", "Put"])

    # Create option
    opt_type = OptionType.CALL if option_type == "Call" else OptionType.PUT

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

    # Pricing Results
    st.markdown("---")
    st.header("💰 Pricing Results")

    # Calculate prices
    bs_model = BlackScholesModel(option)
    bs_result = bs_model.price_with_greeks()

    bin_model = BinomialTreeModel(option, num_steps=100)
    bin_price = bin_model.price()

    mc_model = MonteCarloModel(option, num_simulations=10000, use_antithetic=True)
    mc_price = mc_model.price()

    # Display prices
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Black-Scholes",
            f"${bs_result.price:.4f}",
            help="Analytical pricing model"
        )

    with col2:
        diff_bin = bin_price - bs_result.price
        st.metric(
            "Binomial Tree",
            f"${bin_price:.4f}",
            delta=f"${diff_bin:.4f}",
            help="Discrete-time lattice model"
        )

    with col3:
        diff_mc = mc_price - bs_result.price
        st.metric(
            "Monte Carlo",
            f"${mc_price:.4f}",
            delta=f"${diff_mc:.4f}",
            help="Simulation-based pricing"
        )

    # Greeks
    st.markdown("---")
    st.header("📊 Greeks Analysis")

    greeks = bs_result.greeks

    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Delta (Δ)", f"{greeks.delta:.4f}", help="Sensitivity to spot price")
    with col2:
        st.metric("Gamma (Γ)", f"{greeks.gamma:.4f}", help="Sensitivity of Delta")
    with col3:
        st.metric("Theta (Θ)", f"{greeks.theta:.4f}", help="Time decay per day")
    with col4:
        st.metric("Vega (ν)", f"{greeks.vega:.4f}", help="Sensitivity to volatility")
    with col5:
        st.metric("Rho (ρ)", f"{greeks.rho:.4f}", help="Sensitivity to interest rate")

    # Visualizations
    st.markdown("---")
    st.header("📈 Visualizations")

    tab1, tab2, tab3 = st.tabs(["Greeks Profile", "Price Comparison", "Payoff Diagram"])

    with tab1:
        st.subheader("Delta and Gamma vs Spot Price")

        # Generate data
        spot_range = np.linspace(spot * 0.7, spot * 1.3, 50)
        deltas = []
        gammas = []

        for s in spot_range:
            temp_option = Option(
                spot_price=s,
                strike_price=strike,
                time_to_expiry=expiry,
                volatility=volatility,
                risk_free_rate=rate,
                option_type=opt_type,
                exercise_style=ExerciseStyle.EUROPEAN,
                dividend_yield=dividend,
            )
            temp_model = BlackScholesModel(temp_option)
            deltas.append(temp_model.delta())
            gammas.append(temp_model.gamma())

        # Create figure
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=deltas, name='Delta', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=spot_range, y=gammas, name='Gamma', line=dict(color='red', width=3), yaxis='y2'))

        fig.update_layout(
            title='Greeks Profile',
            xaxis_title='Spot Price ($)',
            yaxis_title='Delta',
            yaxis2=dict(title='Gamma', overlaying='y', side='right'),
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        st.subheader("Option Price vs Spot Price")

        prices_bs = []
        prices_bin = []

        for s in spot_range:
            temp_option = Option(
                spot_price=s,
                strike_price=strike,
                time_to_expiry=expiry,
                volatility=volatility,
                risk_free_rate=rate,
                option_type=opt_type,
                exercise_style=ExerciseStyle.EUROPEAN,
                dividend_yield=dividend,
            )
            prices_bs.append(BlackScholesModel(temp_option).price())
            prices_bin.append(BinomialTreeModel(temp_option, num_steps=50).price())

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=prices_bs, name='Black-Scholes', line=dict(color='blue', width=3)))
        fig.add_trace(go.Scatter(x=spot_range, y=prices_bin, name='Binomial', line=dict(color='green', width=2, dash='dash')))

        fig.update_layout(
            title='Option Price Comparison',
            xaxis_title='Spot Price ($)',
            yaxis_title='Option Price ($)',
            hovermode='x unified',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        st.subheader("Payoff Diagram at Expiration")

        payoffs = []
        for s in spot_range:
            if opt_type == OptionType.CALL:
                payoff = max(s - strike, 0)
            else:
                payoff = max(strike - s, 0)
            payoffs.append(payoff)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=spot_range, y=payoffs, fill='tozeroy', name='Payoff',
                                line=dict(color='green', width=3)))

        fig.add_hline(y=0, line_dash="dash", line_color="gray")
        fig.add_vline(x=strike, line_dash="dash", line_color="red", annotation_text="Strike")

        fig.update_layout(
            title=f'{option_type} Option Payoff at Expiration',
            xaxis_title='Spot Price at Expiration ($)',
            yaxis_title='Payoff ($)',
            height=500
        )

        st.plotly_chart(fig, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Built with ❤️ using Python, Streamlit, and advanced quantitative finance models</p>
        <p>Created by David Hernandez | <a href='https://github.com/davidhernandez04/optionpricing'>GitHub</a></p>
    </div>
""", unsafe_allow_html=True)
