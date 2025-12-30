from langchain_core.messages import HumanMessage
from graph.state import AgentState, show_agent_reasoning
from utils.progress import progress
from tools.api import get_prices, prices_to_df
import json


##### Risk Management Agent #####
def risk_management_agent(state: AgentState):
    """Controls position sizing based on real-world risk factors for multiple tickers."""
    portfolio = state["data"]["portfolio"]
    data = state["data"]
    tickers = data["tickers"]

    # Fetch the latest price for each ticker first
    risk_analysis = {}
    current_prices = {}

    for ticker in tickers:
        progress.update_status("risk_management_agent", ticker, "Fetching prices")

        prices = get_prices(
            ticker=ticker,
            start_date=data["start_date"],
            end_date=data["end_date"],
        )

        if not prices:
            progress.update_status("risk_management_agent", ticker, "Failed: No price data found")
            continue

        prices_df = prices_to_df(prices)
        current_prices[ticker] = prices_df["close"].iloc[-1]

    # Calculate total portfolio value using the fetched prices
    total_portfolio_value = portfolio.get("cash", 0.0)
    for tkr in tickers:
        pos = portfolio.get("positions", {}).get(tkr, {})
        price = current_prices.get(tkr, 0.0)
        long_val = pos.get("long", 0) * price
        short_val = pos.get("short", 0) * price
        total_portfolio_value += abs(long_val) + abs(short_val)

    # Position limit is 20% of the portfolio value
    position_limit = total_portfolio_value * 0.20

    # Now compute risk metrics for each ticker
    for ticker in tickers:
        if ticker not in current_prices:
            continue

        progress.update_status("risk_management_agent", ticker, "Calculating limits")

        price = current_prices[ticker]
        position = portfolio.get("positions", {}).get(ticker, {})
        long_val = position.get("long", 0) * price
        short_val = position.get("short", 0) * price
        current_position_value = abs(long_val) + abs(short_val)

        remaining_position_limit = max(position_limit - current_position_value, 0)
        max_position_size = min(remaining_position_limit, portfolio.get("cash", 0))

        risk_analysis[ticker] = {
            "remaining_position_limit": float(max_position_size),
            "current_price": float(price),
            "reasoning": {
                "portfolio_value": float(total_portfolio_value),
                "current_position": float(current_position_value),
                "position_limit": float(position_limit),
                "remaining_limit": float(remaining_position_limit),
                "available_cash": float(portfolio.get("cash", 0)),
            },
        }

        progress.update_status("risk_management_agent", ticker, "Done")

    message = HumanMessage(
        content=json.dumps(risk_analysis),
        name="risk_management_agent",
    )

    if state["metadata"]["show_reasoning"]:
        show_agent_reasoning(risk_analysis, "Risk Management Agent")

    # Add the signal to the analyst_signals list
    state["data"]["analyst_signals"]["risk_management_agent"] = risk_analysis

    return {
        "messages": state["messages"] + [message],
        "data": data,
    }
