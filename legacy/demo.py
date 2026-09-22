#!/usr/bin/env python3
"""
Demo script showing all capabilities of the trading strategy builder.
Run this to see the full workflow in action.
"""

import json
from strategy_builder import build_strategy_workflow


def demo_basic_workflow():
    """Demonstrate the basic sequential workflow."""
    print("\n" + "="*70)
    print("DEMO: Basic Sequential Workflow")
    print("="*70)

    strategy = """
    Simple Moving Average Crossover Strategy for Stocks:

    Entry: When 20-day SMA crosses above 50-day SMA
    Exit: When 20-day SMA crosses below 50-day SMA
    Position Size: 2% of account per trade
    Stop Loss: 3% below entry
    Take Profit: 5% above entry (or when signal flips)
    Risk Rule: Never risk more than 1% per trade
    """

    result = build_strategy_workflow(strategy)

    print("\n" + "-"*70)
    print("Generated Code Snippet:")
    print("-"*70)
    print(result["generated_code"][:800])
    if len(result["generated_code"]) > 800:
        print("\n[... output truncated ...]")

    return result


def demo_complex_strategy():
    """Demonstrate with a more complex strategy."""
    print("\n\n" + "="*70)
    print("DEMO: Complex Strategy with Multiple Filters")
    print("="*70)

    strategy = """
    Multi-Factor Momentum Strategy for Tech Stocks:

    Filters:
    - Only trade stocks with market cap > $5B
    - Only trade during 9:30-15:00 ET (liquid hours)
    - Skip earnings dates

    Entry Setup:
    - Price breaks above 20-day high (momentum)
    - AND RSI(14) crosses above 50 (strength confirmation)
    - AND Volume > 1.5x average (participation)
    - AND Stock within top 20% momentum (sector filter)

    Position Management:
    - Enter at breakout close
    - First profit target: +3% (take 50% of position)
    - Second profit target: +7% (take 30% of position)
    - Trailing stop: 5% for remaining 20%
    - Time stop: Close after 10 days if not filled

    Risk Parameters:
    - Account risk per trade: 0.5% (tight risk)
    - Max account drawdown: 6%
    - Max concurrent positions: 4
    - Daily loss limit: 2%
    - Max per-sector exposure: 3 positions

    Exit Conditions:
    - Take profit levels (as above)
    - Stop loss: 2% below entry (strict)
    - Technical breakdown: Close below 10-day SMA
    - Sell signal: RSI crosses below 50
    """

    result = build_strategy_workflow(strategy)

    print("\n" + "-"*70)
    print("Validation Results:")
    print("-"*70)
    print(result["validation"][:600])
    if len(result["validation"]) > 600:
        print("\n[... output truncated ...]")

    return result


def demo_crypto_strategy():
    """Demonstrate with a crypto-specific strategy."""
    print("\n\n" + "="*70)
    print("DEMO: Crypto Strategy with Volatility Adjustment")
    print("="*70)

    strategy = """
    Adaptive Volatility Trading Strategy for Crypto:

    Timeframe: 4-hour candles

    Dynamic Entry:
    - Calculate ATR(20) for volatility
    - If ATR is LOW (below 20-day median):
      * Entry: Price breaks above Bollinger Band high
      * Position size: 1.5x normal (leverage opportunity)
    - If ATR is HIGH (above 20-day median):
      * Entry: Mean reversion, price touches Bollinger Band low
      * Position size: 0.5x normal (reduce risk)

    Exit Logic:
    - Profit target scales with volatility:
      * Low volatility: 4% target
      * High volatility: 2% target
    - Stop loss always at 2% (fixed)
    - Breakeven stop after +1% profit

    Risk Management:
    - Account risk: 1% per trade (crypto allows slightly more)
    - Maximum drawdown: 10%
    - Position overlap limit: No more than 20% overlap in time
    - Daily volume filter: Only trade if 24h volume > $100M

    Special Rules:
    - Pause trading during consensus changes or hard forks
    - Use 0.1% position size during weekends (lower liquidity)
    - Scale out 50% at +2%, 30% at +3%, 20% trailing
    """

    result = build_strategy_workflow(strategy)

    print("\n" + "-"*70)
    print("Backtest Recommendations:")
    print("-"*70)
    print(result["backtest_plan"][:600])
    if len(result["backtest_plan"]) > 600:
        print("\n[... output truncated ...]")

    return result


def save_results(results: dict, filename: str = "demo_results.json"):
    """Save all demo results to a file."""
    with open(filename, "w") as f:
        # Convert to JSON-serializable format
        json_results = {
            "basic": results[0],
            "complex": results[1],
            "crypto": results[2],
        }
        json.dump(json_results, f, indent=2, default=str)
    print(f"\n✓ All results saved to {filename}")


def main():
    """Run all demos."""
    print("\n" + "="*70)
    print("TRADING STRATEGY BUILDER - Interactive Demo")
    print("="*70)
    print("""
This demo showcases the full capabilities:
1. Basic strategy (simple moving average crossover)
2. Complex strategy (with multiple filters)
3. Crypto strategy (with volatility adaptation)

Each demo will:
- Parse the strategy description
- Generate production-ready code
- Create backtest recommendations
- Provide validation and risk assessment

This typically takes 30-60 seconds per strategy.
    """)

    input("Press Enter to start the demo...")

    print("\n⏳ Running demos (this may take a minute)...\n")

    results = []

    try:
        results.append(demo_basic_workflow())
        results.append(demo_complex_strategy())
        results.append(demo_crypto_strategy())

        save_results(results)

        print("\n" + "="*70)
        print("✓ DEMO COMPLETE")
        print("="*70)
        print("""
Summary:
- Basic strategy: Generated and validated ✓
- Complex strategy: Multi-filter strategy processed ✓
- Crypto strategy: Volatility-adjusted strategy created ✓

All results saved to: demo_results.json

Next Steps:
1. Review the generated code
2. Customize the strategies for your needs
3. Run backtests with real market data
4. Paper trade before live deployment

Try it yourself:
  python cli.py --interactive
  python cli.py -f example_strategy.txt
  python tool_runner_version.py
        """)

    except Exception as e:
        print(f"\n✗ Demo failed with error:")
        print(f"  {type(e).__name__}: {e}")
        print("\nTroubleshooting:")
        print("1. Verify ANTHROPIC_API_KEY is set")
        print("2. Check your internet connection")
        print("3. Ensure you have API credits available")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
