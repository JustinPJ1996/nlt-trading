import json
from dataclasses import dataclass
from anthropic import Anthropic

client = Anthropic()

@dataclass
class StrategyParsed:
    entry_conditions: str
    exit_conditions: str
    position_sizing: str
    risk_parameters: dict
    raw_description: str


def parse_strategy_tool(strategy_description: str) -> str:
    """Extract strategy components from plain English description."""
    prompt = f"""Analyze this trading strategy description and extract key components:

{strategy_description}

Provide a structured JSON response with these fields:
- entry_conditions: Clear rules for entering positions
- exit_conditions: Rules for closing positions
- position_sizing: How position size is determined
- risk_parameters: Risk management rules (max loss, stop loss, etc.)
- potential_risks: Identified risks or edge cases

Also note any missing information that should be clarified."""

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def generate_code_tool(parsed_strategy: dict, language: str = "python") -> str:
    """Generate executable strategy code from parsed components."""
    strategy_json = json.dumps(parsed_strategy, indent=2)

    prompt = f"""Generate {language} code for this trading strategy:

{strategy_json}

Requirements:
1. Use industry-standard libraries (pandas for Python)
2. Include clear comments explaining each rule
3. Add data validation and error handling
4. Include logging for debugging
5. Return buy/sell signals based on the parsed rules
6. Handle edge cases (missing data, invalid values)

Provide complete, production-ready code that can be backtested."""

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=2000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def backtest_analysis_tool(code: str, test_scenario: str) -> str:
    """Analyze strategy code and suggest backtesting approach."""
    prompt = f"""Given this trading strategy code:

{code}

For this test scenario: {test_scenario}

Provide:
1. Key metrics to track (Sharpe ratio, max drawdown, win rate, etc.)
2. Recommended backtest parameters (date range, starting capital, etc.)
3. Potential data quality issues to watch for
4. Risk management validation points
5. Python code snippet showing how to set up backtesting with this strategy

Focus on realistic assumptions (slippage, commissions, liquidity)."""

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1500,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def validate_strategy_tool(strategy_description: str, generated_code: str) -> str:
    """Validate strategy logic and generated code."""
    prompt = f"""Review this trading strategy for safety and logical consistency:

Original description:
{strategy_description}

Generated code:
{generated_code}

Check for:
1. Logical coherence (do entry/exit rules make sense together?)
2. Risk management (are there adequate safeguards?)
3. Implementation correctness (does code match description?)
4. Potential issues (look-ahead bias, overfitting signs, data leakage)
5. Live trading readiness (what needs adjustment for real money?)

Provide a risk assessment and recommendations."""

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1200,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def deployment_checklist_tool(strategy_description: str, generated_code: str) -> str:
    """Generate live deployment checklist."""
    prompt = f"""Create a deployment checklist for this strategy:

Strategy: {strategy_description}

Code: {generated_code}

Provide a checklist covering:
1. Pre-deployment testing (unit tests, integration tests)
2. Broker API setup (authentication, account limits)
3. Risk safeguards (max position size, daily loss limits, circuit breakers)
4. Monitoring and alerting (what to watch, alert thresholds)
5. Gradual deployment (paper trading -> micro -> full)
6. Kill-switch procedures (when to stop trading)
7. Performance tracking (metrics to monitor)

Format as a markdown checklist."""

    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1000,
        messages=[{"role": "user", "content": prompt}]
    )
    return response.content[0].text


def build_strategy_workflow(strategy_description: str, target_language: str = "python") -> dict:
    """Orchestrate the full strategy building workflow."""
    print("\n" + "="*60)
    print("TRADING STRATEGY BUILDER")
    print("="*60)

    print("\n[1/5] Parsing strategy description...")
    parsed = parse_strategy_tool(strategy_description)
    print("✓ Strategy parsed")
    print(parsed)

    try:
        parsed_dict = json.loads(parsed[parsed.find('{'):parsed.rfind('}')+1])
    except:
        parsed_dict = {"raw_parsed": parsed}

    print("\n[2/5] Generating strategy code...")
    code = generate_code_tool(parsed_dict, target_language)
    print("✓ Code generated")
    print(code[:500] + "..." if len(code) > 500 else code)

    print("\n[3/5] Analyzing backtest approach...")
    backtest_plan = backtest_analysis_tool(code, f"Test on historical {target_language} market data")
    print("✓ Backtest plan created")
    print(backtest_plan[:400] + "..." if len(backtest_plan) > 400 else backtest_plan)

    print("\n[4/5] Validating strategy...")
    validation = validate_strategy_tool(strategy_description, code)
    print("✓ Validation complete")
    print(validation[:400] + "..." if len(validation) > 400 else validation)

    print("\n[5/5] Creating deployment checklist...")
    checklist = deployment_checklist_tool(strategy_description, code)
    print("✓ Checklist created")
    print(checklist[:500] + "..." if len(checklist) > 500 else checklist)

    return {
        "strategy_description": strategy_description,
        "parsed_strategy": parsed_dict,
        "generated_code": code,
        "backtest_plan": backtest_plan,
        "validation": validation,
        "deployment_checklist": checklist,
    }


if __name__ == "__main__":
    example_strategy = """
    I want to build a mean reversion strategy for cryptocurrency trading.

    Entry: When Bitcoin price drops 5% in the last 1 hour from its 24-hour high,
    buy if RSI is below 30 and volume is above average.

    Exit: Take profit at 3% gain or stop loss at 2% loss, whichever comes first.

    Position sizing: Risk 1% of account per trade.

    Risk rules: Never have more than 3 concurrent positions, daily loss limit of 5%.
    """

    result = build_strategy_workflow(example_strategy, target_language="python")

    print("\n" + "="*60)
    print("WORKFLOW COMPLETE - Results saved to: strategy_result.json")
    print("="*60)

    with open("strategy_result.json", "w") as f:
        json.dump(result, f, indent=2)
