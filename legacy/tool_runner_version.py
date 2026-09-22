"""
Advanced version using Claude's Tool Runner for agentic workflow.
This allows Claude to autonomously call tools and iterate on strategies.
"""

import json
from typing import Any
from anthropic import Anthropic

client = Anthropic()

tools = [
    {
        "name": "parse_strategy",
        "description": "Extract and structure strategy components from a plain English description",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_description": {
                    "type": "string",
                    "description": "The trading strategy described in plain English"
                }
            },
            "required": ["strategy_description"]
        }
    },
    {
        "name": "generate_strategy_code",
        "description": "Generate production-ready code implementing the strategy",
        "input_schema": {
            "type": "object",
            "properties": {
                "parsed_components": {
                    "type": "object",
                    "description": "The parsed strategy components (from parse_strategy)",
                    "properties": {
                        "entry_conditions": {"type": "string"},
                        "exit_conditions": {"type": "string"},
                        "position_sizing": {"type": "string"},
                        "risk_parameters": {"type": "object"}
                    }
                },
                "language": {
                    "type": "string",
                    "description": "Programming language (python, pinescript, etc)",
                    "default": "python"
                }
            },
            "required": ["parsed_components"]
        }
    },
    {
        "name": "validate_for_live_trading",
        "description": "Validate strategy for live trading safety and correctness",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_code": {
                    "type": "string",
                    "description": "The generated strategy code"
                },
                "original_description": {
                    "type": "string",
                    "description": "Original strategy description to validate against"
                }
            },
            "required": ["strategy_code", "original_description"]
        }
    },
    {
        "name": "backtest_recommendations",
        "description": "Provide backtesting setup and performance metrics recommendations",
        "input_schema": {
            "type": "object",
            "properties": {
                "strategy_code": {
                    "type": "string",
                    "description": "The strategy code to backtest"
                },
                "market": {
                    "type": "string",
                    "description": "Target market (e.g., 'stocks', 'crypto', 'forex')"
                }
            },
            "required": ["strategy_code", "market"]
        }
    }
]


def execute_tool(tool_name: str, tool_input: dict) -> str:
    """Execute a tool and return the result."""

    if tool_name == "parse_strategy":
        return parse_strategy(tool_input["strategy_description"])

    elif tool_name == "generate_strategy_code":
        parsed = tool_input["parsed_components"]
        language = tool_input.get("language", "python")
        return generate_code(parsed, language)

    elif tool_name == "validate_for_live_trading":
        return validate_strategy(
            tool_input["strategy_code"],
            tool_input["original_description"]
        )

    elif tool_name == "backtest_recommendations":
        return backtest_recommendations(
            tool_input["strategy_code"],
            tool_input["market"]
        )

    else:
        return f"Unknown tool: {tool_name}"


def parse_strategy(description: str) -> str:
    """Parse strategy from description."""
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=800,
        messages=[{
            "role": "user",
            "content": f"""Parse this trading strategy into structured JSON:

{description}

Return ONLY valid JSON with these fields:
- entry_conditions: entry rules
- exit_conditions: exit rules
- position_sizing: position sizing logic
- risk_parameters: risk management rules
- market: target market (stocks/crypto/forex)
- timeframe: trading timeframe"""
        }]
    )
    return response.content[0].text


def generate_code(parsed: dict, language: str) -> str:
    """Generate strategy code."""
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1500,
        messages=[{
            "role": "user",
            "content": f"""Generate {language} code for this strategy:

{json.dumps(parsed, indent=2)}

Requirements:
- Production-ready with error handling
- Clear comments explaining logic
- Data validation
- Realistic trading assumptions
- Return buy/sell signals

Provide complete, working code."""
        }]
    )
    return response.content[0].text


def validate_strategy(code: str, original: str) -> str:
    """Validate strategy for live trading."""
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1000,
        messages=[{
            "role": "user",
            "content": f"""Validate this strategy for live trading:

Original: {original}

Code: {code}

Check for:
1. Logical consistency
2. Risk management sufficiency
3. Implementation correctness
4. Biases (look-ahead, overfitting)
5. Readiness for real money

Provide risk level (green/yellow/red) and key issues."""
        }]
    )
    return response.content[0].text


def backtest_recommendations(code: str, market: str) -> str:
    """Provide backtesting recommendations."""
    response = client.messages.create(
        model="claude-opus-5",
        max_tokens=1200,
        messages=[{
            "role": "user",
            "content": f"""For this {market} strategy:

{code}

Provide:
1. Key performance metrics to track
2. Backtest parameters (dates, starting capital)
3. Data quality checks needed
4. Backtesting code setup (Python example)
5. Realistic assumptions (slippage, commissions)"""
        }]
    )
    return response.content[0].text


def build_strategy_agent(strategy_description: str):
    """Use Claude as an agent to build and validate the strategy."""
    print("\n" + "="*70)
    print("TRADING STRATEGY BUILDER - Agent Mode")
    print("="*70)

    messages = [
        {
            "role": "user",
            "content": f"""Build a complete trading strategy from this description:

{strategy_description}

Steps:
1. Parse the strategy into components
2. Generate production code
3. Validate for live trading
4. Provide backtest recommendations

Be thorough and ask me questions if clarification is needed."""
        }
    ]

    step = 0
    max_steps = 10

    while step < max_steps:
        step += 1
        print(f"\n[Step {step}] Calling Claude...")

        response = client.messages.create(
            model="claude-opus-5",
            max_tokens=2000,
            tools=tools,
            messages=messages
        )

        print(f"Stop reason: {response.stop_reason}")

        if response.stop_reason == "end_turn":
            print("\n" + "="*70)
            print("STRATEGY BUILDING COMPLETE")
            print("="*70)
            for block in response.content:
                if hasattr(block, "text"):
                    print(block.text)
            break

        tool_calls = [block for block in response.content if block.type == "tool_use"]

        if not tool_calls:
            print("No tool calls found, completing...")
            for block in response.content:
                if hasattr(block, "text"):
                    print(block.text)
            break

        messages.append({"role": "assistant", "content": response.content})

        tool_results = []
        for tool_call in tool_calls:
            print(f"  → Executing: {tool_call.name}")
            result = execute_tool(tool_call.name, tool_call.input)
            print(f"    Result length: {len(result)} chars")

            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tool_call.id,
                "content": result
            })

        messages.append({"role": "user", "content": tool_results})

    return {
        "strategy_description": strategy_description,
        "final_response": response
    }


if __name__ == "__main__":
    example_strategy = """
    I want a moving average crossover strategy for stocks.

    Entry: When 20-day MA crosses above 50-day MA and price is above both MAs.
    Exit: When 20-day MA crosses below 50-day MA.
    Position sizing: Risk 1% per trade.
    Stop loss: 2% below entry.
    Take profit: 5% above entry.
    """

    result = build_strategy_agent(example_strategy)
