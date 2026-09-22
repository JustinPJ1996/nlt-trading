# Trading Strategy Builder - LLM-Powered

Build, validate, and backtest trading strategies using Claude AI. Describe your strategy in plain English, and Claude will generate production-ready code, validate it, and provide backtesting recommendations.

## Features

- **Plain English to Code**: Describe your trading strategy, Claude generates the code
- **Strategy Validation**: Automatic validation for live trading safety
- **Backtest Planning**: Recommendations for backtesting setup and metrics
- **Multi-Language Support**: Python, Pine Script (extensible)
- **Risk Assessment**: Built-in safety checks and risk management validation
- **Agent-Mode Workflow**: Let Claude autonomously build and iterate on your strategy

## Quick Start

### 1. Install Dependencies

```bash
pip install -e .
# or with backtesting support:
pip install -e ".[backtesting]"
```

### 2. Set up API Key

```bash
export ANTHROPIC_API_KEY="your-api-key-here"
```

### 3. Build Your First Strategy

**Option A: Interactive Mode**
```bash
python cli.py --interactive
```

**Option B: Command Line**
```bash
python cli.py -s "Buy when RSI < 30, sell when RSI > 70. Risk 1% per trade."
```

**Option C: From File**
```bash
python cli.py -f my_strategy.txt
```

### 4. Run the Agent-Mode Workflow

```bash
python tool_runner_version.py
```

## Project Structure

```
trading-strategy-builder/
├── strategy_builder.py        # Basic sequential workflow
├── tool_runner_version.py     # Advanced agent-based workflow
├── cli.py                     # Command-line interface
├── pyproject.toml            # Dependencies
└── README.md
```

## How It Works

### Strategy Building Workflow

The tool runs through these phases:

1. **Parse Strategy** → Extract components (entry, exit, position sizing, risk rules)
2. **Generate Code** → Create production-ready implementation
3. **Validate** → Check for risks, logical issues, look-ahead bias
4. **Backtest Plan** → Recommend metrics and backtesting approach
5. **Deployment** → Create live-trading checklist

### Example Strategy

```
Entry: Buy when Bitcoin price drops 5% in 1 hour from its 24-hour high,
       and RSI is below 30, and volume is above average.

Exit: Take profit at 3% gain or stop loss at 2% loss, whichever comes first.

Position Sizing: Risk 1% of account per trade.

Risk Rules: 
  - Never hold more than 3 concurrent positions
  - Daily loss limit of 5%
  - Account drawdown limit of 10%
```

## Claude Model Selection

The system uses **Claude Opus 5** (`claude-opus-5`) by default:

- **Strong reasoning**: Understands complex trading logic
- **Code generation**: High-quality, production-ready code
- **Adaptive thinking**: Automatically focuses effort where needed
- **Cost-effective**: Good balance of capability and price

For maximum capability on complex strategies, upgrade to **Claude Fable 5** (`claude-fable-5`).

## Output Examples

The tool generates:

### 1. Parsed Strategy Components
```json
{
  "entry_conditions": "RSI < 30 AND price above 200-day MA",
  "exit_conditions": "RSI > 70 OR stop loss hit",
  "position_sizing": "Risk 1% per trade",
  "risk_parameters": {
    "max_positions": 3,
    "daily_loss_limit": "5%",
    "stop_loss": "2%"
  }
}
```

### 2. Generated Strategy Code
```python
def calculate_signals(data):
    signals = []
    for i in range(len(data)):
        # Entry logic
        if data['rsi'][i] < 30 and data['price'][i] > data['ma200'][i]:
            signals.append({'type': 'BUY', 'confidence': 0.8})
        # Exit logic
        elif data['rsi'][i] > 70:
            signals.append({'type': 'SELL', 'reason': 'overbought'})
    return signals
```

### 3. Validation Report
```
Risk Assessment: YELLOW (Medium)

Issues Found:
- Entry signal lacks confirmation (consider adding volume filter)
- No slippage assumptions in position sizing
- Missing edge case handling for gaps

Recommendations:
- Add volume-weighted entry confirmation
- Adjust position size down 10% for slippage/commissions
- Add weekend/holiday gap handling
```

### 4. Backtest Recommendations
```
Key Metrics:
- Sharpe Ratio (minimum 1.0)
- Maximum Drawdown (should not exceed position sizing risk)
- Win Rate (breakeven ~40-45%)
- Profit Factor (target > 1.5)

Suggested Backtest Parameters:
- Date Range: 3+ years of data
- Starting Capital: $10,000
- Commission: $5/trade (realistic estimate)
- Slippage: 0.05% (crypto) to 0.1% (stocks)
```

## Advanced Usage

### Using the Agent-Based Workflow

The `tool_runner_version.py` uses Claude as an autonomous agent that:
- Parses the strategy
- Generates code
- Validates independently
- Asks clarifying questions if needed
- Iterates based on feedback

```python
from tool_runner_version import build_strategy_agent

strategy = "Buy on breakout above 20-day high..."
result = build_strategy_agent(strategy)
```

### Extending with Custom Tools

Add new tools to `tool_runner_version.py`:

```python
tools = [
    {
        "name": "check_strategy_correlation",
        "description": "Check if strategy correlates with other strategies",
        "input_schema": {...}
    },
    # ... more tools
]
```

### Integration with Real Backtesting

The generated code is compatible with:
- **Backtrader** (Python)
- **VectorBT** (Python, high performance)
- **TradingView** (Pine Script)
- **Alpaca** (API-based)

Example with Backtrader:
```python
import backtrader as bt
from strategy_code import YourStrategy  # Generated code

cerebro = bt.Cerebro()
cerebro.addstrategy(YourStrategy)
cerebro.run()
```

## Safety Guidelines

⚠️ **Before live trading:**

1. **Backtest thoroughly** (3+ years of data minimum)
2. **Test in paper trading** first
3. **Start with small positions** (1-5% of capital)
4. **Use stop losses** on every trade
5. **Monitor continuously** (don't set and forget)
6. **Have a kill switch** (way to stop all trading)

The tool includes built-in risk checks, but **final responsibility is yours**.

## Troubleshooting

### API Key Issues
```bash
# Verify API key is set
echo $ANTHROPIC_API_KEY

# Test connection
python -c "from anthropic import Anthropic; Anthropic().messages.create(model='claude-opus-5', max_tokens=1, messages=[{'role': 'user', 'content': 'hi'}])"
```

### Strategy Generation Issues
- Be specific in your strategy description
- Include entry AND exit conditions
- Specify position sizing method
- Mention risk management rules

### Output Not Matching Expectations
- Try the interactive mode (`--interactive`)
- Claude can ask clarifying questions
- Check the `strategy_result.json` for full output

## Cost Estimates

- Simple strategy: ~$0.10-0.30
- Complex strategy with iterations: ~$0.50-1.00
- Full workflow (parse + validate + backtest plan): ~$1.00-2.00

Claude Opus 5 pricing:
- Input: $5 per 1M tokens
- Output: $25 per 1M tokens

## Next Steps

1. **Customize the workflow** by editing tool definitions
2. **Add more languages** (TradingView, C#, etc.)
3. **Build a web UI** using the Python backend
4. **Add live execution** for paper trading
5. **Extend with market data integration**

## License

This is a hobby project. Use at your own risk.

## Resources

- [Anthropic Claude API Docs](https://docs.anthropic.com)
- [Backtrader Docs](https://www.backtrader.com)
- [VectorBT Docs](https://vectorbt.dev)
- [TradingView Pine Script](https://www.tradingview.com/pine-script-docs)

## Support

For issues with:
- **Claude API**: Check Anthropic docs or open an issue
- **Strategy logic**: Refine your description and try again
- **Code generation**: The prompt can be customized in the tool functions

---

**Happy strategy building!** 📈
