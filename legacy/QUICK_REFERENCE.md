# Quick Reference Guide

## Installation (30 seconds)

```bash
cd trading-strategy-builder
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-api-key-here"
```

## Usage Patterns

### Pattern 1: CLI Interactive
```bash
python cli.py --interactive
```
- Type your strategy
- Get code + validation + backtest plan
- Results saved to `strategy_result.json`

### Pattern 2: From File
```bash
python cli.py -f my_strategy.txt
```
- Prepare strategy file
- Run with CLI
- Get results

### Pattern 3: Programmatic
```python
from tool_runner_version import build_strategy_agent

result = build_strategy_agent("""
Buy when RSI < 30 and price above 200-day MA
Sell when RSI > 70
Risk 1% per trade
""")
```

### Pattern 4: Demo
```bash
python demo.py
```
- See 3 example strategies
- Learn different strategy types
- View sample outputs

## Strategy Description Template

When describing your strategy, include:

```
[Market/Timeframe]
- What asset? (stocks, crypto, forex)
- What timeframe? (1min, 5min, 1hour, 4hour, daily)

[Entry Conditions]
- When do you enter?
- Be specific: indicators, prices, volumes
- Multiple conditions? Use AND/OR

[Exit Conditions]
- When do you exit?
- Take profit level?
- Stop loss level?
- Time-based exit?

[Position Sizing]
- How big is each position?
- Fixed %, risk-based, other?

[Risk Rules]
- Max loss per trade?
- Max positions open?
- Daily loss limit?
- Account drawdown limit?
```

## Generated Outputs

After running, you get:

| File/Output | Contains |
|---|---|
| `strategy_result.json` | Structured strategy components + generated code + validation + recommendations |
| Console output | Summary of each phase |
| Generated Code | Ready-to-backtest Python code |

## Common Commands Cheat Sheet

```bash
# First time? Start here
python cli.py --interactive

# Try the examples
python demo.py

# Build from description
python cli.py -s "Buy on breakout, sell on RSI > 70"

# Build from file
python cli.py -f strategies.txt

# Use agent mode (can ask questions)
python tool_runner_version.py

# Save to custom location
python cli.py -f strategy.txt -o output.json
```

## Result Interpretation

### Parsed Strategy
```json
{
  "entry_conditions": "...",    # What triggers a buy
  "exit_conditions": "...",     # What triggers a sell
  "position_sizing": "...",     # How big each trade
  "risk_parameters": {...}      # Risk limits
}
```

### Risk Assessment
- **GREEN**: Low risk, ready for testing
- **YELLOW**: Medium risk, review recommendations
- **RED**: High risk, needs modification

### Key Performance Indicators (KPIs)
- **Sharpe Ratio**: Higher is better (>1.0 is good)
- **Max Drawdown**: Lower is better (<20% is typical)
- **Win Rate**: % of profitable trades (>45% is decent)
- **Profit Factor**: Gross profit / Gross loss (>1.5 is good)

## Customization

### Change Model
```python
# In any Python file:
model="claude-opus-5"  # Change this

# Options:
# "claude-opus-5" (recommended)
# "claude-fable-5" (more powerful)
# "claude-sonnet-5" (cheaper)
```

### Adjust Token Budget
```python
max_tokens=1000  # Increase for longer responses
```

### Add Your Own Tools
Edit `tool_runner_version.py` and add to `tools` list

## Troubleshooting

| Error | Solution |
|---|---|
| "API key not found" | `export ANTHROPIC_API_KEY="sk-..."` |
| "rate_limit_error" | Wait a minute, try again |
| "Invalid JSON" | Strategy description might be unclear, try interactive mode |
| "No results" | Check console for errors, ensure API key is valid |

## Next Steps After Building

### For Backtesting
```bash
pip install backtrader pandas numpy
# Use generated code as your strategy class
```

### For Paper Trading
1. Sign up for broker paper trading (Alpaca, Interactive Brokers)
2. Modify generated code for broker API
3. Test with real (but not risked) money

### For Live Trading
⚠️ **Only after:**
1. Backtesting 3+ years of data ✓
2. Paper trading results good ✓
3. Risk management rules implemented ✓
4. Kill-switch procedure ready ✓

## Model Capabilities

| Model | Speed | Reasoning | Cost | Best For |
|---|---|---|---|---|
| Haiku 4.5 | Fast | Basic | Cheap | Simple strategies |
| Sonnet 5 | Medium | Good | Moderate | Standard strategies |
| Opus 5 | Slower | Excellent | $$ | Complex strategies |
| Fable 5 | Slow | Best | $$$ | Very complex strategies |

**Recommendation:** Start with Claude Opus 5 (default)

## Example Workflows

### Workflow A: Solo Exploration
```
1. python cli.py --interactive
   └─ Describe strategy
   └─ Get feedback
   └─ Review results

2. Backtest manually with Backtrader
   └─ Install: pip install backtrader
   └─ Use generated code
   └─ Run backtest

3. Iterate based on results
```

### Workflow B: Team Review
```
1. python cli.py -f strategy.txt
   └─ Save strategy file in git

2. Share results with team
   └─ Review generated code
   └─ Review validation report

3. Iterate together
   └─ Update strategy.txt
   └─ Regenerate
   └─ Compare results
```

### Workflow C: Batch Building
```
1. Create strategies.txt with multiple strategies
2. Loop and build each
3. Compare results
4. Pick best ones for backtesting
```

## Monitoring Your Strategy

Key things to track:

### Before Live Trading
- ✓ Backtest Sharpe Ratio > 1.0
- ✓ Max Drawdown < 20%
- ✓ Win Rate > 40%
- ✓ Profit Factor > 1.5
- ✓ All risk rules implemented

### During Paper Trading
- Watch for slippage differences
- Monitor execution quality
- Check for order rejections
- Verify position sizing

### During Live Trading
- Daily P&L vs backtest
- Drawdown vs limit
- Win rate consistency
- Position concentration

---

**Ready to build?** Start with: `python cli.py --interactive`
