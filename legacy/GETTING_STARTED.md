# Getting Started with Trading Strategy Builder

## 5-Minute Quick Start

### Step 1: Install Python Dependencies

```bash
# Navigate to the project directory
cd trading-strategy-builder

# Install the package
pip install -r requirements.txt
```

### Step 2: Set Up Your API Key

Get your Anthropic API key from [https://console.anthropic.com](https://console.anthropic.com)

```bash
# Option 1: Export as environment variable (Linux/Mac)
export ANTHROPIC_API_KEY="sk-ant-..."

# Option 2: Export as environment variable (Windows)
set ANTHROPIC_API_KEY=sk-ant-...

# Option 3: Add to .env file (will be auto-loaded)
echo "ANTHROPIC_API_KEY=sk-ant-..." > .env
```

### Step 3: Run Your First Strategy

**Try the demo:**
```bash
python demo.py
```

**Or build your own:**
```bash
# Interactive mode (recommended for first time)
python cli.py --interactive

# Or from a strategy file
python cli.py -f example_strategy.txt
```

That's it! 🎉

## Understanding the Outputs

When you run the tool, it generates:

### 1. **Parsed Strategy** (strategy_result.json)
The strategy broken down into structured components:
```json
{
  "entry_conditions": "...",
  "exit_conditions": "...",
  "position_sizing": "...",
  "risk_parameters": {...}
}
```

### 2. **Generated Code**
Production-ready Python code implementing your strategy:
- Entry signal generation
- Position sizing calculations
- Risk management checks
- Error handling

### 3. **Validation Report**
Safety assessment including:
- Risk level (Green/Yellow/Red)
- Potential issues found
- Recommendations for improvement

### 4. **Backtest Plan**
Guidance for testing your strategy:
- Key performance metrics to track
- Recommended backtest parameters
- Realistic assumptions (slippage, commissions)
- Data quality checks

## Next Steps: Pick Your Path

### Path 1: Interactive Exploration
```bash
python cli.py --interactive
```
Great for:
- Learning the system
- Exploring different strategy ideas
- Getting feedback on your concepts

### Path 2: Bulk Strategy Building
```bash
# Create a strategy file
cat > my_strategy.txt << 'EOF'
My trading strategy description here...
Entry: ...
Exit: ...
EOF

# Build it
python cli.py -f my_strategy.txt
```

Great for:
- Saving strategy ideas
- Building multiple strategies
- Version control

### Path 3: Agent-Mode Development
```bash
python tool_runner_version.py
```

Edit the `example_strategy` variable and Claude will:
- Ask clarifying questions
- Iterate on the strategy
- Generate and validate autonomously

Great for:
- Complex strategies
- Iterative refinement
- Getting AI feedback on your ideas

## Common Workflows

### Workflow 1: Simple Strategy → Backtest
1. `python cli.py --interactive` - Describe your strategy
2. Claude generates code
3. Take the generated code and backtest manually with Backtrader or VectorBT
4. Share results with Claude for refinement (edit `tool_runner_version.py`)

### Workflow 2: Strategy Collection
```bash
# Keep a file of strategy ideas
echo "Buy on 20/50 MA crossover, sell on RSI > 70" >> strategies.txt

# Convert each to a full strategy
python cli.py -s "Buy on 20/50 MA crossover, sell on RSI > 70"
```

### Workflow 3: Team Collaboration
1. Team describes strategy in plain language
2. Claude generates standardized code
3. Team reviews the generated code together
4. Use version control on the results

## Customization

### Adjust the Model
Edit any Python file and change:
```python
model="claude-opus-5"  # Change to claude-fable-5 for more power
```

### Add More Tools
Edit `tool_runner_version.py` and add to the `tools` list:
```python
tools = [
    {
        "name": "my_custom_tool",
        "description": "What this tool does",
        "input_schema": {...}
    }
]
```

### Change Output Location
```bash
python cli.py -f strategy.txt -o my_custom_output.json
```

## Troubleshooting

### Error: "API key not found"
```bash
# Verify your API key is set
echo $ANTHROPIC_API_KEY

# Should show: sk-ant-...
# If blank, set it:
export ANTHROPIC_API_KEY="your-key-here"
```

### Error: "rate_limit_error"
Wait a moment and try again. If persistent, check your account at console.anthropic.com

### Error: "Cannot parse JSON response"
Claude sometimes returns text instead of structured JSON. Try:
1. Be more specific in your strategy description
2. Use the interactive mode for back-and-forth
3. Try the agent mode (`tool_runner_version.py`)

### Generated code doesn't look right
1. Check the parsed strategy (`strategy_result.json`)
2. Ask Claude to regenerate (edit and re-run)
3. Provide more specific entry/exit conditions

## Tips for Best Results

### Strategy Description Tips
✅ **Good:**
- "Buy when RSI drops below 30 AND price is above 200-day MA"
- "Position size is 1% of account, risk 1% per trade"
- "Max 3 concurrent positions, daily loss limit of 5%"

❌ **Avoid:**
- "Buy good stocks when they're cheap"
- "Sell when it's too risky" (too vague)
- Missing position sizing or risk rules

### For Complex Strategies
1. Start simple, then add filters
2. Use the interactive mode to ask questions
3. Ask Claude to explain each part
4. Let Claude suggest improvements

### For Backtesting
After getting the generated code:
1. Install Backtrader: `pip install backtrader`
2. Use the generated code as your strategy class
3. Add your market data
4. Run backtest

Example:
```python
import backtrader as bt
from generated_strategy import MyStrategy  # Your generated code

cerebro = bt.Cerebro()
cerebro.addstrategy(MyStrategy)
# Add data, set cash, etc.
cerebro.run()
```

## Resources

- **API Documentation**: [https://docs.anthropic.com](https://docs.anthropic.com)
- **Claude Models**: https://docs.anthropic.com/en/docs/about/models/overview
- **Backtrader Guide**: https://www.backtrader.com/
- **Trading Strategy Ideas**: TradingView community, r/algotrading

## Getting Help

1. Check error messages carefully
2. Review the validation report for insights
3. Try the interactive mode for Claude feedback
4. Look at `example_strategy.txt` for format inspiration

## What's Next?

After you get comfortable with building strategies:

1. **Connect Real Data**: Integrate with Alpaca, Interactive Brokers, etc.
2. **Live Paper Trading**: Test with real market data (but not real money)
3. **Backtesting Framework**: Set up Backtrader or VectorBT
4. **Web Interface**: Build a UI around the backend
5. **Multiple Strategies**: Create a portfolio of strategies

Good luck! 🚀
