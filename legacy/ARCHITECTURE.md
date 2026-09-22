# Architecture Overview

## Project Structure

```
trading-strategy-builder/
├── strategy_builder.py          # Basic workflow (sequential)
├── tool_runner_version.py       # Advanced workflow (agent-based)
├── cli.py                       # Command-line interface
├── demo.py                      # Interactive demos
│
├── pyproject.toml              # Project metadata
├── requirements.txt            # Python dependencies
│
├── README.md                   # Full documentation
├── GETTING_STARTED.md         # Quick start guide
├── ARCHITECTURE.md            # This file
│
├── example_strategy.txt       # Example strategy file
├── .gitignore                 # Git ignore patterns
```

## How It Works

### High-Level Flow

```
User Input (Plain English Strategy Description)
    ↓
    ├─→ Parse Strategy Components
    │   ├─→ Entry conditions
    │   ├─→ Exit conditions
    │   ├─→ Position sizing
    │   └─→ Risk parameters
    ↓
    ├─→ Generate Production Code
    │   ├─→ Entry logic
    │   ├─→ Signal generation
    │   └─→ Error handling
    ↓
    ├─→ Validate Strategy
    │   ├─→ Logical consistency
    │   ├─→ Risk management check
    │   └─→ Implementation verification
    ↓
    ├─→ Backtest Recommendations
    │   ├─→ Key metrics
    │   ├─→ Test parameters
    │   └─→ Data quality checks
    ↓
Results (Code + Validation + Recommendations)
```

## Two Implementation Approaches

### 1. Sequential Workflow (`strategy_builder.py`)

**Best for:** Direct, predictable results

```python
def build_strategy_workflow(strategy_description):
    1. parse_strategy_tool()         # Extract components
    2. generate_code_tool()          # Generate code
    3. backtest_analysis_tool()      # Recommend backtests
    4. validate_strategy_tool()      # Validate safety
    5. deployment_checklist_tool()   # Create checklist
    return results
```

**Pros:**
- Simple and predictable
- Each step completes in order
- Easy to understand and debug

**Cons:**
- No iteration or feedback loop
- Claude can't ask clarifying questions
- Less intelligent about complex strategies

**Use when:** You want a straightforward transformation from description → code

### 2. Agent Workflow (`tool_runner_version.py`)

**Best for:** Complex strategies, iteration, feedback

```
User Message
    ↓
Claude reads the strategy
    ↓
Claude decides which tools to call
    ├─→ parse_strategy
    ├─→ generate_strategy_code
    ├─→ validate_for_live_trading
    └─→ backtest_recommendations
    ↓
Claude gets results
    ↓
Claude can ask clarifying questions
    ↓
Claude iterates if needed
    ↓
Final Response with all results
```

**Pros:**
- Claude can ask clarifying questions
- Intelligent tool selection
- Can iterate and improve
- Better for complex strategies
- More natural conversation

**Cons:**
- Takes slightly longer
- Less predictable order
- May require user interaction

**Use when:** Strategy is complex, you want feedback, or Claude should explore options

## API Integration

### Tool Definitions

Each tool is defined with:
```python
{
    "name": "tool_name",
    "description": "What this tool does",
    "input_schema": {
        "type": "object",
        "properties": {...},
        "required": [...]
    }
}
```

### Tool Execution

```python
def execute_tool(tool_name, tool_input):
    # Claude calls tool via API
    # Tool executes locally
    # Results returned to Claude
    # Claude uses results in next step
```

### Models Used

- **Primary:** Claude Opus 5 (`claude-opus-5`)
  - Strong reasoning
  - Good code generation
  - Cost-effective

- **Alternative:** Claude Fable 5 (`claude-fable-5`)
  - Most capable
  - Better on very complex strategies
  - Higher cost

## Code Generation Strategy

### Generated Code Structure

The generated code typically includes:

```python
class TradingStrategy:
    def __init__(self, ...):
        # Initialize parameters
        
    def calculate_signals(self, data):
        # Entry logic
        # Exit logic
        # Return buy/sell signals
        
    def validate_data(self, data):
        # Data quality checks
        
    def size_position(self, account, signal):
        # Calculate position size
        
    def check_risk_limits(self, position):
        # Risk management checks
```

### Key Features

1. **Entry Signal Generation**
   - Parses natural language conditions
   - Converts to mathematical operations
   - Adds confidence scoring

2. **Exit Logic**
   - Take profit levels
   - Stop loss calculations
   - Time-based exits

3. **Position Sizing**
   - Risk-based sizing
   - Account percentage sizing
   - Max position caps

4. **Risk Management**
   - Stop loss enforcement
   - Account drawdown limits
   - Position overlap checks

5. **Error Handling**
   - Data validation
   - Missing data handling
   - Edge case management

## Validation Process

### What Gets Validated

1. **Logical Consistency**
   - Do entry/exit rules make sense together?
   - Are there contradictions?
   - Can both conditions occur simultaneously?

2. **Risk Management**
   - Is risk per trade reasonable?
   - Are there stop losses?
   - Are account limits set?

3. **Implementation Correctness**
   - Does code match description?
   - Are calculations accurate?
   - Is error handling sufficient?

4. **Potential Issues**
   - Look-ahead bias (using future data)
   - Overfitting (too many specific conditions)
   - Data leakage
   - Unrealistic assumptions

5. **Live Trading Readiness**
   - Can be deployed as-is?
   - What additional setup needed?
   - Risk mitigation needed?

## Backtest Planning

### Metrics Recommended

**Performance Metrics:**
- Sharpe Ratio (risk-adjusted return)
- Sortino Ratio (downside risk)
- Maximum Drawdown (peak-to-trough)
- Win Rate (% profitable trades)
- Profit Factor (gross profit / gross loss)
- ROI (return on investment)

**Risk Metrics:**
- Value at Risk (VaR)
- Conditional Value at Risk (CVaR)
- Calmar Ratio (return / max drawdown)
- Recovery Factor (total profit / max drawdown)

**Execution Metrics:**
- Slippage analysis
- Commission impact
- Liquidity requirements
- Execution quality

### Parameters Suggested

**Time Period:**
- Minimum 3 years of historical data
- Walk-forward analysis recommended
- Out-of-sample testing

**Market Conditions:**
- Bull markets
- Bear markets
- Sideways/choppy markets
- Regime changes

**Realistic Assumptions:**
- Bid-ask spreads
- Slippage (0.05% to 0.2% typically)
- Commission (per trade, volume-based)
- Liquidity constraints
- Order rejection rates

## Extensibility

### Adding New Tools

To add a new tool in `tool_runner_version.py`:

```python
tools = [
    # ... existing tools ...
    {
        "name": "your_tool",
        "description": "What it does",
        "input_schema": {
            "type": "object",
            "properties": {...},
            "required": [...]
        }
    }
]

def execute_tool(tool_name, tool_input):
    # ... existing tools ...
    elif tool_name == "your_tool":
        return your_tool_function(tool_input)
```

### Adding New Languages

To add support for other languages (Pine Script, C#, etc.):

1. Update `generate_code` function to handle language parameter
2. Add language-specific code templates
3. Add to prompt instructions for code generation

```python
def generate_code(parsed, language):
    if language == "pinescript":
        # Pine Script specific code generation
    elif language == "csharp":
        # C# specific code generation
    # ... etc
```

### Connecting to Real Brokers

To enable live trading:

1. Add broker-specific tools:
   ```python
   def execute_live_trade(signal, broker, credentials):
       # Use broker API
   ```

2. Add position management tools
3. Add real-time data integration
4. Add order execution tools

## Performance Considerations

### API Costs

**Typical workflow:**
- Parse: ~300 tokens
- Generate code: ~1,500 tokens
- Validate: ~800 tokens
- Backtest plan: ~900 tokens
- **Total: ~3,500 tokens (~$0.10-0.20)**

### Optimization Tips

1. Use caching for repeated strategies
2. Batch similar strategies together
3. Use streaming for large outputs
4. Consider Claude Sonnet for simpler strategies

### Rate Limiting

The Anthropic API has rate limits:
- Handle 429 responses gracefully
- Implement exponential backoff
- Queue requests if needed

## Security Considerations

### API Key Security

```python
# Never hardcode API keys
import os
api_key = os.getenv("ANTHROPIC_API_KEY")

# Safe: loads from environment
client = Anthropic(api_key=api_key)
```

### Generated Code Safety

1. Code is generated by Claude (trusted)
2. Always review generated code before use
3. Test in paper trading first
4. Never auto-deploy to live trading

### Data Handling

- User strategies are sent to Claude API
- Claude doesn't store data between requests
- Review Anthropic privacy policy if sensitive

## Deployment Paths

### Development
```bash
python cli.py --interactive
python demo.py
```

### Production
```bash
# As a module
from tool_runner_version import build_strategy_agent

# Or via CLI
python cli.py -f strategy.txt
```

### Web/API
```python
from flask import Flask
app = Flask(__name__)

@app.route('/build_strategy', methods=['POST'])
def build_strategy():
    strategy = request.json['description']
    result = build_strategy_agent(strategy)
    return result
```

### Scheduled
```bash
# Run nightly strategy builder
0 20 * * * /usr/bin/python3 /path/to/cli.py -f strategies.txt
```

## Future Enhancements

1. **Real-time Backtesting**: Integrated backtrader/vectorbt
2. **Live Trading**: Direct broker API connections
3. **Strategy Optimization**: Auto-tune parameters
4. **Performance Tracking**: Monitor live strategies
5. **Strategy Repository**: Share and discover strategies
6. **Web UI**: Browser-based interface
7. **Multi-agent**: Strategies that create sub-strategies
8. **Market Data Integration**: Real-time data feeds

---

**Last Updated:** September 2024
