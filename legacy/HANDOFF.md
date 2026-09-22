# Project Handoff Document

## Project Overview

**Trading Strategy Builder** — An LLM-powered system that converts plain English trading strategy descriptions into production-ready code, validates them, and provides backtesting recommendations.

**Location:** `/home/justin/trading-strategy-builder/`

**Status:** ✅ Complete (basic implementation, ready for extension)

**Built:** September 22, 2024

## What Was Built

### Core Features Implemented

1. ✅ **Strategy Parsing** — Extract strategy components from plain English
2. ✅ **Code Generation** — Generate Python code implementing strategies
3. ✅ **Strategy Validation** — Check for risks, logical issues, and safety
4. ✅ **Backtest Planning** — Recommend metrics and test parameters
5. ✅ **Deployment Guidance** — Create live-trading checklists

### Two Implementation Paths

1. **Sequential Workflow** (`strategy_builder.py`)
   - Simple, predictable
   - Each step runs in order
   - Good for straightforward strategies

2. **Agent Workflow** (`tool_runner_version.py`)
   - Uses Claude as an autonomous agent
   - Can ask clarifying questions
   - Iterates based on feedback
   - Better for complex strategies

## File Structure

```
trading-strategy-builder/
├── Core Implementation
│   ├── strategy_builder.py          # Sequential workflow
│   ├── tool_runner_version.py       # Agent-based workflow
│   ├── cli.py                       # Command-line interface
│   └── demo.py                      # Interactive demos
│
├── Configuration
│   ├── pyproject.toml              # Project metadata
│   ├── requirements.txt            # Dependencies
│   └── .gitignore                  # Git configuration
│
├── Documentation
│   ├── README.md                   # Full documentation
│   ├── GETTING_STARTED.md         # Quick start guide
│   ├── QUICK_REFERENCE.md         # Commands cheat sheet
│   ├── ARCHITECTURE.md            # Technical deep dive
│   └── HANDOFF.md                 # This file
│
└── Examples
    └── example_strategy.txt       # Sample strategy
```

## Key Technologies

- **Model**: Claude Opus 5 (`claude-opus-5`)
- **SDK**: Anthropic Python SDK (v1.0+)
- **Tool Pattern**: Tool use with agentic loop
- **Language**: Python 3.10+
- **Cost**: ~$0.10-0.50 per strategy

## How to Use

### Installation
```bash
cd /home/justin/trading-strategy-builder
pip install -r requirements.txt
export ANTHROPIC_API_KEY="your-key-here"
```

### Quick Start
```bash
# Interactive mode (best for learning)
python cli.py --interactive

# From strategy file
python cli.py -f example_strategy.txt

# Agent mode (smart iteration)
python tool_runner_version.py

# See demos
python demo.py
```

### Output
Generates `strategy_result.json` containing:
- Parsed strategy components
- Generated production code
- Validation report (risk assessment)
- Backtest recommendations
- Deployment checklist

## Architecture Summary

### Data Flow
```
User Input (Plain English)
    ↓
Parse Strategy (Claude)
    ↓
Generate Code (Claude)
    ↓
Validate Safety (Claude)
    ↓
Plan Backtest (Claude)
    ↓
Output (JSON + Code)
```

### Tool Definitions

All tools are defined in `tool_runner_version.py`:
- `parse_strategy` — Extract components
- `generate_strategy_code` — Generate code
- `validate_for_live_trading` — Safety check
- `backtest_recommendations` — Test recommendations

### Tool Execution
1. Claude decides which tools to call
2. Tools execute locally (API calls to Claude)
3. Results returned to Claude
4. Claude synthesizes final response

## Important Design Decisions

1. **Claude Opus 5 as default**
   - Good balance of capability and cost
   - Can upgrade to Fable 5 for complex strategies
   - Change in any Python file: `model="claude-opus-5"`

2. **Tool Runner Pattern**
   - Gives Claude agency to choose tools
   - Allows iteration and clarification
   - More intelligent than fixed sequence

3. **Local Execution Only**
   - No actual backtesting in this version
   - Generates code that's compatible with Backtrader/VectorBT
   - User provides data and runs backtest separately

4. **Safety First**
   - Validation step included by default
   - Risk assessment before deployment
   - Generated code includes error handling

## Current Limitations & TODOs

### Implemented ✅
- Strategy parsing and code generation
- Validation and safety checks
- Backtest planning
- CLI interface
- Agent-based workflow
- Multiple example strategies

### Not Yet Implemented ❌
- Integrated backtesting (just generates recommendations)
- Live trading execution
- Real-time market data integration
- Broker API connections
- Strategy optimization
- Performance tracking for live strategies
- Web UI
- Multi-strategy portfolio builder

### Future Enhancement Ideas
1. **Backtesting Integration**
   - Add Backtrader/VectorBT integration
   - Run backtests automatically
   - Generate performance charts

2. **Broker Connections**
   - Alpaca API integration
   - Interactive Brokers
   - Paper trading support
   - Live trading (with extreme caution)

3. **Optimization**
   - Parameter optimization
   - Hyperparameter tuning
   - Walk-forward analysis

4. **Interface**
   - Web UI (Flask/FastAPI)
   - Streamlit dashboard
   - Real-time monitoring

5. **Advanced Features**
   - Multiple strategies in portfolio
   - Risk correlation analysis
   - Strategy combination/ensemble
   - Machine learning parameter discovery

## How to Extend

### Add a New Tool
Edit `tool_runner_version.py`:
```python
tools = [
    # ... existing tools ...
    {
        "name": "my_new_tool",
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
    elif tool_name == "my_new_tool":
        return my_new_tool_function(tool_input)
```

### Support New Languages
In `generate_code()` function, add:
```python
if language == "pinescript":
    # Pine Script specific generation
```

### Connect Real Broker
1. Create new tool: `execute_live_trade`
2. Add broker authentication
3. Add position management
4. Add order execution
5. Add risk safeguards

## Testing & Quality

### How to Test
```bash
# Run demo (tests all workflows)
python demo.py

# Test single strategy
python cli.py -s "Your strategy here"

# Test with file
python cli.py -f example_strategy.txt
```

### What to Check
- Strategy components correctly parsed
- Generated code is syntactically correct
- Validation finds obvious issues
- Backtest recommendations are reasonable

### Cost per Test
- ~$0.10-0.20 per strategy
- Watch API key usage at console.anthropic.com

## Security Considerations

### API Key Safety
- Never commit `.env` or hard-coded keys
- Use environment variables
- `.gitignore` already configured

### Generated Code Safety
- Always review generated code before use
- Test in paper trading first
- Never auto-deploy to live trading

### Data Privacy
- Strategy descriptions sent to Claude API
- Data not stored between requests
- Review Anthropic privacy policy if sensitive

## Dependencies

### Core
- `anthropic>=1.0.0` — Claude API
- `pydantic>=2.0.0` — Data validation

### Optional
- `backtrader>=1.9.0` — For backtesting
- `pandas>=2.0.0` — Data processing
- `numpy>=1.24.0` — Numerical computing

### Install Options
```bash
# Core only
pip install -r requirements.txt

# With backtesting
pip install -e ".[backtesting]"
```

## Documentation Map

- **README.md** — Start here for overview
- **GETTING_STARTED.md** — Step-by-step setup
- **QUICK_REFERENCE.md** — Commands and examples
- **ARCHITECTURE.md** — Technical details
- **HANDOFF.md** — This file

## Known Issues & Workarounds

| Issue | Status | Workaround |
|-------|--------|-----------|
| Sometimes returns text instead of JSON | Minor | Use interactive mode for back-and-forth |
| Very complex strategies may exceed token limits | Rare | Break into smaller strategies |
| No actual backtesting built-in | Design choice | Generate code, use external framework |

## Next Development Steps

1. **Short Term (1-2 days)**
   - Add Backtrader integration
   - Auto-run backtest on generated code
   - Show performance charts

2. **Medium Term (1 week)**
   - Web UI with Streamlit
   - Strategy comparison tool
   - Historical result storage

3. **Long Term (ongoing)**
   - Broker integrations
   - Live trading (with safety limits)
   - Strategy optimization
   - Portfolio management

## Contact Points & References

### Claude Anthropic
- API Docs: https://docs.anthropic.com
- Models: Claude Opus 5, Claude Fable 5, Claude Sonnet 5
- Python SDK: `pip install anthropic`

### Trading Frameworks
- Backtrader: https://www.backtrader.com/
- VectorBT: https://vectorbt.dev/
- TradingView: https://www.tradingview.com/

### Brokers (Paper & Live)
- Alpaca: https://alpaca.markets/
- Interactive Brokers: https://www.ibkr.com/
- TradingView: paper trading

## Session Notes

### What Worked Well
- Sequential workflow is straightforward and predictable
- Agent workflow provides good user experience
- Claude's code generation is quite good
- Tool-based approach is extensible

### Lessons Learned
- Trading strategies need multiple validation layers
- Risk management is critical (not optional)
- Position sizing rules must be explicit
- Strategy descriptions need to be detailed

### Decisions Made
- Used Opus 5 (not Fable 5) for cost-effectiveness
- Tool Runner pattern for better user experience
- Agentic workflow as advanced option
- Code generation focused on entry/exit logic

## Quick Commands

```bash
# Setup
cd /home/justin/trading-strategy-builder
pip install -r requirements.txt
export ANTHROPIC_API_KEY="sk-ant-..."

# Use
python cli.py --interactive          # Interactive
python cli.py -f strategy.txt        # From file
python tool_runner_version.py        # Agent mode
python demo.py                       # See examples

# Check API
echo $ANTHROPIC_API_KEY              # Verify key
python -c "from anthropic import Anthropic; Anthropic().messages.create(model='claude-opus-5', max_tokens=1, messages=[{'role': 'user', 'content': 'hi'}])" # Test API
```

## Resume Prompt for Next Context

Use this prompt when resuming in a new Claude conversation:

```
I have a trading strategy builder project at /home/justin/trading-strategy-builder/
that uses Claude API with tool use to convert trading strategies from plain English
into production-ready Python code, validate them, and provide backtesting recommendations.

The project has two implementations:
1. Sequential workflow (strategy_builder.py) - simple, direct
2. Agent workflow (tool_runner_version.py) - smart, iterative

Current status: Basic implementation complete, ready for extensions like:
- Integrated backtesting (Backtrader/VectorBT)
- Web UI
- Broker API connections
- Strategy optimization
- Performance tracking

Project structure, installation, and quick start info is in:
- README.md (overview)
- GETTING_STARTED.md (setup)
- QUICK_REFERENCE.md (commands)
- ARCHITECTURE.md (technical)
- HANDOFF.md (context)

Next task: [specify what you want to work on]

Key context:
- Model: Claude Opus 5 (can change in Python files)
- Using tool use pattern with agentic loop
- All strategies output to strategy_result.json
- No integrated backtesting yet (generates code for external frameworks)
```

---

**Project Status:** ✅ Ready to use / ✅ Ready to extend

**Last Updated:** September 22, 2024, 16:03 UTC

**Maintainer Notes:** The project is well-structured and documented. Focus next development on backtesting integration and web UI.
