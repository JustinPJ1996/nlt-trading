#!/usr/bin/env python3
"""Command-line interface for the trading strategy builder."""

import argparse
import json
import sys
from pathlib import Path
from tool_runner_version import build_strategy_agent


def main():
    parser = argparse.ArgumentParser(
        description="LLM-powered trading strategy builder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Build from string
  python cli.py -s "Buy when RSI < 30 and price above 200-day MA"

  # Build from file
  python cli.py -f strategy.txt

  # Interactive mode
  python cli.py --interactive
        """
    )

    parser.add_argument(
        "-s", "--strategy",
        help="Strategy description as string"
    )
    parser.add_argument(
        "-f", "--file",
        help="Read strategy from file"
    )
    parser.add_argument(
        "-o", "--output",
        default="strategy_result.json",
        help="Output file for results (default: strategy_result.json)"
    )
    parser.add_argument(
        "-i", "--interactive",
        action="store_true",
        help="Interactive mode (ask questions)"
    )

    args = parser.parse_args()

    strategy_description = None

    if args.interactive:
        print("\n" + "="*60)
        print("TRADING STRATEGY BUILDER - Interactive Mode")
        print("="*60)
        print("\nDescribe your trading strategy in detail:")
        print("(Include entry conditions, exit conditions, risk management, etc.)")
        print("(Press Enter twice when done)\n")

        lines = []
        while True:
            try:
                line = input()
                if line:
                    lines.append(line)
                else:
                    if lines and not input("Continue? (press Enter to finish) "):
                        break
            except EOFError:
                break

        strategy_description = "\n".join(lines)

    elif args.file:
        try:
            with open(args.file, "r") as f:
                strategy_description = f.read()
            print(f"✓ Loaded strategy from {args.file}")
        except FileNotFoundError:
            print(f"✗ File not found: {args.file}")
            sys.exit(1)

    elif args.strategy:
        strategy_description = args.strategy

    else:
        parser.print_help()
        sys.exit(1)

    if not strategy_description or not strategy_description.strip():
        print("✗ No strategy provided")
        sys.exit(1)

    print(f"\n{'='*60}")
    print(f"Building strategy from description...")
    print(f"{'='*60}")

    result = build_strategy_agent(strategy_description)

    output_path = Path(args.output)
    with open(output_path, "w") as f:
        json.dump({
            "strategy_description": strategy_description,
            "timestamp": str(Path.cwd())
        }, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == "__main__":
    main()
