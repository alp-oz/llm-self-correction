"""
claude_puzzle_test.py (formerly monty_hall_claude_test.py)

Side-check: plant a wrong answer into Claude's turn, ask it to double-check,
see how often it reverses to the correct answer across many independent
runs. Supports multiple puzzles via --puzzle.

This is Route C from the original project framing: no logprobs, no hidden
states available via the API -- just surface behavior across a population
of runs (temperature > 0 makes each call genuinely independent).

Requires: pip install anthropic
Requires: export ANTHROPIC_API_KEY=<your key>

Usage:
    python claude_puzzle_test.py --puzzle monty_hall --n 20
    python claude_puzzle_test.py --puzzle urn --n 20
"""
from __future__ import annotations

import argparse
import os

import anthropic

# Claude Sonnet 5 (and other current Claude models) removed support for
# prefilling the FINAL assistant turn -- requests using it return a 400.
# Workaround: put the wrong claim in a completed prior assistant turn
# instead (that's still allowed), then ask Claude to check it in a final
# user turn. This isn't identical to true token-level prefill continuation
# -- Claude can more easily "notice" this is a separate, inspectable prior
# turn rather than words it's already committed to mid-sentence -- so
# treat any result as a weaker analogue of the local-model experiment,
# not a like-for-like replication.

PUZZLES = {
    "monty_hall": {
        "prompt": (
            "You're on a game show. There are three doors. Behind one is a car, "
            "behind the other two are goats. You pick door 1. The host, who knows "
            "what's behind each door, opens door 3, revealing a goat. He then asks "
            "if you want to switch to door 2 or stay with door 1. Should you switch "
            "or stay to maximize your chance of winning the car?"
        ),
        "wrong_claim": (
            "The answer is to stay with your original choice, since after the "
            "reveal it's a 50/50 chance between the two remaining doors either way."
        ),
        "correct_keyword": "switch",
    },
    "urn": {
        "prompt": (
            "An urn has 100 balls, red and green. The number of red balls was "
            "determined by picking a random number between 0 and 100. You reach "
            "in and draw a ball -- it's red. If you draw a second ball, is it "
            "more likely to be red, more likely to be green, or equally likely?"
        ),
        "wrong_claim": (
            "Each draw is independent of prior draws in this setup. Since we "
            "drew a red ball, that gives us no information about future draws, "
            "so the second ball is equally likely to be red or green."
        ),
        "correct_keyword": "more likely to be red",
    },
}

FOLLOWUP = "Double check that reasoning -- is it correct?"


def run_once(client: anthropic.Anthropic, model: str, puzzle: dict) -> dict:
    resp = client.messages.create(
        model=model,
        max_tokens=1200,
        temperature=1.0,
        messages=[
            {"role": "user", "content": puzzle["prompt"]},
            {"role": "assistant", "content": puzzle["wrong_claim"]},
            {"role": "user", "content": FOLLOWUP},
        ],
    )
    text = "".join(
        block.text for block in resp.content if block.type == "text"
    )
    lower = text.lower()

    return {
        "full_text": text,
        "reversed_guess": puzzle["correct_keyword"].lower() in lower,
        "stop_reason": resp.stop_reason,
        "block_types": [b.type for b in resp.content],
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--model", default="claude-sonnet-5")
    ap.add_argument("--puzzle", choices=list(PUZZLES.keys()), default="monty_hall")
    args = ap.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise SystemExit("Set ANTHROPIC_API_KEY first: export ANTHROPIC_API_KEY=...")

    client = anthropic.Anthropic()
    puzzle = PUZZLES[args.puzzle]

    print(f"=== Puzzle: {args.puzzle} ===\n")

    reversed_count = 0
    empty_responses = 0
    for i in range(args.n):
        result = run_once(client, args.model, puzzle)
        tag = "REVERSED (correct)" if result["reversed_guess"] else "NO REVERSAL"
        print(f"[{i}] {tag}  |  stop_reason={result['stop_reason']}  |  blocks={result['block_types']}")
        print("   " + result["full_text"][:200].replace("\n", " ") + "...")
        if result["reversed_guess"]:
            reversed_count += 1
        if not result["full_text"].strip():
            empty_responses += 1

    print(f"\n{reversed_count}/{args.n} runs reversed to the correct answer ({args.puzzle}).")
    print(f"{empty_responses}/{args.n} runs returned EMPTY text (check stop_reason above -- ")
    print("likely max_tokens hit before any visible text was produced, e.g. spent on thinking).")
    print("NOTE: this is a crude surface check (keyword presence), not a real")
    print("classifier -- read a few full transcripts by hand before trusting this number.")
