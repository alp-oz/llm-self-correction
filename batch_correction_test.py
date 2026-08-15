"""
batch_correction_test.py

Runs the same (prompt, prefill) pair across many seeds, loading the model
ONCE and reusing it, saving every trace to disk, and printing a summary:
  - how often the model reverses the planted wrong answer ("caught" it)
  - hedge-word counts per run
  - entropy stats per run, split into "before first hedge" vs "from first
    hedge onward" (a first-pass approximation of "initial commitment" vs
    "reconsideration" -- crude, but a start)

Usage:
    python batch_correction_test.py \
      --prompt "A bat and a ball cost one dollar and ten cents in total. The bat costs one dollar more than the ball. How much does the ball cost?" \
      --prefill "**Solution:** Let the ball cost \\( b \\) cents. Then \\( 2b + 100 = 110 \\), so \\( 2b = 10 \\), so \\( b = 10 \\). The ball costs 10 cents." \
      --seeds 10 \
      --correct-answer "5 cents" \
      --wrong-answer "10 cents"
"""
from __future__ import annotations

import argparse
import os
import re
import statistics as stats

from trace_generation import (
    DEFAULT_MODEL, load_model, generate_and_trace, save_trace_json, HEDGE_RE,
)

BOXED_RE = re.compile(r"\\boxed\{([^}]*)\}")


def _extract_number(s: str) -> float | None:
    """Pull a single numeric value out of a phrase, handling simple fractions
    like '2/3' as well as plain numbers like '5' or '0.05'."""
    s = s.replace(" ", "")
    m = re.search(r"-?\d+\.?\d*/\d+\.?\d*", s)
    if m:
        num, den = m.group().split("/")
        try:
            return float(num) / float(den)
        except ZeroDivisionError:
            return None
    m = re.search(r"-?\d+\.?\d*", s)
    return float(m.group()) if m else None


def _normalize(s: str) -> str:
    return re.sub(r"[^a-z0-9.]", "", s.lower())


def _matches_any(candidate: str, patterns: list[str]) -> bool:
    """True if `candidate` (typically a \\boxed{} payload) matches any of the
    given equivalent-phrasing patterns, by numeric equality first and
    normalized substring containment as a fallback (for non-numeric answers
    like 'switch')."""
    cand_num = _extract_number(candidate)
    cand_norm = _normalize(candidate)
    for p in patterns:
        p_num = _extract_number(p)
        if cand_num is not None and p_num is not None and abs(cand_num - p_num) < 1e-9:
            return True
        p_norm = _normalize(p)
        if cand_norm and p_norm and (cand_norm in p_norm or p_norm in cand_norm):
            return True
    return False


def _find_last_boxed(records) -> tuple[str | None, int | None]:
    """Returns (payload, step_t) of the last \\boxed{...} in the run, or
    (None, None) if the model never committed to a boxed answer."""
    text = ""
    cum_len_at_step = []  # cumulative char length after appending step i
    for r in records:
        text += r.token_str
        cum_len_at_step.append(len(text))

    matches = list(BOXED_RE.finditer(text))
    if not matches:
        return None, None

    last = matches[-1]
    commit_t = None
    for i, cum_len in enumerate(cum_len_at_step):
        if cum_len >= last.end():
            commit_t = records[i].t
            break
    return last.group(1).strip(), commit_t


def analyze_run(
    records, correct_patterns: list[str], wrong_patterns: list[str],
    hit_token_limit: bool = False,
) -> dict:
    full_text = "".join(r.token_str for r in records)

    boxed_payload, commit_t = _find_last_boxed(records)
    if boxed_payload is None:
        outcome = "UNRESOLVED"
    elif _matches_any(boxed_payload, correct_patterns):
        outcome = "REVERSED"
    elif _matches_any(boxed_payload, wrong_patterns):
        outcome = "STAYED_WRONG"
    else:
        outcome = "UNCLEAR"

    hedge_positions = [r.t for r in records if r.is_hedge_onset]

    # "Stuck re-litigating": the model kept doubting itself after it had
    # already committed to a boxed answer (renewed doubt post-commitment),
    # or it was actively hedging but ran out of budget before ever
    # committing to anything. Tracked separately from `outcome` and from
    # `hit_token_limit` so the three causes of an "unresolved-looking" run
    # (ran out of tokens / never reconsidered / reconsidered but never
    # re-committed) don't get collapsed into one bucket.
    if commit_t is not None:
        stuck_relitigating = any(t > commit_t for t in hedge_positions)
    else:
        stuck_relitigating = hit_token_limit and len(hedge_positions) > 0

    entropies = [r.entropy for r in records]
    speeds = [r.speed for r in records if r.speed > 0]

    if hedge_positions:
        first_hedge = hedge_positions[0]
        before = [r.entropy for r in records if r.t < first_hedge]
        after = [r.entropy for r in records if r.t >= first_hedge]
    else:
        before, after = entropies, []

    return {
        "outcome": outcome,
        "reversed_to_correct": outcome == "REVERSED",
        "stayed_wrong": outcome == "STAYED_WRONG",
        "hit_token_limit": hit_token_limit,
        "stuck_relitigating": stuck_relitigating,
        "boxed_answer": boxed_payload,
        "boxed_commit_t": commit_t,
        "n_hedges": len(hedge_positions),
        "hedge_positions": hedge_positions,
        "mean_entropy_overall": round(stats.mean(entropies), 3) if entropies else None,
        "mean_entropy_before_first_hedge": round(stats.mean(before), 3) if before else None,
        "mean_entropy_from_first_hedge": round(stats.mean(after), 3) if after else None,
        "max_entropy": round(max(entropies), 3) if entropies else None,
        "mean_speed": round(stats.mean(speeds), 1) if speeds else None,
        "full_text": full_text,
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--puzzle", default=None,
                     help="Name of a puzzle defined in puzzles.py (e.g. 'bat_and_ball', "
                          "'monty_hall'). Supplies --prompt/--prefill/--correct-answer/"
                          "--wrong-answer; any of those flags passed explicitly override "
                          "the puzzle's default.")
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--prefill", default=None)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=600)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seeds", type=int, default=10,
                     help="Number of seeds to run: 0, 1, ..., seeds-1.")
    ap.add_argument("--correct-answer", default=None,
                     help="Comma-separated equivalent phrasings of the right answer, "
                          "e.g. '5 cents,0.05 dollars,$0.05,b = 5'")
    ap.add_argument("--wrong-answer", default=None,
                     help="Comma-separated equivalent phrasings of the planted wrong answer.")
    ap.add_argument("--out-dir", default="traces")
    args = ap.parse_args()

    if args.puzzle:
        from puzzles import PUZZLES
        p = PUZZLES[args.puzzle]
        args.prompt = args.prompt or p["prompt"]
        args.prefill = args.prefill if args.prefill is not None else p["prefill"]
        args.correct_answer = args.correct_answer or p["correct_answer"]
        args.wrong_answer = args.wrong_answer or p["wrong_answer"]

    if not (args.prompt and args.correct_answer and args.wrong_answer):
        ap.error("--prompt/--correct-answer/--wrong-answer are required "
                  "(directly, or via --puzzle)")

    os.makedirs(args.out_dir, exist_ok=True)
    correct_patterns = [p.strip() for p in args.correct_answer.split(",")]
    wrong_patterns = [p.strip() for p in args.wrong_answer.split(",")]

    print(f"Loading model {args.model} once for all {args.seeds} runs...")
    tok, model = load_model(args.model)

    results = []
    for seed in range(args.seeds):
        print(f"\n=== seed {seed} ===")
        recs, hit_token_limit = generate_and_trace(
            tok, model, args.prompt, args.max_new_tokens, args.temperature,
            seed, args.prefill, verbose=False,
        )
        analysis = analyze_run(recs, correct_patterns, wrong_patterns, hit_token_limit)
        analysis["seed"] = seed
        results.append(analysis)

        json_path = os.path.join(args.out_dir, f"trace_seed{seed}.json")
        save_trace_json(json_path, recs, {
            "prompt": args.prompt, "prefill": args.prefill, "model": args.model,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "seed": seed,
        }, hit_token_limit)

        limit_flag = " [hit token limit]" if hit_token_limit else ""
        stuck_flag = " [stuck re-litigating]" if analysis["stuck_relitigating"] else ""
        print(f"  outcome: {analysis['outcome']}{limit_flag}{stuck_flag}  |  "
              f"boxed: {analysis['boxed_answer']!r}  |  "
              f"hedges: {analysis['n_hedges']} at {analysis['hedge_positions']}")
        print(f"  whole-trace mean entropy: {analysis['mean_entropy_overall']}")

    n = len(results)
    n_reversed = sum(1 for r in results if r["outcome"] == "REVERSED")
    n_wrong = sum(1 for r in results if r["outcome"] == "STAYED_WRONG")
    n_unresolved = sum(1 for r in results if r["outcome"] == "UNRESOLVED")
    n_unclear = sum(1 for r in results if r["outcome"] == "UNCLEAR")
    n_hit_limit = sum(1 for r in results if r["hit_token_limit"])
    n_stuck = sum(1 for r in results if r["stuck_relitigating"])

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Total runs:          {n}")
    print(f"Reversed (real):     {n_reversed}  ({100*n_reversed/n:.0f}%)")
    print(f"Stayed wrong:        {n_wrong}  ({100*n_wrong/n:.0f}%)")
    print(f"Unresolved (no box): {n_unresolved}  ({100*n_unresolved/n:.0f}%)")
    print(f"Unclear (boxed but"
          f" matched neither): {n_unclear}  ({100*n_unclear/n:.0f}%)")
    print(f"\nHit token limit:     {n_hit_limit}  ({100*n_hit_limit/n:.0f}%)")
    print(f"Stuck re-litigating: {n_stuck}  ({100*n_stuck/n:.0f}%)")

    reversed_hedges = [r["n_hedges"] for r in results if r["outcome"] == "REVERSED"]
    wrong_hedges = [r["n_hedges"] for r in results if r["outcome"] == "STAYED_WRONG"]
    if reversed_hedges:
        print(f"\nAvg hedges when it self-corrected: {stats.mean(reversed_hedges):.1f}")
    if wrong_hedges:
        print(f"Avg hedges when it stayed wrong:   {stats.mean(wrong_hedges):.1f}")

    # Group-level entropy comparison (avoids the small-sample "before first
    # hedge" split, which is dominated by the single high-entropy
    # sentence-initial token when a hedge fires early).
    reversed_h = [r["mean_entropy_overall"] for r in results
                  if r["outcome"] == "REVERSED" and r["mean_entropy_overall"] is not None]
    wrong_h = [r["mean_entropy_overall"] for r in results
               if r["outcome"] == "STAYED_WRONG" and r["mean_entropy_overall"] is not None]
    if reversed_h:
        print(f"\nAvg whole-trace entropy when it self-corrected: {stats.mean(reversed_h):.3f}")
    if wrong_h:
        print(f"Avg whole-trace entropy when it stayed wrong:   {stats.mean(wrong_h):.3f}")

    print(f"\nAll traces saved under ./{args.out_dir}/")
