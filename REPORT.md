# Report: does visible self-correction track a real reversal?

Scaled run: 25 seeds × 2 puzzles (`bat_and_ball`, `monty_hall`), 600-token
budget, `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, CPU-only. Classifier
and hedge detection as described in `README.md` (last-`\boxed{}`-wins
resolution, sliding-window hedge matching, `hit_token_limit` /
`stuck_relitigating` tracked independently of outcome).

## Headline results

**The original pilot's 7/10 "reversed" result does not survive the fixed
classifier.** Re-running the *same* 10 pilot traces (250-token budget)
through the corrected classifier: **0 REVERSED, 2 STAYED_WRONG, 8
UNRESOLVED**. Every run the old substring-match classifier called
"reversed" was actually the model passing through the correct number while
still mid-oscillation, never committing to it in a final boxed answer,
before the 250-token budget ran out. The pilot's entropy/hedge comparison
was consequently comparing "genuinely committed wrong" against "still
talking," not "genuinely committed wrong" against "genuinely committed
right."

**More budget helps, but even 600 tokens isn't enough most of the time.**

| Puzzle | REVERSED | STAYED_WRONG | UNRESOLVED | UNCLEAR | hit_token_limit | stuck_relitigating |
|---|---|---|---|---|---|---|
| `bat_and_ball` (n=25) | 5 (20%) | 3 (12%) | 17 (68%) | 0 | 19 (76%) | 17 (68%) |
| `monty_hall` (n=25) | 0 (0%) | 0 (0%) | 25 (100%) | 0 | 25 (100%) | 25 (100%) |

**`monty_hall` never converges — 0/25 runs produce a final boxed answer at
all**, even at 600 tokens (mean length: exactly 600, i.e. every single run
was truncated mid-thought). Inspecting the raw text confirms this isn't a
formatting quirk: the model genuinely keeps re-deriving door probabilities
without ever settling, even in runs where it says the correct value
("...the correct probability is 1/3 if you stay and 2/3 if you switch")
in prose — it doubts that statement again a sentence later
(seed 4). This is a real result, not a `\boxed{}` habit mismatch: the
Monty Hall trap induces much deeper, more persistent oscillation in this
model than the arithmetic trap does, to the point of non-convergence
within a budget that lets `bat_and_ball` resolve 32% of the time.

Because `monty_hall` never resolves, **it cannot be used to test the
core theater-vs-real question** (that requires comparing REVERSED against
STAYED_WRONG runs) — the only thing it currently demonstrates is that
"gets stuck re-litigating" generalizes beyond the pilot's puzzle, and does
so far more severely.

## Entropy / hedge signature (bat_and_ball only, small n — read with caution)

| | REVERSED (n=5) | STAYED_WRONG (n=3) |
|---|---|---|
| avg hedges | 4.00 | 1.00 |
| avg whole-trace entropy | 0.291 | 0.179 |

Direction matches the pilot's original claim (more hedging and higher
entropy on genuine reversals) and the magnitude is comparable (~1.6x
entropy here vs. the pilot's reported 1.3–1.7x) — but this is now n=5 vs
n=3 after applying a resolution-based filter to 25 raw runs, not n=7 vs
n=3 out of 10. Read as suggestive, not confirmatory.

For reference, UNRESOLVED runs (the majority in both puzzles) hedge *more*
than either resolved group (9.53 avg for `bat_and_ball`, 6.56 for
`monty_hall`) and run at higher entropy (0.398 / 0.498) — consistent with
"still actively uncertain," as expected for truncated mid-thought traces.

## What this means for the project

1. The pilot's headline number was an artifact of premature truncation
   combined with substring-anywhere matching. The corrected classifier
   removes that artifact, but the cost is a much smaller usable sample
   (8/25 resolved on `bat_and_ball` even at 600 tokens) — token budget is
   now the binding constraint on getting a clean REVERSED-vs-STAYED_WRONG
   comparison, more than seed count is.
2. Puzzle-to-puzzle variance is large and itself informative: this isn't
   just "does the model self-correct," it's "how easily does a given trap
   let the model *finish* correcting itself." That's a dimension the
   original single-puzzle pilot couldn't see.
3. Before adding a second model, the more valuable next step is probably
   giving `bat_and_ball` (and a couple more puzzles) enough token budget
   to see most runs resolve, so the entropy/hedge comparison has a
   real sample size instead of n=5 vs n=3. Whether that means a much
   higher token cap, or a prompt-level nudge to be more concise, is worth
   deciding deliberately rather than just cranking the cap further given
   the CPU-only cost (this scaled run already took ~2.75 hours).

## Raw data

Full per-seed traces (`H_t`, `speed_t`, hedge onsets, full text,
`hit_token_limit`) are under `traces/bat_and_ball/` and
`traces/monty_hall/`. The original 10-seed pilot is preserved under
`traces/trace_seed*.json` for reference.
