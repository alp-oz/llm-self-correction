# Doubt

**Does visible LLM self-correction reflect real internal computation, or is it stylistic theater?**

*Small pilot study: sample sizes are n≤25 per condition throughout, and a
couple of the results below are n=5 per group. See Limitations before
drawing conclusions.*

A small pilot study probing whether the moments where a reasoning model
appears to catch itself mid-answer and reconsider correspond to a real,
measurable change in what's happening inside the model, or whether that
language is just a habit the model has learned to produce, disconnected
from anything actually changing underneath.

## Motivation

Mathias Lindholm asked a simple question that kicked this whole project
off: when a reasoning model says something like "wait, that's not right,
let me reconsider," is that real, or just a verbal habit?

Reasoning models frequently produce visible self-correction: they state an
answer, then interrupt themselves with something like "wait, actually..."
or "hmm, let me reconsider," and sometimes arrive at a different final
answer. It's easy to assume this language tracks something real: a
genuine change in the model's internal computation. But it doesn't have
to. A model could just as easily emit these self-interruption phrases as a
learned stylistic pattern, text that sounds like reconsideration without
any actual shift in what it's computing.

This project tests that directly: plant a wrong answer into a model's own
turn, let it continue, and measure two signals at every generated token
alongside the text:

- **The model's uncertainty score at each step** (`H_t` in the code,
  entropy in the technical sense, see below): how unsure the model is
  about what word to produce next.
- **`speed_t`**: the distance between the model's internal state at one
  step and the next. A rough proxy for how much the model's internal
  "thinking" is actively moving, versus sitting still.

The hypothesis: if self-correction is real, these two signals should show
a distinct pattern around the moment of self-interruption, and critically,
that pattern should behave differently depending on whether the model
actually ends up changing its answer. If it's just a verbal habit, the
moment of self-interruption shouldn't look any different, internally, from
the rest of the generation.

## Method

The model under study is `deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B`, run
locally via `transformers`, CPU-only.

**Why a local model, and not Claude, GPT, or similar?** Those are closed,
proprietary systems: you can only talk to them through an API that sends
back text. There's no way to see what's happening inside them while they
generate that text, not because the internal computation is "encrypted,"
but simply because the companies running them don't expose that level of
detail through their API at all. It isn't offered as an option at any
price. The only way to inspect what a model is actually doing step by
step (how confident it is about each word, how its internal state changes
from one token to the next) is to use a model whose full internal weights
have been published and can be downloaded and run yourself.
DeepSeek-R1-Distill-Qwen-1.5B is one such model, which is why it's the one
used here; this whole project would be impossible to run against Claude
or GPT directly (see "Excluded," below, for what was tried against Claude
instead, and why it can't answer the same question).

**Planting a wrong answer.** For each puzzle, a wrong (but
plausible-sounding) worked solution is planted as the start of the
model's own assistant turn, and the model is left to continue generating
from inside that turn, as if it had already started saying the wrong
thing itself. This is the core manipulation: does it notice, reconsider,
and correct itself to the right answer, or does it perform the *language*
of reconsidering while still landing on the wrong answer?

**The uncertainty score, explained.** At every step, the model doesn't
just pick one next word: internally, it assigns a probability to every
possible next word (thousands of candidates), and then samples from that
distribution. The uncertainty score is a single number summarizing how
spread out that distribution is:

```
H = -sum(p_i * log(p_i))      (summed over every possible next word i)
```

where `p_i` is the probability the model assigns to word `i` being next.
In plain terms: the score is **low** when the model is essentially certain
what comes next (e.g. it's mid-way through a memorized phrase, or the next
token is a forced digit with only one sensible continuation), and **high**
when many different possible next words are all roughly equally plausible
to it. From here on, this document mostly just calls it "the model's
uncertainty score" rather than repeating the technical term.

**Self-correction cues.** This project also looks for the specific words
and phrases where the model appears to interrupt itself and reconsider
what it just said, things like "wait," "actually," "hmm, let me
reconsider," "that doesn't seem right." The code calls a match here a
"hedge," but this document mostly just calls them self-correction cues.
The central question is whether these words correspond to anything real
happening inside the model at that moment (a genuine spike and sustained
change in the uncertainty score), or whether they're just a verbal habit
layered on top of otherwise-unchanged internal behavior.

Detecting these cues runs over a sliding 20-token window of recently
decoded text, not a single token in isolation. BPE tokenization splits
multi-word phrases like these across several tokens (e.g. "Re-evaluating"
becomes `" Re"`, `"-e"`, `"valu"`, `"ating"`), so checking one token at a
time misses almost every multi-word phrase; a cue is only flagged at the
token where a match's span actually completes.

A run's outcome is decided by the **last `\boxed{...}` in the generated
text** (the model's most recent committed answer), not by whether the
correct number appears anywhere in passing. Early runs showed the model
frequently derives the right answer mid-thought, calls it wrong, and never
actually re-commits to it before running out of budget; matching "any
occurrence of the correct answer" against the full text conflates that
with a genuine, deliberate correction. Matching a boxed answer against the
known correct/wrong answer patterns uses numeric equality first (with
basic fraction support) and falls back to normalized substring containment
for non-numeric answers (`batch_correction_test.py`).

Each run also gets two boolean flags, tracked independently of the
categorical outcome:

- **`hit_token_limit`**: generation was cut off by the token budget rather
  than the model reaching a natural stopping point.
- **`stuck_relitigating`**: the model kept producing self-correction cues
  *after* it had already committed to a boxed answer, or was actively
  producing them with no commitment yet when it ran out of budget.

Keeping these separate from `outcome` (rather than folding them into more
outcome categories) keeps "ran out of tokens," "never doubted itself," and
"doubted itself but never resolved" distinguishable in the data, even when
two of those produce a superficially similar-looking truncated trace.

## Findings

Everything below is reported only where there's real statistical backing
(n≥10 per comparison, a clean 0/N or N/N result, or an explicitly-named
single-case illustration of a category). Smaller comparisons are labeled
with their actual n and flagged as suggestive, not confirmatory.

### 1. The original pilot's headline number does not survive the fixed classifier

The first pilot puzzle is the bat-and-ball problem: *"A bat and a ball
cost $1.10 together. The bat costs $1 more than the ball. How much does
the ball cost?"* Most people's first instinct is **$0.10**, but that's
wrong: the correct answer is **$0.05** (if the ball costs $0.05, the bat
costs $1.05, and together that's $1.10). This project plants the
intuitive-but-wrong $0.10 answer into the model's own turn and watches
whether it catches the error.

10 seeds, 250-token budget, this puzzle: originally read as **7/10
corrected themselves to the right answer, 3/10 stayed wrong**, using a
classifier that matched the correct answer anywhere in the generated text.
Re-running the *same* 10 traces through the corrected, last-boxed-answer-wins
classifier gives:

**0 corrected to the right answer, 2 stayed wrong, 8 never reached a final
answer at all.**

Every run the old classifier called "corrected" was actually the model
passing through the right number while still going back and forth, never
committing to it as a final answer before the 250-token budget ran out.
The original 7/10 figure was an artifact of premature truncation combined
with matching the right answer anywhere in the text, not a real rate of
genuine correction.

Spot-checked two of the eight "never reached a final answer" traces by
hand (`trace_seed1.json`, `trace_seed2.json`): both are cut off mid-sentence
at exactly the 250-token limit with no `\boxed{}`, and no near-miss
variant of the word "box," anywhere in the text. `trace_seed1.json` even
reaches the correct number in plain prose ("the ball is 5 cents, not 10")
one sentence before the cutoff, without ever formalizing it as a boxed
final answer, so the "never reached a final answer" label is accurate for
these, not a classifier gap.

### 2. Scaled run, same puzzle, larger budget: partial resolution

25 seeds, 600-token budget, fixed classifier, bat-and-ball puzzle:

| Outcome | Count | % |
|---|---|---|
| Corrected to the right answer | 5 | 20% |
| Stayed wrong | 3 | 12% |
| Never reached a final answer | 17 | 68% |

19/25 (76%) runs hit the token limit; 17/25 (68%) were flagged as still
actively going back and forth when they got cut off. Even at 600 tokens,
most runs don't finish; token budget, not seed count, is the binding
constraint on getting a clean comparison between genuine correction and
staying wrong on this puzzle.

### 3. Uncertainty-score / self-correction signature (n=5 vs n=3, read with caution)

Among the resolved runs from the scaled bat-and-ball batch:

| | Corrected (n=5) | Stayed wrong (n=3) |
|---|---|---|
| avg number of self-correction cues | 4.00 | 1.00 |
| avg whole-trace uncertainty score | 0.291 | 0.179 |

Runs that genuinely corrected themselves used self-correction language
about 4x more often and ran at roughly **1.6x (≈63%) higher** uncertainty
across the whole trace than runs that stayed wrong. This is n=5 vs n=3,
suggestive, not confirmatory, but the direction is consistent with the
idea that visible self-correction language correlates with a real increase
in the model's uncertainty.

### 4. The trend right after a self-correction cue (the sharpest result, n=5 per group)

Pooling every resolved bat-and-ball run across both the original pilot and
the scaled batch (5 corrected, 5 stayed wrong, 10 resolved runs total),
the uncertainty score was compared in three windows around each run's
self-correction cues: a per-trace baseline (all tokens away from any such
cue), the 3 tokens immediately before one starts, and the 3 tokens from
when it starts onward.

| | Corrected (n=5) | Stayed wrong (n=5) |
|---|---|---|
| baseline uncertainty score (away from any cue) | 0.302 | 0.145 |
| 3 tokens **before** a cue starts | 0.939 | 0.630 |
| 3 tokens **after** a cue starts | 1.124 | 0.467 |

Uncertainty rises above baseline right before a self-correction cue fires
**in both groups**, so a rise in uncertainty right before the cue, on its
own, doesn't distinguish a genuine correction from a cosmetic one. What
does distinguish them is what happens **after** the cue starts: runs that
go on to genuinely correct themselves keep climbing (0.939 to 1.124),
consistent with sustained, real uncertainty while the model reworks the
answer. Runs that stay wrong drop back down (0.630 to 0.467), below even
their own pre-cue level, consistent with a brief flicker of doubt that
gets suppressed rather than followed through: the model says the words,
then settles right back into confidently repeating the same wrong answer.

This is n=5 per group. Treat the direction as a real, well-grounded lead:
what happens to the uncertainty score *after* the self-correction language
starts, not just whether it rises at all, is what looks diagnostic, but
not as a settled result.

### 5. The Monty Hall puzzle: non-convergence, not inaccuracy

A second, different puzzle was also tested: the Monty Hall problem (three
doors, one prize; the host reveals a wrong door and offers a switch; the
correct answer is that switching wins 2/3 of the time, not 1/2 as most
people assume). 25 seeds, 600-token budget: **0/25 runs produced a final
answer at all.** Every single run was cut off at exactly the 600-token
limit, still going back and forth. This isn't a formatting artifact:
reading the raw text confirms the model genuinely keeps re-deriving the
door probabilities without ever settling, including cases where it states
the correct answer in plain words ("...the correct probability is 1/3 if
you stay and 2/3 if you switch") and then doubts that statement again a
sentence later.

This is a finding about **never settling on an answer**, not about
accuracy: the Monty Hall puzzle induces much deeper, more persistent
back-and-forth in this model than the bat-and-ball puzzle does, to the
point that a token budget sufficient to resolve bat-and-ball 32% of the
time resolves Monty Hall 0% of the time.

### 6. A three-way outcome split, not two

The project started out assuming two outcomes: genuinely corrects itself,
or stays wrong. The data shows a genuine third pattern: the model commits
to a final boxed answer, then keeps producing self-correction language and
re-deriving the problem *after* that commitment, without ever resolving
the doubt before running out of budget.

Concrete example: `traces/trace_seed5.json` (pilot run, bat-and-ball
puzzle). The continuation opens by restating the planted wrong answer
(`\boxed{10}` cents, at token 12), then immediately re-derives the problem
from scratch, correctly arrives at 5 cents, and calls that correct result
wrong ("However, this result seems incorrect because it leads to a
contradiction. Re-evaluating the equation..."). It then re-runs the same
derivation a second time and is cut off mid-equation at the 250-token
limit, never resolving. By final answer alone this run counts as "stayed
wrong" (its only boxed answer is the early one matching the planted
claim), but that label alone hides what actually happened: the model
worked out the right answer, decided it was wrong, and spiraled without
ever recovering. `outcome`, `hit_token_limit`, and `stuck_relitigating`
are tracked as independent fields specifically so cases like this don't
collapse into an ambiguous "never reached a final answer" bucket.

### Excluded: the Claude API side-experiment

`claude_puzzle_test.py` runs an analogous test against Claude via the
Anthropic API (the Monty Hall puzzle, plus a second, much less famous
probability puzzle, to compare a highly-memorized setup against one Claude
can't simply pattern-match to a known answer). Because the Claude API
doesn't expose any of the internal signals described above (no
uncertainty score, no internal state to measure), this can only ever
produce a count of how often Claude's stated answer flips, not the kind
of internal-signal comparison the rest of this project relies on. The
runs so far are also confounded by the token budget being too small for
the harder puzzle (most runs on that puzzle produced no visible answer
text at all, having spent their entire budget on internal reasoning
before writing anything down). No quantitative claim is made from this
data here; it's noted only as a promising direction for future work.

## Limitations

- **One model.** Everything here is DeepSeek-R1-Distill-Qwen-1.5B,
  CPU-only. No evidence yet on whether findings hold at larger scale or
  on other models.
- **One puzzle with real statistical power.** The bat-and-ball puzzle is
  the only one with enough resolved runs (8 total, pooling pilot + scaled)
  to say anything with even modest confidence. The Monty Hall puzzle and
  the Claude side-checks are exploratory/negative results: meaningful as
  findings about the model never settling on an answer, but not confirmed
  comparisons between genuine correction and cosmetic correction.
- **Small sample sizes throughout.** The uncertainty-score / self-correction
  comparisons in this report are n=3 to 5 per group. They're internally
  consistent and worth pursuing further, not something to generalize
  from.
- **CPU-only hardware constrained scale.** The full 25-seed × 2-puzzle
  scaled run took ~2.75 hours. Token budget, not seed count, is currently
  the main bottleneck on getting more resolved runs; raising it further
  is expensive on this hardware.

## Reproducing

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Single ad-hoc trace, printed live:
python trace_generation.py \
  --prompt "What is 17 * 24? Think step by step." \
  --save-json traces/example.json

# Batch run over a registered puzzle:
python batch_correction_test.py \
  --puzzle bat_and_ball \
  --seeds 25 \
  --max-new-tokens 600 \
  --out-dir traces/bat_and_ball

# Or supply a puzzle ad hoc:
python batch_correction_test.py \
  --prompt "..." --prefill "..." \
  --correct-answer "5 cents,b = 5" --wrong-answer "10 cents,b = 10" \
  --seeds 25

# Claude API side-experiment (separate mechanism, no internal signals
# available, see "Excluded," above):
pip install anthropic
export ANTHROPIC_API_KEY=...
python claude_puzzle_test.py --puzzle monty_hall --n 20
python claude_puzzle_test.py --puzzle urn --n 20
```

Traces are saved as JSON, one file per seed, containing the full per-token
record (uncertainty score, hidden-state movement, self-correction-cue
flags), the generated text, and run metadata (including whether the token
limit was hit).

## Repo structure

```
trace_generation.py      # token-by-token generation + tracing, prefill support
batch_correction_test.py # batch runner over seeds, classifier, summary stats
claude_puzzle_test.py    # Claude API side-experiment (see "Excluded" above)
puzzles.py                # puzzle registry (prompt, prefill, answer patterns)
REPORT.md                 # full write-up of the scaled-run findings
traces/
  trace_seed0.json ... trace_seed9.json   # original 10-seed pilot (250 tok)
  bat_and_ball/                            # scaled run, 25 seeds, 600 tok
  monty_hall/                              # scaled run, 25 seeds, 600 tok
requirements.txt
```
