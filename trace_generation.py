"""
trace_generation.py

Generate text from a local model token-by-token, and at every step record:
  - e_t      : final-layer hidden state after token t   (the "embedding")
  - speed_t  : ||e_t - e_{t-1}||                         (trajectory speed)
  - H_t      : entropy of the softmax over the FULL vocab for predicting
               token t+1, given tokens x_1..x_t          (next-token entropy)

Then flags positions where hedge language ("wait", "actually", ...) starts,
so you can eyeball whether H_t / speed_t spike there or stay flat.

Run on your own machine (needs internet + ~4-8GB RAM for the 1.5B model):

    pip install torch transformers --break-system-packages
    python trace_generation.py --prompt "What is 17 * 24? Think step by step."

First run downloads the model (a few GB) and will be slow to load on CPU.
"""
from __future__ import annotations

import argparse
import re
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"

HEDGE_PATTERNS = [
    r"\bwait\b", r"\bactually\b", r"\bhmm+\b", r"\bhold on\b",
    r"\blet me reconsider\b", r"\bon second thought\b",
    r"\bno[,]? wait\b", r"\bi (?:think|must) (?:have )?made (?:a|an) (?:mistake|error)\b",
    r"\blet me re-?check\b", r"\bcorrection\b",
    r"\bre-?evaluat\w*\b", r"\bre-?examin\w*\b", r"\bre-?deriv\w*\b",
    r"\bseems? (?:incorrect|wrong|off|too \w+)\b",
    r"\b(?:that|this) (?:can'?t|cannot|doesn'?t) be right\b",
    r"\bdoesn'?t (?:seem|look) right\b",
    r"\bleads? to a contradiction\b",
    r"\blet me (?:think again|redo|verify|double-?check)\b",
    r"\bdouble-?check(?:ing)?\b",
    r"\bsomething(?:'s| is) (?:wrong|off)\b",
    r"\bmessed up\b",
    r"\bwhy did i\b",
    r"\bis that correct\??",
    r"\bnot correct\b",
    r"\blet me try again\b",
]
HEDGE_RE = re.compile("|".join(HEDGE_PATTERNS), re.IGNORECASE)


@dataclass
class StepRecord:
    t: int
    token_id: int
    token_str: str
    entropy: float          # H_t: entropy over p(x_{t+1} | x_1..x_t), nats
    speed: float             # ||e_t - e_{t-1}||, None-> 0.0 for t=0
    is_hedge_onset: bool     # True if this token starts a hedge phrase


def entropy_from_logits(logits: torch.Tensor) -> float:
    """logits: (vocab_size,) raw scores for one step. Returns entropy in nats."""
    logits = logits.float()  # upcast for numerically stable softmax
    log_p = F.log_softmax(logits, dim=-1)
    p = log_p.exp()
    h = -(p * log_p).sum().item()
    return h


def load_model(model_name: str = DEFAULT_MODEL):
    """Load tokenizer + model once, so callers can reuse across many runs."""
    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, dtype=torch.float32, low_cpu_mem_usage=True
    )
    model.eval()
    return tok, model


def generate_and_trace(
    tok,
    model,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 0.7,
    seed: int = 0,
    prefill: str | None = None,
    verbose: bool = True,
) -> tuple[list[StepRecord], bool]:
    torch.manual_seed(seed)

    messages = [{"role": "user", "content": prompt}]
    if prefill:
        # Force the assistant's turn to start with our own (wrong) text,
        # then let the model continue generating from there.
        messages.append({"role": "assistant", "content": prefill})
        input_ids = tok.apply_chat_template(
            messages, continue_final_message=True, return_tensors="pt"
        )
    else:
        input_ids = tok.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        )
    if not torch.is_tensor(input_ids):
        # Newer transformers versions return a BatchEncoding-like object here.
        input_ids = input_ids["input_ids"]

    records: list[StepRecord] = []
    prev_hidden = None
    generated_ids = input_ids
    # Bounded window of recently decoded token strings, used to detect
    # multi-word hedge phrases that BPE tokenization splits across several
    # tokens (e.g. "Re-evaluating" -> " Re" "-e" "valu" "ating"). Checking
    # each token in isolation against HEDGE_RE would miss almost every
    # multi-word pattern.
    recent_tokens: list[str] = []
    HEDGE_WINDOW = 20

    if verbose:
        print(f"Generating up to {max_new_tokens} tokens (prints as each one lands)...\n", flush=True)
        print(f"{'t':>4} {'token':<20} {'H_t / speed'}")

    hit_token_limit = False

    with torch.no_grad():
        # Prime the cache with the full prompt in one pass.
        out = model(generated_ids, use_cache=True, output_hidden_states=True)
        past_key_values = out.past_key_values
        logits_last = out.logits[0, -1, :]
        hidden_last = out.hidden_states[-1][0, -1, :].float()

        for t in range(max_new_tokens):
            h_t = entropy_from_logits(logits_last / temperature)
            speed = 0.0 if prev_hidden is None else float(
                torch.norm(hidden_last - prev_hidden).item()
            )
            prev_hidden = hidden_last

            probs = F.softmax(logits_last / temperature, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
            next_str = tok.decode(next_id)

            if next_id.item() == tok.eos_token_id:
                break

            recent_tokens.append(next_str)
            if len(recent_tokens) > HEDGE_WINDOW:
                recent_tokens.pop(0)
            window_text = "".join(recent_tokens)
            cur_token_start = len(window_text) - len(next_str)
            # Only flag onset on the token where a hedge phrase's match
            # actually completes, not on every token it overlaps.
            is_hedge = any(
                m.end() > cur_token_start for m in HEDGE_RE.finditer(window_text)
            )

            records.append(StepRecord(
                t=t,
                token_id=int(next_id.item()),
                token_str=next_str,
                entropy=h_t,
                speed=speed,
                is_hedge_onset=is_hedge,
            ))
            flag = " <-- HEDGE" if records[-1].is_hedge_onset else ""
            if verbose:
                print(f"{t:>4} {next_str!r:<20} H={h_t:6.3f} speed={speed:6.3f}{flag}", flush=True)

            # Feed ONLY the new token; past_key_values carries the rest of
            # the context, so this step is O(1) in sequence length, not O(t).
            out = model(
                next_id.unsqueeze(0),
                past_key_values=past_key_values,
                use_cache=True,
                output_hidden_states=True,
            )
            past_key_values = out.past_key_values
            logits_last = out.logits[0, -1, :]
            hidden_last = out.hidden_states[-1][0, -1, :].float()
        else:
            # Loop ran to completion without ever hitting `break` on EOS.
            hit_token_limit = True

    return records, hit_token_limit


def print_trace(records: list[StepRecord]) -> None:
    full_text = "".join(r.token_str for r in records)
    print("\n--- full generated text ---")
    print(full_text)

    hedges = [r for r in records if r.is_hedge_onset]
    if hedges:
        print(f"\n{len(hedges)} hedge onset(s) detected at t = {[r.t for r in hedges]}")
    else:
        print("\nNo hedge language detected in this generation.")


def save_trace_json(
    path: str, records: list[StepRecord], meta: dict, hit_token_limit: bool = False
) -> None:
    import json
    payload = {
        "meta": meta,
        "hit_token_limit": hit_token_limit,
        "steps": [r.__dict__ for r in records],
        "full_text": "".join(r.token_str for r in records),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--prefill", default=None,
                     help="Text to force as the start of the assistant's answer.")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--max-new-tokens", type=int, default=200)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--save-json", default=None,
                     help="Optional path to save this trace as JSON.")
    args = ap.parse_args()

    tok, model = load_model(args.model)
    recs, hit_token_limit = generate_and_trace(
        tok, model, args.prompt, args.max_new_tokens, args.temperature,
        args.seed, args.prefill,
    )
    print_trace(recs)
    print(f"\nhit_token_limit: {hit_token_limit}")

    if args.save_json:
        save_trace_json(args.save_json, recs, {
            "prompt": args.prompt, "prefill": args.prefill, "model": args.model,
            "max_new_tokens": args.max_new_tokens, "temperature": args.temperature,
            "seed": args.seed,
        }, hit_token_limit)
        print(f"\nSaved trace to {args.save_json}")
