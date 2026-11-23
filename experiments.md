# textfil Evaluation Experiments

Measuring how model size, quantisation, and filter parameters affect
watermark embedding performance.

## Background

textfil encodes binary payloads into existing natural text by substituting
words based on an LLM's ranked predictions. At each candidate word
position, the model predicts the next word given the preceding context.
The top two whole-word predictions (rank-1 and rank-2) form a bit pair:
rank-1 encodes a 0, rank-2 encodes a 1. If the payload bit doesn't match
the word already in the text, the word is swapped for the other prediction.

A fountain code (LT code with systematic encoding) spreads the payload
across more positions than strictly needed, so the message survives when
some positions are unusable. The decoder repeats the same walk over the
text and recovers the payload from the fountain symbols it collects.

### Filter pipeline

Not every word position is suitable for carrying a bit. Each candidate
passes through four filters in order:

1. **Base probability:** rank-2 must have P ≥ `min-r2` (default 0.10).
   If the alternative is too unlikely, the substitution would be
   conspicuous.

2. **Function word filter:** closed-class words (articles, prepositions,
   pronouns, auxiliaries) are skipped. Substituting "the" for "a" is
   conspicuous regardless of probability.

3. **Forward prediction divergence:** after tentatively placing rank-1 vs
   rank-2, the program predicts the *next* word in each case. If the
   continuations differ, the substitution would alter downstream context,
   so the position is skipped.

4. **Embedding similarity:** rank-1 and rank-2 are embedded (via
   `nomic-embed-text`) and their cosine similarity is checked (≥
   `min-sim`, default 0.50). This catches pairs that pass the other
   filters but aren't semantically interchangeable.

The encoder and decoder apply the same filters identically, keeping the
fountain code's RNG synchronised. **This synchronisation is the central
requirement.** If the encoder and decoder disagree about which positions
are usable — because, for example, different model weights produce
different predictions at a borderline position — the fountain stream
desynchronises and the payload is lost.

### Block-sequential processing

Words are processed in blocks separated by gaps. Within a block, earlier
substitutions are visible to later positions. Across blocks, the gap
ensures that no position's context window overlaps a modified word from
another block. This prevents substitutions from contaminating each other's
prediction context.

The gap also means that for naturalness scoring purposes, the context at
every modified position is identical in the original and watermarked text.
This simplifies the substitution quality measurement.


## Hardware constraints

The evaluation runs on an ollama instance with an NVIDIA RTX 2000 Ada
Generation Laptop GPU:

| Property       | Value                             |
|----------------|-----------------------------------|
| GPU            | RTX 2000 Ada Generation Laptop    |
| Total VRAM     | 8188 MiB                          |
| Available VRAM | ~7800 MiB                         |
| Practical limit| ~6–7 GB model                     |

Models that fit within this budget:

| Model              | Quantisation      | Approx. size | Fits? |
|--------------------|-------------------|-------------|-------|
| `llama3.2:1b`      | q4_K_M (default)  | ~1.3 GB     | Yes   |
| `llama3.2:3b`      | q4_K_M (default)  | ~2.0 GB     | Yes   |
| `llama3.2:3b-text-q8_0` | q8_0         | ~3.4 GB     | Yes   |
| `llama3.1:8b`      | q4_0 (default)    | ~4.7 GB     | Yes   |
| `gemma3:4b`        | q4_K_M (default)  | ~3.3 GB     | Yes, different family |
| `gemma4:e4b`       | (default)         | ~6.6 GB     | Yes, different family |
| `llama3.1:8b`      | q8_0              | ~8.5 GB     | No — exceeds VRAM    |

The embedding model `nomic-embed-text` runs alongside the generation
model and uses additional VRAM.


## Metrics

Each trial collects five categories of measurements.

### 1. Round-trip correctness

The fundamental test: encode a known payload, decode from the watermarked
text, check whether the decoded payload matches.

- `round_trip` — Boolean. True if the decoded payload matches the input.
- `decoded` — The hex string the decoder produced, or an error message.
- `recovered_bits`, `total_bits` — On decode failure, how many payload
  bits were recovered out of how many total. Measures how close a failed
  trial came to succeeding.

### 2. Capacity and bandwidth

How many usable bit-carrying positions the text yields, and what fraction
of candidate positions survive each filter stage.

- `enc_total` — Total candidate positions the walker examined.
- `enc_usable` — Positions that passed all four filters.
- `enc_skip_base`, `enc_skip_func`, `enc_skip_div`, `enc_skip_sim` —
  Positions rejected at each filter stage. These four plus `enc_usable`
  sum to `enc_total`.
- `enc_overhead` — Ratio of usable positions to payload bits. Needs to be
  ≥ 1.0 for recovery; > 1.2 provides margin for the fountain code.
- `enc_changed` — Words actually modified (positions where the existing
  word didn't match the required bit).

The decode side reports the same position and filter counts. When encode
and decode use different models, the two sets can diverge — that
divergence is the signal for experiment 4.

### 3. Text naturalness

Measured via substitution log-probability delta. At each position where a
word was changed, a judge model is queried with the preceding context. The
judge independently predicts the next word and returns log-probabilities
for its top-20 predictions. The delta is:

    Δ = log P(substituted word | context) − log P(original word | context)

A negative delta means the substitution is less probable than the original
— the text became less natural at that position. A delta near zero means
the substitution is roughly as probable as the original.

- `mean_logprob_delta` — Average delta across all modified positions.
  Closer to zero is better.
- `worst_logprob_delta` — The most negative delta — the single worst
  substitution.
- `nat_positions` — Number of positions scored (should equal
  `enc_changed`).

If a word is not found in the judge model's top-20 predictions, a default
log-probability of −20.0 is assigned.

**Why not full-text perplexity?** Computing perplexity requires
O(text_length) model queries. For a 3000-word text, that's 3000 API calls
per trial. The substitution delta measures quality at the positions that
actually changed — far cheaper and more targeted. With only ~11 modified
positions in a 3000-word text, global perplexity barely shifts even for
terrible substitutions; the delta catches them directly.

### 4. Compute cost

The harness captures both wall-clock time (from the calling process) and
model-side inference timing (from ollama's response fields).

**Wall-clock:**
- `encode_seconds`, `decode_seconds` — Total elapsed time for the full
  encode and decode operations, including network round-trips and all
  model queries.

**Model-side (from ollama response timing):**
- `enc_predict_calls`, `dec_predict_calls` — Number of generate API calls.
  Each usable position requires 3 predict calls (initial prediction + two
  forward divergence checks); each position rejected at an earlier filter
  stage requires 1.
- `enc_prompt_tokens`, `dec_prompt_tokens` — Total prompt tokens evaluated
  across all predict calls. Grows with context length and number of
  positions.
- `enc_prompt_eval_seconds`, `dec_prompt_eval_seconds` — Cumulative time
  the model spent evaluating prompt tokens (server-side, excluding network).
- `enc_gen_eval_seconds`, `dec_gen_eval_seconds` — Cumulative time the
  model spent generating output tokens.
- `enc_prompt_tokens_per_sec`, `dec_prompt_tokens_per_sec` — Prompt
  evaluation throughput.

**Not currently captured:**
- GPU utilisation percentage and GPU memory usage. These would require
  polling `nvidia-smi` on the GPU host during each trial. See
  [GPU monitoring](#gpu-monitoring) below.
- Energy consumption. The RTX 2000 Ada reports power draw through
  `nvidia-smi --query-gpu=power.draw`. Integrating power × time gives
  energy per trial, but requires the same remote monitoring setup.
- CPU utilisation. Less relevant since the work is GPU-bound, but could
  be captured with the same remote agent.

### 5. Encoder/decoder robustness

Measured in experiment 4. No dedicated metric fields — the standard
`round_trip`, position counts, and `recovered_bits`/`total_bits` tell the
story. When the encode and decode models differ, positions where their
predictions disagree cause desynchronisation. The difference between
`enc_usable` and `dec_usable` quantifies the disagreement.


## Experiment 1: Model and quantisation survey

### Question

How do model size and quantisation level affect watermarking performance?
Do larger models produce more usable positions, better naturalness, or
both? Does quantisation within the same model family change the position
set?

### Design

Fix the text, payload, and all filter parameters. Vary the model across
sizes and quantisation levels available within the hardware budget. Each
model is used for both encode and decode.

| Variable   | Values |
|------------|--------|
| Models     | `llama3.2:1b`, `llama3.2:3b`, `llama3.2:3b-text-q8_0`, `llama3.1:8b`, `gemma3:4b`, `gemma4:e4b` |
| Texts      | Five ~3700-4000 word public domain texts of varied genre (see `texts/`) |
| Payload    | Fixed — `beef` (2 bytes, 16 bits) |
| Parameters | Defaults: `min-r2=0.10`, `min-sim=0.50`, `seed=42` |
| Repetitions| 1 (deterministic with fixed seed) |

### What to look for

- Does the base probability filter reject more or fewer positions on
  larger models? Larger models have sharper probability distributions, so
  rank-2 may fall below the threshold more often.
- Does the forward divergence filter behave differently? Larger models may
  produce more consistent continuations, reducing divergence rejections.
- Do larger models encode faster or slower? Per-token inference is slower,
  but if more positions pass on the first try, total API calls may be
  fewer.
- How does `llama3.2:3b` (q4_K_M) compare to `llama3.2:3b-text-q8_0`?
  If the position sets are identical, quantisation doesn't matter for
  correctness. If they differ, that tells us how much quantisation noise
  enters the predictions.
- Does `gemma3:4b` (a different model family) produce qualitatively
  different filter-stage attrition?
- Prompt throughput (tokens/s) across model sizes — the larger the model,
  the slower per token, but also potentially fewer wasted calls if
  predictions are sharper.

### Running it

~~~sh
for text in texts/*.txt; do
  ./eval -url "$OLLAMA_URL" \
    -text "$text" \
    -payload beef \
    -models "llama3.2:1b,llama3.2:3b,llama3.2:3b-text-q8_0,llama3.1:8b,gemma3:4b,gemma4:e4b" \
    -judge "llama3.1:8b"
done > results/survey.jsonl
~~~


## Experiment 2: Capacity scaling

### Question

How does text length and payload size affect capacity? What's the minimum
text length needed for a given payload? At what point does the fountain
code overhead become the bottleneck?

### Design

Fix the model (`llama3.2:3b-text-q8_0`, the most reliable llama variant
from experiment 1). Vary text length by truncating two texts to different
word counts. Vary payload size across several values.

| Variable     | Values |
|--------------|--------|
| Model        | Fixed — `llama3.2:3b-text-q8_0` |
| Texts        | darwin and doyle, truncated to 500, 1000, 2000, 3000 words |
| Payloads     | `ff` (1B), `beef` (2B), `deadbeef` (4B) |
| Parameters   | Defaults |

This is 2×4×3 = 24 trials. Truncated text files are prepared at word
boundaries in `results/`.

~~~sh
for text in darwin-origin doyle-scandal; do
  for payload in ff beef deadbeef; do
    for words in 500 1000 2000 3000; do
      ./eval -url "$OLLAMA_URL" \
        -text "results/text-${text}-${words}.txt" \
        -payload "$payload" \
        -models "llama3.2:3b-text-q8_0"
    done
  done
done > results/capacity.jsonl
~~~

### What to look for

- Usable positions per 1000 words — is this roughly constant across text
  lengths, or does it depend on the text's content?
- The minimum text length at which a 2-byte payload round-trips.
- Whether a 4-byte payload (32 bits) is achievable at any text length.
- Whether the two texts (darwin — scientific prose, doyle — dialogue-heavy
  fiction) differ in yield scaling.


## Experiment 3: Parameter sensitivity

### Question

How do the two main filter thresholds — `min-r2` (minimum rank-2
probability) and `min-sim` (minimum embedding similarity) — affect the
tradeoff between capacity and naturalness?

### Design

Fix the model (`llama3.2:3b-text-q8_0`) and use two texts (darwin —
median yield, walden — lowest yield). Sweep each parameter independently
while holding the other at its default.

**Sweep A: min-r2**

| Variable  | Values |
|-----------|--------|
| Model     | Fixed — `llama3.2:3b-text-q8_0` |
| Texts     | darwin, walden |
| `min-r2`  | 0.05, 0.08, 0.10, 0.12, 0.15, 0.20 |
| `min-sim` | Fixed — 0.50 |

~~~sh
for text in texts/darwin-origin.txt texts/thoreau-walden.txt; do
  for r2 in 0.05 0.08 0.10 0.12 0.15 0.20; do
    ./eval -url "$OLLAMA_URL" \
      -text "$text" \
      -payload beef \
      -models "llama3.2:3b-text-q8_0" \
      -min-r2 "$r2" \
      -judge "llama3.1:8b"
  done
done > results/sweep-r2.jsonl
~~~

**Sweep B: min-sim**

| Variable  | Values |
|-----------|--------|
| Model     | Fixed — `llama3.2:3b-text-q8_0` |
| Texts     | darwin, walden |
| `min-sim` | 0.30, 0.50, 0.70, 0.85, 0.95 |
| `min-r2`  | Fixed — 0.10 |

~~~sh
for text in texts/darwin-origin.txt texts/thoreau-walden.txt; do
  for sim in 0.30 0.50 0.70 0.85 0.95; do
    ./eval -url "$OLLAMA_URL" \
      -text "$text" \
      -payload beef \
      -models "llama3.2:3b-text-q8_0" \
      -min-sim "$sim" \
      -judge "llama3.1:8b"
  done
done > results/sweep-sim.jsonl
~~~

### What to look for

- **Capacity vs. naturalness tradeoff:** lowering `min-r2` admits more
  positions but with less probable substitutions. At what threshold does
  the mean delta become unacceptable?
- **Filter stage attrition:** how many additional positions does each step
  of the `min-r2` reduction unlock? If most positions are rejected by
  other filters anyway, loosening `min-r2` may gain little.
- **Similarity threshold effect:** raising `min-sim` should improve
  naturalness but reduce capacity. Is there a sweet spot where most of the
  semantic gain is captured without losing many positions?
- **Round-trip reliability:** do any parameter values cause round-trip
  failure?

Both sweeps include `-judge llama3.1:8b` to collect naturalness scores.
If the 8b model is too slow for the full sweep, drop the judge and add it
on a targeted rerun.


## Experiment 4: Cross-quantisation robustness

### Question

If the encoder uses one quantisation of a model and the decoder uses a
different quantisation, does the payload survive? Within the same model
family, how much quantisation divergence can the system tolerate?

### Motivation

In practice, the person encoding and the person decoding may not have
exactly the same model weights. If quantisation differences are small
enough that the top-2 predictions agree at most positions, the fountain
code's redundancy might absorb the disagreement at the remaining
positions.

### Design

Encode with model A, decode with model B, using all pairwise combinations
from two groups: cross-quantisation within the same base model, and
cross-family between llama and gemma.

| Variable    | Values |
|-------------|--------|
| Models      | `llama3.2:3b`, `llama3.2:3b-text-q8_0`, `gemma3:4b`, `gemma4:e4b` |
| Cross pairs | All A→B pairs where A ≠ B, plus same-model baselines |
| Texts       | darwin, doyle |
| Payload     | `beef` (2 bytes) |

~~~sh
for text in texts/darwin-origin.txt texts/doyle-scandal.txt; do
  ./eval -url "$OLLAMA_URL" \
    -text "$text" \
    -payload beef \
    -models "llama3.2:3b,llama3.2:3b-text-q8_0" \
    -cross \
    -judge "llama3.1:8b"
  ./eval -url "$OLLAMA_URL" \
    -text "$text" \
    -payload beef \
    -models "gemma3:4b,gemma4:e4b" \
    -cross \
    -judge "llama3.1:8b"
done > results/cross-quant.jsonl
~~~

The `-cross` flag generates all encode/decode model pairs beyond the
same-model trials. Each group of 2 models produces 2 same-model + 2
cross-model = 4 trials per text, for 16 trials total.

### What to look for

- Does the round-trip succeed for any cross-quantisation pair? If yes, by
  how much do `enc_usable` and `dec_usable` differ?
- If the round-trip fails, how many bits were recovered
  (`recovered_bits`/`total_bits`)? Partial recovery indicates how close
  the two quantisations are.
- Can the difference be absorbed by running the fountain code with more
  overhead (longer text, same payload)?


## Eval harness

The harness is a Go program at `cmd/eval/main.go` within the textfil
module. It shells out to the `textfil` binary for encode/decode operations
(using `-json` and `-debug` flags) and uses the `internal/ollama` package
directly for naturalness scoring.

### Building

~~~sh
cd ~/src/github.com/kortschak/textfil
go build -o textfil .
go build -o eval ./cmd/eval
~~~

### Flags

| Flag           | Default                  | Description |
|----------------|--------------------------|-------------|
| `-url`         | `http://localhost:11434`  | Ollama API base URL |
| `-text`        | (required)               | Input text file |
| `-payload`     | `beef`                   | Hex payload to encode |
| `-models`      | `llama3.2:3b`            | Comma-separated model list |
| `-cross`       | `false`                  | Also test all encode/decode model pairs |
| `-judge`       | (none)                   | Model for naturalness scoring |
| `-embed-model` | `nomic-embed-text`       | Embedding model for similarity filter |
| `-n`           | `1`                      | Repetitions per trial |
| `-textfil`     | (search PATH)            | Path to textfil binary |
| `-seed`        | `42`                     | Fountain coding seed |
| `-min-r2`      | `0.10`                   | Minimum rank-2 probability threshold |
| `-min-sim`     | `0.50`                   | Minimum r1/r2 embedding similarity |

### How it works

For each trial, the harness:

1. Runs `textfil -payload ... -json -debug` and captures the watermarked
   text from JSON stdout and filter/timing statistics from debug stderr.
2. Writes the watermarked text to a temporary file.
3. Runs `textfil -bits ... -json -debug` on the temporary file and
   captures the decoded payload and decode-side statistics.
4. Compares the decoded payload to the input.
5. If a judge model is specified, queries it at each modified position to
   compute the substitution log-probability delta.
6. Writes one JSON line to stdout and a progress line to stderr.

After all trials, a summary table is printed to stderr.

### Trial generation

Without `-cross`, the harness runs one trial per model (same model for
encode and decode). With `-cross`, it additionally generates all ordered
pairs where the encode model differs from the decode model. For *n*
models, that's *n* same-model trials plus *n*(n−1) cross-model trials.


## Output format

Each trial produces one JSON line to stdout. All fields are present on
every line (some may be zero or empty). Naturalness and partial-recovery
fields appear only when applicable.

~~~json
{
  "name": "llama3.2:3b",
  "encode_model": "llama3.2:3b",
  "decode_model": "llama3.2:3b",
  "payload": "beef",
  "text_words": 3011,
  "repetition": 0,
  "round_trip": true,
  "decoded": "beef",
  "enc_total": 1760,
  "enc_usable": 24,
  "enc_skip_base": 1151,
  "enc_skip_func": 512,
  "enc_skip_div": 64,
  "enc_skip_sim": 9,
  "enc_changed": 11,
  "enc_overhead": 1.5,
  "dec_total": 1760,
  "dec_usable": 24,
  "dec_skip_base": 1151,
  "dec_skip_func": 512,
  "dec_skip_div": 64,
  "dec_skip_sim": 9,
  "encode_seconds": 62.3,
  "decode_seconds": 59.8,
  "enc_predict_calls": 5280,
  "enc_prompt_tokens": 42240,
  "enc_prompt_eval_seconds": 23.456,
  "enc_gen_eval_seconds": 1.234,
  "enc_prompt_tokens_per_sec": 1800.3,
  "dec_predict_calls": 5280,
  "dec_prompt_tokens": 42240,
  "dec_prompt_eval_seconds": 23.012,
  "dec_gen_eval_seconds": 1.198,
  "dec_prompt_tokens_per_sec": 1835.1,
  "judge_model": "llama3.1:8b",
  "mean_logprob_delta": -0.42,
  "worst_logprob_delta": -2.1,
  "nat_positions": 11
}
~~~

On decode failure, partial recovery fields appear:

~~~json
{
  "round_trip": false,
  "decoded": "decode failed: 14/16 bits recovered",
  "recovered_bits": 14,
  "total_bits": 16
}
~~~

### Analysing results with jq

~~~sh
# Summary table
jq -r '[.name, .round_trip, .enc_usable, .enc_changed, .enc_overhead,
        .encode_seconds] | @tsv' results.jsonl

# Filter attrition breakdown
jq '{name, base: .enc_skip_base, func: .enc_skip_func,
     div: .enc_skip_div, sim: .enc_skip_sim,
     usable: .enc_usable}' results.jsonl

# Yield (positions per 1000 words)
jq '{name, yield: (.enc_usable / .text_words * 1000)}' results.jsonl

# Cross-model position disagreement
jq 'select(.encode_model != .decode_model) |
    {name, enc: .enc_usable, dec: .dec_usable,
     diff: (.enc_usable - .dec_usable)}' results.jsonl

# Naturalness scores
jq 'select(.mean_logprob_delta) |
    {name, mean: .mean_logprob_delta,
     worst: .worst_logprob_delta}' results.jsonl

# Compute cost comparison
jq '{name, calls: .enc_predict_calls,
     prompt_tok: .enc_prompt_tokens,
     tok_s: .enc_prompt_tokens_per_sec,
     wall_s: .encode_seconds}' results.jsonl

# Partial recovery on failures
jq 'select(.recovered_bits) |
    {name, recovered: .recovered_bits,
     total: .total_bits}' results.jsonl
~~~

The stderr summary table provides a quick visual overview without jq:

~~~
trial                           RT  usable     chg     ovhd   enc(s)   dec(s)     tok/s
----------------------------------------------------------------------------------------
llama3.2:1b                      ✗      18       8    1.12x       45       43      3200
llama3.2:3b                      ✓      24      11    1.50x       62       60      1800
llama3.2:3b-text-q8_0            ✓      24      11    1.50x       68       66      1650
llama3.1:8b                      ✓      26      12    1.62x      120      118       950

Yield (usable positions per 1000 words):
  llama3.2:1b                     6.0 positions/kword
  llama3.2:3b                     8.0 positions/kword
  llama3.2:3b-text-q8_0           8.0 positions/kword
  llama3.1:8b                     8.6 positions/kword
~~~


## GPU monitoring

The eval harness captures model-side inference timing from ollama's
response fields: prompt evaluation time, generation time, and prompt
tokens evaluated. This covers latency and throughput without requiring
access to the GPU host.

For GPU utilisation, memory, and power draw — which are needed for energy
per condition — a separate monitor must run on the GPU host during the
trials. The simplest approach:

~~~sh
# Run on the GPU host in a separate terminal during the experiment
nvidia-smi --query-gpu=timestamp,utilization.gpu,utilization.memory,memory.used,power.draw \
  --format=csv,nounits \
  -l 1 \
  > gpu-log.csv
~~~

This produces a timestamped CSV that can be correlated with the per-trial
wall-clock times in the JSON output. Each trial's start and end times
bracket a window in the GPU log; the mean and peak values within that
window are the trial's GPU metrics.

For automated correlation, the eval harness prints trial start timestamps
to stderr. A post-processing script can join the two datasets.


## Future extensions

### A/B detection test

An LLM-as-judge test: split the text into paragraphs, present original
and watermarked versions of each paragraph in random order, ask the model
which one has been modified. Accuracy near 50% means the watermark is
undetectable; significantly above 50% means substitutions are noticeable.

Requires a chat-capable model and careful prompt design to avoid position
bias.

### Full-text perplexity ratio

Rather than scoring only at modified positions, compute the perplexity of
the entire text before and after watermarking. The ratio
PPL(watermarked) / PPL(original) measures global text quality degradation.
Requires O(text_length) model queries and takes several minutes per trial.

### Text perturbation robustness

Apply common text transformations to the watermarked text before decoding:
normalise whitespace, re-wrap at different widths, apply Unicode
normalisation (NFC/NFD), insert or remove punctuation. Tests whether the
watermark survives the kind of formatting changes that happen when text is
copied through different systems.
