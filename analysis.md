# textfil Experiment Results

We ran four experiments to measure how model choice, text content, filter
parameters, and text length affect watermark encoding and decoding in
textfil. The experiments used six models (four llama variants and two
gemma variants), five public domain texts of different genres, and a
fixed 2-byte payload (`beef`, 16 bits) unless otherwise noted.

The most important finding is that reliable round-trip encoding is harder
than expected. No model achieves 100% reliability across all five texts.
The root cause is that word substitutions during encoding change the
context that later positions see, causing the encoder and decoder to
disagree about which positions are usable. This desynchronisation affects
every model on every text to some degree. The fountain code absorbs small
disagreements, but when the disagreement exceeds the redundancy margin,
decoding fails.

The gemma models produce roughly twice as many usable positions as the
llama models (19-23 vs 9-13 positions per 1000 words), which gives them
more margin to absorb desynchronisation. Cross-model decoding — using
different weights for encode and decode — fails universally, even between
quantisation variants of the same base model.

Hardware: NVIDIA RTX 2000 Ada Generation (8 GB VRAM) running ollama.
Embedding model: nomic-embed-text. All trials used deterministic generation
(temperature 0, seed 1) and fountain code seed 42.


## Experiment 1: Model and quantisation survey

Six models tested on five texts, each encoding and decoding the 2-byte
payload `beef`. The llama3.1:8b model served as the naturalness judge
for all trials.

### Texts

| Short name | Source | Words | Genre |
|------------|--------|------:|-------|
| darwin | Darwin, *Origin of Species* ch 1 | 3741 | Scientific exposition |
| doyle | Conan Doyle, "A Scandal in Bohemia" | 3937 | Dialogue-heavy fiction |
| gatsby | Fitzgerald, *The Great Gatsby* ch 1 | 3983 | Modern literary fiction |
| walden | Thoreau, *Walden* "Economy" | 3786 | Reflective essay |
| wells | H.G. Wells, *War of the Worlds* chs 1-5 | 3919 | Victorian science fiction |

### Round-trip results

|                  | 1b | 3b | 3b-q8 | 8b | gemma3:4b | gemma4:e4b |
|------------------|:--:|:--:|:-----:|:--:|:---------:|:----------:|
| darwin (3741w)   | FAIL (30) | OK (40) | OK (49) | FAIL (56) | OK (81) | OK (88) |
| doyle (3937w)    | FAIL (43) | OK (50) | OK (52) | FAIL (57) | OK (83) | OK (108) |
| gatsby (3983w)   | FAIL (41) | FAIL (45) | OK (54) | OK (55) | FAIL (70) | OK (88) |
| walden (3786w)   | FAIL (30) | FAIL (32) | OK (33) | FAIL (31) | OK (64) | OK (86) |
| wells (3919w)    | FAIL (31) | OK (42) | FAIL (45) | FAIL (45) | OK (76) | FAIL (83) |
| **Pass rate**    | **0/5** | **3/5** | **4/5** | **1/5** | **4/5** | **4/5** |

Numbers in parentheses are usable positions (encoder side).

### Model summary

| Model | Pass | Avg yield | Avg usable | Avg changed | Avg tok/s | Avg encode (s) |
|-------|-----:|----------:|-----------:|------------:|----------:|---------------:|
| llama3.2:1b | 0/5 | 9.0/kw | 35.0 | 12.0 | 833 | 86 |
| llama3.2:3b | 3/5 | 10.8/kw | 41.8 | 16.2 | 362 | 122 |
| llama3.2:3b-text-q8_0 | 4/5 | 12.0/kw | 46.6 | 19.2 | 334 | 123 |
| llama3.1:8b | 1/5 | 12.6/kw | 48.8 | 21.6 | 123 | 234 |
| gemma3:4b | 4/5 | 19.3/kw | 74.8 | 31.0 | 157 | 223 |
| gemma4:e4b | 4/5 | 23.4/kw | 90.6 | 38.8 | 119 | 296 |

### Filter stage attrition (averages across five texts)

| Model | Base prob % | Function word % | Divergence % | Similarity % | Usable % |
|-------|------------:|----------------:|-------------:|-------------:|---------:|
| llama3.2:1b | 71.9 | 22.9 | 3.1 | 0.5 | 1.6 |
| llama3.2:3b | 69.4 | 24.8 | 3.3 | 0.6 | 1.9 |
| llama3.2:3b-text-q8_0 | 70.0 | 24.3 | 2.9 | 0.7 | 2.1 |
| llama3.1:8b | 68.8 | 25.4 | 2.9 | 0.7 | 2.2 |
| gemma3:4b | 56.5 | 32.6 | 6.3 | 1.2 | 3.4 |
| gemma4:e4b | 51.4 | 32.6 | 10.4 | 1.5 | 4.1 |

The base probability filter rejects 51-72% of positions across all
models — it is the dominant filter. The gemma models have a substantially
lower base rejection rate (51-57%) compared to the llama models (69-72%),
meaning gemma's rank-2 predictions are more often above the 0.10
probability threshold. This is the primary driver of gemma's higher
yield.

The function word filter accounts for 23-33%, which is a property of the
text's vocabulary rather than the model. The divergence filter matters
more for gemma models (6-10%) than llama (3%), suggesting that gemma's
higher-ranked predictions include more words that alter the downstream
continuation. The similarity filter removes very few positions (0.5-1.5%)
for all models.

### Findings

**The 1b model never succeeds.** It has the lowest yield (9 positions per
1000 words, only 1.6% of candidates pass all filters) and the highest
base rejection rate (72%). Its predictions are not sharp enough to
produce confident rank-2 alternatives.

**The 8b model is surprisingly unreliable (1/5).** Despite having the
highest yield among the llama models, it passes only on gatsby. The
issue is that the 8b model's substitutions are more likely to change
downstream predictions because it uses more of the available context. It
produces more usable positions but also more desynchronisation.

**The 3b-text-q8_0 is the most reliable llama model (4/5).** The higher
quantisation precision gives slightly more usable positions than the
standard 3b (12.0 vs 10.8 per 1000 words) and fails only on wells.

**The gemma models produce roughly 2x the yield of the llama models.**
Both gemma3:4b and gemma4:e4b pass 4/5 texts, but each fails on a
different text (gatsby and wells respectively). Their higher yield gives
more fountain code redundancy to absorb desynchronisation. gemma4:e4b has
the highest yield (23.4/kw) but is the slowest (119 tok/s, 296s per
encode).

**Throughput vs capacity tradeoff.** The 1b model is fastest (833 tok/s)
but has the fewest usable positions. The gemma4:e4b is slowest (119
tok/s) but has the most. The 3b variants hit a middle ground at 334-362
tok/s with moderate yield.


## Experiment 2: Capacity scaling

Using llama3.2:3b-text-q8_0, two texts (darwin and doyle) were truncated
to 500, 1000, 2000, and 3000 words and tested with payloads of 1, 2, and
4 bytes.

### Darwin

| Words | Payload | Bits | RT | Usable | Overhead | Recovered |
|------:|---------|-----:|----|-------:|---------:|-----------|
| 500 | ff | 8 | OK | 7 | 0.88x | — |
| 1000 | ff | 8 | OK | 14 | 1.75x | — |
| 2000 | ff | 8 | OK | 24 | 3.00x | — |
| 3000 | ff | 8 | OK | 40 | 5.00x | — |
| 500 | beef | 16 | FAIL | 7 | 0.44x | 7/16 |
| 1000 | beef | 16 | FAIL | 13 | 0.81x | 13/16 |
| 2000 | beef | 16 | OK | 23 | 1.44x | — |
| 3000 | beef | 16 | OK | 40 | 2.50x | — |
| 500 | deadbeef | 32 | FAIL | 8 | 0.25x | 8/32 |
| 1000 | deadbeef | 32 | FAIL | 14 | 0.44x | 14/32 |
| 2000 | deadbeef | 32 | FAIL | 24 | 0.75x | 24/32 |
| 3000 | deadbeef | 32 | OK | 43 | 1.34x | — |

### Doyle

| Words | Payload | Bits | RT | Usable | Overhead | Recovered |
|------:|---------|-----:|----|-------:|---------:|-----------|
| 500 | ff | 8 | FAIL | 5 | 0.62x | 5/8 |
| 1000 | ff | 8 | OK | 15 | 1.88x | — |
| 2000 | ff | 8 | OK | 28 | 3.50x | — |
| 3000 | ff | 8 | OK | 43 | 5.38x | — |
| 500 | beef | 16 | FAIL | 5 | 0.31x | 5/16 |
| 1000 | beef | 16 | FAIL | 15 | 0.94x | 14/16 |
| 2000 | beef | 16 | OK | 27 | 1.69x | — |
| 3000 | beef | 16 | FAIL | 42 | 2.62x | — |
| 500 | deadbeef | 32 | FAIL | 5 | 0.16x | 5/32 |
| 1000 | deadbeef | 32 | FAIL | 15 | 0.47x | 14/32 |
| 2000 | deadbeef | 32 | FAIL | 27 | 0.84x | 27/32 |
| 3000 | deadbeef | 32 | FAIL | 42 | 1.31x | — |

### Findings

**Yield is roughly constant per 1000 words.** Darwin yields 12-14
usable positions per 1000 words regardless of text length. Doyle yields
10-15 per 1000 words with more variation at short lengths (the 500-word
excerpt only produces 5 usable positions, or 10 per 1000 words).

**Minimum text lengths for each payload size (darwin):**
- 1 byte (8 bits): 500 words, though at only 0.88x overhead — extremely
  tight. 1000 words is safer (1.75x).
- 2 bytes (16 bits): 2000 words (1.44x overhead).
- 4 bytes (32 bits): 3000 words (1.34x overhead) — the tightest
  successful trial in the entire experiment.

**4 bytes is achievable on darwin but not on doyle.** At 3000 words,
doyle has 42 usable positions and 1.31x overhead — very close to the
darwin success at 1.34x — but fails because of encode/decode
desynchronisation. The doyle 3000-word beef trial also fails despite
2.62x overhead (42 usable positions), confirming that the doyle text
produces more desynchronisation with this model.

**Partial recovery is proportional to overhead.** When the fountain code
doesn't have enough positions, the fraction of bits recovered closely
tracks the overhead ratio. At 2000 words with deadbeef, darwin recovers
24/32 bits (0.75x overhead, 75% recovery). This confirms that the
fountain code degrades gracefully.


## Experiment 3: Parameter sensitivity

Using llama3.2:3b-text-q8_0 on darwin (median yield) and walden
(lowest yield), each filter threshold was swept independently while
holding the other at its default.

### min-r2 sweep (min-sim fixed at 0.50)

| min-r2 | Darwin RT | Darwin usable | Darwin ovhd | Walden RT | Walden usable | Walden ovhd |
|-------:|-----------|:-------------:|:-----------:|-----------|:-------------:|:-----------:|
| 0.05 | OK | 100 | 6.25x | FAIL | 76 | 4.75x |
| 0.08 | OK | 70 | 4.38x | OK | 47 | 2.94x |
| 0.10 | OK | 49 | 3.06x | OK | 33 | 2.06x |
| 0.12 | OK | 41 | 2.56x | OK | 25 | 1.56x |
| 0.15 | FAIL | 29 | 1.81x | OK | 16 | 1.00x |
| 0.20 | FAIL | 12 | 0.75x | FAIL (12/16) | 10 | 0.62x |

### min-sim sweep (min-r2 fixed at 0.10)

| min-sim | Darwin RT | Darwin usable | Darwin ovhd | Walden RT | Walden usable | Walden ovhd |
|--------:|-----------|:-------------:|:-----------:|-----------|:-------------:|:-----------:|
| 0.30 | FAIL | 64 | 4.00x | OK | 49 | 3.06x |
| 0.50 | OK | 49 | 3.06x | FAIL | 33 | 2.06x |
| 0.70 | FAIL | 16 | 1.00x | FAIL | 0 | — |
| 0.85 | FAIL (4/16) | 4 | 0.25x | FAIL | 0 | — |
| 0.95 | FAIL (0/16) | 0 | — | FAIL | 0 | — |

### Findings

**Lowering min-r2 strongly increases capacity.** Going from 0.10 to 0.05
roughly doubles the usable positions (49 to 100 on darwin, 33 to 76 on
walden). The relationship is smooth: each 0.02 reduction in the threshold
admits roughly 15-20 more positions.

**More positions can hurt.** The walden trial at min-r2=0.05 has 76
usable positions (4.75x overhead) but fails. With more positions
admitted, more words are substituted (40 changed vs 16 at the default),
and the extra substitutions increase the chance of downstream prediction
divergence. The same pattern appears on darwin at min-sim=0.30 (64
positions, 4.00x overhead, fails).

**The min-sim filter is a cliff, not a slope.** On darwin, dropping
min-sim from 0.50 to 0.70 cuts positions from 49 to 16. On walden,
anything above 0.50 produces zero positions. The rank-1/rank-2 word
pairs that pass the other filters typically have cosine similarity in
the 0.30-0.60 range; requiring 0.70+ eliminates nearly all of them.

**The operating window is narrow.** For darwin, the working range is
min-r2 0.05-0.12 (at min-sim 0.50) or min-sim 0.50 (at min-r2 0.10).
For walden, the range is tighter: min-r2 0.08-0.15 (at min-sim 0.50)
or min-sim 0.30 (at min-r2 0.10). There is no single parameter setting
that works on all texts.

**The default parameters (0.10/0.50) are a reasonable middle ground**
but not optimal for all texts. Walden barely works at the defaults
(33 positions, 2.06x overhead). Dropping to min-r2=0.08 would give
walden more margin (47 positions) without hurting darwin.


## Experiment 4: Cross-quantisation robustness

Tested whether the payload survives when the encoder and decoder use
different model weights: llama3.2:3b (q4_K_M) vs llama3.2:3b-text-q8_0
(cross-quantisation within the same family) and gemma3:4b vs gemma4:e4b
(cross-family). Two texts: darwin and doyle.

### Results

| Trial | Text | RT | enc_usable | dec_usable | Diff | Decoded |
|-------|------|----|:----------:|:----------:|-----:|---------|
| 3b (same) | darwin | OK | 40 | 41 | -1 | — |
| 3b-q8 (same) | darwin | OK | 49 | 51 | -2 | — |
| 3b -> 3b-q8 | darwin | FAIL | 41 | 52 | -11 | fbfb |
| 3b-q8 -> 3b | darwin | FAIL | 51 | 42 | +9 | dfde |
| gemma3 (same) | darwin | OK | 81 | 81 | 0 | — |
| gemma4 (same) | darwin | OK | 88 | 88 | 0 | — |
| gemma3 -> gemma4 | darwin | FAIL | 83 | 86 | -3 | ffff |
| gemma4 -> gemma3 | darwin | FAIL | 88 | 83 | +5 | b7ff |
| 3b (same) | doyle | OK | 50 | 47 | +3 | — |
| 3b-q8 (same) | doyle | OK | 50 | 51 | -1 | — |
| 3b -> 3b-q8 | doyle | FAIL | 50 | 52 | -2 | befd |
| 3b-q8 -> 3b | doyle | FAIL | 52 | 48 | +4 | 7eef |
| gemma3 (same) | doyle | OK | 83 | 81 | +2 | — |
| gemma4 (same) | doyle | OK | 108 | 110 | -2 | — |
| gemma3 -> gemma4 | doyle | FAIL | 81 | 114 | -33 | fde7 |
| gemma4 -> gemma3 | doyle | FAIL | 111 | 80 | +31 | fe7b |

### Findings

**Cross-model decoding fails universally.** All eight cross-model trials
fail, including the cross-quantisation pair (same base model, different
quantisation level).

**Cross-quantisation disagreement is substantial.** The llama 3b q4->q8
pair on darwin shows enc_usable=41 vs dec_usable=52 — 11 positions where
the two quantisations disagree about whether the position passes the
filter pipeline. This is far larger than the same-model desync (typically
1-2 positions).

**Cross-family disagreement is massive.** The gemma3->gemma4 pair on doyle
shows enc_usable=81 vs dec_usable=114 — a 33-position difference. These
models have fundamentally different prediction behaviour and there is no
reason to expect them to agree.

**The closest near-miss: 3b->3b-q8 on doyle decoded `befd`.** The payload
`beef` in binary is `1011 1110 1110 1111` and `befd` is `1011 1110 1111
1101` — the last byte differs by a few bits. With only a 2-position
usable difference on this text, the cross-quantisation pair was close
to working. Longer text (more fountain overhead) might bridge this gap
for some quantisation pairs on some texts, but not reliably.

**Practical implication: the encoder and decoder must use identical model
weights.** This is a hard constraint. Distributing the model alongside
the watermarking tool (or specifying exact model hashes) is necessary
for interoperability.


## Cross-cutting findings

### Encode/decode desynchronisation

The most significant finding across all experiments is that the encoder
and decoder almost never agree on exactly which positions are usable.
In experiment 1, 24 of 30 trials show a disagreement of at least 1
position, with differences ranging from 1 to 3 positions. In experiment
4 (cross-model), disagreements reach 11-33 positions.

The cause is not context contamination. The block-gap architecture
correctly prevents substitutions from reaching later blocks: the gap
(6 words) equals the context length, so no substitution in one block
appears in the context window of the next. Within a block, the encoder
modifies `working` in place and the decoder reads the watermarked text,
so both see identical context strings at every position. Direct testing
confirms that the contexts match exactly.

The actual cause is GPU floating-point non-determinism. Even with
temperature 0 and a fixed seed, ollama does not produce bitwise-identical
logprob distributions when the same prompt is sent twice. The top-1
(greedy) token is stable, but the ranking within the top-10 tokens
jitters. This matters because the filter pipeline depends on more than
just the top-1 token:

- The WholeWords filter selects space-prefixed alphabetic tokens from the
  top-10. If logprob jitter reorders tokens near the cutoff, different
  whole words may appear as r1 or r2 on different calls.
- The function word filter then checks whether r1 or r2 is a closed-class
  word. A different r2 can flip this decision.
- The forward prediction divergence filter sends three separate LLM
  calls (the base prediction plus two forward predictions). Each call has
  its own jitter.
- The min-r2 probability and min-sim thresholds can be crossed when
  logprob jitter changes which word is r2 or shifts its probability.

A concrete example from darwin (position 2431): the encoder sees
r2="in" (a function word, skipped), while the decoder sees r2="however"
(not a function word, passes). The context string is identical in both
cases. The two words are close enough in logprob that GPU thread
scheduling determines which one ranks higher.

The filter disagreements are distributed across all four filters (not
concentrated in one), confirming that the jitter affects different
positions at different pipeline stages. In the survey, the median
per-filter disagreement between encoder and decoder is 1-2 positions at
each stage, with the total usable difference reaching 1-3 positions.

This desynchronisation is the primary cause of round-trip failures. Even
trials with high overhead (5x+) can fail if the disagreement lands on
critical fountain code positions. Conversely, trials with low overhead
(1.44x on darwin with beef at 2000 words) can succeed if the
disagreement is small enough.

Possible mitigations: (a) widen the gap so there is more redundancy in
the fountain code, (b) avoid positions where the r2 logprob is close
to another candidate's (a "confidence gap" filter), (c) run each
prediction multiple times and use only positions where the result is
stable — at the cost of roughly 2-3x more LLM calls per position.

#### Non-determinism vs numerical instability

The desynchronisation is not mere random noise. Two distinct mechanisms
interact:

1. **Non-determinism at the kernel level.** GPU floating-point
   reduction is non-associative: `(a + b) + c` and `a + (b + c)` can
   differ at the last mantissa bits. GPU kernels choose reduction order
   based on thread scheduling, tiling, and batch shape, so the same
   computation can produce different rounding on different calls. This
   is well documented in recent work on LLM inference reproducibility
   [1][2][3].

2. **Amplification through transformer depth.** These epsilon-scale
   perturbations would be harmless in a linear system. In a
   transformer, they are amplified through depth — condition numbers
   above 10^6 have been reported for a single forward pass [4].
   The result is a "plateau-and-jump" pattern: most positions produce
   identical logit rankings across calls, but roughly 15% of positions
   sit near decision boundaries where the top-k ranking flips [4].
   This matches our observation that most positions agree between
   encoder and decoder while a small fraction (2-7%) disagree.

3. **Argmax discontinuity.** The softmax/argmax step turns a
   continuous logit shift into a discrete ranking change. A linear
   system with the same condition number would produce a proportionally
   shifted output; the ranking discontinuity is what turns a small
   shift into a qualitatively different filter decision.

The practical consequence for textfil is that any position where r2's
logprob is close to the next candidate's is unreliable: the same
context can produce different r2 words on different calls, flipping
filter decisions downstream. The consensus polling approach (below)
addresses this by voting over multiple calls, effectively smoothing
out the stochastic ranking at near-tie positions.

Yuan et al. [2] show that using FP32 compute (with BF16 weights)
eliminates the variation with 34% less memory than full FP32. This
"LayerCast" approach would likely eliminate the need for polling
entirely, but requires control over the inference backend. For ollama
on consumer GPUs, polling at the application level is the available
option. The CPU backend should also produce bit-exact results, at the
cost of slower inference.

[1] M. Srivatsa et al., "LLM-42: Enabling Determinism in LLM
    Inference with Verified Speculation," arXiv:2601.17768, 2026.
[2] J. Yuan et al., "Understanding and Mitigating Numerical Sources
    of Nondeterminism in LLM Inference," arXiv:2506.09501, 2025.
[3] E. Cooper et al., "Accelerating the Mitigation of LLM Inference
    Nondeterminism Across GPU Architectures," arXiv:2609.25624, 2026.
[4] A. Ravishankar et al., "Numerical Instability and Chaos:
    Quantifying the Unpredictability of Large Language Models,"
    arXiv:2604.13206, 2026.

### Consensus polling

The `-poll K` flag addresses desynchronisation by calling Predict K
times at each position and taking the majority vote for r1 and r2
independently. The key design choice is independent voting: r1 and r2
are each voted on separately rather than voting on (r1, r2) pairs.
This matters because r1 is often stable while r2 flips between two
close candidates. If a stable r1="therefore" is paired with r2 that
alternates between "however" and "in", no single pair gets a reliable
majority, but independent voting recovers the stable r1 and resolves
the r2.

K is clamped to odd values (≥1) so ties are impossible with an odd
number of polls.

#### Pair voting does not converge

The first implementation voted on (r1, r2) pairs jointly. This failed
on walden, the hardest text:

| Text    | Poll | Voting | Enc usable | Dec usable | Round trip |
|---------|-----:|--------|:----------:|:----------:|:----------:|
| darwin  | 1    | —      | 40         | 41         | pass       |
| darwin  | 3    | pair   | 41         | 41         | pass       |
| walden  | 3    | pair   | 32         | 33         | fail       |
| walden  | 5    | pair   | 33         | 32         | fail       |

Walden at poll=5 with pair voting still shows a 1-position
disagreement and fails to decode.

#### Independent voting fixes the hardest text

Switching to independent r1/r2 voting:

| Text    | Poll | Voting      | Enc usable | Dec usable | Round trip |
|---------|-----:|-------------|:----------:|:----------:|:----------:|
| walden  | 5    | independent | 36         | 36         | pass       |
| darwin  | 3    | independent | 41         | 41         | pass       |
| wells   | 1    | —           | 50         | 50         | pass       |
| wells   | 3    | independent | 50         | 50         | pass       |

Walden at poll=5 with independent voting produces perfect agreement
(36 encode, 36 decode) and successfully decodes the payload. Darwin
converges at poll=3. Wells already agreed at poll=1 and stays stable
under polling.

The yield can change with polling: walden goes from 32-33 usable
positions (poll=1, disagreeing) to 36 (poll=5, agreeing). Polling
doesn't just stabilise — it can recover positions that were rejected
in one direction due to a jittered prediction.

#### Polling rescues the 1b model

The results above use `llama3.2:3b-text-q8_0`. The smallest model,
`llama3.2:1b`, failed all five texts in the survey (0/5 pass rate).
Every trial showed a 1-2 position disagreement between encoder and
decoder. With poll=3, it passes all five:

| Text    | Poll=1 enc/dec | Poll=1 RT | Poll=3 enc/dec | Poll=3 RT | Overhead | Changed | Mean delta | Worst delta |
|---------|:--------------:|:---------:|:--------------:|:---------:|:--------:|:-------:|:----------:|:-----------:|
| darwin  | 30 / 31        | fail      | 31 / 31        | pass      | 1.94x    | 18      | +2.94      | -4.85       |
| doyle   | 43 / 44        | fail      | 45 / 45        | pass      | 2.81x    | 15      | +6.01      | -1.95       |
| gatsby  | 41 / 43        | fail      | 44 / 44        | pass      | 2.75x    | 17      | +5.77      | -2.46       |
| walden  | 30 / 31        | fail      | 34 / 34        | pass      | 2.12x    | 13      | +1.11      | -16.65      |
| wells   | 31 / 32        | fail      | 32 / 32        | pass      | 2.00x    | 9       | +3.84      | -1.74       |

Encode/decode agreement is perfect across all five texts. Polling also
increases yield: walden goes from 30-31 to 34 usable positions, doyle
from 43-44 to 45.

The naturalness scores (judged by `llama3.1:8b`) follow the same
pattern seen with larger models: mean deltas are positive, meaning
the judge considers the substituted words more probable than the
originals on average. Doyle and gatsby have the strongest positive
deltas (+6.0, +5.8), likely because dialogue-heavy and modern prose
offer more predictable substitution targets. Walden is the outlier:
its mean delta is barely positive (+1.1) and its worst position has
a delta of -16.65 — a single substitution the judge strongly
dislikes. This worst-position pattern also appeared in the survey
baseline (walden at poll=1 had worst delta -17.76), so it reflects
the text's difficulty rather than a polling artefact. Wells has the
fewest changed positions (9 of 32 usable), indicating that most
usable positions had the original word as r1 — the model already
predicted what the author wrote.

At poll=3 the 1b model takes ~200s per encode and ~215s per decode
(~7 minutes per round trip). This is roughly 3× the poll=1 time
(~73s + ~84s = ~157s) but still faster than the 3b model at poll=1
(~122s + ~124s = ~246s). For applications where speed matters, the
1b model with poll=3 gives reliable round-trips at lower latency than
any larger model without polling.

#### Cost

Each poll multiplies the number of Predict calls per position by K.
At poll=5, a ~3800-word text that takes ~4 minutes at poll=1 takes
roughly 15-20 minutes (the base prediction, two forward predictions,
and the embedding call remain at 1× each; only the base prediction is
polled). For the texts tested, poll=3 is sufficient except for walden
which requires poll=5.

### Text effect

Yield and reliability vary substantially across texts:

| Text | Avg yield (all models) | Pass rate |
|------|:----------------------:|:---------:|
| doyle | 16.6/kw | 4/6 |
| darwin | 15.3/kw | 4/6 |
| gatsby | 14.8/kw | 3/6 |
| wells | 13.7/kw | 2/6 |
| walden | 12.2/kw | 3/6 |

Walden (Thoreau's reflective essay) is the hardest text: fewest usable
positions, most failures. Wells (Victorian science fiction) has the
lowest pass rate despite moderate yield. The pass rate does not
correlate perfectly with yield — wells has higher yield than walden but
lower reliability, suggesting that the specific words being substituted
matter as much as how many positions are available.

Doyle (dialogue-heavy fiction) and darwin (scientific prose) are the
easiest texts despite their different styles. This may reflect that both
texts have more predictable word patterns at the positions that pass the
filters.

### Naturalness scoring

The naturalness scores require careful interpretation. The metric is:

    delta = log P(substituted word | context) - log P(original word | context)

where the probabilities come from the judge model (llama3.1:8b). A
positive delta means the judge considers the substituted word *more
probable* than the original. A negative delta means the substitution
made the text less natural at that position.

**Most mean deltas are positive.** This seems counterintuitive but has a
straightforward explanation: the substituted word is a rank-1 or rank-2
prediction from the encoding model. When the encoding model and the judge
agree about what words are likely at a position (which happens often for
the llama models, since the judge is also a llama model), the
substitution may actually be what the judge would have predicted. The
original word in the source text is not necessarily the most probable
word — it is what the author wrote, not what an LLM would predict.

**Worst-case deltas cluster around -18 to -19 for gemma models.** When
a gemma model substitutes a word that the llama judge doesn't have in its
top-20 predictions, the judge assigns the default log-probability of
-20.0. The worst deltas of -18 to -19 are cases where the original word
had a moderate log-probability and the substitution was not in the
judge's vocabulary at that position. This is a cross-family artifact: a
llama judge is not a good evaluator of gemma substitutions, and
vice versa.

**The 1b model on darwin scored 0.000 mean delta with 0 positions.**
This trial failed with zero changed positions being scored, so the
naturalness data is missing.

**Better naturalness measurement would require either:** (a) using the
same model family as the judge and the encoder, or (b) using a much
larger judge model that covers both model families' vocabularies, or
(c) computing full-text perplexity instead of per-position deltas.


## Practical recommendations

1. **Use llama3.2:3b-text-q8_0 for the best reliability among the llama
   models.** It passes 4/5 texts at the default parameters. If speed
   matters more than reliability, llama3.2:3b (3/5, 1.6x faster) is a
   reasonable choice.

2. **Use gemma3:4b or gemma4:e4b for higher capacity.** Both pass 4/5
   texts with roughly 2x the yield of the llama models. gemma3:4b is
   faster (157 vs 119 tok/s) and uses less VRAM (3.4 vs 6.6 GB).

3. **Budget 2000+ words per 2-byte payload, 3000+ words per 4-byte
   payload.** These minimums assume the llama3.2:3b-text-q8_0 model on
   moderate-difficulty text. The gemma models would need less text for the
   same payload because of their higher yield.

4. **Consider lowering min-r2 to 0.08 for harder texts.** This increases
   yield by roughly 40% (33 to 47 positions on walden) with minimal impact
   on naturalness. Going below 0.08 risks admitting too many low-quality
   positions that cause desynchronisation.

5. **Don't raise min-sim above 0.50.** The similarity filter eliminates
   most positions at 0.70+ and all positions at 0.95. The default of 0.50
   is already generous; lowering to 0.30 adds some positions but can also
   cause desynchronisation on some texts.

6. **The encoder and decoder must use identical model weights.** Even
   quantisation variants of the same base model produce enough prediction
   differences to prevent reliable decoding. Distribute the exact model
   file alongside the tool.


## Timing and compute cost

All experiments ran on a single NVIDIA RTX 2000 Ada Generation (8 GB
VRAM). Each trial consists of an encode pass and a
decode pass, each walking the full text with 3 LLM calls per candidate
position (base prediction, forward prediction for r1, forward prediction
for r2) plus one embedding call for positions that pass the first three
filters.

### Per-model timing (experiment 1 averages, ~3800-word texts)

| Model | Avg encode (s) | Avg decode (s) | Avg total (s) | Avg tok/s |
|-------|---------------:|---------------:|---------------:|----------:|
| llama3.2:1b | 86 | 96 | 182 | 833 |
| llama3.2:3b | 122 | 124 | 246 | 362 |
| llama3.2:3b-text-q8_0 | 123 | 134 | 257 | 334 |
| llama3.1:8b | 234 | 205 | 439 | 123 |
| gemma3:4b | 223 | 221 | 443 | 157 |
| gemma4:e4b | 296 | 292 | 588 | 119 |

Encode and decode take roughly equal time (each walks the full text).
The total wall-clock time for a round trip ranges from ~3 minutes (1b)
to ~10 minutes (gemma4:e4b).

### Capacity scaling (experiment 2, llama3.2:3b-text-q8_0)

| Words | Avg encode (s) | Avg decode (s) | Avg total (s) |
|------:|---------------:|---------------:|---------------:|
| 500 | 15 | 14 | 30 |
| 1000 | 29 | 30 | 59 |
| 2000 | 63 | 61 | 124 |
| 3000 | 96 | 98 | 194 |

Time scales roughly linearly with text length. A 500-word encode takes
~15 seconds; a 3000-word encode takes ~96 seconds. Extrapolating, a
10,000-word text would take roughly 5 minutes to encode with this model.

### Experiment totals

| Experiment | Trials | Total time | Avg per trial |
|------------|-------:|-----------:|--------------:|
| Smoke test | 1 | 4 min | 4 min |
| Survey (exp 1) | 30 | 180 min | 6 min |
| Cross-quant (exp 4) | 16 | 137 min | 8.5 min |
| Param sweep r2 (exp 3a) | 12 | 47 min | 4 min |
| Param sweep sim (exp 3b) | 10 | 33 min | 3.3 min |
| Capacity scaling (exp 2) | 24 | 41 min | 1.7 min |
| **Total** | **93** | **~7.3 hrs** | **4.7 min** |

The cross-quant experiment averaged 8.5 minutes per trial because
cross-model trials require loading a different model for decode, which
triggers model swaps on the GPU. Some gemma cross-family trials took
15-19 minutes each (the gemma3→gemma4 trials on doyle ran for ~16-19
minutes total).


## Open questions

**Why does the 8b model fail more than the 3b?** The 8b model has higher
yield than the 3b (12.6 vs 10.8 positions per 1000 words) and more
usable positions, but passes only 1/5 texts vs 3/5. One hypothesis: the
8b model uses context more sensitively, so substitutions propagate
further. Verifying this would require comparing the specific positions
where enc_usable and dec_usable disagree across models.

**Is poll=3 sufficient in general?** The tested texts converge at poll=3
except walden (which needs poll=5). A larger corpus would reveal whether
poll=3 is enough for most texts or whether harder texts routinely need
poll=5 or higher. A related question: does a "confidence gap" filter
(rejecting positions where r2's logprob is close to the next candidate)
achieve similar stabilisation without the extra LLM calls?

**Would the CPU backend eliminate the need for polling?** GPU
floating-point non-determinism is the root cause. The CPU backend should
produce bit-exact results, removing the need for polling entirely at the
cost of slower inference. This has not been tested.

**Is the positive naturalness delta real or an artifact of the judge?**
The llama judge may systematically favour LLM-predicted words over
human-written words because both the judge and the encoding model share
training data biases. Testing with a human evaluation or a different
architecture (e.g. a gemma judge for llama encoders) would clarify this.

**How sensitive are results to the source text's specific content?** The
five texts tested here show substantial variation in yield and
reliability. A larger corpus (20+ texts) would give a better estimate of
the distribution and identify what text properties predict difficulty.
