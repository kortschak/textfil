# `ᵗexᵀfil`

Encode binary data into existing text by substituting words with a local
LLM's alternative predictions. The stegged text reads naturally and the
payload can be extracted from the stegged text alone, without the
original.

## Approach

At each candidate word position, the program feeds the preceding context
to the LLM and gets ranked predictions. The top two whole-word
predictions (rank-1 and rank-2) form a bit state pair: rank-1 encodes a
0, rank-2 encodes a 1. If the desired bit doesn't match the word
already present, the word is swapped.

The decoder repeats the same walk over the stegged text, checks whether
each word matches rank-1 or rank-2 at that position, and recovers the
encoded bits.

A fountain code (LT code with systematic encoding) spreads the payload
across more positions than strictly needed, so the message survives
even when many positions are unusable. The same payload can be encoded
into multiple different documents; at decode time, symbols from all
documents are combined, so the payload can be recovered from any
sufficient subset.

### Filters

Not every word position is suitable, so each candidate passes through four
filters before it can carry a bit:

- Base probability: rank-2 must have probability >= 0.10 (otherwise the
  alternative is too unlikely to be a plausible substitute).
- Function word filter: closed-class words (articles, prepositions,
  pronouns, auxiliaries) are skipped; substituting "the" for "a" is
  conspicuous.
- Forward prediction divergence: after tentatively placing rank-1 vs
  rank-2, the program predicts the *next* word in each case. If the
  continuations differ, the substitution would alter downstream
  context, so the position is skipped.
- Embedding similarity: rank-1 and rank-2 are embedded with
  `nomic-embed-text` and their cosine similarity is checked (>= 0.50).
  This catches pairs that pass the other filters but aren't
  interchangeable.

Positions that fail any filter are skipped by both encoder and decoder
identically, keeping the fountain code's RNG synchronised.

### Block-sequential processing

To prevent substitutions from contaminating each other's context, words
are processed in blocks separated by gaps. Within a block, earlier
substitutions are visible to later positions (the encoder modifies the
working array in place). Across blocks, the gap ensures that no
position's context window overlaps a modified word from another block.

## Requirements

- [ollama](https://ollama.ai) running locally with two models pulled:
  - `llama3.2:3b` (or another model, via `-model`)
  - `nomic-embed-text` (or another embedding model, via `-embed-model`)

## Usage

### Encode

```
textfil -payload <hex> -file <input.text> > stegged.text
```

Reads the input text, encodes the hex payload into it, and writes the
stegged text to stdout. Progress is printed to stderr.

```
textfil -payload beef -file message.text > stegged.text
```

### Decode

```
textfil -bits <n> -file <stegged.text>
```

Extracts the payload from stegged text and prints it as hex to
stdout. `-bits` must match the payload size used during encoding
(payload bytes * 8).

```
textfil -bits 16 -file stegged.text
```

### Ensemble decode

Encode the same payload into several different documents, then decode
from all of them at once. Each document produces a different fountain
symbol stream(seeded by a hash of its first `-ctx` words), so symbols
from all documents combine to improve recovery. The encoder reports how
many symbols each document contributed (e.g. `encoded 13/16 symbols`);
you need roughly 1x to 1.5x the payload bits in total across all
documents.

```
textfil -payload beef -file doc1.text > doc1-wm.text
textfil -payload beef -file doc2.text > doc2-wm.text
textfil -payload beef -file doc3.text > doc3-wm.text

textfil -bits 16 doc1-wm.text doc2-wm.text doc3-wm.text
```

The decoder accepts multiple files as positional arguments (or a single
file via `-file`). A document that individually lacks capacity to carry
the full payload can still contribute symbols; if the combined set is
sufficient, the payload is recovered.

Documents with identical opening words (the first `-ctx` words) would
produce identical symbol streams, contributing no additional
information.

### Debug output

Add `-debug` to either mode to print a position table and statistics to
stderr.

## Capacity

Capacity depends on the input text. Longer texts with more content words
yield more usable positions. In testing, a ~3000-word literary text
produced ~23 usable positions after all four filters, enough to encode
a 2-byte payload with fountain code overhead.

Texts heavy in dialogue, proper nouns, or unusual vocabulary will have
fewer usable positions. Expository prose with common vocabulary works
best.
