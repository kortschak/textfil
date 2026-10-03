// Copyright ©2026 Dan Kortschak. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The textfil command encodes binary payloads into natural text via LLM
// prediction-ranked word substitution, and decodes them back. Multiple
// documents can carry portions of the same payload using fountain codes.
package main

import (
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"hash/fnv"
	"io"
	"math"
	"math/rand/v2"
	"os"
	"strings"
	"unicode"

	"github.com/kortschak/textfil/internal/fountain"
	"github.com/kortschak/textfil/internal/ollama"
)

func main() {
	bits := flag.Int("bits", 0, "payload size in bits (runs decode mode if non-zero)")
	seed := flag.Uint64("seed", 42, "base seed for fountain coding")
	file := flag.String("file", "", "input text file (encode mode, or single-file decode)")
	payloadHex := flag.String("payload", "", "hex payload to encode (runs encode mode if non-empty)")
	debug := flag.Bool("debug", false, "emit position table and stats to stderr")

	cfg := config{client: &ollama.Client{}}
	flag.StringVar(&cfg.client.Model, "model", "llama3.2:3b", "ollama model name")
	flag.StringVar(&cfg.client.URL, "url", "http://localhost:11434", "ollama API base URL")
	flag.IntVar(&cfg.blockSize, "block", 8, "positions per block")
	flag.IntVar(&cfg.gapSize, "gap", 6, "gap words between blocks")
	flag.IntVar(&cfg.ctxLen, "ctx", 6, "context length in words")
	flag.Float64Var(&cfg.minR2, "min-r2", 0.10, "minimum rank-2 probability")
	flag.Float64Var(&cfg.minSim, "min-sim", 0.50, "minimum r1/r2 embedding similarity")
	flag.StringVar(&cfg.embedModel, "embed-model", "nomic-embed-text", "embedding model")
	flag.BoolVar(&cfg.jsonMode, "json", false, "emit output as JSON to stdout")
	flag.Parse()

	if cfg.gapSize < cfg.ctxLen {
		cfg.gapSize = cfg.ctxLen
	}

	if *bits != 0 {
		if *payloadHex != "" {
			fmt.Fprintln(os.Stderr, "-bits and -payload flags cannot be used together")
			os.Exit(1)
		}

		files := flag.Args()
		if *file != "" {
			files = append([]string{*file}, files...)
		}
		if len(files) == 0 {
			files = []string{""}
		}
		decode(cfg, files, *bits, *seed, *debug)
		return
	}

	if *payloadHex == "" {
		fmt.Fprintln(os.Stderr, "-payload is required in encode mode")
		os.Exit(1)
	}
	payload, err := hex.DecodeString(*payloadHex)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid hex payload: %v\n", err)
		os.Exit(1)
	}
	text := readInput(*file)
	words := strings.Fields(text)
	encode(cfg, words, payload, *seed, *debug)
}

// encode walks the text, substitutes words to embed the payload, and
// writes the watermarked text to stdout.
func encode(cfg config, words []string, payload []byte, baseSeed uint64, debug bool) {
	working := make([]string, len(words))
	copy(working, words)
	payloadBits := len(payload) * 8

	seed := docSeed(baseSeed, words, cfg.ctxLen)
	enc := fountain.NewEncoder(payload, rand.NewPCG(baseSeed, seed))

	positions := walkText(cfg, working, func(p *pos) {
		sym := enc.Next()
		p.bit = sym.Value
		bare := stripPunct(working[p.wordIdx])
		isR1 := strings.EqualFold(bare, p.r1)
		switch {
		case p.bit == 0 && isR1:
			p.sub = working[p.wordIdx]
		case p.bit == 0 && !isR1:
			p.sub = transferPunct(working[p.wordIdx], p.r1)
			p.changed = true
		case p.bit == 1 && isR1:
			p.sub = transferPunct(working[p.wordIdx], p.r2)
			p.changed = true
		case p.bit == 1 && !isR1:
			p.sub = working[p.wordIdx]
		}
		working[p.wordIdx] = p.sub
	})

	usable := enc.N()

	if debug {
		printDebug(positions, usable, payloadBits)
	}

	if cfg.jsonMode {
		json.NewEncoder(os.Stdout).Encode(struct {
			Text      string `json:"text"`
			Bits      int    `json:"bits"`
			Positions int    `json:"positions"`
		}{strings.Join(working, " "), payloadBits, usable})
	} else {
		fmt.Fprintf(os.Stderr, "encoded %d/%d symbols\n", usable, payloadBits)
		fmt.Println(strings.Join(working, " "))
	}
}

// decode walks one or more watermarked documents, recovers fountain
// symbols from each, and reconstructs the payload.
func decode(cfg config, files []string, payloadBits int, baseSeed uint64, debug bool) {
	dec := fountain.NewDecoder(payloadBits)
	var allPositions []pos

	for i, path := range files {
		text := readInput(path)
		words := strings.Fields(text)

		if path == "" {
			path = "stdin"
		}
		if len(files) > 1 {
			fmt.Fprintf(os.Stderr, "  [%d/%d] %s (%d words)\n", i+1, len(files), path, len(words))
		}

		seed := docSeed(baseSeed, words, cfg.ctxLen)
		stream := fountain.NewStream(payloadBits, rand.NewPCG(baseSeed, seed))

		positions := walkText(cfg, words, func(p *pos) {
			p.sub = words[p.wordIdx]
			bare := stripPunct(p.sub)
			if strings.EqualFold(bare, p.r1) {
				p.bit = 0
			} else {
				p.bit = 1
			}
			dec.Add(stream.Next(p.bit))
		})

		allPositions = append(allPositions, positions...)
	}

	result := dec.Finish()

	if debug {
		printDebug(allPositions, dec.N(), payloadBits)
	}

	if cfg.jsonMode {
		if result.OK {
			json.NewEncoder(os.Stdout).Encode(struct {
				Payload []byte `json:"payload"`
			}{result.Payload})
		} else {
			json.NewEncoder(os.Stdout).Encode(struct {
				Error string `json:"error"`
			}{fmt.Sprintf("decode failed: %d/%d bits recovered", result.Recovered, result.Total)})
			os.Exit(1)
		}
	} else {
		if !result.OK {
			fmt.Fprintf(os.Stderr, "decode failed: %d/%d bits recovered\n", result.Recovered, result.Total)
			os.Exit(1)
		}
		fmt.Println(hex.EncodeToString(result.Payload))
	}
}

// readInput reads text from path, or stdin if path is empty.
func readInput(path string) string {
	var data []byte
	var err error
	if path != "" {
		data, err = os.ReadFile(path)
	} else {
		data, err = io.ReadAll(os.Stdin)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "reading input: %v\n", err)
		os.Exit(1)
	}
	text := strings.TrimSpace(string(data))
	if text == "" {
		fmt.Fprintln(os.Stderr, "empty input")
		os.Exit(1)
	}
	return text
}

// docSeed derives a per-document fountain seed by hashing the first
// ctxLen words and XORing with baseSeed. These words precede the first
// block, so they are never modified by encoding.
func docSeed(baseSeed uint64, words []string, ctxLen int) uint64 {
	n := min(ctxLen, len(words))
	h := fnv.New64a()
	for _, w := range words[:n] {
		h.Write([]byte(strings.ToLower(stripPunct(w))))
	}
	return baseSeed ^ h.Sum64()
}

func transferPunct(original, replacement string) string {
	bare := stripPunct(original)
	punct := original[len(bare):]
	return replacement + punct
}

func stripPunct(s string) string {
	return strings.TrimRightFunc(s, func(r rune) bool {
		return unicode.IsPunct(r)
	})
}

// walkText iterates over candidate word positions in block-sequential
// order, applying the four-filter pipeline. Positions that pass all
// filters are handed to handle; skipped positions are recorded with
// their skip reason. The working array may be modified by handle
// (the encoder substitutes words in place).
func walkText(cfg config, words []string, handle func(*pos)) []pos {
	var positions []pos
	var scanned int
	blockStart := cfg.ctxLen

	for blockStart < len(words) {
		blockEnd := min(blockStart+cfg.blockSize, len(words))

		for i := blockStart; i < blockEnd; i++ {
			scanned++
			if scanned%20 == 0 {
				fmt.Fprintf(os.Stderr, "\r  word %d/%d", i+1, len(words))
			}

			ctxStart := max(i-cfg.ctxLen, 0)
			ctx := strings.Join(words[ctxStart:i], " ")

			pred, err := cfg.client.Predict(ctx, 10)
			if err != nil {
				exitErr(cfg.jsonMode, "predict at word %d: %v", i, err)
			}

			ww := ollama.WholeWords(pred.Top)
			p := pos{wordIdx: i, context: ctx, orig: words[i]}
			if len(ww) >= 1 {
				p.r1 = ollama.WordText(ww[0])
				p.r1prob = math.Exp(ww[0].Logprob)
			}
			if len(ww) >= 2 {
				p.r2 = ollama.WordText(ww[1])
				p.r2prob = math.Exp(ww[1].Logprob)
			}

			if skip, reason := shouldSkip(p, cfg.minR2); skip {
				p.skipped = true
				p.skipWhy = reason
				p.sub = words[i]
				positions = append(positions, p)
				continue
			}

			next1, err := predictNextWord(cfg.client, ctx+" "+p.r1)
			if err != nil {
				exitErr(cfg.jsonMode, "forward predict r1 at %d: %v", i, err)
			}
			next2, err := predictNextWord(cfg.client, ctx+" "+p.r2)
			if err != nil {
				exitErr(cfg.jsonMode, "forward predict r2 at %d: %v", i, err)
			}
			if !strings.EqualFold(next1, next2) {
				p.skipped = true
				p.skipWhy = fmt.Sprintf("divergent: %s→%s / %s→%s", p.r1, next1, p.r2, next2)
				p.sub = words[i]
				positions = append(positions, p)
				continue
			}

			embs, err := cfg.client.Embed(cfg.embedModel, []string{p.r1, p.r2})
			if err != nil {
				exitErr(cfg.jsonMode, "embed at %d: %v", i, err)
			}
			p.sim = ollama.CosineSimilarity(embs[0], embs[1])
			if p.sim < cfg.minSim {
				p.skipped = true
				p.skipWhy = fmt.Sprintf("low similarity %.4f < %.2f (%s/%s)", p.sim, cfg.minSim, p.r1, p.r2)
				p.sub = words[i]
				positions = append(positions, p)
				continue
			}

			handle(&p)
			positions = append(positions, p)
		}

		blockStart = blockEnd + cfg.gapSize
	}
	fmt.Fprintf(os.Stderr, "\r\033[2K")

	return positions
}

// config holds the parameters for the watermarking pipeline.
type config struct {
	client     *ollama.Client
	blockSize  int
	gapSize    int
	ctxLen     int
	minR2      float64
	minSim     float64
	embedModel string
	jsonMode   bool
}

// shouldSkip reports whether a position should be excluded from
// encoding, and the reason why.
func shouldSkip(p pos, minR2Prob float64) (bool, string) {
	if p.r1 == "" {
		return true, "no predictions"
	}
	if p.r2 == "" {
		return true, "no rank-2"
	}
	if p.r2prob < minR2Prob {
		return true, fmt.Sprintf("r2 prob %.4f < %.2f", p.r2prob, minR2Prob)
	}
	if strings.EqualFold(p.r1, p.r2) {
		return true, "r1/r2 case collision"
	}
	if closedClass[strings.ToLower(p.r1)] || closedClass[strings.ToLower(p.r2)] {
		return true, "function word"
	}
	return false, ""
}

// predictNextWord returns the top whole-word prediction for prompt.
func predictNextWord(c *ollama.Client, prompt string) (string, error) {
	pred, err := c.Predict(prompt, 10)
	if err != nil {
		return "", err
	}
	ww := ollama.WholeWords(pred.Top)
	if len(ww) == 0 {
		return "", nil
	}
	return ollama.WordText(ww[0]), nil
}

// printDebug writes the position table and summary stats to stderr.
func printDebug(positions []pos, usable, payloadBits int) {
	w := os.Stderr
	fmt.Fprintf(w, "%-4s  %-30s  %-12s  %-6s  %-6s  %-3s  %-12s  %s\n",
		"pos", "context", "word", "r1", "r2", "bit", "sub", "note")
	fmt.Fprintln(w, strings.Repeat("-", 100))

	for _, p := range positions {
		ctx := p.context
		if len(ctx) > 30 {
			ctx = ctx[:27] + "..."
		}
		note := ""
		if p.skipped {
			note = p.skipWhy
		} else if p.changed {
			note = "changed"
		}
		bit := "-"
		if !p.skipped {
			bit = fmt.Sprintf("%d", p.bit)
		}
		fmt.Fprintf(w, "%-4d  %-30s  %-12s  %-6s  %-6s  %-3s  %-12s  %s\n",
			p.wordIdx, ctx, p.orig,
			truncate(p.r1, 6), truncate(p.r2, 6),
			bit, p.sub, note)
	}

	var skippedBase, skippedFunc, skippedDiv, skippedSim, changed int
	for _, p := range positions {
		if p.skipped {
			switch {
			case strings.HasPrefix(p.skipWhy, "low similarity"):
				skippedSim++
			case strings.HasPrefix(p.skipWhy, "divergent"):
				skippedDiv++
			case p.skipWhy == "function word":
				skippedFunc++
			default:
				skippedBase++
			}
		}
		if p.changed {
			changed++
		}
	}

	fmt.Fprintln(w)
	fmt.Fprintf(w, "Positions: %d total, %d usable\n", len(positions), usable)
	fmt.Fprintf(w, "Skipped: %d base, %d function-word, %d divergent, %d low-similarity\n",
		skippedBase, skippedFunc, skippedDiv, skippedSim)
	if payloadBits > 0 && usable > 0 {
		fmt.Fprintf(w, "Overhead: %.2fx (%d symbols / %d payload bits)\n",
			float64(usable)/float64(payloadBits), usable, payloadBits)
	}
	if changed > 0 {
		fmt.Fprintf(w, "Changes: %d words modified\n", changed)
	}
}

// pos records the state of one candidate word position.
type pos struct {
	wordIdx int
	context string
	orig    string
	r1, r2  string
	r1prob  float64
	r2prob  float64
	sim     float64
	bit     int
	sub     string
	changed bool
	skipped bool
	skipWhy string
}

func truncate(s string, n int) string {
	if len(s) > n {
		return s[:n]
	}
	return s
}

// exitErr prints an error and exits. In JSON mode the error is written
// to stdout as {"error":"..."}; otherwise it goes to stderr.
func exitErr(jsonMode bool, format string, args ...any) {
	msg := fmt.Sprintf(format, args...)
	if jsonMode {
		json.NewEncoder(os.Stdout).Encode(struct {
			Error string `json:"error"`
		}{msg})
	} else {
		fmt.Fprintln(os.Stderr, msg)
	}
	os.Exit(1)
}

// closedClass lists function words that should not be substituted.
var closedClass = map[string]bool{
	"a": true, "an": true, "the": true,
	"in": true, "on": true, "at": true, "to": true, "for": true,
	"of": true, "by": true, "with": true, "from": true, "into": true,
	"and": true, "or": true, "but": true, "nor": true, "yet": true,
	"so": true, "as": true, "if": true, "then": true, "than": true,
	"is": true, "am": true, "are": true, "was": true, "were": true,
	"be": true, "been": true, "being": true,
	"has": true, "have": true, "had": true, "having": true,
	"do": true, "does": true, "did": true,
	"will": true, "would": true, "shall": true, "should": true,
	"can": true, "could": true, "may": true, "might": true, "must": true,
	"it": true, "he": true, "she": true, "we": true, "they": true,
	"i": true, "me": true, "him": true, "her": true, "us": true, "them": true,
	"my": true, "his": true, "its": true, "our": true, "your": true, "their": true,
	"this": true, "that": true, "these": true, "those": true,
	"who": true, "whom": true, "whose": true, "which": true, "what": true,
	"not": true, "no": true, "some": true, "any": true, "each": true,
	"all": true, "both": true, "few": true, "more": true, "most": true,
	"other": true, "such": true, "only": true, "very": true,
}
