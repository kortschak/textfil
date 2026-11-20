// Copyright ©2026 Dan Kortschak. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The eval command runs textfil encode/decode trials across model
// configurations and collects metrics. Each trial is written as one
// JSON line to stdout; a summary table goes to stderr.
package main

import (
	"bytes"
	"encoding/hex"
	"encoding/json"
	"flag"
	"fmt"
	"os"
	"os/exec"
	"regexp"
	"strconv"
	"strings"
	"time"
	"unicode"

	"github.com/kortschak/textfil/internal/ollama"
)

func main() {
	url := flag.String("url", "http://localhost:11434", "ollama API base URL")
	textFile := flag.String("text", "", "input text file (required)")
	payload := flag.String("payload", "beef", "hex payload to encode")
	models := flag.String("models", "llama3.2:3b", "comma-separated model list")
	cross := flag.Bool("cross", false, "also test all encode/decode model pairs")
	judge := flag.String("judge", "", "judge model for naturalness scoring")
	embedModel := flag.String("embed-model", "nomic-embed-text", "embedding model")
	reps := flag.Int("n", 1, "repetitions per trial")
	bin := flag.String("textfil", "", "path to textfil binary (default: search PATH)")
	seed := flag.Uint64("seed", 42, "fountain coding seed")
	minR2 := flag.Float64("min-r2", 0.10, "minimum rank-2 probability")
	minSim := flag.Float64("min-sim", 0.50, "minimum r1/r2 embedding similarity")
	flag.Parse()

	if *textFile == "" {
		fmt.Fprintln(os.Stderr, "-text is required")
		os.Exit(1)
	}

	textfilBin := *bin
	if textfilBin == "" {
		p, err := exec.LookPath("textfil")
		if err != nil {
			fmt.Fprintln(os.Stderr, "textfil not found in PATH; use -textfil to specify")
			os.Exit(1)
		}
		textfilBin = p
	}

	modelList := splitModels(*models)

	textData, err := os.ReadFile(*textFile)
	if err != nil {
		fmt.Fprintf(os.Stderr, "reading text: %v\n", err)
		os.Exit(1)
	}
	origWords := strings.Fields(strings.TrimSpace(string(textData)))

	payloadBytes, err := hex.DecodeString(*payload)
	if err != nil {
		fmt.Fprintf(os.Stderr, "invalid payload hex: %v\n", err)
		os.Exit(1)
	}
	payloadBits := len(payloadBytes) * 8

	trials := buildTrials(modelList, *cross)

	cfg := trialConfig{
		bin:        textfilBin,
		url:        *url,
		textFile:   *textFile,
		origWords:  origWords,
		payload:    *payload,
		payloadBits: payloadBits,
		embedModel: *embedModel,
		seed:       *seed,
		minR2:      *minR2,
		minSim:     *minSim,
	}

	enc := json.NewEncoder(os.Stdout)
	var results []result

	for _, t := range trials {
		for rep := range *reps {
			fmt.Fprintf(os.Stderr, "  %s (rep %d): encoding...", t.name, rep)
			r := runTrial(cfg, t, rep)

			if *judge != "" && r.encText != "" && r.Error == "" {
				client := &ollama.Client{URL: *url, Model: *judge}
				n := scoreNaturalness(client, origWords, strings.Fields(r.encText), 6)
				r.JudgeModel = *judge
				r.MeanDelta = &n.mean
				r.WorstDelta = &n.worst
				r.NatPositions = &n.count
			}

			enc.Encode(r)
			results = append(results, r)
		}
	}

	printSummary(results)
}

func splitModels(s string) []string {
	parts := strings.Split(s, ",")
	out := make([]string, 0, len(parts))
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func buildTrials(models []string, cross bool) []trial {
	var trials []trial
	for _, m := range models {
		trials = append(trials, trial{name: m, encModel: m, decModel: m})
	}
	if cross && len(models) > 1 {
		for _, enc := range models {
			for _, dec := range models {
				if enc != dec {
					trials = append(trials, trial{
						name:     enc + "→" + dec,
						encModel: enc,
						decModel: dec,
					})
				}
			}
		}
	}
	return trials
}

type trial struct {
	name     string
	encModel string
	decModel string
}

type trialConfig struct {
	bin        string
	url        string
	textFile   string
	origWords  []string
	payload    string
	payloadBits int
	embedModel string
	seed       uint64
	minR2      float64
	minSim     float64
}

type result struct {
	Name     string `json:"name"`
	EncModel string `json:"encode_model"`
	DecModel string `json:"decode_model"`
	Payload  string `json:"payload"`
	Words    int    `json:"text_words"`
	Rep      int    `json:"repetition"`

	RoundTrip bool   `json:"round_trip"`
	Decoded   string `json:"decoded"`

	EncTotal   int     `json:"enc_total"`
	EncUsable  int     `json:"enc_usable"`
	EncBase    int     `json:"enc_skip_base"`
	EncFunc    int     `json:"enc_skip_func"`
	EncDiv     int     `json:"enc_skip_div"`
	EncSim     int     `json:"enc_skip_sim"`
	EncChanged int     `json:"enc_changed"`
	EncOverhead float64 `json:"enc_overhead"`

	DecTotal  int `json:"dec_total"`
	DecUsable int `json:"dec_usable"`
	DecBase   int `json:"dec_skip_base"`
	DecFunc   int `json:"dec_skip_func"`
	DecDiv    int `json:"dec_skip_div"`
	DecSim    int `json:"dec_skip_sim"`

	EncSeconds float64 `json:"encode_seconds"`
	DecSeconds float64 `json:"decode_seconds"`

	EncPredictCalls int     `json:"enc_predict_calls"`
	EncPromptTokens int     `json:"enc_prompt_tokens"`
	EncPromptEvalS  float64 `json:"enc_prompt_eval_seconds"`
	EncGenEvalS     float64 `json:"enc_gen_eval_seconds"`
	EncTokPerSec    float64 `json:"enc_prompt_tokens_per_sec"`

	DecPredictCalls int     `json:"dec_predict_calls"`
	DecPromptTokens int     `json:"dec_prompt_tokens"`
	DecPromptEvalS  float64 `json:"dec_prompt_eval_seconds"`
	DecGenEvalS     float64 `json:"dec_gen_eval_seconds"`
	DecTokPerSec    float64 `json:"dec_prompt_tokens_per_sec"`

	RecoveredBits *int `json:"recovered_bits,omitempty"`
	TotalBits     *int `json:"total_bits,omitempty"`

	JudgeModel   string   `json:"judge_model,omitempty"`
	MeanDelta    *float64 `json:"mean_logprob_delta,omitempty"`
	WorstDelta   *float64 `json:"worst_logprob_delta,omitempty"`
	NatPositions *int     `json:"nat_positions,omitempty"`

	Error string `json:"error,omitempty"`

	encText string
}

func runTrial(cfg trialConfig, t trial, rep int) result {
	r := result{
		Name:     t.name,
		EncModel: t.encModel,
		DecModel: t.decModel,
		Payload:  cfg.payload,
		Words:    len(cfg.origWords),
		Rep:      rep,
	}

	seedStr := strconv.FormatUint(cfg.seed, 10)
	r2Str := strconv.FormatFloat(cfg.minR2, 'f', -1, 64)
	simStr := strconv.FormatFloat(cfg.minSim, 'f', -1, 64)

	encStart := time.Now()
	encOut, encStderr, err := execTextfil(cfg.bin,
		"-payload", cfg.payload,
		"-file", cfg.textFile,
		"-model", t.encModel,
		"-url", cfg.url,
		"-embed-model", cfg.embedModel,
		"-seed", seedStr,
		"-min-r2", r2Str,
		"-min-sim", simStr,
		"-json", "-debug",
	)
	r.EncSeconds = time.Since(encStart).Seconds()

	if err != nil {
		var encErr struct{ Error string `json:"error"` }
		if json.Unmarshal(encOut, &encErr) == nil && encErr.Error != "" {
			r.Error = "encode: " + encErr.Error
		} else {
			r.Error = fmt.Sprintf("encode: %v", err)
		}
		fmt.Fprintf(os.Stderr, " FAIL (%s)\n", r.Error)
		return r
	}

	var encResult struct {
		Text      string `json:"text"`
		Bits      int    `json:"bits"`
		Positions int    `json:"positions"`
	}
	if err := json.Unmarshal(encOut, &encResult); err != nil {
		r.Error = fmt.Sprintf("parsing encode output: %v", err)
		fmt.Fprintf(os.Stderr, " FAIL\n")
		return r
	}
	r.encText = encResult.Text

	es := parseDebugStats(encStderr)
	r.EncTotal = es.total
	r.EncUsable = es.usable
	r.EncBase = es.skipBase
	r.EncFunc = es.skipFunc
	r.EncDiv = es.skipDiv
	r.EncSim = es.skipSim
	r.EncChanged = es.changed
	r.EncOverhead = es.overhead
	r.EncPredictCalls = es.predictCalls
	r.EncPromptTokens = es.promptTokens
	r.EncPromptEvalS = es.promptEvalS
	r.EncGenEvalS = es.genEvalS
	r.EncTokPerSec = es.tokPerSec

	fmt.Fprintf(os.Stderr, " decoding...")

	tmp, err := os.CreateTemp("", "eval-wm-*.txt")
	if err != nil {
		r.Error = fmt.Sprintf("temp file: %v", err)
		fmt.Fprintf(os.Stderr, " FAIL\n")
		return r
	}
	fmt.Fprint(tmp, encResult.Text)
	tmp.Close()
	defer os.Remove(tmp.Name())

	decStart := time.Now()
	decOut, decStderr, _ := execTextfil(cfg.bin,
		"-bits", strconv.Itoa(cfg.payloadBits),
		"-file", tmp.Name(),
		"-model", t.decModel,
		"-url", cfg.url,
		"-embed-model", cfg.embedModel,
		"-seed", seedStr,
		"-min-r2", r2Str,
		"-min-sim", simStr,
		"-json", "-debug",
	)
	r.DecSeconds = time.Since(decStart).Seconds()

	var decResult struct {
		Payload []byte `json:"payload"`
		Error   string `json:"error"`
	}
	if err := json.Unmarshal(decOut, &decResult); err == nil {
		if decResult.Payload != nil {
			decoded := hex.EncodeToString(decResult.Payload)
			r.Decoded = decoded
			r.RoundTrip = strings.EqualFold(decoded, cfg.payload)
		} else if decResult.Error != "" {
			r.Decoded = decResult.Error
			if m := reRecovery.FindStringSubmatch(decResult.Error); m != nil {
				recovered, _ := strconv.Atoi(m[1])
				total, _ := strconv.Atoi(m[2])
				r.RecoveredBits = &recovered
				r.TotalBits = &total
			}
		}
	}

	ds := parseDebugStats(decStderr)
	r.DecTotal = ds.total
	r.DecUsable = ds.usable
	r.DecBase = ds.skipBase
	r.DecFunc = ds.skipFunc
	r.DecDiv = ds.skipDiv
	r.DecSim = ds.skipSim
	r.DecPredictCalls = ds.predictCalls
	r.DecPromptTokens = ds.promptTokens
	r.DecPromptEvalS = ds.promptEvalS
	r.DecGenEvalS = ds.genEvalS
	r.DecTokPerSec = ds.tokPerSec

	if r.RoundTrip {
		fmt.Fprintf(os.Stderr, " OK (%.0fs + %.0fs)\n", r.EncSeconds, r.DecSeconds)
	} else {
		fmt.Fprintf(os.Stderr, " MISMATCH got=%s (%.0fs + %.0fs)\n", r.Decoded, r.EncSeconds, r.DecSeconds)
	}

	return r
}

func execTextfil(bin string, args ...string) (stdout, stderr []byte, err error) {
	cmd := exec.Command(bin, args...)
	var outBuf, errBuf bytes.Buffer
	cmd.Stdout = &outBuf
	cmd.Stderr = &errBuf
	err = cmd.Run()
	return outBuf.Bytes(), errBuf.Bytes(), err
}

type debugStats struct {
	total    int
	usable   int
	skipBase int
	skipFunc int
	skipDiv  int
	skipSim  int
	changed  int
	overhead float64

	predictCalls int
	promptTokens int
	promptEvalS  float64
	genEvalS     float64
	tokPerSec    float64
}

var (
	rePositions  = regexp.MustCompile(`Positions:\s+(\d+)\s+total,\s+(\d+)\s+usable`)
	reSkipped    = regexp.MustCompile(`Skipped:\s+(\d+)\s+base,\s+(\d+)\s+function-word,\s+(\d+)\s+divergent,\s+(\d+)\s+low-similarity`)
	reOverhead   = regexp.MustCompile(`Overhead:\s+([\d.]+)x`)
	reChanged    = regexp.MustCompile(`Changes:\s+(\d+)\s+words modified`)
	reTiming     = regexp.MustCompile(`Timing:\s+(\d+)\s+predict calls,\s+(\d+)\s+prompt tokens,\s+([\d.]+)s\s+prompt eval,\s+([\d.]+)s\s+generation`)
	reThroughput = regexp.MustCompile(`Throughput:\s+([\d.]+)\s+prompt tokens/s`)
	reRecovery   = regexp.MustCompile(`(\d+)/(\d+)\s+bits recovered`)
)

func parseDebugStats(stderr []byte) debugStats {
	s := string(stderr)
	var d debugStats
	if m := rePositions.FindStringSubmatch(s); m != nil {
		d.total, _ = strconv.Atoi(m[1])
		d.usable, _ = strconv.Atoi(m[2])
	}
	if m := reSkipped.FindStringSubmatch(s); m != nil {
		d.skipBase, _ = strconv.Atoi(m[1])
		d.skipFunc, _ = strconv.Atoi(m[2])
		d.skipDiv, _ = strconv.Atoi(m[3])
		d.skipSim, _ = strconv.Atoi(m[4])
	}
	if m := reOverhead.FindStringSubmatch(s); m != nil {
		d.overhead, _ = strconv.ParseFloat(m[1], 64)
	}
	if m := reChanged.FindStringSubmatch(s); m != nil {
		d.changed, _ = strconv.Atoi(m[1])
	}
	if m := reTiming.FindStringSubmatch(s); m != nil {
		d.predictCalls, _ = strconv.Atoi(m[1])
		d.promptTokens, _ = strconv.Atoi(m[2])
		d.promptEvalS, _ = strconv.ParseFloat(m[3], 64)
		d.genEvalS, _ = strconv.ParseFloat(m[4], 64)
	}
	if m := reThroughput.FindStringSubmatch(s); m != nil {
		d.tokPerSec, _ = strconv.ParseFloat(m[1], 64)
	}
	return d
}

type naturalness struct {
	mean  float64
	worst float64
	count int
}

// scoreNaturalness queries the judge model at each position where
// the watermarked text differs from the original, computing the
// log-probability delta of the substitution.
func scoreNaturalness(client *ollama.Client, orig, wm []string, ctxLen int) naturalness {
	n := min(len(orig), len(wm))
	var deltas []float64

	for i := range n {
		if strings.EqualFold(orig[i], wm[i]) {
			continue
		}

		start := max(0, i-ctxLen)
		ctx := strings.Join(wm[start:i], " ")

		pred, err := client.Predict(ctx, 20)
		if err != nil {
			continue
		}

		origWord := strings.ToLower(stripPunct(orig[i]))
		wmWord := strings.ToLower(stripPunct(wm[i]))

		origLP := findLogProb(pred.Top, origWord)
		wmLP := findLogProb(pred.Top, wmWord)

		deltas = append(deltas, wmLP-origLP)
	}

	if len(deltas) == 0 {
		return naturalness{}
	}

	sum := 0.0
	worst := 0.0
	for _, d := range deltas {
		sum += d
		if d < worst {
			worst = d
		}
	}
	return naturalness{
		mean:  sum / float64(len(deltas)),
		worst: worst,
		count: len(deltas),
	}
}

func findLogProb(tokens []ollama.Token, word string) float64 {
	for _, t := range ollama.WholeWords(tokens) {
		if strings.EqualFold(ollama.WordText(t), word) {
			return t.Logprob
		}
	}
	return -20.0
}

func stripPunct(s string) string {
	return strings.TrimRightFunc(s, func(r rune) bool {
		return unicode.IsPunct(r)
	})
}

func printSummary(results []result) {
	fmt.Fprintln(os.Stderr)
	fmt.Fprintf(os.Stderr, "%-30s  %3s  %6s  %6s  %7s  %7s  %7s  %8s",
		"trial", "RT", "usable", "chg", "ovhd", "enc(s)", "dec(s)", "tok/s")

	hasNat := false
	for _, r := range results {
		if r.MeanDelta != nil {
			hasNat = true
			break
		}
	}
	if hasNat {
		fmt.Fprintf(os.Stderr, "  %8s", "mean-Δ")
	}
	fmt.Fprintln(os.Stderr)
	fmt.Fprintln(os.Stderr, strings.Repeat("-", 88))

	for _, r := range results {
		rt := "✗"
		if r.RoundTrip {
			rt = "✓"
		}
		if r.Error != "" {
			rt = "ERR"
		}
		usable := fmt.Sprintf("%d/%d", r.EncUsable, r.DecUsable)
		if r.EncModel == r.DecModel {
			usable = strconv.Itoa(r.EncUsable)
		}
		ovhd := "-"
		if r.EncOverhead > 0 {
			ovhd = fmt.Sprintf("%.2fx", r.EncOverhead)
		}
		tps := "-"
		if r.EncTokPerSec > 0 {
			tps = fmt.Sprintf("%.0f", r.EncTokPerSec)
		}
		fmt.Fprintf(os.Stderr, "%-30s  %3s  %6s  %6d  %7s  %7.0f  %7.0f  %8s",
			r.Name, rt, usable, r.EncChanged, ovhd, r.EncSeconds, r.DecSeconds, tps)
		if hasNat {
			if r.MeanDelta != nil {
				fmt.Fprintf(os.Stderr, "  %8.3f", *r.MeanDelta)
			} else {
				fmt.Fprintf(os.Stderr, "  %8s", "-")
			}
		}
		fmt.Fprintln(os.Stderr)
	}

	// Yield summary across models.
	if len(results) > 1 {
		fmt.Fprintln(os.Stderr)
		fmt.Fprintln(os.Stderr, "Yield (usable positions per 1000 words):")
		seen := make(map[string]bool)
		for _, r := range results {
			if r.Error != "" || seen[r.Name] {
				continue
			}
			seen[r.Name] = true
			if r.Words > 0 {
				yield := float64(r.EncUsable) / float64(r.Words) * 1000
				fmt.Fprintf(os.Stderr, "  %-30s  %.1f positions/kword\n", r.Name, yield)
			}
		}
	}
}

