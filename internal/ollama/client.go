// Copyright ©2026 Dan Kortschak. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package ollama provides a client for the ollama generate and embed APIs.
package ollama

import (
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"math"
	"net/http"
	"strings"
	"unicode"
)

// Client connects to an ollama instance.
type Client struct {
	URL   string
	Model string
}

// Prediction holds the generated token and the top-k alternatives
// returned by the generate API.
type Prediction struct {
	Generated Token
	Top       []Token
}

// Token is a single token with its log-probability.
type Token struct {
	Text    string
	Logprob float64
}

// Predict generates one token from prompt and returns the top-k
// alternatives with log-probabilities. Temperature is fixed at zero
// for deterministic output.
func (c *Client) Predict(prompt string, topK int) (Prediction, error) {
	type generateRequest struct {
		Model      string         `json:"model"`
		Prompt     string         `json:"prompt"`
		Raw        bool           `json:"raw"`
		Stream     bool           `json:"stream"`
		Options    map[string]any `json:"options"`
		Logprobs   bool           `json:"logprobs"`
		TopLogprob int            `json:"top_logprobs"`
	}

	req := generateRequest{
		Model:  c.Model,
		Prompt: prompt,
		Raw:    true,
		Stream: false,
		Options: map[string]any{
			"num_predict": 1,

			// Make generation fully deterministic.
			"temperature": 0, "seed": 1,
		},
		Logprobs:   true,
		TopLogprob: topK,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return Prediction{}, fmt.Errorf("marshalling request: %w", err)
	}

	resp, err := http.Post(c.URL+"/api/generate", "application/json", bytes.NewReader(body))
	if err != nil {
		return Prediction{}, fmt.Errorf("posting to ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return Prediction{}, fmt.Errorf("ollama returned %d: %s", resp.StatusCode, b)
	}

	var gen generateResponse
	err = json.NewDecoder(resp.Body).Decode(&gen)
	if err != nil {
		return Prediction{}, fmt.Errorf("decoding response: %w", err)
	}

	var pred Prediction
	if len(gen.Logprobs) > 0 {
		entry := gen.Logprobs[0]
		pred.Generated = Token{Text: entry.Token, Logprob: entry.Logprob}
		for _, alt := range entry.TopLogprob {
			pred.Top = append(pred.Top, Token{Text: alt.Token, Logprob: alt.Logprob})
		}
	}
	return pred, nil
}

type generateResponse struct {
	Response string         `json:"response"`
	Logprobs []tokenLogprob `json:"logprobs"`
}

type tokenLogprob struct {
	Token      string         `json:"token"`
	Logprob    float64        `json:"logprob"`
	TopLogprob []tokenLogprob `json:"top_logprobs"`
}

// Embed returns embedding vectors for the given texts using the
// specified model.
func (c *Client) Embed(model string, texts []string) ([][]float64, error) {
	type embedRequest struct {
		Model string   `json:"model"`
		Input []string `json:"input"`
	}
	req := embedRequest{Model: model, Input: texts}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, fmt.Errorf("marshalling embed request: %w", err)
	}

	resp, err := http.Post(c.URL+"/api/embed", "application/json", bytes.NewReader(body))
	if err != nil {
		return nil, fmt.Errorf("posting embed to ollama: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		return nil, fmt.Errorf("ollama embed returned %d: %s", resp.StatusCode, b)
	}

	var emb struct {
		Embeddings [][]float64 `json:"embeddings"`
	}
	err = json.NewDecoder(resp.Body).Decode(&emb)
	if err != nil {
		return nil, fmt.Errorf("decoding embed response: %w", err)
	}
	return emb.Embeddings, nil
}

// WholeWords filters tokens to those that represent complete words.
// A whole word is a space-prefixed token containing at least one letter
// and no internal whitespace.
func WholeWords(tokens []Token) []Token {
	var out []Token
	for _, t := range tokens {
		if IsWholeWord(t.Text) {
			out = append(out, t)
		}
	}
	return out
}

// IsWholeWord reports whether token is a space-prefixed complete word
// containing at least one letter.
func IsWholeWord(token string) bool {
	trimmed := strings.TrimLeft(token, " ")
	if len(trimmed) == 0 {
		return false
	}
	if strings.ContainsAny(trimmed, " \t\n\r") {
		return false
	}
	if !strings.HasPrefix(token, " ") {
		return false
	}
	return strings.ContainsFunc(trimmed, unicode.IsLetter)
}

// WordText returns the text of a whole-word token with its leading
// space removed.
func WordText(t Token) string {
	return strings.TrimLeft(t.Text, " ")
}

// CosineSimilarity returns the cosine similarity of two equal-length
// vectors. It returns 0 for empty or mismatched inputs.
func CosineSimilarity(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}
	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 0
	}
	return dot / (math.Sqrt(normA) * math.Sqrt(normB))
}
