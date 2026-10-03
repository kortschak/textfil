// Copyright ©2026 Dan Kortschak. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package fountain implements LT (Luby Transform) fountain codes for
// rate-less erasure coding of bit payloads.
package fountain

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Symbol is one LT-encoded bit with its source mapping.
type Symbol struct {
	Value   int
	Sources []int
}

// Result holds the outcome of LT decoding.
type Result struct {
	Payload   []byte
	Recovered int
	Total     int
	OK        bool
}

// Encoder generates LT-encoded symbols from a payload. The first k
// symbols carry individual source bits in a seed-dependent permuted
// order. Subsequent symbols are XOR combinations with degrees drawn
// from the Robust Soliton distribution.
type Encoder struct {
	bits []int
	k    int
	perm []int
	rnd  *rand.Rand
	n    int
}

// NewEncoder creates an encoder for payload using src to generate the
// systematic permutation and non-systematic degree/source selections.
func NewEncoder(payload []byte, src rand.Source) *Encoder {
	k := len(payload) * 8
	rnd := rand.New(src)
	perm := rnd.Perm(k)
	return &Encoder{
		bits: unpack(payload),
		k:    k,
		perm: perm,
		rnd:  rnd,
	}
}

// N returns the number of symbols generated so far.
func (e *Encoder) N() int { return e.n }

// Next returns the next encoded symbol.
func (e *Encoder) Next() Symbol {
	sources := e.sources()
	val := 0
	for _, s := range sources {
		val ^= e.bits[s]
	}
	e.n++
	return Symbol{Value: val, Sources: sources}
}

func (e *Encoder) sources() []int {
	if e.n < e.k {
		return []int{e.perm[e.n]}
	}
	return choose(e.k, degree(e.k, e.rnd), e.rnd)
}

// Stream regenerates the source mapping for one document's symbol
// sequence. Each call to Next advances the RNG identically to the
// encoder, returning a Symbol with the caller-supplied bit value and
// the reconstructed sources.
type Stream struct {
	k    int
	perm []int
	rnd  *rand.Rand
	n    int
}

// NewStream creates a stream that mirrors an encoder with the same
// payloadBits and src seed. The caller supplies each observed bit;
// the stream reconstructs the corresponding source mapping.
func NewStream(payloadBits int, src rand.Source) *Stream {
	rnd := rand.New(src)
	perm := rnd.Perm(payloadBits)
	return &Stream{
		k:    payloadBits,
		perm: perm,
		rnd:  rnd,
	}
}

// Next returns a symbol pairing bit with the source mapping for the
// current position, then advances the stream.
func (s *Stream) Next(bit int) Symbol {
	var sources []int
	if s.n < s.k {
		sources = []int{s.perm[s.n]}
	} else {
		sources = choose(s.k, degree(s.k, s.rnd), s.rnd)
	}
	s.n++
	return Symbol{Value: bit, Sources: sources}
}

// Decoder collects symbols from any number of streams and recovers
// the payload via peeling.
type Decoder struct {
	k       int
	symbols []Symbol
}

// NewDecoder creates a decoder expecting a payload of payloadBits bits.
func NewDecoder(payloadBits int) *Decoder {
	return &Decoder{k: payloadBits}
}

// Add records a received symbol.
func (d *Decoder) Add(sym Symbol) {
	d.symbols = append(d.symbols, sym)
}

// N returns the number of symbols collected so far.
func (d *Decoder) N() int { return len(d.symbols) }

// Finish runs the peeling decoder and returns the result.
func (d *Decoder) Finish() Result {
	decoded := make([]int, d.k)
	known := make([]bool, d.k)

	type active struct {
		value   int
		sources map[int]bool
	}
	work := make([]active, len(d.symbols))
	for i, s := range d.symbols {
		m := make(map[int]bool, len(s.Sources))
		for _, src := range s.Sources {
			m[src] = true
		}
		work[i] = active{value: s.Value, sources: m}
	}

	for {
		progress := false
		for i := range work {
			for src := range work[i].sources {
				if known[src] {
					work[i].value ^= decoded[src]
					delete(work[i].sources, src)
					progress = true
				}
			}
			if len(work[i].sources) == 1 {
				for src := range work[i].sources {
					decoded[src] = work[i].value
					known[src] = true
					delete(work[i].sources, src)
					progress = true
				}
			}
		}
		if !progress {
			break
		}
	}

	rec := 0
	for _, v := range known {
		if v {
			rec++
		}
	}
	return Result{
		Payload:   pack(decoded),
		Recovered: rec,
		Total:     d.k,
		OK:        rec == d.k,
	}
}

func unpack(data []byte) []int {
	bits := make([]int, len(data)*8)
	for i, b := range data {
		for j := range 8 {
			bits[i*8+j] = int((b >> uint(7-j)) & 1)
		}
	}
	return bits
}

func pack(bits []int) []byte {
	n := (len(bits) + 7) / 8
	out := make([]byte, n)
	for i, b := range bits {
		if b == 1 {
			out[i/8] |= 1 << uint(7-(i%8))
		}
	}
	return out
}

// degree samples a degree from the Robust Soliton distribution.
func degree(k int, rnd *rand.Rand) int {
	if k <= 1 {
		return 1
	}
	c := 0.2
	delta := 0.05

	s := c * math.Log(float64(k)/delta) * math.Sqrt(float64(k))
	if s < 1 {
		s = 1
	}

	probs := make([]float64, k)
	probs[0] = 1.0 / float64(k)
	for d := 2; d <= k; d++ {
		probs[d-1] = 1.0 / float64(d*(d-1))
	}

	kOverS := max(1, min(k, int(math.Round(float64(k)/s))))
	for d := 1; d < kOverS; d++ {
		probs[d-1] += s / (float64(d) * float64(k))
	}
	if kOverS <= k {
		probs[kOverS-1] += s * math.Log(s/delta) / float64(k)
	}

	total := 0.0
	for _, p := range probs {
		total += p
	}

	r := rnd.Float64() * total
	cum := 0.0
	for d := 1; d <= k; d++ {
		cum += probs[d-1]
		if r <= cum {
			return d
		}
	}
	return 1
}

// choose picks d distinct source indices from [0, k) uniformly at
// random and returns them sorted.
func choose(k, d int, rnd *rand.Rand) []int {
	if d >= k {
		s := make([]int, k)
		for i := range s {
			s[i] = i
		}
		return s
	}
	result := rnd.Perm(k)[:d]
	sort.Ints(result)
	return result
}
