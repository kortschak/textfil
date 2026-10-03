// Copyright ©2026 Dan Kortschak. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fountain

import (
	"bytes"
	"math"
	"math/rand/v2"
	"testing"
)

func TestRoundTrip(t *testing.T) {
	for _, test := range []struct {
		name     string
		payload  []byte
		seed     uint64
		overhead float64
	}{
		{name: "two_bytes", payload: []byte{0xbe, 0xef}, seed: 42, overhead: 1},
		{name: "two_bytes_2x", payload: []byte{0xbe, 0xef}, seed: 42, overhead: 2},
		{name: "four_bytes", payload: []byte{0xde, 0xad, 0xbe, 0xef}, seed: 99, overhead: 1.5},
		{name: "single_byte", payload: []byte{0xff}, seed: 1, overhead: 1},
		{name: "eight_bytes", payload: []byte{0x01, 0x23, 0x45, 0x67, 0x89, 0xab, 0xcd, 0xef}, seed: 7, overhead: 1.5},
	} {
		t.Run(test.name, func(t *testing.T) {
			k := len(test.payload) * 8
			n := int(math.Ceil(float64(k) * test.overhead))

			enc := NewEncoder(test.payload, rand.NewPCG(test.seed, test.seed))
			symbols := make([]Symbol, n)
			for i := range symbols {
				symbols[i] = enc.Next()
			}

			stream := NewStream(k, rand.NewPCG(test.seed, test.seed))
			dec := NewDecoder(k)
			for _, sym := range symbols {
				dec.Add(stream.Next(sym.Value))
			}

			result := dec.Finish()
			if !result.OK {
				t.Fatalf("decode failed: recovered %d/%d bits", result.Recovered, result.Total)
			}
			if !bytes.Equal(result.Payload, test.payload) {
				t.Errorf("payload mismatch: got %x, want %x", result.Payload, test.payload)
			}
		})
	}
}

func TestStreamMatchesEncoder(t *testing.T) {
	payload := []byte{0xbe, 0xef}
	seed := uint64(42)
	k := len(payload) * 8
	n := k * 2

	enc := NewEncoder(payload, rand.NewPCG(seed, seed))
	stream := NewStream(k, rand.NewPCG(seed, seed))

	for i := range n {
		sym := enc.Next()
		got := stream.Next(sym.Value)
		if got.Value != sym.Value {
			t.Errorf("symbol %d: value = %d, want %d", i, got.Value, sym.Value)
		}
		if len(got.Sources) != len(sym.Sources) {
			t.Fatalf("symbol %d: sources length = %d, want %d", i, len(got.Sources), len(sym.Sources))
		}
		for j := range sym.Sources {
			if got.Sources[j] != sym.Sources[j] {
				t.Errorf("symbol %d: sources[%d] = %d, want %d", i, j, got.Sources[j], sym.Sources[j])
			}
		}
	}
}

func TestErasureRecovery(t *testing.T) {
	payload := []byte{0xbe, 0xef}
	seed := uint64(42)
	k := len(payload) * 8
	n := k * 3

	enc := NewEncoder(payload, rand.NewPCG(seed, seed))
	symbols := make([]Symbol, n)
	for i := range symbols {
		symbols[i] = enc.Next()
	}

	// Drop every third symbol to simulate erasure.
	stream := NewStream(k, rand.NewPCG(seed, seed))
	dec := NewDecoder(k)
	for i, sym := range symbols {
		mapped := stream.Next(sym.Value)
		if i%3 == 0 {
			continue
		}
		dec.Add(mapped)
	}

	result := dec.Finish()
	if !result.OK {
		t.Fatalf("decode with erasures failed: recovered %d/%d bits", result.Recovered, result.Total)
	}
	if !bytes.Equal(result.Payload, payload) {
		t.Errorf("payload mismatch: got %x, want %x", result.Payload, payload)
	}
}

func TestPackUnpack(t *testing.T) {
	for _, test := range []struct {
		name string
		data []byte
	}{
		{name: "zeros", data: []byte{0x00, 0x00}},
		{name: "ones", data: []byte{0xff, 0xff}},
		{name: "mixed", data: []byte{0xbe, 0xef}},
		{name: "single", data: []byte{0xa5}},
	} {
		t.Run(test.name, func(t *testing.T) {
			bits := unpack(test.data)
			if len(bits) != len(test.data)*8 {
				t.Fatalf("unpack: got %d bits, want %d", len(bits), len(test.data)*8)
			}
			got := pack(bits)
			if !bytes.Equal(got, test.data) {
				t.Errorf("pack(unpack(%x)) = %x", test.data, got)
			}
		})
	}
}
