package scorer_test

import (
	"context"
	"encoding/json"
	"math"
	"testing"

	"github.com/google/go-cmp/cmp"
	"github.com/google/go-cmp/cmp/cmpopts"

	k8stypes "k8s.io/apimachinery/pkg/types"
	fwkdl "sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/datalayer"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"

	"github.com/llm-d/llm-d-inference-scheduler/pkg/plugins/scorer"
	"github.com/llm-d/llm-d-inference-scheduler/test/utils"
)

// kvScore computes the expected KV cache cubic penalty for a given usage and threshold.
func kvScore(kv, threshold float64) float64 {
	if kv >= threshold {
		return 0.0
	}
	r := kv / threshold
	return 1.0 - r*r*r
}

func TestHealthScorer(t *testing.T) {
	approxOpt := cmpopts.EquateApprox(0, 0.01)
	threshold := scorer.KVCacheThresholdDefault
	kvW := scorer.KVWeightDefault
	preW := scorer.PreemptionWeightDefault

	epNoMetrics := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "no-metrics"}},
		nil, nil,
	)
	epEmpty := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "empty"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, PreemptionCount: 0}, nil,
	)
	epModerate := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "moderate"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.5, PreemptionCount: 0}, nil,
	)
	epHigh := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "high"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.84, PreemptionCount: 0}, nil,
	)
	epAtThreshold := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "at-threshold"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.85, PreemptionCount: 0}, nil,
	)
	epOverThreshold := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "over-threshold"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.95, PreemptionCount: 0}, nil,
	)

	tests := []struct {
		name       string
		endpoints  []scheduling.Endpoint
		wantScores map[scheduling.Endpoint]float64
	}{
		{
			name:      "no metrics returns fallback score",
			endpoints: []scheduling.Endpoint{epNoMetrics},
			wantScores: map[scheduling.Endpoint]float64{
				epNoMetrics: scorer.NoMetricsFallbackScore,
			},
		},
		{
			name:      "empty KV (0.0) scores maximum",
			endpoints: []scheduling.Endpoint{epEmpty},
			wantScores: map[scheduling.Endpoint]float64{
				epEmpty: kvW*kvScore(0.0, threshold) + preW*1.0,
			},
		},
		{
			name:      "moderate KV (0.5) scores mid-high",
			endpoints: []scheduling.Endpoint{epModerate},
			wantScores: map[scheduling.Endpoint]float64{
				epModerate: kvW*kvScore(0.5, threshold) + preW*1.0,
			},
		},
		{
			name:      "high KV (0.84) scores very low",
			endpoints: []scheduling.Endpoint{epHigh},
			wantScores: map[scheduling.Endpoint]float64{
				epHigh: kvW*kvScore(0.84, threshold) + preW*1.0,
			},
		},
		{
			name:      "at threshold (0.85) KV score is 0",
			endpoints: []scheduling.Endpoint{epAtThreshold},
			wantScores: map[scheduling.Endpoint]float64{
				epAtThreshold: kvW*0.0 + preW*1.0,
			},
		},
		{
			name:      "over threshold (0.95) KV score is 0",
			endpoints: []scheduling.Endpoint{epOverThreshold},
			wantScores: map[scheduling.Endpoint]float64{
				epOverThreshold: kvW*0.0 + preW*1.0,
			},
		},
		{
			name:      "multiple endpoints scored correctly",
			endpoints: []scheduling.Endpoint{epEmpty, epModerate, epOverThreshold},
			wantScores: map[scheduling.Endpoint]float64{
				epEmpty:         kvW*kvScore(0.0, threshold) + preW*1.0,
				epModerate:      kvW*kvScore(0.5, threshold) + preW*1.0,
				epOverThreshold: kvW*0.0 + preW*1.0,
			},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			s := scorer.NewHealthScorer(utils.NewTestContext(t), &scorer.DefaultHealthScorerParameters)
			got := s.Score(context.Background(), nil, nil, test.endpoints)

			if diff := cmp.Diff(test.wantScores, got, approxOpt); diff != "" {
				t.Errorf("Unexpected output (-want +got): %v", diff)
			}
		})
	}
}

func TestHealthScorerPreemptionDelta(t *testing.T) {
	approxOpt := cmpopts.EquateApprox(0, 0.01)
	kvW := scorer.KVWeightDefault
	preW := scorer.PreemptionWeightDefault

	s := scorer.NewHealthScorer(utils.NewTestContext(t), &scorer.DefaultHealthScorerParameters)

	// Cycle 1: preemption=0, first time seen → preScore=1.0
	ep := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, PreemptionCount: 0}, nil,
	)
	got := s.Score(context.Background(), nil, nil, []scheduling.Endpoint{ep})
	want := kvW*1.0 + preW*1.0
	if diff := cmp.Diff(map[scheduling.Endpoint]float64{ep: want}, got, approxOpt); diff != "" {
		t.Errorf("Cycle 1 (first seen) (-want +got): %v", diff)
	}

	// Cycle 2: preemption still 0, delta=0 → preScore=1.0
	got = s.Score(context.Background(), nil, nil, []scheduling.Endpoint{ep})
	if diff := cmp.Diff(map[scheduling.Endpoint]float64{ep: want}, got, approxOpt); diff != "" {
		t.Errorf("Cycle 2 (no change) (-want +got): %v", diff)
	}

	// Cycle 3: preemption increased to 3, delta=3 → preScore=0.0
	epPreempting := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, PreemptionCount: 3}, nil,
	)
	got = s.Score(context.Background(), nil, nil, []scheduling.Endpoint{epPreempting})
	wantPenalized := kvW*1.0 + preW*0.0
	if diff := cmp.Diff(map[scheduling.Endpoint]float64{epPreempting: wantPenalized}, got, approxOpt); diff != "" {
		t.Errorf("Cycle 3 (preemption spike) (-want +got): %v", diff)
	}

	// Cycle 4: preemption still 3, delta=0 → preScore=1.0 (recovered)
	got = s.Score(context.Background(), nil, nil, []scheduling.Endpoint{epPreempting})
	wantRecovered := kvW*1.0 + preW*1.0
	if diff := cmp.Diff(map[scheduling.Endpoint]float64{epPreempting: wantRecovered}, got, approxOpt); diff != "" {
		t.Errorf("Cycle 4 (recovered) (-want +got): %v", diff)
	}
}

func TestHealthScorerPrunesAbsentEndpoints(t *testing.T) {
	approxOpt := cmpopts.EquateApprox(0, 0.01)
	kvW := scorer.KVWeightDefault
	preW := scorer.PreemptionWeightDefault

	s := scorer.NewHealthScorer(utils.NewTestContext(t), &scorer.DefaultHealthScorerParameters)

	epA := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, PreemptionCount: 5}, nil,
	)
	epB := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-b"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, PreemptionCount: 0}, nil,
	)
	epAWithIncreasedPreemptions := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, PreemptionCount: 6}, nil,
	)

	// Cycle 1: both endpoints present, seeds prevPreemption
	s.Score(context.Background(), nil, nil, []scheduling.Endpoint{epA, epB})

	// Cycle 2: only pod-b remains — pod-a is pruned from history.
	s.Score(context.Background(), nil, nil, []scheduling.Endpoint{epB})

	// Cycle 3: pod-a returns with preemption=6.
	// Since pod-a was pruned, it is treated as first-seen (preScore=1.0).
	got := s.Score(context.Background(), nil, nil, []scheduling.Endpoint{epAWithIncreasedPreemptions, epB})

	wantA := kvW*1.0 + preW*1.0 // first-seen after pruning → healthy
	wantB := kvW*1.0 + preW*1.0 // unchanged
	if diff := cmp.Diff(map[scheduling.Endpoint]float64{epAWithIncreasedPreemptions: wantA, epB: wantB}, got, approxOpt); diff != "" {
		t.Errorf("Pruned endpoint should be treated as first-seen (-want +got): %v", diff)
	}
}

func TestHealthScorerRetainsPreemptionHistoryAcrossMetricsGap(t *testing.T) {
	approxOpt := cmpopts.EquateApprox(0, 0.01)
	kvW := scorer.KVWeightDefault
	preW := scorer.PreemptionWeightDefault

	s := scorer.NewHealthScorer(utils.NewTestContext(t), &scorer.DefaultHealthScorerParameters)

	epWithPreemptions := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, PreemptionCount: 5}, nil,
	)
	epMissingMetrics := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		nil, nil,
	)
	epWithIncreasedPreemptions := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, PreemptionCount: 6}, nil,
	)

	// Cycle 1: establish baseline preemption count.
	s.Score(context.Background(), nil, nil, []scheduling.Endpoint{epWithPreemptions})

	// Cycle 2: endpoint remains a candidate but metrics are temporarily unavailable.
	s.Score(context.Background(), nil, nil, []scheduling.Endpoint{epMissingMetrics})

	// Cycle 3: metrics return with increased preemption count.
	// Expected: delta is still detected and preemption signal is penalized.
	got := s.Score(context.Background(), nil, nil, []scheduling.Endpoint{epWithIncreasedPreemptions})
	want := kvW*1.0 + preW*0.0
	if diff := cmp.Diff(map[scheduling.Endpoint]float64{epWithIncreasedPreemptions: want}, got, approxOpt); diff != "" {
		t.Errorf("Metrics gap should not reset preemption baseline (-want +got): %v", diff)
	}
}

func TestHealthScorerCustomParameters(t *testing.T) {
	approxOpt := cmpopts.EquateApprox(0, 0.01)

	ep := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.5, PreemptionCount: 0}, nil,
	)

	params := &scorer.HealthScorerParameters{
		KVCacheThreshold: 0.7,
		KVWeight:         0.8,
		PreemptionWeight: 0.2,
	}
	s := scorer.NewHealthScorer(utils.NewTestContext(t), params)
	got := s.Score(context.Background(), nil, nil, []scheduling.Endpoint{ep})

	wantKV := kvScore(0.5, 0.7)
	want := 0.8*wantKV + 0.2*1.0
	if diff := cmp.Diff(map[scheduling.Endpoint]float64{ep: want}, got, approxOpt); diff != "" {
		t.Errorf("Custom parameters (-want +got): %v", diff)
	}
}

func TestHealthScorerInvalidWeightsFallbackToDefaults(t *testing.T) {
	approxOpt := cmpopts.EquateApprox(0, 0.01)

	ep := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.5, PreemptionCount: 0}, nil,
	)

	params := &scorer.HealthScorerParameters{
		KVCacheThreshold: 0.7,
		KVWeight:         0.9,
		PreemptionWeight: 0.9, // sum > 1, should fallback to defaults
	}
	s := scorer.NewHealthScorer(utils.NewTestContext(t), params)
	got := s.Score(context.Background(), nil, nil, []scheduling.Endpoint{ep})

	wantKV := kvScore(0.5, 0.7)
	want := scorer.KVWeightDefault*wantKV + scorer.PreemptionWeightDefault*1.0
	if diff := cmp.Diff(map[scheduling.Endpoint]float64{ep: want}, got, approxOpt); diff != "" {
		t.Errorf("Invalid weights should fallback to defaults (-want +got): %v", diff)
	}
}

func TestHealthScorerInvalidKVValuesAreSanitized(t *testing.T) {
	approxOpt := cmpopts.EquateApprox(0, 0.01)
	kvW := scorer.KVWeightDefault
	preW := scorer.PreemptionWeightDefault

	s := scorer.NewHealthScorer(utils.NewTestContext(t), &scorer.DefaultHealthScorerParameters)

	epNegativeKV := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "negative-kv"}},
		&fwkdl.Metrics{KVCacheUsagePercent: -0.2, PreemptionCount: 0}, nil,
	)
	epNaNKV := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "nan-kv"}},
		&fwkdl.Metrics{KVCacheUsagePercent: math.NaN(), PreemptionCount: 0}, nil,
	)
	epInfKV := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "inf-kv"}},
		&fwkdl.Metrics{KVCacheUsagePercent: math.Inf(1), PreemptionCount: 0}, nil,
	)

	got := s.Score(context.Background(), nil, nil, []scheduling.Endpoint{epNegativeKV, epNaNKV, epInfKV})
	want := map[scheduling.Endpoint]float64{
		epNegativeKV: kvW*1.0 + preW*1.0, // clamped to 0.0
		epNaNKV:      kvW*1.0 + preW*1.0, // non-finite treated as 0.0
		epInfKV:      kvW*0.0 + preW*1.0, // clamped to 1.0
	}

	if diff := cmp.Diff(want, got, approxOpt); diff != "" {
		t.Errorf("Invalid KV values should be sanitized (-want +got): %v", diff)
	}
}

func TestHealthScorerNilParameters(t *testing.T) {
	approxOpt := cmpopts.EquateApprox(0, 0.01)

	s := scorer.NewHealthScorer(utils.NewTestContext(t), nil)

	ep := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.0, PreemptionCount: 0}, nil,
	)
	got := s.Score(context.Background(), nil, nil, []scheduling.Endpoint{ep})

	want := scorer.KVWeightDefault*1.0 + scorer.PreemptionWeightDefault*1.0
	if diff := cmp.Diff(map[scheduling.Endpoint]float64{ep: want}, got, approxOpt); diff != "" {
		t.Errorf("Nil parameters should use defaults (-want +got): %v", diff)
	}
}

func TestHealthScorerFactory(t *testing.T) {
	approxOpt := cmpopts.EquateApprox(0, 0.01)

	ep := scheduling.NewEndpoint(
		&fwkdl.EndpointMetadata{NamespacedName: k8stypes.NamespacedName{Name: "pod-a"}},
		&fwkdl.Metrics{KVCacheUsagePercent: 0.5, PreemptionCount: 0}, nil,
	)

	t.Run("default parameters", func(t *testing.T) {
		handle := newFakeHandle(utils.NewTestContext(t))
		p, err := scorer.HealthScorerFactory("my-health", nil, handle)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		hs := p.(*scorer.HealthScorer)
		got := hs.Score(context.Background(), nil, nil, []scheduling.Endpoint{ep})

		kvW := scorer.KVWeightDefault
		preW := scorer.PreemptionWeightDefault
		want := kvW*kvScore(0.5, scorer.KVCacheThresholdDefault) + preW*1.0
		if diff := cmp.Diff(map[scheduling.Endpoint]float64{ep: want}, got, approxOpt); diff != "" {
			t.Errorf("Factory default (-want +got): %v", diff)
		}
		if hs.TypedName().Name != "my-health" {
			t.Errorf("expected name 'my-health', got %q", hs.TypedName().Name)
		}
	})

	t.Run("custom JSON parameters", func(t *testing.T) {
		raw := json.RawMessage(`{"kvCacheThreshold": 0.7, "kvWeight": 0.6, "preemptionWeight": 0.4}`)
		handle := newFakeHandle(utils.NewTestContext(t))
		p, err := scorer.HealthScorerFactory("custom", raw, handle)
		if err != nil {
			t.Fatalf("unexpected error: %v", err)
		}
		hs := p.(*scorer.HealthScorer)
		got := hs.Score(context.Background(), nil, nil, []scheduling.Endpoint{ep})

		want := 0.6*kvScore(0.5, 0.7) + 0.4*1.0
		if diff := cmp.Diff(map[scheduling.Endpoint]float64{ep: want}, got, approxOpt); diff != "" {
			t.Errorf("Factory custom (-want +got): %v", diff)
		}
	})

	t.Run("invalid JSON", func(t *testing.T) {
		raw := json.RawMessage(`{invalid}`)
		handle := newFakeHandle(utils.NewTestContext(t))
		_, err := scorer.HealthScorerFactory("bad", raw, handle)
		if err == nil {
			t.Fatal("expected error for invalid JSON")
		}
	})
}
