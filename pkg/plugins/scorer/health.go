package scorer

import (
	"context"
	"encoding/json"
	"fmt"
	"math"
	"sync"

	"sigs.k8s.io/controller-runtime/pkg/log"
	logutil "sigs.k8s.io/gateway-api-inference-extension/pkg/common/observability/logging"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/plugin"
	"sigs.k8s.io/gateway-api-inference-extension/pkg/epp/framework/interface/scheduling"
)

const (
	// HealthScorerType is the type of the HealthScorer scorer.
	HealthScorerType = "health-scorer"

	// KVCacheThresholdDefault is the KV cache usage threshold above which the score is 0.
	KVCacheThresholdDefault = 0.85

	// KVWeightDefault is the default weight for the KV cache signal.
	KVWeightDefault = 0.55

	// PreemptionWeightDefault is the default weight for the preemption signal.
	PreemptionWeightDefault = 0.45

	// NoMetricsFallbackScore is the score assigned when metrics are unavailable.
	NoMetricsFallbackScore = 0.5
)

// HealthScorerParameters holds the configurable parameters for the health scorer.
type HealthScorerParameters struct {
	KVCacheThreshold float64 `json:"kvCacheThreshold"`
	KVWeight         float64 `json:"kvWeight"`
	PreemptionWeight float64 `json:"preemptionWeight"`
}

// DefaultHealthScorerParameters provides default values for the health scorer.
var DefaultHealthScorerParameters = HealthScorerParameters{
	KVCacheThreshold: KVCacheThresholdDefault,
	KVWeight:         KVWeightDefault,
	PreemptionWeight: PreemptionWeightDefault,
}

// compile-time type assertion
var _ scheduling.Scorer = &HealthScorer{}

// HealthScorerFactory defines the factory function for the HealthScorer.
func HealthScorerFactory(name string, rawParameters json.RawMessage, handle plugin.Handle) (plugin.Plugin, error) {
	parameters := DefaultHealthScorerParameters // copy
	if rawParameters != nil {
		if err := json.Unmarshal(rawParameters, &parameters); err != nil {
			return nil, fmt.Errorf("failed to parse the parameters of the '%s' scorer - %w", HealthScorerType, err)
		}
	}

	return NewHealthScorer(handle.Context(), &parameters).WithName(name), nil
}

// NewHealthScorer creates a new health scorer with KV cache and preemption signals.
func NewHealthScorer(ctx context.Context, params *HealthScorerParameters) *HealthScorer {
	p := DefaultHealthScorerParameters
	if params != nil {
		// Work on a copy to avoid mutating the caller's value.
		p = *params
	}

	if p.KVCacheThreshold <= 0 || p.KVCacheThreshold > 1.0 {
		p.KVCacheThreshold = KVCacheThresholdDefault
		log.FromContext(ctx).V(logutil.DEFAULT).Info(fmt.Sprintf("kvCacheThreshold must be in (0, 1], using default %f", KVCacheThresholdDefault))
	}
	if p.KVWeight < 0 || p.KVWeight > 1.0 ||
		p.PreemptionWeight < 0 || p.PreemptionWeight > 1.0 ||
		p.KVWeight+p.PreemptionWeight <= 0 || p.KVWeight+p.PreemptionWeight > 1.0 {
		p.KVWeight = KVWeightDefault
		p.PreemptionWeight = PreemptionWeightDefault
		log.FromContext(ctx).V(logutil.DEFAULT).Info("weights must be in [0,1], sum in (0,1]; using defaults")
	}

	return &HealthScorer{
		typedName:        plugin.TypedName{Type: HealthScorerType},
		kvCacheThreshold: p.KVCacheThreshold,
		kvWeight:         p.KVWeight,
		preemptionWeight: p.PreemptionWeight,
		prevPreemption:   make(map[string]int),
	}
}

// HealthScorer is a two-signal scorer combining KV cache utilization (cubic penalty)
// and preemption delta detection.
type HealthScorer struct {
	typedName        plugin.TypedName
	kvCacheThreshold float64
	kvWeight         float64
	preemptionWeight float64

	mutex          sync.Mutex
	prevPreemption map[string]int
}

// TypedName returns the typed name of the plugin.
func (s *HealthScorer) TypedName() plugin.TypedName {
	return s.typedName
}

// WithName sets the name of the plugin.
func (s *HealthScorer) WithName(name string) *HealthScorer {
	s.typedName.Name = name
	return s
}

// Category returns Distribution — this scorer distributes load away from unhealthy endpoints.
func (s *HealthScorer) Category() scheduling.ScorerCategory {
	return scheduling.Distribution
}

// Score scores endpoints based on KV cache pressure and preemption activity.
//
// KV cache signal: cubic penalty — score drops sharply as usage approaches the threshold.
//
//	kv >= threshold → 0.0
//	kv < threshold  → 1.0 - (kv / threshold)^3
//
// Preemption signal: binary delta — 0.0 if preemptions increased since last cycle, 1.0 otherwise.
//
// Final score: kvWeight * kvScore + preemptionWeight * preScore
func (s *HealthScorer) Score(_ context.Context, _ *scheduling.CycleState, _ *scheduling.LLMRequest, endpoints []scheduling.Endpoint) map[scheduling.Endpoint]float64 {
	scoredEndpoints := make(map[scheduling.Endpoint]float64, len(endpoints))

	s.mutex.Lock()
	defer s.mutex.Unlock()

	for _, endpoint := range endpoints {
		epName := endpoint.GetMetadata().NamespacedName.String()

		metrics := endpoint.GetMetrics()
		if metrics == nil {
			scoredEndpoints[endpoint] = NoMetricsFallbackScore
			continue
		}

		// KV cache cubic penalty
		kv := metrics.KVCacheUsagePercent
		var kvScore float64
		if kv >= s.kvCacheThreshold {
			kvScore = 0.0
		} else {
			ratio := kv / s.kvCacheThreshold
			kvScore = 1.0 - math.Pow(ratio, 3)
		}

		// Preemption delta
		currentPreemption := metrics.PreemptionCount
		prevPreemption, exists := s.prevPreemption[epName]
		s.prevPreemption[epName] = currentPreemption

		var preScore float64
		if !exists {
			// First time seeing this endpoint — assume healthy
			preScore = 1.0
		} else {
			delta := currentPreemption - prevPreemption
			if delta > 0 {
				preScore = 0.0
			} else {
				preScore = 1.0
			}
		}

		scoredEndpoints[endpoint] = s.kvWeight*kvScore + s.preemptionWeight*preScore
	}

	return scoredEndpoints
}
