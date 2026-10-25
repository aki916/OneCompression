# Copyright 2025-2026 Fujitsu Ltd.

"""Experimental OneComp cascade: logits, then answer probe, then attention probe.

This is a new reference cascade, not a reproduced literature method.
"""

from ..actions import Decision
from .base import ReasoningPolicy, has_cheap_stop, is_confident

# Width of the "nearly confident" band that pays for an attention probe.
ATTENTION_BAND = 0.05


class ProgressivePolicy(ReasoningPolicy):
    name = "progressive"

    @property
    def requires_attention(self):
        return self.config.asag_mode == "full"

    def needs_probe(self, state, observation):
        # Probe only an uncertain margin; a small margin already stops cheaply.
        margin = observation.get("logit_margin")
        return (
            margin is not None
            and self.config.threshold < margin <= self.config.cascade_uncertainty
        )

    def needs_attention_followup(self, probe):
        lower = max(0.0, self.config.confidence_threshold - ATTENTION_BAND)
        return (
            self.requires_attention
            and probe.complete
            and lower <= probe.confidence <= self.config.confidence_threshold
        )

    def decide(self, state, signals):
        c = self.config
        if has_cheap_stop(signals, c):
            return Decision("stop", "cascade_logits")
        if is_confident(signals, c):
            return Decision("stop", "cascade_confidence")
        # Requires confidence as well as convergence; never declare correctness
        # from entropy alone.
        confidence = signals.get("confidence")
        relative_entropy = signals.get("relative_entropy")
        if (
            signals.get("probe_complete", True)
            and confidence is not None
            and confidence >= max(0, c.confidence_threshold - ATTENTION_BAND)
            and relative_entropy is not None
            and relative_entropy < c.entropy_threshold
        ):
            return Decision("stop", "cascade_attention")
        return Decision()
