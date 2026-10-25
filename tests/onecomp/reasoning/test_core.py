"""CPU-only controller regression tests; also runnable with unittest."""

import unittest

from onecomp.reasoning import (
    ReasoningCompressionController,
    ReasoningConfig,
    ReasoningFormat,
    RequestState,
)
from onecomp.reasoning.actions import StopAction
from onecomp.reasoning.boundaries import (
    SentenceBoundaryDetector,
    TokenBoundaryDetector,
    TransitionBoundaryDetector,
)


class CoreTests(unittest.TestCase):
    def test_asag_trap_zero_current_attention(self):
        state = RequestState("zero", ReasoningFormat((10,), (20,)))
        controller = ReasoningCompressionController(ReasoningConfig(policy="asag_reference"))
        result = controller.decide(
            state,
            {
                "relative_entropy": 0.1,
                "previous_attention": 0.2,
                "current_attention": 0.0,
                "backward_ratio": None,
            },
        )
        self.assertEqual(result.kind, "jump")

    def state(self, name="test"):
        return RequestState(name, ReasoningFormat((10, 11), (20, 21), False))

    def test_marker_accounting_and_phase(self):
        s = self.state()
        for t in (10, 11, 1, 2, 20, 21, 3):
            s.append(t)
        self.assertEqual((s.reasoning_tokens, s.answer_tokens, s.marker_tokens), (2, 1, 4))
        self.assertEqual(s.total_generated_tokens, 7)
        self.assertEqual(s.phase, "answer")

    def test_partial_marker_is_counted_until_complete(self):
        s = RequestState("a", ReasoningFormat((10,), (20, 21)))
        s.append(20)
        self.assertTrue(s.pending_end_marker)
        s.append(2)
        self.assertFalse(s.pending_end_marker)
        self.assertEqual(s.reasoning_tokens, 2)

    def test_independent_interleaved_requests(self):
        fmt = ReasoningFormat((10,), (20,))
        a, b = RequestState("a", fmt), RequestState("b", fmt)
        c = ReasoningCompressionController(ReasoningConfig(policy="fixed", budget=1))
        a.append(5)
        self.assertEqual(c.decide(a, {}).kind, "stop")
        self.assertEqual(c.decide(b, {}).kind, "continue")
        self.assertEqual(b.interventions, [])
        self.assertEqual(b.boundaries, [0])

    def test_disabled_identity(self):
        s = self.state()
        c = ReasoningCompressionController()
        self.assertEqual(c.decide(s, {"logit_margin": -100}).kind, "continue")
        self.assertEqual(s.signals, [])
        self.assertEqual(s.interventions, [])

    def test_fixed_budget_zero_and_no_sentence_delay(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="fixed", budget=0))
        self.assertTrue(c.is_boundary(s))
        self.assertEqual(c.decide(s, {}).kind, "stop")
        pending = []
        StopAction(s.format).apply(pending)
        self.assertEqual(pending, [20])

    def test_boundaries(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        s.append(1, "done.\n")
        self.assertTrue(SentenceBoundaryDetector().detect(s))
        self.assertTrue(TransitionBoundaryDetector().detect(s, " Wait"))
        self.assertFalse(TokenBoundaryDetector(2).detect(s))
        s.append(2)
        self.assertTrue(TokenBoundaryDetector(2).detect(s))

    def test_thinkbrake_gap_direction(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="thinkbrake", threshold=0.1))
        self.assertEqual(c.decide(s, {"logit_margin": 0.2}).kind, "continue")
        s.append(1)
        self.assertEqual(c.decide(s, {"logit_margin": 0.05}).kind, "stop")

    def test_stability_uses_answer_and_requires_window(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="stability", stability_window=3))
        for answer in ("42", "42"):
            s.append(1)
            self.assertEqual(c.decide(s, {"answer": answer}).kind, "continue")
        s.append(1)
        self.assertEqual(c.decide(s, {"answer": "42"}).kind, "stop")

    def test_asag_reference_first_entropy_and_actions(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="asag_reference"))
        self.assertEqual(
            c.decide(s, {"confidence": 0.4, "attention_entropy": 10}).kind, "continue"
        )
        s.append(1)
        self.assertEqual(c.decide(s, {"confidence": 0.4, "attention_entropy": 8}).kind, "bias")
        self.assertAlmostEqual(s.signals[-1]["relative_entropy"], -0.2)
        s.append(1)
        self.assertEqual(
            c.decide(s, {"confidence": 0.4, "attention_entropy": 11, "backward_ratio": 2}).kind,
            "jump",
        )
        s.append(1)
        self.assertEqual(
            c.decide(s, {"confidence": 0.4, "attention_entropy": 11, "backward_ratio": 2}).kind,
            "stop",
        )

    def test_first_attention_boundary_has_no_relative_entropy(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="asag_reference"))
        c.decide(s, {"confidence": 0.4, "attention_entropy": 10})
        self.assertIsNone(s.signals[-1]["relative_entropy"])
        self.assertTrue(s.signals[-1]["first_attention_boundary"])

    def test_preamble_tokens_are_neither_reasoning_nor_answer(self):
        s = RequestState("a", ReasoningFormat((10,), (20,), starts_in_reasoning=False))
        for token in (1, 1, 10, 2, 20, 3):
            s.append(token)
        self.assertEqual(
            s.token_kinds, ["preamble", "preamble", "marker", "reasoning", "marker", "answer"]
        )
        self.assertEqual((s.reasoning_tokens, s.answer_tokens, s.marker_tokens), (1, 1, 2))

    def test_repeated_boundary_does_not_accumulate_evidence(self):
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        c = ReasoningCompressionController(ReasoningConfig(policy="stability", stability_window=2))
        s.append(1, "Done.")
        self.assertEqual(c.decide(s, {"answer": "42"}).kind, "continue")
        self.assertFalse(c.is_boundary(s))
        self.assertEqual(c.decide(s, {"answer": "42"}).kind, "continue")
        self.assertEqual(len(s.probe_answers), 1)
        self.assertEqual(len(s.signals), 1)

    def test_incomplete_probes_never_imply_confidence_or_stability(self):
        for policy in ("deer", "stability", "asag_reference", "progressive"):
            c = ReasoningCompressionController(
                ReasoningConfig(
                    policy=policy,
                    confidence_threshold=0,
                    stability_window=1,
                    relative_confidence=True,
                )
            )
            s = RequestState("a", ReasoningFormat((10,), (20,)))
            for _ in range(2):
                s.append(1)
                decision = c.decide(
                    s,
                    {
                        "answer": "42",
                        "confidence": 1,
                        "probe_complete": False,
                        "attention_entropy": 10,
                        "relative_entropy": None,
                    },
                )
                self.assertEqual(decision.kind, "continue")
            self.assertEqual(s.probe_confidences, [])

    def test_asag_strict_confidence_first_boundary_and_absent_ratios(self):
        c = ReasoningCompressionController(ReasoningConfig(policy="asag_reference"))
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        self.assertEqual(
            c.decide(s, {"confidence": 0.95, "attention_entropy": 10}).kind, "continue"
        )
        other = RequestState("b", s.format)
        self.assertEqual(
            c.decide(other, {"confidence": 0.951, "attention_entropy": 10}).kind, "stop"
        )
        s.append(1)
        self.assertEqual(
            c.decide(
                s, {"confidence": 0.99, "attention_entropy": 11, "backward_ratio": None}
            ).kind,
            "continue",
        )

    def test_missing_entropy_in_progressive_is_not_convergence(self):
        c = ReasoningCompressionController(ReasoningConfig(policy="progressive"))
        s = RequestState("a", ReasoningFormat((10,), (20,)))
        self.assertEqual(
            c.decide(s, {"confidence": 0.92, "relative_entropy": None}).kind, "continue"
        )

    def test_thinkbrake_threshold_sweep_is_not_limited_by_cascade(self):
        self.assertEqual(ReasoningConfig(policy="thinkbrake", threshold=10).threshold, 10)

    def test_no_oracle_label_leakage(self):
        c = ReasoningCompressionController(ReasoningConfig(policy="thinkbrake"))
        with self.assertRaises(ValueError):
            c.decide(RequestState("a", ReasoningFormat((10,), (20,))), {"safe_stop": True})

    def test_config_validation(self):
        self.assertEqual(ReasoningConfig.from_dict({}).policy, "none")
        for values in (
            {"bad": 1},
            {"policy": "ASAG"},
            {"budget": -1},
            {"interval": 0},
            {"confidence_threshold": 1.1},
            {"threshold": float("nan")},
            {"budget": True},
        ):
            with self.assertRaises((ValueError, TypeError)):
                ReasoningConfig.from_dict(values)


if __name__ == "__main__":
    unittest.main()
