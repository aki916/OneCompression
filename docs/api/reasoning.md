# Reasoning Compression

Runtime reasoning-length control (experimental). Usage, policies and the
benchmark CLI are described in the [Reasoning Compression user guide](../user-guide/reasoning-compression.md).

## Configuration and request state

::: onecomp.reasoning.ReasoningConfig

::: onecomp.reasoning.ReasoningFormat

::: onecomp.reasoning.RequestState

::: onecomp.reasoning.ReasoningCompressionController

## Policies

::: onecomp.reasoning.policies
    options:
      members:
        - ReasoningPolicy
        - make_policy
        - BudgetPolicy
        - ThinkBrakePolicy
        - DeerPolicy
        - StabilityPolicy
        - AsagPolicy
        - ProgressivePolicy

## Signals

::: onecomp.reasoning.signals
    options:
      members:
        - logit_observation
        - deer_confidence
        - asag_confidence
        - attention_statistics
        - relative_entropy_change

## Backends

::: onecomp.runtime.transformers.TransformersReasoningAdapter

::: onecomp.runtime.vllm.VLLMReasoningAdapter
