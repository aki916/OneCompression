"""

Example: Runtime reasoning compression with the Transformers reference backend

Compares a few training-free policies that shorten a thinking model's reasoning
on one math question. Each policy decodes greedily from the same prompt and
reports reasoning tokens, total model-token evaluations (including answer
probes) and wall-clock latency.

  - none:       vanilla generation (identical tokens to disabled compression)
  - fixed:      force </think> after a fixed reasoning-token budget
  - thinkbrake: stop at a sentence boundary when </think> nearly wins (logits only)
  - deer:       at each "Wait", probe a trial answer; stop if it is confident

The reference backend is serial and batch-size one. For serving, use the
vLLM logits processor (see vllm_inference/example_reasoning_vllm_inference.py).
ASAG (``policy="asag_reference"``) additionally needs the model loaded with
``attn_implementation="eager"``.

A OneComp-quantized model works the same way, e.g.::

    model, tokenizer = QuantizedModelLoader.load_quantized_model("./qwen3-gptq-4bit")

Copyright 2025-2026 Fujitsu Ltd.

"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from onecomp.reasoning import ReasoningConfig, ReasoningFormat
from onecomp.reasoning.data import extract_answer
from onecomp.runtime.transformers import TransformersReasoningAdapter

MODEL_ID = "Qwen/Qwen3-0.6B"
QUESTION = (
    "Solve the following problem. Put only your final answer inside \\boxed{}.\n\n"
    "What is the smallest positive integer n such that 3n is a perfect square "
    "and 2n is a perfect cube?"
)

model = AutoModelForCausalLM.from_pretrained(MODEL_ID, dtype=torch.bfloat16, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
adapter = TransformersReasoningAdapter(model, tokenizer)

# Qwen3 opens reasoning with <think> and closes it with </think>.
reasoning_format = ReasoningFormat.from_tokenizer(tokenizer)
prompt_ids = tokenizer.apply_chat_template(
    [{"role": "user", "content": QUESTION}],
    tokenize=True,
    add_generation_prompt=True,
    enable_thinking=True,
    return_dict=False,
)

policies = [
    ReasoningConfig(),  # policy="none"
    ReasoningConfig(policy="fixed", budget=512),
    # ThinkBrake divides the logit gap by the sampling temperature; 0.6 keeps the
    # paper's tau scale for Qwen3 even though this example decodes greedily.
    ReasoningConfig(policy="thinkbrake", threshold=0.1, temperature=0.6, boundary="sentence"),
    ReasoningConfig(policy="deer", confidence_threshold=0.95, boundary="transition"),
]

for config in policies:
    result = adapter.generate(
        prompt_ids, reasoning_format=reasoning_format, config=config, max_new_tokens=4096
    )
    metrics = result.metrics
    print(
        f"{config.policy:>10}: answer={extract_answer(result.final_answer)!r} "
        f"reasoning_tokens={metrics['reasoning_tokens']} "
        f"model_evaluated_tokens={metrics['model_evaluated_tokens']} "
        f"latency={metrics['latency_s']:.1f}s "
        f"interventions={[i['reason'] for i in result.state.interventions]}"
    )
