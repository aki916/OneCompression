# Copyright 2025-2026 Fujitsu Ltd.

"""Export OneComp's supported symmetric GPTQ 4-bit/group-128 comparison model.

Use a pinned local Qwen3-8B snapshot for --model. This script is intentionally
separate from reasoning calibration; it uses the repository's existing weight
quantizer and calibration machinery without modifying them.
"""

import argparse
from pathlib import Path

from onecomp import GPTQ, CalibrationConfig, ModelConfig, Runner
from onecomp.reasoning.data import write_json
from onecomp.reasoning.experiment import checkpoint_identity, environment_metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--source-revision", required=True, help="Original Qwen3-8B Hub commit SHA"
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--calibration-samples", type=int, default=128)
    args = parser.parse_args()
    if not args.model.is_dir():
        parser.error("--model must be a pinned local snapshot directory")
    if args.output.exists():
        parser.error("--output must not already exist")
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable; cannot run the Qwen3-8B weight-compression experiment"
        )
    runner = Runner(
        model_config=ModelConfig(path=str(args.model), dtype="bfloat16", device="cuda:0"),
        quantizer=GPTQ(wbits=4, groupsize=128, sym=True, actorder=False, mse=False),
        calibration_config=CalibrationConfig(
            num_calibration_samples=args.calibration_samples, max_length=2048, seed=0
        ),
        qep=False,
    )
    runner.run()
    runner.save_quantized_model(str(args.output), pack_weights=True)
    write_json(
        args.output / "reasoning_weight_recipe.json",
        {
            **environment_metadata(),
            "source_model": "Qwen/Qwen3-8B",
            "source_model_revision": args.source_revision,
            "source_checkpoint_sha256": checkpoint_identity(args.model),
            "quantization": "mixed_gptq",
            "bits": 4,
            "group_size": 128,
            "symmetric": True,
            "actorder": False,
            "mse": False,
            "calibration_samples": args.calibration_samples,
            "calibration_seed": 0,
        },
    )


if __name__ == "__main__":
    main()
