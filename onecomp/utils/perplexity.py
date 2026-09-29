"""

Copyright 2025-2026 Fujitsu Ltd.

Author: Keiji Kimura

LLM Perplexity (PPL) measurement module.

Computes per-token perplexity using the sliding-window approach from
Hugging Face Transformers.
https://huggingface.co/docs/transformers/perplexity

Benchmark results (WikiText-2, per-token perplexity):

    ===============  ================================  ===========================  ============
    Model            Default (ml=2048, s=2048)          HF official (ml=MPE, s=512)  Lit. [1][2]
    ===============  ================================  ===========================  ============
    Llama-2-7b       5.4680                             4.7739                         5.47
    Llama-2-13b      4.8800                             4.2762                         4.88
    ===============  ================================  ===========================  ============

    - Default: max_length=2048, stride=2048 (standard setting in quantization papers, matches literature values)
    - HF official: max_length=max_position_embeddings, stride=512 (with overlap -> lower PPL)

References:
    [1] Lin et al., "AWQ: Activation-aware Weight Quantization for LLM Compression
        and Acceleration", arXiv:2306.00978, 2024.
    [2] Tseng et al., "QuIP#: Even Better LLM Quantization with Hadamard Incoherence
        and Lattice Codebooks", arXiv:2402.04396, 2024.

"""

import torch
from datasets import load_dataset
from tqdm import tqdm

from .model_inputs import add_model_specific_inputs


def calculate_perplexity(
    model=None,
    tokenizer=None,
    model_config=None,
    dataset_name="Salesforce/wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="test",
    max_samples=None,
    max_length=2048,
    stride=2048,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    """Calculate perplexity of a Hugging Face Transformers model.

    Based on https://huggingface.co/docs/transformers/perplexity

    Args:
        dataset_name (str): Dataset name (e.g. "Salesforce/wikitext", "allenai/c4").
        dataset_config (str): Dataset configuration.
            - For WikiText: "wikitext-2-raw-v1"
            - For C4: "en/c4-train.00001-of-01024.json.gz" (treated as data_files)
        split (str): Dataset split (e.g. "test", "train", "validation").
        max_samples (int, optional): Maximum number of samples to use. If None, all
            samples are used. 128 or 512 is recommended for C4.
        max_length (int, optional): Maximum length of the sliding window. Default: 2048.
            Aligned with the standard setting in quantization papers.
            If None, model.config.max_position_embeddings is used
            (following the Hugging Face official guide).
        stride (int, optional): Stride of the sliding window. Default: 2048.
            stride < max_length enables overlapping evaluation (yields lower PPL).
            If None, set to the same value as max_length (no overlap).
            The Hugging Face official guide uses stride=512.

    Note:
        Hugging Face official guide parameters (yields lower PPL):
            max_length=model.config.max_position_embeddings, stride=512
            https://huggingface.co/docs/transformers/perplexity

    Example:
        >>> # Evaluate on WikiText-2 (default: standard quantization-paper setting)
        >>> ppl = calculate_perplexity(model=model, tokenizer=tokenizer)
        >>>
        >>> # Evaluate following the Hugging Face official guide
        >>> ppl = calculate_perplexity(
        ...     model=model, tokenizer=tokenizer,
        ...     max_length=model.config.max_position_embeddings,
        ...     stride=512,
        ... )
        >>>
        >>> # Evaluate on C4 (using 128 samples)
        >>> ppl = calculate_perplexity(
        ...     model=model,
        ...     tokenizer=tokenizer,
        ...     dataset_name="allenai/c4",
        ...     dataset_config="en/c4-train.00001-of-01024.json.gz",
        ...     split="train",
        ...     max_samples=128,
        ... )
    """

    # create a `model` and `tokenizer` object from the model config
    if model is None:
        if model_config is None:
            raise ValueError("model_config must be provided if model is not provided")
        model = model_config.load_model()
    if tokenizer is None:
        if model_config is None:
            raise ValueError("model_config must be provided if tokenizer is not provided")
        tokenizer = model_config.load_tokenizer()

    device = next(model.parameters()).device

    encodings = _load_encodings(tokenizer, dataset_name, dataset_config, split, max_samples)
    if max_length is None:
        max_length = model.config.max_position_embeddings
    if stride is None:
        stride = max_length
    seq_len = encodings.input_ids.size(1)
    use_cpu_accum = (
        device.type == "mps" if isinstance(device, torch.device) else str(device).startswith("mps")
    )
    accum_device = torch.device("cpu") if use_cpu_accum else device
    nll_sum = torch.tensor(0.0, dtype=torch.float64, device=accum_device)
    n_tokens = 0
    prev_end_loc = 0
    for begin_loc in tqdm(range(0, seq_len, stride)):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
        input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        with torch.no_grad():
            inputs = add_model_specific_inputs(
                inputs={"input_ids": input_ids, "labels": target_ids},
                model=model,
            )
            outputs = model(**inputs)
            # loss is calculated using CrossEntropyLoss which averages over valid labels
            # N.B. the model only calculates loss over trg_len - 1 labels,
            # because it internally shifts the labels
            # to the left by 1.
            neg_log_likelihood = outputs.loss.to(accum_device).to(torch.float64)
        # Accumulate the total negative log-likelihood and the total number of tokens
        num_valid_tokens = (
            (target_ids != -100).sum().item()
        )  # number of valid tokens in target_ids
        batch_size = target_ids.size(0)
        num_loss_tokens = (
            num_valid_tokens - batch_size
        )  # subtract batch_size due to internal label shift
        nll_sum += neg_log_likelihood * num_loss_tokens
        n_tokens += num_loss_tokens
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break
    avg_nll = nll_sum / n_tokens  # average negative log-likelihood per token
    ppl = torch.exp(avg_nll)
    return ppl.item()


def _load_encodings(tokenizer, dataset_name, dataset_config, split, max_samples):
    """Load the evaluation dataset and tokenize the concatenated texts."""
    # For C4, dataset_config is treated as data_files.
    if dataset_name == "allenai/c4":
        test_dataset = load_dataset(dataset_name, data_files=dataset_config, split=split)
    else:
        test_dataset = load_dataset(dataset_name, dataset_config, split=split)

    # Limit the number of samples
    if max_samples is not None:
        test_dataset = test_dataset.select(range(min(max_samples, len(test_dataset))))
    # Concatenate texts
    return tokenizer("\n\n".join(test_dataset["text"]), return_tensors="pt")


class _ReplayBlock(torch.nn.Module):
    """Stand-in for a transformer block that returns precomputed hidden states.

    Attribute reads fall back to the replaced block (like ``Catcher``), which
    is not registered as a submodule so it is not moved with the model.
    """

    def __init__(self, block, as_tuple):
        super().__init__()
        object.__setattr__(self, "_block", block)
        self.as_tuple = as_tuple
        self.hidden_states = None

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(object.__getattribute__(self, "_block"), name)

    def forward(self, hidden_states, *args, **kwargs):  # pylint: disable=unused-argument
        if self.hidden_states is not None:
            hidden_states = self.hidden_states
        return (hidden_states,) if self.as_tuple else hidden_states


@torch.no_grad()
def calculate_perplexity_offloaded(
    model,
    tokenizer,
    device,
    dataset_name="Salesforce/wikitext",
    dataset_config="wikitext-2-raw-v1",
    split="test",
    max_samples=None,
    max_length=2048,
    stride=2048,
    batch_size=8,
):  # pylint: disable=too-many-arguments, too-many-positional-arguments, too-many-locals
    """Perplexity of a model that does not fit on *device*.

    Uses the same windows and loss as :func:`calculate_perplexity`, but keeps
    the model on CPU and moves one transformer block at a time to *device*:
    every window is pushed through block ``i`` before block ``i + 1`` is
    loaded, so each block is transferred once.  Lazily dequantized layers
    (``NVFP4Linear``) are materialized while their block is on the device.

    Args:
        model: Model on CPU.
        tokenizer: Tokenizer.
        device: Device to run the blocks on.
        batch_size (int): Number of windows forwarded through a block at once.
        Other arguments: See :func:`calculate_perplexity`.

    Returns:
        float: Perplexity.
    """
    from .blockwise import _get_blocks, forward_input, get_blocks_and_inputs
    from .modelopt_checkpoint import materialize_lazy_weights, release_lazy_weights
    from .offload import temporarily_on_device

    device = torch.device(device)
    encodings = _load_encodings(tokenizer, dataset_name, dataset_config, split, max_samples)
    if max_length is None:
        max_length = model.config.max_position_embeddings
    if stride is None:
        stride = max_length
    seq_len = encodings.input_ids.size(1)

    # Same windows as calculate_perplexity: (input_ids, target_ids)
    windows = []
    prev_end_loc = 0
    for begin_loc in range(0, seq_len, stride):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end_loc
        input_ids = encodings.input_ids[:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        windows.append((input_ids, target_ids))
        prev_end_loc = end_loc
        if end_loc == seq_len:
            break

    # Batches of equal-length windows (only the last window may be shorter).
    batches = []
    for i in range(0, len(windows), batch_size):
        chunk = list(range(i, min(i + batch_size, len(windows))))
        lengths = {windows[j][0].size(1) for j in chunk}
        if len(lengths) == 1:
            batches.append(chunk)
        else:
            batches.extend([j] for j in chunk)

    blocks = _get_blocks(model)
    with temporarily_on_device(model, device, exclude=blocks):
        # 1. Inputs of the first block for every batch of windows.
        batch_states = []
        for batch in batches:
            input_ids = torch.cat([windows[j][0] for j in batch]).to(device)
            model_inputs = add_model_specific_inputs({"input_ids": input_ids}, model)
            _, inps, kwargs = get_blocks_and_inputs(model, model_inputs, len(batch))
            batch_states.append((inps, kwargs))

        # 2. Push all windows through each block in turn.
        output_is_tuple = []
        handle = blocks[-1].register_forward_hook(
            lambda _m, _i, out: output_is_tuple.append(isinstance(out, tuple))
        )
        try:
            for block in tqdm(blocks, desc="Blocks"):
                with temporarily_on_device(block, device):
                    materialize_lazy_weights(block)
                    for i, (inps, kwargs) in enumerate(batch_states):
                        batch_states[i] = (
                            forward_input(inps, block, kwargs, batch_size, device),
                            kwargs,
                        )
                    release_lazy_weights(block)
        finally:
            handle.remove()
        as_tuple = bool(output_is_tuple) and output_is_tuple[0]

        # 3. Final norm, LM head and loss, with the blocks replaced by replays.
        original_blocks = list(blocks)
        replays = [_ReplayBlock(b, as_tuple) for b in original_blocks]
        for i, replay in enumerate(replays):
            blocks[i] = replay
        try:
            nll_sum = torch.tensor(0.0, dtype=torch.float64)
            n_tokens = 0
            for batch, (hidden, _) in zip(batches, batch_states):
                for pos, j in enumerate(batch):
                    input_ids, target_ids = (t.to(device) for t in windows[j])
                    replays[0].hidden_states = hidden[pos : pos + 1].to(device)
                    inputs = add_model_specific_inputs(
                        inputs={"input_ids": input_ids, "labels": target_ids}, model=model
                    )
                    loss = model(**inputs).loss.to(torch.float64).cpu()
                    num_loss_tokens = (target_ids != -100).sum().item() - target_ids.size(0)
                    nll_sum += loss * num_loss_tokens
                    n_tokens += num_loss_tokens
                replays[0].hidden_states = None
        finally:
            for i, block in enumerate(original_blocks):
                blocks[i] = block

    return torch.exp(nll_sum / n_tokens).item()
