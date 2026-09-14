import random

import pytest
import torch
from sglang.srt.layers.moe.fused_moe_triton.fused_marlin_moe import (
    _canonicalize_marlin_token_order,
)
from sglang.test.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=5, stage="base-b-kernel-unit", runner_config="1-gpu-large")


def _case(numel, block_size):
    rng = random.Random(17)
    routes = [rng.choice([-1, 0, 2, 64, 128]) for _ in range(numel)]
    expected, shuffled, experts = [], [], []
    for expert in sorted(set(routes)):
        pairs = [index for index, route in enumerate(routes) if route == expert]
        padding = (-len(pairs)) % block_size
        expected.extend(pairs + [numel] * padding)
        rng.shuffle(pairs)
        shuffled.extend(pairs + [numel] * padding)
        experts.extend([expert] * ((len(pairs) + padding) // block_size))
    used = len(expected)
    # Tail entries are deliberately invalid; the original align kernel leaves
    # unused expert entries undefined, and some paths also leave token tails.
    shuffled.extend([-2147483648] * (block_size + 3))
    experts.extend([-2147483648, 2147483647])
    return (
        torch.tensor(shuffled, device="cuda", dtype=torch.int32),
        torch.tensor(experts, device="cuda", dtype=torch.int32),
        torch.tensor([used], device="cuda", dtype=torch.int32),
        torch.tensor(
            expected + [numel] * (block_size + 3), device="cuda", dtype=torch.int32
        ),
    )


@pytest.mark.parametrize(
    "numel,block_size", [(0, 8), (1, 8), (43 * 9, 8), (1031, 32), (4096, 64)]
)
def test_marlin_token_order_preserves_experts_padding_and_filtered_pairs(
    numel, block_size
):
    tokens, experts, total, expected = _case(numel, block_size)
    before = tuple(t.clone() for t in (tokens, experts, total))
    actual = _canonicalize_marlin_token_order(tokens, experts, total, block_size, numel)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for current, original in zip((tokens, experts, total), before):
        torch.testing.assert_close(current, original, rtol=0, atol=0)


def test_marlin_token_order_cuda_graph_replay():
    numel, block_size = 387, 8
    tokens, experts, total, expected = _case(numel, block_size)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            _canonicalize_marlin_token_order(tokens, experts, total, block_size, numel)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        actual = _canonicalize_marlin_token_order(
            tokens, experts, total, block_size, numel
        )
    for _ in range(3):
        graph.replay()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
