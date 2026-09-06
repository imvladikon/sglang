import struct

import pybase64
from sglang_router.mini_lb import _merge_prefill_json


def _encode(values):
    return pybase64.b64encode(struct.pack(f"<{len(values)}i", *values)).decode()


def _decode(value):
    raw = pybase64.b64decode(value, validate=True)
    return struct.unpack(f"<{len(raw) // 4}i", raw)


def test_merge_prefill_json_preserves_complete_r3_and_logprob_metadata():
    prefill = {
        "meta_info": {
            "input_token_logprobs": [[-1.0, 1, None]],
            "routed_experts": _encode([1, 2]),
        },
        "sglext": {"routed_experts": _encode([5, 6])},
    }
    decode = {
        "meta_info": {
            "input_token_logprobs": [[-2.0, 2, None]],
            "routed_experts": _encode([9, 9, 3, 4]),
        },
        "sglext": {"routed_experts": _encode([9, 9, 7, 8])},
    }

    _merge_prefill_json(prefill, decode)

    assert decode["meta_info"]["input_token_logprobs"] == [
        [-1.0, 1, None],
        [-2.0, 2, None],
    ]
    assert _decode(decode["meta_info"]["routed_experts"]) == (1, 2, 3, 4)
    assert _decode(decode["sglext"]["routed_experts"]) == (5, 6, 7, 8)


def test_merge_prefill_json_leaves_decode_metadata_without_prefill_r3_unchanged():
    decode = {
        "meta_info": {
            "input_token_logprobs": [],
            "routed_experts": _encode([3, 4]),
        }
    }

    _merge_prefill_json({"meta_info": {}}, decode)

    assert _decode(decode["meta_info"]["routed_experts"]) == (3, 4)
