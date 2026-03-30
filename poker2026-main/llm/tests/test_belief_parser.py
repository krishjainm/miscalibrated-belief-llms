"""Tests for belief JSON parsing."""

from analysis.buckets import BUCKET_NAMES
from llm.belief_parser import parse_bucket_belief, repair_nonnegative_l1


def _uniform():
    v = 1.0 / len(BUCKET_NAMES)
    return {b: v for b in BUCKET_NAMES}


def test_parse_direct_json():
    d = _uniform()
    raw = '{"' + '","'.join(f'{k}": {v}' for k, v in d.items()) + "}"
    # simpler
    import json

    raw = json.dumps(d)
    out, err = parse_bucket_belief(raw, cot_mode=False)
    assert err is None
    assert out is not None
    assert abs(sum(out.values()) - 1.0) < 1e-6


def test_parse_cot_section():
    import json

    d = _uniform()
    body = json.dumps(d)
    raw = f"REASONING: blah blah.\nPROBABILITIES:\n```json\n{body}\n```"
    out, err = parse_bucket_belief(raw, cot_mode=True)
    assert err is None
    assert out is not None


def test_repair_negative():
    d = _uniform()
    k0 = BUCKET_NAMES[0]
    d[k0] = -0.1
    fixed = repair_nonnegative_l1(d)
    assert all(v >= 0 for v in fixed.values())
    assert abs(sum(fixed.values()) - 1.0) < 1e-6
