"""Unit tests for provider logprob serialization helpers (no live API)."""

from types import SimpleNamespace

from llm.providers import _serialize_gemini_logprobs_result


def test_gemini_logprobs_maps_to_openai_shape():
    lr = SimpleNamespace(
        chosen_candidates=[
            SimpleNamespace(token="Hello", log_probability=-0.05),
        ],
        top_candidates=[
            SimpleNamespace(
                candidates=[
                    SimpleNamespace(token="Hello", log_probability=-0.05),
                    SimpleNamespace(token="Hi", log_probability=-1.2),
                ]
            ),
        ],
    )
    out = _serialize_gemini_logprobs_result(lr)
    assert out is not None
    assert len(out) == 1
    assert out[0]["token"] == "Hello"
    assert out[0]["logprob"] == -0.05
    assert out[0]["bytes"] is None
    assert out[0]["top_logprobs"]["Hello"] == -0.05
    assert out[0]["top_logprobs"]["Hi"] == -1.2
