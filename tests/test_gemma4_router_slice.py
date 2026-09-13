"""gemma-4's router is the one REAP slices through a child module.

Gemma4TextRouter has no `.weight`: a `proj` Linear carries the expert projection
and a separate per_expert_scale vector scales the top-k weights. Miss either and
the pruned checkpoint loads, runs, and quietly mis-routes -- so this asserts both
are narrowed, and that the router still produces the right shapes afterwards.
"""
import torch

from reap.prune import slice_wrapped_router


def _tiny_router(num_experts=8, hidden=16, top_k=2):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRouter
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    cfg = Gemma4TextConfig(
        hidden_size=hidden, num_experts=num_experts, top_k_experts=top_k,
        intermediate_size=32, moe_intermediate_size=8, num_hidden_layers=1,
        num_attention_heads=2, num_key_value_heads=1, vocab_size=32,
    )
    return Gemma4TextRouter(cfg), cfg


def test_router_slice_narrows_proj_and_per_expert_scale():
    router, _ = _tiny_router(num_experts=8)
    with torch.no_grad():          # distinct values so a wrong axis is visible
        router.per_expert_scale.copy_(torch.arange(8, dtype=router.per_expert_scale.dtype))
    keep = [0, 2, 4, 6]

    slice_wrapped_router(router, keep)

    assert router.proj.weight.shape[0] == 4, router.proj.weight.shape
    assert router.proj.out_features == 4
    assert router.per_expert_scale.shape == (4,)
    assert router.per_expert_scale.tolist() == [0.0, 2.0, 4.0, 6.0]
    assert router.config.num_experts == 4


def test_sliced_router_still_routes():
    router, _ = _tiny_router(num_experts=8, hidden=16, top_k=2)
    slice_wrapped_router(router, [1, 3, 5, 7])
    probs, top_k_weights, top_k_index = router(torch.randn(5, 16))
    assert probs.shape == (5, 4), probs.shape          # narrowed expert axis
    assert top_k_index.max().item() < 4                 # never indexes a pruned expert
    assert torch.isfinite(top_k_weights).all()


if __name__ == "__main__":
    test_router_slice_narrows_proj_and_per_expert_scale()
    test_sliced_router_still_routes()
    print("ok")
