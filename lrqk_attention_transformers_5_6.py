import functools

import transformers.cache_utils as cache_utils

import lrqk_attention_transformers_4_52 as _base


def _patch_cache_init_for_tf5():
    init_fn = cache_utils.Cache.__init__
    if getattr(init_fn, "__name__", "") == "_lrqk_tf5_cache_init_compat":
        return

    @functools.wraps(init_fn)
    def _lrqk_tf5_cache_init_compat(
        self,
        layers=None,
        layer_class_to_replicate=None,
        offloading=False,
        offload_only_non_sliding=True,
    ):
        if layers is None and layer_class_to_replicate is None:
            layers = []
        return init_fn(
            self,
            layers=layers,
            layer_class_to_replicate=layer_class_to_replicate,
            offloading=offloading,
            offload_only_non_sliding=offload_only_non_sliding,
        )

    cache_utils.Cache.__init__ = _lrqk_tf5_cache_init_compat


def _patch_attention_forward_kwargs(cls):
    orig_forward = cls.forward
    if getattr(orig_forward, "__name__", "") == "_lrqk_tf5_forward_compat":
        return

    @functools.wraps(orig_forward)
    def _lrqk_tf5_forward_compat(
        self,
        hidden_states,
        position_embeddings,
        attention_mask,
        past_key_value=None,
        cache_position=None,
        **kwargs,
    ):
        if past_key_value is None:
            past_key_value = kwargs.get("past_key_values", None)
        return orig_forward(
            self,
            hidden_states,
            position_embeddings,
            attention_mask,
            past_key_value=past_key_value,
            cache_position=cache_position,
            **kwargs,
        )

    cls.forward = _lrqk_tf5_forward_compat


_patch_cache_init_for_tf5()

for _cls in [
    _base.LRQK_Qwen2Attention,
    _base.LRQK_Qwen3Attention,
    _base.LRQK_LlamaAttention,
    _base.LRQK_MistralAttention,
    _base.LRQK_Phi3Attention,
]:
    _patch_attention_forward_kwargs(_cls)


for _name in dir(_base):
    if not _name.startswith("_"):
        globals()[_name] = getattr(_base, _name)
