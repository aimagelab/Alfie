from typing import Optional, List

import torch
import torch.nn.functional as F
from einops import rearrange

from .cross_heatmap import CrossRawHeatMapCollection, CrossGlobalHeatMap
from .self_heatmap import SelfRawHeatMapCollection, SelfGlobalHeatMap
from .utils import auto_autocast

from diffusers.utils import deprecate, logging
from diffusers.models.attention import Attention

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


class AttnProcessor2_0:
    r"""
    Processor for implementing scaled dot-product attention (enabled by default if you're using PyTorch 2.0).
    """

    def __init__(self, keep_cross_attn_maps: bool = True, keep_self_attn_maps: bool = True, tokenizer = None):
        self.keep_cross_attn_maps = keep_cross_attn_maps
        self.ca_maps_fg = CrossRawHeatMapCollection()
        self.keep_self_attn_maps = keep_self_attn_maps
        self.sa_maps_fg = SelfRawHeatMapCollection()

        self.h = self.w = 32
        self.num_prompt_tokens = None
        self.l_iteration_ca = 0
        self.l_iteration_sa = 0
        self.t = 0

        self.tokenizer = tokenizer

    def __call__(
        self,
        attn: Attention,
        hidden_states: torch.FloatTensor,
        encoder_hidden_states: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        temb: Optional[torch.FloatTensor] = None,
        *args,
        **kwargs,
    ) -> torch.FloatTensor:
        if len(args) > 0 or kwargs.get("scale", None) is not None:
            deprecation_message = "The `scale` argument is deprecated and will be ignored. Please remove it, as passing it will raise an error in the future. `scale` should directly be passed while calling the underlying pipeline component i.e., via `cross_attention_kwargs`."
            deprecate("scale", "1.0.0", deprecation_message)

        residual = hidden_states
        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

        batch_size, sequence_length, _ = (
            hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
        )

        if attention_mask is not None:
            attention_mask = attn.prepare_attention_mask(attention_mask, sequence_length, batch_size)
            # scaled_dot_product_attention expects attention_mask shape to be
            # (batch, heads, source_length, target_length)
            # attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

        query = attn.to_q(hidden_states)

        if encoder_hidden_states is None:
            encoder_hidden_states = hidden_states
        elif attn.norm_cross:
            encoder_hidden_states = attn.norm_encoder_hidden_states(encoder_hidden_states)

        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)

        query = attn.head_to_batch_dim(query)
        key = attn.head_to_batch_dim(key)
        value = attn.head_to_batch_dim(value)
        attention_scores = attn.get_attention_scores(query, key, attention_mask=attention_mask)

        if attn.is_cross_attention and self.keep_cross_attn_maps and self.t > 20:
            maps = rearrange(attention_scores[:, :, :self.num_prompt_tokens], "(b heads) (h w) l -> b heads l h w", b=batch_size, h=self.h)
            maps_fg = maps[-1]  # filter out uncoditionals and background prompt
            self.ca_maps_fg.update(self.l_iteration_ca, maps_fg)
            self.l_iteration_ca += 1
        if not attn.is_cross_attention and self.keep_self_attn_maps and self.t > 20:
            maps = rearrange(attention_scores, '(b n) (h1 w1) (h2 w2) -> b n (h1 w1) h2 w2', b=batch_size, h1=self.h, h2=self.h)
            maps_fg = maps[-1]  # filter out uncoditionals and background prompt
            self.sa_maps_fg.update(self.l_iteration_sa, maps_fg)
            self.l_iteration_sa += 1

        hidden_states = attention_scores @ value
        hidden_states = attn.batch_to_head_dim(hidden_states).to(query.dtype)

        # linear proj
        hidden_states = attn.to_out[0](hidden_states)
        # dropout
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(batch_size, channel, height, width)

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states
    

    def compute_global_cross_heat_map(self, prompt: str, layers: List[int] = None, normalize: bool=False) -> CrossGlobalHeatMap:
        """
        Compute the global heat map for the given prompt, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            prompt: The prompt to compute the heat map for. If none, uses the last prompt that was used for generation.
            layers: Restrict the application to heat maps with this layers indexes in this set. If `None`, use all layers.
            normalize: Whether to normalize the heat map to sum to 1.

        Returns:
            A heat map object for computing word-level heat maps.
        """
        heat_maps = self.ca_maps_fg
        layers = set(range(28)) if layers is None else set(layers)

        all_merges = []

        with auto_autocast(device_type=heat_maps.device_type, dtype=torch.float32):
            for (layer), heat_map in heat_maps:
                if layer in layers:
                    # The clamping fixes undershoot.
                    all_merges.append(F.interpolate(heat_map, size=(self.h, self.w), mode='bicubic').clamp_(min=0))

            try:
                maps = torch.stack(all_merges, dim=0)
            except RuntimeError:
                raise RuntimeError('No heat maps found.')

            maps = maps.mean(dim=(0, 1))
            if normalize:
                maps = maps / (maps[:-1].sum(0, keepdim=True) + 1e-6)  # drop out[PAD] for proper probabilities

        return CrossGlobalHeatMap(self.tokenizer, prompt, maps)
    

    def compute_global_self_heat_map(self, layers: List[int] = None, normalize: bool =False) -> SelfGlobalHeatMap:
        """
        Compute the global heat map for each latent pixel, aggregating across time (inference steps) and space (different
        spatial transformer block heat maps).

        Args:
            layers: Restrict the application to heat maps with layers indexed in this set. If `None`, use all sizes.
            normalize: Whether to normalize the heat map to sum to 1.

        Returns:
            A heat map object for computing latent pixel-level heat maps.
        """
        heat_maps = self.sa_maps_fg
        layers = set(range(28)) if layers is None else set(layers)

        all_merges = []

        with auto_autocast(device_type=heat_maps.device_type, dtype=torch.float32):
            for (layer), heat_map in heat_maps:
                if layer in layers:
                    # The clamping fixes undershoot.
                    all_merges.append(F.interpolate(heat_map, size=(self.h, self.w), mode='bicubic').clamp_(min=0))
            try:
                maps = torch.stack(all_merges, dim=0)
            except RuntimeError:
                raise RuntimeError('No heat maps found.')

            maps = maps.mean(dim=(0, 1))
            if normalize:
                maps = maps / (maps.sum(0, keepdim=True) + 1e-6)  # drop out [SOS] and [PAD] for proper probabilities

        return SelfGlobalHeatMap(maps, maps.shape[0])