import numpy as np
import torch
import torch.nn.functional as F


def compute_trimap_(attention_maps, image_size, sure_fg_threshold, maybe_bg_threshold):
    sure_fg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    unsure = torch.zeros((image_size, image_size), dtype=torch.uint8)
    sure_bg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    for attention_map in attention_maps:
        attention_map = F.interpolate(
            attention_map[None, None, :, :].float(), size=(image_size, image_size), mode='bicubic')[0, 0]

        threshold_sure_fg = sure_fg_threshold * attention_map.max()
        threshold_maybe_bg = maybe_bg_threshold * attention_map.max()
        sure_fg_full_mask += (attention_map > threshold_sure_fg)
        unsure += ((attention_map > maybe_bg_threshold) & (attention_map <= threshold_sure_fg))
        sure_bg_full_mask += (attention_map <= threshold_maybe_bg)

    mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    mask = torch.where(sure_bg_full_mask, 255, mask)
    mask = torch.where(unsure, 128, mask)
    mask = torch.where(sure_fg_full_mask, 0, mask)
    return mask


def compute_trimap(attention_maps, image_size, sure_fg_threshold, maybe_bg_threshold):
    if isinstance(attention_maps, list):
        masks = [
            compute_trimap_(attention_map, image_size, sure_fg_threshold, maybe_bg_threshold) for attention_map in
            attention_maps]
        return torch.stack(masks)
    else:
        return compute_trimap_(attention_maps, image_size, sure_fg_threshold, maybe_bg_threshold)[None]
