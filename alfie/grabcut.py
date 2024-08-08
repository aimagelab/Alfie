import cv2
import torch.nn.functional as F
import logging
from accelerate.logging import get_logger
import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def grabcut(image, attention_maps, image_size, sure_fg_threshold, maybe_fg_threshold, maybe_bg_threshold):

    sure_fg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    maybe_fg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    maybe_bg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    sure_bg_full_mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    for attention_map in attention_maps:
        attention_map = F.interpolate(
            attention_map[None, None, :, :].float(), size=(image_size, image_size), mode='bicubic')[0, 0]

        threshold_sure_fg = sure_fg_threshold * attention_map.max()
        threshold_maybe_fg = maybe_fg_threshold * attention_map.max()
        threshold_maybe_bg = maybe_bg_threshold * attention_map.max()
        sure_fg_full_mask += (attention_map > threshold_sure_fg).to(torch.uint8)
        maybe_fg_full_mask += ((attention_map > threshold_maybe_fg) & (attention_map <= threshold_sure_fg)).to(torch.uint8)
        maybe_bg_full_mask += ((attention_map > threshold_maybe_bg) & (attention_map <= threshold_maybe_fg)).to(torch.uint8)
        sure_bg_full_mask += (attention_map <= threshold_maybe_bg).to(torch.uint8)

    mask = torch.zeros((image_size, image_size), dtype=torch.uint8)
    mask = torch.where(sure_bg_full_mask.bool(), cv2.GC_BGD, mask)
    mask = torch.where(maybe_bg_full_mask.bool(), cv2.GC_PR_BGD, mask)
    mask = torch.where(maybe_fg_full_mask.bool(), cv2.GC_PR_FGD, mask)
    mask = torch.where(sure_fg_full_mask.bool(), cv2.GC_FGD, mask)

    bgdModel = np.zeros((1, 65), np.float64)
    fgdModel = np.zeros((1, 65), np.float64)
    mask = mask.numpy().astype(np.uint8)
    try:
        mask, bgdModel, fgdModel = cv2.grabCut(np.array(image), mask, None, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_MASK)
    except:
        Warning(f'Grabcut failed, using default mask and mode={cv2.GC_INIT_WITH_MASK}')
        mask = np.zeros_like(mask)
        center_rect = (128, 128, 384, 384)
        mask, bgdModel, fgdModel = cv2.grabCut(np.array(image), mask, center_rect, bgdModel, fgdModel, 5, cv2.GC_INIT_WITH_RECT)

    alpha = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')

    return alpha


def save_rgba(rgb, alpha, path):
    alpha = alpha * 255
    if isinstance(alpha, torch.Tensor):
        alpha = np.array(alpha.cpu())
    alpha = alpha.clip(0, 255).astype(np.uint8)
    alpha = Image.fromarray(alpha, mode='L')
    rgb = rgb.copy()
    rgb.putalpha(alpha)
    rgb.save(path)

