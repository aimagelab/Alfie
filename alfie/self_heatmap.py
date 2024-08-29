from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import List, Any, Dict, Tuple, Set, Iterable, Union, Optional
from tqdm import tqdm

from matplotlib import pyplot as plt
import numpy as np
import math
import PIL.Image
import cv2
import torch
import torch.nn.functional as F

from .utils import auto_autocast

__all__ = ['SelfGlobalHeatMap', 'SelfRawHeatMapCollection', 'SelfPixelHeatMap', 'SelfParsedHeatMap', 'SelfSyntacticHeatMapPair']


def plot_overlay_heat_map(im: PIL.Image.Image | np.ndarray, heat_map: torch.Tensor, figsize: Tuple[int, int] = (10,10)):
    with auto_autocast(dtype=torch.float32):
        plt.figure(figsize=figsize)
        plt.axis('off')
        im = np.array(im)
        plt.imshow(heat_map.squeeze().cpu().numpy(), cmap='jet')

        im = torch.from_numpy(im).float() / 255
        im = torch.cat((im, (1 - heat_map.unsqueeze(-1))), dim=-1)
        plt.imshow(im)


class SelfPixelHeatMap:
    def __init__(self, heatmap: torch.Tensor):
        self.heatmap = heatmap

    @property
    def value(self):
        return self.heatmap

    def plot_overlay(self, image, figsize: Tuple[int, int] = (10,10)):
        # type: (PIL.Image.Image | np.ndarray, Path, bool, plt.Axes, Dict[str, Any]) -> None
        plot_overlay_heat_map(image, self.expand_as(image), figsize)

    def expand_as(self, image):
        # type: (PIL.Image.Image, bool, float, bool, Dict[str, Any]) -> torch.Tensor

        im = self.heatmap.unsqueeze(0).unsqueeze(0)
        im = F.interpolate(im.float().detach(), size=(image.size[0], image.size[1]), mode='bicubic')
        im = im[0,0]
        im = (im - im.min()) / (im.max() - im.min() + 1e-8)
        im = im.cpu().detach().squeeze()
        
        return im


@dataclass
class SelfSyntacticHeatMapPair:
    head_heat_map: SelfPixelHeatMap
    dep_heat_map: SelfPixelHeatMap
    head_text: str
    dep_text: str
    relation: str


@dataclass
class SelfParsedHeatMap:
    word_heat_map: SelfPixelHeatMap

class SelfGlobalHeatMap:
    def __init__(self, heat_maps: torch.Tensor, latent_hw: int):
        self.heat_maps = heat_maps.cpu()
        # The dimensions of the latent image on which the heatmap is generated
        self.latent_h = self.latent_w = int(math.sqrt(latent_hw)) 
        # The pixels for which the heatmap is generated (It can be condensed form and thus smaller compared to self.latent_h and latent_w)
        self.inner_latent_h = self.inner_latent_w = int(math.sqrt(heat_maps.shape[0]))
        # Scale Factor
        self.scale_factor = self.latent_h // self.inner_latent_h

    def scale_correction(self, pixel_ids):
        """
        scales the pixel ids according to the pixels of the inner latent image dim i.e. self.inner_latent_h and self.inner_latent_w
        """
        pixel_ids_2d = [(p_id // self.latent_w, p_id % self.latent_w) for p_id in pixel_ids]
        inner_pixel_ids_2d = [(x // self.scale_factor, y // self.scale_factor) for x, y in pixel_ids_2d]
        scaled_pixel_ids = [x * self.inner_latent_w + y for x, y in inner_pixel_ids_2d]
        return scaled_pixel_ids

    def compute_pixel_heat_map(self, latent_pixels: Union[List[int], int]) -> SelfPixelHeatMap:
        """
        Given a list of pixels or pixel id it returns the heatmap for that pixel or mean of all the heatmaps corresponding
        to those pixels.
        The pixel ids should adhere to row-major latent image representation i.e.
        0 1 ... 63
        ..........
        4032...4095
        for SDV2
        """
        if isinstance(latent_pixels, list):
            # scale correction
            merge_idxs = self.scale_correction(latent_pixels)

            return SelfPixelHeatMap(self.heat_maps[merge_idxs].mean(0))
        else:
            merge_idx = self.scale_correction([latent_pixels])
            return SelfPixelHeatMap(self.heat_maps[merge_idx].mean(0))

    def compute_bbox_heat_map(self, x1: int, y1: int, x2: int, y2: int) -> SelfPixelHeatMap:
        """
        Given the top-left coordinates (x1,y1) and bottom-right coordinates (x2,y2) it returns the heatmap for the 
        mean of all the pixels lying inside this bbox.
        These coordinates should be for the latent image
        """
        if x2 < x1 or y2 < y1:
            raise Exception('Enter valid bounding box! (x1,y1) is the top-left corner and (x2,y2) is the bottom-right corner.')
        pix_ids = [x for y in range(y1, y2+1) for x in range((self.latent_w * y) + x1, (self.latent_w * y) + x2 + 1) if x < (self.latent_h * self.latent_w)]
        # scale correction
        pix_ids = self.scale_correction(pix_ids)
        return SelfPixelHeatMap(self.heat_maps[pix_ids].mean(0))

    def inner_pixel_ids(self, 
        pts: Union[List[List[int]], List[int]], 
        image_h: int, image_w: int) -> List:
        """
        Given a contour pts in the  it finds the latent image pixels which lie inside this contour
        pts should be a represent a single polygon multi-piece polygon is not handled by this function, check out `segmentation_heat_map`
          pts should be be a list of [x,y] coordinates of the contour 
              or a list of [x1,y1,x2,y2,...,xn,yn] (i.e. same as [[x1,y1], [x2.y2], ...] just the inner lists are unravelled)
          image_h and image_w is the image height and width respectively of the original image from which contour is taken
        """
        if isinstance(pts[0], int):
            pts = [[pts[i], pts[i+1]] for i in range(0, len(pts), 2)]

        if image_h != image_w:
            raise Exception('Non-Square images not supported yet! `image_h` should be equal to `image_w')

        pts = np.array(np.array(pts) * self.latent_h / image_h, np.int32)
        pts = pts.reshape((-1,1,2))
        inner_pixs = list()
        for i in range(self.latent_h):
            for j in range(self.latent_w):
                dist = cv2.pointPolygonTest(pts, (i, j), False)
                if dist == 1.0:
                    inner_pixs.append((j*self.latent_w) + i)

        return inner_pixs # It returns the inner_pixs according to self.latent_h and self.latent_w needs to be scale corrected for use

    def compute_contour_heat_map(self, 
        pts: Union[List[List[int]], List[int]], 
        image_h: int, image_w: int,
        guide_heatmap: Optional[torch.tensor] = None) -> SelfPixelHeatMap:
          """
          pts should be a represent a single polygon multi-piece polygon is not handled by this function, check out `segmentation_heat_map`
          pts should be be a list of [x,y] coordinates of the contour 
              or a list of [x1,y1,x2,y2,...,xn,yn] (i.e. same as [[x1,y1], [x2.y2], ...] just the inner lists are unravelled)
          image_h and image_w is the image height and width respectively of the original image from which contour is taken
          guide_heatmap is the same as described in `compute_guided_heat_map`, in this case only the pixels of the `guide_heatmap` will be considered which are 
          contained inside the contour
          returns the heatmap for the mean of the pixels lying inside this contour
          """
          
          # get the latent pixel ids lying inside the contour
          inner_pixs = self.inner_pixel_ids(pts, image_h, image_w)
          # scale correction
          inner_pixs = self.scale_correction(inner_pixs)

          if guide_heatmap is None:
              return SelfPixelHeatMap(self.heat_maps[inner_pixs].mean(0))
          else:
              # Finding out the inner_pixs' each pixel's weight as obtained from `guide_heatmap`
              pix_weights = torch.tensor([guide_heatmap[pix_id // self.latent_w, pix_id % self.latent_w] for pix_id in inner_pixs])[:,None,None]
              pix_weights = pix_weights.cpu()
              heatmap = torch.zeros((self.latent_h, self.latent_w)).cpu()
              for idx, pix_id in enumerate(inner_pixs):
                  heatmap += (self.heat_maps[pix_id] * pix_weights[idx])
              heatmap /= pix_weights.sum().item()
              # return the weighted heatmap
              return SelfPixelHeatMap(heatmap)

    def compute_segmentation_heat_map(self, 
        segments: Union[List[List[List[int]]], List[List[int]]], 
        image_h: int, image_w: int, 
        guide_heatmap: Optional[torch.tensor] = None) -> SelfPixelHeatMap:
          """
          Pass in the list of contours like this [[x1,y1,x2,y2,....], [p1,q1,p2,q2,...], ..] or [[[x1,y1],[x2,y2],....], [[p1,q1],[p2,q2],...], ..]
          This finds the mean heatmap for all the pixel heatmaps for the pixels lying inside each of these contours together.
          segments: list of contours in the format explained above
          image_h: the height of the image according to which the `segments` is provided
          image_w: the width of the image according to which the `segments` is provided
          guide_heatmap: the weighing scheme for all the pixels in the latent image while merging pixel heat maps
          """
          segments_inner_pixs = list()
          for segment in segments:
              # Compute heatmap for inner pixels for contour boundary specified
              segment_inner_pixs = self.inner_pixel_ids(segment, image_h, image_w)

              segments_inner_pixs.extend(segment_inner_pixs)

          # scale correction
          segments_inner_pixs = self.scale_correction(segments_inner_pixs)

          if guide_heatmap is None:
              return SelfPixelHeatMap(self.heat_maps[segments_inner_pixs].mean(0))
          else:
              # Finding out the inner_pixs' each pixel's weight as obtained from `guide_heatmap`
              pix_weights = torch.tensor([guide_heatmap[pix_id // self.latent_w, pix_id % self.latent_w] for pix_id in segments_inner_pixs])[:,None,None]
              pix_weights = pix_weights.cpu()
              heatmap = torch.zeros((self.latent_h, self.latent_w)).cpu()
              for idx, pix_id in enumerate(segments_inner_pixs):
                  heatmap += (self.heat_maps[pix_id] * pix_weights[idx])
              heatmap /= pix_weights.sum().item()
              # return the weighted heatmap
              return SelfPixelHeatMap(heatmap)

    def compute_guided_heat_map(self, guide_heatmap: torch.tensor):
        """
        For each pixel in the latent image we have one heatmap. Now, with a guiding heatmap
        we can merge all these pixel heatmaps with a weighted average according to the weights 
        given to each pixel in the guiding heatmap. 

        guide_heatmap: A guiding heatmap of the dimension of the latent image. It should be a 2D torch.tensor
        """

        # convert the latent 2d image from height.width x height x width to 1 x height.weight x height x width
        # i.e. we add the batch dim
        heat_maps2d = self.heat_maps[None, :].clone()

        # weight of the convolution layer that performs attention diffusion (making a copy to prevent changing the heatmap)
        conv_weight = guide_heatmap.to(heat_maps2d.device).view(-1, self.latent_h * self.latent_w).clone()[:, :, None, None]

        # For getting weighted average after 1x1 Kernel convolution below
        conv_weight /= conv_weight.sum(1, keepdims=True)

        # Since `Half` is not supported on cpu, if the dtype is `Half` we do the computation is cuda
        if heat_maps2d.dtype == torch.float16 or conv_weight.dtype == torch.float16:
            # Aggregating all the heatmaps using convolution operation i.e. weighted average using `guide_heatmap` weights
            guided_heatmap = F.conv2d(heat_maps2d.cuda(), conv_weight.cuda())[0,0].cpu()
        else:
            # Aggregating all the heatmaps using convolution operation i.e. weighted average using `guide_heatmap` weights
            guided_heatmap = F.conv2d(heat_maps2d, conv_weight)[0,0]

        return SelfPixelHeatMap(guided_heatmap.cpu() * guide_heatmap.cpu())

    def compute_pixel_diffused_heat_map(self, 
        latent_pixel_id: int, 
        method: str = 'thresholding', 
        n_iter: int = 20, thresh: int = 0.02,
        plot_intermediate: bool = False):
        """
        For the given latent_pixel it iteratively reweights all the pixels of the image 
        based on the attention heatmap of this latent pixel. Now, for this updated
        heatmap we use it to reweight again to generate refined heatmap. This is done for 
        `n_iter` number of iterations
        latent_pixel_id: The latent pixel id from which the heatmap will be diffused throughout the latent image
        method: Currently only has `thresholding` where at each step to remove the misguiding/noisy pixels to
                prevent them from focussing/enhancing wrong heatmaps, we use simple thresholding
        n_iter: For how many interactions to refine the heatmap as described above
        plot_itermediate: If the intermediate heatmaps need to plotted to show the evolutation

        """
        # epicenter is the start of the attention heatmap diffusion
        epicenter = self.compute_pixel_heat_map(latent_pixel_id).heatmap
        if plot_intermediate:
            plt.imshow(epicenter) # Initial Pixel Heatmap
            plt.show()
            
        if method == 'thresholding':
            # We remove noisy pixels based on a threshold compared to the minimum attention weight
            # By setting them to 0
            epicenter[epicenter < epicenter.min()+thresh] = 0
            # Iterating
            for _ in range(n_iter):
                epicenter = self.compute_guided_heat_map(epicenter).heatmap
                if plot_intermediate:
                    plt.imshow(epicenter)
                    plt.show()
                epicenter[epicenter < epicenter.min()+thresh] = 0

        return SelfPixelHeatMap(epicenter)

    def compute_diffused_heat_map(self, 
        method: str = 'thresholding', 
        n_iter: int = 20, thresh: int = 0.02):
        """
        For the entire latent image it iteratively reweights all the pixels of the image 
        based on the attention heatmap of each pixel. Now, for this updated
        heatmaps we use it to reweight again to generate refined heatmaps. This is done for 
        `n_iter` number of iterations. Returns a SelfGlobalHeatMap collection of the attention diffused heatmaps for all pixels
        method: Currently only has `thresholding` where at each step to remove the misguiding/noisy pixels to
                prevent them from focussing/enhancing wrong heatmaps, we use simple thresholding
        n_iter: For how many interactions to refine the heatmap as described above

        """
        # convert the latent 2d image from height.width x height x weight to 1 x height.weight x height x weight
        heat_maps2d = self.heat_maps[None, :].clone()

        # weight of the convolution layer that performs attention diffusion (making a copy to prevent changing the heatmap)
        conv_weight = self.heat_maps.view(-1, self.latent_h * self.latent_w).clone()[:, :, None, None]

        if method == 'thresholding':
            # We remove noisy pixels based on a threshold compared to the minimum attention weight
            # By setting them to 0
            conv_weight[conv_weight < (torch.min(conv_weight, 1, keepdims=True)[0] + thresh)] = 0

            # Iterating
            for _ in tqdm(range(n_iter)):
                # For getting weighted average after 1x1 Kernel convolution below
                conv_weight /= conv_weight.sum(1, keepdims=True)
                # Aggregating all the heatmaps (Maybe conv_weight as input image instead of heat_maps2d could give faster convergence?
                # Need to use conv_weight corresponding to the updated heatmaps then)
                conv_weight = F.conv2d(heat_maps2d, conv_weight)[0].view(self.heat_maps.shape[0], self.heat_maps.shape[0])[:, :, None, None]
                # Cut off noisy values based on threshold
                conv_weight[conv_weight < (torch.min(conv_weight, 1, keepdims=True)[0] + thresh)] = 0

        return SelfGlobalHeatMap(conv_weight.view(self.heat_maps.shape[0], self.latent_h, self.latent_w), self.latent_h*self.latent_w)


RawHeatMapKey = Tuple[int]  # layer


class SelfRawHeatMapCollection:
    def __init__(self):
        self.ids_to_heatmaps: Dict[RawHeatMapKey, torch.Tensor] = defaultdict(lambda: 0.0)
        self.ids_to_num_maps: Dict[RawHeatMapKey, int] = defaultdict(lambda: 0)
        self.device_type = None

    def update(self, layer_idx: int, heatmap: torch.Tensor):
        if self.device_type is None:
            self.device_type = heatmap.device.type
        with auto_autocast(device_type=self.device_type, dtype=torch.float32):
            key = (layer_idx)
            # Instead of simple addition can we do something better ???
            self.ids_to_heatmaps[key] = self.ids_to_heatmaps[key] + heatmap

    def factors(self) -> Set[int]:
        return set(key[0] for key in self.ids_to_heatmaps.keys())

    def layers(self) -> Set[int]:
        return set(key[1] for key in self.ids_to_heatmaps.keys())

    def heads(self) -> Set[int]:
        return set(key[2] for key in self.ids_to_heatmaps.keys())

    def __iter__(self):
        return iter(self.ids_to_heatmaps.items())

    def clear(self):
        self.ids_to_heatmaps.clear()
        self.ids_to_num_maps.clear()
