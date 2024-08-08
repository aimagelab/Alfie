from pathlib import Path

from .pipeline_pixart_sigma import PixArtSigmaPipeline
from .transformer_2d import Transformer2DModel
from transformers import T5EncoderModel, T5Tokenizer
from diffusers.models import AutoencoderKL
from diffusers.schedulers import DPMSolverMultistepScheduler, EulerAncestralDiscreteScheduler, EulerDiscreteScheduler
import nltk

import argparse
import logging
from accelerate.logging import get_logger

import numpy as np
import torch

torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def download_nltk_data():
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')

    try:
        nltk.data.find('taggers/averaged_perceptron_tagger')
    except LookupError:
        nltk.download('averaged_perceptron_tagger')


def get_pipe(image_size, scheduler, device):
    download_nltk_data()
    if image_size == 256:
        model_key = "PixArt-alpha/PixArt-Sigma-XL-2-256x256"
    elif image_size == 512:
        model_key = "PixArt-alpha/PixArt-Sigma-XL-2-512-MS"
    else:
        raise ValueError(f"Invalid image size: {image_size}")
    pipeline_key = "PixArt-alpha/PixArt-Sigma-XL-2-1024-MS"

    text_encoder = T5EncoderModel.from_pretrained(
        pipeline_key,
        subfolder="text_encoder",
        use_safetensors=True,
        torch_dtype=torch.float16)

    tokenizer = T5Tokenizer.from_pretrained(pipeline_key, subfolder="tokenizer")

    vae = AutoencoderKL.from_pretrained(
        pipeline_key,
        subfolder="vae",
        use_safetensors=True,
        torch_dtype=torch.float16)

    model = Transformer2DModel.from_pretrained(
        model_key,
        subfolder="transformer",
        use_safetensors=True,
        torch_dtype=torch.float16)

    for param in text_encoder.parameters():
        param.requires_grad = False
    for param in vae.parameters():
        param.requires_grad = False
    for param in model.parameters():
        param.requires_grad = False

    text_encoder.eval()
    vae.eval()
    model.eval()
    dpm_scheduler = DPMSolverMultistepScheduler.from_pretrained(pipeline_key, subfolder="scheduler")
    if scheduler == 'euler':
        eul_scheduler = EulerDiscreteScheduler.from_config(dpm_scheduler.config)
    elif scheduler == 'euler_ancestral':
        eul_scheduler = EulerAncestralDiscreteScheduler.from_config(dpm_scheduler.config)
    else:
        raise ValueError(f"Invalid scheduler: {scheduler}")

    pipe = PixArtSigmaPipeline.from_pretrained(
        pipeline_key, transformer=model, text_encoder=text_encoder, vae=vae, tokenizer=tokenizer,
        scheduler=eul_scheduler).to(device)

    return pipe


def base_arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--centering", type=str, default='True')
    parser.add_argument("--resample", type=str, default='True')
    parser.add_argument("--scheduler", type=str, default='euler', choices=['euler', 'euler_ancestral'])
    parser.add_argument("--use_neg_prompt", type=str, default='True')
    parser.add_argument("--save_folder", type=str, default='images')
    parser.add_argument("--exclude_generic_nouns", type=str, default='True')
    parser.add_argument("--image_size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--cutout_model", type=str, default='grabcut', choices=['grabcut', 'vit-matte', 'sam'])
    parser.add_argument("--sure_fg_threshold", type=int, default=0.8)
    parser.add_argument("--maybe_fg_threshold", type=int, default=0.3)
    parser.add_argument("--maybe_bg_threshold", type=int, default=0.1)
    parser.add_argument("--num_images", type=int, default=3)
    parser.add_argument("--use_suffix", type=str, default='False', help='Add the suffix on a white background')
    parser.add_argument("--seed", type=int, default=2024, help="random seed")
    parser.add_argument("--vit_matte_key", type=str, default='hustvl/vitmatte-base-composition-1k')
    parser.add_argument("--nouns_to_exclude", nargs='+', default=[
        'image', 'images', 'picture', 'pictures', 'photo', 'photograph', 'photographs', 'illustration',
        'paintings', 'drawing', 'drawings', 'sketch', 'sketches', 'art', 'arts', 'artwork', 'artworks',
        'poster', 'posters', 'cover', 'covers', 'collage', 'collages', 'design', 'designs', 'graphic', 'graphics',
        'logo', 'logos', 'icon', 'icons', 'symbol', 'symbols', 'emblem', 'emblems', 'badge', 'badges', 'stamp',
        'stamps', 'img', 'video', 'videos', 'clip', 'clips', 'film', 'films', 'movie', 'movies', 'meme', 'grand', 
        'sticker', 'stickers', 'banner', 'banners', 'billboard', 'billboards', 'label', 'labels', 'scene', 'art',
        'png', 'jpg', 'jpeg', 'gif', 'www', 'com', 'net', 'org', 'http', 'https', 'html', 'css', 'js', 'php', 
        'scene', 'view', 'm3'])

    return parser


def parse_bool_args(args):
    args.centering = args.centering.lower() == 'true'
    args.exclude_generic_nouns = args.exclude_generic_nouns.lower() == 'true'
    args.use_neg_prompt = args.use_neg_prompt.lower() == 'true'
    args.resample = args.resample.lower() == 'true'
    args.use_suffix = args.use_suffix.lower() == 'true'
    args.nouns_to_exclude = args.nouns_to_exclude if args.exclude_generic_nouns else None
    args.save_folder = Path(args.save_folder)
    args.save_folder.mkdir(exist_ok=True)

    if args.use_suffix:
        args.use_md = False
    return args
