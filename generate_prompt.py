from pathlib import Path
from settings import parse_setting
import json

from alfie.generate import get_pipe, base_arg_parser, parse_bool_args

from transformers import VitMatteImageProcessor, VitMatteForImageMatting

import logging
from accelerate import PartialState
from accelerate.logging import get_logger
from accelerate.utils import set_seed

from alfie.grabcut import grabcut, save_rgba

import torch
from alfie.trimap import compute_trimap
from alfie.utils import normalize_masks

torch.backends.cuda.matmul.allow_tf32 = True

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = get_logger(__name__)


def main():
    parser = base_arg_parser()
    parser.add_argument("--setting_name", type=str, default='centering-rgba-alfie')
    parser.add_argument("--fg_prompt", type=str, required=True)
    args = parser.parse_args()
    settings_dict = parse_setting(args.setting_name)
    vars(args).update(settings_dict)
    args = parse_bool_args(args)

    distributed_state = PartialState()
    args.device = distributed_state.device

    args.save_folder = args.save_folder / 'prompts'
    args.save_folder.mkdir(parents=True, exist_ok=True)

    pipe = get_pipe(
        image_size=args.image_size,
        scheduler=args.scheduler,
        device=args.device)

    suffix = ' on a white background'
    prompt_complete = ["A white background", args.fg_prompt]
    prompt_full = ' '.join(prompt_complete[1].split())
    negative_prompt = ["Blurry, shadow, low-resolution, low-quality"] if args.use_neg_prompt else None
    prompt = prompt_complete if args.centering else prompt_complete[1]
    if args.use_suffix:
        prompt += suffix

    if args.cutout_model == 'vit-matte':
        vit_matte_processor = VitMatteImageProcessor.from_pretrained(args.vit_matte_key)
        vit_matte_model = VitMatteForImageMatting.from_pretrained(args.vit_matte_key)
        vit_matte_model = vit_matte_model.eval()

    base_name = '_'.join([
        prompt_full,
        'centering' if args.centering else '',
        'sz_256' if args.image_size == 256 else 'sz_512'
    ])

    config = vars(args).copy()
    del config['device']
    del config['save_folder']
    del config['seed']
    del config['num_images']
    with open(args.save_folder / f'{base_name}.json', 'w') as f:
        json.dump(config, f, indent=4)

    for seed in range(args.seed, args.seed + args.num_images):
        set_seed(seed)
        generator = torch.Generator(device="cuda").manual_seed(seed)
        name = f'{base_name}_seed_{seed}'

        images, heatmaps = pipe(
            prompt=prompt, negative_prompt=negative_prompt, nouns_to_exclude=args.nouns_to_exclude,
            keep_cross_attention_maps=True, return_dict=False, num_inference_steps=args.steps,
            centering=args.centering, generator=generator)
        
        image = images[0]
        rgb_image_filename = Path(args.save_folder / f"{name}.png")
        if not rgb_image_filename.exists():
            rgb_image_filename.parent.mkdir(parents=True, exist_ok=True)
            image.save(rgb_image_filename)

        torch.cuda.empty_cache()

        if args.cutout_model == 'grabcut':
            alpha_mask = grabcut(
                image=image, attention_maps=list(heatmaps['cross_heatmaps_fg_nouns'].values()), image_size=args.image_size,
                sure_fg_threshold=args.sure_fg_threshold, maybe_fg_threshold=args.maybe_fg_threshold,
                maybe_bg_threshold=args.maybe_bg_threshold)

            alfie_rgba_image_filename =  Path(args.save_folder / f"{name}-rgba-alfie.png")
            alfie_rgba_image_filename.parent.mkdir(parents=True, exist_ok=True)
            alpha_mask_alfie = torch.tensor(alpha_mask)
            alpha_mask_alfie = torch.where(alpha_mask_alfie == 1, normalize_masks(heatmaps['ff_heatmap'] + 1 * heatmaps['cross_heatmap_fg']), 0.)
            save_rgba(image, alpha_mask_alfie, alfie_rgba_image_filename)

        elif args.cutout_model == 'vit-matte':
            trimap = compute_trimap(attention_maps=[list(heatmaps['cross_heatmaps_fg_nouns'].values())],
                                    image_size=args.image_size,
                                    sure_fg_threshold=args.sure_fg_threshold,
                                    maybe_bg_threshold=args.maybe_bg_threshold)

            vit_matte_inputs = vit_matte_processor(images=image, trimaps=trimap, return_tensors="pt").to(args.device)
            vit_matte_model = vit_matte_model.to(args.device)
            with torch.no_grad():
                alpha_mask = vit_matte_model(**vit_matte_inputs).alphas[0, 0]
            alpha_mask = 1 - alpha_mask.cpu().numpy()
            save_rgba(image, alpha_mask, args.save_folder / f"{name}-rgba-vit_matte.png")
        else:
            raise ValueError(f'Invalid cutout model: {args.cutout_model}')

            

        del heatmaps
        torch.cuda.empty_cache()
    
    logger.info("***** Done *****")


if __name__ == '__main__':
    main()
