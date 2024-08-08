# Format: rgb_generation_setting-rgba-rgba_cutout_name


def parse_rgb_generation_setting(rgb_generation_setting: str, setting_dict: dict):
    if 'centering' not in rgb_generation_setting:
        setting_dict['centering'] = 'False'

    if 'no_exc' in rgb_generation_setting:
        setting_dict['exclude_generic_nouns'] = 'False'

    if 'suffix' in rgb_generation_setting:
        setting_dict['use_suffix'] = 'True'

    return setting_dict



def parse_setting(setting: str):
    setting_dict = {
        'centering': 'True',
        'use_neg_prompt': 'True',
        'exclude_generic_nouns': 'True',
        'cutout_model': 'grabcut',
        'use_suffix': 'False'
    }

    if 'rgba' in setting:
        rgb_generation_setting, rgba_cutout_setting = setting.rsplit('-rgba-')
    else:
        rgb_generation_setting = setting
        rgba_cutout_setting = None
    
    setting_dict = parse_rgb_generation_setting(rgb_generation_setting, setting_dict)
    if rgba_cutout_setting:
        if 'alfie' in rgba_cutout_setting:
            setting_dict['cutout_model'] = 'grabcut'
        elif 'vit_matte' in rgba_cutout_setting:
            setting_dict['cutout_model'] = 'vit-matte'
        else:
            raise ValueError(f"Invalid cutout model: {rgba_cutout_setting}. Options: ['grabcut', 'vit_matte']")

    return setting_dict