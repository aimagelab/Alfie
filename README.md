# Alfie

Official PyTorch implementation for [Alfie: Democratising RGBA Image Generation With No $$$](https://arxiv.org/abs/2408.14826) (AI for Visual Arts Workshop, ECCV24)

[![ECCV Paper](https://img.shields.io/badge/ECCV-Paper-blue.svg)]([https://openaccess.thecvf.com/content/CVPR2025/html/Pippi_Zero-Shot_Styled_Text_Image_Generation_but_Make_It_Autoregressive_CVPR_2025_paper.html](https://link.springer.com/chapter/10.1007/978-3-031-92808-6_3))
[![arXiv](https://img.shields.io/badge/arXiv-2408.14826-b31b1b.svg)](https://arxiv.org/abs/2408.14826)
[![ECCV Poster](https://img.shields.io/badge/🖼️-Poster-blue.svg)](https://fabioquattrini.com/posters/Alfie.pdf)

[Setup](#setup) • [Usage](#usage) • [Citation](#citation)

---


## Setup

```bash
conda create --name alfie python==3.11.7
conda activate alfie
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt

```

## Usage:

```python
python generate_prompt.py --setting centering-rgba-alfie --fg_prompt 'A photo of a cat with a hat'
python generate_prompt.py --setting centering-rgba-alfie --fg_prompt 'A large, colorful tree made of money, with lots of yellow and white coins hanging from its branches'

```

## Citation

If you find it useful, please cite it as:
```
@inproceedings{quattrini2024alfie,
  title={{Alfie: Democratising RGBA Image Generation With No $$$}},
  author={Quattrini, Fabio and Pippi, Vittorio and Cascianelli, Silvia and Cucchiara, Rita},
  booktitle{Proceedings of the European Conference on Computer Vision Workshops},
  year={2024},
  organization={Springer}
}
```


Code inspired by [DAAM](https://github.com/castorini/daam)  and [DAAMI2I](https://github.com/RishiDarkDevil/daam-i2i)
