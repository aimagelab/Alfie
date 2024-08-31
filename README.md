# Alfie

This repository contains the reference code and dataset for the paper [Alfie:Democratising RGBA Image Generation With No $$$](https://arxiv.org/abs/2408.14826).
If you find it useful, please cite it as:
```
@inproceedings{quattrini2024alfie,
  title={{Alfie: Democratising RGBA Image Generation With No $$$}},
  author={Quattrini, Fabio and Pippi, Vittorio and Cascianelli, Silvia and Cucchiara, Rita},
  booktitle{Proceedings of the European Conference on Computer Vision Workshops},
  year={2024}
}
```

## Setup

```bash
conda create --name alfie python==3.11.7
conda activate alfie
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia -y
pip install -r requirements.txt

```

## Example usage:

```python
python generate_prompt.py --setting centering-rgba-alfie --fg_prompt 'A photo of a cat with a hat'
python generate_prompt.py --setting centering-rgba-alfie --fg_prompt 'A large, colorful tree made of money, with lots of yellow and white coins hanging from its branches'

```



Code inspired by [DAAM](https://github.com/castorini/daam)  and [DAAMI2I](https://github.com/RishiDarkDevil/daam-i2i)
