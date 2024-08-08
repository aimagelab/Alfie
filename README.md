# Alfie

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

```



Code inspired by [DAAM](https://github.com/castorini/daam)  and [DAAMI2I](https://github.com/RishiDarkDevil/daam-i2i)