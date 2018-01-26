# Recurrent Visual Attention

This is a PyTorch implementation of the Recurrent Attention Model (RAM) as described in [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) by *Volodymyr Mnih, Nicolas Heess, Alex Graves and Koray Kavukcuoglu*.

<p align="center">
 <img src="./plots/bbox.png" alt="Drawing", width=60%>
</p>
<p align="center">
 <img src="./plots/glimpses.png" alt="Drawing", width=40%>
</p>

## Results

I haven't been able to reproduce the paper results. Specifically, on regular MNIST, I am only able to achieve `92%` accuracy vs the `98%+` reported in the paper. I'm still tweaking the hyperparameters and implementing a restriction on the location vector which should improve results significantly.

Here's an animation showing the glimpses extracted by the network on a random batch of the 81'st epoch.

<p align="center">
 <img src="./plots/example.gif" alt="Drawing">
</p>

## Model Description

<p align="center">
 <img src="./plots/model.png" alt="Drawing", width=70%>
</p>

## Requirements

- python 3.5+
- pytorch 0.2+
- tensorboard_logger
- tqdm

## Todo

- [x] GPU support
- [ ] restrict initial random glimpse
- [ ] make the patch extraction code more efficient
- [x] animate glimpses for a given iteration and save
- [ ] reproduce results

## Usage

Here's an example command for training a RAM variant that extracts 7 `8x8` glimpses from an image with tensorboard visualization.

```
python main.py \
--patch_size=8 \
--num_patches=1 \
--glimpse_scale=2 \
--num_glimpses=7 \
--use_tensorboard=True
```

Alternatively, just edit the values in `config.py` and run

```
python main.py
```

to train, with `--resume=True` to reload from the latest checkpoint and

```
python main.py --is_train=False
```
to test the checkpoint that has achieved the best validation accuracy.