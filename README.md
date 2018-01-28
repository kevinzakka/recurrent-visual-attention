# Recurrent Visual Attention

This is a PyTorch implementation of the Recurrent Attention Model (RAM) as described in [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) by *Volodymyr Mnih, Nicolas Heess, Alex Graves and Koray Kavukcuoglu*.

<p align="center">
 <img src="./plots/bbox.png" alt="Drawing", width=60%>
</p>
<p align="center">
 <img src="./plots/glimpses.png" alt="Drawing", width=40%>
</p>

## Results

After training for 124 epochs, I am able to reach `98.117%` accuracy on the validation set and `98.41%` accuracy on the test set.

```
[*] Number of model parameters: 209,677
[*] Loading model from ./ckpt
[*] Loaded ram_6_8x8_2_model_best.pth.tar checkpoint @ epoch 124 with best valid acc of 98.117
[*] Test Acc: 9841/10000 (98.41%)
```

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
- [x] Monte-Carlo sampling for validation and testing
- [ ] restrict initial random glimpse
- [ ] make the patch extraction code more efficient
- [x] animate glimpses for a given iteration and save
- [x] reproduce results

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