# Recurrent Visual Attention

This is a PyTorch implementation of the Recurrent Attention Model (RAM) as described in [Recurrent Models of Visual Attention](https://arxiv.org/abs/1406.6247) by *Volodymyr Mnih, Nicolas Heess, Alex Graves and Koray Kavukcuoglu*.

<p align="center">
 <img src="./plots/bbox.png" alt="Drawing", width=60%>
</p>
<p align="center">
 <img src="./plots/glimpses.png" alt="Drawing", width=40%>
</p>

## Results

After training for 153 epochs, I am able to reach `98.50%` accuracy on the validation set and `98.77%` accuracy on the test set. This is equivalent to `1.23%` test error, compared to `1.29%` reported in the paper. I haven't done random search on the policy standard deviation to tune it so I expect the test error can be reduced to sub `1%` error.

```
[*] Number of model parameters: 209,677
[*] Loading model from ./ckpt
[*] Loaded ram_6_8x8_2_model_best.pth.tar checkpoint @ epoch 153 with best valid acc of 98.500
[*] Test Acc: 9877/10000 (98.77%)
```

Here's an animation showing the glimpses extracted by the network on a random batch of the 135th epoch.

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

Here's an example command for training a RAM variant that extracts 6 `8x8` glimpses from an image with tensorboard visualization.

```
python main.py \
--patch_size=8 \
--num_patches=1 \
--glimpse_scale=2 \
--num_glimpses=6 \
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