# Recurrent Visual Attention

Work In Progress

<p align="center">
 <img src="./plots/glimpses.png" alt="Drawing", width=40%>
</p>

<p align="center">
 <img src="./plots/bbox.png" alt="Drawing", width=70%>
</p>

## Todo

- make the patch extraction code vectorized (don't know if possible)

## Notes

- `fa`: action network
- `fg`: glimpse network
- `fh`: core network
- `fl`: location network
* `fa` trained with cross entropy loss. gradients backpropagated to update `fh`, `fg`
* `fl` trained with reinforce

`fl` -> `fg` -> `fh` -> `fa`

- `fl`
    - i: `h_t`
    - o: `mu`, `l_t`
- `fg`
    - i: `l_t`
    - o: `g_t`
- `fh`
    - i: `g_t`, `h_t_prev`
    - o: `h_t`
- `fa`
    - i: `h_t`
    - o: `y`

loss = loss_action + loss_baseline + loss_reinforce

- loss action = nll_loss(y, gd_truth)
- loss_baseline = mse_loss(b, R)
- loss_reinforce = sum of log of policy * (R - b) averaged over minibatch
