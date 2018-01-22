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

- `location_network`: trained with REINFORCE
- `action_network`: cross entropy loss
- `core_network`: backpropagated gradients from `action network`
- `glimpse_network`: backpropagated gradients from `action network`