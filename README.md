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

- `retina`
    - takes as input an image and a location vector
    - outputs foveated glimpses of this image
- `glimpse network`
    - takes as input an image and a location vector
    - uses the retina
    - feeds output of retina through a fc layer
    - feeds location vector through a fc layer
    - concatenates both and applies nonlinearity
    - outputs a glimpse representation vector
    - trained with backprop
- `core network`
    - takes as input the glimpse representation vector
    - also takes as input the hidden representation at the previous time step
    - basically a 1-layer RNN
    - outputs the hidden representation for the current timestep
    - trained with backprop
- `location network`
    - takes as input the hidden state representation for the current timestep
    - 2 component gaussian with a fixed variance
    - feeds the hidden state through a fc layer to produce 1 output
    - this output is the mean of the location policy at time t
    - the variance is found using random search
    - trained with REINFORCE
- `action network`
    - takes as input the hidden state representation for the current timestep
    - used only at last timestep N
    - linear softmax over the hidden representation
    - divided by a normalizing constant Z
    - train through backprop