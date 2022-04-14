# Inverted Double pendulum
The Double Inverted Pendulum consists of two joint pendulums connected to a cart that
is moving on a track, the agent needs to keeps in equilibrium the Double Inverted Pendulum by interacting with the environment by applying an horizontal force on a cart. At each transition,
additionally to the reward given by the environment, the agent receives a positive, constant signal. A terminal state of the environment is reached when the distance between
the upright and the current state is above a given thresold.
<figure>
<p align="center">
<img src="https://github.com/Julien-Gustin/RL-INFO8003/blob/master/idp/gif/optimal_policy.gif" width="600" height="400" />
  <figcaption>Agent trained for 500 epochs using DDPG algorithm and gamma set to 0.99</figcaption>
</p>

</figure>

## Algorithms available

- Fitted Q-iteration: `fqi`
- Deep Q-learning: `dql`
- Deep Deterministic Policy Gradient: `ddpg`

## Utilisation
Make sure to have installed [pybullet-gym](https://github.com/benelot/pybullet-gym) before using the program.


```
python main.py [-h] [--ddpg] [--fqi] [--dql] [--batchnorm] [--render RENDER] [--gamma GAMMA] [--samples SAMPLES] [--actions ACTIONS] [--seed SEED]
```
`RENDER` should be a file toward a saved model for either **dql** or **ddpg**

 -> *This will render the double pendulum with the given pretrained model.*

`GAMMA` is the discound factor $\gamma \in [0, 1]$

 -> *0.99 give the best results*

`SAMPLES` are the number of sample used when training **fqi**

 -> *Higher is the better but 200k give reasonable good result (computation expensive)*

`ACTIONS` number of discrete action when using either **ddpg** or **fqi**

 -> *Should be an odd number*

`SEED` the seed to use

## Example

### Load a saved model using ddpg

```
python main.py --ddpg --gamma 0.99 --render saved_models/model
```

### Train a dql using 11 discrete actions

```
python main.py --dql --gamma 0.99 --actions 11
```

