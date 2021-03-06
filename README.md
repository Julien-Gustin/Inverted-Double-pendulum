# Inverted Double pendulum
The Double Inverted Pendulum consists of two joint pendulums connected to a cart that
is moving on a track, the agent needs to keeps in equilibrium the Double Inverted Pendulum by interacting with the environment by applying an horizontal force on a cart. At each transition,
additionally to the reward given by the environment, the agent receives a positive, constant signal. A terminal state of the environment is reached when the distance between
the upright and the current state is above a given thresold.

<p align="center">
    <img src="https://github.com/Julien-Gustin/RL-INFO8003/blob/master/gif/optimal_policy.gif" width="600" height="400" />
    <br>
    <em>Agent trained for 500 epochs using DDPG algorithm and gamma set to 0.99</em>
</p>

## Implementation and method

More informations concerning the implementation and method in the [report](report.pdf).

## Algorithms available

- Fitted Q-iteration: `fqi`
- Deep Q-learning: `dql`
- Deep Deterministic Policy Gradient: `ddpg`

## Use
Make sure to have installed [pybullet-gym](https://github.com/benelot/pybullet-gym) before using the program.


```
python main.py [-h] [--ddpg] [--fqi] [--dql] [--batchnorm] [--render RENDER] [--gamma GAMMA] [--samples SAMPLES] [--actions ACTIONS] [--seed SEED]
```
`RENDER` should be a file toward a saved model for either **dql** or **ddpg**

&rarr; *This will render the double pendulum with the given pretrained model.*

`GAMMA` is the discount factor $\gamma \in [0, 1]$

&rarr; *0.99 give the best results*

`SAMPLES` are the number of samples used when training **fqi**

&rarr; *Higher is the better but 200k give reasonable good results (computation expensive)*

`ACTIONS` number of discrete actions when using either **dql** or **fqi**

&rarr; *Should be an odd number*

`SEED` the seed to use

## Examples

### Load a saved model using ddpg

```
python main.py --ddpg --gamma 0.99 --render saved_models/DDPG
```

### Train a dql using 11 discrete actions

```
python main.py --dql --gamma 0.99 --actions 11
```

