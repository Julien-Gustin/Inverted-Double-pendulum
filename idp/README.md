# Inverted Double pendulum

## Algorithm

- Fitted Q-iteration: `fqi`
- Deep Q-learning: `dql`
- Deep Deterministic Policy Gradient: `ddpg`

## Utilisation

```
python main.py [-h] [--ddpg] [--fqi] [--dql] [--batchnorm] [--render RENDER] [--gamma GAMMA] [--samples SAMPLES] [--actions ACTIONS] [--seed SEED]
```
`RENDER` should be a file toward a saved model for either **dql** or **ddpg**

`GAMMA` is the discound factor $\gamma \in [0, 1]$

`SAMPLES` are the number of sample used when training **fqi**

`ACTIONS` number of discrete action when using either **ddpg** or **fqi**

`SEED` the seed use

## Example

### Load a saved model using ddpg

```
python main.py --ddpg --gamma 0.99 --render saved_models/model
```

### Train a dql using 11 discrete actions

```
python main.py --dql --gamma 0.99 --actions 11
```
