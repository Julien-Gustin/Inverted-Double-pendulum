import gym  # open ai gym
import pybulletgym  # register PyBullet enviroments with open ai gym
import keyboard
import numpy

gym.logger.set_level(40)
env = gym.make('InvertedDoublePendulumPyBulletEnv-v0')
env.render() # call this before env.reset, if you want a window showing the environment
env.reset()

while True:
    rng = numpy.random.random()
    if rng < 0.5:
       state, rewards, done, _ = env.step([10])
       print(done)



    else:
        env.step([-10])

    import time
    time.sleep(1)
env.reset()