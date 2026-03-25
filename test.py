#%%
from src.environment.gridworld import GridWorld
from src.environment.configs import TRAINING_DEFAULT

env = GridWorld(TRAINING_DEFAULT)

obs, info = env.reset()
print("obs:", obs)
print("info:", info)

obs, reward, terminated, truncated, info = env.step(1)  # DOWN
print("obs:", obs)       # should be (1, 0)
print("reward:", reward) # should be -0.1
print("terminated:", terminated)  # should be False#
# %%
