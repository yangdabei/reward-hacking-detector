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
from src.environment.gridworld import GridWorld
from src.environment.configs import TEST_COIN_MOVED, TRAINING_DEFAULT
from src.agents.q_learning import QLearningAgent
from src.config import AgentConfig
from src.logging_config import setup_logging
setup_logging()

# Train
env = GridWorld(TRAINING_DEFAULT)
agent = QLearningAgent(AgentConfig(), grid_size=7)
rewards = agent.train(env, num_episodes=5000)

# Test on coin-moved environment
test_env = GridWorld(TEST_COIN_MOVED)
obs, info = test_env.reset()
terminated, truncated = False, False
while not terminated and not truncated:
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = test_env.step(action)

print("Reached goal:", info["reached_goal"])
print("Picked coin:", info["picked_coin"])
print("Steps:", info["steps"])
print("Final epsilon:", agent.epsilon)
print("Q-table size:", len(agent.q_table))
print("Last 10 rewards:", rewards[-10:])
# %%
# Test on TRAINING env (not test env) to confirm it can reach the coin at least
train_test_env = GridWorld(TRAINING_DEFAULT)
obs, info = train_test_env.reset()
terminated, truncated = False, False
while not terminated and not truncated:
    action = agent.select_action(obs)
    obs, reward, terminated, truncated, info = train_test_env.step(action)

print("Reached goal:", info["reached_goal"])
print("Picked coin:", info["picked_coin"])
print("Steps:", info["steps"])
# %%
