from env import TetrisEnvironment

from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env

# env = TetrisEnvironment(render_mode="field_array")

# check_env(env)

# vec_env = make_vec_env(TetrisEnvironment, n_envs=4, env_kwargs={"render_mode" : "field_array"})

# model = PPO("MlpPolicy", vec_env, verbose=1)
# model.learn(total_timesteps=1000)
# model.save("sanity_check")

# del model # remove to demonstrate saving and loading_

model = PPO.load("sanity_check")

env = TetrisEnvironment(render_mode="human")
obs, info = env.reset()
while True:
    action = model.predict(obs["field"])[0]
    obs, rewards, terminated, truncated, info = env.step(action)
    
    if terminated:
        break

env.close()