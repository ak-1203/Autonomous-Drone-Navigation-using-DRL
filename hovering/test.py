from stable_baselines3 import PPO
from env_base import DroneEnv

import os

models_dir = "models/PPO"
model_path = os.path.join(models_dir, "trained_hovering.zip")  # Change this to the desired episode

# Load environment
env = DroneEnv("drone_model/Rotor.urdf")

# Load model
model = PPO.load(model_path, env=env)

# Wrap in DummyVecEnv if needed
from stable_baselines3.common.vec_env import DummyVecEnv
env = DummyVecEnv([lambda: env])

num_trials = 50
total_rewards = []

for trial in range(num_trials):
    obs = env.reset()
    done = False
    trial_reward = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _ = env.step(action)
        trial_reward += reward
    
    total_rewards.append(trial_reward)
    print(f"Trial {trial + 1}: Total Reward = {trial_reward}")

print(f"Average Reward Over {num_trials} Trials: {sum(total_rewards) / num_trials}")
