from env_base import DroneEnv
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
import os

env = DroneEnv("drone_model/Rotor.urdf")
env = Monitor(env)

models_dir = "models/PPO4"
logdir = "logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)
if not os.path.exists(logdir):
    os.makedirs(logdir)

env = DroneEnv("drone_model/Rotor.urdf")
env = Monitor(env)
env.reset()
# Wrap in DummyVecEnv for Stable-Baselines
vec_env = DummyVecEnv([lambda: env])
model = PPO(
    'MlpPolicy',
    env,
    verbose=1,
    tensorboard_log='logs',
    learning_rate=3e-4,        
    n_steps=2048,              
    batch_size=64,            
    n_epochs=10,               
    gamma=0.995,               
    gae_lambda=0.95,          
    clip_range=0.25,            
    ent_coef=0.02,            
    policy_kwargs={'net_arch': [128, 128]},  
    device='cpu',
)
# Training parameters
TIMESTEPS = 100_000
total_steps = 300

# Training loop
for step in range(total_steps):
    model.learn(
        total_timesteps=TIMESTEPS,
        progress_bar=True,
        reset_num_timesteps=False,
        tb_log_name="train3",
        log_interval=10
    )
    model.save(f"{models_dir}/{TIMESTEPS*(step+1)}")

env.close()