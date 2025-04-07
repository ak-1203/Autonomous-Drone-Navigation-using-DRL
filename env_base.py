import gymnasium as gym
from gymnasium import Env, spaces
import numpy as np
import pybullet as p
import time
import pybullet_data



class DroneEnv(gym.Env):
    def __init__(self, drone_path, target_coordinate=(0,0,5), framerate=240):
        super().__init__()
        
        if p.getConnectionInfo()['isConnected'] == 0:
            self.client = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(1. / framerate)
        
        self.framerate = framerate
        self.drone_path = drone_path
        self.target_position = target_coordinate
        self.iteration_max = 5000
        self.iteration = 0
        self.kf=1e-6
        self.max_omega=1000
        self.count=0
        self.success_count = 5
        
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-5, -5, 0, -1, -1, -1, -np.pi, -np.pi, -np.pi, -1, -1, -1]),
            high=np.array([5, 5, 10, 1, 1, 1, np.pi, np.pi, np.pi, 1, 1, 1]),
            dtype=np.float32
        )
        
        self.reset()
        
        
    def _get_observation(self):
        pos, orn = p.getBasePositionAndOrientation(self.drone)
        lin_vel, ang_vel = p.getBaseVelocity(self.drone)
        roll, pitch, yaw = p.getEulerFromQuaternion(orn)
        return np.array([*pos, *lin_vel, roll, pitch, yaw, *ang_vel])
    
    def step(self, action):
        omega=np.full((1,4),1000*action.item())
        
        for i in range(4):
            omega[0, i] = np.abs(omega[0, i]) if i in {0, 3} else -np.abs(omega[0, i])
        motor_forces = self.kf * (omega ** 2)

        for i in range(4):

            p.applyExternalForce(self.drone, i, forceObj=[0, 0, motor_forces[0, i]], posObj=[0, 0, 0], flags=p.LINK_FRAME)

            p.setJointMotorControl2(
            bodyIndex=self.drone,
            jointIndex=i,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=omega[0,i],
            force=10  
            )

        p.stepSimulation()
        self.iteration += 1

        x,y,z,vx,vy,vz,roll,pitch,yaw,w1,w2,w3 = self._get_observation()
        done = z < 0.1 

         # --- Reward Calculation ---
        pos_error = abs(self.target_position[2] - z)
        lateral_drift = np.linalg.norm(np.array((x,y)) - self.target_position[:2])
        velocity_reward = np.dot((vx,vy,vz), np.array([0, 0, 1]))  
        orientation_penalty = abs(roll) + abs(pitch)  
        safe_altitude_bonus = 0.005 if z > 0.15 else -2
        speed_penalty = np.linalg.norm((vx,vy,vz)) ** 2 

        reward_components = np.array([
            np.exp(-pos_error),
            np.exp(-lateral_drift * 2) * 0.3,
            np.exp(-orientation_penalty * 3) * 0.1,
            np.exp(velocity_reward)- 0.1*speed_penalty,
            safe_altitude_bonus
        ])

        reward_components /= (np.max(reward_components) + 1e-8)
        reward = np.sum(reward_components)
    
        truncated = self.iteration > self.iteration_max
        info={}
        

        return self._get_observation(), reward, done,truncated,info
    
    def reset(self,seed=None, options=None):
        p.resetSimulation()
        p.setTimeStep(1./self.framerate) ##
        p.setGravity(0, 0, -9.8)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.loadURDF("plane.urdf")
        self.drone = p.loadURDF(self.drone_path, basePosition=[0, 0, 0.2],globalScaling=5)
        self.iteration = 0
        self.count = 0 ##
        return self._get_observation(),{}
    
    def render(self):
        if p.getConnectionInfo(self.client)["connectionMethod"] != p.GUI:
            p.disconnect(self.client)
            self.client = p.connect(p.GUI)
            p.setAdditionalSearchPath(pybullet_data.getDataPath())
    
    def close(self):
        p.disconnect(self.client)

