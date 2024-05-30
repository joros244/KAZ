from pettingzoo.butterfly import knights_archers_zombies_v10
from gymnasium.vector import SyncVectorEnv
from gymnasium import spaces
from supersuit.aec_vector import vectorize_aec_env_v0

from src.env.Envs import Envs

if __name__ == '__main__':
    envs = Envs(num_envs=2)
    obs = envs.reset(seed=42)
    print(obs)
    envs.step([1, 1])
    envs.close()
