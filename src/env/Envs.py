import numpy as np
import torch
from pettingzoo.butterfly import knights_archers_zombies_v10
from supersuit import vectorize_aec_env_v0

from network.PolicyValueNetwork import PolicyValueNetwork


class Envs:
    # Wrapper for the vectorized environments. Beware AEC logic.
    def __init__(self, num_envs, agent_to_train="archer_0", path_to_model_to_play=None):
        self.episodes_lengths = None
        self._env = knights_archers_zombies_v10.env(
            spawn_rate=20,
            num_archers=1,
            num_knights=1,
            max_zombies=10,
            max_arrows=10,
            killable_knights=False,
            killable_archers=False,
            pad_observation=False,
            line_death=False,
            max_cycles=900,
            vector_state=True,
            use_typemasks=False,
            sequence_space=False)

        self._num_envs = num_envs
        self.envs = vectorize_aec_env_v0(self._env, self._num_envs)

        # Playable agent is the agent we want to train.

        self._playable_agent = agent_to_train

        # Doesn't matter if we use the observation space of the playable agent. They are the same for all agents.
        self.single_observation_space = self.envs.observation_space(self._playable_agent)
        self.single_action_space = self.envs.action_space(self._playable_agent)

        self._agent_to_play = PolicyValueNetwork(self.single_observation_space, self.single_action_space).to(
            torch.device("cuda"))

        if path_to_model_to_play is not None:  # This is the model of the agent that will play with us.
            self._agent_to_play.load_state_dict(torch.load(path_to_model_to_play))

    def get_envs(self):
        return self.envs

    def reset(self, seed=0):
        self.envs.reset(seed=seed)
        agent = self.envs.agent_selection
        self.episodes_lengths = np.zeros(self._num_envs)

        return self.envs.observe(agent)

    def step(self, actions):
        assert len(actions) == self._num_envs, f"Expected {self._num_envs} actions, got {len(actions)}"

        playable_agent = self._playable_agent  # Agent to train

        self.envs.step(actions)

        self.episodes_lengths += 1

        rewards = self.envs.rewards[playable_agent]  # Immediate rewards

        # Other agents still have to play:

        for agent in self.envs.possible_agents:
            if agent != playable_agent:
                observations = self.envs.observe(agent)
                observations = torch.tensor(observations).float().to(torch.device("cuda"))
                actions = self._agent_to_play.get_actions_and_values(observations)[0]
                actions = actions.cpu().numpy()
                self.envs.step(actions)
                rewards += self.envs.rewards[playable_agent]  # Collect possible updates in our rewards

        obs = self.envs.observe(playable_agent)  # Next observations
        terminations = self.envs.terminations[playable_agent]  # Terminations
        truncations = self.envs.truncations[playable_agent]  # Truncations
        infos = self.envs.infos[playable_agent]  # Additional information

        for i, (termination, truncation) in enumerate(zip(terminations, truncations)):
            if termination or truncation:
                infos[i]["final_info"] = {
                    "episode": {"r": self.envs._cumulative_rewards[playable_agent][i], "l": self.episodes_lengths[i]}}
                self.episodes_lengths[i] = 0

        return obs, rewards, terminations, truncations, infos

    def close(self):
        for env in self.envs.envs:
            env.close()
