import torch
from pettingzoo.butterfly import knights_archers_zombies_v10

from network.PolicyValueNetwork import PolicyValueNetwork

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = knights_archers_zombies_v10.env(
        render_mode='human',
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

    rewards = {agent: 0 for agent in env.possible_agents}
    # load policynetwork from state_dict
    agent_A = env.possible_agents[0]
    agent_K = env.possible_agents[1]
    model_A = PolicyValueNetwork(env.observation_space(agent_A), env.action_space(agent_A))
    model_A.load_state_dict(torch.load('src/algorithm/models/KAZ-V3.2-5_19.pt'))
    model_K = PolicyValueNetwork(env.observation_space(agent_K), env.action_space(agent_K))
    model_K.load_state_dict(torch.load('src/algorithm/models/KAZ-V3.1-5_12.pt'))

    env.reset(seed=31415926)

    for agent in env.agent_iter():
        observation, reward, termination, truncation, info = env.last()

        for a in env.agents:
            rewards[a] += env.rewards[a]

        if termination or truncation:
            break
        else:
            if env.agent_name_mapping[agent] == 0:
                observation = torch.tensor(observation).float()
                observation = observation.flatten()
                action = model_A._get_action_and_value(observation)[0]
                action = action.item()
            else:
                observation = torch.tensor(observation).float()
                observation = observation.flatten()
                action = model_K._get_action_and_value(observation)[0]
                action = action.item()
            # action = env.action_space(agent).sample()

        env.step(action)

    env.close()

    avg_reward = sum(rewards.values()) / len(rewards.values())
    avg_reward_per_agent = {
        agent: rewards[agent]  for agent in env.possible_agents
    }
    print(f"Avg reward: {avg_reward}")
    print("Avg reward per agent, per game: ", avg_reward_per_agent)
    print("Full rewards: ", rewards)
