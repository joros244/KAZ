from pettingzoo.butterfly import knights_archers_zombies_v10

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    env = knights_archers_zombies_v10.env(render_mode="human",
                                          spawn_rate=20,
                                          num_archers=1,
                                          num_knights=1,
                                          max_zombies=10,
                                          max_arrows=10,
                                          killable_knights=True,
                                          killable_archers=True,
                                          pad_observation=True,
                                          line_death=False,
                                          max_cycles=900,
                                          vector_state=True,
                                          use_typemasks=False,
                                          sequence_space=False)
    env.reset(seed=42)
    count = 0
    for agent in env.agent_iter():
        count += 1
        observation, reward, termination, truncation, info = env.last()

        # if env.agent_name_mapping[agent] == 0:
        #     print(agent)
        if termination or truncation:
            action = None
        else:
            # this is where you would insert your policy
            action = env.action_space(agent).sample()

        env.step(action)
        print(count)
    env.close()
