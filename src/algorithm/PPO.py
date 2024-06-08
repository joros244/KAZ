import os
import random
import time
from types import SimpleNamespace

import numpy as np
import torch
import wandb
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from src.env.Envs import Envs
from src.network.PolicyValueNetwork import PolicyValueNetwork


class PPO:
    # CleanRL PPO: https://github.com/vwxyzjn/cleanrl/blob/master/cleanrl/ppo.py

    def __init__(self, args: SimpleNamespace):
        self.args = args
        self.args.batch_size = int(self.args.num_envs * self.args.num_steps)
        self.args.minibatch_size = int(self.args.batch_size // self.args.num_minibatches)
        self.args.num_iterations = self.args.total_timesteps // self.args.batch_size
        run_name = f"KAZ__{self.args.exp_name}__{self.args.seed}__{int(time.time())}"

        wandb.init(
            project=self.args.wandb_project_name,
            entity=self.args.wandb_entity,
            sync_tensorboard=True,
            config=vars(self.args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
        self.writer = SummaryWriter(f"runs/{run_name}")
        self.writer.add_text(
            "hyperparameters",
            "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(self.args).items()])),
        )

        # TRY NOT TO MODIFY: seeding
        random.seed(self.args.seed)
        np.random.seed(self.args.seed)
        torch.manual_seed(self.args.seed)
        torch.backends.cudnn.deterministic = self.args.torch_deterministic

        self.device = torch.device("cuda" if torch.cuda.is_available() and self.args.cuda else "cpu")

        self._setup()

    def _setup(self, to_train='', path_to_train=None, path_to_play=None):
        if len(to_train) == 0:
            self.envs = Envs(num_envs=self.args.num_envs)
            self.agent = PolicyValueNetwork(self.envs.single_observation_space, self.envs.single_action_space).to(
                self.device)
        else:
            self.envs = Envs(num_envs=self.args.num_envs, agent_to_train=to_train, path_to_model_to_play=path_to_play)
            self.agent = PolicyValueNetwork(self.envs.single_observation_space, self.envs.single_action_space).to(
                self.device)
            if path_to_train is not None:
                self.agent.load_state_dict(torch.load(path_to_train))

        self.optimizer = optim.Adam(self.agent.parameters(), lr=self.args.learning_rate, eps=1e-5)

        # Storage setup
        self.obs = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_observation_space.shape).to(
            self.device)
        self.actions = torch.zeros((self.args.num_steps, self.args.num_envs) + self.envs.single_action_space.shape).to(
            self.device)
        self.logprobs = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.rewards = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.dones = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)
        self.values = torch.zeros((self.args.num_steps, self.args.num_envs)).to(self.device)

    def train(self):
        path_to_best = None
        # Start game
        global_step = 0
        start_time = time.time()
        next_obs = self.envs.reset(seed=self.args.seed)
        next_obs = torch.Tensor(next_obs).to(self.device)
        next_done = torch.zeros(self.args.num_envs).to(self.device)

        for iteration in range(1, self.args.num_iterations + 1):
            # Annealing the rate if instructed to do so.
            if self.args.anneal_lr:
                frac = 1.0 - (iteration - 1.0) / self.args.num_iterations
                lrnow = frac * self.args.learning_rate
                self.optimizer.param_groups[0]["lr"] = lrnow

            # ROLLOUT PHASE
            for step in range(0, self.args.num_steps):
                global_step += self.args.num_envs
                self.obs[step] = next_obs
                self.dones[step] = next_done

                # ALGO LOGIC: action logic
                with torch.no_grad():
                    action, logprob, _, value = self.agent.get_actions_and_values(next_obs)
                    self.values[step] = value.flatten()
                self.actions[step] = action
                self.logprobs[step] = logprob

                # TRY NOT TO MODIFY: execute the game and log data.
                next_obs, reward, terminations, truncations, infos = self.envs.step(action.cpu().numpy())
                next_done = np.logical_or(terminations, truncations)
                self.rewards[step] = torch.tensor(reward).to(self.device).view(-1)
                next_obs, next_done = torch.Tensor(next_obs).to(self.device), torch.Tensor(next_done).to(self.device)

                for inf in infos:
                    if "final_info" in inf:
                        if inf["final_info"] and "episode" in inf["final_info"]:
                            # print(f"global_step={global_step}, episodic_return={inf['final_info']['episode']['r']}")
                            self.writer.add_scalar("charts/episodic_return", inf["final_info"]["episode"]["r"],
                                                   global_step)
                            self.writer.add_scalar("charts/episodic_length", inf["final_info"]["episode"]["l"],
                                                   global_step)

            # DONE ROLLOUT

            # LEARNING PHASE

            # Generalized Advantage Estimation:
            # bootstrap value if not done
            with torch.no_grad():
                next_value = self.agent.get_values(next_obs).reshape(1, -1)
                advantages = torch.zeros_like(self.rewards).to(self.device)
                lastgaelam = 0
                for t in reversed(range(self.args.num_steps)):
                    if t == self.args.num_steps - 1:  # if t = T-1 -> values are estimated by our network
                        nextnonterminal = 1.0 - next_done
                        nextvalues = next_value
                    else:  # else -> we already have values from what we have seen during rollout
                        nextnonterminal = 1.0 - self.dones[t + 1]
                        nextvalues = self.values[t + 1]
                    delta = self.rewards[t] + self.args.gamma * nextvalues * nextnonterminal - self.values[
                        t]  # TD error (if terminal state -> bootstrap 0). r + gamma*V(s+1) - V(s)

                    advantages[
                        t] = lastgaelam = delta + self.args.gamma * self.args.gae_lambda * nextnonterminal * lastgaelam
                    # Advantage estimation: td error + lambda * gamma * advantage(t+1)
                    # (if terminal state -> bootstrap 0)

                returns = advantages + self.values  # TD(lambda) return estimation

            # flatten the batch
            b_obs = self.obs.reshape((-1,) + self.envs.single_observation_space.shape)
            b_logprobs = self.logprobs.reshape(-1)
            b_actions = self.actions.reshape((-1,) + self.envs.single_action_space.shape)
            b_advantages = advantages.reshape(-1)
            b_returns = returns.reshape(-1)
            b_values = self.values.reshape(-1)

            # MINI-BATCH UPDATE:
            # Optimizing the policy and value network
            b_inds = np.arange(self.args.batch_size)
            clipfracs = []
            continue_training = True
            for epoch in range(self.args.update_epochs):
                np.random.shuffle(b_inds)  # Shuffle the data
                for start in range(0, self.args.batch_size, self.args.minibatch_size):  # Mini-batch iteration
                    end = start + self.args.minibatch_size
                    mb_inds = b_inds[start:end]

                    _, newlogprob, entropy, newvalue = self.agent.get_actions_and_values(b_obs[mb_inds],
                                                                                         b_actions.long()[mb_inds])
                    logratio = newlogprob - b_logprobs[mb_inds]
                    ratio = logratio.exp()  # pi current policy / pi old policy

                    with torch.no_grad():  # Debug and early stopping
                        # calculate approx_kl http://joschu.net/blog/kl-approx.html
                        old_approx_kl = (-logratio).mean()
                        approx_kl = ((ratio - 1) - logratio).mean()
                        clipfracs += [((ratio - 1.0).abs() > self.args.clip_coef).float().mean().item()]

                    if self.args.target_kl is not None and approx_kl > 1.5 * self.args.target_kl:  # If KL divergence is
                        # too high, stop training. Target policy is too far away from the current policy!
                        continue_training = False
                        break

                    mb_advantages = b_advantages[mb_inds]  # Minibatch advantages
                    if self.args.norm_adv:
                        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)  # See
                        # implementation trick 7

                    # Policy loss
                    pg_loss1 = -mb_advantages * ratio
                    pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - self.args.clip_coef,
                                                            1 + self.args.clip_coef)  # Clip ratio
                    pg_loss = torch.max(pg_loss1, pg_loss2).mean()  # Pessimistic policy loss -> minimizing the maximum
                    # of the two losses

                    # Value loss
                    newvalue = newvalue.view(-1)
                    if self.args.clip_vloss:  # Clip loss for the value function. Impl trick 9. Disabled to decrease
                        # complexity (do not want to introduce more hyperparameters)
                        v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                        v_clipped = b_values[mb_inds] + torch.clamp(
                            newvalue - b_values[mb_inds],
                            -self.args.clip_coef,
                            self.args.clip_coef,
                        )
                        v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                        v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                        v_loss = 0.5 * v_loss_max.mean()
                    else:
                        v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                    entropy_loss = entropy.mean()  # Entropy to favor exploration
                    loss = pg_loss - self.args.ent_coef * entropy_loss + v_loss * self.args.vf_coef  # More coefficients
                    # to play with :(

                    self.optimizer.zero_grad()
                    loss.backward()  # Backpropagation
                    nn.utils.clip_grad_norm_(self.agent.parameters(), self.args.max_grad_norm)  # Impl trick 11
                    self.optimizer.step()

                if not continue_training:
                    break

            # Logging stuff
            y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
            var_y = np.var(y_true)
            explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

            self._log(global_step, start_time, v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs,
                      explained_var)

            with torch.no_grad():
                prev_max_reward = 0 if iteration == 1 else prev_max_reward
                avg_reward = torch.sum(self.rewards).item() / self.args.num_envs
                self.writer.add_scalar("charts/average_reward", avg_reward,
                                       global_step)  # Logging average reward (obtained from playing n_steps in each env)

                if not os.path.exists('models'):
                    os.makedirs('models')

                if avg_reward > prev_max_reward:
                    torch.save(self.agent.state_dict(), f"models/{self.args.exp_name}_{iteration}.pt")
                    path_to_best = f"models/{self.args.exp_name}_{iteration}.pt"
                    wandb.save(f"models/{self.args.exp_name}_{iteration}.pt")
                    print(f"Model saved with avg_reward={avg_reward}")
                    prev_max_reward = avg_reward

        self._close()
        return path_to_best

    def _log(self, global_step, start_time, v_loss, pg_loss, entropy_loss, old_approx_kl, approx_kl, clipfracs,
             explained_var):
        # Tracking
        self.writer.add_scalar("charts/learning_rate", self.optimizer.param_groups[0]["lr"], global_step)
        self.writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        self.writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        self.writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        self.writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        self.writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        self.writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        self.writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        self.writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    def _close(self):
        self.envs.close()
        self.writer.close()


if __name__ == "__main__":
    args = SimpleNamespace(
        exp_name='KAZ-V3.3-0',
        seed=31415926,
        wandb_project_name='KAZ-RL',
        wandb_entity="rl2024-umdacs",
        torch_deterministic=True,
        cuda=True,
        learning_rate=4e-4,
        gamma=1.0,
        gae_lambda=0.95,
        clip_coef=0.2,
        norm_adv=True,
        clip_vloss=False,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,
        num_envs=8,
        num_steps=2048,
        num_minibatches=64,
        total_timesteps=327_680,
        update_epochs=10,
        anneal_lr=True
    )

    ppo_trainer = PPO(args)
    archer_path = ppo_trainer.train()
    for b in range(1, 6):
        args.exp_name = f"KAZ-V3.1-{b}"
        ppo_trainer = PPO(args)
        ppo_trainer._setup(to_train='knight_0', path_to_train=None, path_to_play=archer_path)
        knight_path = ppo_trainer.train()
        args.exp_name = f"KAZ-V3.2-{b}"
        ppo_trainer = PPO(args)
        ppo_trainer._setup(to_train='archer_0', path_to_train=archer_path, path_to_play=knight_path)
        archer_path = ppo_trainer.train()
