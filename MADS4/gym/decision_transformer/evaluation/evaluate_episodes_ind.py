import numpy as np
import torch
from decision_transformer.models.s4_muj_ind import *
import logging
import time
import os
import sys
from memory_profiler import memory_usage
logger = logging.getLogger(__name__)

def one_hot(number, dimension):
    one_hot_vector = np.zeros(dimension)
    one_hot_vector[number] = 1
    return np.array(one_hot_vector)

def evaluate_episode(
        env,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        device='cuda',
        target_return=None,
        mode='normal',
        state_mean=0.,
        state_std=1.,
):

    model.eval()
    model.to(device=device)

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32)
    sim_states = []

    episode_return, episode_length = 0, 0
    for t in range(max_ep_len):

        # add padding
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        action = model.get_action(
            (states.to(dtype=torch.float32) - state_mean) / state_std,
            actions.to(dtype=torch.float32),
            rewards.to(dtype=torch.float32),
            target_return=target_return,
        )
        actions[-1] = action
        action = action.detach().cpu().numpy()

        state, reward, done, _ = env.step(action)

        cur_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, cur_state], dim=0)
        rewards[-1] = reward

        episode_return += reward
        episode_length += 1

        if done:
            break
    return episode_return, episode_length

episode = 0
def evaluate_episode_rtg(
        env,
        obs_dim,
        state_dim,
        act_dim,
        model,
        max_ep_len=1000,
        scale=1.0,
        state_mean=0.,
        state_std=1.,
        device='cuda',
        target_return=None,
        mode='normal',):
    global episode
    episode += 1

    model.eval()
    model.to(device=device)
    s4_rec = False
    if isinstance(model, S4_mujoco_wrapper):
        if model.config.single_step_val:
            s4_rec = True
            s4_states = [r.detach() for r in model.get_initial_state((1), device)]

    num_agents = env.num_agents

    for i in range(num_agents):
        state_mean[i] = torch.from_numpy(state_mean[i]).to(device=device)
        state_std[i] = torch.from_numpy(state_std[i]).to(device=device)

    T_rewards, T_wins, steps, episode_dones = 0., 0., 0, 0
    obs, share_obs, available_actions = env.real_env.reset()


    if mode == 'noise':
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # we keep all the histories on the device
    # note that the latest action and reward will be "padding"
    obs_ = [torch.from_numpy(obs[:,i:i+1,:]).to(device=device) for i in range(obs.shape[1])]
    share_obs_ = [torch.from_numpy(share_obs[:, i:i + 1, :]).to(device=device) for i in range(share_obs.shape[1])]

    for i in range(len(obs_)):
        obs_[i] = (obs_[i] - state_mean[i]) / state_std[i]
    obs = obs_
    states = share_obs_
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)# for each agent
    rewards = torch.zeros(0, device=device, dtype=torch.float32)

    ep_return = target_return
    target_return = torch.tensor(ep_return, device=device, dtype=torch.float32).reshape(1, 1)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)

    sim_states = []

    episode_return, episode_length = 0, 0
    actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)# for each agent
    actions = [actions.repeat(1, 1, 1) for _ in range(num_agents)]# for all agents
    rewards = torch.cat([rewards, torch.zeros(1, device=device)])
    rewards = [rewards.repeat(1, 1, 1) for _ in range(num_agents)]  # for all agents
    timesteps = [timesteps for _ in range(num_agents)]
    target_return = [target_return.repeat(1, 1, 1) for _ in range(num_agents)]

    logging.info(f"LOG EVAL TIME: STARTING EVAL :: {device}")
    t = 0
    # for t in range(max_ep_len):
    reward_mean = 0
    first_nan = False
    while True:

        # add padding
        if t > 500 and t <= 503:
            eval_start = time.time()
        if s4_rec:
            action_logits, s4_states = model.get_action(
                obs,
                states,
                actions,
                rewards,
                target_return,
                timesteps,
                s4_states=s4_states)
            if model.config.base_model == "ant_reward_target":
                action = action[0]
            if model.config.discrete > 0:
                action, pred_rtg = action
                maxim = torch.argmax(action, dim=-1).to(dtype=action.dtype)
                action = (maxim * 2 / model.config.discrete - 1 )[0, -1, :model.action_dim]
                pred_state = (maxim / (model.config.discrete / (model.config.state_bound[1] -model.config.state_bound[0])) - model.config.state_bound[0])[0, -1, model.action_dim:]
            if t > 0:
                actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1, device=device)])
        else:

            if t > 0:
                actions = [torch.cat([action, torch.zeros((1, 1, act_dim), device=device)], dim=1) for action in actions]
                rewards = [torch.cat([reward, torch.zeros((1, 1, 1), device=device)], dim=1) for reward in rewards]
                # actions = [torch.cat([torch.zeros((1, 1, act_dim), device=device), action], dim=1) for action in
                #            actions]
                # rewards = [torch.cat([torch.zeros((1, 1, 1), device=device), reward], dim=1) for reward in rewards]
                # actions = [torch.cat([action, torch.zeros((1, 1, act_dim), device=device)], dim=1) for action in
                #            actions]
                # rewards = [torch.cat([reward, torch.zeros((1, 1, 1), device=device)], dim=1) for reward in rewards]

            action_logits = model.get_action(
                obs,
                states,
                actions,
                rewards,
                target_return,
                timesteps,
            )
            # action_logits = model.get_action(
            #     states.to(dtype=torch.float32),
            #     actions.to(dtype=torch.float32),
            #     rewards.to(dtype=torch.float32),
            #     target_return.to(dtype=torch.float32),
            #     timesteps.to(dtype=torch.long),
            # )
            action = [None]*num_agents
            for agent in range(num_agents):
                if available_actions[0,agent,:] is not None:
                    action_logits[agent][available_actions[0,agent,:] == 0] = -1e10
                probs = torch.nn.functional.softmax(action_logits[agent], dim=-1)
                action[agent] = torch.multinomial(probs, num_samples=1)
                # if torch.isnan(probs).any():
                #     if not first_nan:
                #         print('Episode No : ', episode)
                #         print('Time Step : ', t)
                #         first_nan = True
                #
                #     # print(f"yes nan!!")
                #     # print(available_actions[0,agent,:])
                #     avail_actions = torch.where(torch.tensor(available_actions[0,agent,:]) == 1)[0]
                #     action_ind = torch.randint(0, avail_actions.size(0), (1,))
                #     action[agent] = avail_actions[action_ind]
                #     # probs = torch.zeros_like(probs)
                #     # action[agent] = torch.tensor([1]).to(device)
                # else:
                #     action[agent] = torch.multinomial(probs, num_samples=1)


        for i in range(num_agents):
            actions[i][:, -1, :] = torch.from_numpy(one_hot(action[i], act_dim)).to(device)

        action = np.array([a.cpu().numpy()[0] for a in action])#actions as numpy arrays to pass to env.step

        cur_global_obs = share_obs
        cur_local_obs = obs
        cur_ava = available_actions

        obs_, share_obs, rewards_, dones, infos, available_actions = env.real_env.step([action])
        # obs = padding_obs(obs, self.local_obs_dim)
        # share_obs = padding_obs(share_obs, self.global_obs_dim)
        # available_actions = padding_ava(available_actions, self.action_dim)
        t += 1
        # state, reward, done, _ = env.step(action)
        reward_mean += np.mean(rewards_)
        cur_obs = [torch.from_numpy(obs_[:, i:i + 1, :]).to(device=device) for i in range(obs_.shape[1])]
        cur_states = [torch.from_numpy(share_obs[:, i:i + 1, :]).to(device=device) for i in range(share_obs.shape[1])]
        # obs_normalization
        for i in range(len(states)):
            cur_obs[i] = (cur_obs[i] - state_mean[i]) / state_std[i]
        obs = [torch.cat([obs[i], cur_obs[i]], dim=1) for i in range(num_agents)]
        states = [torch.cat([states[i], cur_states[i]], dim=1) for i in range(num_agents)]
        rewards_ = [torch.from_numpy(rewards_[:, i:i + 1, :]).to(device=device) for i in range(rewards_.shape[1])]
        for i in range(num_agents):
            # rewards[i][-1] = rewards_[i][0]
            rewards[i][:, -1, :] = rewards_[i][0][0]
        if mode != 'delayed':
            for i in range(num_agents):
                pred_return = target_return[i][0, -1] - rewards_[i][0, -1] / scale
                target_return[i] = torch.cat(
                    [target_return[i], pred_return.reshape(1, 1, 1)], dim=1)

                timesteps[i] = torch.cat(
                    [timesteps[i],
                     torch.ones((1, 1), device=device, dtype=torch.long) * (t)], dim=1)

            # pred_return = target_return[0,-1] - (reward/scale)
        else:
            pred_return = target_return[0, -1]

        # episode_return += reward
        episode_length += 1
        steps += 1

        if np.all(dones):
            episode_dones = 1
            T_rewards = reward_mean  # mean across agents
            if infos[0][0]['won']:
                T_wins += 1.
            break

    average_diff, last_action_diff = 0, 0
    if s4_rec:
        if model.config.track_step_err:
            actions = actions.to(dtype=torch.float32).reshape(1, -1, act_dim)
            states = states.to(dtype=torch.float32).reshape(1, -1, state_dim)
            target_return = target_return.to(dtype=torch.float32).reshape(1, -1, 1)
            _, predicted_actions, _ = model.forward(
                (states.to(dtype=torch.float32)[0, :-1, :].unsqueeze(0) - state_mean) / state_std,
                actions,
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32)[0, :-1, 0].unsqueeze(0),
                timesteps,
                running=True
            )
            delta = predicted_actions[:, :-1, :] - actions[:, 1:, :]
            average_diff = delta.abs().mean().cpu().item()
            last_action_diff = delta[:, -1, :].abs().mean().cpu().item()

    print(f"evaluation: average return:{T_rewards}, average win rate:{T_wins}")
    return T_rewards, steps, average_diff, last_action_diff
