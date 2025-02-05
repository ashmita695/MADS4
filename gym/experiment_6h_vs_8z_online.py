import gym
import numpy as np
import torch
import wandb

import argparse
import pickle
import random
import sys
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
s4dir = os.path.join(os.path.dirname(currentdir), "s4_module")
sys.path.insert(0, s4dir)

import logging
import time
import datetime

from decision_transformer.evaluation.evaluate_episodes import evaluate_episode, evaluate_episode_rtg
from decision_transformer.models.decision_transformer import DecisionTransformer
from decision_transformer.models.s4_muj import *
from decision_transformer.models.mlp_bc import MLPBCModel
from decision_transformer.training.act_trainer import ActTrainer
from decision_transformer.training.seq_trainer import SequenceTrainer

from data_ma.smac.utils import get_dim_from_space
from decision_transformer.envs.env import Env
from decision_transformer.training.online_training import OnlineTrainer
from decision_transformer.training.online_training_trajectory import OnlineTrainer_Traj


def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0] - 1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t + 1]
    return discount_cumsum


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def one_hot(number, dimension):
    one_hot_vector = np.zeros(dimension)
    one_hot_vector[number] = 1
    return np.array(one_hot_vector)


def experiment(
        exp_prefix,
        variant,
):
    device = variant.get('device', 'cuda')
    log_to_wandb = variant.get('log_to_wandb', False)

    env_name, map_name, dataset = variant['env_name'], variant['map_name'], variant['dataset']
    model_type = variant['model_type']
    group_name = f'{exp_prefix}-{env_name}-{dataset}'
    exp_prefix = f'{group_name}-{random.randint(int(1e5), int(1e6) - 1)}'

    if env_name == 'StarCraft2':
        env = Env(args)
        eval_env = Env(args)
        online_train_env = Env(args)
        global_obs_dim = get_dim_from_space(online_train_env.real_env.share_observation_space)
        local_obs_dim = get_dim_from_space(online_train_env.real_env.observation_space)
        action_dim = get_dim_from_space(online_train_env.real_env.action_space)
        num_agents = online_train_env.num_agents
        # block_size = args.context_length * 1
        print(
            f"global obs dim:{global_obs_dim}; local obs dim:{local_obs_dim}; action dim:{action_dim}; num agents:{num_agents}")

    dataset_path = f'data_ma/smac/6h_vs_8z/6h_vs_8z_1000/good/data_smac.pkl'
    with open(dataset_path, 'rb') as f:
        trajectories = pickle.load(f)

    # save all path information into separate lists
    n_agents = len(trajectories[0]['observations'])
    print(f"number of agents:{n_agents}")

    mode = variant.get('mode', 'normal')
    states, traj_lens, returns = [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in
                                                                                                range(n_agents)]
    for path in trajectories:

        obs_dim = path['observations'][0].shape[1]
        state_dim = path['global_states'][0].shape[1]
        act_dim = path['next_available_actions'][0].shape[1]

        for j in range(n_agents):
            states[j].append(path['observations'][j])
            traj_lens[j].append(len(path['observations'][j]))
            returns[j].append(path['rewards'][j][0])

    print(
        f"offline data: global obs dim:{state_dim}; local obs dim:{obs_dim}; action dim:{action_dim}; num agents:{n_agents}")

    traj_lens, returns = np.array(traj_lens), np.array(returns)
    # used for input normalization
    state_bound = []
    state_mean = []
    state_std = []
    for i in range(n_agents):
        states[i] = np.concatenate(states[i], axis=0)
        state_mean.append(np.mean(states[i], axis=0))
        state_std.append(np.std(states[i], axis=0) + 1e-6)
        nor_states = (states[i] - state_mean[i]) / state_std[i]
        state_bound.append([np.min(nor_states), np.max(nor_states)])
    num_timesteps = sum(traj_lens[0])

    print('=' * 50)
    print(f'Starting new experiment: {env_name} {map_name} {dataset}')
    print(f'{len(traj_lens[0])} trajectories, {num_timesteps} timesteps found')
    print(f'Max timesteps: {np.max(traj_lens[0])}, Min timesteps:{np.min(traj_lens[0])}')
    print(f'Mean timesteps: {np.mean(traj_lens[0])}')
    print(f'Average return: {np.mean(returns[0]):.2f}, std: {np.std(returns[0]):.2f}')
    print(f'Max return: {np.max(returns[0]):.2f}, min: {np.min(returns[0]):.2f}')
    print('=' * 50)

    scale = 1.0
    # env_targets = [np.max(returns[0]) * 1.20, np.min(returns[0]) * 0.90]  # 10% increase and decrease
    env_targets = [12.5, 13.5, 15.0, 20.0, 25]
    max_ep_len = 50
    # got_goals = trajectories[0].get('infos/goal', None)
    # if got_goals is not None:
    #     print(f"Targets sample normalized:")
    #     for l in range(0, 1000, 50):
    #         print(f"Tar {l:4} : {(trajectories[l].get('infos/goal')[5, :] - state_mean[:2]) / state_std[:2]}")
    #     print('=' * 50)

    K = variant['K']
    batch_size = variant['batch_size']
    num_eval_episodes = variant['num_eval_episodes']
    pct_traj = variant.get('pct_traj', 1.)

    # only train on top pct_traj trajectories (for %BC experiment)
    num_timesteps = max(int(pct_traj * num_timesteps), 1)
    sorted_inds = np.argsort(returns[0])  # lowest to highest
    num_trajectories = 1
    timesteps = traj_lens[0][sorted_inds[-1]]  # 47
    ind = len(trajectories) - 2
    while ind >= 0 and timesteps + traj_lens[0][sorted_inds[ind]] < num_timesteps:
        timesteps += traj_lens[0][sorted_inds[ind]]
        num_trajectories += 1
        ind -= 1
    sorted_inds = sorted_inds[-num_trajectories:]

    # used to reweight sampling so we sample according to timesteps instead of trajectories
    p_sample = traj_lens[0][sorted_inds] / sum(traj_lens[0][sorted_inds])

    # def get_batch(batch_size=256, max_len=K):
    #     print(f"max length:{max_len}")
    #     batch_inds = np.random.choice(
    #         np.arange(num_trajectories),
    #         size=batch_size,
    #         replace=True,
    #         p=p_sample,  # reweights so we sample according to timesteps
    #     )
    #     n_agents = len(trajectories[0]['observations'])
    #     s, a, r, d, rtg, timesteps, mask = [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)]
    #     for i in range(batch_size):
    #         traj = trajectories[int(sorted_inds[batch_inds[i]])]
    #         # print(type(traj))#dict
    #         # print(f"length of trajectory:{traj['rewards'].shape[0]}")#length of the episode trajectory:not all equal
    #         si = random.randint(0, traj['rewards'][0].shape[0] - 1)
    #         # print(f"starting index:{si}")
    #         # get sequences from dataset
    #         # print(f"state list:{traj['observations'][si:si + max_len].shape}")# get list of observations from si to end of the episode or max_len
    #         for j in range(n_agents):
    #             s[j].append(traj['observations'][j][si:si + max_len].reshape(1, -1, obs_dim))
    #             acts_ = traj['actions'][j][si:si + max_len]
    #             actions_one_hot = np.array([one_hot(acts_[k][0], act_dim) for k in range(acts_.shape[0])])
    #             a[j].append(actions_one_hot.reshape(1, -1, act_dim))
    #             # a[j].append(traj['actions'][j][si:si + max_len].reshape(1, -1, act_dim))
    #             r[j].append(traj['rewards'][j][si:si + max_len].reshape(1, -1, 1))
    #             if 'terminals' in traj:
    #                 d[j].append(traj['terminals'][j][si:si + max_len].reshape(1, -1))
    #             else:
    #                 d[j].append(traj['dones'][j][si:si + max_len].reshape(1, -1))
    #             # print(s[-1].shape[1]) how many timesteps are included
    #             timesteps[j].append(np.arange(si, si + s[j][-1].shape[1]).reshape(1, -1))
    #             # print(timesteps[-1])# last appended timesteps list
    #             timesteps[j][-1][timesteps[j][-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
    #             # print(s[-1].shape[1])
    #             rtg[j].append(discount_cumsum(traj['rewards'][j][si:], gamma=1.)[:s[j][-1].shape[1] + 1].reshape(1, -1, 1))
    #             if rtg[j][-1].shape[1] <= s[j][-1].shape[1]:
    #                 rtg[j][-1] = np.concatenate([rtg[j][-1], np.zeros((1, 1, 1))], axis=1)
    #
    #             # padding and state + reward normalization
    #             tlen = s[j][-1].shape[1]  # starting index to length of episode
    #             s[j][-1] = np.concatenate([np.zeros((1, max_len - tlen, obs_dim)), s[j][-1]],
    #                                    axis=1)  # concatenated to make all batches size to max_len
    #             s[j][-1] = (s[j][-1] - state_mean[j]) / state_std[j]
    #             # a[j][-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[j][-1]], axis=1)
    #             a[j][-1] = np.concatenate([np.zeros((1, max_len - tlen, act_dim)), a[j][-1]], axis=1)
    #             r[j][-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), r[j][-1]], axis=1)
    #             d[j][-1] = np.concatenate([np.ones((1, max_len - tlen)) * 2, d[j][-1]], axis=1)
    #             rtg[j][-1] = np.concatenate([np.zeros((1, max_len - tlen, 1)), rtg[j][-1]], axis=1) / scale
    #             timesteps[j][-1] = np.concatenate([np.zeros((1, max_len - tlen)), timesteps[j][-1]], axis=1)
    #             mask[j].append(np.concatenate([np.zeros((1, max_len - tlen)), np.ones((1, tlen))], axis=1))
    #
    #     for i in range(n_agents):
    #         s[i] = torch.from_numpy(np.concatenate(s[i], axis=0)).to(dtype=torch.float32, device=device)
    #         a[i] = torch.from_numpy(np.concatenate(a[i], axis=0)).to(dtype=torch.float32, device=device)
    #         r[i] = torch.from_numpy(np.concatenate(r[i], axis=0)).to(dtype=torch.float32, device=device)
    #         d[i] = torch.from_numpy(np.concatenate(d[i], axis=0)).to(dtype=torch.long, device=device)
    #         rtg[i] = torch.from_numpy(np.concatenate(rtg[i], axis=0)).to(dtype=torch.float32, device=device)
    #         timesteps[i] = torch.from_numpy(np.concatenate(timesteps[i], axis=0)).to(dtype=torch.long, device=device)
    #         mask[i] = torch.from_numpy(np.concatenate(mask[i], axis=0)).to(device=device)
    #         # print(s.shape)#batch size*max_sequence_length*obs_dim
    #     return s, a, r, d, rtg, timesteps, mask, None

    def get_batch(batch_size=256, max_len=K):
        print(f"max length:{max_len}")

        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        n_agents = len(trajectories[0]['observations'])
        # s, a, r, d, rtg, timesteps, mask = [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)]
        o, s, a, r, d, rtg, timesteps, mask = [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in
                                            range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in range(n_agents)]
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            # print(type(traj))#dict
            # print(f"length of trajectory:{traj['rewards'].shape[0]}")#length of the episode trajectory:not all equal
            # si = random.randint(0, traj['rewards'][0].shape[0] - 1)
            si = 0
            # print(f"starting index:{si}")
            # get sequences from dataset
            # print(f"state list:{traj['observations'][si:si + max_len].shape}")# get list of observations from si to end of the episode or max_len
            for j in range(n_agents):
                o[j].append(traj['observations'][j][si:si + max_len].reshape(1, -1, obs_dim))
                s[j].append(traj['global_states'][j][si:si + max_len].reshape(1, -1, state_dim))
                # s[j].append(traj['observations'][j][si:si + max_len].reshape(1, -1, obs_dim))
                acts_ = traj['actions'][j][si:si + max_len]
                actions_one_hot = np.array([one_hot(acts_[k][0], act_dim) for k in range(acts_.shape[0])])
                a[j].append(actions_one_hot.reshape(1, -1, act_dim))
                # a[j].append(traj['actions'][j][si:si + max_len].reshape(1, -1, act_dim))
                r[j].append(traj['rewards'][j][si:si + max_len].reshape(1, -1, 1))  # rtg
                if 'terminals' in traj:
                    d[j].append(traj['terminals'][j][si:si + max_len].reshape(1, -1))
                else:
                    d[j].append(traj['dones'][j][si:si + max_len].reshape(1, -1))
                # print(s[-1].shape[1]) how many timesteps are included
                timesteps[j].append(np.arange(si, si + s[j][-1].shape[1]).reshape(1, -1))
                # print(timesteps[-1])# last appended timesteps list
                timesteps[j][-1][timesteps[j][-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff
                # print(s[-1].shape[1])
                # rtg[j].append(discount_cumsum(traj['rewards'][j][si:], gamma=1.)[:s[j][-1].shape[1] + 1].reshape(1, -1, 1))
                rtg[j].append(traj['rewards'][j][si:][:s[j][-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg[j][-1].shape[1] <= s[j][-1].shape[1]:
                    rtg[j][-1] = np.concatenate([rtg[j][-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization
                tlen = s[j][-1].shape[1]  # length of episode
                o[j][-1] = np.concatenate([o[j][-1], np.zeros((1, max_len - tlen, obs_dim))], axis=1)
                o[j][-1] = (o[j][-1] - state_mean[j]) / state_std[j]
                s[j][-1] = np.concatenate([s[j][-1], np.zeros((1, max_len - tlen, state_dim))], axis=1)
                # s[j][-1] = np.concatenate([s[j][-1], np.zeros((1, max_len - tlen, obs_dim))], axis=1)
                # s[j][-1] = (s[j][-1] - state_mean[j]) / state_std[j]

                # a[j][-1] = np.concatenate([np.ones((1, max_len - tlen, act_dim)) * -10., a[j][-1]], axis=1)
                a[j][-1] = np.concatenate([a[j][-1], np.zeros((1, max_len - tlen, act_dim))], axis=1)
                r[j][-1] = np.concatenate([r[j][-1], np.zeros((1, max_len - tlen, 1))], axis=1)
                d[j][-1] = np.concatenate([d[j][-1], np.ones((1, max_len - tlen)) * 1], axis=1)
                rtg[j][-1] = np.concatenate([rtg[j][-1], np.zeros((1, max_len - tlen, 1))], axis=1) / scale
                timesteps[j][-1] = np.concatenate([timesteps[j][-1], np.zeros((1, max_len - tlen))], axis=1)
                mask[j].append(np.concatenate([np.ones((1, tlen)), np.zeros((1, max_len - tlen))], axis=1))

        for i in range(n_agents):
            o[i] = torch.from_numpy(np.concatenate(o[i], axis=0)).to(dtype=torch.float32, device=device)
            s[i] = torch.from_numpy(np.concatenate(s[i], axis=0)).to(dtype=torch.float32, device=device)
            # s[i] = torch.from_numpy(np.concatenate(s[i], axis=0)).to(dtype=torch.float32, device=device)
            a[i] = torch.from_numpy(np.concatenate(a[i], axis=0)).to(dtype=torch.float32, device=device)
            r[i] = torch.from_numpy(np.concatenate(r[i], axis=0)).to(dtype=torch.float32, device=device)
            d[i] = torch.from_numpy(np.concatenate(d[i], axis=0)).to(dtype=torch.long, device=device)
            rtg[i] = torch.from_numpy(np.concatenate(rtg[i], axis=0)).to(dtype=torch.float32, device=device)
            timesteps[i] = torch.from_numpy(np.concatenate(timesteps[i], axis=0)).to(dtype=torch.long, device=device)
            mask[i] = torch.from_numpy(np.concatenate(mask[i], axis=0)).to(device=device)
            # print(s.shape)#batch size*max_sequence_length*obs_dim
        # return s, a, r, d, rtg, timesteps, mask, None
        return o, s, a, r, d, rtg, timesteps, mask, None

    def get_batch_recurrent(batch_size=256):

        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        max_len = max([traj_lens[0][sorted_inds[batch_inds[i]]] for i in range(batch_size)])
        max_len = max(max_len, 3)
        print(f"LOG max len for batch: {max_len}")
        startpoint = 0
        new_len = int(1.0 * variant['partial_traj'] * max_ep_len)
        if variant['partial_traj'] < 1 and max_len > new_len:
            startpoint = max_len - new_len
            max_len = new_len
            print(f"LOG max len for batch (new): {max_len}")

        # s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], [], []
        o, s, a, r, d, rtg, timesteps, mask = [[] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _ in
                                                                                                             range(
                                                                                                                 n_agents)], [
                                                  [] for _ in range(n_agents)], [[] for _ in range(n_agents)], [[] for _
                                                                                                                in
                                                                                                                range(
                                                                                                                    n_agents)], [
                                                  [] for _ in range(n_agents)], [[] for _ in range(n_agents)]
        goals = []

        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            if variant['partial_traj'] < 1:
                si = random.randint(0, max(0, len(traj['observations']) - max_len))
            else:
                si = 0

            # get sequences from dataset
            for j in range(n_agents):
                o[j].append(traj['observations'][j][si:si + max_ep_len].reshape(1, -1, obs_dim))
                s[j].append(traj['global_states'][j][si:si + max_ep_len].reshape(1, -1, state_dim))
                # s[j].append(traj['observations'][j][si:si + max_len].reshape(1, -1, obs_dim))
                acts_ = traj['actions'][j][si:si + max_ep_len]
                actions_one_hot = np.array([one_hot(acts_[k][0], act_dim) for k in range(acts_.shape[0])])
                a[j].append(actions_one_hot.reshape(1, -1, act_dim))
                # a[j].append(traj['actions'][j][si:si + max_len].reshape(1, -1, act_dim))
                r[j].append(traj['rewards'][j][si:si + max_ep_len].reshape(1, -1, 1))  # rtg
                if 'terminals' in traj:
                    d[j].append(traj['terminals'][j][si:si + max_ep_len].reshape(1, -1))
                else:
                    d[j].append(traj['dones'][j][si:si + max_ep_len].reshape(1, -1))
                # print(s[-1].shape[1]) how many timesteps are included
                timesteps[j].append(np.arange(si, si + s[j][-1].shape[1]).reshape(1, -1))
                timesteps[j][-1][timesteps[j][-1] >= max_ep_len] = max_ep_len - 1  # padding cutoff

                rtg[j].append(traj['rewards'][j][si:][:s[j][-1].shape[1] + 1].reshape(1, -1, 1))
                if rtg[j][-1].shape[1] <= s[j][-1].shape[1]:
                    rtg[j][-1] = np.concatenate([rtg[j][-1], np.zeros((1, 1, 1))], axis=1)

                # padding and state + reward normalization
                tlen = s[j][-1].shape[1]  # length of episode
                o[j][-1] = np.concatenate([o[j][-1], np.zeros((1, max_ep_len - tlen, obs_dim))], axis=1)
                o[j][-1] = (o[j][-1] - state_mean[j]) / state_std[j]
                s[j][-1] = np.concatenate([s[j][-1], np.zeros((1, max_ep_len - tlen, state_dim))], axis=1)
                a[j][-1] = np.concatenate([a[j][-1], np.zeros((1, max_ep_len - tlen, act_dim))], axis=1)
                r[j][-1] = np.concatenate([r[j][-1], np.zeros((1, max_ep_len - tlen, 1))], axis=1)
                d[j][-1] = np.concatenate([d[j][-1], np.ones((1, max_ep_len - tlen)) * 1], axis=1)
                rtg[j][-1] = np.concatenate([rtg[j][-1], np.zeros((1, max_ep_len - tlen, 1))], axis=1) / scale
                timesteps[j][-1] = np.concatenate([timesteps[j][-1], np.zeros((1, max_ep_len - tlen))], axis=1)
                mask[j].append(np.concatenate([np.ones((1, tlen)), np.zeros((1, max_ep_len - tlen))], axis=1))

            got_goals = traj.get('infos/goal', None)
            if got_goals is not None:
                goals.append((got_goals[5, :] - state_mean[:2]) / state_std[:2])
            else:
                goals.append(got_goals)

        for i in range(n_agents):
            o[i] = torch.from_numpy(np.concatenate(o[i], axis=0)).to(dtype=torch.float32, device=device)
            s[i] = torch.from_numpy(np.concatenate(s[i], axis=0)).to(dtype=torch.float32, device=device)
            # s[i] = torch.from_numpy(np.concatenate(s[i], axis=0)).to(dtype=torch.float32, device=device)
            a[i] = torch.from_numpy(np.concatenate(a[i], axis=0)).to(dtype=torch.float32, device=device)
            r[i] = torch.from_numpy(np.concatenate(r[i], axis=0)).to(dtype=torch.float32, device=device)
            d[i] = torch.from_numpy(np.concatenate(d[i], axis=0)).to(dtype=torch.long, device=device)
            rtg[i] = torch.from_numpy(np.concatenate(rtg[i], axis=0)).to(dtype=torch.float32, device=device)
            timesteps[i] = torch.from_numpy(np.concatenate(timesteps[i], axis=0)).to(dtype=torch.long, device=device)
            mask[i] = torch.from_numpy(np.concatenate(mask[i], axis=0)).to(device=device)

        if goals[0] is None:
            goals = None
        else:
            goals = torch.from_numpy(np.concatenate([z.reshape((1, 2)) for z in goals], axis=0)).to(dtype=torch.float32,
                                                                                                    device=device)

        return o, s, a, r, d, rtg, timesteps, mask, goals

    def eval_episodes(target_rew):
        def fn(model):
            returns, lengths, all_average_diff, all_last_action_diff = [], [], [], []
            for _ in range(num_eval_episodes):
                with torch.no_grad():

                    if model_type == 'dt' or 's4' in model_type:
                        if isinstance(state_mean[0], np.ndarray):
                            ret, length, average_diff, last_action_diff = evaluate_episode_rtg(
                                env,
                                obs_dim, state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                scale=scale,
                                target_return=target_rew / scale,
                                mode=mode,
                                state_mean=[s_mean for s_mean in state_mean],
                                state_std=[s_std for s_std in state_std],
                                device=device if not bool(variant['cpu_inf']) else 'cpu',
                            )
                        else:
                            ret, length, average_diff, last_action_diff = evaluate_episode_rtg(
                                env,
                                obs_dim, state_dim,
                                act_dim,
                                model,
                                max_ep_len=max_ep_len,
                                scale=scale,
                                target_return=target_rew / scale,
                                mode=mode,
                                state_mean=[s_mean.numpy() for s_mean in state_mean],
                                state_std=[s_std.numpy() for s_std in state_std],
                                device=device if not bool(variant['cpu_inf']) else 'cpu',
                            )

                    else:
                        average_diff, last_action_diff = 0, 0
                        ret, length = evaluate_episode(
                            env,
                            obs_dim, state_dim,
                            act_dim,
                            model,
                            max_ep_len=max_ep_len,
                            target_return=target_rew / scale,
                            mode=mode,
                            state_mean=state_mean,
                            state_std=state_std,
                            device=device,
                        )
                returns.append(ret)
                lengths.append(length)
                all_average_diff.append(average_diff)
                all_last_action_diff.append(last_action_diff)
            return {
                f'current avg returns': np.mean(returns),
                f'target_{target_rew}_return_mean': np.mean(returns),
                f'target_{target_rew}_return_std': np.std(returns),
                f'target_{target_rew}_length_mean': np.mean(lengths),
                f'target_{target_rew}_length_std': np.std(lengths),
                f'target_{target_rew}_diff_average_mean': np.mean(all_average_diff),
                f'target_{target_rew}_diff_average_std': np.std(all_average_diff),
                f'target_{target_rew}_diff_av_last_action_mean': np.mean(all_last_action_diff),
                f'target_{target_rew}_diff_av_last_action_std': np.std(all_last_action_diff),
            }

        return fn

    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    logging.info(f"LOG EVAL MEMMODPRE: tot_m: {tot_m} , used_m: {used_m} , free_m: {free_m}")
    if 's4' in model_type:
        s4_config = S4_config(s4_layers=variant['s4_layers'],
                              dropout=variant['s4dropoutval'],
                              len_corr=bool(variant['s4_singlestep']),
                              single_step_val=bool(variant['s4_singlestep']),
                              layer_norm_s4=bool(variant['layer_norm_s4']),
                              s4_onpolicy=bool(variant['s4_onpolicy']),
                              s4_resnet=bool(variant['s4_resnet']),
                              s4_trainable=bool(variant['s4_trainable']) or variant['s4_onpolicy_afteroff'] > 0,
                              n_ssm=variant['s4_n_ssm'],
                              base_model=variant['s4_ablation_model'],
                              train_noise=variant['s4_train_noise'],
                              precision=variant['s4_precision'],
                              track_step_err=bool(variant['track_step_err']),
                              discrete=variant['s4_discrete'],
                              state_bound=state_bound,
                              recurrent_mode=bool(variant['s4_recurrent_mode_train']),
                              s4_ant_multi_lr=str(variant['s4_ant_multi_lr']) if variant[
                                                                                     's4_ant_multi_lr'] != 'none' else None)
    if model_type == 'dt':
        model = DecisionTransformer(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            max_ep_len=max_ep_len,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
            n_head=variant['n_head'],
            n_inner=4 * variant['embed_dim'],
            activation_function=variant['activation_function'],
            n_positions=1024,
            resid_pdrop=variant['dropout'],
            attn_pdrop=variant['dropout'],
        )
    elif model_type == 'bc':
        model = MLPBCModel(
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            hidden_size=variant['embed_dim'],
            n_layer=variant['n_layer'],
        )
    elif model_type == 's4v3':

        model = S4_mujoco_wrapper_v3(
            config=s4_config,
            obs_dim=obs_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            l_max=K,
            s4_weight_decay=variant['s4weight_decay'],
            n_embd=variant['embed_dim'],
            d_state=variant['s4internal'],
            H=variant['s4interface'],
            kernel_mode=variant['s4_mode'],
        )
        if bool(variant['s4_onpolicy']):
            model_target = S4_mujoco_wrapper_v3(
                config=s4_config,
                state_dim=state_dim,
                act_dim=act_dim,
                max_length=K,
                l_max=K,
                s4_weight_decay=variant['s4weight_decay'],
                n_embd=variant['embed_dim'],
                d_state=variant['s4internal'],
                H=variant['s4interface'],
                kernel_mode=variant['s4_mode'],
            )
            model_target = model_target.to(device)
    else:
        raise NotImplementedError

    logging.info("Using model type of " + model_type + ": " + str(type(model)))
    logging.info(
        "Using model of size: " + str(sum(param.numel() for param in model.parameters() if param.requires_grad)))
    if 's4' in model_type:
        to_print = model.reprr()
        logging.info(to_print)
    model = model.to(device=device)

    warmup_steps = variant['warmup_steps']

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )

    if variant['s4_load_model'] != "none":
        file_modl = variant['s4_load_model'] + "_actor.pkl"
        # mujoco_hopp_107500_online_latest_critic.pkl
        print(f"Using model from {file_modl}")
        logging.info(f"Using model from {file_modl}")
        loadedfile = torch.load(file_modl, torch.device(device))
        model = loadedfile['model'].to(device)
        optimizer = loadedfile['optimizer']
        for g in optimizer.param_groups:
            g['lr'] = variant['learning_rate']
        if bool(variant['s4_onpolicy']):
            loadedfile = torch.load(file_modl, torch.device(device))
            model_target = loadedfile['model'].to(device)

    # file_path = 'smac_best_all_dep_no_tanh_only_prev_act_act_prev_rew_end.pkl'
    # file_path = f"6h_8z_good_agent_id_obs_embed_prev_act_embed_prev_mem_embed_act_rew_embed_offline.pkl"
    # # # Load the state dictionary
    # state_dict = torch.load(file_path)

    # Load the state dictionary into the model
    # model.load_state_dict(state_dict)
    # print(model)
    model = model.to(device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=variant['learning_rate'],
        weight_decay=variant['weight_decay'],
    )

    tot_m, used_m, free_m = map(int, os.popen('free -t -m').readlines()[-1].split()[1:])
    logging.info(f"LOG EVAL MEMMODPOST: tot_m: {tot_m} , used_m: {used_m} , free_m: {free_m}")

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda steps: min((steps + 1) / warmup_steps, 1)
    )

    critic, critic2 = None, None
    critic_optimizer = None
    critic_scheduler = None
    if bool(variant['s4_onpolicy']) or bool(variant['s4_offpol_actorcritic']) or bool(variant['s4_onpolicy_afteroff']):
        print(
            f"Building Critic of {variant['online_critictype']}: {state_dim:5d} {act_dim:5d} {int((state_dim + act_dim) / 2):5d}")
        q_net = QNet(obs_dim=obs_dim, act_dim=act_dim, n_agents=num_agents)
        v_net = VNet(obs_dim, n_agents)
        mix_net = MixNet(state_shape=state_dim, n_agents=n_agents, hyper_hidden_dim=64,num_action=act_dim)
        critic = Critic(q_net, v_net, mix_net)
        critic_target = Critic(q_net, v_net, mix_net)
        # if variant['online_critictype'] == "resnet":
        #     critic = FC_critic_resnet(obs_dim=obs_dim, action_dim=act_dim,
        #                               obs_enc_size=obs_dim, action_enc_size=act_dim,
        #                               mutual_enc_size=obs_dim + act_dim)
        #     critic_target = FC_critic_resnet(obs_dim=obs_dim, action_dim=act_dim,
        #                                      obs_enc_size=obs_dim, action_enc_size=act_dim,
        #                                      mutual_enc_size=obs_dim + act_dim)
        #     mix_net = MixNet(state_shape=state_dim, n_agents=num_agents, hyper_hidden_dim=64, num_action=act_dim)
        # elif variant['online_critictype'] == "shallow":
        #     critic = FC_critic_shallowA(states_dim=state_dim, action_dim=act_dim,
        #                                 state_enc_size=state_dim, action_enc_size=act_dim,
        #                                 mutual_enc_size=int((state_dim + act_dim) / 2))
        #     critic_target = FC_critic_shallowA(states_dim=state_dim, action_dim=act_dim,
        #                                        state_enc_size=state_dim, action_enc_size=act_dim,
        #                                        mutual_enc_size=int((state_dim + act_dim) / 2))
        # elif variant['online_critictype'] == "cat":
        #     critic = FC_critic_cat(states_dim=state_dim, action_dim=act_dim,
        #                            state_enc_size=state_dim, action_enc_size=act_dim,
        #                            mutual_enc_size=int((state_dim + act_dim) / 2))
        #     critic_target = FC_critic_cat(states_dim=state_dim, action_dim=act_dim,
        #                                   state_enc_size=state_dim, action_enc_size=act_dim,
        #                                   mutual_enc_size=int((state_dim + act_dim) / 2))
        # elif variant['online_critictype'] == "cat_exp":
        #     critic = FC_critic_cat_expanded(states_dim=state_dim, action_dim=act_dim,
        #                                     state_enc_size=state_dim, action_enc_size=act_dim,
        #                                     mutual_enc_size=int((state_dim + act_dim) / 2))
        #     critic_target = FC_critic_cat_expanded(states_dim=state_dim, action_dim=act_dim,
        #                                            state_enc_size=state_dim, action_enc_size=act_dim,
        #                                            mutual_enc_size=int((state_dim + act_dim) / 2))
        # elif variant['online_critictype'] == "cat_exp_rtg":
        #     critic = FC_critic_cat_expanded_rtg(states_dim=state_dim, action_dim=act_dim)
        #     critic2 = FC_critic_cat_expanded_rtg(states_dim=state_dim, action_dim=act_dim)
        #     critic2 = critic2.to(device=device)
        #     critic_target = FC_critic_cat_expanded_rtg(states_dim=state_dim, action_dim=act_dim)
        # else:
        #     critic = FC_critic_shallowA_diff(states_dim=state_dim, action_dim=act_dim,
        #                                      state_enc_size=state_dim, action_enc_size=act_dim,
        #                                      mutual_enc_size=(state_dim + act_dim) * 2)
        #     critic_target = FC_critic_shallowA_diff(states_dim=state_dim, action_dim=act_dim,
        #                                             state_enc_size=state_dim, action_enc_size=act_dim,
        #                                             mutual_enc_size=(state_dim + act_dim) * 2)
        critic = critic.to(device=device)
        critic_target = critic_target.to(device=device)

        warmup_steps = variant['warmup_steps']
        critic_optimizer = torch.optim.AdamW(
            critic.parameters(),
            lr=variant['online_critic_lr'],
            weight_decay=variant['online_critic_wd'],
        )
        if variant['s4_load_model'] != "none" and variant['fine_tune_critic_steps'] <= 0:
            file_crit = variant['s4_load_model'] + "_critic.pkl"
            # mujoco_hopp_107500_online_latest_critic.pkl
            print(f"Using model from {file_crit}")
            logging.info(f"Using model from {file_crit}")
            loadedfile = torch.load(file_crit)
            critic = loadedfile['critic'].to(device)
            critic_optimizer = loadedfile['critic_optimizer']
            loadedfile = torch.load(file_crit)
            critic_target = loadedfile['critic'].to(device)

        critic_scheduler = torch.optim.lr_scheduler.LambdaLR(
            critic_optimizer,
            lambda steps: min((steps + 1) / warmup_steps, 1)
        )

    if model_type == 'dt' or 's4' in model_type:
        if variant['s4_onpolicy_afteroff'] > 0:
            pass
        elif not bool(variant['s4_onpolicy']):
            # loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            trainer = SequenceTrainer(
                model=model,
                optimizer=optimizer,
                batch_size=batch_size,
                get_batch=get_batch if not bool(variant['s4_singlestep']) else get_batch_recurrent,
                scheduler=scheduler,
                loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.functional.cross_entropy(a_hat, a),
                eval_fns=[eval_episodes(tar) for tar in env_targets],
                runlabel=variant['online_postfix'],
                critic=[critic, critic2],
                critic_optimizer=critic_optimizer,
                critic_scheduler=critic_scheduler,
                reward_scale=scale,
                variant=variant,
                rtg_set_all=bool(variant['s4_ant_rtg_all_base']),
            )
        else:
            if variant['s4_onpolicy'] == 2:
                trainer = OnlineTrainer_Traj(
                    env=env,
                    model=model,
                    model_target=model_target,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    state_mean=state_mean,
                    state_std=state_std,
                    scheduler=scheduler if not bool(variant['flat_lr']) else None,
                    episodes_per_iteration=variant['episodes_per_iteration'],
                    steps_between_model_swap=variant['online_steps_model_swap'],
                    steps_between_trains=variant['online_steps_between_trains'],
                    min_amount_to_train=variant['online_min_train_step'],
                    online_exploration_type=variant['online_exploration_type'],
                    trains_per_step=variant['online_trains_per_step'],
                    replay_memory_max_size=variant['online_replay_memory_max_size'],
                    s4_load_model=variant['s4_load_model'],
                    eval_fns=[eval_episodes(tar) for tar in env_targets],
                    game_name=env_name,
                    online_savepostifx=variant['online_postfix'],
                    rtg_variation=variant['online_rtg_variation'],
                    online_soft_update=variant['online_soft_update'],
                    base_target_reward=env_targets[0] * 0.95,
                    fine_tune_critic_steps=variant['fine_tune_critic_steps'],
                    online_step_partial_advance=variant['online_step_partial_advance'],
                    fine_tune_critic_steps_speedup=variant['fine_tune_critic_steps_speedup'],

                    eval_base_func=eval_episodes,
                )
            else:
                trainer = OnlineTrainer(
                    env=env,
                    model=model,
                    model_target=model_target,
                    critic=critic,
                    critic_target=critic_target,
                    optimizer=optimizer,
                    batch_size=batch_size,
                    state_mean=state_mean,
                    state_std=state_std,
                    scheduler=scheduler if not bool(variant['flat_lr']) else None,
                    critic_scheduler=critic_scheduler if not bool(variant['flat_lr']) else None,
                    critic_optimizer=critic_optimizer,
                    episodes_per_iteration=variant['episodes_per_iteration'],
                    steps_between_model_swap=variant['online_steps_model_swap'],
                    steps_between_trains=variant['online_steps_between_trains'],
                    min_amount_to_train=variant['online_min_train_step'],
                    online_exploration_type=variant['online_exploration_type'],
                    trains_per_step=variant['online_trains_per_step'],
                    replay_memory_max_size=variant['online_replay_memory_max_size'],
                    s4_load_model=variant['s4_load_model'],
                    eval_fns=[eval_episodes(tar) for tar in env_targets],
                    game_name=env_name,
                    online_savepostifx=variant['online_postfix'],
                    rtg_variation=variant['online_rtg_variation'],
                    online_soft_update=variant['online_soft_update'],
                    base_target_reward=env_targets[0] * 0.95,
                    fine_tune_critic_steps=variant['fine_tune_critic_steps'],
                    online_step_partial_advance=variant['online_step_partial_advance'],
                    fine_tune_critic_steps_speedup=variant['fine_tune_critic_steps_speedup'],

                    eval_base_func=eval_episodes,
                )
    elif model_type == 'bc':
        trainer = ActTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size,
            get_batch=get_batch,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.mean((a_hat - a) ** 2),
            eval_fns=[eval_episodes(tar) for tar in env_targets],
        )

    if log_to_wandb:
        wandb.init(
            name=exp_prefix,
            group=group_name,
            project='decision-transformer',
            config=variant
        )
        # wandb.watch(model)  # wandb has some bug

    if variant['s4_onpolicy_afteroff'] == 0:
        print(f"Trainer Type: {type(trainer)}")
    else:
        pass
    print(f"model:{model}")
    # print(f"online after offline:{variant['s4_onpolicy_afteroff']}")
    if variant['s4_onpolicy_afteroff'] == 0:
        print(f"training only offline: max epochs:{variant['max_iters']}")
        print(f"max iterations:{variant['max_iters']}, num steps per iter:{variant['num_steps_per_iter']}")

        best_avg_returns = 0
        current_model = model
        for iter in range(variant['max_iters']):
            outputs, current_model, current_avg_returns = trainer.train_iteration(
                num_steps=variant['num_steps_per_iter'], iter_num=iter + 1,
                print_logs=True)

            logging.info('=' * 80)
            logging.info(f'Iteration {iter + 1}')
            for k, v in outputs.items():
                logging.info(f'{k}: {v}')

            if log_to_wandb:
                wandb.log(outputs)

            # save best model
            if current_avg_returns > best_avg_returns:
                best_avg_returns = current_avg_returns
                print(f"best average returns updated: {best_avg_returns}")
                current_model.to(torch.float32)
                torch.save(current_model.state_dict(),
                           f"6h_8z_good_agent_id_rew_embed_global_state_emb_obs_embed_prev_act_embed_act_offline_nplr_emb_64_iter_1000_with_before_good.pkl")
            print(f"best average returns seen so far: {best_avg_returns}")

            logging.info(f'Iteration {iter}: return = {current_avg_returns}: best return so far = {best_avg_returns}')
    else:
        print(f"training online after offline")
        trainer1 = SequenceTrainer(
            model=model,
            optimizer=optimizer,
            batch_size=batch_size, get_batch=get_batch,
            # get_batch=get_batch if not bool(variant['s4_singlestep']) else get_batch_recurrent,
            scheduler=scheduler,
            loss_fn=lambda s_hat, a_hat, r_hat, s, a, r: torch.nn.functional.cross_entropy(a_hat, a),
            critic=[None, None],
            critic_optimizer=None,
            critic_scheduler=None,
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            runlabel=variant['online_postfix'],
            rtg_set_all=bool(variant['s4_ant_rtg_all_base']),
        )
        print("LOGX starting off policy")
        best_avg_returns = 0
        current_model = model
        best_model = model
        for iter in range(variant['max_iters']):
            outputs, current_model, current_avg_returns = trainer1.train_iteration(
                num_steps=variant['num_steps_per_iter'], iter_num=iter + 1,
                print_logs=True)
            if True:
                logging.info('=' * 80)
                logging.info(f'Iteration {iter + 1}')
                for k, v in outputs.items():
                    logging.info(f'{k}: {v}')
            if log_to_wandb:
                wandb.log(outputs)

            # save best model
            if current_avg_returns > best_avg_returns:
                best_avg_returns = current_avg_returns
                print(f"best average returns updated: {best_avg_returns}")
                current_model.to(torch.float32)
                best_model.load_state_dict(current_model.state_dict())
                # torch.save(current_model.state_dict(), f"smac_best_all_dep_no_tanh_no_act_embed_ablated_2.pkl")
                torch.save(current_model.state_dict(),
                           f"6h_8z_good_agent_id_rew_emb_obs_embed_global_state_emb_prev_act_embed_act_offline_before_online.pkl")
            print(f"best average returns seen so far: {best_avg_returns}")

        ##
        # best_avg_returns = 11.581
        new_s4_train = bool(variant['s4_trainable'])
        s4_config.s4_trainable = new_s4_train
        new_model = S4_mujoco_wrapper_v3(
            config=s4_config,
            obs_dim=obs_dim,
            state_dim=state_dim,
            act_dim=act_dim,
            max_length=K,
            l_max=K,
            s4_weight_decay=variant['s4weight_decay'],
            n_embd=variant['embed_dim'],
            d_state=variant['s4internal'],
            H=variant['s4interface'],
            kernel_mode=variant['s4_mode'],
        )

        # file_path = f"6h_8z_good_agent_id_rew_emb_obs_embed_global_state_emb_prev_act_embed_act_offline_before_online.pkl"
        # # # # Load the state dictionary
        # state_dict = torch.load(file_path)

        # Load the state dictionary into the model
        # model.load_state_dict(state_dict)
        new_model = new_model.to(device=device)
        new_model.load_state_dict(best_model.state_dict())
        # new_model.load_state_dict(state_dict)
        # model_target.load_state_dict(model.state_dict())
        model_target = new_model
        model = new_model

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=variant['learning_rate'],
            weight_decay=variant['weight_decay'],
        )
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda steps: min((steps + 1) / warmup_steps, 1)
        )
        ##
        trainer2 = OnlineTrainer(
            env=env.real_env,
            model=model,
            model_target=model_target,
            critic=critic,
            critic_target=critic_target,
            optimizer=optimizer,
            batch_size=batch_size,
            state_mean=[s_mean for s_mean in state_mean],
            state_std=[s_std for s_std in state_std],
            scheduler=scheduler if not bool(variant['flat_lr']) else None,
            lr=variant['online_critic_lr'],
            weight_decay=variant['online_critic_wd'],
            critic_scheduler=critic_scheduler if not bool(variant['flat_lr']) else None,
            critic_optimizer=critic_optimizer,
            episodes_per_iteration=variant['episodes_per_iteration'],
            steps_between_model_swap=variant['online_steps_model_swap'],
            steps_between_trains=variant['online_steps_between_trains'],
            min_amount_to_train=variant['online_min_train_step'],
            online_exploration_type=variant['online_exploration_type'],
            trains_per_step=variant['online_trains_per_step'],
            replay_memory_max_size=variant['online_replay_memory_max_size'],
            s4_load_model=variant['s4_load_model'],
            eval_fns=[eval_episodes(tar) for tar in env_targets],
            game_name=env_name,
            online_savepostifx=variant['online_postfix'],
            rtg_variation=variant['online_rtg_variation'],
            online_soft_update=variant['online_soft_update'],
            base_target_reward=env_targets[0],  # from 0.9*
            fine_tune_critic_steps=variant['fine_tune_critic_steps'],
            online_step_partial_advance=variant['online_step_partial_advance'],
            fine_tune_critic_steps_speedup=variant['fine_tune_critic_steps_speedup'],
            eval_base_func=eval_episodes,
        )
        print("LOGX starting on policy")
        for iter in range(20):
            outputs, current_model, current_avg_returns = trainer2.train_iteration(num_steps=variant['num_steps_per_iter'], iter_num=iter + 1,
                                               print_logs=True)
            if True:
                logging.info('=' * 80)
                logging.info(f'Iteration {iter + 1}')
                for k, v in outputs.items():
                    logging.info(f'{k}: {v}')
            if log_to_wandb:
                wandb.log(outputs)

            # save best model
            if current_avg_returns > best_avg_returns:
                best_avg_returns = current_avg_returns
                print(f"best average returns updated: {best_avg_returns}")
                current_model.to(torch.float32)
                # torch.save(current_model.state_dict(), f"smac_best_all_dep_no_tanh_no_act_embed_ablated_2.pkl")
                torch.save(current_model.state_dict(),
                           f"6h_8z_good_agent_id_rew_emb_obs_embed_global_emb_prev_act_embed_act_online_after_offline.pkl")

            print(f"best average returns seen so far: {best_avg_returns}")

    model.to(torch.float32)
    torch.save(model.state_dict(), f"mujfinal_{variant['online_postfix']}_model_end.pkl")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--env_name', type=str, default='StarCraft2')
    parser.add_argument('--map_name', type=str, default='6h_vs_8z')
    parser.add_argument('--dataset', type=str, default='good')  # good, medium, poor
    parser.add_argument('--mode', type=str, default='normal')  # normal for standard setting, delayed for sparse
    parser.add_argument('--K', type=int, default=50)  # max sequence length for training #50
    parser.add_argument('--pct_traj', type=float, default=1.)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--batch_size', type=int, default=64) #64
    parser.add_argument('--model_type', type=str, default='s4v3')
    parser.add_argument('--embed_dim', type=int, default=96)  # 64
    # parser.add_argument('--n_layer', type=int, default=4) #3, 4
    parser.add_argument('--n_layer', type=int, default=3)
    parser.add_argument('--n_head', type=int, default=1)
    parser.add_argument('--activation_function', type=str, default='relu')
    parser.add_argument('--dropout', type=float, default=0.1)  # 0.1
    # parser.add_argument('--learning_rate', '-lr', type=float, default=1e-5) #1e-4, 1e-5
    parser.add_argument('--learning_rate', '-lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', '-wd', type=float, default=1e-4)  # 1e-4
    parser.add_argument('--warmup_steps', type=int, default=10000)
    parser.add_argument('--num_eval_episodes', type=int, default=30)
    parser.add_argument("--n_eval_rollout_threads", type=int, default=1,
                        help="Number of parallel envs for evaluating rollouts")
    parser.add_argument('--max_iters', type=int, default=15)
    parser.add_argument('--num_steps_per_iter', type=int, default=1000)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--log_to_wandb', '-w', type=bool, default=True)

    # # S4 related
    # parser.add_argument('--s4weight_decay', type=float, default=0.05) #0.0, 0.05
    # parser.add_argument('--s4internal', type=int, default=128)#hidden state dim 96, 128
    # parser.add_argument('--s4interface', type=int, default=96)#input dim 96
    # parser.add_argument('--layer_norm_s4', type=int, default=0)
    # parser.add_argument('--s4_singlestep', type=int, default=0)
    # parser.add_argument('--s4_n_ssm', type=int, default=1)
    # parser.add_argument('--s4_precision', type=int, default=1)
    # parser.add_argument('--s4dropoutval', type=float, default=0.1) #0.0, 0.1
    # parser.add_argument('--s4_layers', type=int, default=4) #3, 4
    # parser.add_argument('--s4_resnet', type=int, default=0)
    # parser.add_argument('--s4_mode', type=str, default='nplr')
    # parser.add_argument('--s4_trainable', type=int, default=1)
    # parser.add_argument('--s4_load_model', type=str, default='none')
    # parser.add_argument('--s4_ablation_model', type=str, default='s4')
    # parser.add_argument('--s4_train_noise', type=float, default=0.0)
    # parser.add_argument('--s4_offpol_actorcritic', type=int, default=0)
    # parser.add_argument('--s4_discrete', type=int, default=0)
    # parser.add_argument('--s4_recurrent_mode_train', type=int, default=0)
    # parser.add_argument('--s4_ant_train_bias', type=float, default=0)
    # parser.add_argument('--s4_ant_multi_lr', type=str, default='none')
    # parser.add_argument('--s4_ant_rtg_all_base', type=int, default=0)

    # S4 related
    parser.add_argument('--s4weight_decay', type=float, default=0.0)  # 0.0, 0.05
    parser.add_argument('--s4internal', type=int, default=96)  # hidden state dim 96, 128
    parser.add_argument('--s4interface', type=int, default=96)  # input dim 96
    parser.add_argument('--layer_norm_s4', type=int, default=0) # 0 mamba does not run without it!!!
    parser.add_argument('--s4_singlestep', type=int, default=0)#1
    parser.add_argument('--s4_n_ssm', type=int, default=1)
    parser.add_argument('--s4_precision', type=int, default=1)
    parser.add_argument('--s4dropoutval', type=float, default=0.0)  # 0.0, 0.1
    parser.add_argument('--s4_layers', type=int, default=3)  # 3, 4
    parser.add_argument('--s4_resnet', type=int, default=0)
    parser.add_argument('--s4_mode', type=str, default='real')# try real!!!
    parser.add_argument('--s4_trainable', type=int, default=1)
    parser.add_argument('--s4_load_model', type=str, default='none')
    parser.add_argument('--s4_ablation_model', type=str, default='s4')
    parser.add_argument('--s4_train_noise', type=float, default=0.0)
    parser.add_argument('--s4_offpol_actorcritic', type=int, default=0)
    parser.add_argument('--s4_discrete', type=int, default=0)
    parser.add_argument('--s4_recurrent_mode_train', type=int, default=0)
    parser.add_argument('--s4_ant_train_bias', type=float, default=0)
    parser.add_argument('--s4_ant_multi_lr', type=str, default='none')
    parser.add_argument('--s4_ant_rtg_all_base', type=int, default=0)

    # online
    parser.add_argument('--s4_onpolicy', type=int, default=0)
    parser.add_argument('--s4_onpolicy_afteroff', type=int, default=1)
    parser.add_argument('--online_postfix', type=str, default='latest_dependent')
    parser.add_argument('--online_steps_model_swap', type=int, default=2000)#2000
    parser.add_argument('--episodes_per_iteration', type=int, default=200) #200
    parser.add_argument('--online_steps_between_trains', type=int, default=10)#10
    parser.add_argument('--online_min_train_step', type=int, default=5000)#5000
    parser.add_argument('--online_exploration_type', type=str, default="0.75,0.3,18000,60000,90000")
    parser.add_argument('--online_trains_per_step', type=int, default=4)
    parser.add_argument('--online_critic_lr', type=float, default=1e-4)
    parser.add_argument('--online_replay_memory_max_size', type=float, default=70000)
    parser.add_argument('--online_critictype', type=str, default="resnet")
    parser.add_argument('--online_rtg_variation', type=float, default=0.15)
    parser.add_argument('--online_soft_update', type=float, default=-1)
    parser.add_argument('--online_critic_wd', type=float, default=1e-3)
    parser.add_argument('--flat_lr', type=int, default=0)
    parser.add_argument('--fine_tune_critic_steps', type=int, default=0) #0, 5000
    parser.add_argument('--fine_tune_critic_steps_speedup', type=int, default=10)
    parser.add_argument('--online_step_partial_advance', type=str, default="none")

    # others:
    parser.add_argument('--overhead_rtg_test', type=int, default=0)
    parser.add_argument('--track_step_err', type=int, default=0)
    parser.add_argument('--partial_traj', type=float, default=1.0)
    parser.add_argument('--cpu_inf', type=int, default=0)

    parser.add_argument("--add_move_state", action='store_true', default=False)
    parser.add_argument("--add_local_obs", action='store_true', default=False)
    parser.add_argument("--add_distance_state", action='store_true', default=False)
    parser.add_argument("--add_enemy_action_state", action='store_true', default=False)
    parser.add_argument("--add_agent_id", action='store_true', default=True)
    parser.add_argument("--add_visible_state", action='store_true', default=False)
    parser.add_argument("--add_xy_state", action='store_true', default=False)
    parser.add_argument("--use_state_agent", action='store_true', default=True)
    parser.add_argument("--use_mustalive", action='store_false', default=True)
    parser.add_argument("--add_center_xy", action='store_true', default=True)
    parser.add_argument("--stacked_frames", type=int, default=1,
                        help="Dimension of hidden layers for actor/critic networks")
    parser.add_argument("--use_stacked_frames", action='store_true',
                        default=False, help="Whether to use stacked_frames")
    parser.add_argument("--use_obs_instead_of_state", action='store_true',
                        default=False, help="Whether to use global state or concatenated obs")

    # to remove
    parser.add_argument('--s4activate', type=str, default="none")

    args = parser.parse_args()

    prefix1 = datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S')
    prefix2 = "onpol" if bool(vars(args)['s4_onpolicy']) else "offpol"
    # fn = "runlog_" + str(vars(args)['env_name']) + "_" + str(vars(args)['map_name']) + "_" + str(
    #     vars(args)['dataset']) + f"_{prefix2}_{prefix1}_" + "dep_all" + str(int(time.time() * 100000) % 100000) + ".log"

    # fn = "runlog_" + str(vars(args)['env_name']) + "_" + str(vars(args)['map_name']) + "_" + str(
    #     vars(args)['dataset']) + f"_{prefix2}_{prefix1}_" + "dep_all_no_tanh_no_act_embed_ablated_2" + ".log"

    fn = "runlog_" + str(vars(args)['env_name']) + "_" + str(vars(args)['map_name']) + "_" + str(
        vars(args)[
            'dataset']) + f"_{prefix2}_{prefix1}_" + "agent_id_rew_embed_obs_embed_global_state_emb_prev_act_embed_act_online_after_offline_nplr_emb_96_iter_1000_good" + ".log"

    # fn = "test.log"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        filename=fn,
    )

    print(f"Exporting log to {fn}")
    logging.info("ARGS ON RUN")
    for k in vars(args).keys():
        logging.info(f"{k:32} :: {vars(args)[k]:35}")
    logging.info("ARGS END")
    set_seed(vars(args)['seed'])
    sys.stdout.flush()
    experiment('gym-experiment', variant=vars(args))
