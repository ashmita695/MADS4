import random
import numpy as np
import torch
import glob
import os
from torch.nn import functional as F
from gym.spaces.discrete import Discrete

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import yaml

def to_device(*params):
    return [x.to(device) for x in params]

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def list2array(input_list):
    return np.array(input_list)


FLOAT = torch.FloatTensor

# from [g, o, a, r, d, ava] to [g, o, a, rtgs, d, ava]
# def get_episode(index, bias, data_dir=None, min_return=0):
#     index += bias
#     path_files = glob.glob(os.path.join(data_dir, "*"))
#     episode = torch.load(path_files[index])
#
#     for agent_trajectory in episode:
#         rtgs = 0
#         for i in reversed(range(len(agent_trajectory))):
#             rtgs += agent_trajectory[i][3][0]
#             agent_trajectory[i][3][0] = rtgs # rewards to go instead of rewards
#
#     return episode

# from [g, o, a, r, d, ava] to [g, o, a, r, d, ava, rtgs]
def get_episode(index, bias, data_dir=None, min_return=0):
    index += bias
    path_files = glob.glob(os.path.join(data_dir, "*"))
    episode = torch.load(path_files[index])

    for agent_trajectory in episode:
        rtgs = 0
        returns_to_go = [0] * len(agent_trajectory)

        for i in reversed(range(len(agent_trajectory))):
            rtgs += agent_trajectory[i][3][0]
            returns_to_go[i] = rtgs

        # Append the returns to go as a separate entry in each timestep
        for i in range(len(agent_trajectory)):
            agent_trajectory[i].append([returns_to_go[i]])

    return episode


def create_dataset(episode_num, bias, data_dir=None, min_return=0):
    global_states = []
    local_obss = []
    actions = []
    rtgs = []
    done_idxs = []
    time_steps = []

    for episode_idx in range(episode_num):
        episode = get_episode(episode_idx, bias, data_dir, min_return)
        for agent_trajectory in episode:
            time_step = 0
            rtgs_trajectory = np.zeros(len(agent_trajectory))
            for step in agent_trajectory:
                g, o, a, r, d, ava = step
                if type(g) is np.ndarray:
                    g = g.tolist()
                if type(o) is np.ndarray:
                    o = o.tolist()
                if type(a) is np.ndarray:
                    a = a.tolist()
                if type(r) is np.ndarray:
                    r = r[0]
                if type(ava) is np.ndarray:
                    ava = ava.tolist()
                global_states.append(g)
                local_obss.append(o)
                actions.append(a)
                rtgs_trajectory[0:time_step + 1] += r
                time_steps.append(time_step)
                time_step += 1
            rtgs.extend(list(rtgs_trajectory))
            done_idxs.append(len(global_states))

    states = np.concatenate((global_states, local_obss), axis=1)

    return states, actions, done_idxs, rtgs, time_steps

def load_data(episode_num, bias, data_dir=None, min_return=0, n_agents=0):
    global_states = [[] for i in range(n_agents)]
    local_obss = [[] for i in range(n_agents)]
    actions = [[] for i in range(n_agents)]
    rewards = [[] for i in range(n_agents)]
    returns = [[] for i in range(n_agents)]
    done_idxs = [[] for i in range(n_agents)]
    time_steps = [[] for i in range(n_agents)]
    next_global_states = [[] for i in range(n_agents)]
    next_local_obss = [[] for i in range(n_agents)]
    next_available_actions = [[] for i in range(n_agents)]

    mark_rewards = []

    data = []
    for episode_idx in range(episode_num):
        episode = get_episode(episode_idx, bias, data_dir, min_return)
        # print(len(episode))#6 trajectories for 6 agents
        if not isinstance(data_dir, str) and episode is None:
            continue
        length = len(episode[0])# not all episodes have same number of time steps
        # print(length)
        # print(type(episode))#list
        flag = True
        episode_dict = {}
        episode_dict['observations'] = [[] for i in range(n_agents)]
        episode_dict['global_states'] = [[] for i in range(n_agents)]
        episode_dict['next_observations'] = [[] for i in range(n_agents)]
        episode_dict['next_global_states'] = [[] for i in range(n_agents)]
        episode_dict['actions'] = [[] for i in range(n_agents)]
        episode_dict['rewards'] = [[] for i in range(n_agents)]
        episode_dict['terminals'] = [[] for i in range(n_agents)]
        episode_dict['returns'] = [[] for i in range(n_agents)]
        episode_dict['next_available_actions'] = [[] for i in range(n_agents)]

        for agent_trajectory in episode:
            if not len(agent_trajectory) == length:
                flag = False
        if flag:

            for j, agent_trajectory in enumerate(episode):#each agent trajectory
                # print(j, type(agent_trajectory))

                time_step = 0

                mark_reward = []
                # print(len(agent_trajectory))#length of the episode
                for i in range(len(agent_trajectory)):
                    # print(len(agent_trajectory[i]))#6 elements stored for each agent at each time step
                    g, o, a, r, d, ava, rtg = agent_trajectory[i]


                    if i < len(agent_trajectory) - 1:
                        g_next = agent_trajectory[i + 1][0]
                        o_next = agent_trajectory[i + 1][1]
                        ava_next = agent_trajectory[i + 1][5]
                    else:
                        g_next = g
                        o_next = o
                        ava_next = ava

                    global_states[j].append(g)
                    local_obss[j].append(o)
                    actions[j].append(a)
                    rewards[j].append(r[0])# rewards
                    returns[j].append(rtg[0]) #returns to go
                    time_steps[j].append(time_step)
                    time_step += 1
                    next_global_states[j].append(g_next)
                    next_local_obss[j].append(o_next)
                    next_available_actions[j].append(ava_next)

                    episode_dict['observations'][j].append(o)
                    episode_dict['actions'][j].append(a)
                    episode_dict['global_states'][j].append(g)
                    episode_dict['rewards'][j].append(r[0])
                    episode_dict['returns'][j].append(rtg[0])
                    episode_dict['next_global_states'][j].append(g_next)
                    episode_dict['next_observations'][j].append(o_next)
                    episode_dict['next_available_actions'][j].append(ava_next)
                    episode_dict['terminals'][j].append(0.0)

                    # mark_reward.append(r[0])
                episode_dict['observations'][j]=list2array(episode_dict['observations'][j])
                episode_dict['actions'][j] = list2array(episode_dict['actions'][j])
                episode_dict['global_states'][j] = list2array(episode_dict['global_states'][j])
                episode_dict['rewards'][j] = list2array(episode_dict['rewards'][j])
                episode_dict['returns'][j] = list2array(episode_dict['returns'][j])
                episode_dict['next_global_states'][j] = list2array(episode_dict['next_global_states'][j])
                episode_dict['next_observations'][j] = list2array(episode_dict['next_observations'][j])
                episode_dict['next_available_actions'][j] = list2array(episode_dict['next_available_actions'][j])
                episode_dict['terminals'][j] = list2array(episode_dict['terminals'][j])

                # done_idxs[j].append(len(global_states[j]))

                # if j == 0:
                #     mark_rewards.append(np.mean(mark_reward))

            data.append(episode_dict)
        else:
            print("invalid data\n")
    # print(len(global_states[0]))#list of global states for 6 agents: each list consists of steps of 50 episodes: total 1871 steps
    # actions = list2array(actions).swapaxes(1, 0).tolist()
    # done_idxs = list2array(done_idxs).swapaxes(1, 0).tolist()
    # rewards = list2array(rewards).swapaxes(1, 0).tolist()
    # time_steps = list2array(time_steps).swapaxes(1, 0).tolist()
    # next_available_actions = list2array(next_available_actions).swapaxes(1, 0).tolist()
    # global_states = list2array(global_states).swapaxes(1, 0).tolist()
    # local_obss = list2array(local_obss).swapaxes(1, 0).tolist()
    # next_global_states = list2array(next_global_states).swapaxes(1, 0).tolist()
    # next_local_obss = list2array(next_local_obss).swapaxes(1, 0).tolist()


    # [s, o, a, d, r, t, s_next, o_next, ava_next]
    # return global_states, local_obss, actions, done_idxs, rewards, time_steps, next_global_states, next_local_obss, \
    #        next_available_actions, mark_rewards
    return data

def get_dim_from_space(space):
    if isinstance(space[0], Discrete):
        return space[0].n
    elif isinstance(space[0], list):
        return space[0][0]


