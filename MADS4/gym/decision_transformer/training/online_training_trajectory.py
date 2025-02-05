import numpy as np
import torch
import time
from decision_transformer.models.s4_muj import *
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import logging
logger = logging.getLogger(__name__)

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class OnlineTrainer_Traj:
    def __init__(self,
                 env,
                 model,
                 model_target,
                 batch_size=100,
                 max_ep_len=1000,
                 scale=1000.,
                 state_mean=0.,
                 state_std=1.,
                 device='cuda',
                 eval_fns = None,
                 optimizer = None,
                 scheduler = None,
                 target_return=None,
                 steps_between_model_swap=2000,
                 steps_between_trains=10,
                 min_amount_to_train=5000,
                 replay_memory_max_size=60000,
                 online_exploration_type="0.75,0.3,18000,60000,90000",
                 s4_load_model="none",
                 trains_per_step=4,
                 mode='normal',
                 game_name="def",
                 online_savepostifx="latest",
                 rtg_variation=0,
                 base_target_reward=3600,
                 online_soft_update=-1,
                 fine_tune_critic_steps=0,
                 online_step_partial_advance = 'none',
                 fine_tune_critic_steps_speedup = 10,
                 episodes_per_iteration=200,
                 eval_base_func=None,
                 ):
        self.env = env
        self.game_name = game_name
        self.base_target_reward = base_target_reward
        self.curr_target_reward = base_target_reward
        self.device = device
        self.state_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.model = model
        self.model.batch_mode = False
        self.model_target = model_target
        self.model_target.eval()
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.reward_scale = scale
        self.state_mean = torch.from_numpy(state_mean).to(device=device)
        self.state_std = torch.from_numpy(state_std).to(device=device)
        self.mode = mode
        self.eval_fns = eval_fns
        self.eval_base_func = eval_base_func
        self.diagnostics = dict()
        self.start_time = time.time()

        self.replay_memory_max_size = replay_memory_max_size
        self.episodes_per_iteration = episodes_per_iteration
        self.min_amount_to_train = min_amount_to_train ### 20000
        self.online_model_export_freq = 100000
        self.gamma = 0.99
        self.online_savepostifx = online_savepostifx
        self.rtg_variation = rtg_variation
        self.tau = online_soft_update
        self.fine_tune_critic_steps = fine_tune_critic_steps
        self.fine_tune_critic_steps_speedup = fine_tune_critic_steps_speedup
        self.train_reward_target = 0

        self.s4_load_model = s4_load_model
        # mujoco_hopp_107500_online_latest_critic.pkl
        self.all_steps = 0 if s4_load_model == "none" else int(self.s4_load_model.split("_")[-3])
        self.replay_memory_max_size = 2000
        self.replay_memory = []

        temp = online_exploration_type.split(",")
        self.start_variation= float(temp[0])
        self.end_variation = float(temp[1])
        self.explore_breakstart = int(temp[2])
        self.explore_breakend = int(temp[3])
        self.explore_breakend_simple = int(temp[4])
        self.trains_per_step = trains_per_step
        torch.autograd.set_detect_anomaly(True)
        return

    def update_module_param(self):
        if self.tau <= 0:
            self.model_target.load_state_dict(self.model.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            if self.all_steps > self.fine_tune_critic_steps:
                for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

    def get_variation(self):
        if self.all_steps < self.explore_breakstart:
            currprob = self.start_variation
        elif self.all_steps < self.explore_breakend:
            currprob = 1.0 * (self.end_variation - self.start_variation) / (self.explore_breakend - self.explore_breakstart) * (
                        self.all_steps - self.explore_breakstart) + self.start_variation
        elif self.all_steps < self.explore_breakend_simple:
            currprob = 1.0 * (0.005 - self.end_variation) / (self.explore_breakend_simple - self.explore_breakend) * (
                        self.all_steps - self.explore_breakend) + self.end_variation + 0.005
        else:
            currprob = 0.005
        return currprob

    def train_iteration(self, num_steps=100, episodes_to_run=200, mode="partial", iter_num=0, print_logs=True ):
        ## run to create dataset:
        print(f"LOG variation: {self.get_variation()}")
        logger.info(f"LOG variation: {self.get_variation()}")
        if iter_num==1:
            episodes_to_run = 1
        else:
            episodes_to_run = self.episodes_per_iteration
        if self.rtg_variation>0:
            run_target_return = self.curr_target_reward * (np.ones((episodes_to_run)) + self.rtg_variation * ( 2 * np.random.random((episodes_to_run)) -1) )
        else:
            run_target_return = self.curr_target_reward * np.ones((episodes_to_run))
        past_diffs = []
        seen_returns, seen_lengths = [], []
        for i in range(episodes_to_run):
            episode_return, episode_length, data, diff = self.run_train_episode(float(run_target_return[i]/self.reward_scale))
            self.replay_memory.append(data)
            if len(self.replay_memory) > self.replay_memory_max_size:
                for temp in range(100):
                    self.replay_memory.pop(0)
            logger.info(f"Iternum,innerepisode {iter_num:4} {i:4} length,diff,rewards :: {episode_length:6d} {diff:.3f} {episode_return:.3f}")
            past_diffs.append(float(diff))
            seen_returns.append(episode_return)
            seen_lengths.append(episode_length)
        self.latest_episode_return_mean = np.array(seen_returns).mean()
        self.latest_episode_length_mean = np.array(seen_lengths).mean()
        self.train_reward_target = max(np.array(seen_returns).mean() + 0.4 * np.array(seen_returns).std(),
                                       self.train_reward_target)
        logger.info(f"Average evaluation return: {self.latest_episode_return_mean}, Length: {self.latest_episode_length_mean}")
        logger.info(f"Target return for iter {iter_num:3} : {self.train_reward_target}")

        logs = dict()
        eval_start = time.time()

        if iter_num == 1:
            train_losses = [0]
        else:
            logs["return_pivot"] = self.train_reward_target
            train_losses = []
            training_dat = self.prepare_replay_data()
            self.model.train()
            train_start = time.time()
            for _ in range(self.min_amount_to_train):
                train_loss = self.train_step(training_dat)
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()

            logs['time/training'] = time.time() - train_start

        self.model.eval()
        self.model.pre_val_setup()

        #f'target_{target_rew}_return_mean': np.mean(returns),
        #f'target_{target_rew}_return_std': np.std(returns),
        #f'target_{target_rew}_length_mean': np.mean(lengths),
        #f'target_{target_rew}_length_std': np.std(lengths),
        #### new_target generate:
        old_target_reward = self.curr_target_reward
        oldvals = self.eval_base_func(self.curr_target_reward)(self.model)
        if iter_num == 1:
            to_add = oldvals[f'target_{self.curr_target_reward}_return_mean']
            self.curr_target_reward = int(1.0 * (self.curr_target_reward + 2*oldvals[f'target_{self.curr_target_reward}_return_mean']) / 3)
        else:
            #self.curr_target_reward = int(1.0 * ( to_add + 2 * self.curr_target_reward + oldvals[f'target_{self.curr_target_reward}_return_mean'] ) /4)
            #self.curr_target_reward = min(self.base_target_reward,
                #1.0 * (2 * self.curr_target_reward + oldvals[f'target_{self.curr_target_reward}_return_mean']) / 3 + 400)
            self.curr_target_reward = max( 3900, min(self.base_target_reward, #5300
                1.0 * (oldvals[f'target_{self.curr_target_reward}_return_mean']) + 400))
        #### new_target generate:
        newvals = self.eval_base_func(self.curr_target_reward)(self.model)

        logging.info('=' * 80)
        logging.info(f'Iteration {iter_num + 1} pre Validation calibration')
        logging.info(f'Average diff over iteration: {np.array(past_diffs).mean()}')
        logging.info(f'Pre calibration target:  {old_target_reward}')
        for k, v in oldvals.items():
            logging.info(f'{k}: {v}')
        logging.info('~' * 50)
        logging.info(f'Post calibration target: {self.curr_target_reward}')
        for k, v in newvals.items():
            logging.info(f'{k}: {v}')
        logging.info('=' * 80)

        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num+1}')
            for k, v in logs.items():
                print(f'{k}: {v}')
        return logs

    def run_train_episode(self, target_return):
        s4_states = [r.detach() for r in self.model.get_initial_state((1), self.device)]
        state = self.env.reset()
        if self.mode == 'noise':
            state = state + np.random.normal(0, 0.1, size=self.state.shape)
        # return observations_stack

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"
        states = torch.from_numpy(state).reshape(1, self.state_dim).to(device=self.device, dtype=torch.float32)
        actions = torch.zeros((1, self.act_dim), device=self.device, dtype=torch.float32)
        rewards = torch.zeros(1, device=self.device, dtype=torch.float32)

        ep_return = target_return
        target_return = torch.tensor(ep_return, device=self.device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        sim_states = []

        episode_return, episode_length = 0, 0
        actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])
        for t in range(self.max_ep_len):

            # add padding
            action, s4_states = self.model.get_action(
                (states.to(dtype=torch.float32) - self.state_mean) / self.state_std,
                actions.to(dtype=torch.float32),
                rewards.to(dtype=torch.float32),
                target_return.to(dtype=torch.float32),
                timesteps.to(dtype=torch.long),
                s4_states=s4_states
            )
            if t > 0:
                actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)
                rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])
            ### Add proc to decide the upcoming action according to training method
            if self.all_steps > self.fine_tune_critic_steps:
                action = action + self.get_variation() * torch.randn(action.shape, dtype=action.dtype, device=action.device)
                action = torch.minimum(torch.maximum(action, -torch.ones_like(action)), torch.ones_like(action))
            actions[-1] = action
            action = action.detach().cpu().numpy()

            state, reward, done, _ = self.env.step(action)

            cur_state = torch.from_numpy(state).to(device=self.device).reshape(1, self.state_dim)
            states = torch.cat([states, cur_state], dim=0)
            rewards[-1] = reward
            pred_return = target_return[0, -1] - (reward / self.reward_scale)
            target_return = torch.cat(
                [target_return, pred_return.reshape(1, 1)], dim=1)
            timesteps = torch.cat(
                [timesteps,
                 torch.ones((1, 1), device=self.device, dtype=torch.long) * (t + 1)], dim=1)

            episode_return += reward
            episode_length += 1
            self.all_steps += 1

            if done:
                break
        dones = torch.zeros_like(rewards, device=self.device, dtype=torch.int)
        dones[-1] = 1
        data = {'observations': np.array(states.detach().cpu()), 'rewards': np.array(rewards.detach().cpu()),
                'actions': np.array(actions.detach().cpu()), 'dones': np.array(dones.detach().cpu())}
        diff = float(ep_return-episode_return/self.reward_scale)
        return episode_return, episode_length, data, diff

    def loss_fn(self, a_hat, a, totreward):
        weighted_lens = torch.maximum(torch.minimum(torch.exp_(totreward / self.train_reward_target-1), torch.ones_like(totreward)*10), torch.ones_like(totreward)*0.01)
        return torch.mean(torch.mean((a_hat - a) ** 2, dim=1)*weighted_lens)

    def train_step(self, train_data):
        states, actions, rewards, dones, rtg, timesteps, attention_mask = self.get_batch(train_data, self.batch_size)
        action_target = torch.clone(actions)
        #print(f"LOGXXX states {states.shape}")
        #print(f"LOGXXX actions {actions.shape}")
        #print(f"LOGXXX rewards {rewards.shape}")
        #print(f"LOGXXX rtg {rtg.shape}")
        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        attention_mask = attention_mask[..., 1:]
        action_target = action_target[:, 1:, :]
        action_preds = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        totrewards = torch.sum(rewards, dim=1)

        loss = self.loss_fn(action_preds, action_target, totrewards)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
        self.optimizer.step()

        with torch.no_grad():
            self.diagnostics['training/action_error'] = torch.mean(
                (action_preds - action_target) ** 2).detach().cpu().item()

        return loss.detach().cpu().item()

    def prepare_replay_data(self):
        # start train_step:
        states, traj_lens, returns = [], [], []
        for path in self.replay_memory:
            if self.mode == 'delayed':  # delayed: all rewards moved to end of trajectory
                path['rewards'][-1] = path['rewards'].sum()
                path['rewards'][:-1] = 0.
            states.append(path['observations'])
            traj_lens.append(len(path['observations']))
            returns.append(path['rewards'].sum())
        traj_lens, returns = np.array(traj_lens), np.array(returns)

        # used for input normalization
        states = np.concatenate(states, axis=0)
        #state_mean, state_std = np.mean(states, axis=0), np.std(states, axis=0) + 1e-6
        num_timesteps = sum(traj_lens)

        print('=' * 50)
        print(f'Starting training')
        print(f'{len(traj_lens)} trajectories, {num_timesteps} timesteps found')
        print(f'Average return: {np.mean(returns):.2f}, std: {np.std(returns):.2f}')
        print(f'Max return: {np.max(returns):.2f}, min: {np.min(returns):.2f}')
        print('=' * 50)

        pct_traj = 1

        # only train on top pct_traj trajectories (for %BC experiment)
        num_timesteps = max(int(pct_traj * num_timesteps), 1)
        sorted_inds = np.argsort(returns)  # lowest to highest
        num_trajectories = 1
        timesteps = traj_lens[sorted_inds[-1]]
        ind = len(self.replay_memory) - 2
        while ind >= 0 and timesteps + traj_lens[sorted_inds[ind]] < num_timesteps:
            timesteps += traj_lens[sorted_inds[ind]]
            num_trajectories += 1
            ind -= 1
        sorted_inds = sorted_inds[-num_trajectories:]
        return [num_trajectories, traj_lens, sorted_inds]


    def get_batch(self, data, batch_size=256):
        trajectories = self.replay_memory
        num_trajectories, traj_lens, sorted_inds = data
        p_sample = traj_lens[sorted_inds]**2 / sum(traj_lens[sorted_inds]**2)
        ####
        batch_inds = np.random.choice(
            np.arange(num_trajectories),
            size=batch_size,
            replace=True,
            p=p_sample,  # reweights so we sample according to timesteps
        )
        #print(f"log sorted ind: {sorted_inds}")
        #print(f"log traj: {len(trajectories)}")
        #print(f"log sorted ind: {trajectories[sorted_inds[-1]]['rewards'].sum()}")
        #print(f"log sorted ind: {trajectories[sorted_inds[0]]['rewards'].sum()}")

        #if self.training_mode == "epsilon":
        #    batch_inds = [sorted_inds[-1], sorted_inds[-2]] * int(batch_size/2)
        max_len = max([traj_lens[sorted_inds[batch_inds[i]]] for i in range(batch_size)])
        print(f"LOG max len for batch: {max_len}")

        s, a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], []
        for i in range(batch_size):
            traj = trajectories[int(sorted_inds[batch_inds[i]])]
            #si = random.randint(0, traj['rewards'].shape[0] - 1)
            si = 0

            # get sequences from dataset
            s.append(traj['observations'][si:si + max_len].reshape(1, -1, self.state_dim))
            a.append(traj['actions'][si:si + max_len].reshape(1, -1, self.act_dim))
            r.append(traj['rewards'][si:si + max_len].reshape(1, -1, 1))
            if 'terminals' in traj:
                d.append(traj['terminals'][si:si + max_len].reshape(1, -1))
            else:
                d.append(traj['dones'][si:si + max_len].reshape(1, -1))
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.max_ep_len] = self.max_ep_len - 1  # padding cutoff
            rtg.append(discount_cumsum(traj['rewards'][si:], gamma=1.)[:s[-1].shape[1] + 1].reshape(1, -1, 1))
            if rtg[-1].shape[1] <= s[-1].shape[1]:
                rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, 1, 1))], axis=1)

            # print(f"LOG b4 pad s: {s[-1].shape}")
            # print(f"LOG b4 pad a: {a[-1].shape}")
            # print(f"LOG b4 pad r: {r[-1].shape}")
            # print(f"LOG b4 pad d: {d[-1].shape}")
            # print(f"LOG b4 pad rtg: {rtg[-1].shape}")

            # padding and state + reward normalization
            tlen = s[-1].shape[1]
            s[-1] = np.concatenate([s[-1], np.zeros((1, max_len - tlen, self.state_dim))], axis=1)
            s[-1] = (s[-1] - np.array(self.state_mean.detach().cpu())) / np.array(self.state_std.detach().cpu())
            a[-1] = np.concatenate([a[-1], np.zeros((1, max_len - tlen, self.act_dim))], axis=1)
            r[-1] = np.concatenate([r[-1], np.zeros((1, max_len - tlen, 1))], axis=1)
            d[-1] = np.concatenate([d[-1], np.ones((1, max_len - tlen)) * 2], axis=1)
            rtg[-1] = np.concatenate([rtg[-1], np.zeros((1, max_len - tlen, 1))], axis=1) / self.reward_scale
            timesteps[-1] = np.concatenate([timesteps[-1], np.zeros((1, max_len - tlen))], axis=1)
            mask.append(np.concatenate([np.ones((1, tlen)), np.zeros((1, max_len - tlen))], axis=1))

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=self.device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=self.device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=self.device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=self.device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=self.device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=self.device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=self.device)

        return s, a, r, d, rtg, timesteps, mask