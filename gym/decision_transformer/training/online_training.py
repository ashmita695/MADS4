import numpy as np
import torch
import time
from decision_transformer.models.s4_muj import *
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Dataset
import logging
from gym.spaces.discrete import Discrete
from tqdm import tqdm

logger = logging.getLogger(__name__)

def one_hot(number, dimension):
    one_hot_vector = np.zeros(dimension)
    one_hot_vector[number] = 1
    return np.array(one_hot_vector)

def get_dim_from_space(space):
    if isinstance(space[0], Discrete):
        return space[0].n
    elif isinstance(space[0], list):
        return space[0][0]

def discount_cumsum(x, gamma):
    discount_cumsum = np.zeros_like(x)
    discount_cumsum[-1] = x[-1]
    for t in reversed(range(x.shape[0]-1)):
        discount_cumsum[t] = x[t] + gamma * discount_cumsum[t+1]
    return discount_cumsum

class OnlineTrainer:
    def __init__(self,
                 env,
                 model,
                 model_target,
                 critic,
                 critic_target,
                 batch_size=100,
                 max_ep_len=1000,
                 scale=1.,
                 state_mean=0.,
                 state_std=1.,
                 device='cuda',
                 eval_fns = None,
                 optimizer = None,
                 scheduler = None,
                 lr = None,
                 weight_decay = None,
                 critic_scheduler = None,
                 critic_optimizer = None,
                 target_return=None,
                 steps_between_model_swap=2000,
                 steps_between_trains=10,
                 min_amount_to_train=5000, #5000
                 replay_memory_max_size=60000,
                 online_exploration_type="0.75,0.3,18000,60000,90000",
                 s4_load_model="none",
                 trains_per_step=4,
                 mode='normal',
                 game_name="def",
                 online_savepostifx="latest",
                 rtg_variation=0,
                 base_target_reward=20.0,
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

        self.global_obs_dim = get_dim_from_space(env.share_observation_space)
        self.local_obs_dim = get_dim_from_space(env.observation_space)
        self.action_dim = get_dim_from_space(env.action_space)

        self.act_dim = self.action_dim
        self.state_dim = self.local_obs_dim

        self.model = model
        self.model_target = model_target
        self.model.config.recurrent_mode = False
        self.model_target.config.recurrent_mode = False
        self.model_target.eval()
        #self.critic = critic.to(device=device)
        #self.critic_target = critic_target.to(device=device)
        self.critic = critic
        self.critic_target = critic_target
        self.critic_rtg_type = "rtg" in str(type(self.critic_target))
        # logger.info(f"Critic type: {str(type(self.critic_target))}")
        # logger.info(f"Critic rtg activation: {self.critic_rtg_type}")
        self.critic_target.eval()
        self.critic_lr = lr
        self.critic_wd = weight_decay
        self.v_param = list(self.critic.v.parameters())
        self.q_param = list(self.critic.q.parameters()) + list(self.critic.mix.parameters())

        self.optimizer = optimizer
        self.scheduler = scheduler
        self.critic_scheduler = critic_scheduler
        self.critic_optimizer = critic_optimizer
        self.v_optimizer = torch.optim.AdamW(self.v_param, lr=self.critic_lr, weight_decay=self.critic_wd)
        self.q_optimizer = torch.optim.AdamW(self.q_param, lr=self.critic_lr, weight_decay=self.critic_wd)
        # to add v and q scheduler
        self.batch_size = batch_size
        self.max_ep_len = max_ep_len
        self.reward_scale = scale
        self.state_mean = [torch.from_numpy(s_).to(device=device) for s_ in state_mean]
        self.state_std = [torch.from_numpy(std_).to(device=device) for std_ in state_std]
        self.mode = mode
        self.eval_fns = eval_fns
        self.eval_base_func = eval_base_func
        self.diagnostics = dict()
        self.start_time = time.time()

        self.replay_memory_max_size = replay_memory_max_size
        self.episodes_per_iteration = episodes_per_iteration
        self.steps_between_model_swap = steps_between_model_swap
        self.steps_between_trains = steps_between_trains
        self.min_amount_to_train = min_amount_to_train ### 20000
        self.online_model_export_freq = 100000
        self.gamma = 0.99
        self.online_savepostifx = online_savepostifx
        self.rtg_variation = rtg_variation
        self.tau = online_soft_update
        self.fine_tune_critic_steps = fine_tune_critic_steps
        self.fine_tune_critic_steps_speedup = fine_tune_critic_steps_speedup
        self.online_step_partial_advance = online_step_partial_advance
        self.step_partial_dat = [1, 1, 1, 1]
        if self.online_step_partial_advance != "none":
            self.step_partial_dat = [int(x) for x in self.online_step_partial_advance.split("_")]

        self.s4_load_model = s4_load_model
        # mujoco_hopp_107500_online_latest_critic.pkl
        self.all_steps = 0 if s4_load_model == "none" else int(self.s4_load_model.split("_")[-3])
        self.train_dataset = StateActionReturnDataset_Online_Qlearn(self.replay_memory_max_size, "smac",
                                                                    123, self.device)

        temp = online_exploration_type.split(",")
        self.start_variation= float(temp[0])
        self.end_variation = float(temp[1])
        self.explore_breakstart = int(temp[2])
        self.explore_breakend = int(temp[3])
        self.explore_breakend_simple = int(temp[4])
        self.trains_per_step = trains_per_step
        torch.autograd.set_detect_anomaly(True)

        if torch.cuda.is_available():
            self.device = torch.cuda.current_device()
            # self.model.batch_mode = True
            #self.model = torch.nn.DataParallel(self.model).to(self.device)
            if hasattr(self.model, "module"):
                self.raw_model = self.model.module
                self.raw_model_target = self.raw_model_target
            else:
                self.raw_model = self.model
                self.raw_model_target = self.raw_model
        self.alpha = 10.0
        self.grad_norm_clip = 1.00

        return

    def update_module_param(self):
        if self.tau <= 0:
            self.model_target.load_state_dict(self.model.state_dict())
            self.critic_target.load_state_dict(self.critic.state_dict())
        else:
            if self.all_steps > self.fine_tune_critic_steps:
                for target_param, param in zip(self.model_target.parameters(), self.raw_model.parameters()): # was_model
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
        if iter_num == 1:
            episodes_to_run = 1
        else:
            episodes_to_run = self.episodes_per_iteration
        if self.rtg_variation > 0:
            run_target_return = self.curr_target_reward * (np.ones((episodes_to_run)) + self.rtg_variation * ( 2 * np.random.random((episodes_to_run)) -1) )
        else:
            run_target_return = self.curr_target_reward * np.ones((episodes_to_run))
        past_diffs = []
        for i in range(episodes_to_run):
            episode_return, episode_length, diff = self.run_train_episode(float(run_target_return[i]/self.reward_scale))
            logger.info(f"Iternum,innerepisode {iter_num:4} {i:4} length,diff,rewards :: {episode_length:6d} {diff[0, 0].item():.3f} {episode_return:.3f}")
            print(f"Average evaluation return: {episode_return}, Length: {episode_length}")
            past_diffs.append(float(diff[0, 0].item()))

        logs = dict()
        eval_start = time.time()

        self.model.eval()
        self.raw_model.pre_val_setup() #was_model

        #f'target_{target_rew}_return_mean': np.mean(returns),
        #f'target_{target_rew}_return_std': np.std(returns),
        #f'target_{target_rew}_length_mean': np.mean(lengths),
        #f'target_{target_rew}_length_std': np.std(lengths),
        #### new_target generate:
        old_target_reward = self.curr_target_reward
        oldvals = self.eval_base_func(self.curr_target_reward)(self.raw_model)  #was_model
        if iter_num == 1:
            to_add = oldvals[f'target_{self.curr_target_reward}_return_mean']
            self.curr_target_reward = int(1.0 * (self.curr_target_reward + 2*oldvals[f'target_{self.curr_target_reward}_return_mean']) / 3)
        else:
            #self.curr_target_reward = int(1.0 * ( to_add + 2 * self.curr_target_reward + oldvals[f'target_{self.curr_target_reward}_return_mean'] ) /4)
            #self.curr_target_reward = min(self.base_target_reward,
                #1.0 * (2 * self.curr_target_reward + oldvals[f'target_{self.curr_target_reward}_return_mean']) / 3 + 400)
            self.curr_target_reward = max(15, min(self.base_target_reward, #5300
                1.0 * (oldvals[f'target_{self.curr_target_reward}_return_mean']) + 5))
        #### new_target generate:
        newvals = self.eval_base_func(self.curr_target_reward)(self.raw_model)  #was_model

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

        avg_returns = 0
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.raw_model)  #was_model
            current_avg_returns = outputs[f'current avg returns']
            if current_avg_returns > avg_returns:
                avg_returns = current_avg_returns
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num+1}')
            for k, v in logs.items():
                print(f'{k}: {v}')
        return logs, self.model, avg_returns

    def run_train_episode(self, target_return):
        self.model.eval()
        self.raw_model.pre_val_setup() #was_model
        s4_states = [r.detach() for r in self.raw_model.get_initial_state((1), self.device)] # was_model
        self.model_target.pre_val_setup()

        T_rewards, T_wins, steps, episode_dones = 0., 0., 0, 0
        obs, share_obs, available_actions = self.env.reset()
        self.num_agents = obs.shape[1]
        s4_states = [s4_states] * self.num_agents

        if self.mode == 'noise':
            obs = obs + np.random.normal(0, 0.1, size=self.state.shape)
        #return observations_stack

        # we keep all the histories on the device
        # note that the latest action and reward will be "padding"

        obs_ = [torch.from_numpy(obs[:, i:i + 1, :]).to(device=self.device) for i in range(obs.shape[1])]
        share_obs_ = [torch.from_numpy(share_obs[:, i:i + 1, :]).to(device=self.device) for i in range(share_obs.shape[1])]
        for i in range(len(obs_)):
            obs_[i] = (obs_[i] - self.state_mean[i]) / self.state_std[i]
        obs = obs_
        states = share_obs_
        actions = torch.zeros((0, self.act_dim), device=self.device, dtype=torch.float32)  # for each agent
        rewards = torch.zeros(0, device=self.device, dtype=torch.float32)

        ep_return = target_return
        target_return = torch.tensor(ep_return, device=self.device, dtype=torch.float32).reshape(1, 1)
        timesteps = torch.tensor(0, device=self.device, dtype=torch.long).reshape(1, 1)

        sim_states = []

        episode_return, episode_length = 0, 0
        actions = torch.cat([actions, torch.zeros((1, self.act_dim), device=self.device)], dim=0)  # for each agent
        actions = [actions.repeat(1, 1, 1) for _ in range(self.num_agents)]  # for all agents
        rewards = torch.cat([rewards, torch.zeros(1, device=self.device)])
        rewards = [rewards.repeat(1, 1, 1) for _ in range(self.num_agents)]  # for all agents
        timesteps = [timesteps for _ in range(self.num_agents)]
        target_return = [target_return.repeat(1, 1, 1) for _ in range(self.num_agents)]

        t = 0
        # for t in range(max_ep_len):
        reward_mean = 0
        # first_nan = False
        while True:
            # add padding
            if t > 0:
                actions = [torch.cat([action, torch.zeros((1, 1, self.act_dim), device=self.device)], dim=1) for action in actions]
                rewards = [torch.cat([reward, torch.zeros((1, 1, 1), device=self.device)], dim=1) for reward in rewards]

            action_logits, next_s4_states = self.raw_model.get_action(obs, states, actions, rewards,
                target_return,
                timesteps,
                s4_states=s4_states)

            action = [None] * self.num_agents
            for agent in range(self.num_agents):
                if available_actions[0, agent, :] is not None:
                    action_logits[agent][available_actions[0, agent, :] == 0] = -1e10
                probs = torch.nn.functional.softmax(action_logits[agent], dim=-1)
                action[agent] = torch.multinomial(probs, num_samples=1)

            ### Add proc to decide the upcoming action according to training method
            # if self.all_steps > self.fine_tune_critic_steps:
            #     action = action + self.get_variation() * torch.randn(action.shape, dtype=action.dtype, device=action.device)
            # action = torch.minimum(torch.maximum(action, -torch.ones_like(action)), torch.ones_like(action))
            # actions[-1] = action
            # action = action.detach().cpu().numpy

            for i in range(self.num_agents):
                actions[i][:, -1, :] = torch.from_numpy(one_hot(action[i], self.act_dim)).to(self.device)

            action = np.array([a.cpu().numpy()[0] for a in action])  # actions as numpy arrays to pass to env.step

            # state, reward, done, _ = self.env.step(action)
            obs_, share_obs, rewards_, dones, infos, available_actions = self.env.step([action])
            t += 1

            reward_mean += np.mean(rewards_)
            cur_obs = [torch.from_numpy(obs_[:, i:i + 1, :]).to(device=self.device) for i in range(obs_.shape[1])]
            cur_states = [torch.from_numpy(share_obs[:, i:i + 1, :]).to(device=self.device) for i in
                          range(share_obs.shape[1])]
            for i in range(len(states)):
                cur_obs[i] = (cur_obs[i] - self.state_mean[i]) / self.state_std[i]
            obs = [torch.cat([obs[i], cur_obs[i]], dim=1) for i in range(self.num_agents)]
            states = [torch.cat([states[i], cur_states[i]], dim=1) for i in range(self.num_agents)]
            rewards_ = [torch.from_numpy(rewards_[:, i:i + 1, :]).to(device=self.device) for i in range(rewards_.shape[1])]
            for i in range(self.num_agents):
                rewards[i][:, -1, :] = rewards_[i][0][0]
            # cur_state = torch.from_numpy(state).to(device=self.device).reshape(1, self.state_dim)
            # states = torch.cat([states, cur_state], dim=0)
            # rewards[-1] = reward
            for i in range(self.num_agents):
                pred_return = target_return[i][0,-1] - rewards_[i][0,-1]/self.reward_scale
                target_return[i] = torch.cat(
                    [target_return[i], pred_return.reshape(1, 1, 1)], dim=1)

                timesteps[i] = torch.cat(
                    [timesteps[i],
                     torch.ones((1, 1), device=self.device, dtype=torch.long) * (t)], dim=1)

            # pred_return = target_return[0, -1] - (reward/self.reward_scale)
            # target_return = torch.cat(
            #     [target_return, pred_return.reshape(1, 1)], dim=1)
            # timesteps = torch.cat(
            #     [timesteps,
            #      torch.ones((1, 1), device=self.device, dtype=torch.long) * (t + 1)], dim=1)

            # diff = self.train_dataset.add_obs_shift(states[-2, ...].cpu(), actions[-2, ...].cpu(), target_return[0, -2].cpu(),
            #                                         [x.cpu() for x in s4_states],
            #                                         states[-1, ...].cpu(), actions[-1, ...].cpu(), target_return[0, -1].cpu(),
            #                                         [x.cpu() for x in next_s4_states], int(done))

            diff = self.train_dataset.add_obs_shift(torch.stack(obs, dim=1)[:,:, -2, :].cpu(),
                                                    torch.stack(states, dim=1)[:,:, -2, :].cpu(),
                                                    torch.stack(target_return, dim=1)[:,:, -2, :].cpu(),
                                                    [[tensor.cpu() for tensor in s4_state_agent] for s4_state_agent in s4_states],
                                                    torch.stack(actions, dim=1)[:,:, -1, :].cpu(),
                                                    torch.stack(obs, dim=1)[:,:, -1, :].cpu(),
                                                    torch.stack(states, dim=1)[:,:, -1, :].cpu(),
                                                    torch.stack(target_return, dim=1)[:,:, -1, :].cpu(),
                                                    [[tensor.cpu() for tensor in s4_state_agent] for s4_state_agent in next_s4_states],
                                                    dones)

            episode_length += 1
            steps += 1
            # episode_return += reward

            s4_states = next_s4_states

            self.all_steps += 1
            if self.all_steps % 1000 == 0:
                print(f"Steps {self.all_steps:8d} :: Var {self.get_variation():.3f}")
                logger.info(f"Steps {self.all_steps:8d} :: Var {self.get_variation():.3f}")
            if self.all_steps % self.steps_between_trains == 0 and self.all_steps >= self.min_amount_to_train and len(self.train_dataset) > self.batch_size: #*100
                loss1, loss2, loss3 = 0, 0, 0
                self.model.train()
                self.model_target.eval()
                self.critic_target.eval()
                self.critic.train()
                to_run_steps = self.trains_per_step
                if self.all_steps <= self.fine_tune_critic_steps:
                    to_run_steps = to_run_steps * self.fine_tune_critic_steps_speedup
                for jj in range(to_run_steps):
                    print(f"online training begins")
                    logger.info("***** Online Training Begin ******")
                    train_loss1, train_loss2, train_loss3 = self.train_step()
                    loss1 += train_loss1
                    loss2 += train_loss2
                    loss3 += train_loss3
                    #logger.info(f"PASSED {jj}")
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.critic_scheduler is not None:
                    self.critic_scheduler.step()
                print(f"training log: {self.all_steps:10} losses: {loss1/to_run_steps:4f} {loss2/to_run_steps:4f} {loss3/to_run_steps:4f}")
                logger.info(f"training log: {self.all_steps:10} losses: {loss1/to_run_steps:4f} {loss2/to_run_steps:4f} {loss3/to_run_steps:4f}")
                self.model.eval()
                self.critic.q.eval()
                self.critic.v.eval()


            if self.all_steps % self.steps_between_model_swap == 0 and self.all_steps >= self.min_amount_to_train:
                print(f"updated Tar_model {self.all_steps:10}")
                #logger.info(f"updated Tar_model {self.all_steps:10}")
                # self.target_model = copy.deepcopy(self.model).to(device=self.device)
                self.update_module_param()

            if self.all_steps % self.online_model_export_freq == 0 and self.all_steps >= self.min_amount_to_train:
                fileoutname1 = f"mujoco_{self.game_name}_{self.all_steps}_online_{self.online_savepostifx}_actor.pkl"
                fileoutname2 = f"mujoco_{self.game_name}_{self.all_steps}_online_{self.online_savepostifx}_critic.pkl"
                torch.save(self.model_target.state_dict(), fileoutname1)
                torch.save(self.critic_target.state_dict(), fileoutname2)
                print(f"Saved latest dict : {fileoutname1}")
                logger.info(f"Saved latest dict : {fileoutname1}")

            if np.all(dones):
                episode_dones = 1
                T_rewards = reward_mean  # mean across agents
                if infos[0][0]['won']:
                    T_wins += 1.
                break
        return T_rewards, episode_length, diff

    def train_step(self):
        #states, actions, rewards, dones, rtg, timesteps, attention_mask = self.train_dataset.get_batch(self.batch_size)
        loader = DataLoader(self.train_dataset, shuffle=True, pin_memory=True,
                            batch_size=self.batch_size,
                            num_workers=2)
        for zz in loader:
            break

        #out_state, out_action, out_rtg, out_s4state, out_next_state, out_next_action, out_next_rtg, out_next_s4state, rewards, dones = self.train_dataset.get_batch(self.batch_size)
        out_obs_, out_state_, out_action_, out_rtg_, out_s4state_, out_next_state_, out_next_obs_, out_next_action_, out_next_rtg_, out_next_s4state_, rewards_, dones_ = [q.to(self.device) for q in zz]
        forward_s4_base_ = [out_s4state_[:, :, :,  x, :, :].squeeze(3) for x in range(self.raw_model.s4_amount)]  #was_model
        forward_s4_next_ = [out_next_s4state_[:, :, :,  x, :, :].squeeze(3) for x in range(self.raw_model.s4_amount)]  #was_model

        # modify to be used in the step method of S4 blocks
        # Split the tensor along the third dimension
        # split_tensors = torch.split(tensor, 1, dim=2)
        # Remove the singleton dimensions from each tensor
        # result_tensors = [t.squeeze(1).squeeze(1) for t in split_tensors]
        out_obs_split = torch.split(out_obs_, 1, dim=2)
        out_obs = [t.squeeze(1).squeeze(1) for t in out_obs_split]
        out_state_split = torch.split(out_state_, 1, dim=2)
        out_state = [t.squeeze(1).squeeze(1) for t in out_state_split]
        out_action_split = torch.split(out_action_, 1, dim=2)
        out_action = [t.squeeze(1).squeeze(1) for t in out_action_split]
        out_rtg_split = torch.split(out_rtg_, 1, dim=2)
        out_rtg = [t.squeeze(1).squeeze(1) for t in out_rtg_split]
        s4_state_split = [[t.squeeze(1) for t in torch.unbind(tensor, dim=2)] for tensor in forward_s4_base_]
        # Step 2: Transpose the resulting list of lists to get 6 lists, each containing 3 tensors
        forward_s4_base = list(map(list, zip(*s4_state_split)))

        out_obs_split_next = torch.split(out_next_obs_, 1, dim=2)
        out_next_obs = [t.squeeze(1).squeeze(1) for t in out_obs_split_next]
        out_state_split_next = torch.split(out_next_state_, 1, dim=2)
        out_next_state = [t.squeeze(1).squeeze(1) for t in out_state_split_next]
        out_action_split_next = torch.split(out_next_action_, 1, dim=2)
        out_next_action = [t.squeeze(1).squeeze(1) for t in out_action_split_next]
        out_rtg_split_next = torch.split(out_next_rtg_, 1, dim=2)
        out_next_rtg = [t.squeeze(1).squeeze(1) for t in out_rtg_split_next]
        s4_state_split_next = [[t.squeeze(1) for t in torch.unbind(tensor, dim=2)] for tensor in forward_s4_next_]
        # Step 2: Transpose the resulting list of lists to get 6 lists, each containing 3 tensors
        forward_s4_next = list(map(list, zip(*s4_state_split_next)))

        rewards_split = torch.split(rewards_, 1, dim=2)
        rewards = [t.squeeze(1).squeeze(1) for t in rewards_split]

        dones_split = torch.split(dones_, 1, dim=2)
        dones = [t.squeeze(1) for t in dones_split]
        ## Critic update

        loss1, loss2 = 0, 0
        if self.all_steps % self.step_partial_dat[3] < self.step_partial_dat[2] or self.all_steps <= self.fine_tune_critic_steps:
            self.critic_optimizer.zero_grad()
            with torch.no_grad():
                actor_actions_next, _ = self.model_target.step_forward(out_next_obs, out_next_state, out_next_action, None, out_next_rtg, None, s4_states=forward_s4_next)
            # if self.critic_rtg_type:
            #     Q_critic = self.critic(out_state, out_action, out_rtg)
            #     Q_critic_next = self.critic_target(out_next_obs, out_next_state, actor_actions_next, out_next_rtg)
            # else:
            one_hot_agent_id = torch.eye(self.num_agents)
            # print(one_hot_agent_id)
            if torch.cuda.is_available():
                one_hot_agent_id = one_hot_agent_id.cuda()

            obs_with_id = []
            next_obs_with_id = []
            for i in range(self.num_agents):
                one_hot_id = one_hot_agent_id[i].unsqueeze(0).expand(out_obs[i].shape[0], -1)
                obs_with_id.append(torch.cat((out_obs[i], one_hot_id), dim=1))
                next_obs_with_id.append(torch.cat((out_next_obs[i], one_hot_id), dim=1))

            Q_critic = [self.critic.q(obs_with_id[i]) for i in range(self.num_agents)]

            current_Q = []
            for q, action in zip(Q_critic, out_next_action):
                # Get the indices of the actions
                action_indices = action.argmax(dim=1, keepdim=True)
                # Gather the Q-values using the indices
                gathered_q = q.gather(1, action_indices)
                # Append to the list
                current_Q.append(gathered_q)

            out_state_stack = torch.stack(out_state, dim=1)
            out_state_input = out_state_stack.unsqueeze(1).permute(0, 1, 2, 3)
            w_q, b_q = self.critic.mix(out_state_input)
            current_Q = torch.stack(current_Q, dim=2).unsqueeze(3)
            Q_total = (w_q * current_Q).sum(dim=-2) + b_q.squeeze(dim=-1)

            v_next = [self.critic_target.v(next_obs_with_id[i]) for i in range(self.num_agents)]
            out_next_state_stack = torch.stack(out_next_state, dim=1)
            out_next_state_input = out_next_state_stack.unsqueeze(1).permute(0, 1, 2, 3)
            w_next, b_next = self.critic_target.mix(out_next_state_input)
            v_next = torch.stack(v_next, dim=2).unsqueeze(3)
            v_next_total = (w_next * v_next).sum(dim=-2) + b_next.squeeze(dim=-1)

            dones = torch.stack(dones, dim=1)
            exp_q_total= rewards[0].unsqueeze(-1) + self.gamma*(1-(torch.min(dones, dim=1, keepdim=True).values).unsqueeze(-1))*v_next_total.detach()
            # y = rewards + self.gamma * Q_critic_next * (1 - dones)
            # losscritic = torch.pow(Q_critic - y, 2).mean()

            q_loss = ((Q_total - exp_q_total.detach())**2).mean()

            Q_critic_target = [self.critic_target.q(obs_with_id[i]) for i in range(self.num_agents)]
            target_Q = []
            for q, action in zip(Q_critic_target, out_next_action):
                # Get the indices of the actions
                action_indices = action.argmax(dim=1, keepdim=True)
                # Gather the Q-values using the indices
                gathered_q = q.gather(1, action_indices)
                # Append to the list
                target_Q.append(gathered_q)
            w_target_q, b_target_q = self.critic_target.mix(out_state_input)
            target_Q = torch.stack(target_Q, dim=2).unsqueeze(3)

            v = [self.critic.v(obs_with_id[i]) for i in range(self.num_agents)]
            v = torch.stack(v, dim=2).unsqueeze(3)
            z = 1 / self.alpha * (w_target_q.detach() * target_Q.detach() - w_target_q.detach() * v)
            z = torch.clamp(z, min=-10.0, max=10.0)
            max_z = torch.max(z)
            if torch.cuda.is_available():
                max_z = torch.where(max_z < -1.0, torch.tensor(-1.0).cuda(), max_z)
            else:
                max_z = torch.where(max_z < -1.0, torch.tensor(-1.0), max_z)
            max_z = max_z.detach()

            v_loss = torch.exp(z - max_z) + torch.exp(-max_z) * w_target_q.detach() * v / self.alpha
            v_loss = v_loss.mean()

            self.q_optimizer.zero_grad()
            q_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.q_param, self.grad_norm_clip)
            self.q_optimizer.step()
            loss1 = q_loss.detach().cpu().item()

            self.v_optimizer.zero_grad()
            v_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.v_param, self.grad_norm_clip)
            self.v_optimizer.step()
            loss2 = v_loss.detach().cpu().item()
            # # Updatevars:
            # losscritic.backward()
            # torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .5) # .25
            # self.critic_optimizer.step()
            # loss1 = losscritic.detach().cpu().item()
        loss3 = None
        if self.all_steps > self.fine_tune_critic_steps and self.all_steps % self.step_partial_dat[1] < self.step_partial_dat[0]:
            self.optimizer.zero_grad()
            exp_a = torch.exp(z).detach().squeeze(-1)
            actor_actions, _ = self.model.step_forward(out_obs, out_state, out_action, None, out_rtg, None,
                                                       s4_states=forward_s4_base) # was step forward

            actor_actions = torch.stack(actor_actions, dim=1)
            actor_actions = actor_actions.unsqueeze(1)

            dist = torch.distributions.Categorical(logits=actor_actions)
            out_next_action_stack = torch.stack(out_next_action, dim=1).unsqueeze(1)
            log_probs = dist.log_prob(out_next_action_stack.max(-1).indices)
            entropy = dist.entropy().mean()

            # Define entropy coefficient (you can tune this value)
            entropy_coef = 0.01  # A small positive value to encourage exploration

            # actor_loss = -(exp_a * log_probs).mean()
            actor_loss = -(exp_a * log_probs).mean() - entropy_coef * entropy

            actor_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_norm_clip)
            self.optimizer.step()
            # if self.critic_rtg_type:
            #     Q_critic_now = -self.critic(out_state, actor_actions, out_rtg).mean()
            # else:
                # Q_critic_now = -self.critic(out_state, actor_actions).mean()
                # Q_critic_now.backward()  # retain_graph=Tru
                # torch.nn.utils.clip_grad_norm_(self.model.parameters(), .5)  # .25
                # self.optimizer.step()

            self.raw_model.pre_val_setup() #was_model
            self.raw_model_target.pre_val_setup() #was_model
            loss3 = actor_loss.detach().cpu().item()

        ## Actor update
        #self.critic.eval()
        #for param in self.critic.parameters():
        #    param.requires_grad = False


        # Updatevars:
        #self.optimizer.zero_grad()
        #Q_critic_now.backward() #retain_graph=True
        #torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1) # .25
        #self.optimizer.step()


        #action_target = torch.clone(actions)
        #print(f"LOGXXX states {states.shape}")
        #print(f"LOGXXX actions {actions.shape}")
        #print(f"LOGXXX rewards {rewards.shape}")
        #print(f"LOGXXX rtg {rtg.shape}")

        #with torch.no_grad():
        #    self.diagnostics['training/action_error'] = torch.mean(
        #        (action_preds - action_target) ** 2).detach().cpu().item()
        return [loss1, loss2, loss3]

class StateActionReturnDataset_Online_Qlearn(Dataset):
    def __init__(self, max_buffer_size, game_name, seed, device):
        Dataset.__init__(self)
        self.max_buffer_size = max_buffer_size
        #self.vocab_size = max(actions) + 1
        self.obs_stack = []
        self.state_stack = []
        self.actions_stack = []
        self.rtg_stack = []
        self.s4_state_stack = []

        self.next_obs_stack = []
        self.next_state_stack = []
        self.next_actions_stack = []
        self.next_rtg_stack = []
        self.s4_next_state_stack = []

        self.done_marker_stack = []
        self.rewards_stack = []
        #self.all_totrewards = []
        self.game_name = game_name
        self.seed = seed
        self.device = device

        self.currhist_obs = []
        self.currhist_state = []
        self.currhist_action = []
        self.currhist_rtg = []
        self.currhist_reward = []
        self.currhist_s4_states = []
        self.currhist_dones = []

        return

    def __len__(self):
        return len(self.state_stack)

    def __getitem__(self, idx):
        stacked_s4_state = torch.stack(self.s4_state_stack[idx], dim=1)
        stacked_s4_next_state = torch.stack(self.s4_next_state_stack[idx], dim=1)
        return [torch.tensor(self.obs_stack[idx], dtype=torch.float32).detach(),
                torch.tensor(self.state_stack[idx], dtype=torch.float32).detach(),
                torch.tensor(self.actions_stack[idx], dtype=torch.float32).detach(),
                torch.tensor(self.rtg_stack[idx], dtype=torch.long).detach(),
                torch.tensor(stacked_s4_state, dtype=stacked_s4_state.dtype).detach(),
                torch.tensor(self.next_state_stack[idx], dtype=torch.float32).detach(),
                torch.tensor(self.next_obs_stack[idx], dtype=torch.float32).detach(),
                torch.tensor(self.next_actions_stack[idx], dtype=torch.float32).detach(),
                torch.tensor(self.next_rtg_stack[idx], dtype=torch.long).detach(),
                torch.tensor(stacked_s4_next_state, dtype=stacked_s4_next_state.dtype).detach(),
                torch.tensor(self.rewards_stack[idx], dtype=torch.long).detach(),
                torch.tensor(self.done_marker_stack[idx], dtype=torch.float).detach()]

    def clean_buffer(self, amount=500):
        for z in range(amount):
            self.obs_stack.pop(0)
            self.state_stack.pop(0)
            self.actions_stack.pop(0)
            self.rtg_stack.pop(0)
            self.s4_state_stack.pop(0)

            self.next_obs_stack.pop(0)
            self.next_state_stack.pop(0)
            self.next_actions_stack.pop(0)
            self.next_rtg_stack.pop(0)
            self.s4_next_state_stack.pop(0)

            self.rewards_stack.pop(0)
            self.done_marker_stack.pop(0)

        return

    def add_observation(self, obs, state, rtg, s4_state, action,
                            next_obs, next_state, next_rtg, s4_next_state, next_action, done):
        if len(self.state_stack) > self.max_buffer_size:
            self.clean_buffer()

        self.obs_stack.append(obs)
        self.state_stack.append(state)
        self.actions_stack.append(action)
        self.rtg_stack.append(rtg)
        # self.s4_state_stack.append(torch.cat([[x.unsqueeze(0) for x in s4_state_agent] for s4_state_agent in s4_state], dim=1))
        self.s4_state_stack.append([torch.cat([x.unsqueeze(0) for x in s4_state_agent], dim=1) for s4_state_agent in s4_state])

        self.next_obs_stack.append(next_obs)
        self.next_state_stack.append(next_state)
        self.next_actions_stack.append(next_action)
        self.next_rtg_stack.append(next_rtg)
        self.s4_next_state_stack.append([torch.cat([x.unsqueeze(0) for x in s4_state_agent], dim=1) for s4_state_agent in s4_next_state])

        self.rewards_stack.append(rtg-next_rtg)
        self.done_marker_stack.append(done)

        return


    def add_obs_shift(self, obs, state, rtg, s4_state, action,
                            next_obs, next_state, next_rtg, s4_next_state, done):
        if len(self.currhist_state) == 0:
            self.currhist_obs.append(obs)
            self.currhist_state.append(state)
            pre_actions = list(np.zeros_like(action))
            pre_actions = torch.tensor(pre_actions, dtype=torch.int64)
            self.currhist_action.append(pre_actions)
            self.currhist_rtg.append(rtg)
            #self.currhist_s4_states.append(torch.cat([x.unsqueeze(0) for x in s4_state], dim=0))
            self.currhist_s4_states.append(s4_state)
        self.currhist_obs.append(next_obs)
        self.currhist_state.append(next_state)
        self.currhist_action.append(action)
        self.currhist_rtg.append(next_rtg)
        #self.currhist_s4_states.append(torch.cat([x.unsqueeze(0) for x in s4_next_state], dim=0))
        self.currhist_s4_states.append(s4_next_state)
        #self.currhist_reward.append(rtg - next_rtg)
        self.currhist_dones.append(done)

        if bool(done.all()):
            return self.add_curr_to_buffer()
        return None

    def add_curr_to_buffer(self):
        rtg_shift = self.currhist_rtg[-1]
        new_rtg = [x-rtg_shift for x in self.currhist_rtg]
        for z in range(len(self.currhist_state)-1):

            self.add_observation(self.currhist_obs[z], self.currhist_state[z],
                                 new_rtg[z],
                                 self.currhist_s4_states[z],
                                 self.currhist_action[z],
                                 self.currhist_obs[z+1],
                                 self.currhist_state[z+1],
                                 new_rtg[z+1],
                                 self.currhist_s4_states[z+1],
                                 self.currhist_action[z+1],
                                 self.currhist_dones[z]
                                 )
        self.currhist_obs = []
        self.currhist_state = []
        self.currhist_action = []
        self.currhist_rtg = []
        self.currhist_reward = []
        self.currhist_s4_states = []
        self.currhist_dones = []
        return rtg_shift


    def get_batch(self, batch_size):
        batch_inds = np.random.choice(
            np.arange(len(self)),
            size=batch_size,
            replace=True,
        )
        flag = True
        for x in batch_inds:
            if flag:
                out_state, out_action, out_rtg, out_s4state, out_next_state, out_next_action, out_next_rtg, out_next_s4state, rewards, dones = self.__getitem__(x)
                out_state = out_state.unsqueeze(0)
                out_action = out_action.unsqueeze(0)
                out_rtg = out_rtg.unsqueeze(0)
                out_s4state = out_s4state.unsqueeze(0)

                out_next_state = out_next_state.unsqueeze(0)
                out_next_action = out_next_action.unsqueeze(0)
                out_next_rtg = out_next_rtg.unsqueeze(0)
                out_next_s4state = out_next_s4state.unsqueeze(0)

                rewards = rewards.unsqueeze(0)
                dones = dones.unsqueeze(0)
                flag = False
            else:
                out_state_n, out_action_n, out_rtg_n, out_s4state_n, out_next_state_n, out_next_action_n, out_next_rtg_n, out_next_s4state_n, rewards_n, dones_n = self.__getitem__(x)
                out_state = torch.cat([out_state, out_state_n.unsqueeze(0)], dim=0)
                out_action = torch.cat([out_action, out_action_n.unsqueeze(0)], dim=0)
                out_rtg = torch.cat([out_rtg, out_rtg_n.unsqueeze(0)], dim=0)
                out_s4state = torch.cat([out_s4state, out_s4state_n.unsqueeze(0)], dim=0)

                out_next_state = torch.cat([out_next_state, out_next_state_n.unsqueeze(0)], dim=0)
                out_next_action = torch.cat([out_next_action, out_next_action_n.unsqueeze(0)], dim=0)
                out_next_rtg = torch.cat([out_next_rtg, out_next_rtg_n.unsqueeze(0)], dim=0)
                out_next_s4state = torch.cat([out_next_s4state, out_next_s4state_n.unsqueeze(0)], dim=0)

                rewards = torch.cat([rewards, rewards_n.unsqueeze(0)], dim=0)
                dones = torch.cat([dones, dones_n.unsqueeze(0)], dim=0)
        return out_state.to(self.device), out_action.to(self.device), out_rtg.to(self.device), out_s4state.to(self.device),\
               out_next_state.to(self.device), out_next_action.to(self.device), out_next_rtg.to(self.device),\
               out_next_s4state.to(self.device), rewards.to(self.device), dones.to(self.device)