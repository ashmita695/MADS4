import numpy as np
import torch
from decision_transformer.models.s4_muj_ind import *
import copy
import time


class Trainer:

    def __init__(self, model, optimizer, batch_size, get_batch, loss_fn,
                 scheduler=None, eval_fns=None, runlabel = "default", critic=None,
                 critic_optimizer=None,
                 critic_scheduler=None,
                 reward_scale = None,
                 variant=None,
                 rtg_set_all=False):
        self.model = model
        self.rtg_set_all = rtg_set_all
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.get_batch = get_batch
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.eval_fns = [] if eval_fns is None else eval_fns
        self.diagnostics = dict()
        self.runlabel = runlabel
        self.critic_optimizer = critic_optimizer
        self.critic_scheduler = critic_scheduler
        self.train_step_c = 0
        self.reward_scale = reward_scale
        self.variant = variant
        self.critic, self.critic2 = critic
        if self.critic is not None:
            self.critic_tar = copy.deepcopy(self.critic)
            self.model_tar = copy.deepcopy(self.model)

            self.critic2_tar = copy.deepcopy(self.critic2)
            self.critic_optimizer2 = torch.optim.AdamW(
                self.critic2.parameters(),
                lr=1e-4,
                weight_decay=1e-4,
            )
            self.critic_scheduler2 = torch.optim.lr_scheduler.LambdaLR(
                self.critic_optimizer2,
                lambda steps: min((steps + 1) / 10000, 1)
            )
            self.critic_tar.eval()
            self.critic2_tar.eval()
            self.model_tar.eval()
            print(f"Created copies")
        else:
            self.critic = None

        self.start_time = time.time()

    def train_iteration(self, num_steps, iter_num=0, print_logs=False):
        train = True
        logs = dict()
        if train:
            train_losses = []
            logs = dict()

            train_start = time.time()

            self.model.train()
            if self.critic is not None:
                self.critic.train()
            for _ in range(num_steps):
                self.train_step_c += 1
                if self.train_step_c + 1 == num_steps:
                    self.train_step_c = -1

                train_loss = self.train_step()
                print(f"training step:{_/num_steps}:train loss:{train_loss}")
                train_losses.append(train_loss)
                if self.scheduler is not None:
                    self.scheduler.step()
                if self.critic_scheduler is not None:
                    self.critic_scheduler.step()
                    self.critic_scheduler2.step()
            self.train_step_c = 0

            logs['time/training'] = time.time() - train_start

            logs['training/train_loss_mean'] = np.mean(train_losses)
            logs['training/train_loss_std'] = np.std(train_losses)

        print(f"evaluation")
        eval_start = time.time()

        self.model.eval()
        if isinstance(self.model, S4_mujoco_wrapper):
            if self.model.config.single_step_val:
                self.model.pre_val_setup()

        avg_returns = 0
        for eval_fn in self.eval_fns:
            outputs = eval_fn(self.model)
            current_avg_returns = outputs[f'current avg returns']
            if current_avg_returns > avg_returns:
                avg_returns = current_avg_returns
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/total'] = time.time() - self.start_time
        logs['time/evaluation'] = time.time() - eval_start
        # logs['time/total'] = time.time() - self.start_time
        # logs['time/evaluation'] = time.time() - eval_start
        # logs['training/train_loss_mean'] = np.mean(train_losses)
        # logs['training/train_loss_std'] = np.std(train_losses)

        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]
        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')
        return logs, self.model, avg_returns

    def train_step(self):
        states, actions, rewards, dones, attention_mask, returns = self.get_batch(self.batch_size)
        print(f"states:{states}")
        state_target, action_target, reward_target = torch.clone(states), torch.clone(actions), torch.clone(rewards)

        state_preds, action_preds, reward_preds = self.model.forward(
            states, actions, rewards, masks=None, attention_mask=attention_mask, target_return=returns,
        )
        print(f"states_pred:{state_preds}")
        # note: currently indexing & masking is not fully correct
        loss = self.loss_fn(
            state_preds, action_preds, reward_preds,
            state_target[:,1:], action_target, reward_target[:,1:],
        )
        print(f"loss:{loss}")
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.detach().cpu().item()
