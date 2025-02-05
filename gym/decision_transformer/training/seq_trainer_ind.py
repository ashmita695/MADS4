import numpy as np
import torch

from decision_transformer.training.trainer_ind import Trainer
from decision_transformer.models.s4_muj_ind import *
import logging
logger = logging.getLogger(__name__)

class SequenceTrainer(Trainer):

    def train_step(self):
        print(f"training step batch size:{self.batch_size}")
        obs, states, actions, rewards, dones, rtg, timesteps, attention_mask, goals = self.get_batch(self.batch_size)
        # print(f"states dim:{states.shape}")
        # if states.shape[0] <= 2: why?
        #     return 10000

        # action_target = torch.clone(actions)
        # make one-hot actions
        action_target = [tensor.clone() for tensor in actions]

        # print(f"action targets dim:{action_target.shape}")

        do_dicrete = False
        s4_model = isinstance(self.model, S4_mujoco_wrapper)
        if s4_model:
            if self.model.config.discrete > 0:
                print(f"yes discrete")
                do_dicrete = True
        if do_dicrete > 0:
            state_target = torch.clone(states[:,1:,...])
            rtg_target = torch.clone(rtg[:,1:-1,...])

        ###############

        target_goal = None
        # if self.model.config.base_model == "ant_con":
        #     pattn = torch.cat([attention_mask,
        #                        torch.zeros((states.shape[0], 1), dtype=attention_mask.dtype, device=attention_mask.device)],
        #                       dim=1)
        #     uu = pattn.argmin(dim=1) - 1
        #     #print(f"LOGX ITER ORIGMASK {attention_mask.clone().cpu()}")
        #     #print(f"LOGX ITER MASK VAL {uu.clone().cpu()}")
        #     state_target_mid = torch.zeros_like(states[:, 0, :2])
        #     for tempb in range(states.shape[0]):
        #         state_target_mid[tempb, :] = states[tempb, int(uu[tempb]), :2].clone()
        #     state_target_mid = state_target_mid.clone().unsqueeze(1).expand(states.shape[0], states.shape[1], 2)
        #     target_goal = state_target_mid.clone()

        to_pass_rtg = [_[:,:-1] for _ in rtg]
        # if self.rtg_set_all:
        #     print(f"{self.rtg_set_all}")
        #     to_pass_rtg = torch.ones_like(to_pass_rtg) * rtg[:,:-1].max()

        state_preds, action_preds, reward_preds = self.model.forward(
            obs, states, actions, rewards, to_pass_rtg, timesteps, attention_mask=attention_mask, goals=goals,
            target_goal=target_goal
        )
        # print(f"states_pred:{state_preds}")

        action_preds_orig = [action_preds_agent.clone().detach() for action_preds_agent in action_preds]
        if s4_model:
            attention_mask = [attention_mask_agent[...,1:] for attention_mask_agent in attention_mask]
            action_target = [action_target_agent[:, 1:, ...] for action_target_agent in action_target]

        n_agents = len(states)
        state_dim, act_dim = states[0].shape[-1], action_preds[0].shape[2]
        if self.critic is None:
            if not do_dicrete:
                print(f"not do discrete")
                for i in range(n_agents):
                    action_preds[i] = action_preds[i].reshape(-1, act_dim)[attention_mask[i].reshape(-1) > 0]
                    action_target[i] = action_target[i].reshape(-1, act_dim)[attention_mask[i].reshape(-1) > 0]
            else:
                print(f"do discrete")
                action_target = ((1 + action_target) * (self.model.config.discrete / 2)).to(dtype=torch.long)
                action_target = torch.maximum(torch.minimum(action_target, self.model.config.discrete * torch.ones_like(action_target) -1), torch.zeros_like(action_target))
                action_target = action_target.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0].reshape(-1)


                state_target = ((state_target -self.model.config.state_bound[0]) * (self.model.config.discrete / (self.model.config.state_bound[1] -self.model.config.state_bound[0]))).to(dtype=torch.long)
                state_target = torch.maximum(torch.minimum(state_target,self.model.config.discrete * torch.ones_like(state_target) - 1), torch.zeros_like(state_target))
                state_target = state_target.reshape(-1, state_dim)[attention_mask.reshape(-1) > 0].reshape(-1)

                action_preds = action_preds.reshape(-1, act_dim, self.model.config.discrete)[
                    attention_mask.reshape(-1) > 0].reshape(-1, self.model.config.discrete)
                state_preds = state_preds.reshape(-1, state_dim, self.model.config.discrete)[
                    attention_mask.reshape(-1) > 0].reshape(-1, self.model.config.discrete)

                reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
                rtg_target = rtg_target.reshape(-1, 1)[attention_mask.reshape(-1) > 0]

                lossf = torch.nn.CrossEntropyLoss()
                #topr = f"LOGXZZ pred {action_preds.shape} targ {action_target.shape} X min {action_target.min().cpu().item():4} max {action_target.max().cpu().item():4}"
                #print(topr)
                #logger.info(topr)
                loss = lossf(torch.cat([action_preds, state_preds], dim=0), torch.cat([action_target, state_target], dim=0))
                loss += torch.mean((reward_preds - rtg_target) ** 2)

            reward_losses_enable = False
            if s4_model:
                if self.model.config.base_model == "ant_reward_target":
                    reward_losses_enable = True
            if reward_losses_enable:
                print(f"reward_losses_enable:{reward_losses_enable}")
                loss_actions = torch.mean((action_preds - action_target) ** 2)

                reward_preds = reward_preds.reshape(-1, 1)[attention_mask.reshape(-1) > 0]
                reward_target = torch.clone(rtg)[:, 1:-1].reshape(-1, 1)[attention_mask.reshape(-1) > 0]

                loss_reward = torch.mean((reward_preds - reward_target) ** 2)

                loss_improve = torch.mean(torch.pow(reward_preds,2))

                if self.model.config.s4_ant_multi_lr is None:
                    fc1, fc2, fc3 = 1, 1, 0.05
                else:
                    temp = self.model.config.s4_ant_multi_lr.split(",")
                    fc1, fc2, fc3 = [float(x) for x in temp]
                loss = fc1 * loss_actions + fc2 * loss_reward + fc3 * loss_improve
                print(f"LOGX ITER MULTILOSS act: {loss_actions.cpu().item():.5f} X reward {loss_reward.cpu().item():.5f} X reward_opt {loss_improve.cpu().item():.5f}")
            elif self.model.config.base_model == "ant_con":
                loss_actions = torch.mean((action_preds - action_target) ** 2)

                state_preds = state_preds.reshape(-1, 2)[attention_mask.reshape(-1) > 0]
                #state_target_mid = states[:, real_end_mask, :2].clone().reshape(states.shape[0], 1, 2).expand(states.shape[0], states.shape[1]-1, 2)
                #state_target_mid = state_target_mid.clone().reshape(states.shape[0], 1, 2).expand(states.shape[0], states.shape[1]-1, 2)
                #state_target_mid = state_target_mid.clone().unsqueeze(1).expand(states.shape[0], states.shape[1]-1, 2)
                state_target_mid = state_target_mid[:,:-1,:]
                state_target = state_target_mid.reshape(-1, 2)[attention_mask.reshape(-1) > 0]
                newdiff_mid = rtg[:, 0, :].clone().reshape(-1,1,1).expand(states.shape[0], states.shape[1]-1, 2)
                newdiff = (1*(newdiff_mid > 0).clone()).reshape(-1, 2)[attention_mask.reshape(-1) > 0]

                loss_target = torch.mean(newdiff*torch.abs(state_preds - state_target)**1.2)

                if self.model.config.s4_ant_multi_lr is None:
                    fc1, fc2 = 1, 200
                else:
                    temp = self.model.config.s4_ant_multi_lr.split(",")
                    fc1, fc2 = [float(x) for x in temp]
                loss = fc1 * loss_actions + fc2 * loss_target
                print(f"LOGX ITER MULTILOSS act: {loss_actions.cpu().item():.5f} X target {loss_target.cpu().item():.5f} X TAR {self.model.target_goal.cpu()}")
                for tt, item in enumerate([[2,0], [-3,3]]):
                    print(f"LOGX ITER MULTILOSS TARREAL{tt} {state_target_mid[item[0],item[1],:].cpu()} Y {newdiff_mid[item[0],item[1],:]} Y {newdiff_mid[item[0],item[1],:]>0}")
            elif goals is None:
                if not do_dicrete:
                    print(f"goals none and not do discrete")
                    loss = 0
                    for i in range(n_agents):
                        loss = loss + self.loss_fn(
                            None, action_preds[i], None,
                            None, action_target[i], None,
                        )
            else:
                # B L A
                # B 2
                target_achieved_factor = 1.0/6.8
                # print("here!")
                statep = states[:,:,:2].transpose(-1,-2)
                goalsp = goals.unsqueeze(-1).expand((goals.shape[0], goals.shape[1], states.shape[1]))
                diff = torch.sum(torch.pow(statep - goalsp, 2), dim=1)
                diff = diff < target_achieved_factor
                #argsrise = torch.argmax(diff, dim=1) #was 9, 1, 0.5
                newdiff = 5 * (torch.sum(diff, dim=1) > 0).detach() + 1
                newdiff = newdiff.unsqueeze(-1).expand(states.shape[0], states.shape[1]-1) * (1+ -1 *diff[:,:-1]) + diff[:,:-1]*0.2
                newdiff = newdiff.unsqueeze(-1).expand(states.shape[0], states.shape[1]-1, act_dim).reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
                print(f"LOGX diff. 10: {int(torch.sum((newdiff==10))):5} 1: {int(torch.sum((newdiff==1))):5} 0.5: {int(torch.sum((newdiff==0.5))):5} dim {newdiff.shape}")
                loss = torch.mean(newdiff * ((action_preds - action_target) ** 2))

                if self.model.config.base_model == "ant_con_auto":
                    target_preds = state_preds.reshape(-1, 2)[attention_mask.reshape(-1) > 0]
                    target_target = torch.clone(goalsp)[:,:,1:].transpose(-1,-2)
                    target_target = target_target.reshape(-1, 2)[attention_mask.reshape(-1) > 0]

                    diff = torch.sum(torch.pow(statep - goalsp, 2), dim=1)
                    diff = diff > target_achieved_factor
                    newdiff = 5*diff.detach()+0.01
                    newdiff = newdiff.unsqueeze(-1).expand(states.shape[0], states.shape[1], 2)[:,1:,:].reshape(-1, 2)[
                        attention_mask.reshape(-1) > 0]

                    loss = loss + torch.mean(newdiff * ((target_preds - target_target) ** 2))



            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()

            with torch.no_grad():
                #self.diagnostics['training/action_error'] = torch.mean((action_preds-action_target)**2).detach().cpu().item()
                self.diagnostics['training/action_error'] = loss.detach().cpu().item()
            retloss = loss.detach().cpu().item()
            print(f"LOGX ITER LOSS {retloss}")
        else:
            _, action_preds_from_tar, _ = self.model_tar.forward(
                states, actions, rewards, rtg[:, :-1], timesteps, attention_mask=attention_mask,
            )
            gamma = 0.99
            batch_len = states.shape[1]
            action_preds_from_tar = torch.cat([action_preds_from_tar,
                                      torch.zeros((self.batch_size, 1, act_dim), dtype=action_preds_from_tar.dtype,
                                                  device=action_preds_from_tar.device)], dim=1)
            action_preds = torch.cat([action_preds,
                                      torch.zeros((self.batch_size, 1, act_dim), dtype=action_preds.dtype,
                                                  device=action_preds.device)], dim=1)
            usemask = attention_mask.reshape(self.batch_size, -1) > 0
            states = states.reshape(-1, state_dim)
            actions = actions.reshape(-1, act_dim)
            action_preds_clone = action_preds.reshape(-1, act_dim)
            action_preds_from_tar = action_preds_from_tar.reshape(-1, act_dim)
            rtg = rtg[:,:-1,:].reshape(-1, 1)
            #1
            y = rewards.clone().detach().squeeze(-1)[:, [rewards.shape[1]-1]+list(range(0,rewards.shape[1]-1))] / float(self.reward_scale) + gamma * self.critic_tar(states.clone().detach(), action_preds_from_tar.clone().detach(), rtg.clone().detach()).reshape(self.batch_size, -1)
            criticloss = self.critic(states.clone().detach(), actions.clone().detach(), rtg.clone().detach()).reshape(self.batch_size, -1)[:, :-1] - y[:, 1:]
            criticloss = torch.mean(torch.pow(criticloss[usemask.clone().detach()], 2))

            self.critic_optimizer.zero_grad()
            criticloss.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .25)
            self.critic_optimizer.step()
            #2
            y = rewards.clone().detach().squeeze(-1)[:, [rewards.shape[1]-1]+list(range(0,rewards.shape[1]-1))] / float(self.reward_scale) + gamma * self.critic2_tar(states.clone().detach(), action_preds_from_tar.clone().detach(), rtg.clone().detach()).reshape(self.batch_size, -1)
            criticloss2 = self.critic2(states.clone().detach(), actions.clone().detach(), rtg.clone().detach()).reshape(self.batch_size, -1)[:, :-1] - y[:, 1:]
            criticloss2 = torch.mean(torch.pow(criticloss2[usemask.clone().detach()], 2))

            self.critic_optimizer2.zero_grad()
            criticloss2.backward(retain_graph=False)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), .25)
            self.critic_optimizer2.step()

            actorloss = -self.critic(states, action_preds_clone.clone(), rtg).reshape(self.batch_size, -1)[:, :-1][usemask]
            actorloss = torch.mean(actorloss)

            actorloss2 = -self.critic(states, action_preds_clone.clone(), rtg).reshape(self.batch_size, -1)[:, :-1][usemask]
            actorloss2 = torch.mean(actorloss2)

            optloss = actorloss if actorloss.item() > actorloss2.item() else actorloss2

            self.optimizer.zero_grad()
            optloss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), .25)
            self.optimizer.step()
            retloss = criticloss2.detach().cpu().item() + criticloss.detach().cpu().item() + optloss.detach().cpu().item()
            print(f"LOGX ITER LOSS {retloss} : {criticloss.detach().cpu().item():.5f} X {criticloss2.detach().cpu().item():.5f} X {optloss.detach().cpu().item():.5f}")
            if self.train_step_c % 100 == 0:
                #self.model_tar.load_state_dict(self.model.state_dict())
                #self.model_tar.load_state_dict(self.model.state_dict())
                print(f"Updating target neworks")
                self.tau = 0.05
                for target_param, param in zip(self.model_tar.parameters(), self.model.parameters()):  # was_model
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                for target_param, param in zip(self.critic_tar.parameters(), self.critic.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)
                for target_param, param in zip(self.critic2_tar.parameters(), self.critic2.parameters()):
                    target_param.data.copy_(target_param.data * (1.0 - self.tau) + param.data * self.tau)

        if self.train_step_c == -1 and isinstance(self.model, S4_mujoco_wrapper):
            if self.model.config.track_step_err:
                batch_len = states.shape[1]
                self.model.eval()
                self.model.pre_val_setup()
                s4_states = [r.detach() for r in self.model.get_initial_state((self.batch_size), states.device)]
                step_pred_actions = torch.zeros_like(action_preds_orig)
                states = states.reshape(self.batch_size, -1, state_dim)
                actions = actions.reshape(self.batch_size, -1, act_dim)
                rtg = rtg.reshape(self.batch_size, -1, 1)
                for stp in range(batch_len-1):
                    z, s4_states = self.model.step_forward(states[:, stp+1, :], actions[:, stp, :], None, rtg[:, stp+1, :],
                                                    torch.zeros_like(rtg), s4_states)
                    #print(f"LOGX step_pred_actions {step_pred_actions.shape}")
                    #print(f"LOGX z {z.shape}")
                    step_pred_actions[:, stp, :] = z
                tp = f"Average Diff  L2: {torch.mean(torch.pow(action_preds_orig - step_pred_actions, 2))}"
                print(tp); logger.info(tp)
                tp = f"Average Diff  L1: {torch.mean(torch.abs(action_preds_orig - step_pred_actions))}"
                print(tp); logger.info(tp)
                tp = f"Average first L1: {torch.mean(torch.abs((action_preds_orig - step_pred_actions)[:, 0, :]))}"
                print(tp); logger.info(tp)
                tp = f"Average last  L1: {torch.mean(torch.abs((action_preds_orig - step_pred_actions)[:, -2, :]))}"
                print(tp); logger.info(tp)
        return retloss
