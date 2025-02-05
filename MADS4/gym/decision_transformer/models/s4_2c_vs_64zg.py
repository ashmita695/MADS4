import logging
from functools import partial
import math
import numpy as np
from scipy import special as ss
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils as U
import opt_einsum as oe

if __name__ == '__main__':
    import sys
    import os
    import inspect
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    print(currentdir)
    s4dir = os.path.join(os.path.dirname(currentdir), "s4_module")
    sys.path.insert(0, s4dir)
    from s4_module import *
    from model import TrajectoryModel
else:
    from s4_module import *
    from decision_transformer.models.model import TrajectoryModel
contract = oe.contract

class S4_config():

    def __init__(self, **kwargs):
        self.dropoutval = 0
        self.activation = None
        self.layer_norm_s4 = False
        self.single_step_val = False
        self.setup_c = False
        self.s4_resnet = False
        self.s4_onpolicy = False
        self.s4_layers = 2
        self.len_corr = False
        self.s4_trainable = True
        self.track_step_err = False
        self.recurrent_mode = False
        self.n_ssm = 1
        self.precision = 1
        self.train_noise = 0
        self.base_model = "s4"#s4
        self.discrete = 0
        self.s4_ant_multi_lr = None
        for k,v in kwargs.items():
            if k == "activation":
                if v == "gelu":
                    self.activation = F.gelu
                elif v == "relu":
                    self.activation = F.relu_
                else:
                    self.activation = None
            else:
                setattr(self, k, v)
    def reprr(self):
        out = ""
        out += "Singlestep: " + str(self.single_step_val) + " X "
        out += "setup_c: " + str(self.setup_c) + " X "
        out += "length_corr: " + str(self.len_corr) + " X "
        out += "layer_norm: " + str(self.layer_norm_s4) + " X "
        out += "dropoutval: " + str(self.dropoutval) + " X "
        out += "s4_layers: " + str(self.s4_layers) + " X "
        out += "s4_resnet: " + str(self.s4_resnet) + " X "
        out += "s4_trainable: " + str(self.s4_trainable) + " X "
        out += "s4_ssm: " + str(self.n_ssm) + " X "
        out += "s4_precision: " + str(self.precision) + " X "
        out += "s4_base_model: " + str(self.base_model) + " X "
        out += "activation: " + str(self.activation) + " X "
        out += "s4_ant_multi_lr: " + str(self.s4_ant_multi_lr)
        return out

class S4_mujoco_wrapper(TrajectoryModel):

    def __init__(
            self,
            config,
            state_dim,
            act_dim,
            max_length,
            n_embd=128,
            H=10,
            l_max=None,
            # Arguments for SSM Kernel
            d_state=64,
            measure='legs',
            dt_min=0.001,
            dt_max=0.1,
            rank=1,
            trainable=None,
            lr=None,
            use_state=True,
            stride=1,
            s4_weight_decay=0.0, # weight decay on the SS Kernel
            weight_norm=False,  # weight normalization on FF
            **kwargs
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, or inconvenient to pass in,
          set l_max=None and length_correction=True
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, D) or (B, D, L) [B=batch size, L=sequence length, D=feature dimension]
        Other options are all experimental and should not need to be configured
        """

        super().__init__(state_dim, act_dim, max_length=max_length)

        self.h = H
        self.state_dim = state_dim
        #self.config = S4_config(**kwargs)
        self.config = config
        #self.single_step_val = single_step_val
        #self.setup_c = setup_c
        self.s4_weight_decay = s4_weight_decay
        #self.layer_norm_s4 = layer_norm_s4
        #self.activation = activation
        #self.dropout_en = dropout_en

        ##Added for pre/post processing to inject the S4, taken as default from the DT:
        self.input_emb_size = n_embd
        self.n = d_state
        self.action_dim = act_dim

        self.state_encoder = nn.Sequential(nn.Linear(self.state_dim, self.input_emb_size),
                                           nn.Tanh())
        self.ret_emb = nn.Sequential(nn.Linear(1, self.input_emb_size), nn.Tanh())

        self.action_embeddings = nn.Sequential(nn.Linear(self.action_dim, self.input_emb_size), nn.Tanh())
        nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        self.input_projection = nn.Linear(self.input_emb_size*4+6, self.h)
        self.input_projection_ind = nn.Linear(self.input_emb_size * 3 + 6, self.h)

        if self.config.layer_norm_s4:
            self.input_norm_layer = nn.LayerNorm(self.h)
            self.output_norm_layer = nn.LayerNorm(self.h)
        if self.config.dropoutval>0:
            self.dropoutlayer = nn.Dropout(self.config.dropoutval)
        if self.config.activation is not None:
            self.input_proj2 = nn.Linear(self.h, self.h)

        self.l_max = l_max
        if self.config.single_step_val:
            l_max = None
        self.s4_mod = S4(H, l_max=l_max, d_state=d_state, measure=measure,
            dt_min=dt_min, dt_max=dt_max, rank=rank, trainable=trainable, lr=lr, length_correction=self.config.len_corr,
            stride=stride, weight_decay=s4_weight_decay, weight_norm=weight_norm, use_state=use_state
        )
        self.output_projection = nn.Linear(self.h,self.action_dim, bias=False)
        self.curr_state = None
        if self.config.setup_c:
            self.pre_val_setup()

    def pre_val_setup(self):
        self.s4_mod.kernel.krylov._setup()

    def forward(self, states, actions, rewards, rtg, timestep, state=None, running=False, cache=None, **kwargs): # absorbs return_output and transformer src mask
        #### forward(self, states, actions, rewards, returns_to_go, timesteps, attention_mask=None)

        #### preprocess for the S4:
        # u (batch, l_max, 84*84*4) -> inputsize (84, 84, 4), (batch, l_max, 1), (batch, l_max, 1) - state, action, reward
        del_r = 1
        batchsize = states.shape[0]
        input_len = states.shape[1]
        #if v[1] == None:
        #    return torch.ones((1,1,self.action_vocab), device=v[0].device), state
        if running:
            del_r = 0
        state_embed = self.state_encoder(states[:,del_r:,...].reshape(-1, self.state_dim).type(torch.float32).contiguous())
        if actions == None:
            action_embed = self.action_embeddings(torch.zeros_like(v[2], device=state_embed.device).reshape(-1, self.action_dim).type(torch.float))
        else:
            action_embed = self.action_embeddings(actions[:,:input_len-del_r,...].reshape(-1,self.action_dim).type(torch.float))
        reward_embed = self.ret_emb(rtg[:,del_r:,...].reshape(-1,1).type(torch.float32))
        if input_len >= 2:
            action_embed = action_embed.squeeze(-2)
            reward_embed = reward_embed.squeeze(-2)
        u = torch.zeros((batchsize*(input_len-del_r), 3*self.input_emb_size), dtype=torch.float32, device=state_embed.device)
        u[..., :self.input_emb_size] = state_embed
        u[..., self.input_emb_size: 2*self.input_emb_size] = action_embed
        u[..., 2*self.input_emb_size: 3*self.input_emb_size] = reward_embed


        u = self.input_projection(u).reshape(batchsize, input_len-del_r, self.h)
        if self.config.layer_norm_s4:
            u = self.input_norm_layer(u)
        if self.config.activation is not None:
            u = self.config.activation(u)
        if self.config.dropoutval:
            u = self.dropoutlayer(u)
        if self.config.activation is not None:
            u = self.input_proj2(u)

        u = u.transpose(-1, -2)
        # print(u.shape)
        y, next_state = self.s4_mod(u)
        ret_y = y.transpose(-1, -2)

        if self.config.layer_norm_s4:
            ret_y =self.output_norm_layer(ret_y)

        ret_y = self.output_projection(ret_y.reshape(-1,self.h))
        if self.config.discrete > 0:
            ret_y = ret_y.reshape(batchsize, input_len - del_r, self.action_dim, self.config.discrete)
        else:
            ret_y = ret_y.reshape(batchsize, input_len - del_r, self.action_dim)

        return None, ret_y, next_state

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, state=None, running=False, **kwargs):
        # get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs)
        assert not self.training

        states = states.reshape(1, -1, self.state_dim)
        actions = actions.reshape(1, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(1, -1, 1)
        timesteps = timesteps.reshape(1, -1)

        if not self.config.single_step_val:
            if self.l_max is not None:
                states = states[:, -self.l_max:]
                actions = actions[:, -self.l_max:]
                returns_to_go = returns_to_go[:, -self.l_max:]
                timesteps = timesteps[:, -self.l_max:]
            return self.forward(states, actions, rewards, returns_to_go, timesteps, state, running=True, **kwargs)[1][0,-1,:]

        # run single step. need to reconfigure
        u = torch.zeros((1,3*self.input_emb_size), device=v[0].device)
        state_embed = self.state_encoder(v[0][:, -1, ...].reshape(-1, 4, 84, 84).type(torch.float32).contiguous())
        if v[1] == None:
            action_embed = self.action_embeddings(torch.zeros_like(v[2][:, -1, ...], device=state_embed.device).reshape(-1, 1).type(torch.float))
        else:
            action_embed = self.action_embeddings(v[1][:, -1, ...].reshape(-1,1).type(torch.float))
        reward_embed = self.ret_emb(v[2][:, -1, ...].reshape(-1,1).type(torch.float32))
        ### print("LOG v[0] shape " + str(v[0].shape))
        ### print("LOG v[1] shape " + str(v[1].shape))
        ### print("LOG v[2] shape " + str(v[2].shape))
        ### print("LOG u before shape " + str(u.shape))
        # need to change here : input previous agent action
        u[:, :self.input_emb_size] = state_embed
        u[:, self.input_emb_size: 2*self.input_emb_size] = action_embed
        u[:, 2*self.input_emb_size: 3*self.input_emb_size] = reward_embed
        u = self.input_projection(u)
        if self.config.layer_norm_s4:
            u = self.input_norm_layer(u)
        if self.config.activation is not None:
            u = self.config.activation(u)
        if self.config.dropoutval:
            u = self.dropoutlayer(u)
        if self.config.activation is not None:
            u = self.input_proj2(u)
        ### print("LOG u after_proj shape " + str(u.shape))
        #u = u.transpose(0,1)
        #u = u.squeeze(1)
        ### print("LOG u after shape " + str(u.shape))
        if self.curr_state is None:
            self.reset_state()
        ### print("LOG u device" + str(u.device))
        ### print("LOG state device" + str(self.curr_state.device))
        inner_output, next_state = self.s4_mod.step(u, self.curr_state)
        self.curr_state = next_state
        #inner_output = inner_output.transpose(0,1)
        if self.config.layer_norm_s4:
            inner_output = self.output_norm_layer(inner_output)
        ret_y = self.output_projection(inner_output).unsqueeze(1)
        #print("LOG u ret_y shape: " + str(u.shape) + " X " + str(ret_y.shape))
        ### print("LOG state shape " + str(self.curr_state.shape))
        return ret_y, next_state

    # need to validate it in migpt.utils, if need to change
    def reset_state(self, device):
        self.curr_state = torch.zeros((1, self.h, self.n)).to(device=device, dtype=torch.cfloat)

    def get_block_size(self):
        return self.l_max

    ##added optimizer to match the DT original structure need to edit:
    def configure_optimizers(self, train_config):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        s4_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        # parameters that need to be configured:
        # 'kernel.krylov.C', 'output_linear.weight', 'kernel.krylov.w', 'kernel.krylov.B', 'D', 'kernel.krylov.log_dt'
        # original:
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, nn.Linear)
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        #S4_kernel_modules = (krylov)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name

                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)
                elif "s4_mod" in mn and self.s4_weight_decay>0:
                    s4_decay.add(fpn)
                elif "s4_mod" in mn and self.s4_weight_decay<=0:
                    no_decay.add(fpn)

        # special case the position embedding parameter in the root GPT module as not decayed
        #no_decay.add('pos_emb')
        #no_decay.add('global_pos_emb')
        #for r in ["s4_mod.kernel.krylov.C", "s4_mod.output_linear.weight", "s4_mod.kernel.krylov.w", "s4_mod.kernel.krylov.B", "s4_mod.D", "s4_mod.kernel.krylov.log_dt"]:
        #    if self.s4_weight_decay > 0:
        #        decay.add(r)
        #    else:
        #        no_decay.add(r)

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        union_params = decay | no_decay | s4_decay
        for d1, d2 in [(decay, no_decay), (decay, s4_decay), (s4_decay, decay)]:
            inter_params = decay & no_decay
            assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": train_config.weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(s4_decay))], "weight_decay": self.s4_weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=train_config.learning_rate, betas=train_config.betas)
        return optimizer

    def reprr(self):
        to_print = "Env state dimension: " + str(self.state_dim) + " X Internal size: " + str(self.n) + " X Interface size: " + str(
            self.h) + " X S4weight size: " + str(self.s4_weight_decay) + " X context length: " + str(
            self.l_max) + " X embedding size: " + str(self.input_emb_size)
        to_print += "\n"
        to_print += self.config.reprr()
        return to_print

###############################

##### Recurrent model:

class RNN_Block(nn.Module):
    def __init__(self, config, H, l_max, d_state, **kwargs):
        super().__init__()
        self.h = H
        self.config = config
        self.s4_mod_in = RNN_wrapper(input_size=self.h, hidden_size=d_state, batch_first=True)
        self.afterblock = nn.Sequential(
            nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity(),
            nn.GELU(),
            nn.Linear(self.h, 3 * self.h),
            nn.GELU(),
            nn.Linear(3 * self.h, self.h),
            nn.Dropout(self.config.dropoutval) if self.config.dropoutval>0 else nn.Identity(),
        )
        return

    def forward(self, u):
        y, next_state = self.s4_mod_in(u)
        if self.config.s4_resnet:
            y = y + 0.5 * u
        y = self.afterblock(y) + 0.5 * y
        return y, next_state

    def step(self, u, state):
        inner_output, new_state = self.s4_mod_in.step(u, state)
        if self.config.s4_resnet:
            inner_output = inner_output + 0.5 * u
        inner_output = self.afterblock(inner_output) + 0.5 * inner_output
        #### TEST
        #inner_output = self.normalizer(inner_output+u)
        return inner_output, state

class RNN_wrapper(nn.Module):
    def __init__(self,input_size, hidden_size, batch_first):
        super().__init__()
        self.hidden_size = hidden_size
        self.rnn_mod = nn.RNN(input_size=input_size, hidden_size=hidden_size, batch_first=batch_first)
        self.sizecorrector = nn.Linear(hidden_size, input_size)
        return

    def forward(self, u):
        input_state = self.default_state(u.shape[0]).to(device=u.device)

        output, next_state = self.rnn_mod(u, input_state)
        output = self.sizecorrector(output)
        return output, next_state

    def step(self,u ,state):
        #print(f"LOGX state  {state.shape}")
        #print(f"LOGX ushape {u.shape}")
        u = u.reshape(state.shape[0],-1,u.shape[-1])
        state = state.transpose(0,1).contiguous()
        output, next_state = self.rnn_mod(u, state); # squeeze
        output = self.sizecorrector(output)
        return output.reshape(-1, u.shape[-1]), next_state; #.unsqueeze(1)

    def setup_step(self):
        return

    def default_state(self, batchsize):
        return torch.zeros((1,batchsize,self.hidden_size));


class Linear_Block(nn.Module):
    def __init__(self, config, H, **kwargs):
        super().__init__()
        self.h = H
        self.s4_mod_in = S4_dummy()
        self.config = config
        self.afterblock = nn.Sequential(
            nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity(),
            nn.GELU(),
            nn.Linear(self.h, self.h),
            nn.GELU(),
            nn.Linear(self.h, self.h),
            nn.Dropout(self.config.dropoutval) if self.config.dropoutval>0 else nn.Identity(),
        )
        return

    def forward(self, u):
        y = self.afterblock(u) + u
        return y, None

    def step(self, u, state):
        y = self.afterblock(u) + u
        return y, state

class S4_dummy:
    def __init__(self):
        return
    def setup_step(self):
        return
    def default_state(self, batchsize):
        return torch.zeros((batchsize));


class S4_Block(nn.Module):
    """ an unassuming Transformer block """
    def __init__(self, config, H,
                 l_max, d_state, measure, dt_min, dt_max, rank, trainable, lr, weight_norm
                 ,s4mode='nplr'):
        super().__init__()
        self.h = H
        self.n = d_state
        self.config =config
        #self.beforeblock = nn.BatchNorm1d(self.h) if self.config.layer_norm_s4 else nn.Identity()
        self.beforeblock = nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity()
        self.afterblock = nn.Sequential(

            nn.GELU(),
            nn.Dropout(self.config.dropoutval) if self.config.dropoutval > 0 else nn.Identity(),
            nn.Linear(self.h, self.h),
            nn.GELU(),
            #nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity(),
            #nn.Linear(self.h, 3 * self.h),
            #nn.GELU(),
            #nn.Linear(3 * self.h, self.h),
            #nn.Tanh(),
        )
       #self.afterblock = nn.Sequential(
       #     nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity(),
       #     nn.GELU(),
       #     nn.Linear(self.h, 3 * self.h),
       #     nn.GELU(),
       #     nn.Linear(3 * self.h, self.h),
            #nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity(),
            #nn.Linear(self.h, 3 * self.h),
            #nn.GELU(),
            #nn.Linear(3 * self.h, self.h),
            #nn.Tanh(),
        #    nn.Dropout(self.config.dropoutval) if self.config.dropoutval>0 else nn.Identity(),
        #)
        self.l_max = l_max
        if self.config.single_step_val:
            l_max = None
        self.s4_mod_in = S4(H, l_max=l_max, d_state=d_state, measure=measure,
            dt_min=dt_min, dt_max=dt_max, rank=rank, trainable=trainable, lr=lr,
            weight_norm=weight_norm, linear=True, mode=s4mode, precision=self.config.precision, n_ssm=self.config.n_ssm,
        )
        #### TEST
        #self.normalizer = nn.LayerNorm(self.h)

    def forward(self, u):#takes only input, no hidden state
        # x: (B H L) if self.transposed else (B L H)
        # state: (H N)
        # Returns: same shape as x

        #y = u.transpose(-1, -2)
        y = u
        y = self.beforeblock(y)
        y = y.transpose(-1, -2)
        # self.config.base_model = "seq"
        if "seq" in self.config.base_model:
            self.s4_mod_in.setup_step()
            s4_state = self.s4_mod_in.default_state(y.shape[0]).to(device=y.device)
            out = []
            for i in range(y.shape[2]):
                yt, s4_state = self.s4_mod_in.step(y[:,:,i], s4_state)
                out.append(yt.unsqueeze(-1))
            y = torch.cat(out, dim=-1)
            next_state = s4_state
        else:
            # self.s4_mod_in.setup_step()
            # s4_state = self.s4_mod_in.default_state(y.shape[0]).to(device=y.device)
            y, next_state = self.s4_mod_in(y)# no state is being passed
        if self.config.train_noise>0:
            y = y + self.config.train_noise * torch.randn(y.shape, device=y.device)
        y = y.transpose(-1, -2)
        y = self.afterblock(y)
        if self.config.s4_resnet:
            y = y + u
        #### TEST
        #y = self.normalizer(y+u.transpose(-1, -2))
        return y, next_state

    def step(self, u, state):
        #print(f"LOGZZ u1 {u.shape}")
        y = self.beforeblock(u)
        #print(f"LOGZZ u2 {y.shape}")
        inner_output, new_state = self.s4_mod_in.step(y, state)
        #print(f"LOGZZ u3 {inner_output.shape}")
        inner_output = self.afterblock(inner_output)
        #print(f"LOGZZ u4 {inner_output.shape}")
        if self.config.s4_resnet:
            inner_output = inner_output + u
        #print(f"LOGZZ u5 {inner_output.shape}")
        #### TEST
        #inner_output = self.normalizer(inner_output+u)
        return inner_output, new_state


class S4_mujoco_wrapper_v3(S4_mujoco_wrapper):
    def __init__(
            self,
            config,
            state_dim,
            act_dim,
            max_length,
            n_embd=128,
            H=10,
            l_max=None,
            # Arguments for SSM Kernel
            d_state=64,
            measure='legs',
            dt_min=0.001,
            dt_max=0.1,
            rank=1,
            lr=None,
            stride=1,
            s4_weight_decay=0.0, # weight decay on the SS Kernel
            weight_norm=False,  # weight normalization on FF
            kernel_mode='nplr'
        ):
        """
        d_state: the dimension of the state, also denoted by N
        l_max: the maximum sequence length, also denoted by L
          if this is not known at model creation, or inconvenient to pass in,
          set l_max=None and length_correction=True
        dropout: standard dropout argument
        transposed: choose backbone axis ordering of (B, L, D) or (B, D, L) [B=batch size, L=sequence length, D=feature dimension]
        Other options are all experimental and should not need to be configured
        """
        TrajectoryModel.__init__(self, state_dim, act_dim, max_length=max_length)

        self.h = H
        self.state_dim = state_dim
        self.config = config
        self.s4_weight_decay = s4_weight_decay
        self.s4_mode = kernel_mode
        self.batch_mode = False

        ##Added for pre/post processing to inject the S4, taken as default from the DT:
        self.input_emb_size = n_embd
        self.n = d_state
        self.action_dim = act_dim
        self.l_max = None #not used

        if self.config.base_model == "ant_con":
            logging.info("Using 2 state dim enhancement")
            self.target_goal = nn.Parameter(torch.ones(2))
            #self.target_dict = [[1.2292455, -1.1228857], [[0.89684707, 1.5145522 ]], [-1.428631, 0.6352183]]
            self.state_dim += 2
        if self.config.base_model == "ant_reward_target":
            logging.info("Using ant reward target")
            self.output_projection_reward = nn.Linear(self.h, 1)

        # self.state_encoder = nn.Sequential(nn.Linear(self.state_dim, self.input_emb_size), nn.Tanh())
        # self.ret_emb = nn.Sequential(nn.Linear(1, self.input_emb_size), nn.Tanh())
        # self.action_embeddings = nn.Sequential(nn.Linear(self.action_dim, self.input_emb_size), nn.Tanh())
        # self.prev_action_embeddings = nn.Sequential(nn.Linear(self.action_dim*5, self.input_emb_size), nn.Tanh())

        self.state_encoder = nn.Linear(self.state_dim, self.input_emb_size)
        self.ret_emb = nn.Linear(1, self.input_emb_size)
        self.action_embeddings = nn.Linear(self.action_dim, self.input_emb_size)
        self.prev_action_embeddings = nn.Linear(self.action_dim, self.input_emb_size)
        # self.prev_action_embeddings = nn.Linear(self.action_dim*5, self.input_emb_size) #ablated versions
        #nn.init.normal_(self.action_embeddings[0].weight, mean=0.0, std=0.02)
        # self.input_projection = nn.Linear(self.input_emb_size*4+6, self.h)
        self.input_projection = nn.Linear(self.input_emb_size * (2+1) + 2 + 70, self.h)
        # self.input_projection = nn.Linear(self.input_emb_size * (3 + 5) + 6, self.h)
        # self.input_projection = nn.Linear(self.input_emb_size * 2 + 6 + 14 + 14 * 5, self.h)
        # self.input_projection = nn.Linear(self.input_emb_size * 3 + 6 + 14, self.h)
        # self.input_projection_ind = nn.Linear(self.input_emb_size * 3 + 6, self.h)
        # self.input_projection_ind = nn.Linear(self.input_emb_size * 2 + 6 + 14, self.h)
        self.beforeblock = nn.Sequential(
            nn.GELU(),
            nn.Linear(self.h, self.h),
            nn.Dropout(self.config.dropoutval),
        )
        # self.beforeblock = nn.Sequential(
        #     nn.GELU(),
        #     nn.Linear(390, self.h),
        #     nn.Dropout(self.config.dropoutval),
        # )
        # self.beforeblock = nn.LayerNorm(self.h) if self.config.layer_norm_s4 else nn.Identity()
        if self.config.s4_onpolicy:
            lr = 0.001
        #S4 configuration
        self.s4_amount = self.config.s4_layers
        trainable = None if self.config.s4_trainable else False
        if self.config.base_model == "lin":
            logging.info(f"S4 model abl: using linear core x {self.s4_amount}")
            self.s4_mods = nn.ModuleList([Linear_Block(self.config, H=H) for _ in range(self.s4_amount)])
        elif self.config.base_model == "rnn":
            logging.info(f"S4 model abl: using RNN core x {self.s4_amount}")
            self.s4_mods = nn.ModuleList([RNN_Block(self.config, H=H, l_max=l_max, d_state=d_state) for _ in
                                          range(self.s4_amount)])
        else:
            if self.config.base_model== "random":
                logging.info(f"S4 model abl: using S4 random core x {self.s4_amount}")
                logging.info(f"Setting measure to \"random\"")
                measure = "random"
            self.s4_mods = nn.ModuleList([S4_Block(self.config, H=H, l_max=l_max, d_state=d_state, measure=measure,
                                                   dt_min=dt_min, dt_max=dt_max, rank=rank, trainable=trainable, lr=lr,
                                                   weight_norm=weight_norm, s4mode=self.s4_mode) for _ in
                                          range(self.s4_amount)])
        if self.config.discrete > 0:
            #self.output_projection = nn.Linear(self.h, self.action_dim * self.config.discrete, bias=False)
            self.output_projection = nn.Linear(self.h, (self.action_dim + self.state_dim ) * self.config.discrete, bias=False)
            self.output_projection_rtg = nn.Linear(self.h, 1)
        else:
            self.output_projection = nn.Linear(self.h, self.action_dim, bias=False)#action logits

    def pre_val_setup(self):
        for mod in self.s4_mods:
            mod.s4_mod_in.setup_step()
        return

    def forward(self, states, actions, rewards, rtg, timestep, s4_states=None, running=False, cache=None, goals=None, target_goal=None, **kwargs): # absorbs return_output and transformer src mask
        """
        u: (B H L) if self.transposed else (B L H)
        state: (H N) never needed unless you know what you're doing: memory
        Returns: same shape as u
        """
        #### preprocess for the S4:
        # u (batch, l_max, 84*84*4) -> inputsize (84, 84, 4), (batch, l_max, 1), (batch, l_max, 1) - state, action, reward
        # print("model forward") here
        # print(self.batch_mode)#False
        if self.batch_mode:
            return self.step_forward(states, actions, rewards, rtg, timestep, s4_states=s4_states, running=running, **kwargs)
        del_r = 1
        n_agents = len(states)

        ret_act = [None]*n_agents
        ret_st = [None]*n_agents
        ret_rtg = [None]*n_agents
        prev_agent_act = [None]*n_agents
        if running:
            # for each agent
            del_r = 0
            # batch_size = states.shape[0]
            # sequence_len = states.shape[1]

        batch_size = states[0].shape[0]
        sequence_len = states[0].shape[1]


        for i in range(n_agents):

            # states[i] = torch.cat((states[i], one_hot_agent_id.to(device=states[i].device)), dim=-1)
            state_embed = self.state_encoder(states[i][:,del_r:,...].reshape(-1, self.state_dim).type(torch.float32).contiguous())
            one_hot_agent_id = F.one_hot(torch.tensor(i), num_classes=n_agents).unsqueeze(dim=0).unsqueeze(dim=0).expand(batch_size, sequence_len, n_agents).to(device=state_embed.device)
            # global_state_embed = self.state_encoder(states[i][:,del_r:,...].reshape(-1, self.state_dim).type(torch.float32).contiguous())
            # state_embed = torch.cat((state_embed, one_hot_agent_id[:,del_r:,...].reshape(-1, n_agents)), dim=-1)
            state_embed = torch.cat((one_hot_agent_id[:, del_r:, ...].reshape(-1, n_agents), state_embed), dim=-1)
            # state_embed = torch.cat((one_hot_agent_id[:, del_r:, ...].reshape(-1, n_agents), states[i][:,del_r:,...].reshape(-1, self.state_dim)), dim=-1)
            if i == 0:#1st agent has no previous agent action
                prev_agent_act[i] = None
            else:
                # prev_agent_act[i] = ret_act[i-1]
                prev_agent_act[i] = ret_act[:i]

            if prev_agent_act[i] == None:
                # prev_action_embed = torch.zeros_like(actions[i][:, :sequence_len - del_r, ...]).reshape(-1,self.action_dim)
                # prev_action_embed = self.prev_action_embeddings(torch.zeros_like(actions[i][:,:sequence_len-del_r,...], device=state_embed.device).reshape(-1, self.action_dim))
                # prev_action_embed = torch.zeros((state_embed.size(0), (n_agents-1)*self.act_dim), device=state_embed.device)
                prev_action_embed = torch.zeros((state_embed.size(0), (n_agents - 1) * self.input_emb_size),
                                                device=state_embed.device)
            else:
                # for k in range(len(prev_agent_act[i])):
                #     # prev_action_embed += prev_agent_act[i][k][:, :sequence_len - del_r, ...].reshape(-1, self.action_dim)
                #     prev_action_embed += self.prev_action_embeddings(prev_agent_act[i][k][:, :sequence_len - del_r, ...].reshape(-1, self.action_dim))
                # prev_action_embed = prev_action_embed/len(prev_agent_act[i])

                prev_action_embed_list = []
                for k in range(len(prev_agent_act[i])):
                    prev_action_embed = self.prev_action_embeddings(prev_agent_act[i][k][:, :sequence_len - del_r, ...].reshape(-1, self.action_dim))
                    prev_action_embed_list.append(prev_action_embed)

                concat_prev_action_embed = torch.cat(prev_action_embed_list, dim=-1)

                pad_size = (n_agents - 1) * self.input_emb_size - concat_prev_action_embed.size(-1)
                if pad_size > 0:
                    pad_tensor = torch.zeros((concat_prev_action_embed.size(0), pad_size),
                                             dtype=concat_prev_action_embed.dtype, device=concat_prev_action_embed.device)
                    concat_prev_action_embed = torch.cat([concat_prev_action_embed, pad_tensor], dim=-1)

                prev_action_embed = concat_prev_action_embed

                # concatenated_actions = torch.cat(prev_agent_act[i], dim=-1)
                #
                # # Pad concatenated actions with zeros if necessary
                # pad_size = (n_agents-1) * self.action_dim - concatenated_actions.size(-1)
                # if pad_size > 0:
                #     pad_tensor = torch.zeros((concatenated_actions.size(0), concatenated_actions.size(1), pad_size),
                #                              dtype=concatenated_actions.dtype, device=concatenated_actions.device)
                #     concatenated_actions = torch.cat([concatenated_actions, pad_tensor], dim=-1)
                #
                # # Update prev_action_embed
                # prev_action_embed = concatenated_actions.reshape(-1, concatenated_actions.size(-1))
            # prev_action_embed = self.prev_action_embeddings(prev_action_embed)

            if actions == None:
                # action_embed = self.action_embeddings(torch.zeros_like(rtg[i], device=state_embed.device).reshape(-1, 1))
                action_embed = torch.zeros_like(rtg[i], device=state_embed.device).reshape(-1, 1)
            else:
                # action_embed = self.action_embeddings(actions[i][:,:sequence_len-del_r,...].reshape(-1,self.action_dim))
                action_embed = actions[i][:, :sequence_len - del_r, ...].reshape(-1, self.action_dim)

            #reward_embed = self.ret_emb(rtg[:,del_r:,...].reshape(-1,1).type(torch.float32))
            reward_embed = self.ret_emb(rtg[i][:, :sequence_len-del_r, ...].reshape(-1, 1).type(torch.float32))
            action_embed = action_embed.reshape(batch_size * (sequence_len - del_r), -1)
            reward_embed = reward_embed.reshape(batch_size * (sequence_len - del_r), -1)

            # u = torch.cat([state_embed, prev_action_embed, action_embed, reward_embed], dim=-1)
            u = torch.cat([state_embed, action_embed, prev_action_embed, reward_embed], dim=-1)
            # u = torch.cat([state_embed, reward_embed, action_embed, prev_action_embed], dim=-1) # used before!
            # u = torch.cat([state_embed, action_embed, reward_embed], dim=-1)
            # d = 390
            u = self.input_projection(u).reshape(batch_size, sequence_len-del_r, self.h)
            #
            # u = self.input_projection_ind(u).reshape(batch_size, sequence_len - del_r, self.h)
            # u = u.reshape(batch_size, sequence_len-del_r, d)
            ret_y = self.beforeblock(u)
            # print(self.s4_mods)# 3 S4 blocks
            for mod in self.s4_mods:
                ret_y, hidden_state = mod(ret_y)
            ret_temp = self.output_projection(ret_y.reshape(-1,self.h))# action logits

            if self.config.discrete > 0:
                ret_temp = ret_temp.reshape(batch_size, sequence_len - del_r, self.action_dim + self.state_dim, self.config.discrete)
                ret_rtg = self.output_projection_rtg(ret_y.reshape(-1,self.h)).reshape(batch_size, sequence_len - del_r, 1)
                ret_st = ret_temp[:, :, self.action_dim:, :]
                ret_act = ret_temp[:, :, :self.action_dim, :]

            else:
                ret_act[i] = ret_temp.reshape(batch_size, sequence_len - del_r, self.action_dim)#action logits
                ret_st[i] = None
                ret_rtg[i] = None

            # if self.config.base_model == "ant_reward_target":
            #     ret_rtg = self.output_projection_reward(ret_y.reshape(-1,self.h)).reshape(batch_size, sequence_len - del_r, 1)
            # if "ant_con" in self.config.base_model:
            #     ret_st = self.target_goal.unsqueeze(0).unsqueeze(0).expand(batch_size, sequence_len - del_r, 2)
        return ret_st, ret_act, ret_rtg

    def get_action(self, states, actions, rewards, returns_to_go, timesteps, s4_states=None, running=False, targets=None, **kwargs):
        # get_action(self, states, actions, rewards, returns_to_go, timesteps, **kwargs)
        #assert not self.training

        # if self.config.base_model == "ant_con_old":
        #     last_st = states[-1,:2]
        #     self.target_goal = torch.as_tensor(self.target_dict[self.curr_target % len(self.target_dict)],
        #                                        dtype=states.dtype, device=states.device)
        #     diff = torch.sum(torch.pow(last_st - self.target_goal, 2))
        #     if diff < 0.9 / 6.8:
        #         logging.info(f"Antmaze change target {self.curr_target:2} -> {self.curr_target+1:2} . {self.target_dict[self.curr_target]}")
        #         self.curr_target = (self.curr_target + 1) % len(self.target_dict)
        #         self.target_goal = torch.as_tensor(self.target_dict[self.curr_target % len(self.target_dict)],
        #                                            dtype=states.dtype, device=states.device)
        #         s4_states = [r.detach() for r in self.get_initial_state((1), s4_states[0].device)]
        #     states = torch.cat([states, self.target_goal.reshape(1,2).expand(states.shape[0],2)], dim=-1)
        #
        # if self.config.base_model == "ant_con":
        #     states = torch.cat([states, self.target_goal.reshape(1,2).expand(states.shape[0],2)], dim=-1)

        # states = states.reshape(1, -1, self.state_dim)
        # actions = actions.reshape(1, -1, self.act_dim)
        # returns_to_go = returns_to_go.reshape(1, -1, 1)
        # timesteps = timesteps.reshape(1, -1)


        if not self.config.single_step_val:
            if self.l_max is not None:
                states = states[:, -self.l_max:]
                actions = actions[:, -self.l_max:]
                returns_to_go = returns_to_go[:, -self.l_max:]
                timesteps = timesteps[:, -self.l_max:]
            actions_ = self.forward(states, actions, rewards, returns_to_go, timesteps, running=True, **kwargs)[1]
            actions = [a[0,-1,:] for a in actions_]# return the actions ins the current timestep
            return actions
            # return actions_ # return the whole sequence


            # return self.forward(states, actions, rewards, returns_to_go, timesteps, running=True, **kwargs)[1][0,-1,:]#actions

        # run single step. need to reconfigure
        state_embed = self.state_encoder(states[:, -1, ...].reshape(-1, self.state_dim).type(torch.float32).contiguous())
        if actions == None:
            action_embed = self.action_embeddings(
                torch.zeros_like(returns_to_go, device=state_embed.device)[:, -1, ...].reshape(-1, self.action_dim))
        else:
            action_embed = self.action_embeddings(actions[:, -1, ...].reshape(-1, self.action_dim))
        reward_embed = self.ret_emb(returns_to_go[:, -1, ...].reshape(-1, 1).type(torch.float32))
        u = torch.cat([state_embed, action_embed, reward_embed], dim=-1)
        u = self.input_projection(u)
        ret_y = self.beforeblock(u)
        output_states = []
        for i, mod in enumerate(self.s4_mods):
            input_state = None
            if s4_states is not None:
                input_state = s4_states[i]
            ret_y, new_state = mod.step(ret_y, input_state)
            output_states.append(new_state)

        if self.config.discrete > 0:
            ret_rtg = self.output_projection_rtg(ret_y).reshape(1, -1, 1)
            ret_y = self.output_projection(ret_y).reshape(1, -1, self.action_dim + self.state_dim, self.config.discrete)
            return [ret_y, ret_rtg], output_states
        else:
            ret_act = self.output_projection(ret_y).unsqueeze(1)
            if self.config.base_model == "ant_reward_target":
                ret_target = self.output_projection_reward(ret_y).unsqueeze(1)
                return [ret_act[0, -1, :], ret_target], output_states
            return ret_act[0, -1, :], output_states

    def step_forward(self, states, actions, rewards, returns_to_go, timesteps, s4_states=None, running=False, **kwargs):

        states = states.unsqueeze(1)
        actions = actions.unsqueeze(1)
        returns_to_go = returns_to_go.reshape(-1, 1, 1)
        # print("here")
        # run single step. need to reconfigure
        state_embed = self.state_encoder(states[:, -1, ...].reshape(-1, self.state_dim).type(torch.float32).contiguous())
        if actions == None:
            action_embed = self.action_embeddings(
                torch.zeros_like(returns_to_go, device=state_embed.device)[:, -1, ...].reshape(-1, self.action_dim))
        else:
            action_embed = self.action_embeddings(actions[:, -1, ...].reshape(-1, self.action_dim))
        reward_embed = self.ret_emb(returns_to_go[:, -1, ...].reshape(-1, 1).type(torch.float32))
        u = torch.cat([state_embed, action_embed, reward_embed], dim=-1)
        u = self.input_projection(u)
        ret_y = self.beforeblock(u)
        output_states = []
        for i, mod in enumerate(self.s4_mods):
            input_state = None
            if s4_states is not None:
                input_state = s4_states[i]
                #print(f"LOGXX {i} realsizes: {self.h:3}x{self.n:3}")
                #print(f"LOGXX {i} actlsizes: {ret_y.shape} over {input_state.shape}")
            ret_y, new_state = mod.step(ret_y, input_state)
            output_states.append(new_state)

        ret_y = self.output_projection(ret_y)
        return ret_y, output_states

    def get_initial_state(self, batchsize, device='cpu'):
        if not self.config.single_step_val:
            return None

        return [mod.s4_mod_in.default_state(batchsize).to(device=device) for mod in self.s4_mods]

    def get_block_size(self):
        return self.l_max

    ##added optimizer to match the DT original structure need to edit:
    def get_optim_group(self, lr, all_decay_rate):

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        s4_decay = set()
        # whitelist_weight_modules = (torch.nn.Linear, )
        # parameters that need to be configured:
        # 'kernel.krylov.C', 'output_linear.weight', 'kernel.krylov.w', 'kernel.krylov.B', 'D', 'kernel.krylov.log_dt'
        # original:
        whitelist_weight_modules = (torch.nn.Linear, torch.nn.Conv2d, nn.Linear)
        blacklist_weight_modules = (S4)
        #S4_kernel_modules = (krylov)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                #if pn.endswith('bias'):
                #    # all biases will not be decayed
                #    no_decay.add(fpn)
                #if isinstance(m, blacklist_weight_modules):
                if "kernel" in fpn:
                    # weights of whitelist modules will be weight decayed
                    no_decay.add(fpn)
                    print(f"nod {mn:40} paramn {pn:50}")
                else:
                    # weights of blacklist modules will NOT be weight decayed
                    decay.add(fpn)
                    print(f"yod {mn:40} paramn {pn:50}")

        param_dict = {pn: p for pn, p in self.named_parameters()}
        union_params = decay | no_decay
        inter_params = decay & no_decay
        print(f"no decay: {no_decay}")
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params),)
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": all_decay_rate},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        optimizer = torch.optim.AdamW(optim_groups, lr=lr)
        return optimizer

############################################################################################################
############################################################################################################
# Critic model for online learning:

class FC_critic_resnet(nn.Module):
    def __init__(self, states_dim=5, action_dim=3, state_enc_layers=2,
                 action_enc_layers=2, mutual_layers=3, state_enc_size=10, action_enc_size=8, mutual_enc_size=10):
        super(FC_critic_resnet, self).__init__()

        self.state_projection = nn.Linear(states_dim, state_enc_size)
        self.action_projection = nn.Linear(action_dim, action_enc_size)

        self.state_enc = nn.Sequential(nn.GELU(),
                                       nn.Linear(state_enc_size, state_enc_size*2),
                                       nn.GELU(),
                                       nn.Linear(state_enc_size*2, state_enc_size)
                                       )
        self.action_enc = nn.Sequential(nn.GELU(),
                                        nn.Linear(action_enc_size, action_enc_size),
                                        )
        self.state_action_to_mutual = nn.Sequential(nn.Linear(state_enc_size + action_enc_size, mutual_enc_size), nn.Tanh())
        self.mutual_enc = nn.Sequential(nn.Linear(mutual_enc_size, mutual_enc_size),
                                        nn.Tanh())
        self.output_projection = nn.Linear(mutual_enc_size, 1)

        return

    def forward(self, states, actions):
        stateforward = self.state_projection(states)
        actionforward = self.action_projection(actions)

        stateforward = self.state_enc(stateforward) + stateforward
        actionforward = self.action_enc(actionforward) + actionforward

        mutualforward = self.state_action_to_mutual(torch.cat([stateforward, actionforward], dim=-1))
        mutualforward = self.mutual_enc(mutualforward) + mutualforward
        return self.output_projection(mutualforward)

class FC_critic_shallowA_diff(nn.Module):
    def __init__(self, states_dim=5, action_dim=3, state_enc_layers=2, action_pass=True,
                 action_enc_layers=2, mutual_layers=3, state_enc_size=10, action_enc_size=8, mutual_enc_size=10):
        super(FC_critic_shallowA_diff, self).__init__()
        self.action_pass = action_pass

        self.state_projection = nn.Linear(states_dim, state_enc_size)
        self.action_projection = nn.Linear(action_dim, action_enc_size*2)

        self.state_enc = nn.Sequential(nn.GELU(),
                                       nn.Linear(state_enc_size, state_enc_size * 2),
                                       nn.GELU(),
                                       )
        self.action_enc = nn.Sequential(nn.GELU())

        self.state_action_to_mutual = nn.Linear(state_enc_size*2 + action_enc_size*2, mutual_enc_size)
        self.mutual_enc = nn.Sequential(nn.Tanh(),
                                        nn.Linear(mutual_enc_size, mutual_enc_size))
        self.mutual_end = nn.Linear(mutual_enc_size, 1)

        return

    def forward(self, states, actions):
        stateforward = self.state_projection(states)
        actionforward = self.action_projection(actions)

        stateforward = self.state_enc(stateforward)

        actionforward = self.action_enc(actionforward)

        mutualforward = self.state_action_to_mutual(torch.cat([stateforward, actionforward], dim=-1))
        if self.action_pass:
            mutualforward = self.mutual_enc(mutualforward) + mutualforward
        else:
            mutualforward = self.mutual_enc(mutualforward)
        return self.mutual_end(mutualforward)

class FC_critic_shallowA(nn.Module):
    def __init__(self, states_dim=5, action_dim=3, state_enc_layers=2,
                 action_enc_layers=2, mutual_layers=3, state_enc_size=10, action_enc_size=8, mutual_enc_size=10):
        super(FC_critic_shallowA, self).__init__()

        self.state_projection = nn.Linear(states_dim, state_enc_size)
        self.action_projection = nn.Linear(action_dim, action_enc_size*2)

        self.state_enc = nn.Sequential(nn.GELU(),
                                       nn.Linear(state_enc_size, state_enc_size*2),
                                       nn.GELU(),
                                       nn.Linear(state_enc_size*2, state_enc_size*2),
                                       nn.GELU(),
                                       )
        self.action_enc = nn.Sequential(nn.GELU())

        self.state_action_to_mutual = nn.Linear(state_enc_size*2 + action_enc_size*2, mutual_enc_size)
        self.mutual_enc = nn.Sequential(nn.Tanh(),
                                        nn.Linear(mutual_enc_size, mutual_enc_size//2),
                                        nn.Tanh(),
                                        nn.Linear(mutual_enc_size//2, 1))

        return

    def forward(self, states, actions):
        stateforward = self.state_projection(states)
        actionforward = self.action_projection(actions)

        stateforward = self.state_enc(stateforward)
        actionforward = self.action_enc(actionforward)

        mutualforward = self.state_action_to_mutual(torch.cat([stateforward, actionforward], dim=-1))
        mutualforward = self.mutual_enc(mutualforward)
        return mutualforward

class FC_critic_cat(nn.Module):
    def __init__(self, states_dim=5, action_dim=3, state_enc_layers=2,
                 action_enc_layers=2, mutual_layers=3, state_enc_size=10, action_enc_size=8, mutual_enc_size=10):
        super(FC_critic_cat, self).__init__()

        self.fc1 = nn.Linear(states_dim+action_dim, states_dim+action_dim)
        self.act1 = nn.GELU()
        self.fc2 = nn.Linear(states_dim+action_dim, states_dim+action_dim)
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(states_dim+action_dim, states_dim+action_dim)
        self.act3 = nn.Tanh()
        self.fc4 = nn.Linear(states_dim+action_dim, 1)
        return

    def forward(self, states, actions):
        y = self.fc1(torch.cat([states, actions], dim=-1))
        y = self.act1(y)
        y = self.fc2(y) + y
        y = self.act2(y)
        y = self.fc3(y) + y
        y = self.act3(y)
        return self.fc4(y)

class FC_critic_cat_expanded(nn.Module):
    def __init__(self, states_dim=5, action_dim=3, state_enc_layers=2,
                 action_enc_layers=2, mutual_layers=3, state_enc_size=10, action_enc_size=8, mutual_enc_size=10):
        super(FC_critic_cat_expanded, self).__init__()
        self.layersize = states_dim+action_dim
        self.fc1 = nn.Linear(self.layersize, self.layersize)
        self.act1 = nn.GELU()
        self.fc2 = nn.Sequential(nn.Linear(self.layersize, 2*self.layersize),
                                 nn.GELU(),
                                 nn.Linear(2*self.layersize, self.layersize))
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(self.layersize, 1)
        return

    def forward(self, states, actions):
        y = self.fc1(torch.cat([states, actions], dim=-1))
        y = self.act1(y)
        y = self.fc2(y) + y
        y = self.act2(y)
        return self.fc3(y)

class FC_critic_cat_expanded_rtg(nn.Module):
    def __init__(self, states_dim=5, action_dim=3, state_enc_layers=2,
                 action_enc_layers=2, mutual_layers=3, state_enc_size=10, action_enc_size=8, mutual_enc_size=10):
        super(FC_critic_cat_expanded_rtg, self).__init__()
        self.layersize = states_dim+action_dim+1
        self.fc1 = nn.Linear(self.layersize, self.layersize)
        self.act1 = nn.GELU()
        self.fc2 = nn.Sequential(nn.Linear(self.layersize, 2*self.layersize),
                                 nn.GELU(),
                                 nn.Linear(2*self.layersize, self.layersize))
        self.act2 = nn.GELU()
        self.fc3 = nn.Linear(self.layersize, 1)
        return

    def forward(self, states, actions, rtg):
        rtg = rtg.reshape(-1,1)
        inp = torch.cat([states, actions, rtg], dim=-1)
        y = self.fc1(inp) + inp
        y = self.act1(y)
        y = self.fc2(y) + y
        y = self.act2(y)
        return self.fc3(y)

if __name__ == '__main__':
    print(f"inside s4_muj")
    config = S4_config(layer_norm_s4=True,
                       single_step_val=True,
                       s4_layers=3,
                       precision=1,
                       s4_resnet=True,
                       base_model='s4')
    action_size = 12
    state_size = 13
    seq_len = 1000

    model = S4_mujoco_wrapper_v3(
            config, state_size, action_size, 1,
            n_embd=7, H=13,
            d_state=17,
            kernel_mode='diag')
    model.eval()
    u = torch.rand((1,seq_len,state_size+action_size+1))
    _, out1, _ = model(u[:,:,0:state_size], u[:,:,state_size:-1], None, u[:,:,-1], None, running=True)
    ###
    out2 = torch.zeros(1,seq_len,action_size)
    model.pre_val_setup()
    s4_states = [r.detach() for r in model.get_initial_state((1), "cpu")]
    for x in range(1,1+seq_len):
        z, s4_states = model.get_action(u[:,:x,0:state_size], u[:,:x,state_size:-1], None, u[:,:x,-1], torch.zeros_like(u[:,:x,-1]), s4_states)
        out2[:,x-1,:] = z
    print("#"*15 +f"Diff out1| out2" + "#"*15)
    print(f"Sizes: out1 {out1.shape} | out2 {out2.shape}")
    totnumbers = (out1.shape[1] * out1.shape[2])
    print(f"Average Diff  L2: {torch.sum(torch.pow(out1 - out2, 2)) / totnumbers}")
    print(f"Average Diff  L1: {torch.sum(torch.abs(out1 - out2)) / totnumbers}")
    print(f"Average first L1: {torch.sum(torch.abs(out1 - out2)[0,0,:]) / out1.shape[2]}")
    print(f"Average last  L1: {torch.sum(torch.abs(out1 - out2)[0,-1,:]) / out1.shape[2]}")
    #print(f"Diff enlarged:\n{out1 - out2}")
    print(f"#"*100)
    #print(f"Out1:\n{out1}")
    #print(f"Out2:\n{out2}")

    ##########

    batch_size = 5
    u = torch.rand((batch_size,seq_len,state_size+action_size+1))

    _, out1, _ = model(u[:,:,0:state_size], u[:,:,state_size:-1], None, u[:,:,-1], None, running=False)
    config.recurrent_mode = True
    _, out3, _ = model(u[:,:,0:state_size], u[:,:,state_size:-1], torch.zeros_like(u[:,:,state_size:-1]), u[:,:,-1], torch.zeros_like(u[:,:,state_size:-1]), running=False)
    config.recurrent_mode = False
    ###

    print("#"*15 +f"Diff out1| out3 EVALL" + "#"*15)
    print(f"Sizes: out1 {out1.shape} | out3 {out3.shape}")
    totnumbers = (out1.shape[1] * out1.shape[2])
    print(f"Average Diff  L2: {torch.sum(torch.pow(out1 - out3, 2)) / totnumbers}")
    print(f"Average Diff  L1: {torch.sum(torch.abs(out1 - out3)) / totnumbers}")
    print(f"Average first L1: {torch.sum(torch.abs(out1 - out3)[0,0,:]) / out1.shape[2]}")
    print(f"Average last  L1: {torch.sum(torch.abs(out1 - out3)[0,-1,:]) / out1.shape[2]}")
    #print(f"Diff enlarged:\n{out1 - out2}")
    print(f"#"*100)

    ###########

    batch_size = 5
    u = torch.rand((batch_size,seq_len,state_size+action_size+1))
    u2 = u.clone()

    model.train()
    _, out1, _ = model(u[:,:,0:state_size], u[:,:,state_size:-1], None, u[:,:,-1], None, running=False)
    config.recurrent_mode = True
    _, out3, _ = model(u2[:,:,0:state_size], u2[:,:,state_size:-1], torch.zeros_like(u2[:,:,state_size:-1]), u2[:,:,-1], torch.zeros_like(u2[:,:,state_size:-1]), running=False)
    config.recurrent_mode = False
    ###

    print("#"*15 +f"Diff out1| out3 TRAIN" + "#"*15)
    print(f"Sizes: out1 {out1.shape} | out3 {out3.shape}")
    totnumbers = (out1.shape[1] * out1.shape[2])
    print(f"Average Diff  L2: {torch.sum(torch.pow(out1 - out3, 2)) / totnumbers}")
    print(f"Average Diff  L1: {torch.sum(torch.abs(out1 - out3)) / totnumbers}")
    print(f"Average first L1: {torch.sum(torch.abs(out1 - out3)[0,0,:]) / out1.shape[2]}")
    print(f"Average last  L1: {torch.sum(torch.abs(out1 - out3)[0,-1,:]) / out1.shape[2]}")
    #print(f"Diff enlarged:\n{out1 - out2}")
    print(f"#"*100)