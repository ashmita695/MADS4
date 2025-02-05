from utils import load_data

data_dir = './6h_vs_8z/6h_vs_8z_1000/good/'
bias = 0
num_step = 20
episodes = 1000
num_agents = 6
global_states, local_obss, actions, done_idxs, rewards, time_steps, next_global_states, next_local_obss, \
                next_available_actions, _ = load_data(int(episodes), bias, data_dir, n_agents=num_agents)

print(type(global_states))#list
print(len(global_states))#38003 total steps with 1000 episodes
print(type(global_states[0]))
print(len(global_states[0]))# num agents
# print(f"states shape:{global_states.shape}")