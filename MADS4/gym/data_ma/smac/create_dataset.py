# create multi-agent dataset as .pkl files for learning in S4 with fixed max timesteps per agent trajectory as 50(?) for smac
import pickle
from utils import load_data

data_dir = './6h_vs_8z/6h_vs_8z_1000/good/' #2c_vs_64zg poor has less than 1000 episodes!!
file_path = f'./6h_vs_8z/6h_vs_8z_1000/good/6h_vs_8z.pkl'
bias = 0
# num_step = 20
episodes = 1000
num_agents = 6
data = load_data(int(episodes), bias, data_dir, n_agents=num_agents)


print(type(data))
print(len(data))
print(type(data[0]))

keys = data[0].keys()
print(keys)

for key, value in data[0].items():
    # print(key, ':', value)
    print(f"type of {key}:{type(value)}")
    print(f"len of {key}:{len(value)}")# number of agents
    for i in range(len(value)):
        print(f"shape of {key}{i}: {value[i].shape}")

with open(file_path, 'wb') as f:
    pickle.dump(data, f)