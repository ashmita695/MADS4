# create multi-agent dataset as .pkl files for learning in S4 with fixed max timesteps per agent trajectory as 50(?) for smac
import pickle
from utils import load_data

data_dir = './AlberDICE Dataset (Public)/warehouse/rware-tiny/rware-tiny-4ag-easy-v4-expert'
file_path = f'./rware_tiny/rware_tiny_1000/expert/rware_tiny_4ag_easy_v4_expert.pkl'


episodes = 1000
num_agents = 4
data = load_data(int(episodes), data_dir, n_agents=num_agents)


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