import numpy as np
import matplotlib.pyplot as plt
import json
import os

data_dir = './AlberDICE Dataset (Public)/warehouse/rware-tiny/rware-tiny-2ag-easy-v4-expert'

# Load the .npy files
actions = np.load(os.path.join(data_dir, 'actions.npy')) # 1000, 501, 6, 1
actions_onehot = np.load(os.path.join(data_dir, 'actions_onehot.npy')) #1000, 501, 6, 5
avail_actions = np.load(os.path.join(data_dir, 'avail_actions.npy')) #1000, 501, 6, 5
filled = np.load(os.path.join(data_dir, 'filled.npy')) #1000, 501, 1
obs = np.load(os.path.join(data_dir, 'obs.npy')) # 1000, 501, 6, 71
reward = np.load(os.path.join(data_dir, 'reward.npy')) #1000, 501, 1
state = np.load(os.path.join(data_dir, 'state.npy')) #1000, 501, 42
terminated = np.load(os.path.join(data_dir, 'terminated.npy')) #1000, 501, 1

# # Load the meta.json file
# with open('meta.json') as f:
#     meta = json.load(f)

# Example visualizations

# 1. Visualize actions (for instance, the first 100 actions)
plt.figure(figsize=(10, 5))
plt.plot(actions[:100])
plt.title('Actions (First 100)')
plt.xlabel('Timestep')
plt.ylabel('Action Value')
plt.show()

# 2. Visualize rewards (for instance, the first 100 timesteps)
plt.figure(figsize=(10, 5))
plt.plot(reward[:100])
plt.title('Rewards (First 100)')
plt.xlabel('Timestep')
plt.ylabel('Reward Value')
plt.show()

# 3. Visualize the state or observations (assuming the data is a matrix)
# Visualize the first observation/state
plt.imshow(obs[0], cmap='viridis')
plt.title('First Observation')
plt.colorbar()
plt.show()

# 4. Print the metadata from the meta.json file
print("Metadata from meta.json:")
print(json.dumps(meta, indent=4))
