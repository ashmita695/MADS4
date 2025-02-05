import pickle

# Load the object from the .pkl file
with open('walker2d-medium-v2.pkl', 'rb') as f:
    data = pickle.load(f)

# Now you can inspect the loaded object
print(type(data))

print(len(data))

print(type(data[0]))

keys = data[0].keys()
print(keys)

for key, value in data[0].items():
    # print(key, ':', value)
    print(f"shape of {key}:{value.shape}")