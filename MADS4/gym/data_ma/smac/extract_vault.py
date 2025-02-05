import jax
import jax.numpy as jnp
import flashbax as fbx
from flashbax.vault import Vault


vlt = Vault(rel_dir="vaults/smac_v1", vault_name="3m.vlt", vault_uid="Good")
all_data = vlt.read()
offline_data = all_data.experience

jax.tree_map(lambda x: x.shape, offline_data)
print('yes')

terminals = offline_data['terminals']
print(len(terminals[0]))