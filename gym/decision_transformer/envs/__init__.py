from functools import partial
from .multiagentenv import MultiAgentEnv
from .warehouse.warehouse_env import WareHouseEnv

import sys
import os


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


REGISTRY = {

    "warehouse": partial(env_fn, env=WareHouseEnv)
}