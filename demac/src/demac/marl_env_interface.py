"""The interface for defining the shared environment, which is what the user must extend for the coordinator to properly
interact with the shared environment.

This interface outlines the necessary instance variables and functions that the user must define for the coordinator
to properly interact with the shared environment. The user must extend this interface and implement the following
in order to properly use the DeMAC framework. There are examples provided for the user to see how to properly implement
this interface (see the DeMAC README for more information).

Usage example:
    See the trivial environment (more information on the DeMAC README)
"""
import gym

from demac.src.demac.demac_agent_env_wrapper import AgentEnvWrapper


class MARLEnvInterface(gym.Env):
    names = []  # A list of names in the order of the agents (order matters here!)
    observation_space = {}  # {agent name: obs space}
    action_space = {}  # {agent name: act space}
    agent_envs = {}  # {agent name: wrapper env}
    render = False  # Whether to render the environment

    def reset(self):
        pass

    def step(self, action):
        pass

    def close(self):
        pass

    # Links the shared environment to all the agent wrapper environments. Can change but not recommended.
    def add_agent_env(self, agent_name: str, env: AgentEnvWrapper):
        self.agent_envs[agent_name] = env

    # DO NOT TOUCH: Allows us to query for attributes based on name
    def _getattribute(self, agent_name, attr):
        return object.__getattribute__(self, attr)[agent_name]
