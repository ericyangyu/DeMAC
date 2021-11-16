import gym

from demac.src.demac.demac_agent_env_wrapper import AgentEnvWrapper


class MARLEnvInterface(gym.Env):
    names = []  # must init a list of names in the order of the agents
    observation_space = {}  # {name: obs space}
    action_space = {}  # {name: act space}
    agent_envs = {}  # {name: wrapper env}
    render = False  # whether to render the environment
    spec = None  # ignore me

    def reset(self):
        pass

    def step(self, action):
        pass

    def render(self):
        pass

    def close(self):
        pass

    # DO NOT TOUCH: allows us to query for attributes based on name
    def getattribute(self, agent_name, attr):
        return object.__getattribute__(self, attr)[agent_name]

    def add_agent_env(self, agent_name: str, env: AgentEnvWrapper):
        self.agent_envs[agent_name] = env
