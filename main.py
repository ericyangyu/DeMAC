import argparse
from multiprocessing import Process
from pathlib import Path

import yaml

from demac.src.demac.demac_agent_env_wrapper import AgentEnvWrapper
from demac.src.demac.demac_coordinator import Coordinator
from sample_envs.callbacks.agent_env_logging_callbacks import AgentEnvLoggingCallback
from sample_envs.gridnav.grid import Grid
from sample_envs.gridnav.gridnav import GridNav
from sample_envs.meteor.meteor_env import MeteorEnv
from sample_envs.trivial.trivial_env import TrivialEnv

from stable_baselines3 import A2C
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import CheckpointCallback

parser = argparse.ArgumentParser(description='Check if we are training or testing.')
parser.add_argument('--test', dest='model_name', type=str, default=None)
parser.add_argument('--env', dest='env', type=str, default='trivial')
parser.add_argument('--exp_path', dest='exp_path', type=str, default='exp0/')
args = parser.parse_args()

# Map the correct sample environment to user argument
env_map = {
    'trivial': ('sample_envs/trivial', TrivialEnv),
    'gridnav': ('sample_envs/gridnav', Grid) if not args.model_name else ('sample_envs/gridnav', GridNav),
    'meteor': ('sample_envs/meteor', MeteorEnv),
}
env_path, env = env_map[args.env]
configs = yaml.load(open(env_path + '/config/config.yaml', 'r'), Loader=yaml.SafeLoader)
env = env()

# Start up the DeMAC coordinator, linking the shared environment to the coordinator
coordinator = Coordinator(env, exp_path=args.exp_path, test=args.model_name is not None)

# Initialize agents and their wrapper environments
envs, agents = [], []
for i in range(configs['num_agents']):
    envs.append(AgentEnvWrapper(str(i), coordinator))
    if args.model_name:
        # Load an existing agent model
        agents.append(A2C.load(f'./{args.exp_path}/{str(i)}/models/{args.model_name}', envs[i], device='cpu'))
    else:
        # Initialize a new agent model
        agents.append(A2C(MlpPolicy, envs[i], verbose=1, device='cpu'))

# Start up the coordinator server to start listening for agent requests
coordinator.start()

if args.model_name:
    # Evaluate the given model
    env.evaluate(num_eps=1000, agents=agents, envs=envs)
else:
    # Begin agent learning from scratch
    for i, agent in enumerate(agents):
        model_path = Path(f'./{args.exp_path}/{i}/models/')
        model_path.mkdir(parents=True)

        checkpoint_callback = CheckpointCallback(save_freq=1e4, save_path=model_path,
                                                 name_prefix='rl_model')
        logging_callback = AgentEnvLoggingCallback(env_wrapper=envs[i])
        p = Process(target=agent.learn, args=(1e7,), kwargs={'callback': [logging_callback, checkpoint_callback]})
        p.start()
