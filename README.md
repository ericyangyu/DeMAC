# DeMAC Framework

The Decentralized Multi-Agent Coordination (DeMAC) Framework is a lightweight tool designed to easily coordinate multiple agents with decentralized policies in a shared multi-agent environment. This framework targets anyone looking to quickly set up a robust multi-agent environment and train/test workflow. 

This open source project is currently in its infancy, and certainly has many issues. Thus, any work to extend this project's functionality is welcome.

## How it works

In single agent scenarios, the agent-environment interaction process is very well-defined: the agent sends a request to the environment, the environment sends a response to the agent. The agent has full control for stepping and resetting the environment. However, in a multi-agent scenario where there are many agents to one shared environment, this workflow is not as trivial. Some issues include how to coordinate agent requests with the environment (no one agent has full control over the environment anymore), how the shared environment should handle an arbitrary number of agents, etc. That is where the DeMAC framework comes in. 

![](images/demac_workflow.png?raw=true)

On a high level, the DeMAC framework works to formalize the agent-environment interaction process in a multi-agent setting. We can achieve this by having a coordinator to act as the middle man between the agents and environment. This builds the illusion that each agent exists in a single agent setting, i.e. a 1-1 mapping between agent and environment, which reduces complexity. The DeMAC workflow can be visualized in the diagram above, but to elaborate on the steps:

1. The coordinator sets up a message queue server, in which agents can publish individual requests to the coordinator. The coordinator will collect the individual agent requests, while the agents wait for a response.
2. Once the coordinator receives a request from each agent, it will send a batch request to the shared environment. The current iteration of DeMAC assumes that requests are either only step or reset, where reset from a single agent overrides all other agent requests (i.e. step), but this can be changed. The coordinator then receives a batch response from the shared environment. 
3. The coordinator extracts out individual agent responses from the shared environment's batch response, to which the coordinator maps each response to the corresponding agent's callback message queue.
4. Each agent consumes their response from their respective callback message queue, which then they extract and return that response to the original query.

Note that each agent only has to worry about sending a request and getting a response; in other words, each agent is given the illusion of existing in a single agent setting. The request/response workflow for each agent is fully abstracted with the use of wrapper agent environments, in that each agent is fed a wrapper environment rather than the actual environment. Then, any requests the agent makes passes through the wrapper environment, which is designed to interact with the coordinator and delegate/propagate requests/responses. 

Since the coordinator of the DeMAC framework acts as a middle man, the user is responsible for defining the learning algorithms, agents, and coordinator on the client-side, and shared environment on the server-side. It is important that the shared environment extends the [**MARLEnvInterface**](./demac/src/demac/marl_env_interface.py) interface, and individual agents only have access to a wrapper environment instance defined by [**AgentEnvWrapper**](./demac/src/demac/demac_agent_env_wrapper.py). Both the interface and wrapper environment classes use [gym](https://gym.openai.com/) because it is a great toolkit to build well-defined environments. A more detailed guide to coding with the DeMAC framework can be seen in the following section.

## How to begin using DeMAC

First, create a shared environment that extends the [**MARLEnvInterface**](./demac/src/demac/marl_env_interface.py) interface, initializing any instance variables or functions in the interface. Then, create a main file similar to [main.py](./main.py) that defines the workflow for learning/evaluating multiple agents. 

The [sample envs](./sample_envs) folder gives a couple examples ([trivial](./sample_envs/trivial/), [meteor](./sample_envs/meteor), [gridnav](./sample_envs/gridnav)) for how to define the shared environment. [main.py](./main.py) gives a good example for how to set up the DeMAC workflow (e.g. initializing the coordinator, linking the coordinator to the agents, initializing each agent with a wrapper environment, etc.) and define the learning/evaluating workflow.

## Installation
To install, first install the following dependencies:
* [Docker](https://docs.docker.com/engine/install/)
* [RabbitMQ](https://rabbitmq.com/download.html)
* [Anaconda](https://docs.anaconda.com/anaconda/install/) (Optional)

Docker will help us install RabbitMQ, which is how we manage our message queue system in DeMAC. Anaconda is an optional dependency to set up a virtual environment with the instructions below.
  
Assuming you have the 3 dependencies above, setting up your virtual environment is simple. Simply run the following:

```
conda create -n demac python=3.7
conda activate demac
pip install -r demac/requirements.txt
```

If you do not want to install Anaconda (a very large but useful package!), and prefer to use Python's built-in virtual environment system, run the following:
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Usage
For the given repo, we will use the sample env `MeteorEnv` as an example of how to use DeMAC.

To train:
```
python main.py --env meteor
```

To test:
```
python main.py --env meteor --test <model file name>
```

## Sample Environments
### Trivial
A minimum implementation of a shared environment using the DeMAC framework. This contrived example can be used as a basic reference for how to get set up with the shared environment, and run quick tests. 

### Meteor
![Alt Text](images/meteor.gif)

This is a game where a group of agents must avoid falling meteors from the sky. Agents live on the last row of an NxN grid, and can observe meteors falling in intervals from the top row. The collective goal of the agents is to avoid the meteors for as long as possible.

Agents are able to move left and right, but are not on top of each other. Hence, the challenge is for the agents to learn cooperative behavior (e.g. moving for another agent to dodge a meteor even if the current agent is out of harm's way) to maximize the collective return.

### GridNav
![Alt Text](images/gridnav.gif)

This task involves a group of agents that must navigate an NxN grid and reach some goal without colliding into each other or into obstacles. The collective goal of the agents is to reach the goal as quickly as possible; there is a big bonus for an agent that reaches the goal, and big penalty for colliding agents. 

Agents are able to move up, down, left, and right, but will collide if they move on top of a spot that is occupied by another agent or obstacle.
