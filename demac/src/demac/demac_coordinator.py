"""The definition for the Coordinator, which acts as the middle man between the shared environment and agents.

This module defines the coordinator and starts up the message queue server for listening to agent requests. Once the
coordinator receives a request from each agent, it pools up their requests into a batch request, and sends it
to the shared environment. Once the shared environment returns a batch response, the coordinator extracts a response for
each agent and returns the response in each agent's callback message queue.

We follow the continuation-passing style outlined in section 2.1.2 of pika docs:
https://buildmedia.readthedocs.org/media/pdf/pika/latest/pika.pdf

Usage example:
    coordinator = Coordinator(env, exp_path='./exp0', test=False)
"""
import traceback
from threading import Thread

import gym
import pika
import json
import logging
import sys
import os

from pika import SelectConnection, BasicProperties
from pika.channel import Channel
from pika.frame import Method

from demac.src.demac.demac_agent_env_wrapper import AgentEnvWrapper
from demac.src.demac.marl_env_interface import MARLEnvInterface
from demac.src.utils.logging_utils import create_std_logger
from demac.src.utils.os_utils import init_dir

sys.path.append('..')


class Coordinator(gym.Env):
    """Coordinates the DeMAC workflow by acting as the middle man between the shared environment and agents.

    It works by pooling agent wrapper environment requests (i.e. step, reset), and batch requesting the user-defined
    shared environment once it has received a request from all agents. When the user-defined shared environment returns
    a batch response, the Coordinator unpacks the requests and returns the corresponding request to each individual
    agent wrapper environment.
    """

    def __init__(self, env: MARLEnvInterface, exp_path: str, test: bool) -> None:
        """Initializes the coordinator.

        Initializes the coordinator. Links the shared environment to the coordinator, and initializes the specified
        experiment directory.

        Args:
            env: the shared environment defined by the user (must extend MARLEnvInterface)
            exp_path: The relative path to the experiment directory
            test: Specifies whether we are training or testing
        """
        super(Coordinator, self).__init__()

        # Keep track of whether the coordinator server is set up yet
        self.ready = False

        self.env = env
        self.test = test

        # Sets up experiment directory with coordinator logging directory; deletes any files in it if it exists
        self.exp_path = exp_path + '/'
        self.coord_path = self.exp_path + 'Coordinator/'

        # Initialize experiment directories, with reprompts for user confirmation
        if not self.test:
            if os.path.exists(self.exp_path):
                if input(f'{self.exp_path} already exists! If you continue with training, you will '
                                        'override this directory. Are you sure? (y/n) ').lower() == 'y':
                    print(f'Initializing {self.exp_path}...')
                    init_dir(self.exp_path)
                    init_dir(self.coord_path)
                else:
                    print(f'Not overriding {self.exp_path}.')

        # Set up a logger for the coordinator
        self.logger = create_std_logger('Coordinator', self.coord_path + 'coordinator.log', logging.DEBUG)

        # The message queue we'll use to pool all agent function calls and
        # send a response to each agent
        self.msg_map = {}

    def reset(self) -> dict:
        """Resets the shared environment.

        A wrapper function for resetting the shared environment. This will only be called once all agent requests are
        pooled.

        Returns:
            A dict mapping agent names to their observations in the current timestep. For example:
            {
                'Agent 0': (1, 2, 3),
                'Agent 1': (4, 5, 6),
                'Agent 2': (7, 8, 9)
            }

            where each agent receives a tuple representing its observation.
        """
        return self.env.reset()

    def step(self, action: dict) -> dict:
        """Steps the shared environment.

        A wrapper function for stepping the shared environment. This will only be called once all agent requests are
        pooled.

        Args:
            action: A dict mapping agent names to actions

        Returns:
            A dict mapping agent name to a tuple containing the observation, reward, done, and info, respectively.
            For example:
            {
                'Agent 0': ((1, 2, 3), 1, False, {}),
                'Agent 1': ((4, 5, 6), 0, False, {}),
                'Agent 2': ((7, 8, 9), 1, False, {}),
            }
        """
        return self.env.step(action)

    def render(self, mode='human') -> None:
        """Renders the environment.

        Calls render to the shared environment, where the user defines how to render the task.

        Args:
            mode: The rendering type
        """
        self.env.render(mode=mode)

    def close(self) -> None:
        """Closes the environment.

        Specify any closing routines here.
        """
        self.env.close()

    def start(self) -> None:
        """Starts up the coordinator message queue server

        Starts up a thread for starting up the message queue server, and linking the coordinator to the server. This
        will wait until the server is done setting up.
        """
        self.logger.info("Starting up coordinator message queue server...")

        # Start up message queue server
        t = Thread(target=self.__connect_server, args=())
        t.daemon = True
        t.start()

        # Wait until the message queue server is set up
        while True:
            if self.ready:
                self.logger.info("Coordinator server is ready!")
                return

    def __connect_server(self) -> None:
        """Step #1: Connects to RabbitMQ using the default parameters.

        Establishes a connection with the RabbitMQ message queue server.
        """
        parameters = pika.ConnectionParameters()
        connection = pika.SelectConnection(parameters, on_open_callback=self.__on_connected)

        self.logger.info(f'Coordinator is listening on port {parameters.port}...')

        try:
            # Loop so we can communicate with RabbitMQ
            connection.ioloop.start()
        except KeyboardInterrupt:
            # Gracefully close the connection
            connection.close()

    def __on_connected(self, connection: SelectConnection) -> None:
        """Step #2: Called when we are fully connected to RabbitMQ.

        Opens a channel once we are fully connected to RabbitMQ.

        Args:
            connection: A newly established connection to RabbitMQ
        """
        # Open a channel
        connection.channel(on_open_callback=self.__on_channel_open)

    def __on_channel_open(self, new_channel: Channel) -> None:
        """Step #3: Called when our channel has opened.

        Once the channel has been opened, declare a message queue to for the coordinator to listen on.

        Args:
            new_channel: The newly created channel
        """
        self.channel = new_channel
        # Remove any existing channels first
        self.channel.queue_delete(queue='demac_coordinator_queue')
        self.channel.queue_declare(queue='demac_coordinator_queue',
                                   durable=True, exclusive=False,
                                   auto_delete=False,
                                   callback=self.__on_queue_declared)

    def __on_queue_declared(self, frame: Method) -> None:
        """Step #4: Called when RabbitMQ has told us our Queue has been declared.

        Lets the coordinator know that the server is done setting up, and begins to consume from the queue.

        Args:
            frame: The response from RabbitMQ
        """
        self.logger.info('Coordinator message queue server done setting up.')
        self.ready = True

        self.channel.basic_consume('demac_coordinator_queue',
                                   self.__handle_delivery)

    def __handle_delivery(self, channel: Channel, method: Method, props: BasicProperties, body: bytes) -> None:
        """Step #5: Called when we receive a message from RabbitMQ

        This gets called every time we consume a new message from the message queue. We expect these to be agent
        requests. The coordinator will pool agent requests, and batch send them to the shared environment once it
        receives a request from all agents.

        Args:
            channel: The channel we opened for RabbitMQ
            method: The AMQP Method frame
            props: The BasicProperties telling us how to respond to agents
            body: The body of the agent request
        """
        # Extract the body of the agent request
        body = json.loads(body.decode())

        # Package up a callback into our coordinator msg queue
        agent_name = list(body.keys())[0]
        exchange = ''
        routing_key = props.reply_to
        properties = pika.BasicProperties(correlation_id=props.correlation_id)
        delivery_tag = method.delivery_tag

        self.msg_map[agent_name] = (channel.basic_publish, exchange, routing_key, properties, body, channel.basic_ack,
                                    delivery_tag)

        # Send a batch request to the shared environment if we have received a request from all agents
        if len(self.msg_map) == len(self.env.names):

            # Either reset or step, where reset from a single agent request overrides all other requests
            resp = None
            try:
                if 'reset' in list(body[agent_name].keys()):
                    resp = self.reset()
                else:
                    resp = self.step({name: self.msg_map[name][4][name]['step'][0] for name in self.msg_map})
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                print("Press CTRL + C to exit.", file=sys.stderr)
                exit(1)

            # Publish the corresponding response to each each agent's callback queue
            for name, (channel.basic_publish, exchange, routing_key, properties, body, channel.basic_ack,
                       delivery_tag) in self.msg_map.items():
                channel.basic_publish(exchange=exchange,
                                      routing_key=routing_key,
                                      properties=properties,
                                      body=json.dumps(resp)
                                      )
                channel.basic_ack(delivery_tag=delivery_tag)
            self.msg_map = {}

    def link_env(self, agent_name: str, env: AgentEnvWrapper) -> None:
        """Links an agent wrapper environment to the shared environment.

        Calls the user-defined add_agent_env to link the agent wrapper environment to the shared environment.

        Args:
            agent_name: The name of the agent
            env: The agent wrapper environment
        """
        self.env.add_agent_env(agent_name, env)

    def _getattribute(self, agent_name: str, attr: str) -> None:
        """Define our own __getattribute__ default function.

        Define our own __getattribute__ default function so we don't have to mess with the default __getattribute__
        function. We expect to delegate any attribute dereferences to the user-defined shared environment.

        Args:
            agent_name: The name of the agent whose attributes to retrieve
            attr: The name of the attribute
        """
        return self.env._getattribute(agent_name, attr)
