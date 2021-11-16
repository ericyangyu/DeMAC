import traceback
from threading import Thread

import gym
import pika
import json
import logging
import sys
import os

from demac.src.demac.demac_agent_env_wrapper import AgentEnvWrapper
from demac.src.utils.logging_utils import create_std_logger
from demac.src.utils.os_utils import init_dir

sys.path.append('..')


class Coordinator(gym.Env):
    """
    Coordinates the MARL workflow by pooling agent requests (i.e. step, reset), batch requesting the user-defined
    MARL environment, and returning individual agent requests to each agent wrapper environment.
    """
    ready = False  # Keeps track of whether the coordinator server is set up yet

    def __init__(self, msarl_env, exp_path, test=False):
        super(Coordinator, self).__init__()

        self.env = msarl_env

        self.agent_names = msarl_env.names

        # Keep track of the wrapper environments for each agent for logging purposes.
        self.agent_envs = {}

        # The message queue we'll use to pool all agent function calls and
        # send a response to each agent
        # NOTE: structure is {
        #     'agent name': [a1, a2]
        # }
        self.msg_map = {}

        # Keep track of whether we are training or testing
        self.test = test

        # Sets up experiment directory with coordinator logging directory; deletes any files in it if it exists
        self.exp_path = exp_path + '/'
        self.coord_path = self.exp_path + 'Coordinator/'

        if not self.test:
            # Initialize experiment directories, with reprompts to user for confirmation
            should_override = False
            if os.path.exists(self.exp_path):
                should_override = input(f'{self.exp_path} already exists! If you train, you will override this directory. Are you sure? (y/n) ')
            if not should_override or should_override.lower() == 'y':
                print(f'Initializing {self.exp_path}...')
                init_dir(self.exp_path)
                init_dir(self.coord_path)
            else:
                print(f'Not overriding {self.exp_path}.')

        # Set up a logger
        self.logger = create_std_logger('Coordinator', self.coord_path + 'coordinator.log', logging.DEBUG)

    def reset(self):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def render(self):
        self.env.render()

    def close(self):
        """
        Closes the environment

        Returns:
            None
        """
        self.env.close()

    def start(self):
        """
        Starts up the DeMAC server

        Returns:
            None
        """
        # Start up message queue server
        self.logger.info("Starting up DeMAC server...")
        # _thread.start_new_thread(self.__connect_server, ())

        t = Thread(target=self.__connect_server, args=())
        t.daemon = True
        t.start()

        while True:
            if self.ready:
                self.logger.info("DeMAC server is ready!")
                return

    def __connect_server(self):
        """
        Step #1: Connects to RabbitMQ using the default parameters
        """
        parameters = pika.ConnectionParameters()
        self.logger.info(f'Coordinator is listening on port {parameters.port}...')
        self.connection = pika.SelectConnection(parameters, on_open_callback=self.__on_connected)

        try:
            # Loop so we can communicate with RabbitMQ
            self.connection.ioloop.start()
        except KeyboardInterrupt:
            # Gracefully close the connection
            self.connection.close()

    def __on_connected(self, connection):
        """
        Step #2: Called when we are fully connected to RabbitMQ
        """
        # Open a channel
        self.connection.channel(on_open_callback=self.__on_channel_open)

    def __on_channel_open(self, new_channel):
        """
        Step #3: Called when our channel has opened
        """
        self.channel = new_channel
        # Remove any existing channels first
        self.channel.queue_delete(queue='demac_coordinator_queue')
        self.channel.queue_declare(queue='demac_coordinator_queue',
                                   durable=True, exclusive=False,
                                   auto_delete=False,
                                   callback=self.__on_queue_declared)

    def __on_queue_declared(self, frame):
        """
        Step #4: Called when RabbitMQ has told us our Queue has been declared,
        frame is the response from RabbitMQ
        """
        self.logger.info('Server done setting up.')
        Coordinator.ready = True
        self.logger.info(f'Coordinator.ready is now {Coordinator.ready}')

        self.channel.basic_consume('demac_coordinator_queue',
                                   self.__handle_delivery)

    def __handle_delivery(self, channel, method, props, body):
        """
        Step #5: Called when we receive a message from RabbitMQ
        """
        body = json.loads(body.decode())

        # Package up a callback into our coordinator msg queue
        agent_name = list(body.keys())[0]
        exchange = ''
        routing_key = props.reply_to
        properties = pika.BasicProperties(correlation_id=props.correlation_id)
        delivery_tag = method.delivery_tag

        a1 = (channel.basic_publish, exchange, routing_key, properties, body)
        a2 = (channel.basic_ack, delivery_tag)

        self.msg_map[agent_name] = (a1, a2, body)

        if len(self.msg_map) == len(self.env.names):

            sys.stdout.flush()

            # Either reset or step, where reset from a single agent request overrides all other requests
            ret = None
            try:
                if 'reset' in list(body[agent_name].keys()):
                    ret = self.reset()
                elif 'step' in list(body[agent_name].keys()):
                    ret = self.step({name: self.msg_map[name][2][name]['step'][0] for name in self.msg_map})
            except Exception as e:
                print(traceback.format_exc(), file=sys.stderr)
                print("Press CTRL + C to exit.", file=sys.stderr)
                exit(1)

            for name, (a1, a2, _) in self.msg_map.items():
                a1[0](exchange=a1[1],
                      routing_key=a1[2],
                      properties=a1[3],
                      body=json.dumps(ret, default=True)
                      )
                a2[0](delivery_tag=a2[1])
            self.msg_map = {}

    def getattribute(self, agent_name, attr):
        """
        Defines our own __getattribute__ default function so we don't have to mess with the default
        __getattribute__ function

        Args:
            agent_name: the name of the agent whose attributes to retrieve
            attr: the name of the attribute

        Returns:

        """
        return self.env.getattribute(agent_name, attr)

    def link_env(self, agent_name: str, env: AgentEnvWrapper):
        self.env.add_agent_env(agent_name, env)

