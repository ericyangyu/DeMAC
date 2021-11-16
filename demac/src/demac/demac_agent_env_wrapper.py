import gym
import json
import pika
import uuid
import logging
import sys

sys.path.append('..')

from demac.src.utils.logging_utils import create_std_logger
from demac.src.utils.os_utils import init_dir


class AgentEnvWrapper(gym.Env):
    """
    An agent wrapper environment for interacting with the DeMAC server
    """
    def __init__(self, name, coordinator):
        super(AgentEnvWrapper, self).__init__()

        self.channel_open = False
        self.name = name
        self.coordinator = coordinator

        # Set up experiment directory for current agent
        self.agent_path = self.coordinator.exp_path + self.name + '/'

        if not coordinator.test:
            init_dir(self.agent_path)

        # Initialize logger for this class
        self.__init_logger()

        # Links this wrapper environment to the coordinator
        self.coordinator.link_env(name, self)

        self.logger.info(f'Agent {self.name} wrapper environment set up.')

    def reset(self):
        """
        Resets the environment

        Returns:
            the user-defined return value for reset, typically (observation)

        """
        data = {
            self.name: {
                'reset': None
            }
        }
        ret = self.query(data)
        return ret

    def step(self, action):
        """
        Steps through one timestep in the environment

        Args:
            action: the action to take, assumed to be an int (you can modify this function)

        Returns:
            the user-defined return value for step, typically (observation, reward, done, info)
        """
        data = {
            self.name: {
                'step': [int(action)]
            }
        }
        ret = self.query(data)
        return ret

    def render(self):
        """
        Renders the environment

        Returns:
            None
        """
        self.coordinator.render()

    def close(self):
        """
        Closes the environment and DeMAC server.

        Returns:
            None
        """
        self.coordinator.close()

        self.channel.close()
        self.logger.info('Finished learning; closing connection to the DeMAC server.')


    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # vvvvvvvvvv DO NOT TOUCH THESE FUNCTIONS vvvvvvvvvv
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    def query(self, body):
        """
        Publishes a message to the DeMAC server message queue, and consumes the return message from the DeMAC server.

        Args:
            body: the message body to send to the DeMAC server to consume

        Returns:
            the message from the DeMAC server, which is given by the user-defined MARL environment
        """
        if not self.channel_open:
            self.__init_channel()
            self.channel_open = True

        body = json.dumps(body)

        self.response = ''
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key='demac_coordinator_queue',
                                   properties=pika.BasicProperties(reply_to=self.callback_queue,
                                                                   correlation_id=self.corr_id, ), body=body)

        while not self.response:
            self.connection.process_data_events()
        self.response = json.loads(self.response)[self.name]
        return self.response

    # This func will be called every time we try to access a
    # variable inside this class
    def __getattribute__(self, attr):
        """
        Override the class __getattribute__ function to call the coordinator's getattribute if this class does not
        contain the input attribute

        Args:
            attr: the name of the attribute

        Returns:
            the attribute value returned by this class or the coordinator
        """
        # Get attribute from current class instance normally
        ret = object.__getattribute__(self, attr)

        # If attr not found in current class instance, then must be
        # in coordinator instance
        if ret is None:
            ret = self.coordinator.getattribute(self.name, attr)

        return ret

    def __init_logger(self):
        """
        Initializes the log file for this agent

        Returns:
            None
        """
        self.logger = create_std_logger(self.name, self.agent_path + f'/{self.name}.log', logging.DEBUG)

    def __init_channel(self):
        """
        Sets up a connection to the DeMAC server

        Returns:
            None
        """
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='127.0.0.1'))

        self.channel = self.connection.channel()

        result = self.channel.queue_declare(queue='', exclusive=True)
        self.callback_queue = result.method.queue

        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.__on_response,
            auto_ack=True)

        self.logger.info('Connection set up to DeMAC server.')

    def __on_response(self, ch, method, props, body):
        if self.corr_id == props.correlation_id:
            self.response = body

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^ DO NOT TOUCH THESE FUNCTIONS ^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
