"""The definition for the agent wrapper environment, which acts as the messenger between an agent and the coordinator.

This module defines the agent wrapper environment, which is what the agent interacts with. The wrapper environment
fosters the illusion that the agent is operating in a single-agent space, where there is a 1-1 mapping between the agent
and the wrapper environment. This wrapper environment will intercept all agent requests (e.g. variable dereferences,
function calls), and instead either delegate to the coordinator or publish a request to the coordinator queue. Any
responses from the coordinator in the callback queue can then be relayed back to the agent, in the format that the
agent expects.

Usage example:
    AgentEnvWrapper("Agent 0", coordinator)
"""

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
    """The agent wrapper environment for interacting with the coordinator message queue server.

    This class is a wrapper for the environment that each agent expects to have access to. This wrapper environment
    provides the illusion that each agent is operating in a single agent setting. Behind the scenes, this wrapper
    environment takes the requests of the agent it is mapped to, and publishes/consumes messages from the coordinator
    server.
    """

    def __init__(self, name: str, coordinator) -> None:
        """Defines a wrapper environment for the agent.

        Args:
            name: The name of the agent
            coordinator: The instance of the coordinator we are linking to (no type hint to avoid circular import)
        """
        super(AgentEnvWrapper, self).__init__()

        self.name = name
        self.coordinator = coordinator

        # Checks to see if we have already opened a channel for this agent
        self.channel_open = False

        # Set up new experiment directory for current agent only during training
        self.agent_path = self.coordinator.exp_path + self.name + '/'
        if not coordinator.test:
            init_dir(self.agent_path)

        # Initialize logger for this class
        self.__init_logger()

        # Links this wrapper environment to the shared environment via the coordinator
        self.coordinator.link_env(name, self)

        self.logger.info(f'Agent {self.name} wrapper environment set up.')

    def reset(self):
        """Sends a request to reset in the shared environment.

        Intercepts the reset call by the currently mapped agent, and publishes a reset request to the coordinator queue.

        Returns:
            The observation for the agent, typically represented as a tuple.
        """
        data = {
            self.name: {
                'reset': None
            }
        }
        resp = self.__query(data)
        return resp

    def step(self, action: int) -> None:
        """Steps through one timestep in the environment.

        Intercepts the step call by the currently mapped agent, and publishes a step request to the coordinator queue.

        Args:
            action: the action to take, assumed to be an int (recommended to modify this function to fit your needs)

        Returns:
            the user-defined return value for step, typically (observation, reward, done, info)
        """
        data = {
            self.name: {
                'step': [int(action)]
            }
        }
        ret = self.__query(data)
        return ret

    def render(self, mode='human') -> None:
        """Renders the environment.
        """
        self.coordinator.render(mode=mode)

    def close(self) -> None:
        """Closes the environment and DeMAC server.
        """
        self.coordinator.close()
        self.channel.close()
        self.logger.info('Finished learning. Disconnecting from the coordinator server.')

    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
    # vvvvvvvvvv DO NOT TOUCH THESE FUNCTIONS vvvvvvvvvv
    # vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

    def __query(self, body):
        """DO NOT TOUCH: Publishes a message to the DeMAC server message queue, and consumes the return message from
        the coordinator.

        This is the primary way in which the agent wrapper environment can interact with the coordinator. The wrapper
        environment publishes agent requests to the coordinator message queue here, and returns a response from the
        coordinator here. It will wait until it receives a request.

        Args:
            body: the message body to publish for the coordinator to consume

        Returns:
            the message from the coordinator, which is given by the shared environment
        """
        # Open a new channel if not open yet
        if not self.channel_open:
            self.__init_channel()
            self.channel_open = True

        # Formulate a publish request to the coordinator queue
        self.response = ''
        self.corr_id = str(uuid.uuid4())
        self.channel.basic_publish(exchange='',
                                   routing_key='demac_coordinator_queue',
                                   properties=pika.BasicProperties(reply_to=self.callback_queue,
                                                                   correlation_id=self.corr_id, ),
                                   body=json.dumps(body))

        # Wait for a response from the coordinator
        while not self.response:
            self.connection.process_data_events()

        # Send a response back to the wrapper environment
        self.response = json.loads(self.response)[self.name]
        return self.response

    # This func will be called every time we try to access a
    # variable inside this class
    def __getattribute__(self, attr: str) -> object:
        """DO NOT TOUCH: Override the class __getattribute__ function to call the coordinator's getattribute if this
        class does not contain the input attribute.

        This function will get called every time the agent tries to access a variable inside the environment.
        The wrapper environment will instead intercept the dereference request, and delegate to the coordinator.

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
            ret = self.coordinator._getattribute(self.name, attr)

        return ret

    def __init_logger(self):
        """DO NOT TOUCH: Initializes the log file for this agent at the agent path initialized earlier.
        """
        self.logger = create_std_logger(self.name, self.agent_path + f'/{self.name}.log', logging.DEBUG)

    def __init_channel(self):
        """DO NOT TOUCH: Initializes a connection to the coordinator server.

        Starts up a blocking connection to the coordinator. This also defines and listens to a callback queue that
        the coordinator will use to send responses back to the wrapper environment.
        """
        # Set up a blocking connection to the coordinator
        self.connection = pika.BlockingConnection(
            pika.ConnectionParameters(host='127.0.0.1'))
        self.channel = self.connection.channel()

        # Set up a callback queue for the coordinator to send responses back to the agent wrapper environment
        self.callback_queue = self.channel.queue_declare(queue='', exclusive=True).method.queue

        # Start consuming on the callback queue
        self.channel.basic_consume(
            queue=self.callback_queue,
            on_message_callback=self.__on_response,
            auto_ack=True)

        self.logger.info('Connection set up to coordinator.')

    def __on_response(self, ch, method, props, body):
        """DO NOT TOUCH: A callback function to populate self.response if a response is received from the callback queue.

        Args:
            ch: The channel we opened for the coordinator
            method: The AMQP Method frame
            props: The BasicProperties of the callback queue
            body: The body of the response
        """
        if self.corr_id == props.correlation_id:
            self.response = body

    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # ^^^^^^^^^^ DO NOT TOUCH THESE FUNCTIONS ^^^^^^^^^^
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
