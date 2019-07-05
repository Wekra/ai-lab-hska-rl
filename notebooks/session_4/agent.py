"""
Atari DQN-Learning implementation with keras.
"""

import random
from collections import deque
from typing import Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras.backend import set_session
from tensorflow.keras.layers import Conv2D, Flatten, Dense, Lambda, multiply, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.losses import huber_loss

from lib import AbstractAgent
from lib.atari_helpers import LazyFrames


class AtariDQN(AbstractAgent):

    def __init__(self, action_size: int, state_size: int,
                 gamma: float = None, epsilon: float = None, epsilon_decay: float = None, epsilon_min: float = None,
                 alpha: float = None, batch_size=None, memory_size=None, start_replay_step=None,
                 target_model_update_interval=None, train_freq=None):
        self.action_size = action_size
        self.state_size = state_size

        # randomly remembered states and rewards (used because of efficiency and correlation)
        self.memory = deque(maxlen=memory_size)

        # discount factor (how much discount future reward)
        self.gamma = gamma

        # initial exploration rate of the agent (exploitation vs. exploration)
        self.epsilon = epsilon

        # decay epsilon over time to shift from exploration to exploitation
        self.epsilon_decay = epsilon_decay

        # minimal epsilon: x% of the time take random action
        self.epsilon_min = epsilon_min

        # step size also called learning rate alpha
        self.alpha = alpha

        # can be any multiple of 32 (smaller mini-batch size usually leads to higher accuracy/ NN performs better)
        self.batch_size = batch_size

        # number of steps played
        self.step = 0

        # after how many played steps the experience replay should start
        self.start_replay_step = start_replay_step

        # after how many steps should the weights of the target model be updated
        self.target_model_update_interval = target_model_update_interval

        # at which frequency (interval) the model should be trained (steps)
        self.train_freq = train_freq

        assert self.start_replay_step >= self.batch_size, \
            "The number of steps to start replay must be at least as large as the batch size."

        self.action_mask = np.ones((1, self.action_size))
        self.action_mask_batch = np.ones((self.batch_size, self.action_size))

        config = tf.ConfigProto(intra_op_parallelism_threads=8,
                                inter_op_parallelism_threads=4,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        session = tf.Session(config=config)
        set_session(session)  # set this TensorFlow session as the default session for Keras

        self.model = self._build_model()
        self.target_model = self._build_model()

    def _build_model(self):
        atari_shape = (84, 84, 4)
        # With the functional API we need to define the inputs. Sequential API no longer works because of merge mask
        frames_input = Input(atari_shape, name='frames')
        action_mask = Input((self.action_size,), name='action_mask')

        # Assuming that the input frames are still encoded from 0 to 255. Transforming to [0, 1].
        normalized = Lambda(lambda x: x / 255.0, name='normalization')(frames_input)

        conv1 = Conv2D(filters=32,
                       kernel_size=(8, 8),
                       strides=(4, 4),
                       activation='relu',
                       kernel_initializer=tf.variance_scaling_initializer(scale=2)
                       )(normalized)

        conv2 = Conv2D(filters=64,
                       kernel_size=(4, 4),
                       strides=(2, 2),
                       activation='relu',
                       kernel_initializer=tf.variance_scaling_initializer(scale=2)
                       )(conv1)

        conv3 = Conv2D(filters=64,
                       kernel_size=(3, 3),
                       strides=(1, 1),
                       activation='relu',
                       kernel_initializer=tf.variance_scaling_initializer(scale=2)
                       )(conv2)

        # Flattening the last convolutional layer.
        conv_flattened = Flatten()(conv3)

        # "The final hidden layer is fully-connected and consists of 256 rectifier units."
        hidden = Dense(units=256, activation='relu')(conv_flattened)

        # "The output layer is a fully-connected linear layer with a single output for each valid action."
        output = Dense(self.action_size)(hidden)

        # Finally, we multiply the output by the mask!
        # "The main drawback of [passing the action as an input] is that a separate forward pass is required
        # to compute the Q-value of each action, resulting in a cost that scales linearly with the number of
        # actions. We instead use an architecture in which there is a separate output unit for each possible
        # action, and only the state representation is an input to the neural network.
        # The outputs correspond to the predicted Q-values of the individual action for the input state.
        # The main advantage of this type of architecture is the ability to compute Q-values for
        # all possible actions in a given state with only a single forward pass through the network.
        filtered_output = multiply([output, action_mask])

        model = Model(inputs=[frames_input, action_mask], outputs=filtered_output)
        model.compile(loss=huber_loss, optimizer=Adam(lr=self.alpha, clipnorm=10), metrics=None)

        return model

    def _remember(self, experience: Tuple[LazyFrames, int, LazyFrames, float, bool]) -> None:
        self.memory.append(experience)

    def _replay(self) -> None:
        mini_batch = random.sample(self.memory, self.batch_size)
        states_lf, actions_lf, next_states_lf, rewards_lf, dones_lf = zip(*mini_batch)

        # Todo: Convert the parts of the mini-batch into corresponding numpy arrays.
        #  Note that the states are of type 'LazyFrames' due to memory efficiency
        #  and must therefore be converted individually.

        actions = np.array(actions_lf)
        
        states = []
        next_states = []
        
        for frame in states_lf:
            states.append(np.array(frame))
            
        for frame in next_states_lf:
            next_states.append(np.array(frame))

        states = np.array(states)
        next_states = np.array(next_states)
        #states = np.array(states_lf)
#         next_states = np.array(next_states_lf)
        rewards = np.array(rewards_lf)
        dones = np.array(dones_lf)
        
#         print("States: ", states)
#         print("NStates: ", next_states)
#         print("Rwds: ", rewards)
#         print("Dns: ", dones)
        # The following assert statements are intended to support further implementation,
        #  but can also be removed/adjusted if necessary.
        assert all(isinstance(x, np.ndarray) for x in (states, actions, rewards, next_states, dones)), \
            "All experience batches should be of type np.ndarray."
        assert states.shape == (self.batch_size, 84, 84, 4), \
            f"States shape should be: {(self.batch_size, 84, 84, 4)}"
        assert actions.shape == (self.batch_size,), f"Actions shape should be: {(self.batch_size,)}"
        assert rewards.shape == (self.batch_size,), f"Rewards shape should be: {(self.batch_size,)}"
        assert next_states.shape == (self.batch_size, 84, 84, 4), \
            f"Next states shape should be: {(self.batch_size, 84, 84, 4)}"
        assert dones.shape == (self.batch_size,), f"Dones shape should be: {(self.batch_size,)}"

        # Todo: Predict the Q values of the next states (choose the right model!). Passing ones as the action mask
        #  Note that a suitable mask has already been created in '__init__'.
        next_q_values = self.target_model.predict([next_states, self.action_mask_batch])

        # Todo: Calculate the Q values, remember
        #  - the Q values of each non-terminal state is the reward + gamma * the max next state Q value
        #  - and the Q values of terminal states should be the reward (Hint: 1.0 - dones) makes sure that if the game is
        #    over, targetQ = rewards
        # Depending on the implementation, the axis must be specified to get the max q-value for EACH batch element!

        # Todo: Create a one hot encoding of the actions (the selected action is 1 all others 0)
        #  Hint look at the imports. A Keras help function will be imported there.

        # Todo: Create the target Q values based on the one hot encoding of the actions and the calculated q-values
        #  Hint you have to "reshape" the q_values to match the shape
        ######### own code below:
#         next_q_values = []
#         q_values = []
#         one_hot_actions = []
#         target_q_values = []
#         for i in range(len(next_q_values_original)):
#             if dones[i]:
#                 # Done: Set the Q values of terminal states to 0 (by definition)
#                 # Final/terminal states: The states that have no available actions are final/terminal states. each action=0
#                 next_q_values.append(np.zeros(self.action_size))
#                 q_values.append(0)
#             else:
#                 # Done: Calculate the Q values, you must
#                 #  remember the Q values of each non-terminal state is the reward + gamma * the max next state Q value
#                 # Depending on the implementation, the axis must be specified to get the max q-value for EACH batch element!
#                 next_q_values.append(next_q_values_original[i])
#                 q_values.append(rewards[i] + self.gamma * np.max(next_q_values[i]))
            
#             one_hot_actions.append([])
#             target_q_values.append([])
            
#             idx = np.array(np.argmax(next_q_values[i]))
#             idxmax = idx[0] if len(idx.shape) > 0 else idx
#             #print("IDX",idx,"IDXMAX",idxmax)
                
#             for j in range(self.action_size):
#                 # Done: Create a one hot encoding of the actions (the selected action is 1 all others 0)
#                 one_hot_actions[i].append(1           if j == idxmax else 0)
#                 # Done: Create the target Q values based on the one hot encoding of the actions and the calculated q-values
#                 target_q_values[i].append(q_values[i] if j == idxmax else 0)
    
        next_q_values[dones] = 0.0
        q_values = rewards + self.gamma*np.max(next_q_values, axis=1)
        one_hot_encoding = to_categorical(actions, num_classes=self.action_size)
        target_q_values = one_hot_encoding * q_values.reshape(self.batch_size, 1)

    
#         print("DONES: ", dones)
#         print("NEXT Q VALUES: ", next_q_values)
#         print("Q_Values: ", q_values)
#         print("ONE HOT ACTIONS: ", one_hot_actions)
#         print("TARGET_Q_VALUES ", target_q_values)


    # Todo: fit the model with the right x and y values
        self.model.fit(
            x= (states, one_hot_encoding),  # states and mask
            y=[target_q_values],  # target Q values
            batch_size=self.batch_size,
            verbose=0
        )

    def act(self, state_lf: LazyFrames) -> int:
        """Selects the action to be executed based on the given state.

        Implements epsilon greedy exploration strategy, i.e. with a probability of
        epsilon, a random action is selected.

        Args:
            state: LazyFrames object representing the state based on 4 stacked observations (images)

        Returns:
            Action.
        """
        state = np.array(state_lf)
        if random.random() < self.epsilon:
            action = random.randrange(self.action_size)
        else:
            # Todo: Use the model to get the Q values for the state and determine the action based on the max Q value.
            #  Hint: You have to convert the state to a list of numpy arrays before you can pass it to the model
            # action = 1
#             print('State: ', state)
#             print('AM: ', self.action_mask)
            action = np.argmax(self.model.predict([[state], self.action_mask]))
#             print("PREDICTED ACTION: ", action)
        return action

    def train(self, experience: Tuple[LazyFrames, int, LazyFrames, float, bool]) -> None:
        
        """Stores the experience in memory. If memory is full trains network by replay.
        Args:
            experience: Tuple of state, action, next state, reward, done.
        Returns:
            None
        """
        self._remember(experience)

        # Todo: As soon as enough steps are played:
        #  - Update epsilon as long as it is not minimal
        #  - update weights of the target model (syn of the two models)
        #  - execute replay -> include train_frequency
        if self.step > self.start_replay_step:

            if self.step % self.train_freq == 0:
                self._replay()

            if self.step % self.target_model_update_interval == 0:
                self.target_model.set_weights(self.model.get_weights())

            if self.epsilon > self.epsilon_min:
                self.epsilon = self.epsilon - self.epsilon_decay

        self.step += 1
