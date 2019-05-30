"""
AbstractBaseClass (Interface) for all RL Agents used in this Lab.
"""

from abc import ABC, abstractmethod


class AbstractAgent(ABC):

    @abstractmethod
    def act(self, state) -> int:
        """Selects an action based on the given state.

        Args:
            state: Environment state.

        Returns:
            The selected action.
        """

    @abstractmethod
    def train(self, exploration) -> None:
        """Trains based on the given exploration.

        Args:
            exploration: Tuple consists of 
                state, action, next_state, reward and done.

        Returns:
            None
        """
