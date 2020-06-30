import gym
from gym import spaces
from gym import utils
from gym.utils import seeding
import numpy as np

import logging
logger = logging.getLogger(__name__)


class InventoryEnv(gym.Env, utils.EzPickle):
    """Inventory control with lost sales environment

    TO BE EDITED

    This environment corresponds to the version of the inventory control
    with lost sales problem described in Example 1.1 in Algorithms for
    Reinforcement Learning by Csaba Szepesvari (2010).
    https://sites.ualberta.ca/~szepesva/RLBook.html
    """
 
    def __init__(self, initial_stock=100, fixed_entry_cost=5, item_cost=2, inventory_cost=2, payment=3, lam=8, random_seed=None):
        self.initial_stock = initial_stock
        self.action_space = spaces.Discrete(initial_stock)
        self.observation_space = spaces.Discrete(initial_stock)
        self.maximum = initial_stock
        self.state = initial_stock
        self.fixed_entry_cost = fixed_entry_cost
        self.item_cost = item_cost
        self.inventory_cost = inventory_cost
        self.payment = payment
        self.lam = lam
        if random_seed:
            self._seed(random_seed)
        else:
            self._seed()

        # Start the first round
        self.reset()

    def demand(self):
        return self.np_random.poisson(self.lam)

    def transition(self, x, a, d):
        maximum = self.maximum
        return max(min(x + a, maximum) - d, 0)

    def reward(self, x, action, y):
        fixed_entry_cost = self.fixed_entry_cost
        maximum = self.maximum
        item_cost = self.item_cost
        inventory_cost = self.inventory_cost
        payment = self.payment
        r = -fixed_entry_cost * (action > 0) - item_cost * max(min(x + action, maximum) - x, 0) - inventory_cost * x + payment * max(min(x + action, maximum) - y, 0)
        return r

    def _seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        assert self.action_space.contains(action)
        observation = self.state
        demand = self.demand()
        new_observation = self.transition(observation, action, demand)
        self.state = new_observation
        reward = self.reward(observation, action, new_observation)
        done = 0
        last_demand = demand
        return new_observation, last_demand, reward, done, {}

    def reset(self):
        self.state = self.initial_stock
        return self.state
