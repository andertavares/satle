from copy import copy

import gym
from gym import spaces
import numpy as np
from pysat.formula import CNF

from .util import unit_propagation, encode


class SATState:
    def __init__(self, formula, model):
        """
        Creates a new state with a formula and a model (partial or full)
        :param formula: pysat.formula.CNF
        :param model: tuple with partial or full assignment to variables
        """
        self.formula = formula.copy()
        self.model = copy(model)

    def valid_actions(self):
        """
        Returns the valid actions that can be performed in this state
        :return:
        """
        raise NotImplementedError
        """free_literals = [v for v in range(1, self.formula.nv + 1) if v not in self.model]
        free_literals += [-v for v in range(1, self.formula.nv + 1) if v not in self.model]
        return free_literals"""

    def terminal(self):
        """
        Returns whether the current state is terminal
        :return:
        """
        return self.is_sat() or self.is_unsat()  # terminal states are sat or unsat nodes

    def is_sat(self):
        """
        Returns whether the formula in this state is satisfiable.
        Note that this is not the negation of is_unsat if this state is not terminal.
        :return:
        """
        return len(self.formula.clauses) == 0  # an empty formula is satisfiable

    def is_unsat(self):
        """
        Returns whether the formula in this state is unsatisfiable.
        Note that this is not the negation of is_sat if this state is not terminal.
        :return:
        """
        return any([len(c) == 0 for c in self.formula.clauses])  # a formula with an empty clause is unsat

    def reward(self):
        """
        Returns the reward for reaching the current state
        :return:
        """
        if self.terminal():
            return 0
        return -1


class SATEnv(gym.Env):
    """
    SAT gym environment. The goal is to find a solution to the
    formula, preferably with the least number of steps (variable assignments).
    Finding an unsat state yields -1 of reward.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, formula):
        """

        :param formula: a satisfiable formula
        """
        super(SATEnv, self).__init__()
        self.formula = formula
        self.state = SATState(self.formula, np.zeros(shape=(self.formula.nv, ), dtype=np.uint8))

        # literals is a list: [-nv,...,-1, +1, ..., nv] (without zero)
        self.literals = list(range(-self.formula.nv, 0)) + list(range(1, self.formula.nv+1))

        # 2 actions per variable (asserted or negated)
        self.action_space = spaces.Discrete(2 * self.formula.nv)

        # obs space is a dict{'formula': adj_matrix, 'model': model}
        # adj matrix is a vars x clauses matrix with 0,-1,+1 if var is absent, negated, asserted in clause
        # model contains 0,-1,+1 in each position if the corresponding variable is unassigned, negated, asserted
        # more info on gym spaces: https://github.com/openai/gym/tree/master/gym/spaces
        self.observation_space = spaces.Dict({
            'formula': spaces.Box(low=-1, high=1, shape=(self.formula.nv, len(self.formula.clauses)), dtype=np.uint8),
            'model': spaces.Box(low=-1, high=1, shape=(self.formula.nv,), dtype=np.uint8)  # array with (partial) model
        })

    def encode_action(self, var_index, value):
        """
        Returns an action in interval 0, 2*n_vars corresponding to assigning
        the truth-value to the given variable
        :param var_index: variable index (i.e. ranging from 0 to n_vars-1)
        :param value: bool corresponding to the value the variable will take
        :return:
        """
        # offsets by n_vars if value is positive, because the first 'n_vars' actions
        # correspond to assigning False to variables
        offset = self.formula.nv if value else 0

        return var_index + offset

    def var_and_polarity(self, action):
        """
        Translates an action into a tuple (var,value)
        Where var is the variable index (from 0 to num_vars-1) and value is (+1 or -1)
        meaning True or False, respectively

        :param action:
        :return:
        """
        lit = self.literals[action]  # translates from [0,..., 2*num_vars] to [-num_vars, ...,-1, +1,...,num_vars]
        return abs(lit) - 1, +1 if lit > 0 else -1

    def step(self, action):
        """
        Processes the action (selected variable) by adding it to the
        partial solution and returning the resulting formula
        and associated reward, done and info.
        :param action:
        :return:
        """

        new_model = copy(self.state.model)

        # adds the literal to the partial solution
        var, value = self.var_and_polarity(action)
        new_model[var] = value

        # creates new formula with the result of adding the corresponding literal to the previous
        new_formula = unit_propagation(self.formula, self.literals[action])

        self.state = SATState(new_formula, new_model)

        # reward: 1, -1, 0 for sat, unsat, non-terminal, respectively
        reward = 1 if self.state.is_sat() else -1 if self.state.is_unsat() else 0
        obs = {'formula': encode(self.formula.nv, self.state.formula.clauses), 'model': self.state.model}
        return obs, reward, self.state.terminal(), {'clauses': self.state.formula.clauses}

    def reset(self):
        """
        Resets to the initial state and returns it
        :return:
        """
        self.state = SATState(self.formula, np.zeros(shape=(self.formula.nv, ), dtype=np.uint8))
        return encode(self.state.formula.nv, self.state.formula.clauses)

    def render(self, mode='human'):
        print('#vars', self.state.formula.nv)
        print('clauses', self.state.formula.clauses)
        print('model', self.state.model)
        print(f'sat={self.state.is_sat()}, unsat={self.state.is_unsat()}')
        print(encode(self.state.formula.nv, self.state.formula.clauses))

    def close(self):
        pass

