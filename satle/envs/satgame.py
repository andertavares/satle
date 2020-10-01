from copy import copy, deepcopy

import gym
from gym import spaces
import numpy as np
from pysat.formula import CNF

from .util import unit_propagation, encode, num_vars, vars_and_indices


class SATState:
    def __init__(self, clauses):
        """
        Creates a new state with a original_clauses and a model (partial or full)
        :param clauses: list(list(int))
        """
        self.clauses = deepcopy(clauses)
        self.n_vars = num_vars(self.clauses)

    def terminal(self):
        """
        Returns whether the current state is terminal
        :return:
        """
        return self.is_sat() or self.is_unsat()  # terminal states are sat or unsat nodes

    def is_sat(self):
        """
        Returns whether the original_clauses in this state is satisfiable.
        Note that this is not the negation of is_unsat if this state is not terminal.
        :return:
        """
        return len(self.clauses) == 0  # an empty original_clauses is satisfiable

    def is_unsat(self):
        """
        Returns whether the original_clauses in this state is unsatisfiable.
        Note that this is not the negation of is_sat if this state is not terminal.
        :return:
        """
        return any([len(c) == 0 for c in self.clauses])  # a original_clauses with an empty clause is unsat

    def reward(self):
        """
        Returns the reward for reaching the current state
        :return:
        """
        return 1 if self.is_sat() else -1 if self.is_unsat() else 0


class SATEnv(gym.Env):
    """
    SAT gym environment. The goal is to find a solution to the
    original_clauses, preferably with the least number of steps (variable assignments).
    Finding an unsat state yields -1 of reward.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, clauses):
        """

        :param clauses: a satisfiable original_clauses
        """
        super(SATEnv, self).__init__()
        self.original_clauses = clauses
        self.n_vars = num_vars(self.original_clauses)  # fixed throughout execution

        # each position stores the value of a variable: 0,1,-1 for unassigned,True,False
        self.model = np.zeros(shape=(self.n_vars,), dtype=np.int8)

        self.var_to_idx, self.idx_to_var = vars_and_indices(self.original_clauses)

        # literals is a list: [1,...,n_vars, -1,...,-n_vars] (DIMACS notation without zero)
        self.literals = list(range(1, self.n_vars + 1)) + list(range(-1, -self.n_vars-1, -1))

        # 2 actions per variable (assign True or False to it)
        self.action_space = spaces.Discrete(2 * self.n_vars)

        # obs space the adjacency matrix of the factor graph
        # adj matrix is a vars x original_clauses matrix with 0,-1,+1 if var is absent, negated, asserted in clause
        # more info on gym spaces: https://github.com/openai/gym/tree/master/gym/spaces
        self.observation_space = spaces.Box(low=-1, high=1, shape=(self.n_vars, len(self.original_clauses)), dtype=np.int8),

        self.state = None  # is initialized at reset
        self.reset()

    def encode_action(self, var_index, value):
        """
        Returns an action in interval 0, 2 * n_vars corresponding to assigning
        the truth-value to the given variable
        :param var_index: variable index (i.e. ranging from 0 to n_vars-1)
        :param value: bool corresponding to the value the variable will take
        :return:
        """
        # offsets by n_vars if value is negative, because the first 'n_vars' actions
        # correspond to assigning True to variables
        offset = self.n_vars if not value else 0

        return var_index + offset

    def action_to_literal(self, action):
        """
        Translates an action (ranging from 0..2*n_vars)
        to a DIMACS literal ([1,...,orig_vars]+[-1,...,-orig_vars])
        where n_vars refers to the current clauses and orig_vars
        to the number of variables in the original clauses (passed to __init__)
        :param action:
        :return:
        """
        var = self.idx_to_var[action]

        # the first n_vars are positive literals, the remaining are negative
        return var if action < self.n_vars else -var

    def var_and_signal(self, action):
        """
        Translates an action into a tuple (var,signal)
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
        partial solution and returning the resulting original_clauses
        and associated reward, done and info.
        :param action:
        :return:
        """

        # adds the literal to the partial solution
        literal = self.action_to_literal(action)
        # index is zero-based where literal is 1-based; assigned value is the literal signal
        var_idx, var_sign = abs(literal)-1, 1 if literal > 0 else -1

        # returns data for this unchanged state if action is out of range or
        # in an attempt to assign a value to a non-free variable
        if not self.action_space.contains(action) or self.model[var_idx] != 0:
            obs = encode(self.state.clauses)
            info = {'model': self.model, 'clauses': self.state.clauses}
            return obs, self.state.reward(), self.state.terminal(), info

        # assigns the intended value to the variable
        self.model[var_idx] = var_sign

        # creates new original_clauses with the result of adding the corresponding literal to the previous
        new_clauses = unit_propagation(self.state.clauses, var_idx, var_sign)

        # updates indices
        self.var_to_idx, self.idx_to_var = vars_and_indices(new_clauses)
        self.state = SATState(new_clauses)

        # updates action and observation spaces
        self.action_space = spaces.Discrete(2 * self.state.n_vars)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.state.n_vars, len(self.state.clauses)), dtype=np.int8
        )

        obs = encode(self.state.clauses)
        info = {'model': self.model, 'clauses': self.state.clauses}
        return obs, self.state.reward(), self.state.terminal(), info

    def reset(self):
        """
        Resets to the initial state and returns it
        :return:
        """

        self.model = np.zeros(shape=(self.n_vars,), dtype=np.int8)
        self.var_to_idx, self.idx_to_var = vars_and_indices(self.original_clauses)
        self.action_space = spaces.Discrete(2 * self.n_vars)
        self.observation_space = spaces.Box(
            low=-1, high=1, shape=(self.n_vars, len(self.original_clauses)), dtype=np.int8
        )
        self.state = SATState(self.original_clauses)
        return encode(self.state.clauses)

    def render(self, mode='human'):
        print('#vars', self.state.original_clauses.nv)
        print('original_clauses', self.state.original_clauses.original_clauses)
        print('model', self.state.model)
        print(f'sat={self.state.is_sat()}, unsat={self.state.is_unsat()}')
        print(encode(self.state.original_clauses.original_clauses))

    def close(self):
        pass


