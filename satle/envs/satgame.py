from copy import copy, deepcopy

import gym
from gym import spaces
import numpy as np

from .util import unit_propagation, num_vars, vars_and_indices


class SATState:
    def __init__(self, clauses):
        """
        Creates a new state from the clauses
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

    def encode(self):
        """
        Encodes the state as a factor graph and returns the dense
        v x c x 2 tensor representing the graph, where v is #variables and c is
        #clauses. Entry i,j contains:
        - positive edge [0,1] if var i is asserted in clause j
        - negative edge [1,0] if var i is negated in clause j
        - no edge [0,0] if var i is not present in clause j
        Variable indexes in the clauses are according to their occurence.
        E.g., if clauses is: [[-5, 1], [2, -7, 5]] then the index of
        variables 5,1,2,7 become 0,1,2,3 respectively.
        :return: np.array with the representation matrix (v x c x 2)
        """

        # maps each variable to its index in the matrix
        var_to_idx, _ = vars_and_indices(self.clauses)
        adj = np.zeros((len(var_to_idx), len(self.clauses), 2), dtype=np.uint8)  # v x c x 2 tensor matrix (v=#vars, c=#clauses)

        for c_num, clause in enumerate(self.clauses):
            for literal in clause:
                var_idx = var_to_idx[abs(literal)]
                adj[var_idx][c_num] = [1,0] if literal < 0 else [0,1]

        return adj


class SATGame(gym.Env):
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
        super(SATGame, self).__init__()
        self.original_clauses = clauses
        self.n_vars = num_vars(self.original_clauses)  # fixed throughout execution

        # each position stores the value of a variable: 0,1,-1 for unassigned,True,False
        self.model = np.zeros(shape=(self.n_vars,), dtype=np.int8)

        self.var_to_idx, self.idx_to_var = vars_and_indices(self.original_clauses)

        # literals is a list: [1,...,n_vars, -1,...,-n_vars] (DIMACS notation without zero)
        self.literals = list(range(1, self.n_vars + 1)) + list(range(-1, -self.n_vars-1, -1))

        # 2 actions per variable (assign True or False to it)
        self.action_space = spaces.Discrete(2 * self.n_vars)

        # obs space is the v x c x 2 tensor representing of the factor graph
        # where v=#vars, c=#caluses. Entry i,j contains:
        # [0,0]/[1,0]/[0,1] if var i is absent/negated/asserted in clause j
        # more info on gym spaces: https://github.com/openai/gym/tree/master/gym/spaces
        self.observation_space = spaces.Box(low=0, high=1, shape=(self.n_vars, len(self.original_clauses), 2), dtype=np.uint8),

        self.state = None  # is initialized at reset
        self.reset()

    def encode_action(self, dimacs_var, value):
        """
        Returns an action in interval 0, 2 * n_vars corresponding to assigning
        the truth-value to the given dimacs variable.
        Especially useful if one is looking at the clauses rather than the observation
        :param dimacs_var: variable index (i.e. ranging from 1 to n_vars)
        :param value: bool corresponding to the value the variable will take
        :return:
        """
        # offsets by n_vars if value is negative, because the first 'n_vars' actions
        # correspond to assigning True to variables
        idx = self.var_to_idx[dimacs_var]
        offset = self.state.n_vars if not value else 0

        return idx + offset

    def var_and_signal(self, action):
        """
        Translates an action (ranging from 0..2*n_vars)
        to a DIMACS literal ([1,...,orig_vars]+[-1,...,-orig_vars])
        where n_vars refers to the current clauses and orig_vars
        to the number of variables in the original clauses (passed to __init__)

        Returns -1,0 (invalid action and signal) if action is invalid
        :param action:
        :return:int,int
        """
        # returns invalid values for invalid action
        if not self.action_space.contains(action):
            return -1, 0
        # the first n_vars are positive assignments, the remaining are negative
        assign_true = action < self.state.n_vars
        idx = action if assign_true else action // 2
        var = self.idx_to_var[idx]

        return abs(var)-1, 1 if assign_true else -1

    def step(self, action):
        """
        Processes the action (selected variable) by adding it to the
        partial solution and returning the resulting original_clauses
        and associated reward, done and info.
        :param action:
        :return:
        """

        # adds the literal to the partial solution
        # index is zero-based where literal is 1-based; assigned value is the literal signal
        var_idx, var_sign = self.var_and_signal(action)

        # returns data for this unchanged state if action is out of range or
        # in an attempt to assign a value to a non-free variable
        if not self.action_space.contains(action) or self.model[var_idx] != 0:
            obs = self.state.encode()
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
            low=0, high=1, shape=(self.state.n_vars, len(self.state.clauses), 2), dtype=np.uint8
        )

        obs = self.state.encode()
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
            low=0, high=1, shape=(self.n_vars, len(self.original_clauses), 2), dtype=np.uint8
        )
        self.state = SATState(self.original_clauses)
        return self.state.encode()

    def render(self, mode='human'):
        print('#vars', self.state.original_clauses.nv)
        print('original_clauses', self.state.original_clauses.original_clauses)
        print('model', self.state.model)
        print(f'sat={self.state.is_sat()}, unsat={self.state.is_unsat()}')
        print(self.state.encode())

    def close(self):
        pass


