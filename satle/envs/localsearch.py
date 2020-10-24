from copy import copy, deepcopy

import gym
from gym import spaces
import numpy as np

from .util import unit_propagation, num_vars, vars_and_indices


class LocalSearchSAT(gym.Env):
    """
    Local search SAT gym environment. It starts with a random assignment to a formula,
    the goal is to flip variable values until a satisfying assignment is found.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, clauses, seed=None):
        """

        :param clauses: a satisfiable original_clauses
        """
        super(LocalSearchSAT, self).__init__()
        # original_clauses and n_vars are static (fixed) throughout execution
        self.original_clauses = clauses
        self.n_vars = num_vars(clauses)

        # if seed is not None, every reset will restore the formula to the same initial state
        self.seed = seed

        # map and reverse map of DIMACS variable to index in factor graph
        self.var_to_idx, self.idx_to_var = vars_and_indices(self.original_clauses)

        # literals is a list: [1,...,n_vars, -1,...,-n_vars] (DIMACS notation without zero)
        self.literals = list(range(1, self.n_vars + 1)) + list(range(-1, -self.n_vars - 1, -1))

        # 1 actions per variable (flip it)
        self.action_space = spaces.Discrete(self.n_vars)

        # obs space contains the factor graph representation matrix and the model
        # matrix is a vars x clauses x 2 matrix with entry i,j  = [0,0],[1,0],[0,1] if
        # var i is absent, negated, asserted in clause j.
        # model is an array with -1,1 if var is False or True in the solution (0 is invalid)
        # more info on gym spaces: https://github.com/openai/gym/tree/master/gym/spaces
        self.observation_space = spaces.Dict({
            'graph': spaces.Box(low=0, high=1, shape=(self.n_vars, len(self.original_clauses), 2), dtype=np.uint8),
            'model': spaces.Box(low=-1, high=1, shape=(self.n_vars,), dtype=np.int8)
        })

        # model is initialized at reset
        self.model = None
        self.reset()

    def apply_model(self):
        """
        Applies the model (current tentative solution)
        to the original formula and returns the resulting formula
        :return:list(list(int))
        """
        # easiest way: unit propagate the model's assignments
        formula = self.original_clauses
        for idx, value in enumerate(self.model):
            if value == 0:  # skips free (unassigned) variables
                continue
            formula = unit_propagation(formula, self.idx_to_var[idx], value == 1)
        return formula

    def is_sat_state(self):
        """
        Returns whether the current assignment satisfies the formula
        :return:
        """
        # if unit-propagating the assignments made the formula empty, it is a satisfying one
        return len(self.apply_model()) == 0

    def reward(self):
        """
        Returns the reward for the current state
        :return: 1 if this state corresponds to a sat assignment, else 0
        """
        return 1 if self.is_sat_state() else 0

    def encode_state(self):
        """
        Encodes the state as a factor graph and returns a v x c x 2
        representation matrix.
        Position i,j of the matrix contains
        - positive edge [0,1] if var i is asserted in clause j
        - negative edge [1,0] if var i is negated in clause j
        - no edge [0,0] if var i is not present in clause j
        Variable indexes in the clauses are according to their occurence.
        E.g., if clauses is: [[-5, 1], [2, -7, 5]] then the index of
        variables 5,1,2,7 become 0,1,2,3 respectively.
        :return: np.array with the adjacency matrix (#vars x #clauses x 2), with [0,0], [1,0], [0,1] in position v,c if variable v is absent/asserted/negated in clause c
        """

        # v x c x 2 tensor (v=#vars, c=#clauses)
        repr = np.zeros((len(self.var_to_idx), len(self.original_clauses), 2), dtype=np.uint8)

        for c_num, clause in enumerate(self.original_clauses):
            for literal in clause:
                var_idx = self.var_to_idx[abs(literal)]
                repr[var_idx][c_num] = [1,0] if literal < 0 else [0,1]

        return {'graph': repr, 'model': self.model}

    def step(self, action):
        """
        Processes the action (selected variable) by adding flipping the corresponding bit
        partial solution and returning the resulting original_clauses
        and associated reward, done and info.
        :param action:int
        :return:observation,reward,done,info
        """

        # returns data for this unchanged state if action is out of range or
        # in an attempt to assign a value to a non-free variable
        if not self.action_space.contains(action):
            obs = self.encode_state()
            info = {'clauses': self.apply_model()}
            return obs, self.reward(), self.is_sat_state(), info

        # flips the corresponding bit
        self.model[action] = -self.model[action]

        # creates new original_clauses with the result of adding the corresponding literal to the previous
        # new_clauses = unit_propagation(self.state.clauses, var_idx, var_sign)

        obs = self.encode_state()
        info = {'clauses': self.apply_model()}
        return obs, self.reward(), self.is_sat_state(), info

    def reset(self):
        """
        Resets to the initial state and returns it
        :return:
        """

        # if seed is not None, will generate the same model everytime reset is called
        if self.seed is not None:
            np.random.seed(self.seed)
        # each position of the model stores 1 or -1 for True or False assignment to the var
        # starts with a random unsatisfying model
        self.model = np.array(np.random.choice([-1, 1], self.n_vars), dtype=np.int8)
        while self.is_sat_state():  # repeats the assignment until unsat
            self.model = np.array(np.random.choice([-1, 1], self.n_vars), dtype=np.int8)

        return self.encode_state()

    def render(self, mode='human'):
        print('#vars', self.action_space.n)
        print('clauses', self.apply_model())
        print('model', self.model)
        print(f'sat={self.is_sat_state()}')
        print(self.encode_state())

    def close(self):
        pass


