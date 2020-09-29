from copy import copy

import gym
from gym import spaces
from pysat.formula import CNF
from itertools import compress


def unit_propagation(formula, literal):
    """
    Performs unit propagation of literal l in formula f.
    That is, removes all clauses with l and removes ~l from the clauses it occurs
    :param formula:
    :param literal:
    :return:
    """
    new_f = formula.copy()

    # removes clauses with literal
    occurrences = [literal not in c for c in new_f.clauses]  # creates an array mask to keep clauses without literal
    new_f.clauses = list(compress(new_f.clauses, occurrences))  # filters the old list with the mask

    # removes occurrences of ~literal
    for c in new_f.clauses:
        # python triggers ValueError if the element is not on the list
        try:
            c.remove(-literal)
        except ValueError:
            pass  # ignore the error
    return new_f


class SATState:
    def __init__(self, formula, model):
        """
        Creates a new state with a formula and a model (partial or full)
        :param formula: pysat.formula.CNF
        :param model: dict with a partial or full assignment of values to variables
        """
        self.formula = formula.copy()
        self.model = copy(model)

    def valid_actions(self, positives_only=True):
        """
        Returns the valid actions that can be performed in this state
        :return:
        """
        free_literals = [v for v in range(1, self.formula.nv + 1) if v not in self.model]
        if positives_only:
            return free_literals

        free_literals += [-v for v in range(1, self.formula.nv + 1) if v not in self.model]
        return free_literals

    def apply_action(self, action):
        """
        Returns the two resulting states of applying action to the current state.
        This corresponds to branching on the variable denoted by 'action'.
        The resulting states correspond to the branch nodes resulting by branching on
        asserted and negated action
        :param action: int, denotes the variable to branch on
        :return: tuple(SATState, SATState) containing the two resulting states
        """
        # creates two new models from the current model
        m1, m2 = copy(self.model), copy(self.model)

        # adds the asserted literal to m1 and the negated to m2
        m1[abs(action)] = action
        m2[abs(action)] = -action

        # creates two formulas with the result of adding action to m1 and -action to m2
        f1 = unit_propagation(self.formula, action)
        f2 = unit_propagation(self.formula, -action)

        # returns the two resulting states
        return SATState(f1, m1), SATState(f2, m2)

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


def encode(formula):
    """
    TODO encodes a formula into a graph
    :param formula:
    :return:
    """
    raise NotImplementedError


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
        self.formula = CNF(from_file=formula)
        self.state = SATState(self.formula, {})

        # literals is a list: [-nv,...,-1, +1, ..., nv] (without zero)
        self.literals = list(range(-self.formula.nv, 0)) + list(range(1, self.formula.nv+1))
        # 2 actions per variable (use it asserted or negated)
        self.action_space = spaces.Discrete(2 * self.formula.nv)

        # TODO: define the obs. space (see: https://github.com/openai/gym/tree/master/gym/spaces)
        # use dict of MultiBinary for the bi-adjacency matrix? Selsam's? Cameron's?
        self.observation_space = None

    def get_literal(self, action):
        """
        Translates an action into the corresponding literal,
        i.e., from [0,..., 2* num_vars] to [-num_vars, ...,-1, +1,...,num_vars]
        :param action:
        :return:
        """
        return self.literals[action]

    def step(self, action):
        """
        Processes the action (selected variable) by adding it to the
        partial solution and returning the resulting formula
        and associated reward, done and info.
        :param action:
        :return:
        """
        lit = self.get_literal(action)
        new_model = copy(self.state.model)

        # adds the literal to the partial solution
        new_model[abs(lit)] = lit

        # creates two formulas with the result of adding action to m1 and -action to m2
        new_formula = unit_propagation(self.formula, lit)

        self.state = SATState(new_formula, new_model)

        # reward: 1, -1, 0 for sat, unsat, non-terminal, respectively
        reward = 1 if self.state.is_sat() else -1 if self.state.is_unsat() else 0
        info = {'formula': self.state.formula, 'model': self.state.model}
        return encode(self.state), reward, self.state.terminal(), info

    def reset(self):
        """
        Resets to the initial state and returns it
        :return:
        """
        self.state = SATState(self.formula, {})  # resets internal state
        return encode(self.state)

    def render(self, mode='human'):
        pass

    def close(self):
        pass


class SharpSATEnv:
    """
    A gym-like interface for solving a #SAT problem (i.e. couting #models)
    """

    def __init__(self, formula):
        """
        Initializes the environment with the formula to be model-counted.
        The initial state corresponds to the original formula and an empty model
        :param formula:
        """

        self.initial_state = SATState(formula, {})
        self.solutions = 0  # number of solutions found so far
        self.n_vars = formula.nv  # number of variables in the original formula

    def count_models(self, state):
        """
        Counts the number of solutions in this state.
        Only sat states have solutions (equivalent to the number of free variables).
        :param state:
        :return:
        """
        if state.is_sat():
            return 2 ** len([free for free in range(1, self.n_vars + 1) if free not in state.model])
        return 0

    def step(self, state, action):
        """
        Implements the action (variable to branch on) in the given state (formula).
        The action incurs in two new states, which are the result of applying
        unit propagations on state + {action} and state + {-action} respectively.
        The list of transitions is returned afterwards.
        Each transition is a tuple(next_state, reward, done, info).
        Currently, info is an empty dict
        :param state:
        :param action:
        :return: list of transitions: tuple(next_state, reward, done, info)
        """
        s1, s2 = state.apply_action(action)

        for next_s in [s1, s2]:
            self.solutions += self.count_models(next_s)

        return [(s1, s1.reward(), s1.terminal(), {'model': s1.model}),
                (s2, s2.reward(), s2.terminal(), {'model': s2.model})]

    def reset(self):
        self.solutions = 0
        return self.initial_state

    def render(self, mode='human'):
        raise NotImplementedError

    def close(self):
        raise NotImplementedError
