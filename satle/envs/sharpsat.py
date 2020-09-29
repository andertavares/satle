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
