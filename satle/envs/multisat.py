import gym
from pysat.formula import CNF

from .util import create_sat_problem


class MultiSATEnv(gym.Env):
    """
    Gym environment for SAT problems.
    It wraps a sub-environment (e.g. SATGame or LocalSearchSAT) such that at each
    reset, that environment is re-created with a new formula.
    The formulas will be given by the files from the dataset or
    will be generated with cnfgen, being from a specific family,
    e.g. 3-CNFs on Gnp graphs.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env_class, from_dataset=None, *problem_args):
        """
        This SAT env wraps a sub-environment such that at each
        reset, that environment is re-created with a new formula.
        The formulas will be given by the files from the dataset or
        will be generated with cnfgen, being from a specific family,
        e.g. 3-CNFs on Gnp graphs.

        :param env_class: which environment is to be wrapped (e.g. SATGame, LocalSearchSAT, etc)
        :param from_dataset: list of .cnf DIMACS files to be used in this environment
        :param problem_args: arguments required by cnfgen to generate problem instances
        (must be passed as if cnfgen is being called from command line, e.g. 'kcolor', 3, 'gnp', 5, 0.5)
        """
        super(MultiSATEnv, self).__init__()

        self.env_class = env_class
        self.problem_args = problem_args
        self.current_instance = None
        self.dataset = from_dataset
        self.dataset_index = 0
        self.action_space = None        # not defined until reset
        self.observation_space = None   # not defined until reset

    def step(self, action):
        obs, reward, done, info = self.current_instance.step(action)

        # updates observation and action space in case they're dynamic
        self.observation_space = self.current_instance.observation_space
        self.action_space = self.current_instance.action_space

        return obs, reward, done, info

    def reset(self):
        """
        Re-creates the environment with a new formula and returns its initial state
        :return:
        """
        # creates a new formula and then a new sub-environment with that formula
        if self.dataset is not None:
            clauses = CNF(self.dataset[self.dataset_index]).clauses
            self.dataset_index += 1
            if self.dataset_index >= len(self.dataset):  # wraps when reaches the end
                self.dataset_index = 0
        else:
            clauses = create_sat_problem(*self.problem_args)

        self.current_instance = self.env_class(clauses)

        obs = self.current_instance.reset()
        self.observation_space = self.current_instance.observation_space
        self.action_space = self.current_instance.action_space
        return obs

    def render(self, mode='human'):
        print('Problem: ', self.problem_args)
        self.current_instance.render()

    def close(self):
        self.current_instance.close()

    def seed(self, seed=None):
        self.current_instance.seed(seed)


