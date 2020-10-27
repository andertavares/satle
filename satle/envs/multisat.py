import gym
from .util import create_sat_problem


class MultiSATEnv(gym.Env):
    """
    Gym environment for SAT problems.
    It wraps a sub-environment (e.g. SATGame or LocalSearchSAT) such that at each
    reset, that environment is re-created with a new formula.
    The formulas will be from a specific family, e.g. 3-CNFs on Gnp graphs, and
    are generated with cnfgen.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self, env_class, *problem_args):
        """
        This SAT env wraps a sub-environment such that at each
        reset, that environment is re-created with a new formula.
        The formulas will be from a specific family, e.g. 3-CNFs on Gnp graphs, and
        are generated with cnfgen.

        :param env_class: which environment is to be wrapped (e.g. SATGame, LocalSearchSAT, etc)
        :param problem_args: arguments required by cnfgen to generate problem instances
        (must be passed as if cnfgen is being called from command line, e.g. 'kcolor', 3, 'gnp', 5, 0.5)
        """
        super(MultiSATEnv, self).__init__()

        self.env_class = env_class
        self.problem_args = problem_args
        self.current_instance = None
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


