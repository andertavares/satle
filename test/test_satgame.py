import unittest

import numpy as np
from pysat.formula import CNF

from satle.envs.satgame import SATEnv


class TestSATEnv(unittest.TestCase):

    def test_encode_action(self):
        env = SATEnv([[-1, -2], [2], [2, -3, -4]])

        for var in range(0, 4):  # true assignments
            self.assertEqual(var, env.encode_action(var, True), msg=f'expected={var}')

        for var in range(0, 4):  # false assignments
            self.assertEqual(4+var, env.encode_action(var, False), msg=f'expected={var}')

    def test_step(self):
        f = CNF(from_clauses=[[-1, -2], [2], [2, -3, -4]])

        env = SATEnv(f.clauses)
        initial_state = env.state  #env.reset()

        # that's how two np arrays should be compared for equality...
        self.assertTrue((np.zeros(f.nv) == env.model).all(), f'model: {env.model}')

        self.assertEqual([[-1, -2], [2], [2, -3, -4]], env.state.clauses)

        # will add variable 2 (index=1) to solution with positive value
        exp_model = np.zeros(f.nv)
        exp_model[1] = 1

        # resulting original_clauses has a single clause ([-1])
        expected_matrix = np.zeros((1, 1))
        expected_matrix[0, 0] = -1  # -1 on 1st clause
        obs, reward, done, info = env.step(env.encode_action(1, True))

        self.assertTrue((expected_matrix == obs).all())
        self.assertEqual([[-1]], info['clauses'])
        self.assertTrue((exp_model == info['model']).all(), f'exp={exp_model}, actual={info["model"]}')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)

        # if I try to set 2 to True again, the state will be the same
        obs, reward, done, info = env.step(env.encode_action(1, True))
        self.assertTrue((expected_matrix == obs).all())
        self.assertEqual([[-1]], info['clauses'])
        self.assertTrue((exp_model == info['model']).all(), f'exp={exp_model}, actual={info["model"]}')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)

        # now, I'll try to set 1 to False and I'll be done with reward 1
        obs, reward, done, info = env.step(env.encode_action(0, False))
        expected_matrix = np.zeros((0, 0))  # empty matrix
        exp_model[0] = -1
        self.assertTrue((expected_matrix == obs).all())
        self.assertEqual([], info['clauses'])
        self.assertTrue((exp_model == info['model']).all(), f'exp={exp_model}, actual={info["model"]}')
        self.assertEqual(1, reward)
        self.assertEqual(True, done)


if __name__ == '__main__':
    unittest.main()
