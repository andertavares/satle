import unittest

import numpy as np
from pysat.formula import CNF

from satle.envs.localsearch import LocalSearchSAT


class TestSATEnv(unittest.TestCase):
    def test_encode1(self):
        env = LocalSearchSAT([[-1, 2], [-2, 1]])
        repr = env.encode_state()['graph']
        # variable 1 (index 0) is asserted in clause 1, var 2 is asserted in clause 0
        positives = [(0, 1), (1, 0)]

        # var 1 is negated in clause 0, var 2 is negated in clause 1
        negatives = [(0, 0), (1, 1)]

        for p in positives:
            self.assertTrue((repr[p[0],p[1]] == [0, 1]).all())

        for n in negatives:
            self.assertTrue((repr[n[0], n[1]] == [1, 0]).all())

        # zero elsewhere
        for i in range(len(repr)):
            for j in range(len(repr[i])):
                if (i, j) not in positives and (i, j) not in negatives:
                    self.assertTrue((repr[i, j] == [0, 0]).all() )

    def test_encode_ordered(self):
        env = LocalSearchSAT([[-1, -2], [2], [2, -3, -4]])

        expected_matrix = np.zeros((4, 3, 2))  # 4 vars and 3 clauses
        expected_matrix[0, 0] = [1, 0]  # -1 on 1st clause
        expected_matrix[1, 0] = [1, 0]  # -2 on 1st clause
        expected_matrix[1, 1] = [0, 1]  # 2 on 2nd clause
        expected_matrix[1, 2] = [0, 1]  # 2 on 3rd clause
        expected_matrix[2, 2] = [1, 0]  # -3 on 3rd clause
        expected_matrix[3, 2] = [1, 0]  # -4 on 3rd clause

        actual = env.encode_state()['graph']
        self.assertTrue((expected_matrix == actual).all())

    def test_step_flips_correct_var(self):
        f = CNF(from_clauses=[[-1, -2], [2], [2, -3, -4]])
        env = LocalSearchSAT(f.clauses)
        obs = env.reset()
        exp_model = np.copy(obs['model'])
        # flips the first var in the expected model and compares if it is equal to the environment's
        obs, reward, done, info = env.step(0)
        exp_model[0] = -exp_model[0]
        self.assertTrue((exp_model == obs['model']).all(), f'exp=\n{exp_model}\nactual=\n{obs["model"]}')




    def test_step(self):
        f = CNF(from_clauses=[[-1, -2], [2], [2, -3, -4]])

        env = LocalSearchSAT(f.clauses)

        # forces a model because it is otherwise randomly initiated

        # that's how two np arrays should be compared for equality...
        self.assertTrue((np.zeros(f.nv) == env.model).all(), f'model: {env.model}')
        self.assertEqual([[-1, -2], [2], [2, -3, -4]], env.state.clauses)

        # will add variable 2 (index=1) to solution with positive value
        exp_model = np.zeros(f.nv)
        exp_model[1] = 1

        # resulting original_clauses has a single clause ([-1])
        expected_matrix = np.zeros((4, 3, 2))
        expected_matrix[0, 0] = [1,0]  # -1 on 1st clause
        obs, reward, done, info = env.step(env.encode_action(2, True))

        self.assertTrue((expected_matrix == obs).all(), f'exp={expected_matrix}, \nactual=\n{obs}')
        self.assertEqual([[-1]], info['clauses'])
        self.assertTrue((exp_model == info['model']).all(), f'exp={exp_model}, actual={info["model"]}')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)
        self.assertEqual(8, env.action_space.n)  # action space does not change

        # performing an invalid action keeps the environment at its previous state
        obs, reward, done, info = env.step(8)  # valid actions are 0..7 only
        self.assertTrue((expected_matrix == obs).all())
        self.assertEqual([[-1]], info['clauses'])
        self.assertTrue((exp_model == info['model']).all(), f'exp={exp_model}, actual={info["model"]}')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)
        self.assertEqual(8, env.action_space.n)

        # now, I'll try to set 1 to False and I'll be done with reward 1
        obs, reward, done, info = env.step(env.encode_action(1, False))
        expected_matrix = np.zeros((4, 3, 2))
        exp_model[0] = -1
        self.assertTrue((expected_matrix == obs).all())
        self.assertEqual([], info['clauses'])
        self.assertTrue((exp_model == info['model']).all(), f'exp={exp_model}, actual={info["model"]}')
        self.assertEqual(1, reward)
        self.assertEqual(True, done)
        self.assertEqual(8, env.action_space.n)


if __name__ == '__main__':
    unittest.main()
