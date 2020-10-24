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
        exp_model = np.copy(env.model)
        # flips the first var in the expected model and compares if it is equal to the environment's
        obs, reward, done, info = env.step(0)
        exp_model[0] = -exp_model[0]
        self.assertTrue((exp_model == obs['model']).all(), f'exp=\n{exp_model}\nactual=\n{obs["model"]}')


    def test_step(self):
        f = CNF(from_clauses=[[-1, -2], [2], [-3, -4]])

        env = LocalSearchSAT(f.clauses)
        expected_matrix = np.zeros((4, 3, 2))  # 4 vars x 3 clauses
        expected_matrix[0, 0] = [1, 0]  # -1 on 1st clause
        expected_matrix[1, 0] = [1, 0]  # -2 on 1st clause
        expected_matrix[1, 1] = [0, 1]  # 2 on 2nd clause
        expected_matrix[2, 2] = [1, 0]  # -3 on 3rd clause
        expected_matrix[3, 2] = [1, 0]  # -4 on 3rd clause
        obs = env.reset()
        self.assertTrue((expected_matrix == obs['graph']).all(), f'exp={expected_matrix}, \nactual=\n{obs["graph"]}')
        self.assertEqual([[-1, -2], [2], [-3, -4]], env.original_clauses)

        # forces a model because it is otherwise randomly initiated
        test_values = np.array([1, -1, 1, 1])  # T,F,T,T
        env.model = np.copy(test_values)
        self.assertTrue((test_values == env.model).all(), f'model: {env.model}')


        # will flip variable 2 (index=1), its expected value is 1
        test_values[1] = 1
        # resulting formula is [[1],[-3,-4]], but representation is based on the original
        obs, reward, done, info = env.step(1)
        self.assertTrue((expected_matrix == obs['graph']).all(), f'exp={expected_matrix}, \nactual=\n{obs["graph"]}')
        #self.assertEqual([[], []], info['clauses'])
        self.assertTrue((test_values == obs['model']).all(), f'exp={test_values}, actual={obs["model"]}')
        self.assertEqual(0, reward)
        self.assertFalse(done)
        self.assertEqual(4, env.action_space.n)  # action space does not change

        # will flip variable 1 (index=0), its expected value is -1
        test_values[0] = -1
        # resulting formula is [[]]
        obs, reward, done, info = env.step(0)
        self.assertTrue((expected_matrix == obs['graph']).all(), f'exp={expected_matrix}, \nactual=\n{obs["graph"]}')
        #self.assertEqual([[]], info['clauses'])
        self.assertTrue((test_values == obs['model']).all(), f'exp={test_values}, actual={obs["model"]}')
        self.assertEqual(0, reward)
        self.assertFalse(done)

        # now flips variable 4 (index=3), this will solve the formula
        test_values[3] = -1  # [-1,
        obs, reward, done, info = env.step(3)
        self.assertTrue((expected_matrix == obs['graph']).all(), f'exp={expected_matrix}, \nactual=\n{obs["graph"]}')
        #self.assertEqual([], info['clauses'])
        self.assertTrue((test_values == obs['model']).all(), f'exp={test_values}, actual={obs["model"]}')
        self.assertEqual(1, reward)
        self.assertTrue(done)


if __name__ == '__main__':
    unittest.main()
