import unittest

import numpy as np
from pysat.formula import CNF

from satle.envs.satgame import SATEnv, SATState


class TestSATState(unittest.TestCase):
    def test_encode1(self):
        state = SATState([[-1, 2], [-2, 1]])
        adj_matrix = state.encode()
        # variable 1 (index 0) is asserted in clause 1, var 2 is asserted in clause 0
        positives = [(0, 1), (1, 0)]

        # var 1 is negated in clause 0, var 2 is negated in clause 1
        negatives = [(0, 0), (1, 1)]

        for p in positives:
            self.assertEqual(1, adj_matrix[p[0], p[1]])

        for n in negatives:
            self.assertEqual(-1, adj_matrix[n[0], n[1]])

        # zero elsewhere
        for i in range(len(adj_matrix)):
            for j in range(len(adj_matrix[i])):
                if (i, j) not in positives and (i, j) not in negatives:
                    self.assertEqual(0, adj_matrix[i, j])

    def test_encode_ordered(self):
        state = SATState([[-1, -2], [2], [2, -3, -4]])

        expected_matrix = np.zeros((4, 3))  # 4 vars and 3 clauses
        expected_matrix[0, 0] = -1  # -1 on 1st clause
        expected_matrix[1, 0] = -1  # -2 on 1st clause
        expected_matrix[1, 1] = 1  # 2 on 2nd clause
        expected_matrix[1, 2] = 1  # 2 on 3rd clause
        expected_matrix[2, 2] = -1  # -3 on 3rd clause
        expected_matrix[3, 2] = -1  # -4 on 3rd clause

        actual = state.encode()
        self.assertTrue((expected_matrix == actual).all())

    def test_encode_unordered(self):
        """
        Same original_clauses as test_encode_ordered, but variables occur in different order
        and are labeled differently
        :return:
        """
        clauses = [[-7, -3, 5], [-1, -5], [5]]

        expected_matrix = np.zeros((4, 3))  # there are 4 vars and 3 original_clauses
        expected_matrix[0, 0] = -1  # -7 on 1st clause
        expected_matrix[1, 0] = -1  # -3 on 1st clause
        expected_matrix[2, 0] = 1  # 5 on 1st clause
        expected_matrix[3, 1] = -1  # -1 on 2nd clause
        expected_matrix[2, 1] = -1  # -5 on 2nd clause
        expected_matrix[2, 2] = 1  # 5 on 3rd clause
        actual = SATState(clauses).encode()
        self.assertEqual(expected_matrix.shape, actual.shape)
        self.assertTrue((expected_matrix == actual).all())


class TestSATEnv(unittest.TestCase):
    # TODO write a test var_and_signal after state changes

    def test_encode_action(self):
        env = SATEnv([[-1, -2], [2], [2, -3, -4]])

        for idx in range(0, 4):  # true assignments
            self.assertEqual(idx, env.encode_action(idx+1, True), msg=f'expected={idx}')

        for idx in range(0, 4):  # false assignments
            self.assertEqual(4+idx, env.encode_action(idx+1, False), msg=f'expected={idx}')

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
        obs, reward, done, info = env.step(env.encode_action(2, True))

        self.assertTrue((expected_matrix == obs).all())
        self.assertEqual([[-1]], info['clauses'])
        self.assertTrue((exp_model == info['model']).all(), f'exp={exp_model}, actual={info["model"]}')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)
        self.assertEqual(2, env.action_space.n)  #two actions (set the single remaning var True or False)

        # performing an invalid action keeps the environment at its previous state
        obs, reward, done, info = env.step(2)  # valid actions are 0,1 only
        self.assertTrue((expected_matrix == obs).all())
        self.assertEqual([[-1]], info['clauses'])
        self.assertTrue((exp_model == info['model']).all(), f'exp={exp_model}, actual={info["model"]}')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)
        self.assertEqual(2, env.action_space.n)  # two actions (set the single remaning var True or False)

        # now, I'll try to set 1 to False and I'll be done with reward 1
        obs, reward, done, info = env.step(env.encode_action(1, False))
        expected_matrix = np.zeros((0, 0))  # empty matrix
        exp_model[0] = -1
        self.assertTrue((expected_matrix == obs).all())
        self.assertEqual([], info['clauses'])
        self.assertTrue((exp_model == info['model']).all(), f'exp={exp_model}, actual={info["model"]}')
        self.assertEqual(1, reward)
        self.assertEqual(True, done)
        self.assertEqual(0, env.action_space.n)


if __name__ == '__main__':
    unittest.main()
