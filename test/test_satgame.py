import unittest

import numpy as np
from pysat.formula import CNF

from satle.envs.satgame import SATEnv
from satle.envs.util import encode


class TestSATState(unittest.TestCase):
    def test_encode(self):
        formula = CNF(from_clauses=[[-1, 2], [-2, 1]])
        adj_matrix = encode(formula.nv, formula.clauses)
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


class TestSATEnv(unittest.TestCase):

    def test_step(self):
        f = CNF(from_clauses=[[-1, -2], [2], [2, -3, -4]])

        env = SATEnv(f)
        initial_state = env.state  #env.reset()

        # that's how two np arrays should be compared for equality...
        self.assertTrue((np.zeros(f.nv) == initial_state.model).all(), f'model: {initial_state.model}')

        self.assertEqual([[-1, -2], [2], [2, -3, -4]], initial_state.formula.clauses)

        # will add variable 2 to solution with positive polarity
        exp_done = False
        exp_model = np.zeros(f.nv)
        exp_model[1] = 1
        obs, reward, done, info = env.step(env.encode_action(2, True))
        self.assertEqual([[-1]], info['clauses'])  #not testing obs.formula
        self.assertTrue((exp_model == obs['model']).all(), f'exp={exp_model}, actual={obs["model"]}')
        self.assertEqual(0, reward)
        self.assertEqual(False, done)



if __name__ == '__main__':
    unittest.main()
