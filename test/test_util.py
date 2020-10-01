import unittest

import numpy as np
from pysat.formula import CNF

from satle.envs.util import encode, unit_propagation, num_vars


class TestUtil(unittest.TestCase):
    """
    TODO test vars_and_indices
    """
    def test_encode1(self):
        formula = CNF(from_clauses=[[-1, 2], [-2, 1]])
        adj_matrix = encode(formula.clauses)
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
        f = CNF(from_clauses=[[-1, -2], [2], [2, -3, -4]])

        expected_matrix = np.zeros((f.nv, len(f.clauses)))
        expected_matrix[0, 0] = -1  # -1 on 1st clause
        expected_matrix[1, 0] = -1  # -2 on 1st clause
        expected_matrix[1, 1] = 1  # 2 on 2nd clause
        expected_matrix[1, 2] = 1  # 2 on 3rd clause
        expected_matrix[2, 2] = -1  # -3 on 3rd clause
        expected_matrix[3, 2] = -1  # -4 on 3rd clause

        actual = encode(f.clauses)
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
        actual = encode(clauses)
        self.assertEqual(expected_matrix.shape, actual.shape)
        self.assertTrue((expected_matrix == actual).all())

    def test_unit_propagation_until_empty(self):
        f = [[-1, -2], [2]]
        # assigning True to 2
        f = unit_propagation(f, 1, 1)
        self.assertEqual([[-1]], f)

        #  assigning False to 1
        f = unit_propagation(f, 0, -1)
        self.assertEqual([], f)

    def test_unit_propagation_multiple_clauses_removed(self):
        f = [[-1, -2], [2], [2, -3, -4]]
        # setting 2 to True
        f = unit_propagation(f, 1, 1)
        self.assertEqual([[-1]], f)

    def test_unit_propagation_negated_literal(self):
        f = [[-1, -2], [2], [2, -3, -4]]
        # setting 2 to false
        f = unit_propagation(f, 1, -1)
        self.assertEqual([[], [-3, -4]], f)

    def test_num_vars(self):
        """
        Counts the actual number of variables in the original_clauses
        :param clauses:
        :return:
        """
        self.assertEqual(4, num_vars([[-1, -2], [2], [2, -3, -4]]))
        self.assertEqual(2, num_vars([[-1, -4], [1], [1, 4]]))
        self.assertEqual(2, num_vars([[-1, -4], [], [1, 4]]))
        self.assertEqual(0, num_vars([]))
        self.assertEqual(0, num_vars([[], []]))

if __name__ == '__main__':
    unittest.main()
