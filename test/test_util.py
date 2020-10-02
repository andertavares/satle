import unittest

import numpy as np
from pysat.formula import CNF

from satle.envs.util import unit_propagation, num_vars, vars_and_indices


class TestUtil(unittest.TestCase):
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

    def test_vars_and_indices_ordered(self):
        f = [[-1, -2], [2], [2, -3, -4]]
        var_to_idx, idx_to_var = vars_and_indices(f)
        self.assertEqual({1: 0, 2: 1, 3: 2, 4: 3}, var_to_idx)
        self.assertEqual({0: 1, 1: 2, 2: 3, 3: 4}, idx_to_var)

    def test_vars_and_indices_unordered(self):
        f = [[-7, -3, 5], [-1, -5], [5]]  # same as before, but different labels & clause ordering
        var_to_idx, idx_to_var = vars_and_indices(f)
        self.assertEqual({7: 0, 3: 1, 5: 2, 1: 3}, var_to_idx)
        self.assertEqual({0: 7, 1: 3, 2: 5, 3: 1}, idx_to_var)


if __name__ == '__main__':
    unittest.main()
