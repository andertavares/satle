import unittest

import subprocess
from pysat.formula import CNF
from pysat.solvers import Solver

from satle.envs.util import unit_propagation, num_vars, vars_and_indices, create_sat_problem


class MockupCall:
    """
    Fakes a subprocess.call routine
    """
    def __init__(self):
        pass

    def set_desired_formula(self, clauses, output_file):
        self.formula = CNF(from_clauses=clauses)
        self.output_file = output_file

    def call(self, *args):
        """
        Writes the pre-set formula to the pre-set output file
        :return:
        """
        self.formula.to_file(self.output_file)


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

    def test_desired_formula(self):
        """
        Checks if create_sat_problem returns the desired formula
        :return:
        """
        clauses = [[-7, -3, 5], [-1, -5], [5]]

        # sets up a fake subprocess.call that will write the desired formula
        fake_call = MockupCall()
        fake_call.set_desired_formula(clauses, 'tmp.cnf')
        old_call = subprocess.call
        subprocess.call = fake_call.call
        self.assertEqual(clauses, create_sat_problem(True, 'foo', 'bar'))

        # returns things back to normal
        subprocess.call = old_call

    def test_create_satifiable_problems(self):
        """
        Checks whether create_sat_problem returns a satisfiable formula
        when prompted to do so
        :return:
        """
        # runs 10 attempts to generate satisfiable problems
        for _ in range(10):
            clauses = create_sat_problem(True, 'kcolor', 3, 'gnp', 20, 0.2)
            with Solver(name='Glucose3', bootstrap_with=clauses) as solver:
                self.assertTrue(solver.solve())

    def test_create_problems(self):
        """
        Checks whether create_sat_problem returns a both sat and unsat problems
        :return:
        """
        # runs 10 attempts to generate satisfiable problems
        sat, unsat = 0, 0
        for _ in range(10):
            # creates 3-CNFs with 20 vars & 90 clauses (phase transition)
            clauses = create_sat_problem(False, 'randkcnf', 3, 20, 91)
            with Solver(name='Glucose3', bootstrap_with=clauses) as solver:
                if solver.solve():
                    sat += 1
                else:
                    unsat += 1
        self.assertTrue(sat > 0)
        self.assertTrue(unsat > 0)


if __name__ == '__main__':
    unittest.main()
