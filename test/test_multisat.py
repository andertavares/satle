import unittest

from pysat.formula import CNF

from satle.envs import util
from satle.envs.localsearch import LocalSearchSAT
from satle.envs.multisat import MultiSATEnv
from satle.envs.satgame import SATGame


class FakeUtil:
    """
    To emulate a method in util package (create_sat_problem)
    """
    def __init__(self, instances):
        self.instances = instances
        self.index = -1  # will be incremented to zero below
        self.sat_required = None
        self.args = None

    def fake_create_sat_problem(self, sat_required, *args):
        # registers received params
        self.sat_required = sat_required
        self.args = args

        # increments index and return an instance
        self.index += 1
        return self.instances[self.index % len(self.instances)]


class TestMultiSAT(unittest.TestCase):
    def test_from_dataset(self):
        """
        Simple test with a 2-instance dataset
        :return:
        """
        dataset = [
            'instances/sat_00001_k3_v20_c91.cnf',
            'instances/sat_00002_k3_v20_c91.cnf'
        ]
        env = MultiSATEnv(LocalSearchSAT, from_dataset=dataset)
        self.assertEqual(dataset, env.dataset)
        self.assertEqual(0, env.dataset_index)

        # at reset, the environment will load the first instance
        env.reset()
        self.assertEqual(CNF(dataset[0]).clauses, env.current_instance.original_clauses)
        self.assertEqual(1, env.dataset_index)

        # now the second instance
        env.reset()
        self.assertEqual(CNF(dataset[1]).clauses, env.current_instance.original_clauses)
        self.assertEqual(0, env.dataset_index)      # index wrapped around

        # first instance again (wrapped)
        env.reset()
        self.assertEqual(CNF(dataset[0]).clauses, env.current_instance.original_clauses)
        self.assertEqual(1, env.dataset_index)

    def test_with_cnfgen(self):
        instances = [
            [[1, 2], [-1, 3]],
            [[-1, -2], [1, -3]],
        ]
        futil = FakeUtil(instances)
        util.create_sat_problem = futil.fake_create_sat_problem

        env = MultiSATEnv(SATGame, cnfgen_args=['kcolor', 3, 'gnp', 20, 0.4])

        env.reset()  # will retrieve the first instance
        self.assertEqual(instances[0], env.current_instance.original_clauses)

        env.reset()  # will retrieve the 2nd instance
        self.assertEqual(instances[1], env.current_instance.original_clauses)

        env.reset()  # will retrieve the first instance again (wrapped in FakeUtils)
        self.assertEqual(instances[0], env.current_instance.original_clauses)


if __name__ == '__main__':
    unittest.main()
