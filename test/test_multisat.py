import unittest

from pysat.formula import CNF

from satle.envs.localsearch import LocalSearchSAT
from satle.envs.multisat import MultiSATEnv


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


if __name__ == '__main__':
    unittest.main()
