import unittest

from pysat.formula import CNF

from satle.envs.sharpsat import SharpSATEnv, SATState


class TestSATState(unittest.TestCase):
    def test_apply_action(self):
        f = CNF(from_clauses=[[-1, -2], [2], [2, -3, -4]])
        s = SATState(formula=f, model={})

        # branches on variable '2' on s
        next_s1, next_s2 = s.apply_action(2)
        expected_clauses1 = [[-1]]
        expected_clauses2 = [[], [-3, -4]]
        self.assertEqual({2: 2}, next_s1.model)
        self.assertEqual(expected_clauses1, next_s1.formula.clauses, msg='next_s1')

        self.assertEqual({2: -2}, next_s2.model)
        self.assertEqual(expected_clauses2, next_s2.formula.clauses, msg='next_s2')


class TestSharpSATEnv(unittest.TestCase):

    def test_step(self):
        f = CNF(from_clauses=[[-1, -2], [2], [2, -3, -4]])

        env = SharpSATEnv(f)
        initial_state = env.reset()

        self.assertEqual({}, initial_state.model)
        self.assertEqual([[-1, -2], [2], [2, -3, -4]], initial_state.formula.clauses)

        # will select variable 2 to branch on, these are the expected results
        exp_clauses = [
            [[-1]],
            [[], [-3, -4]]
        ]
        exp_rewards = [-1, 0]
        exp_done = [False, True]
        exp_models = [{2: 2}, {2: -2}]
        transitions = env.step(initial_state, 2)

        for idx, (state, reward, done, info) in enumerate(transitions):
            self.assertEqual(exp_clauses[idx], state.formula.clauses)
            self.assertEqual(exp_rewards[idx], reward)
            self.assertEqual(exp_done[idx], done)
            self.assertEqual(exp_models[idx], info['model'])


if __name__ == '__main__':
    unittest.main()
