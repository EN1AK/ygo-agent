from types import SimpleNamespace
import unittest

import numpy as np

from ygoai.rl.candidate_search import SearchFailure
from ygoai.rl.replay_search import ReplaySearchBackend


class FakeEnv:
    def __init__(self):
        self.turn = 0
        self.closed = False

    def view(self):
        return {'actions_': np.zeros((1,1,4),dtype=np.uint8),
                'selection_':np.array([[2 if self.turn==1 else 1]])}

    def info(self):
        return {'to_play':[self.turn%2], 'num_options':[1],
                'invalid_game':[0], 'termination_reason':[1 if self.turn==3 else 0]}

    def reset(self):
        return self.view(), self.info()

    def step(self, action):
        self.turn += 1
        return self.view(), np.array([1. if self.turn==3 else 0.]), np.array([self.turn==3]), self.info()

    def close(self):
        self.closed = True


class FakeModel:
    root_player = 0
    search_agent = SimpleNamespace(init_rnn_state=lambda _:0)

    def __init__(self, prefix=()):
        self.row={'snapshot':{'seed':1,'actions':list(prefix)}}
        self.calls=[]

    def make_env(self):
        self.env=FakeEnv()
        return self.env

    def advance_model(self, obs, info, ra, rb, evaluator):
        player=info['to_play'][0]
        self.calls.append((player,ra,rb))
        return player,ra+int(player==0),rb+int(player==1),np.array([[0.]]),np.array([.25])


class ReplaySearchTest(unittest.TestCase):
    def test_separate_recurrent_states_and_evaluation_cache(self):
        model=FakeModel(prefix=[0])
        backend=ReplaySearchBackend(model)
        with backend.branch(0,4) as branch:
            self.assertEqual(branch.evaluate().prompt,'select_chain')
            branch.evaluate()
            branch.step(0)
            branch.step(0)
            terminal=branch.evaluate()
            self.assertEqual(terminal.terminal_root_return,1.)
        self.assertEqual(model.calls,[(0,0,0),(1,1,0),(0,1,1)])
        self.assertTrue(model.env.closed)
        self.assertEqual(backend.active,0)
        with self.assertRaises(SearchFailure):
            branch.evaluate()
        backend.close()
        with self.assertRaises(SearchFailure):
            with backend.branch(0,4):
                pass

    def test_bad_prefix_releases_environment(self):
        model=FakeModel(prefix=[9])
        backend=ReplaySearchBackend(model)
        with self.assertRaises(SearchFailure):
            with backend.branch(0,4):
                pass
        self.assertTrue(model.env.closed)
        self.assertEqual(backend.active,0)

    def test_terminal_prefix_rejected_and_active_close_rejected(self):
        model=FakeModel(prefix=[0,0,0])
        backend=ReplaySearchBackend(model)
        with self.assertRaisesRegex(SearchFailure,'snapshot_prefix_terminated'):
            with backend.branch(0,4):
                pass
        self.assertTrue(model.env.closed)
        model.row['snapshot']['actions']=[]
        with backend.branch(0,4):
            with self.assertRaises(SearchFailure):
                backend.close()


if __name__ == '__main__':
    unittest.main()
