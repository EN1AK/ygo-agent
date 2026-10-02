"""Replay reference backend using the existing single-checkpoint evaluator."""
from contextlib import contextmanager
import json

import numpy as np

from ygoai.rl.candidate_search import SearchFailure, SearchPosition
from ygoai.rl.counterfactual import observation_digest, unpack_step


def scalar(value):
    return np.asarray(value).reshape(-1)[0].item()


class ReplaySearchBackend:
    information_mode = 'oracle_exact_state'

    def __init__(self, model):
        self.model = model
        # Replay snapshot storage is the seed/action record, not a native arena.
        self.snapshot_bytes = len(json.dumps(model.row, sort_keys=True).encode())
        self.closed = False
        self.active = 0

    def close(self):
        if self.active:
            raise SearchFailure('cannot_close_backend_with_active_branch')
        self.closed = True

    @contextmanager
    def branch(self, particle, seed):
        if self.closed:
            raise SearchFailure('closed_backend')
        if particle < 0:
            raise SearchFailure('invalid_particle')
        env = self.model.make_env()
        branch = None
        self.active += 1
        try:
            branch = ReplayBranch(self.model, env)
            for action in self.model.row['snapshot']['actions']:
                branch.step(int(action))
                if branch.terminal is not None:
                    raise SearchFailure('snapshot_prefix_terminated')
            yield branch
        finally:
            if branch is not None:
                branch.closed = True
            try:
                env.close()
            finally:
                self.active -= 1


class ReplayBranch:
    def __init__(self, model, env):
        self.model, self.env = model, env
        self.closed = False
        self.obs, self.info = env.reset()
        self.ra = model.search_agent.init_rnn_state(1)
        self.rb = model.search_agent.init_rnn_state(1)
        self.terminal = None
        self.cached = None

    def evaluate(self):
        if self.closed:
            raise SearchFailure('closed_branch')
        if self.cached is not None:
            return self.cached
        if self.terminal is not None:
            self.cached = SearchPosition(None, (), (), None, observation_digest(self.obs),
                                         'terminal', self.terminal)
            return self.cached
        player, self.ra, self.rb, logits, value = self.model.advance_model(
            self.obs, self.info, self.ra, self.rb, evaluator='root')
        count = int(scalar(self.info['num_options']))
        if count < 1:
            raise SearchFailure('empty_live_menu')
        selection = self.obs.get('selection_')
        prompt_id = int(np.asarray(selection)[0, 0]) if selection is not None else 0
        if not prompt_id:
            prompt_id = int(np.asarray(self.obs['actions_'])[0, 0, 3])
        self.cached = SearchPosition(int(player), tuple(range(count)),
                                     tuple(float(x) for x in np.asarray(logits)[0,:count]),
                                     float(scalar(value)), observation_digest(self.obs),
                                     'select_chain' if prompt_id==2 else f'prompt-{prompt_id}')
        return self.cached

    def step(self, action):
        position = self.evaluate()  # Advance the acting player's RNN exactly once.
        if position.terminal_root_return is not None:
            raise SearchFailure('step_after_terminal')
        if action not in position.legal_actions:
            raise SearchFailure('illegal_branch_action')
        self.obs, reward, done, self.info = unpack_step(self.env.step(np.asarray([action])))
        self.cached = None
        if bool(scalar(done)):
            if int(scalar(self.info['invalid_game'])) or int(scalar(self.info['termination_reason'])) != 1:
                raise SearchFailure('invalid_branch_terminal')
            value = float(scalar(reward))
            if value not in (-1.,0.,1.):
                raise SearchFailure('terminal_reward_scale')
            self.terminal = value if position.player==self.model.root_player else -value
