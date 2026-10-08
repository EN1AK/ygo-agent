"""Direct decision-model RL experiment. No existing PPO actor is used.

Only the three explicitly known, synthetic starter scenarios are supported.
Reference continuations are macro actions, not unrestricted duel play. Native
cores run in subprocesses because their callbacks are process-global.
"""
from __future__ import annotations

import hashlib
import json
import math
import random
import subprocess
import sys


SCENARIOS = {
    'interaction': {
        'player': 1,
        'goal': 'Prevent Engage returning to the opponent hand when this chain finishes.',
        'context': 'Opponent Kagari has activated its recovery effect targeting Engage in the graveyard. No further opponent responses in this exercise.',
        'actions': {'veiler': 'Activate Effect Veiler on Kagari.',
                    'ogre': 'Activate Ghost Ogre in response to Kagari.',
                    'pass': 'Do not respond.'}},
    'battle': {
        'player': 0, 'goal': 'Win during this battle phase.',
        'context': 'Hayate effect is active. Opponent does not respond. Resolve the selected attack, then end the battle phase.',
        'actions': {'direct': 'Attack directly with Hayate.',
                    'attack_monster': 'Attack Blue-Eyes with Hayate.'}},
    'battle-negated': {
        'player': 0, 'goal': 'Survive this battle phase.',
        'context': 'Effect Veiler has already negated Hayate until end of turn. Opponent does not respond.',
        'actions': {'end_battle': 'End the battle phase.',
                    'attack_monster': 'Attack Blue-Eyes with Hayate.'}},
}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     allow_nan=False).encode('utf-8')).hexdigest()


def visible_state(state, player):
    """Discard hidden identities, physical deck order and opponent hand cards."""
    cards, counts = [], {}
    for card in state['cards']:
        owner, loc, pos = card['player'], card['location'], card['position']
        counts[f'{owner}:{loc}'] = counts.get(f'{owner}:{loc}', 0) + 1
        public = loc in (4, 8, 16, 32) and not (pos & 10)
        own_known = owner == player and loc in (2, 4, 8, 64)
        if public or own_known:
            row = {k: card[k] for k in ('code', 'player', 'location', 'sequence', 'position')}
            row['zone'] = {2:'hand',4:'monster zone',8:'spell/trap zone',16:'graveyard',32:'banished',64:'extra deck'}[loc]
            row['owner'] = 'self' if owner == player else 'opponent'
            row['position_name'] = {1:'face-up attack',2:'face-down attack',4:'face-up defense',8:'face-down defense'}.get(pos, 'other')
            cards.append(row)
    return dict(acting_player=player, lp=state['lp'], phase=state['phase'], turn_player=state['turn_player'],
                winner=state['winner'], cards=cards, zone_counts=counts)


class BranchEnvironment:
    def __init__(self, core, database, scripts, *, max_probes=2, probe_cost=0.02,
                 timeout=45, runner=None):
        if max_probes < 0 or not math.isfinite(probe_cost) or probe_cost < 0 or timeout <= 0:
            raise ValueError('invalid experiment budget')
        self.assets = [str(core), str(database), str(scripts)]
        self.max_probes, self.probe_cost, self.timeout = max_probes, probe_cost, timeout
        self.runner = runner

    def _run(self, branch, root_only=False):
        if self.runner is not None:
            return self.runner(self.case, branch, root_only)
        cmd = [sys.executable, '-m', 'scripts.run_jev_branch', '--case', self.case,
               '--branch', branch, '--core', self.assets[0], '--database', self.assets[1],
               '--scripts', self.assets[2]]
        if root_only:
            cmd.append('--root-only')
        out = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8',
                             timeout=self.timeout, check=True)
        return json.loads(out.stdout)

    def reset(self, case):
        self.case = case
        self.spec = SCENARIOS[case]
        self.probes, self.history, self.evidence = set(), [], []
        self.done = False
        root = self._run(next(iter(self.spec['actions'])), root_only=True)
        self.root_digest = root['root_digest']
        self.runtime = root['runtime']
        self.script_hashes = dict(root.get('script_hashes', {}))
        self.state = root['visible_root']
        self.card_texts = root['card_texts']
        self.evidence.append(root)

    def request(self, rng):
        if self.done:
            raise ValueError('episode is terminal')
        operations = [('commit', a) for a in self.spec['actions']]
        if len(self.probes) < self.max_probes:
            operations += [('probe', a) for a in self.spec['actions'] if a not in self.probes]
        rng.shuffle(operations)
        criteria = {f'option_{i}': ('Simulate only: ' if op == 'probe' else 'Commit: ') +
                    self.spec['actions'][a] for i, (op, a) in enumerate(operations)}
        state = dict(scenario_contract='known synthetic mechanism exercise; fixed opponent and scripted continuations',
                     goal=self.spec['goal'], context=self.spec['context'], root=self.state,
                     card_texts=self.card_texts, simulated_branches=self.history,
                     probes_remaining=self.max_probes-len(self.probes))
        return dict(state=state, criteria=criteria, operations=operations)

    def step(self, operation):
        if self.done:
            raise ValueError('episode is terminal')
        kind, branch = operation
        if branch not in self.spec['actions'] or kind not in ('probe', 'commit'):
            raise ValueError('unknown operation')
        if kind == 'probe' and (branch in self.probes or len(self.probes) >= self.max_probes):
            raise ValueError('probe budget or duplicate violation')
        result = self._run(branch)
        if result['root_digest'] != self.root_digest or result['runtime'] != self.runtime:
            raise ValueError('branch root or runtime drift')
        for name, value in result.get('script_hashes', {}).items():
            if name in self.script_hashes and self.script_hashes[name] != value:
                raise ValueError('card script drift')
            self.script_hashes[name] = value
        if result['status'] not in ('success', 'verified_failure'):
            raise ValueError('unverified branch must not become a reward')
        self.evidence.append(result)
        if kind == 'probe':
            self.probes.add(branch)
            self.history.append(dict(action=self.spec['actions'][branch],
                                     resulting_observation=result['visible_final']))
            return -self.probe_cost
        self.done = True
        self.success = result['status'] == 'success'
        return 1.0 if self.success else -1.0


def returns_to_go(rewards):
    total, values = 0., []
    for reward in reversed(rewards):
        if not math.isfinite(reward):
            raise ValueError('nonfinite reward')
        total += reward
        values.append(total)
    return list(reversed(values))


def leave_one_out_baselines(totals):
    if len(totals) < 2 or not all(math.isfinite(x) for x in totals):
        raise ValueError('need at least two finite on-policy episode returns')
    return [(sum(totals)-value)/(len(totals)-1) for value in totals]
