"""Opt-in eligibility wrapper. Default PPO/evaluation paths do not import it."""
from dataclasses import dataclass
import math

import numpy as np

from ygoai.rl.candidate_search import SearchBudget, candidate_search, _probabilities


@dataclass(frozen=True)
class SearchPolicy:
    budget: SearchBudget = SearchBudget()
    minimum_entropy: float = .1
    strategic_prompts: tuple[str,...] = ('select_chain','select_idlecmd','select_battlecmd')
    policy_kl: float = 1.
    magnet_kl: float = 0.
    fair_play: bool = True
    strict: bool = False

    def choose(self, root, backend, *, seed, external_opponent=False):
        if not math.isfinite(self.minimum_entropy) or self.minimum_entropy<0:
            raise ValueError('Entropy threshold must be finite and nonnegative')
        raw=_probabilities(root.logits)
        entropy=float(-np.sum(raw*np.log(np.maximum(raw,1e-300))))
        raw_action=root.legal_actions[int(np.argmax(raw))]
        reason=None
        if external_opponent: reason='unsupported_external_opponent_state'
        elif len(root.legal_actions)<2: reason='single_legal_action'
        elif root.prompt not in self.strategic_prompts and entropy<self.minimum_entropy:
            reason='ineligible_prompt_and_low_entropy'
        if reason:
            return {'schema':'ygo-search-policy-v1','eligible':False,'completed':False,
                    'raw_action':raw_action,'selected_action':raw_action,'fallback_reason':reason,
                    'raw_logits':list(root.logits),'raw_value':root.value,'raw_policy':raw.tolist(),
                    'legal_actions':list(root.legal_actions),'entropy':entropy,
                    'root_observation_digest':root.observation_digest}
        report=candidate_search(root,backend,self.budget,seed=seed,policy_kl=self.policy_kl,
                                magnet_kl=self.magnet_kl,fair_play=self.fair_play,strict=self.strict)
        return {'schema':'ygo-search-policy-v1','eligible':True,'completed':report['completed'],
                'raw_action':raw_action,'selected_action':report['selected_action'],
                'fallback_reason':report['fallback_reason'],'entropy':entropy,'search':report}
