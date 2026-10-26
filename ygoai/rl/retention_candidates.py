"""Observed goal regressions are search candidates, never action supervision."""
import copy
from collections import Counter


def strict_outcomes(record,goals,controlled):
    cards=[c for c in record['final']['cards'] if c['player']==controlled]
    monsters=Counter(c['code'] for c in cards if c['location']==4 and c['position']&5)
    spells=Counter(c['code'] for c in cards if c['location']==8 and c['position']&5)
    hand=Counter(c['code'] for c in cards if c['location']==2)
    return [dict(monsters=not(Counter(g['monsters'])-monsters),spells=not(Counter(g['spells'])-spells),wind_in_hand=hand[56651978]>=1,hand_resources=sum(hand.values())>=g['hand']) for g in goals]


def normal_regression(before,after,case):
    assert before['schema']==after['schema']=='ygo-elfnote-search-route-v1'
    assert before['case']==after['case']==case['name']
    assert before['definition']==after['definition']==case['definition'],'Different benchmark contexts are not a retention comparison'
    assert before['status']=='success' and after['status']=='goal_not_reached','Only completed observed goal loss is a regression candidate'
    controlled=case['definition']['controlled'];assert controlled==0
    positive=strict_outcomes(before,case['goals'],controlled);negative=strict_outcomes(after,case['goals'],controlled)
    assert any(all(r.values()) for r in positive) and not any(all(r.values()) for r in negative)
    assert positive==[r['checks'] for r in before['outcomes']] and negative==[r['checks'] for r in after['outcomes']]
    divergence=None
    for i,(a,b) in enumerate(zip(before['decisions'],after['decisions'])):
        if any(a[k]!=b[k] for k in ['player','menu','observation_sha256']):break
        if a['action']!=b['action']:
            if b['player']==controlled:
                divergence=dict(step=i,prior_action=a['action'],current_action=b['action'],current_observation_sha256=b['observation_sha256'],current_rnn_sha256=b['rnn_sha256'])
            break
    return dict(schema='ygo-retention-candidate-v1',case=case['name'],family='normal_strict_goal',split=case['split'],status='observed_strict_goal_regression',training_ready=False,
        definition=copy.deepcopy(case['definition']),goals=copy.deepcopy(case['goals']),prior_goal_checks=positive,current_goal_checks=negative,first_observed_route_divergence=divergence,
        label_semantics='Loss of this declared strict benchmark goal, not proof of a strategic error. Prior-policy actions are not correction labels; no prior hidden state may be reused.',
        required_admission=['Freeze a new current-parent discovery protocol','Reconstruct the current-model history/RNN and independently search or verify a valid continuation','Compare actual functional alternatives before diagnosing strategic failure','Double replay and independent native audit before any new teaching mask'])
