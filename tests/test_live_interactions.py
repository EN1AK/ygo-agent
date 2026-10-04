from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from collect_live_interactions import raw_chains


def packet(message, payload, index=0, step=4):
    return {'message': message, 'packet_hex': bytes([1, message, *payload]).hex(),
            'index': index, 'after_decision_step': step}


def source(code, actor, link, step=4):
    return packet(70, [*code.to_bytes(4, 'little'), actor, *([0] * 10), link], step=step)


def test_sources_follow_link_not_latest_activation():
    rows = [source(63288573, 1, 1), source(97268402, 0, 2, step=5),
            packet(76, [1]), packet(74, [])]
    result = list(raw_chains(rows))[0]
    assert result['context_step'] == 4
    assert result['links'][1]['code'] == 63288573
    assert result['disabled_links'] == [1]
    assert 'effect_success' not in result


def test_new_chain_does_not_inherit_previous_context_or_source():
    rows = [source(1, 0, 1), packet(74, []), source(2, 1, 1, step=8), packet(74, [])]
    a, b = list(raw_chains(rows))
    assert a['context_step'] == 4 and b['context_step'] == 8
    assert b['links'] == {1: {'code': 2, 'actor': 1}}


@pytest.mark.parametrize('rows', [[packet(76, [1])], [source(1, 0, 1)],
                                [source(1, 0, 1), packet(76, [2]), packet(74, [])]])
def test_orphan_incomplete_and_unknown_links_rejected(rows):
    with pytest.raises(AssertionError):
        list(raw_chains(rows))


def test_opponent_before_first_observation_keeps_unknown_context():
    r = list(raw_chains([source(1, 1, 1, step=None), packet(74, [])]))[0]
    assert r['context_step'] is None
