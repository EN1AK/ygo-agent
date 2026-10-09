from pathlib import Path
import json
import sqlite3
import subprocess
import sys
import pytest

from ygoai.rl.effect_semantics import FIELDS, WIDTH, parse_effects, validate_assets
from ygoai.rl.observation_schema import manifest, tensor_contract


def script(body):
    return 'local s,id=GetID()\nfunction s.initial_effect(c)\n' + body + '\nend\n'


def features(raw, slot):
    return dict(zip(FIELDS, raw[slot * WIDTH:(slot + 1) * WIDTH]))


def test_distinct_effects_and_clone_override():
    raw, report = parse_effects(script('''
local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(id,0))
e1:SetCategory(CATEGORY_SPECIAL_SUMMON)
e1:SetType(EFFECT_TYPE_QUICK_O)
e1:SetRange(LOCATION_MZONE)
e1:SetCost(s.cost)
c:RegisterEffect(e1)
local e2=e1:Clone()
e2:SetDescription(aux.Stringid(id,1))
e2:SetType(EFFECT_TYPE_TRIGGER_O)
e2:SetRange(LOCATION_GRAVE)
c:RegisterEffect(e2)'''), 123)
    assert report['slots'] == [0, 1]
    assert features(raw, 0)['quick'] == 1
    assert features(raw, 1)['quick'] == 0
    assert features(raw, 1)['trigger'] == features(raw, 1)['range_grave'] == 1
    assert features(raw, 0)['range_grave'] == 0


def test_ambiguous_description_is_unknown():
    raw, report = parse_effects(script('''
local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(id,0))
c:RegisterEffect(e1)
local e2=e1:Clone()
c:RegisterEffect(e2)'''), 123)
    assert not any(raw)
    assert report['ambiguous_slots'] == [0]


def test_comments_strings_dynamic_and_foreign_binding_are_not_evidence():
    for body in (
        '-- local e1=Effect.CreateEffect(c)\n-- e1:SetDescription(aux.Stringid(id,0))',
        'local text="e1:SetDescription(aux.Stringid(id,0))"',
        'if FLAG then\nlocal e1=Effect.CreateEffect(c)\nend',
        'local e1=Effect.CreateEffect(c)\ne1:SetDescription(aux.Stringid(999,0))\nc:RegisterEffect(e1)',
    ):
        assert not any(parse_effects(script(body), 123)[0])


def test_unregistered_effect_and_unknown_mutator_are_not_bound():
    base = 'local e1=Effect.CreateEffect(c)\ne1:SetDescription(aux.Stringid(id,0))'
    assert not any(parse_effects(script(base), 123)[0])
    assert not any(parse_effects(script(base+'\nhelper(e1)\nc:RegisterEffect(e1)'), 123)[0])


def test_v2_preserves_every_v1_tensor_and_has_explicit_new_contract():
    old = tensor_contract('structured-lite-v1')
    new = tensor_contract('structured-state-v2')
    assert all(new[k] == v for k, v in old.items())
    assert len(new) == len(old) + 3
    assert manifest('structured-state-v2')['public_state_version'] == 'public-chain-turn-v1'


def test_real_reviewed_raye_declarations():
    # Literal excerpt: no card database, Lua execution or hidden-state labels.
    raw, _ = parse_effects('''function c26077387.initial_effect(c)
local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(26077387,0))
e1:SetType(EFFECT_TYPE_QUICK_O)
e1:SetRange(LOCATION_MZONE)
e1:SetCost(c26077387.spcost1)
c:RegisterEffect(e1)
local e2=Effect.CreateEffect(c)
e2:SetDescription(aux.Stringid(26077387,1))
e2:SetType(EFFECT_TYPE_FIELD+EFFECT_TYPE_TRIGGER_O)
e2:SetRange(LOCATION_GRAVE)
c:RegisterEffect(e2)
end''', 26077387)
    assert features(raw, 0)['has_cost'] == 1
    assert features(raw, 1)['has_cost'] == 0
    assert features(raw, 0)['range_monster'] == features(raw, 1)['range_grave'] == 1


def test_builder_is_deterministic_and_rejects_tampered_assets(tmp_path):
    db = tmp_path / 'cards.cdb'
    with sqlite3.connect(db) as c:
        c.execute('CREATE TABLE datas(id INTEGER, type INTEGER, atk INTEGER, def INTEGER, level INTEGER, race INTEGER, attribute INTEGER)')
        c.execute('INSERT INTO datas VALUES(123,33,1000,1000,4,1,1)')
    codes = tmp_path / 'codes.txt'
    codes.write_text('123\n')
    scripts = tmp_path / 'lua'
    scripts.mkdir()
    (scripts / 'c123.lua').write_text(script('''local e1=Effect.CreateEffect(c)
e1:SetDescription(aux.Stringid(id,0))
e1:SetCategory(CATEGORY_DRAW)
c:RegisterEffect(e1)'''))
    builder = Path(__file__).resolve().parents[1] / 'scripts/build_structured_semantics.py'
    for name in ('one', 'two'):
        subprocess.run([sys.executable, str(builder), '--cards-db', str(db), '--code-list', str(codes),
                        '--scripts', str(scripts), '--output-dir', str(tmp_path / name), '--effect-descriptions'],
                       check=True, stdout=subprocess.DEVNULL)
        validate_assets(tmp_path / name)
    assert (tmp_path/'one/metadata.json').read_bytes() == (tmp_path/'two/metadata.json').read_bytes()
    table = tmp_path / 'one/effect-descriptions.u8'
    table.write_bytes(bytes(len(table.read_bytes())))
    with pytest.raises(ValueError, match='digest mismatch'):
        validate_assets(tmp_path / 'one')
