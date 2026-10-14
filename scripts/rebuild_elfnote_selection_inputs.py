"""Reconstruct immutable replay inputs after the completed-selection scope fix.

Every engine response, menu and outcome must remain unchanged. Only inactive
selection counts/references/overflow may change; live selections are retained.
No optimization or policy inference is performed.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

import numpy as np
from scripts.collect_teaching_exercise import load_native, execute
from scripts.convert_human_replay import convert
from scripts.eval_capability_exercises import run as synthetic_run
from scripts.teach_elfnote_combo import load_row, save, audit_actor
from scripts.teach_three_capabilities import BRANCHES
from ygoai.rl.exercise_actor import array_digest
from ygoai.rl.exercise_core import sha
from ygoai.rl.elfnote_combo import GOALS, grade


def migrated_inputs(record, obs):
    arrays = {k:v.copy() for k,v in obs.items()}
    # These four protocols use init_multi_select; completed callbacks set idx=-1.
    # Exact equality with the newly rebuilt native writer is required below.
    live_messages = {15, 20, 23, 25}
    for i, d in enumerate(record['decisions']):
        if d['message'] not in live_messages:
            arrays['selection_'][i,8] = 0
            arrays['action_group_refs_'][i,:,3] = 0
            arrays['action_group_mask_'][i,:,3] = 0
            arrays['structured_diagnostics_'][i,5] = 0
    migrated = copy.deepcopy(record)
    for i, d in enumerate(migrated['decisions']):
        d['observation_sha256'] = array_digest({k:v[i:i+1] for k,v in arrays.items()})
    if 'observation_archive_digest' in migrated:
        migrated['observation_archive_digest'] = array_digest(arrays)
    return migrated, arrays


def same_inputs(actual, expected):
    assert actual.keys() == expected.keys()
    for k in actual: assert np.array_equal(actual[k],expected[k]), k


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('root','source','intake'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    for k,v in vars(a).items():
        if isinstance(v,Path):setattr(a,k,v.resolve())
    release=a.root/'dist/runtime-releases/exercise-study-e9994b4-20261008'
    original=a.root/'training-runs/elfnote-intake-20261014'
    a.native=a.intake/'exercise_actor_native.cpython-310-x86_64-linux-gnu.so'
    assert sha(a.native) != sha(original/a.native.name)
    a.database=release/'repo/assets/locale/zh/cards.cdb';a.scripts=release/'repo/scripts/script'
    a.code_list=release/'code_list.txt';a.semantics=release/'semantics'
    plan=json.loads((a.source/'assets/exercises/teaching-v1/elfnote-single-card.json').read_text())
    old_dir=a.root/'training-runs/route-repair-study-20261013/correct_retain/endpoint/old'
    actor_dir=a.intake/'actor-v1';actor_dir.mkdir()
    old_out=a.intake/'retention';old_out.mkdir()
    shutil.copytree(original/'replay-v2',a.intake/'replay-v2')
    native=load_native(a,release/'opening-02/registration.ydk')
    live_retained=0;results=[]
    for entry in plan['demonstrations']:
        old_record, old_obs=load_row(original/'actor-v1'/(entry['name']+'.json'))
        _,expected=migrated_inputs(old_record,old_obs)
        a.input=a.intake/'replay-v2'/(entry['name']+'.json')
        record,obs=convert(a,native,json.loads(a.input.read_text()))
        repeated,obs2=convert(a,native,json.loads(a.input.read_text()))
        assert record==repeated; same_inputs(obs,obs2);same_inputs(obs,expected)
        assert record['status']=='converted' and grade(record['final'],GOALS[entry['goal']])['success']
        for key in ('definition','events','final','core_responses','core_seed'):
            assert record[key]==old_record[key],key
        assert [(d['player'],d['message'],d['menu'],d['action']) for d in record['decisions']]==[(d['player'],d['message'],d['menu'],d['action']) for d in old_record['decisions']]
        path=save(actor_dir,entry['name'],record,obs);audit_actor(a,path)
        entry['actor_record_sha256']=sha(path)
        results.append(dict(name=entry['name'],source_record_sha256=sha(original/'actor-v1'/(entry['name']+'.json')),record_sha256=sha(path),physical_trajectory_unchanged=True))
        live_retained+=sum(int(obs['action_group_mask_'][i,:,3].sum()) for i,d in enumerate(record['decisions']) if d['message'] in {15,20,23,25})
    for i,entry in enumerate(plan['retention_anchors']):
        path=old_dir/entry['name'];old_record,old_obs=load_row(path)
        migrated,expected=migrated_inputs(old_record,old_obs)
        if i==4:
            record,obs=synthetic_run(a,native,None,'combo','reference','policy',replay_record=migrated)
            assert record['status']=='success' and record['final_state']==old_record['final_state']
        else:
            last=len(migrated['decisions'])-1
            record,obs=execute(native,migrated['definition'],a.semantics,BRANCHES[migrated['definition']['kind']][0],None,'teacher',replay_record=migrated,replay_until=last,root_action=migrated['decisions'][last]['action'])
            assert record['result']['success']
            assert record['final_state']==old_record['final_state']
        same_inputs(obs,expected)
        assert record['events']==old_record['events'] and record['core_responses']==old_record['core_responses']
        record['input_semantics_migration']=dict(source_record_sha256=sha(path),native_sha256=sha(a.native),scope='only completed selection state removed; exact raw actions and physical outcome retained')
        target=save(old_out,Path(entry['name']).stem,record,obs)
        if i==4:
            from scripts.verify_exercise_actor_responses import verify
            proof=verify(target,a)
        else:
            from scripts.audit_teaching_exercise import audit
            proof=audit(target,a)
        assert proof['passed'],proof
        with target.with_name(target.stem+'.audit.json').open('x') as f:json.dump(proof,f,indent=2)
        entry['sha256']=sha(target);entry['observations_sha256']=sha(target.with_suffix('.npz'))
        live_retained+=sum(int(obs['action_group_mask_'][j,:,3].sum()) for j,d in enumerate(record['decisions']) if d['message'] in {15,20,23,25})
    assert live_retained>0,'live multi-selection references were not exercised'
    plan.update(id='elfnote-single-card-selected-scope-20261014',native_sha256=sha(a.native))
    with (a.intake/'single-card-plan.json').open('x') as f:json.dump(plan,f,ensure_ascii=False,indent=2)
    with (a.intake/'reconstruction.json').open('x') as f:json.dump(dict(passed=True,native_sha256=sha(a.native),demonstrations=results,retention=5,live_selected_references_preserved=live_retained,allowed_changes=['inactive selection_[8]','inactive action_group_refs_ group 3','inactive action_group_mask_ group 3','inactive structured_diagnostics_[5]']),f,indent=2)
    print(json.dumps(dict(passed=True,demonstrations=5,retention=5,live_selected_references_preserved=live_retained)),flush=True)


if __name__=='__main__':main()
