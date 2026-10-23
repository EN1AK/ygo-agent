"""Frozen conditional exercise expansion and evidence-gated supervision masks."""
import copy
import numpy as np
from scripts.study_elfnote_coverage import variant,digest
from ygoai.rl.elfnote_search import action_signature


def make_cases(plan,base_cases,previous_cases):
    cases=[]
    for base in base_cases:
        source=copy.deepcopy(base);source['split']='fit';cases.append(source)
        for i,settings in enumerate(plan['resource_variants']):
            case=variant(base,settings,base['name']+f'-resource-{i}','fresh_resource',100*len(cases)+i)
            case['intervention']=base['intervention'];cases.append(case)
        case=copy.deepcopy(base);case.update(name=base['name']+'-late',split='fresh_timing',intervention_skip=1)
        cases.append(case)
    # New pairings retain the exposed elfnote lineage; no new-mechanism claim.
    from scripts.study_elfnote_recovery import cases_for
    extra_plan={**plan,'pairs':plan['extra_pairs']}
    for case in cases_for(extra_plan,previous_cases):
        case['split']='fresh_pair';cases.append(case)
    for case in cases:case['definition_sha256']=digest(case['definition'])
    assert len({c['name'] for c in cases})==len(cases)
    return cases


def correction_steps(record):
    return [i for i,d in enumerate(record['decisions']) if d.get('searchable',False) and d['player']==record['definition']['controlled']
            and action_signature(d['menu'][d['action']])!=action_signature(d['menu'][int(np.argmax(d['logits']))])]


def correction_row(record,observations):
    record=copy.deepcopy(record)
    if record['status']!='success' or not record['functional_evidence']['success'] or not record.get('intervention_evidence',{}).get('success'):
        raise ValueError('Functional evidence required before supervision')
    root,probe=record['root_step'],record['probe_start_step']
    steps=correction_steps(record)
    if root is None or probe is None or not steps or not all(root<=i<probe for i in steps):
        raise ValueError('No verified post-response pre-probe correction')
    if any(d.get('searchable',False) and (d['player']!=record['definition']['controlled'] or not root<=i<probe) for i,d in enumerate(record['decisions'])):
        raise ValueError('Search weight outside the controlled public recovery')
    for i,d in enumerate(record['decisions']):d['teaching']=i in steps
    record['curriculum_admission']=dict(kind='conditional_semantic_correction',steps=steps,probe_supervision=False)
    return (record,observations),steps
