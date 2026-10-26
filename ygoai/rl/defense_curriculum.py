"""The frozen66-history response-teaching comparison, with explicit group means."""
from collections import Counter
import copy
import math


COUNTS = dict(successful_normal=49, successful_old_skill=4, synthetic_old_KL_only=1,
              functional_recovery_CE=4, functional_context_KL=4, public_response_CE=4)


def coefficients(rows, response_coef):
    if isinstance(response_coef,bool) or not math.isfinite(response_coef) or not 0<=response_coef<=1:
        raise ValueError('Invalid response coefficient')
    if Counter(r['kind'] for r in rows)!=Counter(COUNTS):
        raise ValueError('Expected49/4/1/4/4/4 complete-history groups')
    ce=dict(successful_normal=1/49,successful_old_skill=1/4,synthetic_old_KL_only=0.,
            functional_recovery_CE=1/4,functional_context_KL=0.,public_response_CE=response_coef/4)
    protected={'successful_normal','successful_old_skill','synthetic_old_KL_only','functional_context_KL'}
    return [ce[r['kind']] for r in rows],[1/58 if r['kind'] in protected else 0. for r in rows]


def descriptors(base, collection, plan):
    assert base['histories']==62 and base['parent_sha256']==plan['parent_sha256']
    assert collection['admitted'] and len(collection['results'])==4
    rows=copy.deepcopy(base['rows'])
    for case in plan['cases']:
        row=next(r for r in collection['results'] if r['case']==case['name'])
        assert row['teaching_steps']==[case['root_step']] and row['parent_sha256']==plan['parent_sha256']
        rows.append(dict(path=row['path'],record_sha256=row['record_sha256'],observations_sha256=row['observations_sha256'],
            controlled=0,teaching_steps=list(row['teaching_steps']),kind='public_response_CE'))
    cc,kc=coefficients(rows,0.)
    for row,c,k in zip(rows,cc,kc):row.update(ce_coef=c,kl_coef=k)
    return rows
