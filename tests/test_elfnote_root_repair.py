import copy
import json
import unittest
from collections import Counter
from pathlib import Path
from scripts.study_elfnote_root_repair import correction_indices,cases_for
from scripts.study_elfnote_coverage import make_cases
from ygoai.rl.elfnote_search import make_cases as original_cases
from scripts.teach_elfnote_combo import reserved_cases


class RepairTests(unittest.TestCase):
    def test_deck_alias_is_not_a_correction_but_hand_field_is(self):
        menu=[dict(code=1,act='Select',spec='101'),dict(code=1,act='Select',spec='102'),dict(code=1,act='Select',spec='h1')]
        record=dict(definition=dict(controlled=0),decisions=[dict(player=0,menu=menu,action=1,logits=[3,2,1]),dict(player=0,menu=menu,action=2,logits=[3,2,1]),dict(player=1,menu=menu,action=2,logits=[3,2,1])])
        self.assertEqual(correction_indices(record),[1])

    def test_new_conditions_preserve_registration_and_do_not_mutate_sources(self):
        plan=json.loads((Path(__file__).resolve().parents[1]/'assets/exercises/teaching-v1/elfnote-root-repair.json').read_text())
        def definition(a,b):
            cards=[14558128,97556336,25661743,59581480,13597785,59438930,97268402,59438930,a,b]
            return dict(controlled=0,initial=[[x,0,1,0,8] for x in cards],seed_sequence=[1]*8)
        previous=[]
        for name,a,b in [('red-blue',13597785,59581480),('red-gate',13597785,25661743),('blue-gate',59581480,25661743),('gate-innocent',25661743,97556336)]:
            for i in range(6):previous.append(dict(name=f'{name}-train-{i}',split='train',intervention=None,definition=definition(a,b),goals=[]))
            for i in range(2):previous.append(dict(name=f'{name}-fresh-{i}',split='fresh',intervention=None,definition=definition(a,b),goals=[]))
        for name,a,b in [('red-innocent',13597785,97556336),('blue-innocent',59581480,97556336)]:
            previous.append(dict(name=name+'-exposed',split='diagnostic',intervention=None,definition=definition(a,b),goals=[]))
            for i in range(2):previous.append(dict(name=f'{name}-fresh-{i}',split='fresh',intervention=None,definition=definition(a,b),goals=[]))
        # Existing histories have distinct seeds; source duplicates must be rejected.
        for i,c in enumerate(previous):c['definition']['seed_sequence']=[i+1]*8
        original=copy.deepcopy(previous);cases=cases_for(plan,previous)
        self.assertEqual(previous,original);self.assertEqual(len(cases),48)
        self.assertEqual(sum(c['split']=='fresh_repair' for c in cases),6)
        self.assertEqual(sum(c['split']=='fresh_triple' for c in cases),4)
        for i,c in enumerate(cases[-4:]):
            source=next(x for x in previous if x['name']==plan['triples'][i]['source'])
            self.assertEqual(Counter(map(tuple,c['definition']['initial'])),Counter(map(tuple,source['definition']['initial'])))


if __name__=='__main__':unittest.main()
