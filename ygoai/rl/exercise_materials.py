"""Fixed material-combination split within the existing Halq combo family."""
from copy import deepcopy
from ygoai.rl.exercise_contexts import context_opening, ContextReference
from ygoai.rl.exercise_teaching import CYBER
from ygoai.rl.exercise_starters import VEILER

FORWARD, THRASHER, JET, CONTROLLER, GILA = 56410040, 65367484, 9742784, 68505803, 45894482
LINEUPS = [
    dict(name='forward-veiler', starter=FORWARD, tuner=VEILER, split='fit'),
    dict(name='thrasher-veiler', starter=THRASHER, tuner=VEILER, split='fit'),
    dict(name='cyber-jet', starter=CYBER, tuner=JET, split='fit'),
    dict(name='forward-controller', starter=FORWARD, tuner=CONTROLLER, split='fit'),
    dict(name='forward-jet', starter=FORWARD, tuner=JET, split='reserved-combination'),
    dict(name='thrasher-controller', starter=THRASHER, tuner=CONTROLLER, split='reserved-combination'),
    dict(name='gila-veiler', starter=GILA, tuner=VEILER, split='reserved-material'),
    dict(name='gila-jet', starter=GILA, tuner=JET, split='reserved-material'),
]


def material_opening(database, code_list, lineup, seat):
    profile=dict(name=f'{lineup["name"]}-seat{seat}', controlled=seat,
                 place_last=seat==0, variant=seat, split=lineup['split'])
    d=context_opening(database,code_list,'combo',profile)
    mapping={CYBER:lineup['starter'], VEILER:lineup['tuner']}
    d['decks'][seat]=[mapping.get(c,c) for c in d['decks'][seat]]
    d['initial']=[[mapping.get(c,c) if p==seat and loc==1 else c,p,loc,seq,pos]
                  for c,p,loc,seq,pos in d['initial']]
    assert all(r[0] in set(map(int,code_list.read_text().split())) for r in d['initial'])
    d['context']['materials']=deepcopy(lineup)
    d['material_split']=lineup['split']
    return d


class MaterialReference(ContextReference):
    def choose(self,snap,events):
        own=self.definition['controlled']; lineup=self.definition['context']['materials']
        if snap['player']!=own: return super().choose(snap,events)
        if snap['message']==12 and any(a['code']==GILA for a in snap['menu']):
            return next(i for i,a in enumerate(snap['menu']) if a['act']=='Cancel')
        # Teacher-only role substitution; actor tensors and stored menus stay native.
        mapping={lineup['starter']:CYBER,lineup['tuner']:VEILER}
        mapped=dict(snap,menu=[dict(a,code=mapping.get(a['code'],a['code'])) for a in snap['menu']])
        return super().choose(mapped,events)
