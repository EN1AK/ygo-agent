import struct
import unittest

from ygoai.rl.jev_duel import read_deck, request
from ygoai.rl.jev_duel_observation import decode_fields, visible_cards, visible_event


class DuelInterfaceTests(unittest.TestCase):
    def test_engine_dynamic_attack_and_disabled_state_survive_projection(self):
        # Serialized native QUERY_CODE|POSITION|ATTACK|DEFENSE|STATUS.
        payload=struct.pack('<II4BiiI',0x80303,123,0,4,2,1,2700,1600,1)
        cards=decode_fields([struct.pack('<I',len(payload)+4)+payload])
        card=visible_cards(cards,0)['cards'][0]
        self.assertEqual(card['attack'],2700)
        self.assertTrue(card['disabled'])

    def test_hidden_card_fields_never_enter_policy_state(self):
        rows=[dict(code=100+p*10+loc,player=p,location=loc,sequence=0,position=pos,attack=999,status=1)
              for p,loc,pos in ((0,1,8),(1,1,8),(1,2,1),(1,8,8),(0,2,1))]
        view=visible_cards(rows,0)
        self.assertEqual(len(view['cards']),2)
        self.assertNotIn('code',view['cards'][0])
        self.assertNotIn('attack',view['cards'][0])
        self.assertEqual(view['cards'][1]['code'],102)

    def test_opponent_draw_and_hidden_movement_are_redacted(self):
        draw=bytes([90,1,1])+struct.pack('<I',12345678)
        self.assertNotIn('codes',visible_event(draw,0))
        self.assertEqual(visible_event(draw,1)['codes'],[12345678])
        move=bytes([50])+struct.pack('<I',12345678)+bytes([1,1,0,8,1,2,0,1])+struct.pack('<I',0)
        self.assertNotIn('code',visible_event(move,0))

    def test_real_chain_resolution_events_are_present(self):
        raw=bytes([70])+struct.pack('<I',63288573)+bytes([1,4,5,1,1,4,5])+struct.pack('<I',63288573*16)+bytes([1])
        event=visible_event(raw,0)
        self.assertEqual(event['code'],63288573)
        self.assertEqual(event['chain'],1)
        self.assertEqual(visible_event(bytes([76,1]),0)['event'],'effect negated')
        self.assertEqual(visible_event(bytes([74]),0)['event'],'chain ended')

    def test_model_operations_preserve_native_indices_and_snapshot_history(self):
        snap=dict(done=False,invalid=False,state={'cards':[]},menu=[dict(index=0,description='a'),dict(index=1,description='b')])
        history=[]
        r=request(snap,history,1)
        self.assertEqual(r['operations'],[('commit',0),('commit',1),('probe',0),('probe',1)])
        history.append(dict(native_action=0))
        self.assertEqual(r['state']['simulated_branches'],[])
        self.assertNotIn(('probe',0),request(snap,history,1)['operations'])
        self.assertEqual(len(request(snap,history,0)['operations']),2)

    def test_repository_oldschool_is_a_standard_size_deck(self):
        main,extra=read_deck('assets/deck/unused/OldSchool.ydk')
        self.assertEqual(len(main),40)
        self.assertEqual(extra,[])


if __name__=='__main__': unittest.main()
