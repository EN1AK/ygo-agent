import unittest
from scripts.studies.crystron_curriculum.freeze_supervision import components
from scripts.studies.crystron_curriculum.catalog import turn_segments
from scripts.studies.crystron_curriculum.goals import achieved,own_attacks


class CrystronCurriculumTests(unittest.TestCase):
    def test_resource_goal_cannot_borrow_a_second_normal_summon_next_turn(self):
        state=dict(winner=None,lp=[8000,8000],cards=[dict(player=0,location=4,code=42,position=1)])
        goal=dict(kind='settled_resource_inventory',cards=[dict(location=4,code=42,count=1)])
        snapshot=dict(message=11,player=0)
        self.assertTrue(achieved(goal,state,snapshot,[dict(op=40)],0))
        self.assertFalse(achieved(goal,state,snapshot,[dict(op=40),dict(op=40)],0))

    def test_attack_declaration_and_canceled_attack_are_not_completed_attacks(self):
        attack=dict(op=110,raw='6e0004000101040001')
        self.assertEqual(own_attacks([attack],0),0)
        self.assertEqual(own_attacks([attack,dict(op=112),dict(op=114)],0),0)
        self.assertEqual(own_attacks([attack,dict(op=114)],0),1)
        self.assertEqual(own_attacks([attack,dict(op=114)],1),0)
        self.assertEqual(own_attacks([attack,dict(op=114),dict(op=40)],0),0)

    def test_player_deck_and_match_links_are_transitive_split_units(self):
        rows=[dict(source_sha256=s,player=p,deck_sha256=d,match_id=m) for s,p,d,m in [
            ('a','alice','deck1','game1'),('b','alice','deck2','game2'),
            ('c','bob','deck2','game3'),('d','carol','deck3','game3'),
            ('e','eve','deck4','game4')]]
        groups=components(rows)
        self.assertEqual(len({groups[s] for s in ('a','b','c','d')}),1)
        self.assertNotEqual(groups['a'],groups['e'])
        self.assertEqual(groups,components(list(reversed(rows))))

    def test_partial_turn_and_lp_assignment_are_not_terminal_or_damage(self):
        events=[dict(op=40,raw='2800'),dict(op=94,raw='5e01401f0000'),
                dict(op=91,raw='5b01d0070000'),dict(op=40,raw='2801'),
                dict(op=110,raw='6e0104000100040000')]
        turns=turn_segments(events,0)
        self.assertTrue(turns[0]['boundary_complete'])
        self.assertFalse(turns[1]['boundary_complete'])
        self.assertEqual(turns[0]['opponent_damage'],2000)
        self.assertEqual(turns[1]['attacks'],1)


if __name__=='__main__':unittest.main()
