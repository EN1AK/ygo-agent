import unittest
from extra_deck_aliases import unique_extra_cards,board_and_extra_inventory,extra_menu_permutation,normalize_extra_events


def move(code,source,destination,reason):
    return dict(op=50,raw=(bytes([50])+code.to_bytes(4,'little')+bytes(source)+bytes(destination)+reason.to_bytes(4,'little')).hex())


def summon(code):
    return dict(op=62,raw=(bytes([62])+code.to_bytes(4,'little')+bytes([0,4,2,1])).hex())


class ExtraAliasTests(unittest.TestCase):
    def test_xyz_alias_is_bound_to_the_actual_unique_summon(self):
        unique={(0,1000),(0,2000)}
        def route(slot,parent=1000):
            return [move(3000,[0,4,1,1],[0,192,slot,0],0x200008),
                move(parent,[0,64,slot,8],[0,4,2,1],8),summon(parent)]
        self.assertEqual(normalize_extra_events(route(5),unique),normalize_extra_events(route(6),unique))
        self.assertNotEqual(normalize_extra_events(route(5),unique),normalize_extra_events(route(6,2000),unique))
        self.assertNotEqual(normalize_extra_events(route(5),{(0,2000)}),normalize_extra_events(route(6),{(0,2000)}))
        self.assertNotEqual(normalize_extra_events(route(5)[:1],unique),normalize_extra_events(route(6)[:1],unique))
        self.assertEqual(normalize_extra_events(route(5)[:1],unique,True),normalize_extra_events(route(6)[:1],unique,True))

    def test_intervening_other_gameplay_prevents_parent_guessing(self):
        route=[move(3000,[0,4,1,1],[0,192,5,0],0x200008),
            move(4000,[0,2,1,8],[0,16,2,1],1),summon(1000)]
        self.assertEqual(normalize_extra_events(route,{(0,1000)})[0],route[0])

    def test_extra_permutation_never_hides_a_main_deck_change(self):
        a=dict(cards=[dict(player=0,location=1,sequence=0,code=9,position=8),
            dict(player=0,location=64,sequence=0,code=1,position=8),dict(player=0,location=64,sequence=1,code=2,position=8)],lp=[8000,8000])
        b=dict(a,cards=[dict(a['cards'][0]),dict(a['cards'][2],sequence=0),dict(a['cards'][1],sequence=1)])
        self.assertEqual(board_and_extra_inventory(a),board_and_extra_inventory(b))
        b['cards'][0]['sequence']=1
        self.assertNotEqual(board_and_extra_inventory(a),board_and_extra_inventory(b))

    def test_only_unique_extra_card_aliases_can_be_rebound(self):
        unique=unique_extra_cards(dict(initial=[[1000,0,64,0,8],[2000,0,64,1,8]]))
        old=[dict(code=1000,spec='x1',act='SpSummon',response=0)]
        new=[dict(code=1000,spec='x2',act='SpSummon',response=1)]
        self.assertEqual(extra_menu_permutation(old,new,0,unique),[0])
        with self.assertRaises(ValueError):extra_menu_permutation(old,new,0,set())


if __name__=='__main__':unittest.main()
