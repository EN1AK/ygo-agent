import lzma
from pathlib import Path
import struct
import tempfile
import unittest

from ygoai.rl.demonstrations import parse_yrp2, make_manifest_record
from ygoai.rl.exercise_core import messages
from ygoai.rl.human_replay import canonical_response, comparable_game_events, card_spec, response_action
from ygoai.rl.elfnote_combo import GOALS, grade


class HumanReplayTests(unittest.TestCase):
    def test_battle_commands_preserve_attack_index_across_direct_flags(self):
        menu = [dict(act='Activate',phase='None'),
                dict(act='Attack',phase='None'),
                dict(act='DirectAttack',phase='None'),
                dict(act='None',phase='Main2'),dict(act='None',phase='End')]
        snapshot = dict(message=10,player=1,menu=menu)
        prompt = dict(op=10,player=1)
        for response, expected in ((0,0),(1,1),(65537,2),(2,3),(3,4)):
            self.assertEqual(response_action(snapshot,prompt,struct.pack('<i',response).hex()),expected)
        for response in (131073,-1,65538,4):
            with self.assertRaises(ValueError):
                response_action(snapshot,prompt,struct.pack('<i',response).hex())

    def replay(self, folder, *, flags=17, header_version=1, response=b'\x04\xff\xff\xff\xff'):
        payload = b'\0' * 80 + struct.pack('<4I', 8000, 5, 1, 327696)
        payload += struct.pack('<2I', 1, 13597785) + struct.pack('<I', 0)
        payload += struct.pack('<2I', 1, 3557275) + struct.pack('<I', 0) + response
        compressed = lzma.compress(payload, format=lzma.FORMAT_ALONE)
        header = struct.pack('<6I', 846230137, 4962, flags, 0, len(payload), 123)
        header += compressed[:5] + b'\0' * 3
        header += struct.pack('<8I', *range(8)) + struct.pack('<4I', header_version, 0, 0, 0)
        path = Path(folder) / 'demo.yrp'
        path.write_bytes(header + compressed[13:])
        return path

    def test_compressed_payload_size_and_corruption_are_checked(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.replay(folder)
            original = path.read_bytes()
            size = struct.unpack_from('<I', original, 16)[0]
            for advertised in (size - 1, size + 1):
                data = bytearray(original)
                struct.pack_into('<I', data, 16, advertised)
                path.write_bytes(data)
                with self.assertRaises((ValueError, lzma.LZMAError)):
                    parse_yrp2(path)
            data = bytearray(original)
            data[80] = 0xff  # Invalid LZMA range-coder initialization.
            path.write_bytes(data)
            with self.assertRaises((ValueError, lzma.LZMAError)):
                parse_yrp2(path)

    def test_extended_seed_is_preserved_and_header_version_is_not_seat(self):
        with tempfile.TemporaryDirectory() as folder:
            path = self.replay(folder)
            replay = parse_yrp2(path)
            self.assertEqual(replay.seed_sequence, tuple(range(8)))
            self.assertEqual(replay.header_version, 1)
            row = make_manifest_record(path, 'combo', 'elfnote')
            self.assertIsNone(row['start_player'])
            self.assertEqual(row['engine']['seed_sequence'], list(range(8)))
            self.assertFalse(row['supervision']['value'])

    def test_unsupported_variants_and_oversized_responses_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            for kwargs in (dict(flags=19), dict(header_version=2), dict(response=b'\0'),
                           dict(response=b'\x41' + b'\0' * 65)):
                with self.assertRaises(ValueError):
                    parse_yrp2(self.replay(folder, **kwargs))

    def test_weighted_selection_consumes_exact_payload_before_next_message(self):
        raw = bytes.fromhex('1700000800000001070101d5bc0000040204000000017097d00500040004000000')
        rows = list(messages(raw + bytes.fromhex('290400')))
        self.assertEqual([r['op'] for r in rows], [23, 41])
        self.assertEqual(rows[0]['target'], 8)
        self.assertEqual(rows[0]['must'][0]['code'], 12375297)
        self.assertEqual(rows[0]['cards'][0]['extra'], 4)
        with self.assertRaises(ValueError):
            list(messages(raw[:-1]))

    def test_mapping_uses_prompt_alias_not_physical_deck_order(self):
        card = dict(player=0, location=1, sequence=2, code=12375297)
        self.assertEqual(card_spec(card, 0), '3')
        prompt = dict(op=15, player=0, cards=[card])
        snapshot = dict(message=15, player=0, menu=[dict(spec='3', finish=False)])
        self.assertEqual(response_action(snapshot, prompt, '0100'), 0)
        with self.assertRaises(ValueError):
            response_action(dict(snapshot, player=1), prompt, '0100')

    def test_outcome_accepts_order_and_extra_materials_but_requires_resources(self):
        target = GOALS['red']
        cards = [dict(code=code, player=0, location=4, position=1) for code in reversed(target['monsters'])]
        cards += [dict(code=code, player=0, location=8, position=5) for code in target['spells']]
        cards += [dict(code=code, player=0, location=2, position=8) for code in [56651978]+[59438930]*4]
        state = dict(cards=cards)
        self.assertTrue(grade(state, target)['success'])
        cards.append(dict(code=12375297, player=0, location=4, position=1))
        self.assertTrue(grade(state, target)['success'])
        cards[2]['position'] = 8
        self.assertFalse(grade(state, target)['success'])

    def test_ordinary_selection_preserves_recorded_order(self):
        cards = [dict(player=0, location=4, sequence=i, code=100+i) for i in range(3)]
        prompt = dict(op=15, player=0, cards=cards)
        menu = [dict(spec='m'+str(i+1), finish=False) for i in range(3)]
        snapshot = dict(message=15, player=0, menu=menu)
        self.assertEqual(response_action(snapshot, prompt, '020200'), 2)
        snapshot['menu'] = menu[:2]
        self.assertEqual(response_action(snapshot, prompt, '020200'), 0)

    def test_overlay_material_identity_includes_material_index(self):
        card = dict(player=0, location=132, sequence=0, extra=1)
        self.assertEqual(card_spec(card, 0), 'm1b')
        self.assertEqual(card_spec(card, 1), 'om1b')
        with self.assertRaises(ValueError):
            card_spec(dict(player=0, location=132, sequence=0), 0)

    def test_unweighted_tribute_preserves_recorded_selection_order(self):
        cards = [dict(player=0, location=2, sequence=i, extra=1) for i in (0, 3)]
        prompt = dict(op=20, player=0, min=2, max=2, cards=cards)
        menu = [dict(spec='h1', finish=False), dict(spec='h4', finish=False)]
        self.assertEqual(response_action(dict(message=20, player=0, menu=menu),
                                         prompt, '020100'), 1)
        cancel = dict(act='Cancel', finish=False)
        self.assertEqual(response_action(dict(message=20, player=0, menu=[cancel]),
                                         prompt, 'ffffffff'), 0)

    def test_announce_card_follows_exact_branch_membership_then_leaf(self):
        prompt = dict(op=142, player=0)
        # Noncontiguous branches must not be inferred from a numeric range.
        menu = [dict(announce_codes=[10, 90]), dict(announce_codes=[20, 30])]
        snap = dict(message=142, player=0, menu=menu)
        self.assertEqual(response_action(snap, prompt, struct.pack('<I', 90).hex()), 0)
        with self.assertRaises(ValueError):
            response_action(snap, prompt, struct.pack('<I', 40).hex())
        snap['menu'] = [dict(response=10), dict(response=90)]
        self.assertEqual(response_action(snap, prompt, struct.pack('<I', 90).hex()), 1)

    def test_counted_selection_padding_does_not_change_card_indices(self):
        prompt = dict(op=15)
        self.assertEqual(canonical_response(prompt, '0100000000'), '0100')
        self.assertEqual(canonical_response(prompt, 'ffffffff'), 'ffffffff')
        self.assertEqual(canonical_response(dict(op=142), '01000000'), '01000000')
        for value in ('0200', '010001'):
            with self.assertRaises(ValueError):
                canonical_response(prompt, value)

    def test_display_normalization_keeps_game_boundaries_and_hint_values(self):
        def events(*rows):
            return [dict(op=op, raw=value) for op, value in rows]
        original = events((2, 'hint'), (160, 'a'), (160, 'b'), (50, 'move'))
        reordered = events((2, 'hint'), (2, 'hint'), (160, 'b'), (160, 'a'), (50, 'move'))
        self.assertEqual(comparable_game_events(original), comparable_game_events(reordered))
        for changed in (events((2, 'other'), (160, 'a'), (160, 'b'), (50, 'move')),
                        events((2, 'hint'), (160, 'a'), (50, 'move'), (160, 'b'))):
            self.assertNotEqual(comparable_game_events(original), comparable_game_events(changed))


if __name__ == '__main__':
    unittest.main()
