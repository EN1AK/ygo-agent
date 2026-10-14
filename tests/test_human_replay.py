import lzma
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch

from ygoai.rl.demonstrations import parse_yrp2, make_manifest_record
from ygoai.rl.exercise_core import messages
from ygoai.rl.human_replay import card_spec, response_action
from ygoai.rl.elfnote_combo import GOALS, grade


class HumanReplayTests(unittest.TestCase):
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
        # These tests isolate header/response semantics. Python's LZMA_ALONE
        # writer emits an unknown-size end marker; YGOPro writes a known-size
        # stream. The supplied real recordings exercise decompression separately.
        decoder = patch('ygoai.rl.demonstrations.lzma.LZMADecompressor')
        decoder.start().return_value.decompress.return_value = payload
        self.addCleanup(decoder.stop)
        return path

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


if __name__ == '__main__':
    unittest.main()
