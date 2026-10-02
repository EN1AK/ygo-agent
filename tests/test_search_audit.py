import copy
import json
from pathlib import Path
import unittest

from ygoai.rl.search_audit import upgrade_legacy_search, validate_envelope


class SearchAuditTest(unittest.TestCase):
    def test_real_legacy_puct_roundtrip_is_lossless(self):
        source=json.loads((Path(__file__).parent/'fixtures/search/legacy-cycle-puct.json').read_text())
        original=copy.deepcopy(source)
        envelope=json.loads(json.dumps(upgrade_legacy_search(source)))
        validate_envelope(envelope)
        self.assertEqual(source,original)
        self.assertEqual(envelope['source_payload'],original)
        self.assertEqual(envelope['source_payload']['leaf_value_puct']['actions'],original['leaf_value_puct']['actions'])
        envelope['source_payload']['restored']['state_value']+=1
        with self.assertRaises(ValueError): validate_envelope(envelope)

    def test_oracle_cannot_be_promoted_by_envelope(self):
        report={'schema':'ygo-counterfactual-search-v1','leaf_value_puct':{}}
        envelope=upgrade_legacy_search(report)
        envelope['fair_play_eligible']=True
        with self.assertRaises(ValueError): validate_envelope(envelope)


if __name__=='__main__': unittest.main()
