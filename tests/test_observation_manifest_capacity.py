import copy,unittest
from ygoai.rl.observation_schema import manifest,validate_manifest,LEGACY_SCHEMA,STRUCTURED_LITE_SCHEMA


class CapacityManifestTests(unittest.TestCase):
    def test_nondefault_capacity_roundtrips_for_both_schemas(self):
        for schema in [LEGACY_SCHEMA,STRUCTURED_LITE_SCHEMA]:
            result=manifest(schema,max_options=128,history_actions=48)
            self.assertEqual(tuple(result['tensors']['actions_']['shape']),(128,12))
            self.assertEqual(result['capacities'],dict(history_actions=48,max_options=128))
            validate_manifest(result)
            default=manifest(schema);self.assertNotIn('capacities',default);validate_manifest(default)

    def test_shape_and_capacity_disagreement_is_rejected(self):
        result=manifest(STRUCTURED_LITE_SCHEMA,max_options=128)
        wrong=copy.deepcopy(result);wrong['tensors']['actions_']['shape']=(127,12)
        with self.assertRaises(ValueError):validate_manifest(wrong)
        del result['capacities']
        with self.assertRaises(ValueError):validate_manifest(result)

    def test_invalid_capacity_metadata_is_rejected(self):
        for capacity in [0,-1,True,128.0]:
            with self.assertRaises(ValueError):manifest(STRUCTURED_LITE_SCHEMA,max_options=capacity)
        result=manifest(STRUCTURED_LITE_SCHEMA);result['capacities']={'surprise':128}
        with self.assertRaises(ValueError):validate_manifest(result)


if __name__=='__main__':unittest.main()
