import unittest
from ygoai.rl.defense_curriculum import coefficients,COUNTS


class DefenseCurriculumTests(unittest.TestCase):
    def test_preservation_and_response_targets_do_not_conflict(self):
        rows=[dict(kind=kind) for kind,n in COUNTS.items() for _ in range(n)]
        cc,kc=coefficients(rows,1.)
        self.assertAlmostEqual(sum(cc),4.)
        self.assertAlmostEqual(sum(kc),1.)
        for row,c,k in zip(rows,cc,kc):
            if row['kind']=='synthetic_old_KL_only':self.assertEqual(c,0.)
            if row['kind']=='public_response_CE':self.assertEqual((c,k),(.25,0.))
        control,_=coefficients(rows,0.)
        self.assertEqual([i for i,(a,b) in enumerate(zip(cc,control)) if a!=b],list(range(62,66)))

    def test_missing_groups_or_nonfinite_weight_cannot_silently_pass(self):
        rows=[dict(kind=kind) for kind,n in COUNTS.items() for _ in range(n)]
        for value in [float('nan'),True,-1.,2.]:
            with self.assertRaises(ValueError):coefficients(rows,value)
        with self.assertRaises(ValueError):coefficients(rows[:-1],1.)


if __name__=='__main__':unittest.main()
