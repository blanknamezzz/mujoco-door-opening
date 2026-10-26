import unittest
import torch
from probe_hjepa_spatial import spatial_features,standardize,probe_metrics,fit_logistic


class SpatialProbeTests(unittest.TestCase):
    def test_replay_accepts_complete96_and_rejects_truncated_or_training_rows(self):
        from replay_hjepa_reference import validate_replay_cohort
        report=dict(test_scene_count=96,decisions={'control':[dict(scene_id=str(i)) for i in range(96)]})
        manifest=dict(scenes=[dict(scene_id=str(i),split='test') for i in range(96)])
        self.assertEqual(len(validate_replay_cohort(report,manifest,'control')),96)
        report['decisions']['control'].pop()
        with self.assertRaises(ValueError):validate_replay_cohort(report,manifest,'control')
        report['test_scene_count']=95;manifest['scenes'].pop();manifest['scenes'][0]['split']='train'
        with self.assertRaises(ValueError):validate_replay_cohort(report,manifest,'control')

    def test_named_seed42_streams_are_repeatable_and_geometry_disjoint(self):
        from structural_scene import independent_test_scenes
        from prepare_hjepa_splits import assert_scene_disjoint
        first=independent_test_scenes(42,2,stream='pooling_v1')
        self.assertEqual(first,independent_test_scenes(42,2,stream='pooling_v1'))
        assert_scene_disjoint(first,independent_test_scenes(42,2,stream='pooling_v2'))
        assert_scene_disjoint(first,independent_test_scenes(42,2))
        with self.assertRaises(ValueError):independent_test_scenes(42,2,stream='../unsafe')

    def test_quadrants_preserve_spatial_information_lost_by_mean(self):
        patches=torch.zeros(1,16,16,192)
        patches[:,:8,:8]=1;patches[:,:8,8:]=2
        patches[:,8:,:8]=3;patches[:,8:,8:]=4
        hidden=torch.cat([torch.full((1,1,192),7.),patches.flatten(1,2)],1)
        projection=torch.cat([torch.eye(192),torch.zeros(576,192)])
        before=spatial_features(hidden,torch.nn.Identity(),projection)
        swapped=hidden.clone();swapped[:,1:]=patches.flip(1).flatten(1,2)
        after=spatial_features(swapped,torch.nn.Identity(),projection)
        self.assertTrue(torch.equal(before['patch_mean'],after['patch_mean']))
        self.assertFalse(torch.equal(before['grid2x2_full'],after['grid2x2_full']))
        torch.testing.assert_close(before['grid2x2_full'].reshape(1,4,192)[0,:,0],torch.tensor([1.,2.,3.,4.]))
        torch.testing.assert_close(before['grid2x2_random192'],torch.ones(1,192))
        torch.testing.assert_close(before['backbone_cls'],torch.full((1,192),7.))

    def test_validation_extremes_do_not_change_training_normalization(self):
        train=torch.tensor([[1.,4.],[3.,4.]],dtype=torch.float64)
        a,_,mean,std=standardize(train,torch.tensor([[1e9,-1e9]],dtype=torch.float64))
        b,_,other_mean,other_std=standardize(train,torch.zeros(1,2,dtype=torch.float64))
        torch.testing.assert_close(a,b);torch.testing.assert_close(mean,other_mean);torch.testing.assert_close(std,other_std)
        self.assertTrue(torch.isfinite(a).all())

    def test_side_probe_distinguishes_wrong_side_and_false_refusal(self):
        labels=torch.tensor([[0.,1.],[1.,0.],[0.,0.],[1.,1.]])
        logits=torch.tensor([[1.,-1.],[-1.,-2.],[-1.,-1.],[1.,2.]])
        m=probe_metrics(logits,labels)
        self.assertEqual(m['correct_side_decisions'],2)
        self.assertEqual(m['wrong_side'],1);self.assertEqual(m['false_refusals'],1)
        self.assertEqual(m['correct_refusals'],1)

    def test_frozen_linear_probe_learns_observed_separation(self):
        x=torch.tensor([[-2.],[-1.],[1.],[2.]],dtype=torch.float64)
        y=torch.tensor([0.,0.,1.,1.],dtype=torch.float64)
        w,b=fit_logistic(x,y,.01)
        self.assertTrue(torch.equal(x.flatten()>0,x@w+b>0))


if __name__=='__main__':unittest.main()
