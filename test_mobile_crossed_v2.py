"""Crossed group isolation, label contracts and coordinated-control bounds."""
import copy
import random
import unittest
import numpy as np
import mujoco
import hjepa_runtime
import torch

from mobile_crossed_v2 import (LAYOUTS,MotionDoor,make_manifest,restore_scene,
                               geometry_identity,factorized_targets,factorized_loss)
from run_mobile_crossed_experiment import sample_training,decision
from door_stress_scenes import scenario_by_id
from grasp_plan import GraspPlan
from test_mobile_door import action_table


class CrossedContracts(unittest.TestCase):
    def test_group_layouts_share_appearance_and_never_cross_splits(self):
        manifest=make_manifest(dict(train=2,val=1,test=1),'crossed_contract_seed42')
        groups={}
        for s in manifest['scenes']:groups.setdefault(s['group_id'],[]).append(s)
        self.assertEqual(len(groups),4)
        self.assertEqual(len({s['geometry_sha256'] for s in manifest['scenes']}),16)
        for members in groups.values():
            self.assertEqual({s['layout'] for s in members},set(LAYOUTS))
            self.assertEqual(len({s['split'] for s in members}),1)
            for key in ('door_style','handle_style','handle_x','handle_z','handle_friction','hinge_resistance'):
                self.assertEqual(len({s[key] for s in members}),1)
            by_layout={s['layout']:s for s in members}
            self.assertEqual(by_layout['free']['obstacles'],())
            a=by_layout['front_blocked']['obstacles'];b=by_layout['rear_blocked']['obstacles']
            self.assertEqual(len(a),len(b))
            for left,right in zip(a,b):
                np.testing.assert_allclose(left['pos'],np.array(right['pos'])*[1,-1,1])
            self.assertEqual(len(by_layout['both_blocked']['obstacles']),len(a)+len(b))

    def test_generator_streams_and_geometry_identity_are_stable(self):
        one=make_manifest(dict(train=1),'stream_a')
        same=make_manifest(dict(train=1),'stream_a')
        other=make_manifest(dict(train=1),'stream_b')
        self.assertEqual(one,same)
        self.assertTrue({s['geometry_sha256'] for s in one['scenes']}.isdisjoint(s['geometry_sha256'] for s in other['scenes']))
        spec=one['scenes'][0];changed=dict(spec,scene_id='renamed',description='changed metadata')
        self.assertEqual(geometry_identity(spec),geometry_identity(changed))
        self.assertEqual(restore_scene(spec).handle_style,spec['handle_style'])

    def test_binary_targets_ignore_partial_progress_and_total_legacy_reward(self):
        records=action_table();a=factorized_targets(records)
        for e in records:e.update(reward=-1e9,safe_opening_fraction=1.,shortfall=0.,milestones=[True]*3)
        b=factorized_targets(records)
        for key in a:np.testing.assert_array_equal(a[key],b[key])

    def test_binary_gradients_separate_good_side_bad_base_and_missing_grasp(self):
        records=action_table()
        for e in records[6:]:e.update(nav_success=False,opening_tested=False,grasp_observed=False)
        target=factorized_targets(records)
        p=torch.zeros(5,requires_grad=True);g=torch.zeros(4,2,requires_grad=True)
        factorized_loss(p,g,target).backward()
        self.assertLess(p.grad[2],0.)
        self.assertGreater(p.grad[3],0.)
        self.assertLess(g.grad[2,0],0.)
        self.assertGreater(g.grad[2,1],0.)
        torch.testing.assert_close(g.grad[3],torch.zeros(2))

    def test_group_sampler_uses_one_complete_quad_and_iid_same_cohort(self):
        training=[dict(spec=s) for s in make_manifest(dict(train=3),'batch_contract')['scenes']]
        grouped=sample_training(training,'grouped',random.Random(42))
        self.assertEqual(len({s['spec']['group_id'] for s in grouped}),1)
        self.assertEqual({s['spec']['layout'] for s in grouped},set(LAYOUTS))
        iid=sample_training(training,'iid',random.Random(42))
        self.assertEqual(len(iid),4)
        self.assertTrue(all(s in training for s in iid))

    def test_orbit_plan_keeps_passive_hinge_and_rejects_excess_speed(self):
        exp=MotionDoor(scenario_by_id('oak-lever-wall'),planner='orbit')
        self.assertEqual(exp.m.neq,0)
        self.assertNotIn(exp.m.joint('door_hinge').id,exp.m.actuator_trnid[:,0])
        with self.assertRaisesRegex(ValueError,'speed limit'):
            exp.trial_continuous(GraspPlan((.8,-.11,.85)))
        with self.assertRaises(ValueError):MotionDoor(scenario_by_id('oak-lever-wall'),planner='unknown')

    def test_common_union_exposes_planner_limitation_instead_of_false_success(self):
        data=action_table(())
        from staged_mobile_reward import credit_targets
        scene=dict(spec=dict(scene_id='x',group_id='g',layout='free'),
                   planners={'fixed':dict(records=data,credit=credit_targets(data))},
                   union_bases=np.array([True,False,False,False]),union_sides=np.array([True,False]))
        selected=decision(scene,'fixed',0,0)
        self.assertTrue(selected['planner_limitation'])
        self.assertFalse(selected['success'])
        refusal=decision(scene,'fixed',4,None)
        self.assertTrue(refusal['feasible'])
        self.assertFalse(refusal['conditional_feasible'])


if __name__=='__main__':unittest.main()
