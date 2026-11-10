"""Fusion support, unknown-space and immutable-query contracts."""
import math
import unittest
import numpy as np
import mujoco
from mobile_partial_map_v5 import MapSettings,PartialDoorMap,cad_prior
from mobile_partial_fusion_v7 import union_voxels,fused_endpoints,FusedPartialDoorMap
from door_stress_scenes import scenario_suite


def depth_frame(depth=2.):
    return dict(depth_m=np.full((5,5),depth),K=np.array([[2.,0,2.5],[0,2.,2.5],[0,0,1.]]),
                world_from_camera=np.eye(4))


class PartialFusionContracts(unittest.TestCase):
    def test_repeats_and_frame_order_do_not_multiply_occupied_voxels(self):
        points=np.array([[.001,.002,.003],[.004,.005,.006],[.1,.1,.1]])
        expected=union_voxels(points,.025)
        np.testing.assert_array_equal(expected,union_voxels(np.r_[points,points],.025))
        np.testing.assert_array_equal(expected,union_voxels(points[::-1],.025))
        self.assertEqual(len(expected),2)

    def test_free_rays_clear_endpoints_but_surface_and_unknown_are_retained(self):
        points=np.array([[0.,0.,-1.],[0.,0.,-2.],[10.,0.,-1.]])
        cloud,stats=fused_endpoints([points],[depth_frame()],MapSettings(),True)
        self.assertEqual(stats['free_ray_endpoints_removed'],1)
        self.assertTrue(any(np.array_equal(p,[0,0,-2]) for p in cloud))
        self.assertTrue(any(np.array_equal(p,[10,0,-1]) for p in cloud))
        unknown,_=fused_endpoints([points],[depth_frame(float('nan'))],MapSettings(),True)
        self.assertEqual(len(unknown),3)

    def test_query_is_repeatable_and_restores_per_frame_storage(self):
        prior=cad_prior(scenario_suite()[0]);mapper=FusedPartialDoorMap(prior)
        mapper.d.qpos[mapper.door_q]=math.pi/4;mujoco.mj_forward(mapper.m,mapper.d)
        point=mapper.d.geom_xpos[mapper.m.geom('door_panel').id].copy()[None]
        mapper.points=[point.copy()];mapper.frames=[depth_frame()]
        first=mapper.query();original=mapper.points
        mapper.points.append(point.copy());mapper.frames.append(depth_frame())
        twice=mapper.query();again=mapper.query()
        self.assertIs(mapper.points,original)
        self.assertEqual(first['mask'],twice['mask'])
        self.assertEqual(twice['mask'],again['mask'])
        self.assertEqual(twice['fusion_statistics']['unique_occupied_voxels'],1)
        # In the historical concatenation, one voxel repeated in two frames
        # crosses the minimum-two-hit threshold. This is the intended change.
        historical=PartialDoorMap(prior);historical.points=[point.copy(),point.copy()]
        historical.frames=[depth_frame(),depth_frame()]
        self.assertTrue(historical.query()['queries'][0]['observed_obstruction'])
        self.assertFalse(twice['queries'][0]['observed_obstruction'])

    def test_prior_rejects_hidden_obstacles_and_profile_keeps_default_margin(self):
        scene=scenario_suite()[0]
        with self.assertRaises(ValueError):FusedPartialDoorMap(scene)
        mapper=FusedPartialDoorMap(cad_prior(scene))
        self.assertEqual(mapper.settings.point_radius_m,.015)
        self.assertEqual(FusedPartialDoorMap(cad_prior(scene),'tight_voxel_union').settings.point_radius_m,.003)


if __name__=='__main__':unittest.main()
