"""Information boundaries and depth geometry for partial opening maps."""
from dataclasses import replace
import unittest
import numpy as np
import mujoco
from mobile_partial_map_v5 import (cad_prior,geom_sdf,backproject,observed_free,PartialDoorMap)


def frame(depth=2.,translation=(0,0,0)):
    T=np.eye(4);T[:3,3]=translation
    return dict(depth_m=np.full((5,5),depth),K=np.array([[5.,0,2.5],[0,5.,2.5],[0,0,1]]),world_from_camera=T)


class PartialMapContracts(unittest.TestCase):
    def test_camera_pixel_centers_and_negative_camera_z(self):
        p=backproject(frame(),1)
        np.testing.assert_allclose(p[12],[0,0,-2.])
        np.testing.assert_allclose(p[0],[-.8,.8,-2.])
        np.testing.assert_allclose(backproject(frame(translation=(1,2,3)),1)[12],[1,2,1])

    def test_occluded_and_out_of_view_samples_are_unknown(self):
        points=np.array([[0,0,-1.],[0,0,-3.],[5,0,-1.],[0,0,1.]])
        np.testing.assert_array_equal(observed_free(points,[frame()]),[True,False,False,False])
        np.testing.assert_array_equal(observed_free(points,[]),[False]*4)

    def test_blended_depth_at_a_silhouette_is_not_an_obstacle_endpoint(self):
        f=frame();f['depth_m'][:,:2]=1.;f['depth_m'][:,3:]=3.
        points=backproject(f,1)
        self.assertGreater(len(points),0)
        self.assertFalse(np.isclose(points[:,2],-2.).any())

    def test_additional_view_can_reveal_previously_occluded_space(self):
        points=np.array([[0,0,-1.],[0,0,-3.]])
        a=observed_free(points,[frame()]);b=observed_free(points,[frame(),frame(4.)])
        self.assertTrue(np.all(b>=a));np.testing.assert_array_equal(b,[True,True])

    def test_signed_distances_respect_shape_and_rotation(self):
        R=np.array([[0,-1,0],[1,0,0],[0,0,1.]])
        box=geom_sdf([[0,0,0],[0,2,0]],mujoco.mjtGeom.mjGEOM_BOX,[1,.2,.3],np.zeros(3),R)
        np.testing.assert_allclose(box,[-.2,1.])
        cap=geom_sdf([[0,0,.5],[0,0,.7]],mujoco.mjtGeom.mjGEOM_CAPSULE,[.1,.5,0],np.zeros(3),np.eye(3))
        np.testing.assert_allclose(cap,[-.1,.1])

    def test_external_obstacles_are_rejected_at_mapper_boundary(self):
        from door_stress_scenes import scenario_by_id
        scene=scenario_by_id('oak-lever-wall')
        with self.assertRaises(ValueError):PartialDoorMap(scene)
        clean=cad_prior(scene)
        self.assertFalse(clean.obstacles);self.assertEqual(clean.wall_layout,'unknown')
        hidden=replace(scene,obstacles=tuple(reversed(scene.obstacles)),wall_layout='both_blocked',expected_open_side='front')
        self.assertEqual(cad_prior(scene),cad_prior(hidden))

    def test_same_observation_has_same_query_despite_hidden_layout_metadata(self):
        from door_stress_scenes import scenario_by_id
        scene=scenario_by_id('oak-lever-wall');state=np.zeros(19,np.float32)
        state[1:5]=[0,-np.pi/2,-2*np.pi/3,0];state[5:7]=.04;state[14:18]=[.6,-1.65,1,0]
        f=frame();a=PartialDoorMap(cad_prior(scene));b=PartialDoorMap(cad_prior(replace(scene,obstacles=(),wall_layout='free')))
        a.add_frame(f,state);b.add_frame(f,state)
        qa,qb=a.query(),b.query()
        self.assertEqual(qa,qb)
        self.assertTrue(any(q['status']=='unknown' for q in qa['queries']))

    def test_free_scene_rgbd_does_not_invent_a_sweep_obstacle(self):
        from door_stress_scenes import scenario_by_id
        from mobile_crossed_v2 import MotionDoor
        from mobile_door import reached_view
        scene=replace(scenario_by_id('oak-lever-wall'),obstacles=(),door_style='brushed-steel')
        exp=MotionDoor(scene,start_pose=(.6,-1.65,np.pi/2))
        observation,_,state=reached_view(exp)
        mapper=PartialDoorMap(cad_prior(scene));mapper.add_frame(observation,state)
        np.testing.assert_array_equal(mapper.query()['mask'],mapper.base_allowed)
        self.assertEqual(exp.reset_count,1)


if __name__=='__main__':unittest.main()
