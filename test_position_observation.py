import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import torch
from position_observation import encode_position_views,position_features,observe_rear_camera


class PositionObservationTests(unittest.TestCase):
    def test_front_alignment_accepts_only_sparse_one_level_raster_roundoff(self):
        from prepare_hjepa_views import front_alignment
        a=np.zeros((224,224,3),np.uint8);b=a.copy();b[0,0,0]=1
        self.assertTrue(front_alignment(a,b)['acceptable'])
        b[0,0,0]=2;self.assertFalse(front_alignment(a,b)['acceptable'])
        b=a.copy();b[:2,:,0]=1;self.assertFalse(front_alignment(a,b)['acceptable'])

    def test_single_view_features_preserve_exact_old_inputs(self):
        z=torch.randn(2,192);commands=torch.randn(2,5,4)
        self.assertTrue(torch.equal(position_features(z,commands),torch.cat([z[:,None].expand(-1,5,-1),commands],-1)))

    def test_candidate_side_uses_actual_sensor_and_refusal_uses_both(self):
        z=torch.tensor([[[1.,2.],[3.,4.]]],requires_grad=True)
        commands=torch.tensor([[[.6,-1.,1.,0.],[.6,1.,-1.,0.],[0.,0.,0.,0.]]])
        features=position_features(z,commands)
        torch.testing.assert_close(features[0,:,:2],torch.tensor([[1.,2.],[3.,4.],[2.,3.]]))
        features[:,:,:2].sum().backward()
        torch.testing.assert_close(z.grad,torch.full_like(z,1.5))

    def test_paired_policy_requires_actual_rear_pixels(self):
        class Model:
            config={'position_views':'paired'}
            def encode(self,pixels,states):return pixels.mean((-1,-2))
        pixels=torch.zeros(1,3,4,4)
        with self.assertRaises(ValueError):encode_position_views(Model(),pixels,torch.zeros(1,1))
        z=encode_position_views(Model(),pixels,torch.zeros(1,1),torch.ones_like(pixels))
        self.assertEqual(tuple(z.shape),(1,2,3));self.assertEqual(float(z[:,1].mean()),1.)

    def test_repeated_front_matches_single_view_at_inference(self):
        class Model:
            config={'position_views':'repeat-front'}
            def encode(self,pixels,states):return pixels.mean((-1,-2))
        pixels=torch.randn(2,3,4,4);conditions=torch.randn(2,5,4);conditions[:,4]=0
        z=encode_position_views(Model(),pixels,torch.zeros(2,1))
        self.assertTrue(torch.equal(position_features(z,conditions),position_features(pixels.mean((-1,-2)),conditions)))

    def test_rear_camera_restores_calibration_and_robot_state(self):
        from structural_scene import structural_scenes
        e=structural_scenes()['pillar-back'].experiment()
        cid=e.m.camera('perception').id
        before=[a.copy() for a in (e.d.qpos,e.d.qvel,e.d.ctrl,e.m.cam_pos[cid],e.m.cam_quat[cid])]
        time=e.d.time
        rear=observe_rear_camera(e)
        self.assertGreater(rear['world_from_camera'][1,3],0)
        self.assertEqual(e.d.time,time)
        for a,b in zip((e.d.qpos,e.d.qvel,e.d.ctrl,e.m.cam_pos[cid],e.m.cam_quat[cid]),before):np.testing.assert_array_equal(a,b)

    def test_rear_test_arrays_are_excluded_before_loading(self):
        from train_hjepa import read_data
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)
            for s in ('train','val'):
                np.savez(p/(s+'.npz'),initial_rgb=np.zeros((4,4,3),np.uint8))
                np.save(p/(s+'.npy'),np.ones((4,4,3),np.uint8))
            m=dict(scenes=[dict(scene_id=s,split=s) for s in ('train','val','test')],
                episodes=[dict(scene_id=s,split=s,file=s+'.npz') for s in ('train','val','test')],
                position_rear_files={s:s+'.npy' for s in ('train','val','test')})
            (p/'manifest.json').write_text(json.dumps(m))
            _,records=read_data(p,required_splits=('train','val'),include_splits=('train','val'))
            self.assertEqual(len(records),2);self.assertTrue(all(x['initial_rear_rgb'].mean()==1 for _,x in records))

    def test_endpoint_collection_matches_temporal_physics(self):
        from collect_hjepa import collect
        from hjepa_scene import DoorScene
        specs=[dict(scene_id='parity',split='train',handle_x=.8,handle_z=.85,friction=.8,hinge_resistance=1.2)]
        with tempfile.TemporaryDirectory() as folder,patch('collect_hjepa.scene_set',return_value=specs):
            p=Path(folder)
            a=collect(p/'temporal',scenes=3,bases=1,grasp_count=1,image_size=64,observation_interval=.5)
            b=collect(p/'endpoint',scenes=3,bases=1,grasp_count=1,image_size=64,observation_interval=.5,endpoint_only=True)
            for k in ('result','reward','success','collision','safe_opening_fraction','shortfall','milestones'):
                self.assertEqual(a['episodes'][0][k],b['episodes'][0][k])
            self.assertGreater(a['episodes'][0]['frames'],0);self.assertEqual(b['episodes'][0]['frames'],0)
            self.assertFalse(b['temporal_sequences_available'])


if __name__=='__main__':unittest.main()
