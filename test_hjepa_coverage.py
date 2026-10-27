import json
from pathlib import Path
import random
import tempfile
import unittest
import numpy as np
import torch
from experiment_reproducibility import seed_everything
from prepare_hjepa_coverage import compose
from train_hjepa import read_data


class CoverageTests(unittest.TestCase):
    def test_seed_42_repeats_python_numpy_torch(self):
        a=seed_everything(42);x=(random.random(),np.random.rand(),torch.rand(3))
        b=seed_everything(42);y=(random.random(),np.random.rand(),torch.rand(3))
        self.assertEqual(x[:2],y[:2]);self.assertTrue(torch.equal(x[2],y[2]));self.assertEqual(a,b)
        self.assertEqual(a['cuda_seed'],42)
        if torch.cuda.is_available():
            seed_everything(42);a=torch.rand(3,device='cuda')
            seed_everything(42);b=torch.rand(3,device='cuda')
            self.assertTrue(torch.equal(a,b))

    def test_policy_reader_deduplicates_sensors_without_opening_temporal_files(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);rgb=np.zeros((8,8,3),np.uint8)
            for i in range(2):np.savez(p/f'{i}.npz',initial_rgb=rgb,arrival_rgb=rgb,initial_proprio=np.zeros(19),
                arrival_proprio=np.zeros(19),base=np.zeros(4),grasp=np.zeros(5),pixels=np.ones((100,8,8,3)))
            m=dict(scenes=[dict(scene_id='s',split='train')],episodes=[dict(scene_id='s',split='train',base_id=0,
                file=f'{i}.npz',pixels_file='MUST_NOT_OPEN.npy') for i in range(2)])
            (p/'manifest.json').write_text(json.dumps(m))
            _,r=read_data(p,required_splits=('train',),include_splits=('train',),policy_only=True)
            self.assertIs(r[0][1]['initial_rgb'],r[1][1]['initial_rgb'])
            self.assertIs(r[0][1]['arrival_rgb'],r[1][1]['arrival_rgb'])
            self.assertNotIn('pixels',r[0][1]);self.assertFalse(r[0][1]['initial_rgb'].flags.writeable)
            bad=np.ones_like(rgb);np.savez(p/'1.npz',initial_rgb=bad)
            with self.assertRaisesRegex(ValueError,'Inconsistent'):read_data(p,required_splits=('train',),policy_only=True)

    def test_composed_policy_corpus_excludes_tests_and_translates_sensor_paths(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder);sources=[]
            for n in ('old','new'):
                d=p/n;d.mkdir();sources.append(d)
                scenes=[];episodes=[];rear={}
                for split in ('train','val','test'):
                    sid=n+split;scenes.append(dict(scene_id=sid,split=split,family='f',example_geometry=sid))
                    np.savez(d/(sid+'.npz'),initial_rgb=np.zeros((8,8,3),np.uint8));np.save(d/(sid+'.npy'),np.zeros((8,8,3),np.uint8))
                    episodes.append(dict(scene_id=sid,split=split,file=sid+'.npz',pixels_file=None));rear[sid]=sid+'.npy'
                m=dict(seed=42,image_size=8,base_budget=5,grasp_budget=4,scenes=scenes,episodes=episodes,position_rear_files=rear)
                (d/'manifest.json').write_text(json.dumps(m))
            out=p/'mixed';m=compose([(sources[0],('train',)),(sources[1],('train','val'))],out)
            self.assertEqual(m['split_counts'],dict(train=2,val=1));self.assertFalse(m['temporal_sequences_available'])
            self.assertEqual(set(m['position_rear_files']),{'oldtrain','newtrain','newval'})
            self.assertTrue(all((out/e['file']).is_file() for e in m['episodes']))
            with self.assertRaisesRegex(ValueError,'Only development'):compose([(sources[0],('test',))],p/'bad')
            with self.assertRaisesRegex(ValueError,'Repeated scene'):compose([(sources[0],('train',)),(sources[0],('train',))],p/'dup')


if __name__=='__main__':unittest.main()
