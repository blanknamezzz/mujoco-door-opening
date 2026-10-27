"""Action-grid identity, CLI compatibility and variable-size model contracts."""
import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
from grasp_angles import LEGACY_ANGLES, DIVERSE_ANGLES, add_angle_arguments, resolve_angles
from train_grasp_rl import Policy, load_checkpoint, save_checkpoint
from compare_grasp_learning import RewardModel, load_model, save_model


def arguments(*argv):
    parser = argparse.ArgumentParser()
    add_angle_arguments(parser)
    return parser.parse_args(argv)


class AngleGridTests(unittest.TestCase):
    def test_expansion_preserves_all_old_actions_and_zero(self):
        values = resolve_angles(arguments(), require_zero=True)
        self.assertEqual(len(values), 29)
        self.assertTrue(set(LEGACY_ANGLES).issubset(values))
        self.assertEqual(set(values)-set(LEGACY_ANGLES), {-90.,-60.,-45.,-30.,30.,45.,60.,90.})
        self.assertEqual(resolve_angles(arguments('--angle-grid','legacy21')), list(LEGACY_ANGLES))

    def test_explicit_range_and_invalid_inputs(self):
        self.assertEqual(resolve_angles(arguments('--angle-min','-60','--angle-max','60','--angle-step','5')), list(range(-60,61,5)))
        self.assertEqual(resolve_angles(arguments('--angles-deg','30','0','-30')), [-30.,0.,30.])
        for argv in [('--angles-deg','0','0'), ('--angles-deg','0','nan'),
                     ('--angle-step','0'), ('--angle-step','3'),
                     ('--angle-grid','legacy21','--angles-deg','0','1')]:
            with self.subTest(argv=argv), self.assertRaises(ValueError):
                resolve_angles(arguments(*argv))
        with self.assertRaises(ValueError):
            resolve_angles(arguments('--angles-deg','-30','30'), require_zero=True)

    def test_both_training_entrypoints_use_same_default_and_legacy_flags(self):
        import collect_grasp_data as collection
        import train_grasp_rl as rl
        for module, argv, func in [(collection,['collect_grasp_data.py'],'collect'),
                                   (rl,['train_grasp_rl.py','train'],'train')]:
            with patch('sys.argv',argv), patch.object(module,func) as call:
                module.main()
                self.assertEqual(resolve_angles(call.call_args.args[0]), list(DIVERSE_ANGLES))
        with patch('sys.argv',['train_grasp_rl.py','train','--angle-min','-20','--angle-max','20','--angle-step','2']), patch.object(rl,'train') as call:
            rl.main()
            self.assertEqual(resolve_angles(call.call_args.args[0]),list(LEGACY_ANGLES))

    def test_old_and_new_checkpoints_keep_action_index_identity(self):
        for values in (LEGACY_ANGLES, DIVERSE_ANGLES):
            with self.subTest(count=len(values)), tempfile.TemporaryDirectory() as tmp:
                config=dict(schema_version=2, angles_deg=list(values))
                p=Path(tmp)/'rl.npz'
                policy=Policy(len(values),np.random.default_rng(42))
                policy.params['b2'][:]=np.arange(len(values))
                save_checkpoint(p,policy,config)
                loaded,c=load_checkpoint(p)
                self.assertEqual(c['angles_deg'],list(values))
                np.testing.assert_array_equal(policy.forward(np.zeros(4))[1],loaded.forward(np.zeros(4))[1])
                model=RewardModel(len(values),np.random.default_rng(42))
                metadata=dict(config,method='supervised')
                save_model(p,model,metadata)
                loaded,c=load_model(p)
                self.assertEqual(loaded.forward(np.zeros(4))[1].shape,(len(values),))
                self.assertEqual(c,metadata)

    def test_resume_cannot_reinterpret_existing_action_indices(self):
        from collect_grasp_data import collect
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)
            (p/'manifest.json').write_text(json.dumps(dict(angles_deg=list(LEGACY_ANGLES))))
            args=arguments('--angle-grid','diverse29')
            args.output=p;args.resume=True
            with self.assertRaisesRegex(ValueError,'Cannot change the angle grid'):
                collect(args)


if __name__ == '__main__':
    unittest.main()
