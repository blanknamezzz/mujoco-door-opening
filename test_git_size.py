"""A staged large blob must not escape the size guard via a small working file."""
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import check_git_size


class GitSizeGuard(unittest.TestCase):
    def test_ignored_weights_and_staged_blob_size(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prefix = ['git', '-c', f'safe.directory={root.as_posix()}']
            subprocess.run([*prefix, 'init', '--quiet', str(root)], check=True)
            (root/'.gitignore').write_text('*.pt\n')
            weight = root/'weights.pt'
            weight.write_bytes(b'x'*2048)
            with patch.object(check_git_size, 'ROOT', root):
                self.assertTrue(check_git_size.audit(.001)['passed'])
                subprocess.run([*prefix, 'add', '-f', 'weights.pt'], cwd=root, check=True)
                weight.write_bytes(b'small')
                report = check_git_size.audit(.001)
                self.assertFalse(report['passed'])
                self.assertEqual(report['oversized'][0]['staged_bytes'], 2048)
                weight.unlink()
                self.assertFalse(check_git_size.audit(.001)['passed'])


if __name__ == '__main__':
    unittest.main()
