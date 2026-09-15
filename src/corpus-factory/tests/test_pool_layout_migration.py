import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from common import paths
from pool_layout_migration import apply_moves


class PoolLayoutMigrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.old = 'corpus/pools/example-20260914-v1'
        self.new = 'corpus/pools/candidates/mixed/2026-09-14-example-r01'
        source = self.root / self.old
        source.mkdir(parents=True)
        (source / 'data.bin').write_bytes(b'\x00\xff\x01\x02')
        (source / 'RELEASE.json').write_text('{"original_id":"example-v1"}')
        self.out = self.root / 'receipt'
        self.out.mkdir()
        manifest = self.root / '.internal/registry/migrations/2026-09-corpus-pools.json'
        manifest.parent.mkdir(parents=True)
        manifest.write_text(json.dumps({'exact': {self.old: self.new}}))

    def test_move_preserves_bytes_inode_and_old_paths_resolve(self):
        original = self.root / self.old / 'data.bin'
        inode = original.stat().st_ino
        result = apply_moves(self.root, {self.old:self.new}, self.out)
        self.assertEqual(result['status'], 'moved_and_verified')
        self.assertFalse((self.root/self.old).exists())
        with patch.object(paths, 'ROOT', self.root):
            for ref in [self.old+'/data.bin', str(original)]:
                resolved = paths.resolve_repo_path(ref)
                self.assertEqual(resolved, self.root/self.new/'data.bin')
                self.assertEqual(resolved.stat().st_ino, inode)
                self.assertEqual(resolved.read_bytes(), b'\x00\xff\x01\x02')
            self.assertEqual(paths.resolve_repo_path(self.old+'-unrelated/data.bin'),
                             self.root/(self.old+'-unrelated/data.bin'))

    def test_older_alias_chains_into_new_folder(self):
        apply_moves(self.root,{self.old:self.new},self.out)
        manifest=self.root/'.internal/registry/migrations/2026-09-four-domain.json'
        manifest.write_text(json.dumps({'prefix':{'artifacts/pools/':'corpus/pools/'}}))
        with patch.object(paths,'ROOT',self.root):
            self.assertEqual(paths.resolve_repo_path('artifacts/pools/example-20260914-v1/data.bin'),
                             self.root/self.new/'data.bin')

    def test_destination_collision_never_overwrites(self):
        target=self.root/self.new
        target.mkdir(parents=True)
        (target/'keep').write_text('keep')
        with self.assertRaises(ValueError):
            apply_moves(self.root,{self.old:self.new},self.out)
        self.assertTrue((self.root/self.old/'data.bin').exists())
        self.assertEqual((target/'keep').read_text(),'keep')

    def test_escape_is_rejected(self):
        with self.assertRaises(ValueError):
            apply_moves(self.root,{self.old:'corpus/pools/../../outside'},self.out)
        self.assertTrue((self.root/self.old).exists())
