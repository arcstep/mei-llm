import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from training.cpt.rolling_store import RollingStore


class RollingStoreTest(unittest.TestCase):
    def test_bounds_and_protects_milestones(self):
        with tempfile.TemporaryDirectory() as d:
            store=RollingStore(d,'owner')
            for step in range(1,9):
                path=Path(d)/f'state-{step:07d}.pt';payload=str(step).encode()
                path.write_bytes(payload)
                path.with_suffix('.json').write_text(json.dumps({'sha256':hashlib.sha256(payload).hexdigest(),'fingerprint':'owner'}))
                store.committed(path,step=step,tokens=step*100,milestone=step in (3,6))
            self.assertEqual([r['step'] for r in store.state['checkpoints']],[3,6,7,8])
            self.assertEqual(len(list(Path(d).glob('*.pt'))),4)
            self.assertEqual(len(RollingStore(d,'owner').state['checkpoints']),4)
            with self.assertRaises(ValueError):RollingStore(d,'other')

    def test_corruption_does_not_retire_previous_checkpoint(self):
        with tempfile.TemporaryDirectory() as d:
            store=RollingStore(d,'owner')
            path=Path(d)/'state-0000001.pt';path.write_bytes(b'bad')
            path.with_suffix('.json').write_text(json.dumps({'sha256':'wrong','fingerprint':'owner'}))
            with self.assertRaises(ValueError):store.committed(path,step=1,tokens=1)
            self.assertEqual(store.state['checkpoints'],[])
            self.assertTrue(path.exists())

    def test_refuses_foreign_directory(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d)/'old.pt').write_bytes(b'protected')
            with self.assertRaises(ValueError):RollingStore(d,'owner')


if __name__=='__main__':unittest.main()
