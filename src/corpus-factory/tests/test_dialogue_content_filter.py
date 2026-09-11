import hashlib
import importlib.util
import json
import tempfile
import unittest
from array import array
from pathlib import Path

module_path = Path(__file__).resolve().parents[1] / "quality/dialogue_content_filter.py"
spec = importlib.util.spec_from_file_location("dialogue_content_filter", module_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class ContentFilterTests(unittest.TestCase):
    def test_filter_preserves_selected_token_bytes_and_rebases_offsets(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            source.mkdir()
            for split in ("train", "valid"):
                (source / f"{split}.bin").write_bytes(array("H", [2, 10, 1, 2, 20, 1]).tobytes())
                rows = [{"text": "丢弃这条", "token_offset": 0, "tokens": 3},
                        {"text": "保留这条", "token_offset": 3, "tokens": 3}]
                (source / f"{split}.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")
            manifest = {"artifacts": {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in source.iterdir()}, "tokenizer_sha256": "test"}
            receipt = source / "receipt.json"
            receipt.write_text(json.dumps(manifest))
            config = root / "config.json"
            output = root / "output"
            config.write_text(json.dumps({"output": str(output), "rules": {"excluded": "丢弃"}}))
            result = module.filter_candidate(receipt, config)
            self.assertEqual(result["tokens"], {"train": 3, "valid": 3})
            self.assertEqual((output / "train.bin").read_bytes(), array("H", [2, 20, 1]).tobytes())
            self.assertEqual(json.loads((output / "train.jsonl").read_text())["token_offset"], 0)
            self.assertFalse(result["training_adoption_eligible"])
            with self.assertRaises(FileExistsError):
                module.filter_candidate(receipt, config)
