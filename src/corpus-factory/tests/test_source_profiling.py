from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import sys
import unittest
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq

MODULE = Path(__file__).resolve().parents[1] / "sources" / "profiling.py"
spec = importlib.util.spec_from_file_location("source_profiling", MODULE)
p = importlib.util.module_from_spec(spec)
spec.loader.exec_module(p)


class SourceProfilingTests(unittest.TestCase):
    def test_http_catalog_is_frozen_but_does_not_claim_population_completeness(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);catalog=root/'catalog.json';catalog.write_text('{"files":[]}')
            config={'sources':[{'source_id':'archive','catalog_path':'catalog.json',
                'catalog_sha256':p.digest(catalog),'http_files':[{'path':'data.zip','url':'https://example.test/data.zip'}]}]}
            with patch.object(p,'ROOT',root),patch.object(p,'fetch') as fetch:
                good=p.profile(config,root/'good',network=True,metadata_only=True)
                self.assertEqual((root/'good/archive/supporting-catalog.raw').read_bytes(),catalog.read_bytes())
                self.assertFalse(json.loads((root/'good/archive/frame.json').read_text())['complete_for_patterns'])
                catalog.write_text('changed')
                bad=p.profile(config,root/'bad',network=True,metadata_only=True)
                self.assertIn('catalog hash changed',str(bad['sources'][0]['errors']))
                fetch.assert_not_called()

    def test_document_survey_preserves_format_and_does_not_invent_relations(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'guide.md'; text='字段001不能改成1\n```sql\nselect 0 where false;\n```\n'
            path.write_text(text)
            sample=p.sample_text(path,{'path':'guide.md','shard_probability':0.25},{'github_repository':'owner/project'})
            self.assertEqual(sample['rows'],1)
            self.assertEqual(sample['samples'][0]['text'],text)
            self.assertEqual(sample['samples'][0]['weight'],4)
            self.assertNotIn('relationships',sample['samples'][0]['original'])

    def test_audit_detects_split_leak_without_adopting_unbound_files(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            for split in ("train", "test"):
                pq.write_table(pa.table({"text": ["保持字段001，不补猜缺失值"]}), root / f"{split}.parquet")
            config = {"sources": [{"source_id": split, "official_split": split,
                                    "local_globs": [f"{split}.parquet"]} for split in ("train", "test")]}
            out = root / "survey"
            with patch.object(p, "ROOT", root):
                p.profile(config, out)
            (out / "unbound").mkdir()
            (out / "unbound/profile.json").write_text("not a bound profile")
            result = p.audit_coverage([out, out])
            self.assertEqual(result["observed_train_test_exact_leak_groups"], 1)
            self.assertEqual(result["observed_cross_source_exact_overlap_groups"], 1)
            self.assertEqual(result["unique_observed_positions"], 2)
            self.assertEqual(result["unique_observed_nonempty_text_hashes"], 1)
            self.assertFalse(result["sample_positions_are_independent_capacity"])
            self.assertTrue(all(c["total_variation_distance"]["language"] is None for c in result["comparisons"]))

    def test_git_relation_requires_frozen_blob_bytes(self):
        raw = b'[{"schema":{"type":"integer"},"tests":[{"data":"001","valid":false}]}]'
        blob = p.hashlib.sha1(b"blob " + str(len(raw)).encode() + b"\0" + raw).hexdigest()
        frame = {"kind": "github_tree_metadata", "revision": "fixed", "complete_for_patterns": True,
                 "files": [{"path": "tests/test.json", "bytes": len(raw), "git_blob_sha": blob}]}
        config = {"sources": [{"source_id": "test", "github_repository": "org/repo", "format": "relation",
                                "relation_kind": "json-schema-tests", "official_split": "test"}]}
        sys.path.insert(0, str(MODULE.parent))
        try:
            with tempfile.TemporaryDirectory() as folder:
                out = Path(folder) / "ok"
                with patch.object(p, "github_frame", return_value=frame), patch.object(p, "fetch", return_value=(raw, {})):
                    result = p.profile(config, out, network=True)
                self.assertEqual(result["sources"][0]["sampled_shards"], 1)
                sample = json.loads(next((out / "test").glob("sample-*.json")).read_text())
                self.assertEqual(sample["samples"][0]["split"], "test")
                self.assertEqual(sample["samples"][0]["original"]["annotations"]["expected_valid"], [False])
                with patch.object(p, "github_frame", return_value=frame), patch.object(p, "fetch", return_value=(raw.replace(b"001", b"002"), {})):
                    failed = p.profile(config, Path(folder) / "bad", network=True)
                self.assertEqual(failed["sources"][0]["sampled_shards"], 0)
                self.assertIn("frozen blob", failed["sources"][0]["errors"][0]["error"])
        finally:
            sys.path.pop(0)

    def test_single_wave_uses_marginal_not_union_probability(self):
        a = {"shard": {"path": "a", "wave_probability": .1, "shard_probability": .2},
             "samples": [{"row_index": 0, "purpose": "population", "inclusion_probability": .02,
                          "utf8_bytes": 10, "text_sha256": "x", "metadata": {}}]}
        b = {"shard": {"path": "b", "wave_probability": 1, "shard_probability": 1},
             "samples": [{"row_index": 0, "purpose": "population", "inclusion_probability": .1,
                          "utf8_bytes": 100, "text_sha256": "y", "metadata": {}}]}
        self.assertAlmostEqual(p.summarize([a, b])["represented_record_total"], 60)
        self.assertAlmostEqual(p.summarize([a, b], single_wave=True)["represented_record_total"], 110)
        self.assertAlmostEqual(p.summarize([a, b], single_wave=True)["represented_utf8_bytes"], 2000)

    def test_range_cache_crosses_blocks_and_reuses_columns(self):
        payload = bytes(range(256)) * 8192
        calls = []
        def fake_fetch(url, budget, *, start=None, length=0):
            calls.append((start, length))
            return payload[start:start + length], {"Content-Range": f"bytes {start}-{start + length - 1}/{len(payload)}",
                                                  "X-Mei-Resolved-URL": url}
        with patch.object(p, "fetch", side_effect=fake_fetch):
            reader = p.RangeReader("https://example.com/x", None)
            reader.seek((1 << 20) - 5)
            self.assertEqual(reader.read(10), payload[(1 << 20) - 5:(1 << 20) + 5])
            self.assertEqual(len(calls), 3)
            self.assertEqual(reader.read(20), payload[(1 << 20) + 5:(1 << 20) + 25])
            self.assertEqual(len(calls), 3)
            reader.seek(-4, 2)
            self.assertEqual(reader.read(10), payload[-4:])

    def test_resume_adopts_only_matching_frame_and_design(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder);old = root / "old";old.mkdir();(old / "test").mkdir()
            config = {"sources": [{"source_id": "test", "repository": "org/repo", "patterns": ["*.parquet"]}]}
            frame = {"kind": "hf_revision_listing", "revision": "abc", "complete_for_patterns": True,
                     "files": [{"path": "a.parquet"}]}
            shard = p.select_shards(frame["files"], 20260913)[0]
            name = p.hashlib.sha256(b"a.parquet").hexdigest()[:20]
            p.write_new(old / "config.json", config)
            (old / "implementation.py.snapshot").write_bytes(b"old source")
            p.write_new(old / "test/frame.json", frame)
            p.write_new(old / f"test/sample-{name}.json", {"shard": shard, "samples": []})
            with patch.object(p, "hf_frame", return_value=frame), patch.object(p, "RangeReader", side_effect=AssertionError("must adopt")):
                result = p.profile(config, root / "new", network=True, resume_from=old)
            self.assertEqual(result["sources"][0]["sampled_shards"], 1)
            adopted = json.loads((root / f"new/test/sample-{name}.json").read_text())
            self.assertEqual(adopted["adopted_sample"]["sha256"], p.digest(old / f"test/sample-{name}.json"))

    def test_dialogue_reservoir_preserves_whole_records(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "data.jsonl"
            with path.open("w") as f:
                for i in range(1000):
                    f.write(json.dumps([f"第{i}项", "不是零，是001\n需要复查"]) + "\n")
            shard = {"path": "dialogue-fixture.jsonl", "shard_probability": 1}
            a = p.sample_jsonl(path, shard, 8, 20)
            b = p.sample_jsonl(path, shard, 8, 20)
            self.assertEqual(a, b)
            self.assertEqual(a["rows"], 1000)
            self.assertEqual(len(a["samples"]), 20)
            self.assertNotEqual([r["row_index"] for r in a["samples"]], list(range(20)))
            self.assertTrue(all(r["inclusion_probability"] == .02 for r in a["samples"]))
            self.assertTrue(all(r["original"][1] == "不是零，是001\n需要复查" for r in a["samples"]))

    def test_ignored_http_range_never_reads_body(self):
        class Response:
            headers = {}
            status = 200
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def geturl(self): return "https://example.com/file"
            def read(self, *args): raise AssertionError("must not download ignored range")
        with tempfile.TemporaryDirectory() as folder:
            b = p.Budget(Path(folder), 100, 0)
            with patch.object(p.urllib.request, "urlopen", return_value=Response()):
                with self.assertRaisesRegex(RuntimeError, "honor exact byte range"):
                    p.fetch("https://example.com/file", b, start=0, length=1)

    def test_spread_and_disjoint_waves(self):
        files = [{"path": f"{i:04d}.parquet"} for i in range(975)]
        selected = p.select_shards(files, 9)
        self.assertEqual(selected, p.select_shards(list(reversed(files)), 9))
        self.assertEqual(len(selected), 128)
        self.assertEqual(len({s["path"] for s in selected}), 128)
        self.assertEqual({s["stratum"] for s in selected}, set(range(64)))
        self.assertEqual({s["stratum"] for s in selected[:64]}, set(range(64)))
        self.assertTrue(all(s["wave"] == 1 for s in selected[:64]))
        self.assertEqual(sum(1 / s["shard_probability"] for s in selected), 975)
        self.assertTrue(any(int(s["path"][:4]) > 950 for s in selected))

    def test_small_frame_is_census(self):
        rows = p.select_shards([{"path": str(i)} for i in range(6)], 1)
        self.assertEqual(len(rows), 6)
        self.assertTrue(all(r["shard_probability"] == 1 for r in rows))
        self.assertEqual(p.select_shards([], 1), [])
        with self.assertRaises(ValueError):
            p.select_shards([{"path": "a"}, {"path": "a"}], 1)

    def test_unequal_groups_and_preserved_values(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "input.parquet"
            values = ["001\n空 值\t℃"] * 91 + ["long" * 100] * 10
            pq.write_table(pa.table({"text": values, "dump": ["a"] * 91 + ["b"] * 10}), path, row_group_size=13)
            shard = {"path": str(path), "shard_probability": 1}
            result = p.sample_parquet(path, shard, 17, 200)
            stats = p.summarize([result])
            self.assertEqual(stats["sample_records"], 101)
            self.assertEqual(stats["represented_record_total"], 101)
            self.assertEqual(result["samples"][0]["text"], values[0])
            self.assertAlmostEqual(stats["document_distributions"]["dump"]["a"], 91 / 101)
            self.assertLess(stats["byte_distributions"]["dump"]["a"], 0.5)

    def test_targeted_samples_do_not_change_population(self):
        row = {"row_index": 0, "purpose": "population", "inclusion_probability": .5,
               "utf8_bytes": 10, "text_sha256": "a", "metadata": {"dump": "old"}}
        extra = {**row, "row_index": 1, "purpose": "targeted", "metadata": {"dump": "new"}}
        a = p.summarize([{"shard": {"path": "a"}, "samples": [row]}])
        b = p.summarize([{"shard": {"path": "a"}, "samples": [row, extra]}])
        self.assertEqual(a, b)
        self.assertEqual(a["document_distributions"]["topic"], {"unknown": 1.0})

    def test_duplicate_positions_fail(self):
        row = {"row_index": 0, "purpose": "population", "inclusion_probability": 1,
               "utf8_bytes": 0, "text_sha256": "a", "metadata": {}}
        with self.assertRaises(ValueError):
            p.summarize([{"shard": {"path": "a"}, "samples": [row, row]}])

    def test_hash_audit_and_write_once(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            report = {"source_id": "x", "process_status": "hold"}
            (root / "x").mkdir()
            p.write_new(root / "x/profile.json", report)
            p.write_new(root / "survey.json", {"artifacts": {"x/profile.json": p.digest(root / "x/profile.json")}})
            self.assertFalse(p.audit_coverage([root])["m1_passed"])
            with self.assertRaises(FileExistsError):
                p.write_new(root / "survey.json", {})
            (root / "x/profile.json").write_text("{}")
            with self.assertRaises(ValueError):
                p.audit_coverage([root])

    def test_budget_reserves_uncertain_reads(self):
        with tempfile.TemporaryDirectory() as folder:
            budget = p.Budget(Path(folder), 10, 0)
            budget.reserve(8)
            with self.assertRaises(RuntimeError):
                budget.reserve(3)
            self.assertEqual(budget.reserved, 8)

    def test_missing_raw_column_is_not_silently_serialized(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "table.parquet"
            pq.write_table(pa.table({"amount": [1, 2]}), path)
            with self.assertRaisesRegex(ValueError, "relation adapter"):
                p.sample_parquet(path, {"path": str(path), "shard_probability": 1}, 1)


if __name__ == "__main__":
    unittest.main()
