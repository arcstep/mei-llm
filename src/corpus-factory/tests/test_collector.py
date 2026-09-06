from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[3]


def load_module(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


collector = load_module("mei_51m_source_collector", "src/corpus-factory/sources/collector.py")
sources = load_module(
    "mei_51m_source_manager", "src/corpus-factory/sources/source_manager.py"
)
registry = load_module(
    "mei_51m_source_registry", "src/corpus-factory/sources/registry.py"
)


class ResolveTests(unittest.TestCase):
    def test_static_manifest_resolves_with_absolute_manifest_path(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            manifest = root / "static.json"
            manifest.write_text(
                json.dumps(
                    {
                        "schema": "mei-51m-source-manifest-v1",
                        "source_id": "sqllogictest",
                        "files": [
                            {
                                "filename": "test/select1.test",
                                "url": "https://example.com/select1.test",
                                "sha256": "a" * 64,
                                "bytes": 12,
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            entry = {
                "source_id": "sqllogictest",
                "manifest_source": {
                    "type": "static_manifest",
                    "manifest": str(manifest),
                },
            }
            batch = collector.resolve_batch(entry)
            self.assertEqual(1, len(batch["files"]))
            self.assertIn("sqllogictest:", batch["batch_token"])

    def test_batch_token_is_sensitive_to_listing(self) -> None:
        entry = {
            "source_id": "x",
            "manifest_source": {"type": "direct_urls"},
            "urls": [{"filename": "a.txt", "url": "https://example.com/a.txt"}],
        }
        first = collector.resolve_batch(entry)
        entry["urls"].append({"filename": "b.txt", "url": "https://example.com/b.txt"})
        second = collector.resolve_batch(entry)
        self.assertNotEqual(first["batch_token"], second["batch_token"])

    def test_hf_api_resolves_with_explicit_filenames_without_network(self) -> None:
        entry = {
            "source_id": "fineweb2-hq-cmn-hani",
            "repository": "epfml/FineWeb2-HQ",
            "manifest_source": {"type": "hf_api", "repo_type": "dataset", "patterns": ["*.parquet"]},
        }
        with patch.object(collector, "list_hf_files") as list_files:
            batch = collector.resolve_batch(
                entry, filenames=["data/cmn_Hani/train-000.parquet"]
            )
        list_files.assert_not_called()
        self.assertEqual(1, len(batch["files"]))

    def test_hf_api_filters_listing_by_patterns_and_subset(self) -> None:
        entry = {
            "source_id": "fineweb2-hq-cmn-hani",
            "repository": "epfml/FineWeb2-HQ",
            "manifest_source": {
                "type": "hf_api",
                "repo_type": "dataset",
                "patterns": ["data/cmn_Hani/*.parquet"],
            },
        }
        listing = [
            "data/cmn_Hani/train-000.parquet",
            "data/cmn_Hani/train-001.parquet",
            "data/en/train-000.parquet",
        ]
        with patch.object(
            collector, "list_hf_files", return_value=listing
        ) as list_files:
            batch = collector.resolve_batch(entry, subset="data/cmn_Hani/train-00[01]*.parquet")
        list_files.assert_called_once()
        self.assertEqual(
            ["data/cmn_Hani/train-000.parquet", "data/cmn_Hani/train-001.parquet"],
            [row["filename"] for row in batch["files"]],
        )


class DownloadTests(unittest.TestCase):
    def entry(self, urls: list[dict[str, str]]) -> dict:
        return {
            "source_id": "test-source",
            "role": "structured",
            "manifest_source": {"type": "direct_urls"},
            "urls": urls,
        }

    def batch(self, urls: list[dict[str, str]]) -> dict:
        return collector.resolve_batch(self.entry(urls))

    def test_refuses_bad_token_before_any_network(self) -> None:
        urls = [{"filename": "a.json", "url": "https://example.com/a.json"}]
        batch = self.batch(urls)
        with tempfile.TemporaryDirectory() as raw:
            with patch("urllib.request.urlopen") as urlopen:
                with self.assertRaisesRegex(
                    collector.CollectorError, "download refused"
                ):
                    collector.download_batch(
                        batch,
                        Path(raw) / "out",
                        entry=self.entry(urls),
                        authorization="test-source:0000000000000000",
                    )
            urlopen.assert_not_called()

    def test_direct_url_download_verifies_hash_and_pins(self) -> None:
        payload = b'{"name": "nod"}\n'
        import hashlib

        expected = hashlib.sha256(payload).hexdigest()
        urls = [
            {"filename": "a.json", "url": "https://example.com/a.json", "sha256": expected}
        ]
        batch = self.batch(urls)
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "out"
            with patch(
                "urllib.request.urlopen",
                return_value=Mock(
                    __enter__=Mock(return_value=Mock(read=Mock(side_effect=[payload, b""]))),
                    __exit__=Mock(return_value=False),
                ),
            ):
                manifest = collector.download_batch(
                    batch, out, entry=self.entry(urls), authorization=batch["batch_token"]
                )
            self.assertEqual("mei-51m-download-manifest-v1", manifest["schema"])
            self.assertTrue(manifest["files"][0]["hash_pinned"])
            self.assertEqual(expected, manifest["files"][0]["sha256"])
            self.assertFalse(manifest["distribution_clearance_asserted"])
            self.assertTrue((out / "download-manifest.json").is_file())

    def test_hash_drift_fails_closed_without_artifacts(self) -> None:
        urls = [
            {
                "filename": "a.json",
                "url": "https://example.com/a.json",
                "sha256": "b" * 64,
            }
        ]
        batch = self.batch(urls)
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "out"
            with patch(
                "urllib.request.urlopen",
                return_value=Mock(
                    __enter__=Mock(
                        return_value=Mock(
                            read=Mock(side_effect=[b"changed-content", b""])
                        )
                    ),
                    __exit__=Mock(return_value=False),
                ),
            ):
                with self.assertRaisesRegex(collector.CollectorError, "hash drift"):
                    collector.download_batch(
                        batch, out, entry=self.entry(urls), authorization=batch["batch_token"]
                    )
            self.assertFalse(out.exists())

    def test_unpinned_download_records_hash_pinned_false(self) -> None:
        urls = [{"filename": "a.json", "url": "https://example.com/a.json"}]
        batch = self.batch(urls)
        with tempfile.TemporaryDirectory() as raw:
            out = Path(raw) / "out"
            with patch(
                "urllib.request.urlopen",
                return_value=Mock(
                    __enter__=Mock(
                        return_value=Mock(
                            read=Mock(side_effect=[b"some-bytes", b""])
                        )
                    ),
                    __exit__=Mock(return_value=False),
                ),
            ):
                manifest = collector.download_batch(
                    batch, out, entry=self.entry(urls), authorization=batch["batch_token"]
                )
            self.assertFalse(manifest["files"][0]["hash_pinned"])


class DownloadHqDelegationTests(unittest.TestCase):
    def test_download_hq_keeps_legacy_authorization_but_delegates(self) -> None:
        fake = Mock()
        fake.CollectorError = collector.CollectorError
        fake.resolve_batch = Mock(return_value={"batch_token": "t:ok", "files": []})
        fake.download_batch = Mock(
            return_value={"schema": "mei-51m-download-manifest-v1"}
        )
        with tempfile.TemporaryDirectory() as raw:
            with patch.object(sources, "collector", return_value=fake):
                result = sources.download_hq(
                    ["data/cmn_Hani/train-000.parquet"],
                    Path(raw) / "out",
                    sources.DOWNLOAD_AUTHORIZATION,
                )
        fake.download_batch.assert_called_once()
        self.assertEqual(
            "t:ok", fake.download_batch.call_args.kwargs["authorization"]
        )
        self.assertEqual("mei-51m-download-manifest-v1", result["schema"])

    def test_download_hq_wraps_collector_errors(self) -> None:
        fake = Mock()
        fake.CollectorError = collector.CollectorError
        fake.resolve_batch = Mock(
            side_effect=collector.CollectorError("no files")
        )
        with tempfile.TemporaryDirectory() as raw:
            with patch.object(sources, "collector", return_value=fake):
                with self.assertRaisesRegex(sources.SourceError, "no files"):
                    sources.download_hq(
                        ["data/cmn_Hani/train-000.parquet"],
                        Path(raw) / "out",
                        sources.DOWNLOAD_AUTHORIZATION,
                    )


class DownloadCommandTests(unittest.TestCase):
    def test_dry_run_prints_token_without_network(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            urls = [
                {
                    "filename": "a.json",
                    "url": "https://example.com/a.json",
                    "sha256": "a" * 64,
                    "bytes": 10,
                }
            ]
            entry = {
                "source_id": "test-source",
                "role": "structured",
                "manifest_source": {"type": "direct_urls"},
                "urls": urls,
            }
            with patch.object(
                sources, "entry_for", return_value=entry
            ), patch.object(collector, "download_batch") as download:
                result = sources.download(
                    "test-source",
                    None,
                    subset=None,
                    filenames=None,
                    authorization=None,
                    dry_run=True,
                    cache_dir=None,
                )
            download.assert_not_called()
            self.assertEqual("dry_run", result["status"])
            self.assertEqual(10, result["total_bytes"])
            self.assertIn("test-source:", result["batch_token"])

    def test_download_requires_authorization_and_out(self) -> None:
        entry = {
            "source_id": "test-source",
            "manifest_source": {"type": "direct_urls"},
            "urls": [{"filename": "a.json", "url": "https://example.com/a.json"}],
        }
        with patch.object(sources, "entry_for", return_value=entry):
            with self.assertRaisesRegex(sources.SourceError, "authorize-download"):
                sources.download(
                    "test-source", None, subset=None, filenames=None,
                    authorization=None, dry_run=False, cache_dir=None,
                )


if __name__ == "__main__":
    unittest.main()
