import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("survey_relations", Path(__file__).resolve().parents[1] / "sources/relations.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RelationSurveyTests(unittest.TestCase):
    def test_dialogue_labels_are_not_visible_facts(self):
        value = {"a": {"goal": ["hidden"], "messages": [{"role": "usr", "content": "还是改成两间", "dialog_act": ["gold"]}]}}
        item = next(r.relation_units(value, "crosswoz"))
        self.assertEqual(item["input"], {"messages": [{"role": "usr", "content": "还是改成两间"}]})
        self.assertEqual(item["annotations"]["messages"][0]["dialog_act"], ["gold"])
        self.assertEqual(item["original"], value["a"])

    def test_false_zero_and_empty_are_not_conflated(self):
        group = {"schema": {"type": "integer"}, "tests": [{"data": 0, "valid": True}, {"data": "", "valid": False}, {"data": False, "valid": False}]}
        item = next(r.relation_units([group], "json-schema-tests"))
        self.assertEqual(item["annotations"]["expected_valid"], [True, False, False])
        self.assertTrue(all("valid" not in case for case in item["input"]["cases"]))

    def test_table_headers_not_invented(self):
        item = next(r.relation_units({"rows": [["001", None, 0]], "annotations": {"column_labels": ["id"]}}, "table-bundle"))
        self.assertNotIn("headers", item["input"])
        self.assertEqual(item["input"]["rows"], [["001", None, 0]])

    def test_document_links_require_real_endpoints(self):
        with self.assertRaises(ValueError):
            list(r.relation_units({"documents": [{"id": "a", "revision": "rev", "sha256": "hash"}],
                                  "relationships": [{"from": "a", "to": "missing"}]}, "document-bundle"))


if __name__ == "__main__": unittest.main()
