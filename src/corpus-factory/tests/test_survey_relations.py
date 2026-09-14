import importlib.util
from pathlib import Path
import unittest
import hashlib
import io
import json

spec = importlib.util.spec_from_file_location("survey_relations", Path(__file__).resolve().parents[1] / "sources/relations.py")
r = importlib.util.module_from_spec(spec)
spec.loader.exec_module(r)


class RelationSurveyTests(unittest.TestCase):
    def test_literary_page_preserves_revision_and_does_not_invent_missing_work(self):
        page={'pageid':7,'title':'话剧','revisions':[{'revid':9,'slots':{'main':{'*':'{{前言}}\n甲：不是零。'}}}]}
        value={'query':{'pages':{'7':page,'-1':{'title':'未找到','missing':''}}}}
        units=list(r.relation_units(value,'wikisource-pages'))
        self.assertEqual(len(units),1)
        self.assertEqual(units[0]['original'],page)
        self.assertEqual(units[0]['annotations']['revision_id'],9)
        self.assertFalse(units[0]['annotations']['complete_work_verified'])
        self.assertFalse(units[0]['annotations']['transclusions_resolved'])

    def test_array_stream_and_tool_dialogue_do_not_leak_future_results(self):
        raw = '[{"content":["改成001", "好的"]}, false, 1.25e-3]'.encode()
        for chunk in (1, 7, 65536):
            h=hashlib.sha256()
            self.assertEqual([v for _,v in r.iter_json_object(io.BytesIO(raw), h, chunk_size=chunk, mapping=False)], json.loads(raw))
            self.assertEqual(h.hexdigest(), hashlib.sha256(raw).hexdigest())
        original={'system':'tools: []','conversations':[{'from':'user','value':'查温度'}, {'from':'assistant','value':'[read()]'}, {'from':'tool','value':'25'}]}
        item=next(r.relation_units([original],'toolace'))
        self.assertEqual(item['original'],original)
        self.assertNotIn('25',json.dumps(item['input']))
        self.assertEqual(item['annotations']['non_user_turns'][1]['turn_index'],2)

    def test_streamed_mapping_preserves_unicode_and_whole_nested_values(self):
        raw = '{"甲":{"messages":["不是0，是001","转义\\\"不拆句"]},"乙":1.25e-3,"丙":false}'.encode()
        for chunk in (1, 3, 31, 65536):
            h = hashlib.sha256()
            actual = dict(r.iter_json_object(io.BytesIO(raw), h, chunk_size=chunk))
            self.assertEqual(actual, json.loads(raw))
            self.assertEqual(h.hexdigest(), hashlib.sha256(raw).hexdigest())
        for raw in (b'{"a":1,}', b'{"a":1} trailing', b'{"a":1,"a":2}'):
            with self.assertRaises(ValueError):
                list(r.iter_json_object(io.BytesIO(raw), hashlib.sha256(), chunk_size=1))

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
